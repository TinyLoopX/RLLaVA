"""Synchronous RL controller shared by RLVR, SFT–RL and Agentic tasks."""

from contextlib import contextmanager
from tqdm import tqdm

from .base import BaseController
from rllava.distribute.resource_manager import ResourceManager
from rllava.distribute.batch_dispatcher import BatchDispatcher
from rllava.distribute.dist_utils import is_rank0
from rllava.rollout import RolloutManager
from rllava.roles.reward import Reward
from rllava.train.train import Learner
from rllava.rollout.processors import PrefixRolloutProcessor, HPTBatchRolloutProcessor
from rllava.utils.metrics import (
    compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics,
)
from rllava.utils.py_functional import timer
from rllava.utils.logger.aggregate_logger import print_rank_0


def build_rollout(config, reward, tokenizer, processor):
    """Algorithm configuration chooses sampling transforms for every entry point."""
    rollout_processor = None
    if config.algorithm.unify_strategy == "switch":
        rollout_processor = HPTBatchRolloutProcessor(config, tokenizer)
    elif config.algorithm.prefix_strategy != "none":
        rollout_processor = PrefixRolloutProcessor(config, tokenizer)
    rollout = RolloutManager(config.rollout, reward, tokenizer, processor,
                             rollout_processor=rollout_processor)
    rollout.configure_sampling(config.algorithm, config.data.train_batch_size,
                               config.trainer.max_try_make_batch)
    return rollout


class Controller(BaseController):
    def __init__(self, config, tokenizer, processor, train_dataloader, val_dataloader):
        self.config = config
        self.resources = ResourceManager.from_config(config)
        self.dispatcher = BatchDispatcher(self.resources)
        self.rollout = None
        self._closed = False
        try:
            reward = Reward(config.reward, tokenizer)
            self.rollout = build_rollout(config, reward, tokenizer, processor)
            self.learner = Learner(config, tokenizer, processor,
                                   resources=self.resources, dispatcher=self.dispatcher)
            training_steps = (config.trainer.max_steps if config.trainer.max_steps is not None
                              else len(train_dataloader) * config.trainer.total_epochs)
            layout = self.resources.roles["rollout"]
            print_rank_0(
                f"Using {self.resources.world_size} GPUs for training; generation uses "
                f"{layout.dp_size} replicas with {layout.parallel.tp_size} GPUs each."
            )
            self.rollout.initialize(config.actor.model.model_path, self.dispatcher)
            self.learner.initialize(training_steps)
            reward.initialize()
            super().__init__(self.learner, config, train_dataloader, val_dataloader,
                             rollout=self.rollout, reward=reward)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def generate_context(self):
        with self.learner.actor.unwrap_model_for_generation() as model:
            self.rollout.generator.load(model)
        try:
            yield
        finally:
            self.rollout.generator.offload()

    def collect(self, data_iterator, global_step=None):
        with self.generate_context():
            batch = self.rollout.collect(data_iterator, global_step=global_step)
        return self.learner._to_training_device(self.dispatcher.scatter(batch))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.rollout is not None:
                self.rollout.generator.close()
        finally:
            self.resources.close()

    def run(self):
        if self.config.trainer.val_before_train or self.config.trainer.val_only:
            val_metrics = self.validate()
            self.logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.val_only:
                return

        self.data_iterator = self.iter_train_batches()

        for _ in tqdm(
            range(self.global_steps, self.training_steps),
            initial=self.global_steps,
            total=self.training_steps,
            desc="Training Progress",
            disable=not is_rank0(),
        ):
            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    batch = self.collect(self.data_iterator, global_step=self.global_steps)

                batch, output = self.learner.step(
                    batch, global_step=self.global_steps, timing_raw=timing_raw
                )

                batch = self.dispatcher.gather(batch)
                output = self.dispatcher.gather(output)

                self.global_steps += 1

            if (self.config.trainer.val_freq > 0 and self.global_steps % self.config.trainer.val_freq == 0) \
                        or self.global_steps >= self.training_steps:
                with timer("validation", timing_raw):
                    self.validate(metrics)

            if (self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0) \
                        or self.global_steps >= self.training_steps:
                with timer("save_checkpoint", timing_raw):
                    self.save_checkpoint()

            if is_rank0():
                # Distributed gathers above finish on all ranks before logging.
                for key in batch.batch.keys():
                    if isinstance(key, str) and key.startswith("reward_"):
                        metrics[f"reward/{key[len('reward_'):]}"] = batch.batch[key].mean().detach().item()
                metrics.update(batch.meta_info.get("adv_metrics", {}))
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.learner.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=self.resources.world_size))
                metrics.update(reduce_metrics(output.non_tensor_batch))

                self.logger.log(data=metrics, step=self.global_steps)


