from __future__ import annotations

import torch
import torch.distributed as dist
import numpy as np
from tensordict import TensorDict
from rllava.data.protocol import DataProto
from rllava.distribute.dist_utils import CoordinatorError, broadcast_object
from rllava.roles.generator import Generator
from typing import Callable, TYPE_CHECKING
from collections import defaultdict
from .agents import get_agent
from .rollout_worker import RolloutWorker
from .batching import RolloutBatching
from copy import deepcopy

if TYPE_CHECKING:
    from rllava.rollout.config import RolloutConfig

__all__ = ["RolloutManager"]


class RolloutManager:
    def __init__(
        self,
        config: RolloutConfig,
        reward,
        tokenizer,
        processor,
        agent_cls=None,
        rollout_processor=None,
        *,
        workflow=None,
    ):
        self.config = config
        self.n = config.n
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward = reward
        self.generator = Generator(config, tokenizer, processor)
        if workflow is not None:
            if agent_cls is not None:
                raise ValueError("Specify agent_cls or the legacy workflow adapter, not both")
            agent_cls = getattr(workflow, "agent_cls", None)
            if agent_cls is None:
                raise ValueError("Custom workflows must be migrated to the rollout agent interface")
        # Also accept the former positional workflow class during migration.
        self.agent_cls = getattr(agent_cls, "agent_cls", agent_cls) or get_agent(
            getattr(config, "agent", "single_turn")
        )
        self.batching = RolloutBatching(tokenizer.pad_token_id)
        if rollout_processor is not None:
            self.rollout_processor = rollout_processor
        else:
            self.rollout_processor = None
        self.algorithm_config = None
        self.train_batch_size = None
        self.max_try_make_batch = 0

    def configure_sampling(self, algorithm_config, train_batch_size, max_try_make_batch):
        self.algorithm_config = algorithm_config
        self.train_batch_size = train_batch_size
        self.max_try_make_batch = max_try_make_batch

    def filter(self, batch):
        config = self.algorithm_config
        if config is None or not config.online_filtering:
            return batch
        scores = batch.batch[f"reward_{config.filter_key}"].detach().cpu().tolist()
        uids = batch.non_tensor_batch["uid"]
        grouped = defaultdict(list)
        for uid, score in zip(uids, scores):
            grouped[uid].append(score)
        keep = {uid for uid, values in grouped.items()
                if config.filter_low < np.mean(values) < config.filter_high}
        return batch[[i for i, uid in enumerate(uids) if uid in keep]]

    def collect(self, data_iterator, global_step: int | None = None) -> DataProto:
        def collect():
            batches, count, attempts = [], 0, 0
            target = self.train_batch_size * self.n
            while count < target:
                attempts += 1
                data = DataProto.from_single_dict(next(data_iterator))
                batch = self.generate_one_batch(data, global_step=global_step)
                if len(batch):
                    batches.append(batch)
                    count += len(batch)
                limit = self.max_try_make_batch
                if count < target and 0 < limit <= attempts:
                    raise RuntimeError(f"Online filtering collected {count}/{target} responses after {attempts} attempts.")
            self.batching._align_trajectory_lengths(batches)
            batch = DataProto.concat(batches)[:target]
            batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
            return batch

        if self.train_batch_size is None:
            raise RuntimeError("configure_sampling must be called before collect")
        return self.run(collect)

    def initialize(self, model_path, dispatcher):
        self.dispatcher = dispatcher
        self.resources = dispatcher.resources
        self.generator.initialize(model_path, self.resources.roles["rollout"])

    def run_workers(self, prompts, source_data=None, *, n=1, agent_cls=None, worker_config=None):
        """Schedule isolated trajectory workers and batch their model requests."""
        if n < 1:
            raise ValueError("rollout n must be positive")
        agent_cls = agent_cls or self.agent_cls
        worker_config = worker_config or self.config
        original_size = len(source_data) if source_data is not None else len(prompts)
        if len(prompts) == original_size:
            prompts = prompts.repeat(repeat_times=n, interleave=True)
        elif len(prompts) != original_size * n:
            raise ValueError("Processed prompts do not match the requested trajectory count")
        if source_data is not None:
            source_data = source_data.repeat(repeat_times=n, interleave=True)

        def make_env():
            from rllava.rollout.envs import initialize_env_from_config
            return initialize_env_from_config(worker_config.env_config_path)

        def make_tools():
            from rllava.rollout.tools import initialize_tools_from_config
            return initialize_tools_from_config(worker_config.tool_config_path)

        sessions, requests, results = [], {}, {}
        try:
            for index in range(len(prompts)):
                data = deepcopy(prompts[[index]])
                if source_data is not None:
                    source = source_data[[index]]
                    for key, value in source.non_tensor_batch.items():
                        if key not in data.non_tensor_batch:
                            data.non_tensor_batch[key] = deepcopy(value)
                agent = agent_cls(self.tokenizer, self.processor)
                multi_turn = agent.multi_turn
                worker = RolloutWorker(
                    agent,
                    max_turns=worker_config.max_turns if multi_turn else 1,
                    env_factory=make_env if multi_turn and worker_config.env_config_path else None,
                    tools_factory=make_tools if multi_turn and worker_config.tool_config_path else None,
                    evaluate_env=getattr(getattr(self.reward, "config", None), "reward_type", None) == "env",
                )
                session = worker.run(data)
                sessions.append(session)
                requests[index] = next(session)

            while requests:
                indices = list(requests)
                active = [requests[i] for i in indices]
                self.batching._align_active_samples(active, list(range(len(active))))
                batch = DataProto.concat(active)
                batch.meta_info["n"] = 1
                output = self.generate_sequences(batch)
                if len(output) != len(indices):
                    raise ValueError("Generator must return one response per active trajectory")
                for row, index in enumerate(indices):
                    try:
                        requests[index] = sessions[index].send(output[[row]])
                    except StopIteration as completed:
                        results[index] = completed.value
                        del requests[index]
            samples = [results[i] for i in range(len(sessions))]
            self.batching._align_trajectory_lengths(samples)
            result = DataProto.concat(samples)
            result.batch = TensorDict(source=result.batch.to_dict(), batch_size=result.batch.batch_size)
            return result
        finally:
            # Covers reset, generation, execution and assembly failures, as well
            # as trajectories still suspended at a generation request.
            for session in sessions:
                session.close()

    def _broadcast(self, value):
        return broadcast_object(value, src=self.resources.coordinator_rank, group=self.resources.control_group)

    def run(self, collect_on_coordinator):
        if self.resources.rank == self.resources.coordinator_rank:
            try:
                result = collect_on_coordinator()
            except Exception as exc:
                self._broadcast(("abort", f"{type(exc).__name__}: {exc}"))
                raise
            self._broadcast(("finish", None))
            return result
        while True:
            command, error = self._broadcast(None)
            if command == "finish":
                return None
            if command == "abort":
                raise RuntimeError(error)
            # A failed round returns all workers to this command loop; the
            # coordinator broadcasts abort instead of stranding them in gather.
            try:
                self._generate_round(None)
            except CoordinatorError:
                continue

    def generate(self, prompts):
        self._broadcast(("generate", None))
        result, error = self._generate_round(prompts)
        if error:
            raise RuntimeError(error)
        return result

    def _generate_round(self, prompts):
        layout = self.resources.roles["rollout"]
        local = self.dispatcher.scatter(prompts, "rollout")
        if self.generator.collective_generate and self.resources.world_size > 1:
            local = broadcast_object(local, src=layout.input_rank, group=layout.groups["replica"])
        active = self.generator.collective_generate or (local is not None and len(local) > 0)
        prepared, error = None, None
        try:
            if active:
                prepared = self.generator.prepare(local)
        except Exception as exc:
            error = f"rank {self.resources.rank}: {type(exc).__name__}: {exc}"
        error = self._agree_error(error)
        if error:
            return None, error

        output, error = None, None
        try:
            if active:
                output = self.generator.generate(prepared)
        except Exception as exc:
            error = f"rank {self.resources.rank}: {type(exc).__name__}: {exc}"
        error = self._agree_error(error)
        if error:
            return None, error
        return self.dispatcher.gather(output, "rollout"), None

    def _agree_error(self, error):
        errors = [error]
        if self.resources.world_size > 1:
            errors = [None] * self.resources.world_size
            dist.all_gather_object(errors, error, group=self.resources.control_group)
        return next((error for error in errors if error), None)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the rollout engine.
        
        Args:
            prompts: Input prompts for generation
            
        Returns:
            Generated sequences
        """
        self.generator.prepare_prompts(prompts)

        output = self.generate(prompts)
        output = output.to("cpu")
        return output


    def generate_one_batch(self, data: DataProto, filter: Callable | None = None, val=False, global_step: int | None = None) -> DataProto:  
        # uid
        import uuid
        data.non_tensor_batch["uid"] = np.array([
            str(uuid.uuid4()) for _ in range(len(data.batch))
        ], dtype=object)

        # pop keys for generation
        batch_keys = ["input_ids", "attention_mask", "position_ids"]
        saved_tgt_input_ids = None
        if "tgt_input_ids" in data.batch.keys():
            saved_tgt_input_ids = data.batch["tgt_input_ids"]
            batch_keys.append("tgt_input_ids")

        gen_batch = data.pop(
            batch_keys=batch_keys,
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        if val:
            gen_batch.meta_info = dict(self.config.val_override_config)
        else:
            gen_batch.meta_info = dict()
        if global_step is not None:
            gen_batch.meta_info["global_step"] = int(global_step)

        gen_batch.meta_info["min_pixels"] = self.config.min_pixels
        gen_batch.meta_info["max_pixels"] = self.config.max_pixels
        gen_batch.meta_info["video_fps"] = self.config.video_fps

        if self.rollout_processor is not None and val is False:
            gen_batch = self.rollout_processor.pre_process(gen_batch)

        rollout_n = self.config.val_override_config.get("n", 1) if val else self.config.n
        gen_batch_output = self.run_workers(gen_batch, data, n=rollout_n)

        if self.rollout_processor is not None and val is False:
            gen_batch_output = self.rollout_processor.post_process(gen_batch_output, gen_batch)

        if val:
            data = data.repeat(repeat_times=rollout_n, interleave=True)
            data.batch = TensorDict(source={}, batch_size=data.batch.batch_size)
            new_batch = data.union(gen_batch_output)
            return new_batch
        
        if self.config.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.generate_sequences(gen_baseline_batch)
            new_batch = data.union(gen_baseline_output)
            reward_baseline_tensor, _ = self.reward.compute_rewards(new_batch)
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output
        
        # restore tgt_input_ids for downstream off-policy processing
        if saved_tgt_input_ids is not None:
            data.batch["tgt_input_ids"] = saved_tgt_input_ids

        data = data.repeat(repeat_times=rollout_n, interleave=True)
        # Preserve any restored target IDs while replacing stale empty storage.
        data.batch = TensorDict(source=data.batch.to_dict(), batch_size=data.batch.batch_size)
        new_batch = data.union(gen_batch_output)

        # compute rewards first
        reward_tensor, reward_metrics = self.reward.compute_rewards(new_batch)
        new_batch.batch["token_level_scores"] = reward_tensor
        # store per-sample reward metrics into batch for later slicing/aggregation
        for k, v in reward_metrics.items():
            new_batch.batch[f"reward_{k}"] = torch.tensor(v, dtype=torch.float32, device=reward_tensor.device)

        new_batch = (self.filter if filter is None else filter)(new_batch)
        if self.rollout_processor is not None and val is False:
            new_batch = self.rollout_processor.post_rollout(
                new_batch,
                self.reward,
                self,
            )
        return new_batch
