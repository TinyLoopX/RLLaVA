from __future__ import annotations

import torch
import torch.distributed as dist
import numpy as np
from tensordict import TensorDict
from rllava.data.protocol import DataProto
from rllava.distribute.dist_utils import CoordinatorError, broadcast_object
from rllava.utils.model_utils import print_gpu_memory_usage
from typing import Callable, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from ..config import RolloutConfig

__all__ = ["Rollout"]


class Rollout():
    def __init__(
        self,
        config: RolloutConfig,
        reward,
        tokenizer,
        processor,
        workflow=None,
        rollout_processor=None,
    ):
        self.config = config
        self.n = config.n
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward = reward
        self.rollout_engine = None
        if workflow is not None:
            self.workflow = workflow(
                self.reward,
                self.tokenizer,
                self.processor,
                self.config.max_turns,
                self.config.discount,
                self.config.env_config_path,
            )
        else:
            self.workflow = None
        if rollout_processor is not None:
            self.rollout_processor = rollout_processor
        else:
            self.rollout_processor = None
        self.algorithm_config = None
        self.data_config = None

    def initialize(self, model_path, dispatcher):
        from rllava.engine import EngineFactory
        from transformers import GenerationConfig

        self.dispatcher = dispatcher
        self.resources = dispatcher.resources
        self.model_path = model_path
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        """Initialize the rollout engine for sequence generation."""
        print_gpu_memory_usage(f"Before {self.config.name} rollout engine init")
        engine_class = EngineFactory(self.config.name)
        self.rollout_engine = engine_class(
            model_name_or_path=self.model_path,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            layout=dispatcher.resources.roles["rollout"],
        )
        print_gpu_memory_usage(f"After {self.config.name} rollout engine init")

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
        if self.rollout_engine.collective_generate and self.resources.world_size > 1:
            local = broadcast_object(local, src=layout.input_rank, group=layout.groups["replica"])
        active = self.rollout_engine.collective_generate or (local is not None and len(local) > 0)
        prepared, error = None, None
        try:
            if active:
                prepared = self.rollout_engine.prepare(local)
        except Exception as exc:
            error = f"rank {self.resources.rank}: {type(exc).__name__}: {exc}"
        error = self._agree_error(error)
        if error:
            return None, error

        output, error = None, None
        try:
            if active:
                output = self.rollout_engine.generate(prepared)
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
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        output = self.generate(prompts)
        output = output.to("cpu")
        return output


    def generate_one_batch(self, data: DataProto, filter: Callable = lambda sample: sample, val=False, global_step: int | None = None) -> DataProto:  
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

        if self.workflow is not None:
            # Pre-expand both gen_batch and data by n so every trajectory is independent.
            # Inside the workflow each turn runs with n=1 (enforced via meta_info).
            rollout_n = self.config.val_override_config.get("n", 1) if val else self.config.n
            if rollout_n > 1:
                gen_batch = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
                data = data.repeat(repeat_times=rollout_n, interleave=True)
            gen_batch_output = self.workflow.run(self, gen_batch, data)
        else:
            gen_batch_output = self.generate_sequences(gen_batch)

        if self.rollout_processor is not None and val is False:
            gen_batch_output = self.rollout_processor.post_process(gen_batch_output, gen_batch)

        if val:
            # data already expanded for workflow path; repeat only for non-workflow path
            if self.workflow is None:
                data = data.repeat(repeat_times=self.config.val_override_config.get("n", 1), interleave=True)
            # After pop(), data.batch is an empty TensorDict whose internal
            # state is corrupted (stale flat-buffer / offset metadata).
            # Assigning keys into it via union then calling .cuda() produces
            # wrong data.  Replace with a fresh empty TensorDict first.
            if self.workflow is not None:
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

        # repeat to align with repeated responses in rollout
        # workflow path already pre-expanded data by n; only repeat for non-workflow path
        if self.workflow is None:
            data = data.repeat(repeat_times=self.config.n, interleave=True)
        else:
            data.batch = TensorDict(source={}, batch_size=data.batch.batch_size)
        new_batch = data.union(gen_batch_output)

        # compute rewards first
        reward_tensor, reward_metrics = self.reward.compute_rewards(new_batch)
        new_batch.batch["token_level_scores"] = reward_tensor
        # store per-sample reward metrics into batch for later slicing/aggregation
        for k, v in reward_metrics.items():
            new_batch.batch[f"reward_{k}"] = torch.tensor(v, dtype=torch.float32, device=reward_tensor.device)

        new_batch = filter(new_batch)
        if self.rollout_processor is not None and val is False:
            new_batch = self.rollout_processor.post_rollout(
                new_batch,
                self.reward,
                self,
            )
        return new_batch
