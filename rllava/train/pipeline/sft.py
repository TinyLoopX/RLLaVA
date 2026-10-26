import torch
import numpy as np
import uuid
from torch.utils.data import DataLoader
from tqdm import tqdm
from rllava.ppo.utils.metrics import compute_throughout_metrics, compute_timing_metrics, reduce_metrics, compute_length_metrics
from rllava.utils.py_functional import timer
from rllava.train.pipeline.base import Pipeline
from rllava.utils.config import BaseConfig, init_config
from rllava.distribute.dist_utils import init_dist, is_rank0, on_coordinator
from rllava.ppo import PPOConfig, PPO, Rollout
from rllava.data.protocol import DataProto
from rllava.ppo.plugins import get_policy_loss
from rllava.ppo.role.reward import Reward
from rllava.data.data_loader import create_dataloader
from rllava.utils.tokenizer import load_tokenizer_and_processor



class SFTPipeline(Pipeline):

    def __init__(self, model, config: BaseConfig, train_dataloader: DataLoader, val_dataloader: DataLoader):
        model.initialize(train_dataloader)
        
        super().__init__(model, config, train_dataloader, val_dataloader)
        self.config: PPOConfig = self.config  # Explicitly specify the type of self.config
        self.model: PPO = self.model

    def _prepare_sft_batch(self, batch):
        pad_id = self.model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.model.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for SFT training.")

        prompt_input_ids = batch.batch["input_ids"].clone()
        prompt_attention_mask = batch.batch["attention_mask"].clone()
        prompt_position_ids = batch.batch["position_ids"].clone()
        max_response_length = self.config.data.max_response_length

        if "tgt_input_ids" in batch.batch.keys():
            responses = batch.batch["tgt_input_ids"]
            response_mask = responses.ne(pad_id)
        elif "tgt_input_ids" in batch.non_tensor_batch.keys():
            padded_targets = []
            for target_ids in batch.non_tensor_batch["tgt_input_ids"]:
                if isinstance(target_ids, torch.Tensor):
                    token_ids = target_ids.tolist()
                elif hasattr(target_ids, "tolist"):
                    token_ids = target_ids.tolist()
                else:
                    token_ids = list(target_ids)
                token_ids = token_ids[:max_response_length]
                if len(token_ids) < max_response_length:
                    token_ids = token_ids + [pad_id] * (max_response_length - len(token_ids))
                padded_targets.append(token_ids)
            responses = torch.tensor(padded_targets, dtype=torch.long, device=prompt_input_ids.device)
            response_mask = responses.ne(pad_id)
        elif "labels" in batch.batch.keys():
            labels = batch.batch["labels"]
            ignore_index = -100
            response_mask = labels.ne(ignore_index) & labels.ne(pad_id)
            responses = torch.where(response_mask, labels, torch.full_like(labels, pad_id))
        else:
            raise ValueError("SFT training requires `tgt_input_ids` or `labels` in the batch.")

        response_length = responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=prompt_input_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(prompt_input_ids.size(0), -1)
        if prompt_position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(prompt_input_ids.size(0), 1, -1).expand(
                prompt_input_ids.size(0), prompt_position_ids.size(1), -1
            )
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id

        new_input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
        new_attention_mask = torch.cat(
            [prompt_attention_mask, response_mask.to(dtype=prompt_attention_mask.dtype)], dim=-1
        )
        new_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        new_response_mask = response_mask.to(dtype=prompt_attention_mask.dtype)

        batch.batch["prompts"] = prompt_input_ids
        batch.batch["responses"] = responses
        batch.batch["response_mask"] = new_response_mask
        batch.batch["input_ids"] = new_input_ids
        batch.batch["attention_mask"] = new_attention_mask
        batch.batch["position_ids"] = new_position_ids

        # Keep actor.update_policy contract in SFT mode.
        batch.batch["advantages"] = batch.batch["response_mask"].float()
        if "old_log_probs" not in batch.batch.keys():
            batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["advantages"])

        if "multi_modal_data" in batch.non_tensor_batch and "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch["input_ids"]))],
                dtype=object,
            )

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["min_pixels"] = self.config.data.min_pixels
        batch.meta_info["max_pixels"] = self.config.data.max_pixels
        batch.meta_info["video_fps"] = self.config.data.video_fps

        from tensordict import TensorDict
        batch.batch = TensorDict(
            {k: v for k, v in batch.batch.items()},
            batch_size=[prompt_input_ids.size(0)],
        )

        return batch

    def run(self):

        if self.config.trainer.val_before_train:
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
                batch = on_coordinator(
                    lambda: self._prepare_sft_batch(DataProto.from_single_dict(next(self.data_iterator))),
                    src=self.model.resources.coordinator_rank, group=self.model.resources.control_group,
                )
                batch = self.model._to_training_device(self.model.dispatcher.scatter(batch))

                with timer("update", timing_raw):
                    output = self.model.update_model(batch, global_steps=self.global_steps)
                    
                batch = self.model.dispatcher.gather(batch)
                output = self.model.dispatcher.gather(output)

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
                metrics.update(compute_length_metrics(batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=self.model.resources.world_size))
                metrics.update(reduce_metrics(output.non_tensor_batch))

                self.logger.log(data=metrics, step=self.global_steps)



def main():
    
    # Initialize configuration using PPOConfig structure
    config = init_config(PPOConfig)
    # SFT pipeline should not depend on reference-model KL terms.
    config.algorithm.use_kl_loss = False
    config.algorithm.use_kl_in_reward = False
    if not config.data.dataset_class:
        config.data.dataset_class = "SFTDataset"

    config.rollout.n = 1

    # Load tokenizer and processor based on model path and chat template
    tokenizer, processor = load_tokenizer_and_processor(config.actor.model.model_path, config.data.override_chat_template)
    
    # Create training and validation dataloaders
    train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

    # Define the policy loss function for supervised training.
    policy_loss = get_policy_loss(getattr(config.actor.policy_loss, "loss_mode", "sft"))
    # Initialize the reward model using the configuration and tokenizer
    reward = Reward(config.reward, tokenizer)
    # Initialize the Rollout module which manages data generation and processing during training
    rollout = Rollout(
        config.rollout,
        reward,
        tokenizer,
        processor,
        rollout_processor=None,
    )
    # Initialize the PPO model with all components (config, tokenizer, reward, rollout, etc.)
    model = PPO(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        reward=reward,
        rollout=rollout,
        policy_loss=policy_loss,
    )

    # Initialize and run the RL training pipeline with the configured model and dataloaders
    try:
        pipeline = SFTPipeline(config=config,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                model=model)
        pipeline.run()
    finally:
        model.close()



if __name__ == "__main__":
    init_dist()
    main()
