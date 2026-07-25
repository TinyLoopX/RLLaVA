import torch
import itertools
from collections import defaultdict
from typing import Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from rllava.utils.logger.aggregate_logger import print_rank_0
from rllava.ppo.utils.metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, compute_length_metrics
from RLLaVA.rllava.workflow.think_with_image import MultiTurnWorkflow
from rllava.utils.py_functional import timer
from rllava.train.pipeline.base import Pipeline
from rllava.utils.config import BaseConfig, init_config
from rllava.utils.dist_utils import init_dist, is_rank0, dist_batch, gather_batch
from rllava.ppo import PPOConfig, PPO, Rollout
from rllava.ppo.plugins import get_adv_estimator, get_policy_loss, PrefixRolloutProcessor, HPTBatchRolloutProcessor
from rllava.ppo.role.reward import Reward
from rllava.data.data_loader import create_dataloader
from rllava.utils.tokenizer import load_tokenizer_and_processor



class RLVRPipeline(Pipeline):

    def __init__(self, model, config: BaseConfig, train_dataloader: DataLoader, val_dataloader: DataLoader):
        model.initialize(train_dataloader)
        
        super().__init__(model, config, train_dataloader, val_dataloader)
        self.config: PPOConfig = self.config  # Explicitly specify the type of self.config
        self.model: PPO = self.model

    def validate(self, metrics: dict[str, Any]=dict()) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print_rank_0("Start validation...")
        with self.model.generate_context():
            iterator = iter(self.val_dataloader)
            while True:
                test_batch = dist_batch(iterator)
                if test_batch is None:
                    break

                test_batch = self.model.rollout.generate_one_batch(test_batch, val=True)
                test_batch = gather_batch(test_batch)
                
                if is_rank0():
                    # evaluate using reward_function
                    reward_tensor, reward_metrics = self.model.compute_rewards(test_batch)
    
                    # store generations
                    input_ids = test_batch.batch["prompts"]
                    input_texts = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                    output_ids = test_batch.batch["responses"]
                    output_texts = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    scores = reward_tensor.sum(-1).cpu().tolist()
                    sample_inputs.extend(input_texts)
                    sample_outputs.extend(output_texts)
                    sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
                    sample_scores.extend(scores)
    
                    reward_tensor_lst.append(reward_tensor)
                    for key, value in reward_metrics.items():
                        reward_metrics_lst[key].extend(value)
    
                    for key, value in compute_length_metrics(test_batch).items():
                        length_metrics_lst[key].append(value)

                break

        if is_rank0():
            self.maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
    
            self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
            val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
            val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
            
            metrics.update({"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics})
            return metrics

    def run(self):

        if self.config.trainer.val_before_train:
            val_metrics = self.validate()
            self.logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.val_only:
                return

        self.data_iterator = itertools.cycle(self.train_dataloader)
        
        for _ in tqdm(range(self.training_steps), initial=self.global_steps, desc="Training Progress", disable=not is_rank0()):
            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    batch = self.model.rollout_batch(self.data_iterator, global_step=self.global_steps)

                with timer("log_prob", timing_raw):
                    batch = self.model.compute_log_probs(batch)

                with timer("adv", timing_raw):
                    batch = self.model.compute_advantage(batch)

                with timer("update", timing_raw):
                    output = self.model.update_model(batch, self.training_steps)
                    
                batch = gather_batch(batch)
                output = gather_batch(output)

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
                # collect metrics (must run on all ranks before rank0 logs to avoid deadlocks)
                for key in batch.batch.keys():
                    if isinstance(key, str) and key.startswith("reward_"):
                        metrics[f"reward/{key[len('reward_'):]}"] = batch.batch[key].mean().detach().item()
                metrics.update(batch.meta_info.get("adv_metrics", {}))
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.model.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=1))
                metrics.update(reduce_metrics(output.non_tensor_batch))

                self.logger.log(data=metrics, step=self.global_steps)



def main():
    
    # Initialize configuration using PPOConfig structure
    config = init_config(PPOConfig)

    # Load tokenizer and processor based on model path and chat template
    tokenizer, processor = load_tokenizer_and_processor(config.actor.model.model_path, config.data.override_chat_template)
    
    # Create training and validation dataloaders
    train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

    # Initialize advantage estimator (e.g., GRPO) specified in the configuration
    adv_estimator = get_adv_estimator(getattr(config.algorithm, "adv_estimator", "grpo")) 
    # Define the policy loss function (e.g., 'vanilla' PPO loss)
    policy_loss = get_policy_loss(getattr(config.actor.policy_loss, "loss_mode", 'vanilla'))
    # Initialize the reward model using the configuration and tokenizer
    reward = Reward(config.reward, tokenizer)
    # Initialize the Rollout module which manages data generation and processing during training
    rollout = Rollout(
        config.rollout,
        reward,
        tokenizer,
        processor,
        workflow=MultiTurnWorkflow,
        rollout_processor=None,
    )
    # Initialize the PPO model with all components (config, tokenizer, reward, rollout, etc.)
    model = PPO(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        reward=reward,
        rollout=rollout,
        adv_estimator=adv_estimator,
        policy_loss=policy_loss,
    )

    # Initialize and run the RL training pipeline with the configured model and dataloaders
    pipeline = RLVRPipeline(config=config,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            model=model)
    pipeline.run()



if __name__ == "__main__":
    init_dist()
    main()