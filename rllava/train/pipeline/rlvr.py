from torch.utils.data import DataLoader
from tqdm import tqdm
from rllava.ppo.utils.metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from rllava.utils.py_functional import timer
from rllava.train.pipeline.base import Pipeline
from rllava.utils.config import BaseConfig, init_config
from rllava.distribute.dist_utils import init_dist, is_rank0
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
                with timer("gen", timing_raw):
                    batch = self.model.rollout_batch(self.data_iterator, global_step=self.global_steps)

                with timer("log_prob", timing_raw):
                    batch = self.model.compute_log_probs(batch)

                with timer("adv", timing_raw):
                    batch = self.model.compute_advantage(batch)

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
                # collect metrics (must run on all ranks before rank0 logs to avoid deadlocks)
                for key in batch.batch.keys():
                    if isinstance(key, str) and key.startswith("reward_"):
                        metrics[f"reward/{key[len('reward_'):]}"] = batch.batch[key].mean().detach().item()
                metrics.update(batch.meta_info.get("adv_metrics", {}))
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.model.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=self.model.resources.world_size))
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
    try:
        pipeline = RLVRPipeline(config=config,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                model=model)
        pipeline.run()
    finally:
        model.close()



if __name__ == "__main__":
    init_dist()
    main()
