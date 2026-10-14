import torch
import os
import numpy as np
import torch.distributed as dist
from contextlib import contextmanager
from collections import defaultdict
from .config import PPOConfig
from .role.rollout import Rollout
from .role.actor import Actor
from .role.critic import Critic
from .role.reward import Reward
from .role.ref import Ref
from .utils.core_algos import kl_penalty, get_kl_controller
from rllava.data.protocol import DataProto
from rllava.data.data_utils import process_image, process_video
from rllava.utils import torch_functional as VF
from rllava.utils.dist_utils import (
    dist_gather_then_scatter,
    dist_batch,
    gather_batch,
)
from rllava.utils.logger.aggregate_logger import print_rank_0



class PPO():
    
    def __init__(self, config: PPOConfig, tokenizer, processor, reward: Reward, rollout: Rollout=None, adv_estimator=None, policy_loss=None):
        self.config = config
        self.train_batch_size = config.data.train_batch_size
        self.tokenizer = tokenizer
        self.processor = processor
        self._cache = {}

        if config.algorithm.use_kl_in_reward and config.algorithm.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_critic = self.config.critic is not None
        
        if self.config.data.train_batch_size % self.config.actor.ppo_mini_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")
        
        if (
            self.config.data.train_batch_size * self.config.rollout.n
        ) % self.config.actor.log_prob_micro_batch_size != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )
        
        self.config.actor.ppo_mini_batch_size *= config.rollout.n
        print_rank_0(f"Actor will use global batch size {self.config.actor.ppo_mini_batch_size}.")

        if self.use_critic:
            if self.config.data.train_batch_size % self.config.critic.ppo_mini_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                self.config.data.train_batch_size * self.config.rollout.n
            ) % self.config.critic.log_prob_micro_batch_size != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        # Injected plugins/components (with sensible fallbacks)
        self.adv_estimator = adv_estimator

        self.actor = Actor(config.actor, 
                           tokenizer=tokenizer,
                           processor=processor,
                           policy_loss=policy_loss)
        self.critic = Critic(config.critic, 
                             tokenizer=tokenizer, 
                             processor=processor) if config.critic is not None else None

        self.ref = None
        if config.algorithm.use_kl_loss or config.algorithm.use_kl_in_reward:
            if not self.actor.config.model.use_peft:
                self.ref = Ref(config.actor, tokenizer=tokenizer, processor=processor)

        self.reward = reward
        self.rollout = rollout

        def online_sampling(batch: DataProto) -> DataProto:
            if self.config.algorithm.online_filtering:
                key = f"reward_{self.config.algorithm.filter_key}"
                # convert to python list for numpy ops
                filter_scores = batch.batch[key].detach().cpu().tolist()
                uids = batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)
                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                batch = batch[kept_sample_idxs]
            return batch
        
        self.filter = online_sampling

    def initialize(self, train_dataloader):
        self.train_dataloader = train_dataloader
        self.training_steps = self.get_training_steps(train_dataloader)

        self.rollout.initialize(self.config.actor.model.model_path)

        self.config.actor.optim.training_steps = self.training_steps
        self.actor.initialize()

        if self.critic is not None:
            self.config.critic.optim.training_steps = self.training_steps
            self.critic.initialize()

        if self.ref is not None:
            self.ref.initialize(self.actor)
        
        self.reward.initialize()

    def get_training_steps(self, train_dataloader):  
        if self.config.trainer.max_steps is not None:
            return self.config.trainer.max_steps
        else:
            return len(train_dataloader) * self.config.trainer.total_epochs

    def load_checkpoint(self, checkpoint_path: str):
        actor_path = os.path.join(checkpoint_path, "actor")
        self.actor.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(checkpoint_path, "critic")
            self.critic.load_checkpoint(critic_path)

    def save_checkpoint(self, checkpoint_path: str):
        actor_path = os.path.join(checkpoint_path, "actor")
        self.actor.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)
        if self.use_critic:
            critic_path = os.path.join(checkpoint_path, "critic")
            self.critic.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

    def rollout_batch(self, data_iterator, global_step: int | None = None) -> DataProto:
        batch = None
        num_try_make_batch = 0
        print_rank_0("Start generating batch...")
        with self.generate_context():
            while True:
                num_try_make_batch += 1
                batch_in = dist_batch(data_iterator)

                new_batch = self.rollout.generate_one_batch(batch_in, self.filter, global_step=global_step)
                new_batch = gather_batch(new_batch)

                batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
                current_batch_size = len(batch) // self.rollout.n
                rollout_batch_size = self.train_batch_size
                if current_batch_size < rollout_batch_size:
                    print_rank_0(f"{current_batch_size=} < {rollout_batch_size=}")
                    max_try_make_batch = self.config.trainer.max_try_make_batch
                    if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                        print_rank_0(f"{num_try_make_batch=}. Continue generating...")
                    else:
                        raise ValueError(f"{num_try_make_batch=} >= {max_try_make_batch=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                else:
                    print_rank_0(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                    result = batch[: self.train_batch_size * self.rollout.n]
                    result.meta_info["global_token_num"] = torch.sum(result.batch["attention_mask"], dim=-1).tolist()
                    result = result.chunk(dist.get_world_size())[dist.get_rank()]
                    return result

        
    @contextmanager
    def generate_context(self):
        
        with self.actor.unwrap_model_for_generation() as unwrapped_model:
            self.rollout.rollout_engine.load(unwrapped_model)

        yield
        self.rollout.rollout_engine.offload()

    def compute_values(self, data: DataProto):
        self._process_multi_modal_inputs(data)

        return self.critic.compute_values(data)

    def compute_rewards(self, data: DataProto):
        return self.reward.compute_rewards(data)

    def compute_log_probs(self, data: DataProto):
        self._process_multi_modal_inputs(data)
        temperature = self.config.rollout.temperature
        temperatures = data.batch["rollout_temperature"]
        if temperatures.shape != (data.batch["responses"].shape[0],) or not torch.allclose(
            temperatures, torch.full_like(temperatures, temperature), rtol=1e-5, atol=1e-6
        ):
            raise ValueError("Rollout temperature differs from the temperature used to recompute actor probabilities.")
        mask = data.batch["response_mask"].bool()
        if mask.shape != data.batch["responses"].shape or not torch.equal(mask, data.batch["attention_mask"][:, -mask.shape[1]:].bool()):
            raise ValueError("Response tokens, response_mask and attention_mask are misaligned.")
        # Freeze the pre-update policy once, also for single-minibatch runs and
        # KL-in-reward. The sampler probabilities remain a separate tensor.
        output = self.actor.compute_log_probs(data, temperature=temperature)
        data.batch["old_log_probs"] = output.detach().float().masked_fill(~mask, 0.0)
        data.meta_info["tis_imp_ratio_cap"] = self.config.actor.tis_imp_ratio_cap

        # compute ref_log_probs if needed
        if self.ref is not None:
            output = self.ref.compute_log_probs(data, temperature=temperature)
            data = data.union(DataProto.from_dict(tensors={"ref_log_probs": output}))
        return data

    def apply_kl_penalty(self, data: DataProto, kl_ctrl, kl_p="kl"):
        """Apply KL penalty to the token-level rewards.

        This function computes the KL divergence between the reference policy and current policy,
        then applies a penalty to the token-level rewards based on this divergence.
    
        Args:
            data (DataProto): The data containing batched model outputs and inputs.
            kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
            kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
    
        Returns:
            tuple: A tuple containing:
                - The updated data with token-level rewards adjusted by KL penalty
                - A dictionary of metrics related to the KL penalty
        """
        response_mask = data.batch["response_mask"]
        token_level_scores = data.batch["token_level_scores"]
        batch_size = data.batch.batch_size[0]
        

        kld = kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_p)
        kld = kld * response_mask
        beta = kl_ctrl.value

        data.batch["token_level_rewards"] = token_level_scores - beta * kld

        current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)
        current_kl = torch.mean(current_kl, dim=0).item()
        
        kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
        metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
        
        return data, metrics
    
    @dist_gather_then_scatter()
    def compute_advantage(self, data: DataProto):
        adv_metrics = {}
        # apply kl penalty if needed
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = self.apply_kl_penalty(
                data, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty)
            adv_metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        advantages, returns = self.adv_estimator(data, self.config.algorithm)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["adv_metrics"] = adv_metrics
        return data
    
    def update_model(self, data: DataProto, global_steps) -> DataProto:
        self._process_multi_modal_inputs(data)

        data.meta_info["use_kl_loss"] = self.config.algorithm.use_kl_loss
        data.meta_info["temperature"] = self.config.rollout.temperature
        outputs = []

        if self.use_critic:
            critic_output = self.critic.update(data)
            outputs.append(critic_output)

        if self.config.trainer.critic_warmup <= global_steps:
            actor_output = self.actor.update(data)
            outputs.append(actor_output)

        if len(outputs) == 0:
            return DataProto()
            
        merged = outputs[0]
        for out in outputs[1:]:
            merged = merged.union(out)
        return merged
    
    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        if "uid" in self._cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
            self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}  # avoid repeated processing for n > 1 samples
            for index, multi_modal_data in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
            ):  # process multi modal data per sample
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    if "images" in multi_modal_data and multi_modal_data["images"] is not None:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels, self.processor))

                    if "videos" in multi_modal_data and multi_modal_data["videos"] is not None:
                        for video in multi_modal_data["videos"]:
                            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

                    if len(images) != 0:
                        # it's necessary to add `dict` to properly convert batch features to dict
                        # otherwise the batch features will be converted to dict keys
                        # see https://github.com/hiyouga/EasyR1/pull/339
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif len(videos) != 0:
                        multi_modal_inputs = dict(
                            self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                        )
                    else:
                        multi_modal_inputs = {}

                    for key in multi_modal_inputs:
                        if isinstance(multi_modal_inputs[key], torch.Tensor):
                            multi_modal_inputs[key] = multi_modal_inputs[key].to(data.batch["input_ids"].device)

                    multi_modal_inputs_cache[index] = multi_modal_inputs

                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]
