"""Learner: learning-signal preparation and policy/value updates."""

from __future__ import annotations

import torch
import os
import numpy as np
from dataclasses import replace
from typing import TYPE_CHECKING
from rllava.roles.actor import Actor
from rllava.roles.critic import Critic
from rllava.roles.ref import Ref
from rllava.train.core_algos import kl_penalty, get_kl_controller
from rllava.data.protocol import DataProto
from rllava.data.data_utils import process_image, process_video
from rllava.utils import torch_functional as VF
from rllava.distribute.dist_utils import on_coordinator
from rllava.utils.py_functional import timer
from .advantage import get_adv_estimator
from .policy_loss import get_policy_loss

if TYPE_CHECKING:
    from rllava.controller.config import RLConfig


class Learner:

    def __init__(self, config: RLConfig, tokenizer, processor, *, resources, dispatcher, adv_estimator=None, policy_loss=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._cache = {}
        self.resources = resources
        self.dispatcher = dispatcher
        policy_loss = policy_loss or get_policy_loss(config.actor.policy_loss.loss_mode)

        if config.algorithm.use_kl_in_reward and config.algorithm.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_critic = self.config.critic is not None

        if config.algorithm.adv_estimator == "gae" and not self.use_critic:
            raise ValueError("GAE requires a critic configuration")
        self.adv_estimator = adv_estimator or get_adv_estimator(config.algorithm.adv_estimator)
        actor_layout = self.resources.roles["actor"]
        actor_config = replace(config.actor, ppo_mini_batch_size=config.actor.ppo_mini_batch_size * config.rollout.n // actor_layout.dp_size)
        self.actor = Actor(actor_config, tokenizer=tokenizer, processor=processor,
                           policy_loss=policy_loss, layout=actor_layout)
        self.critic = None
        if self.use_critic:
            critic_layout = self.resources.roles["critic"]
            critic_config = replace(config.critic, ppo_mini_batch_size=config.critic.ppo_mini_batch_size // critic_layout.dp_size)
            self.critic = Critic(critic_config, tokenizer=tokenizer, processor=processor, layout=critic_layout)

        self.ref = None
        if config.algorithm.use_kl_loss or config.algorithm.use_kl_in_reward:
            ref_config = replace(actor_config, parallel=self.resources.roles["ref"].parallel)
            self.ref = Ref(ref_config, tokenizer=tokenizer, processor=processor,
                           layout=self.resources.roles["ref"])

    def initialize(self, training_steps):
        self.training_steps = training_steps
        self.actor.config.optim.training_steps = training_steps
        self.actor.initialize()
        if self.critic is not None:
            self.critic.config.optim.training_steps = training_steps
            self.critic.initialize()
        if self.ref is not None:
            self.ref.initialize(self.actor)

    def step(self, data: DataProto, global_step: int, timing_raw=None):
        """Prepare learning signals and update; return the batch and metrics."""
        timing_raw = {} if timing_raw is None else timing_raw
        self._cache.clear()
        try:
            with timer("log_prob", timing_raw):
                data = self.compute_log_probs(data)
            if self.use_critic:
                with timer("values", timing_raw):
                    data.batch["values"] = self.compute_values(data).detach()
            with timer("adv", timing_raw):
                data = self.compute_advantage(data)
            with timer("update", timing_raw):
                output = self.update_model(data, global_steps=global_step)
            return data, output
        finally:
            self._cache.clear()
            data.non_tensor_batch.pop("multi_modal_inputs", None)

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

    @staticmethod
    def _to_training_device(data):
        return data.to(torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

    def compute_values(self, data: DataProto):
        self._process_multi_modal_inputs(data)

        return self.critic.compute_values(data)

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

    def compute_advantage(self, data: DataProto):
        # Redistribute raw experience; rebuild derived model inputs on each rank.
        data.non_tensor_batch.pop("multi_modal_inputs", None)
        batch = self.dispatcher.gather(data)
        batch = on_coordinator(lambda: self._compute_advantage(batch),
                               src=self.resources.coordinator_rank, group=self.resources.control_group)
        return self._to_training_device(self.dispatcher.scatter(batch))

    def _compute_advantage(self, data: DataProto):
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

        data.non_tensor_batch.pop("multi_modal_inputs", None)

        if len(outputs) == 0:
            return DataProto()

        merged = outputs[0]
        for out in outputs[1:]:
            merged = merged.union(out)
        return merged

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        if self._cache.get("batch") is not data:
            self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            for multi_modal_data in data.non_tensor_batch["multi_modal_data"]:
                multi_modal_data = multi_modal_data or {}
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

                batch_multi_modal_inputs.append(multi_modal_inputs)

            self._cache["batch"] = data
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]
