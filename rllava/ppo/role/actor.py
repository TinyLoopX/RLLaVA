import torch
import psutil
import numpy as np
import logging
import os

import torch.nn.functional as F
from ..config import ActorConfig
from tqdm import tqdm
from codetiming import Timer
from collections import defaultdict
from ..utils.core_algos import kl_penalty, agg_loss
from rllava.ppo.plugins.policy_loss import (
    get_policy_loss_extra_keys,
    build_policy_loss_kwargs,
)
from rllava.data.protocol import DataProto
from rllava.utils.device import get_torch_device
from rllava.model.builder import build_model, build_optimizer
from rllava.utils.py_functional import append_to_dict
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from rllava.utils.logger.aggregate_logger import print_rank_0
from .base import PolicyRole



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


class Actor(PolicyRole):
    def __init__(self, config: ActorConfig, tokenizer=None, processor=None, policy_loss=None):
        super().__init__(config, tokenizer, processor)
        self.policy_loss = policy_loss
        self.optimizer = None
        self.lr_scheduler = None

        self.config.ppo_mini_batch_size = self.config.ppo_mini_batch_size // self.accelerator.num_processes
        if self.config.ppo_mini_batch_size == 0:
            raise ValueError(f"Actor mini batch size on per device must be larger than 0.")

        if self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size != 0:
            raise ValueError(f"Actor mini batch size on per device must be divisible by the micro batch size.")
        
        print_rank_0(f"Actor will use mini batch size on per device {self.config.ppo_mini_batch_size}.")

    def initialize(self):
        log_gpu_memory_usage(f"Before init Actor from HF AutoModel", logger=logger)
        self.model, self.flops_counter = build_model(self.config.model, 
                                                     self.tokenizer, 
                                                     device_mesh=self.accelerator.device_mesh)

        log_gpu_memory_usage(f"After init Actor from HF AutoModel", logger=logger)
        self.model = self.accelerator.prepare(self.model)
        log_gpu_memory_usage(f"After Actor Accelerator prepare", logger=logger)
        
        self.optimizer, self.lr_scheduler = build_optimizer(self.config.optim, self.model)
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)
        log_gpu_memory_usage(f"After Optimizer and LR Scheduler Accelerator prepare", logger=logger)

    @torch.no_grad()
    def compute_log_probs(self, data: DataProto, temperature: float = None) -> torch.Tensor:
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch.keys() else []

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.model.dynamic_batching:
            max_token_len = self.config.log_prob_micro_batch_size * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.log_prob_micro_batch_size)

        log_probs_lst = []
        if self.accelerator.is_main_process:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        model = self.model
        with self.accelerator.eval(model):
            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_probs, _ = self.forward_batch(model=model, micro_batch=model_inputs, temperature=temperature)
                log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.model.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
        return log_probs
    
    def update(self, data: DataProto):
        # Perform training with Accelerator
        with Timer(name="update_policy", logger=None) as timer:
            with self.accelerator.train(model=self.model, optimizer=self.optimizer):
                metrics = self.update_policy(data=data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        
        # Use accelerator's world size
        world_size = self.accelerator.num_processes
        metrics["perf/mfu/actor"] = estimated_flops * self.config.ppo_epochs / promised_flops / world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        lr = self.lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr
        self.lr_scheduler.step()

        output = DataProto(
               non_tensor_batch={
                   key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
               }
           )

        return output
    
    def update_policy(self, data: DataProto):
        """Update policy using Accelerator for unified training management."""

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        use_kl_loss = data.meta_info["use_kl_loss"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "advantages",
            "old_log_probs",
        ]
        for key in get_policy_loss_extra_keys(self.policy_loss):
            if key in data.batch.keys():
                select_keys.append(key)
        
        if use_kl_loss:
            select_keys.append("ref_log_probs")

        if self.config.tis_imp_ratio_cap > 0:
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.ppo_mini_batch_size)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1, disable=not self.accelerator.is_main_process)

            for mini_batch in mini_batches:
                if self.config.model.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.ppo_micro_batch_size * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.optimizer.zero_grad()

                # Convert to list to get length for determining last step
                micro_batches_list = list(micro_batches)
                num_micro_batches = len(micro_batches_list)
                
                micro_batches_iter = tqdm(enumerate(micro_batches_list), total=num_micro_batches, desc="Update policy", position=2, disable=not self.accelerator.is_main_process)
                for micro_batch_idx, micro_batch in micro_batches_iter:
                    is_last_micro_batch = (micro_batch_idx == num_micro_batches - 1)
                    micro_batch_metrics = defaultdict(list)
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    if self.config.model.dynamic_batching:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if self.config.entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_probs, logits_slice = self.forward_batch(
                        model=self.model,
                        micro_batch=model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_logits=False,
                    )

                    old_log_probs = model_inputs["old_log_probs"].detach()

                    policy_loss_kwargs = build_policy_loss_kwargs(
                        self.policy_loss, model_inputs=model_inputs, metrics=micro_batch_metrics
                    )

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = self.policy_loss(
                        old_log_prob=old_log_probs,
                        log_prob=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                        entropy=entropy,
                        **policy_loss_kwargs,
                    )

                    if self.config.entropy_coeff != 0:
                        # entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode='token-mean')
                        micro_batch_metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if use_kl_loss:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_probs,
                            ref_logprob=ref_log_probs,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
     
                    loss = policy_loss * loss_scale_factor
                    # For DeepSpeed: pass is_last_step to trigger allreduce on the last micro_batch
                    self.accelerator.backward(loss, is_last_step=is_last_micro_batch)

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac_higher": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self.accelerator.optimizer_step(self.model, self.optimizer)
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.optimizer.zero_grad()
        return metrics

