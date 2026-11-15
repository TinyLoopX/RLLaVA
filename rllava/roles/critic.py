import os
import torch
import logging
import numpy as np
import torch.distributed as dist
from rllava.roles.config import CriticConfig
from tqdm import tqdm
from codetiming import Timer
from collections import defaultdict
from rllava.train.core_algos import compute_value_loss
from transformers import AutoModelForTokenClassification
from rllava.data.protocol import DataProto
from rllava.model.builder import build_model, build_optimizer
from rllava.utils.py_functional import append_to_dict
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from rllava.utils.logger.aggregate_logger import print_rank_0
from rllava.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import Role

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


class Critic(Role):
    def __init__(self, config: CriticConfig, tokenizer=None, processor=None, *, layout):
        super().__init__(config, tokenizer, processor, layout=layout)
        self.optimizer = None
        self.lr_scheduler = None

        print_rank_0(f"Critic will use mini batch size on per device {self.config.ppo_mini_batch_size}.")

    def initialize(self):
        log_gpu_memory_usage(f"Before init Critic from HF AutoModel", logger=logger)
        self.model, self.flops_counter = build_model(self.config.model, 
                                                     self.tokenizer, 
                                                     device_mesh=self.accelerator.device_mesh,
                                                     model_class=AutoModelForTokenClassification)

        log_gpu_memory_usage(f"After init Critic from HF AutoModel", logger=logger)
        self.model = self.accelerator.prepare(self.model)
        log_gpu_memory_usage(f"After Critic Accelerator prepare", logger=logger)
        
        self.optimizer, self.lr_scheduler = build_optimizer(self.config.optim, self.model)
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)
        log_gpu_memory_usage(f"After Optimizer and LR Scheduler Accelerator prepare", logger=logger)

    def load_checkpoint(self, checkpoint_path: str):
        self.accelerator.load_state(self.model, self.optimizer, self.lr_scheduler, checkpoint_path) 

    def save_checkpoint(self, checkpoint_path: str, save_model_only: bool = False):
        if save_model_only:
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
        else:
            # Call save_state on ALL ranks; Accelerate will coordinate and only write once.
            self.accelerator.save_state(self.model, self.optimizer, self.lr_scheduler, checkpoint_path)
        self.accelerator.wait_for_everyone()

    def forward_batch(self, model, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = self._prepare_multi_modal_inputs(micro_batch)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.config.model.padding_free:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.config.model.ulysses_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.config.model.ulysses_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(model, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.config.model.ulysses_size > 1:
                    values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1 : -1]
            else:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                if hasattr(model, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length - 1 : -1].squeeze(-1)  # (bsz, response_length, vocab_size)

        return values

    @torch.no_grad()
    def compute_values(self, data: DataProto) -> torch.Tensor:
        data = data.to(torch.cuda.current_device())
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "response_mask" in data.batch:
            select_keys.append("response_mask")
        non_tensor_select_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch.keys() else []

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.model.dynamic_batching:
            max_token_len = self.config.log_prob_micro_batch_size * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len, dp_group=self.layout.groups.get("data"))
        else:
            size = self.config.log_prob_micro_batch_size
            micro_batches = [data[start:start + size] for start in range(0, len(data), size)]

        values_lst = []
        if self.accelerator.is_main_process:
            micro_batches = tqdm(micro_batches, desc="Compute values", position=1)

        model = self.model
        with self.accelerator.eval(model):
            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                values = self.forward_batch(model=model, micro_batch=model_inputs)
                values_lst.append(values)

        values = torch.concat(values_lst, dim=0)

        if self.config.model.dynamic_batching:
            values = restore_dynamic_batch(values, batch_idx_list)

        if "response_mask" in data.batch:
            values = values * data.batch["response_mask"]  

        return values
    
    def _compute_values(self, module, data: DataProto):
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.log_prob_micro_batch_size
        )
        values_lst = []
        if self.worker.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute values", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            values = self.forward_batch(model=module, micro_batch=model_inputs)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length:]
        return values
    
    def _update(self, data: DataProto):
        self.model.train()

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.accelerator.is_main_process:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.model.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.ppo_micro_batch_size * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len, dp_group=self.layout.groups.get("data"))
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                if self.accelerator.is_main_process:
                    micro_batches = tqdm(micro_batches, desc="Update critic", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    vpreds = self.forward_batch(model=self.model, micro_batch=model_inputs)
                    vf_loss, vf_metrics = compute_value_loss(
                        vpreds=vpreds,
                        returns=returns,
                        values=values,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    loss = vf_loss * torch.sum(response_mask) * self.accelerator_manager.accelerator.num_processes / total_response_tokens
                    loss.backward()

                    batch_metrics = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_metrics["vf_clipfrac"],
                        "critic/vpred_mean": vf_metrics["vpred_mean"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        return metrics
    
    def update(self, data: DataProto) -> DataProto:
        """Update critic network.
        
        This method contains the algorithm logic for critic updates.
        Distributed operations are handled by the worker.
        
        Args:
            batch: Training batch data
            
        Returns:
            Training metrics
        """
        # Execute with distributed operations handled by worker
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self._update(data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu_critic"] = (
            estimated_flops * self.config.ppo_epochs / (promised_flops * self.worker.world_size)
        )
        
        # Add learning rate and other metrics
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        metrics["critic/lr"] = lr
        
        # Wrap metrics in DataProto
        return DataProto(
            non_tensor_batch={
                key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
            }
        )