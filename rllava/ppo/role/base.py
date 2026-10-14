import os
import math
import torch

from collections import defaultdict
from typing import Dict, Optional, Tuple

from rllava.engine import EngineFactory
from rllava.utils import torch_functional as VF
from rllava.utils.device import get_device_name
from rllava.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


class Role:
    def __init__(self, config, tokenizer=None, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device_name = get_device_name()

        self.accelerator = EngineFactory(config.strategy)(config)

    def prepare_multi_modal_inputs(self, micro_batch: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        if "multi_modal_inputs" not in micro_batch:
            return {}

        multi_modal_inputs = defaultdict(list)
        for input_dict in micro_batch["multi_modal_inputs"]:
            for key, value in input_dict.items():
                multi_modal_inputs[key].append(value)

        prepared_inputs = {}
        for key, value in multi_modal_inputs.items():
            prepared_inputs[key] = torch.cat(value, dim=0) if len(value) != 0 else None

        return prepared_inputs

    def unwrap_model(self):
        return self.accelerator.unwrap_model(self.model)

    def unwrap_model_for_generation(self):
        return self.accelerator.unwrap_model_for_generation(self.model, self.config.model.use_peft)

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


class PolicyRole(Role):
    def __init__(self, config, tokenizer=None, processor=None):
        super().__init__(config, tokenizer, processor)

        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def forward_batch(
        self,
        model,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_logits: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if temperature is None or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Actor/reference log probabilities require a finite positive temperature.")
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = self.prepare_multi_modal_inputs(micro_batch)

        logits_slice = None
        logits_for_entropy = None

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

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.config.model.ulysses_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.config.model.ulysses_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.config.model.ulysses_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)
                # ((total_nnz / sp) + pad)
                log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.config.model.ulysses_size > 1:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad back to (bsz, seqlen)
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                need_logits = return_logits or calculate_entropy
                if need_logits:
                    if self.config.model.ulysses_size > 1:
                        logits_rmpad = gather_outputs_and_unpad(
                            logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    full_logits = pad_input(
                        hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )  # (bsz, seqlen, vocab)
                    logits_slice = full_logits[:, -response_length - 1 : -1, :]
                    if calculate_entropy:
                        logits_for_entropy = logits_slice
            else:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits: torch.Tensor = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = self.log_probs_from_logits(logits, micro_batch["responses"])  # (bsz, response_length)
                logits_slice = logits if return_logits else None
                if calculate_entropy:
                    logits_for_entropy = logits

        # Calculate entropy if requested
        entropy = None
        if calculate_entropy and logits_for_entropy is not None:
            probs = torch.softmax(logits_for_entropy, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (bsz, response_length)

        return entropy, log_probs, logits_slice
