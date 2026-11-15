import torch
import logging
import os

from rllava.roles.config import ActorConfig
from rllava.data.protocol import DataProto
from tqdm import tqdm
from .actor import Actor
from rllava.model.patch.monkey_patch import apply_monkey_patch
from rllava.model.builder import build_model
from rllava.utils.model_utils import print_model_size
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from contextlib import nullcontext
from .base import PolicyRole



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


class Ref(PolicyRole):
    def __init__(self, config: ActorConfig, tokenizer=None, processor=None, *, layout):
        super().__init__(config, tokenizer, processor, layout=layout)
        self.model = None

    def initialize(self, actor: Actor):
        self.actor = actor

        if not actor.config.model.use_peft:
            log_gpu_memory_usage(f"Before init Ref from HF AutoModel", logger=logger)
            self.model, self.flops_counter = build_model(self.config.model, 
                                                         self.tokenizer, 
                                                         device_mesh=self.accelerator.device_mesh,
                                                         torch_dtype=torch.bfloat16,
                                                         trainerble=False)
            
            log_gpu_memory_usage(f"After init Ref from HF AutoModel", logger=logger)
            self.model = self.accelerator.prepare(self.model)
            log_gpu_memory_usage(f"After Ref Accelerator prepare", logger=logger)

    @torch.no_grad()
    def compute_log_probs(self, data: DataProto, temperature: float = None) -> torch.Tensor:
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if "multi_modal_inputs" in data.non_tensor_batch.keys() else []

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.model.dynamic_batching:
            max_token_len = self.config.log_prob_micro_batch_size * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len, dp_group=self.layout.groups.get("data"))
        else:
            size = self.config.log_prob_micro_batch_size
            micro_batches = [data[start:start + size] for start in range(0, len(data), size)]

        log_probs_lst = []
        if self.accelerator.is_main_process:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        if self.actor.config.model.use_peft:
            ctx = self.actor.model.disable_adapter()
            model = self.actor.model
        else:
            ctx = nullcontext()
            model = self.model
        with ctx, self.accelerator.eval(model):
            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_probs, _ = self.forward_batch(model=model, micro_batch=model_inputs, temperature=temperature)
                log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.model.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
        return log_probs
