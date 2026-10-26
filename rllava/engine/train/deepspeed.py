import os
import logging
import itertools
import torch
import torch.nn as nn
from types import MethodType
from contextlib import contextmanager
from typing import Any, Dict, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from trl.models.utils import is_deepspeed_available
from collections import OrderedDict
from accelerate.utils import is_compiled_module, has_compiled_regions
from accelerate.utils.transformer_engine import convert_model
from .base import TrainEngine
from .. import register_engine
from rllava.utils.config import DeepSpeedConfig
from rllava.utils.logger.aggregate_logger import log_with_rank
from rllava.utils.memory_utils import aggressive_empty_cache

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except ImportError:
    deepspeed = None


DS_INTERNAL_TAG = "deepspeed"


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


@register_engine("deepspeed")
class DeepSpeedAccelerator(TrainEngine):
    """Native DeepSpeed engine mirroring the TrainEngine contract."""

    def __init__(self, config, *, layout):
        super().__init__(config, layout=layout)
        if not is_deepspeed_available():
            raise ImportError(
                "DeepSpeed is not installed. Please install it before selecting the 'deepspeed' strategy."
            )

        self.ds_config: DeepSpeedConfig = getattr(config, "deepspeed", DeepSpeedConfig())
        self.ds_config_dict = self._build_ds_config_dict(config)

        self.module: Optional[nn.Module] = None
        self.ds_engine: DeepSpeedEngine = None
        self.ds_optimizer = None
        self.ds_scheduler = None

    def prepare(self, *args: Any, **kwargs: Any):
        prepared_items = [self._prepare_item(arg, **kwargs) for arg in args]
        self._maybe_initialize_engine()
        finalized_items = [self._finalize_item(arg) for arg in prepared_items]
        if len(finalized_items) == 1:
            return finalized_items[0]
        return tuple(finalized_items)

    def unwrap_model(self, model):
        if isinstance(model, DeepSpeedEngine):
            return model.module
        return model

    @contextmanager
    def unwrap_model_for_generation(self, model, is_peft_model: bool = False):
        aggressive_empty_cache(force_sync=True)

        unwrapped_model = self.unwrap_model(model)
        if is_peft_model:
            unwrapped_model.pretrained_model.disable_adapter()
        if self.ds_config.zero_stage == 3:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield self.unwrap_model(model)
                add_hooks(model)
        else:
            yield unwrapped_model

    @contextmanager
    def eval(self, model):
        model.eval()
        yield

    @contextmanager
    def train(self, model, optimizer: Optimizer):
        model.train()
        yield

    def load_state(self, model, optimizer, lr_scheduler, local_path: Optional[str]):
        if local_path is None:
            return

        _, client_state = self.ds_engine.load_checkpoint(local_path, tag=DS_INTERNAL_TAG)
        if lr_scheduler is not None and client_state is not None:
            lr_state = client_state.get("lr_scheduler")
            if lr_state is not None:
                lr_scheduler.load_state_dict(lr_state)
        self.wait_for_everyone()
        log_with_rank(f"Loaded DeepSpeed checkpoint from {local_path}", rank=self.rank, logger=logger)

    def save_state(self, model, optimizer, lr_scheduler, local_path: Optional[str]):
        if local_path is None:
            return
        os.makedirs(local_path, exist_ok=True)
        client_state = {"lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None}
        self.ds_engine.save_checkpoint(local_path, tag=DS_INTERNAL_TAG, client_state=client_state)
        self.wait_for_everyone()
        log_with_rank(f"Saved DeepSpeed checkpoint to {local_path}", rank=self.rank, logger=logger)

    def backward(self, loss: torch.Tensor, is_last_step: bool = True):
        """Perform backward pass with gradient accumulation support.
        
        Args:
            loss: The loss tensor to backpropagate.
            is_last_step: If True, this is the last micro_batch in the accumulation,
                         triggers allreduce. If False, gradients are accumulated locally.
        """
        # Set boundary based on whether this is the last accumulation step
        # boundary=True triggers allreduce in DeepSpeed ZeRO
        self.ds_engine.set_gradient_accumulation_boundary(is_boundary=is_last_step)
        kwargs = {"scale_wrt_gas": False}
        self.ds_engine.backward(loss, **kwargs)

    def optimizer_step(self, model: DeepSpeedEngine, optimizer):
        model.step()
        
        return model.get_global_grad_norm()

    def _build_ds_config_dict(self, config) -> Dict[str, Any]:
        per_device_mini_batch = max(1, getattr(config, "ppo_mini_batch_size", 1))
        micro_batch_size = max(1, getattr(config, "ppo_micro_batch_size", 1))
        gradient_accumulation_steps = max(1, per_device_mini_batch // micro_batch_size)
        
        train_batch_size = (
            micro_batch_size * gradient_accumulation_steps * self.layout.dp_size
        )

        dtype = (self.ds_config.torch_dtype or "").lower()
        bf16_enabled = dtype in {"bf16", "bfloat16"}
        fp16_enabled = dtype in {"fp16", "float16", "half"}
        zero_stage = int(self.ds_config.zero_stage)
        zero_config = {
            "stage": zero_stage,
            "contiguous_gradients": self.ds_config.contiguous_gradients,
            "overlap_comm": self.ds_config.overlap_comm,
            "reduce_scatter": self.ds_config.reduce_scatter,
        }
        if self.ds_config.enable_cpu_offload:
            zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}
        else:
            zero_config["offload_optimizer"] = {"device": "none"}
            zero_config["offload_param"] = {"device": "none"}

        if zero_stage >= 3:
            zero_config.setdefault(
                "stage3_prefetch_bucket_size", float(self.ds_config.stage3_prefetch_bucket_size)
            )
            zero_config.setdefault(
                "stage3_param_persistence_threshold", float(self.ds_config.stage3_param_persistence_threshold)
            )

        optimizer_config = {
            "params": {
                "lr": self.config.optim.lr,
                "betas": list(self.config.optim.betas),
                "weight_decay": self.config.optim.weight_decay,
            }
        }
        if self.config.optim.strategy == "adamw":
            optimizer_config["type"] = "AdamW"
        else:
            raise NotImplementedError(f"Optimizer {self.config.optim.strategy} not supported in DeepSpeed.")

        scheduler_config = self._build_scheduler_config()

        ds_dict = {
            "train_batch_size": int(train_batch_size),
            "train_micro_batch_size_per_gpu": int(micro_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "gradient_clipping": float(getattr(self.config, "max_grad_norm", 1.0)),
            "optimizer": optimizer_config,
            "zero_optimization": zero_config,
            "bf16": {"enabled": bf16_enabled},
            "fp16": {"enabled": fp16_enabled},
            "wall_clock_breakdown": False,
        }
        if scheduler_config is not None:
            ds_dict["scheduler"] = scheduler_config

        return ds_dict

    def _prepare_item(self, obj: Any, **kwargs: Any):
        if isinstance(obj, nn.Module):
            return self._prepare_module(obj, **kwargs)
        if isinstance(obj, Optimizer):
            return self.ds_optimizer
        if isinstance(obj, LRScheduler):
            return self.ds_scheduler
        return obj

    def _finalize_item(self, obj: Any):
        if self.ds_engine is not None and self.module is not None and obj is self.module:
            return self.ds_engine
        if isinstance(obj, Optimizer):
            return self.ds_optimizer
        if isinstance(obj, LRScheduler):
            return self.ds_scheduler
        return obj

    def _prepare_module(self, module: nn.Module, **kwargs: Any):
        self.wait_for_everyone()
        self.module = module
        return module

    def _maybe_initialize_engine(self):
        if self.ds_engine is not None:
            return
        if self.module is None:
            return

        params = [p for p in self.module.parameters() if p.requires_grad]
        if not params:
            # set all params to trainable for DeepSpeed init
            for param in self.module.parameters():
                param.requires_grad = True

        log_with_rank("Initializing DeepSpeed engine...", rank=self.rank, logger=logger)
        ds_engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
            model=self.module,
            config=self.ds_config_dict,
        )
        self.ds_engine = ds_engine
        self.ds_optimizer = ds_optimizer
        self.ds_scheduler = ds_scheduler

    def _build_scheduler_config(self) -> Optional[Dict[str, Any]]:
        optim_cfg = getattr(self.config, "optim", None)
        if optim_cfg is None:
            return None

        total_steps = max(1, getattr(optim_cfg, "training_steps", 1))
        if optim_cfg.lr_warmup_steps is not None:
            warmup_steps = optim_cfg.lr_warmup_steps
        else:
            warmup_steps = int(optim_cfg.lr_warmup_ratio * total_steps)
        warmup_steps = max(0, warmup_steps)

        base_lr = float(optim_cfg.lr)
        warmup_min_lr = 0.0 if warmup_steps > 0 else base_lr
        warmup_max_lr = base_lr

        scheduler_type = optim_cfg.lr_scheduler_type
        if scheduler_type == "constant":
            return {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": warmup_min_lr,
                    "warmup_max_lr": warmup_max_lr,
                    "warmup_num_steps": warmup_steps,
                },
            }

        if scheduler_type == "cosine":
            min_lr_ratio = optim_cfg.min_lr_ratio if optim_cfg.min_lr_ratio is not None else 0.0
            min_lr = base_lr * float(min_lr_ratio)
            return {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": warmup_min_lr,
                    "warmup_max_lr": warmup_max_lr,
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": total_steps,
                    "min_lr": min_lr,
                },
            }

        raise NotImplementedError(f"LR scheduler {scheduler_type} is not supported in DeepSpeed.")

    

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]

def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_deepspeed_module(optimizer_offload.module)