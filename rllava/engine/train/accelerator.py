"""
Accelerator-based engine for PPO with FSDP and DeepSpeed support
"""
import torch
from contextlib import contextmanager
from typing import Dict
from accelerate import Accelerator
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin, DeepSpeedPlugin
from contextlib import nullcontext
from trl.models.utils import add_hooks, remove_hooks, is_deepspeed_available
from .base import TrainEngine
from .. import register_engine
try:
    # Prefer the top-level import if available (PyTorch >= 2.0)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:
    try:
        # Fallback older path
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    except Exception:
        FSDP = None


if is_deepspeed_available():
    import deepspeed



@register_engine("accelerator")
class HFAccelerator(TrainEngine):
    """
    HFAccelerator class that uses HuggingFace Accelerator for distributed training.
    Supports FSDP and DeepSpeed automatically through Accelerator configuration.
    """
    
    def __init__(self, config, *, layout):
        super().__init__(config, layout=layout)

        self.device_mesh = None
        if config.strategy == "fsdp":
            # Build FSDP configuration using only existing FSDPConfig fields
            fsdp_config = {
                "cpu_offload": config.fsdp.enable_cpu_offload,
                "use_orig_params": config.fsdp.use_orig_params,
            }
            
            # Build mixed precision policy from existing FSDPConfig fields
            mixed_precision_policy = None
            if config.fsdp.mp_param_dtype or config.fsdp.mp_reduce_dtype or config.fsdp.mp_buffer_dtype:
                # Helper function to convert string dtype to torch dtype
                def get_torch_dtype(dtype_str):
                    if not dtype_str:
                        return None
                    dtype_map = {
                        "bf16": torch.bfloat16,
                        "fp16": torch.float16,  
                        "fp32": torch.float32,
                        "float32": torch.float32,
                        "bfloat16": torch.bfloat16,
                        "half": torch.float16,
                    }
                    return dtype_map.get(dtype_str, torch.float32)
                
                mixed_precision_policy = {}
                if config.fsdp.mp_param_dtype:
                    mixed_precision_policy["param_dtype"] = get_torch_dtype(config.fsdp.mp_param_dtype)
                if config.fsdp.mp_reduce_dtype:
                    mixed_precision_policy["reduce_dtype"] = get_torch_dtype(config.fsdp.mp_reduce_dtype)
                if config.fsdp.mp_buffer_dtype:
                    mixed_precision_policy["buffer_dtype"] = get_torch_dtype(config.fsdp.mp_buffer_dtype)
                    
                if mixed_precision_policy:
                    fsdp_config["mixed_precision_policy"] = mixed_precision_policy
            
            # Create FSDP plugin
            fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_config)
            
            # Initialize accelerator with FSDP plugin
            self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        elif config.strategy == "deepspeed":
            # Convert custom DeepSpeedConfig to DeepSpeedPlugin with complete ZeRO config
            deepspeed_config = {
                "zero_optimization": {
                    "stage": config.deepspeed.zero_stage,
                    "offload_optimizer": {
                        "device": "cpu" if config.deepspeed.enable_cpu_offload else "none",
                    },
                    "offload_param": {
                        "device": "cpu" if config.deepspeed.enable_cpu_offload else "none", 
                    },
                },
                "bf16": {
                    "enabled": config.deepspeed.torch_dtype == "bfloat16"
                },
                "fp16": {
                    "enabled": config.deepspeed.torch_dtype == "float16"
                },
                "gradient_clipping": 1.0,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": config.ppo_micro_batch_size,
            }
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=deepspeed_config,
                zero3_init_flag=True,
            )
            self.accelerator = Accelerator(
                deepspeed_plugin=deepspeed_plugin
            )
        else:
            self.accelerator = Accelerator()

    @property
    def rank(self):
        """Get rank from Accelerator."""
        return self.accelerator.process_index
    
    @property
    def num_processes(self):
        """Get number of processes from Accelerator."""
        return self.accelerator.num_processes
    
    @property
    def is_main_process(self):
        """Check if current process is the main process."""
        return self.accelerator.is_main_process
    
    def wait_for_everyone(self):
        """Wait for all processes to complete."""
        self.accelerator.wait_for_everyone()

    def prepare(self, *args, **kwargs):
        return self.accelerator.prepare(*args, **kwargs)

    def get_init_weight_context(self, use_meta_tensor=True):
        # For FSDP and DeepSpeed, avoid meta initialization to prevent compatibility issues
        if getattr(self.config, "strategy", None) in ["fsdp", "deepspeed"]:
            return nullcontext
        
        from accelerate import init_empty_weights
    
        cpu_init_weights = lambda: torch.device("cpu")
        if use_meta_tensor:
            init_context = init_empty_weights if (torch.distributed.is_initialized() and self.rank != 0) else cpu_init_weights
        else:
            init_context = cpu_init_weights
        return init_context

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model)

    @contextmanager
    def unwrap_model_for_generation(self, model, is_peft_model: bool = False):
        # Always unwrap the model for a clean interface downstream
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        if is_peft_model and hasattr(unwrapped_model, "pretrained_model"):
            unwrapped_model.pretrained_model.disable_adapter()
        
        if self.accelerator.state.deepspeed_plugin is not None and self.accelerator.state.deepspeed_plugin.zero_stage == 3:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield self.accelerator.unwrap_model(model)
                add_hooks(model)
        elif self.accelerator.state.fsdp_plugin is not None:
            yield model

    @contextmanager
    def eval(self, model):
        model.to(torch.cuda.current_device())
        model.eval()

        yield

        model.to('cpu')

    @contextmanager
    def train(self, model, optimizer):
        model.to(torch.cuda.current_device())
        model.train()

        yield

        model.to('cpu')

    def load_state(self, model, optimizer, lr_scheduler, checkpoint_path):
        self.accelerator.load_state(checkpoint_path)

    def save_state(self, model, optimizer, lr_scheduler, checkpoint_path):
        self.accelerator.save_state(checkpoint_path)

    def backward(self, loss, is_last_step: bool = True):
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss)

    def clip_grad_norm_(self, model, max_norm):
        return self.accelerator.clip_grad_norm_(model.parameters(), max_norm)
    
    def _rename_weight_keys(self, weights: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
        """Convert state dict keys for compatibility.
        
        Args:
            actor_weights: Model weights dictionary
            model: The model to get conversion mapping from
            
        Returns:
            Converted weights dictionary
        """
        # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return weights

        import re
        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights