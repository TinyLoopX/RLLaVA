import torch
import logging
import os
from trl import get_peft_config
from peft import get_peft_model
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizer
from rllava.utils.flops_counter import FlopsCounter
from rllava.utils.torch_dtypes import PrecisionType
from rllava.model.config import ModelConfig, OptimConfig
from rllava.utils.logger.aggregate_logger import print_rank_0
from rllava.model.patch.monkey_patch import apply_monkey_patch
from rllava.utils.train_utils import find_all_linear_names
from rllava.distribute.dist_utils import is_rank0
from rllava.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from rllava.utils.model_utils import print_model_size



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


def build_model(config: ModelConfig, tokenizer: PreTrainedTokenizer, device_mesh: DeviceMesh=None, model_class=None, torch_dtype=None, trainerble=True):
    model_config = AutoConfig.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation=config.attn_implementation,
        **config.override_config,
    )
    print_rank_0(f"Model config: {model_config}")
    
    flops_counter = FlopsCounter(model_config)

    if model_class is None:
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM
    
    if torch_dtype is None:
        torch_dtype = config.torch_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
    
    init_weight = get_init_weight_context(
        use_meta_tensor=not model_config.tie_word_embeddings, mesh=device_mesh)
    with init_weight():
        model = model_class.from_pretrained(
            config.model_path,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attn_implementation,
        )

        apply_monkey_patch(
            model=model,
            use_remove_padding=config.padding_free,
            ulysses_sp_size=config.ulysses_size
        )

        model.to(torch_dtype)

    if not trainerble:
        for param in model.parameters():
            param.requires_grad = False

    if config.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    config.lora_target_modules = find_all_linear_names(model, ['visual', 'connector', 'vision_tower'] )
    peft_config = get_peft_config(config)
    
    if peft_config is not None:
        model.enable_input_require_grads()
        # If PEFT is used, wrap the model with PEFT
        peft_model = get_peft_model(model, peft_config)
        model = peft_model

    if is_rank0(): 
        print_model_size(model)

    return model, flops_counter


def build_optimizer(config: OptimConfig, model: nn.Module):
    if config.strategy == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            fused=True,
        )
    elif config.strategy == "adamw_bf16":
        from utils.torch_functional import AnyPrecisionAdamW
        optimizer = AnyPrecisionAdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.strategy} not supported.")
        
    # Create learning rate scheduler   
    if config.lr_warmup_steps is not None:
        num_warmup_steps = config.lr_warmup_steps
    else:
        num_warmup_steps = int(config.lr_warmup_ratio * config.training_steps)
    
    if config.lr_scheduler_type == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps
        )
    elif config.lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=config.training_steps,
            min_lr_ratio=config.min_lr_ratio,
            num_cycles=config.num_cycles,
        )
    return optimizer, lr_scheduler


def get_init_weight_context(use_meta_tensor: bool = True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights
    
    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if not is_rank0() else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context