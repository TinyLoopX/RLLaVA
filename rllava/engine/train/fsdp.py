"""
Native FSDP training engine.

The class mirrors the `TrainEngine` interface and prepares FSDP-specific
configuration that will be consumed when models/optimizers are attached.
Subsequent steps will extend this skeleton with full training utilities.
"""

from __future__ import annotations

import os
import functools
import logging
import warnings
import torch
import torch.nn as nn

from contextlib import contextmanager
from typing import Any, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers.trainer_pt_utils import get_module_class_from_name
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from rllava.engine.train.base import TrainEngine
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.logger.aggregate_logger import log_with_rank
from rllava.utils.config import FSDPConfig
from rllava.model.config import CheckpointConfig
from rllava.utils.device import get_device_id, get_torch_device, get_device_name, is_cuda_available
from rllava.utils.memory_utils import aggressive_empty_cache
from rllava.utils.fs import local_mkdir_safe, copy_to_local
from rllava.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType, ShardedOptimStateDictConfig
from torch.distributed.fsdp._runtime_utils import _lazy_init
from .. import register_engine



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


@register_engine("fsdp")
class FSDPAccelerator(TrainEngine):
    """Native FSDP engine used by PPO-style training loops.

    The engine mirrors the ``TrainEngine`` interface and exposes a compact API
    that the actor/critic roles rely on.  Highlights:

    * Transparent wrapping of modules with :class:`FullyShardedDataParallel`.
    * Helper utilities to gather parameters for generation/export and to clip
      gradients across the participating FSDP modules.
    * Lightweight checkpoint orchestration that persists model, optimizer and
      scheduler states without depending on ``accelerate``.

    The implementation is intentionally minimalist so that more advanced
    sharding behaviour borrowed from external projects can be integrated later
    without breaking the current training stack.
    """

    def __init__(self, config, *, layout):
        super().__init__(config, layout=layout)
        self.fsdp_config = getattr(config, "fsdp", FSDPConfig())
        self.checkpoint_config = getattr(config, "checkpoint", CheckpointConfig())

        self.device_mesh = create_device_mesh(world_size=self.world_size, 
                                              fsdp_size=self.fsdp_config.fsdp_size)

        self.layout.groups["parameter_mesh"] = self.device_mesh

        self.wrapped_models: list[FSDP] = []
        self.optimizers: list[Optimizer] = []
        self.lr_schedulers: list[LRScheduler] = []

    def prepare(self, *args: Any, **kwargs: Any):
        """Wrap supported training artefacts with FSDP.

        Parameters are processed in-order to mirror the ``accelerate`` API.
        Supported artefacts:

        - ``nn.Module`` instances are wrapped with FSDP according to the
          configuration provided via ``config.fsdp``.
        - ``Optimizer`` / ``LRScheduler`` instances are cached so their state
          can be saved/restored alongside the model.
        """
        prepared_items = [self._prepare_item(arg, **kwargs) for arg in args]
        if len(prepared_items) == 1:
            return prepared_items[0]
        return tuple(prepared_items)

    def unwrap_model(self, model):
        if isinstance(model, FSDP):
            return model.module
        return model

    @contextmanager
    def unwrap_model_for_generation(self, model, is_peft_model: bool = False):
        aggressive_empty_cache(force_sync=True)
        
        if self.fsdp_config.offload_params:
            log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
            self.load_fsdp_model_to_gpu(model)
            log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        yield model  # we use wrapped model for FSDP
        
        if self.fsdp_config.offload_params:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            self.offload_fsdp_model_to_cpu(model)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

    @contextmanager
    def eval(self, model: FSDP):
        if self.fsdp_config.offload_params:
            self.load_fsdp_model_to_gpu(model)
        model.eval()

        yield

        if self.world_size > 1:
            model._handle.reshard(True)

        if self.fsdp_config.offload_params:
            self.offload_fsdp_model_to_cpu(model)

    @contextmanager
    def train(self, model: FSDP, optimizer: Optimizer):
        if self.fsdp_config.offload_params:
            self.load_fsdp_model_to_gpu(model)
        if self.fsdp_config.offload_optimizer:
            self.load_fsdp_optimizer(optimizer, device_id=get_device_id())
        model.train()

        yield

        if self.world_size > 1:
            model._handle.reshard(True)

        if self.fsdp_config.offload_params:
            self.offload_fsdp_model_to_cpu(model)
        if self.fsdp_config.offload_optimizer:
            self.offload_fsdp_optimizer(optimizer)

    def load_state(self, model: FSDP, optimizer: Optimizer, lr_scheduler: LRScheduler, local_path: str):
        if local_path is None:
            return

        if self.fsdp_config.offload_params:
            self.load_fsdp_model_to_gpu(model)
        if self.fsdp_config.offload_optimizer:
            self.load_fsdp_optimizer(optimizer, device_id=get_device_id())

        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if 'model' in self.checkpoint_config.load_contents
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if 'optimizer' in self.checkpoint_config.load_contents
            else None
        )
        with self.get_fsdp_state_ctx(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if 'model' in self.checkpoint_config.load_contents:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)
            if 'optimizer' in self.checkpoint_config.load_contents:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if 'extra' in self.checkpoint_config.load_contents:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                BaseCheckpointManager.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and lr_scheduler is not None:
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        self.wait_for_everyone()

        if self.fsdp_config.offload_params:
            self.offload_fsdp_model_to_cpu(model)
        if self.fsdp_config.offload_optimizer:
            self.offload_fsdp_optimizer(optimizer)

    def save_state(self, model: FSDP, optimizer: Optimizer, lr_scheduler: LRScheduler, local_path: str):
        if local_path is None:
            return

        if self.fsdp_config.offload_params:
            self.load_fsdp_model_to_gpu(model)
        if self.fsdp_config.offload_optimizer:
            self.load_fsdp_optimizer(optimizer, device_id=get_device_id())
            
        local_path = local_mkdir_safe(local_path)
        self.wait_for_everyone()

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.get_fsdp_state_ctx(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if 'model' in self.checkpoint_config.save_contents:    
                    model_state_dict = model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)
        
                if 'optimizer' in self.checkpoint_config.save_contents:
                    optimizer_state_dict = optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if 'extra' in self.checkpoint_config.save_contents:
                    lr_scheduler_state_dict = lr_scheduler.state_dict() if lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": BaseCheckpointManager.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        self.wait_for_everyone()

        if self.fsdp_config.offload_params:
            self.offload_fsdp_model_to_cpu(model)
        if self.fsdp_config.offload_optimizer:
            self.offload_fsdp_optimizer(optimizer)

    def backward(self, loss: torch.Tensor, is_last_step: bool = True):
        # is_last_step is ignored for FSDP, only used by DeepSpeed
        loss.backward()

    def get_fsdp_state_ctx(self, model, state_type, state_cfg, optim_cfg):
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)

    def optimizer_step(self, model: FSDP, optimizer: Optimizer):
        """Perform optimizer step with two-level gradient management.

        1. clip_grad_norm_ clips gradients to ``max_grad_norm`` (default 1.0).
        2. If the *pre-clip* grad norm exceeds ``skip_grad_norm`` (default 80.0),
           the update is skipped entirely as a safety valve against gradient
           explosions (mirrors the original Unify-Post-Training implementation).
        """
        grad_norm = self.clip_grad_norm_(
                        model, self.config.max_grad_norm
                    )

        skip_threshold = getattr(self.config, "skip_grad_norm", 80.0)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
            optimizer.zero_grad()
        elif skip_threshold > 0 and grad_norm > skip_threshold:
            print(f"Gradient norm is too large: {grad_norm} > {skip_threshold}. Skip update.")
            optimizer.zero_grad()
        else:
            optimizer.step()

        return grad_norm

    def clip_grad_norm_(self, model: FSDP, max_norm: float):
        grad_norm = model.clip_grad_norm_(max_norm)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        
        return grad_norm

    def _prepare_item(self, obj: Any, **kwargs: Any):
        if isinstance(obj, nn.Module):
            return self._prepare_module(obj, **kwargs)
        if isinstance(obj, Optimizer):
            self._register_optimizer(obj, **kwargs)
            return obj
        if isinstance(obj, LRScheduler):
            self._register_lr_scheduler(obj, **kwargs)
            return obj
        return obj

    def _prepare_module(self, module: nn.Module, **kwargs: Any):
        from torch.distributed.fsdp import MixedPrecision, CPUOffload
        from rllava.utils.torch_dtypes import PrecisionType

        self.wait_for_everyone()

        forward_only = kwargs.get("forward_only", False)

        mixed_precision_config = self.fsdp_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = self.get_fsdp_wrap_policy(
            module=module,
            config=self.fsdp_config.wrap_policy,
            is_lora=self.config.model.use_peft
        )

        cpu_offload = None
        if forward_only:
            cpu_offload = CPUOffload(offload_params=True)

        sharding_strategy = get_sharding_strategy(self.device_mesh)
        wrapped_module = FSDP(
                module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.fsdp_config.use_orig_params,
                forward_prefetch=self.fsdp_config.forward_prefetch
            )

        if wrapped_module not in self.wrapped_models:
            self.wrapped_models.append(wrapped_module)

        if self.config.model.enable_activation_offload:
            enable_gradient_checkpointing = self.config.model.enable_gradient_checkpointing
            enable_activation_offloading(wrapped_module, 'fsdp', enable_gradient_checkpointing)

        if self.world_size == 1:
            FSDP.set_state_dict_type(
                wrapped_module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        else:
            FSDP.set_state_dict_type(
                wrapped_module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        if not forward_only:
            if self.fsdp_config.offload_params:
                self.offload_fsdp_model_to_cpu(wrapped_module)
                log_gpu_memory_usage("After offload model during init", logger=logger)

        return wrapped_module

    def _register_optimizer(self, optimizer: Optimizer, **kwargs: Any) -> None:
        if optimizer not in self.optimizers:
            self.optimizers.append(optimizer)
            if self.fsdp_config.offload_optimizer:
                self.offload_fsdp_optimizer(optimizer=optimizer)
                log_gpu_memory_usage("After offload optimizer during init", logger=logger)

    def _register_lr_scheduler(self, scheduler: LRScheduler, **kwargs: Any) -> None:
        if scheduler not in self.lr_schedulers:
            self.lr_schedulers.append(scheduler)

    @torch.no_grad()
    def load_fsdp_model_to_gpu(self, model: FSDP):
        # lazy init FSDP model
        _lazy_init(model, model)
        assert model._is_root, "Only support root model loading to GPU"
        device_id = get_device_id()
        for handle in model._all_handles:
            if handle._offload_params:
                continue
            flat_param = handle.flat_param
            handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)
            # the following still keeps id(._local_shard) != id(.data)
            flat_param._local_shard = flat_param.data
    
    @torch.no_grad()
    def offload_fsdp_model_to_cpu(self, model: FSDP, empty_cache: bool = True):
        # lazy init FSDP model
        _lazy_init(model, model)
        assert model._is_root, "Only support root model offloading to CPU"
        for handle in model._all_handles:
            if handle._offload_params:
                continue
            flat_param = handle.flat_param
            assert (
                flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
                and id(flat_param.data) != id(flat_param._local_shard)
                and flat_param.data.size() == flat_param._local_shard.size()
            )
            handle.flat_param_to(torch.device("cpu"), non_blocking=True)
            # the following still keeps id(._local_shard) != id(.data)
            flat_param._local_shard = flat_param.data
            assert id(flat_param._local_shard) != id(flat_param.data)
        if empty_cache:
            get_torch_device().empty_cache()

    @torch.no_grad()
    def offload_fsdp_optimizer(self, optimizer):
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)

    @torch.no_grad()
    def load_fsdp_optimizer(self, optimizer, device_id):
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)

    # Copyright 2020-present the HuggingFace Inc. team.
    # Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
    def get_fsdp_wrap_policy(self, module, config=None, is_lora=False):
        """Get FSDP wrap policy for the module.
    
        Args:
            module: The module to get wrap policy for
            config: Configuration for wrap policy
            is_lora: Whether to enable lambda policy for LoRA modules
        """
        if config is None:
            config = {}
    
        # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this
        # once we have make all config in verl from OmegaConf to data class.
        def _get_attr(attr_name, default_value=None):
            if hasattr(config, "get"):
                return config.get(attr_name, default_value)
            else:
                return config.__getattribute__(attr_name)
    
        if _get_attr("disable", False):
            return None
    
        default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = _get_attr(
            "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
        )
        min_num_params = _get_attr("min_num_params", 0)
        auto_wrap_policy = None
    
        policies = []
    
        from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy
    
        # Add lambda policy for LoRA modules if is_lora is True
        if is_lora:
    
            def lambda_policy_fn(module):
                return bool(
                    len(list(module.named_children())) == 0
                    and getattr(module, "weight", None) is not None
                    and module.weight.requires_grad
                )
    
            lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
            policies.append(lambda_policy)
    
        if min_num_params > 0:
            size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
            policies.append(size_policy)
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(module, layer_class)
                if transformer_cls is None:
                    raise Exception("Could not find the transformer layer class to wrap in the model.")
                else:
                    transformer_cls_to_wrap.add(transformer_cls)
    
            transformer_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_cls_to_wrap,
            )
            policies.append(transformer_policy)
    
        if len(policies) > 0:
            auto_wrap_policy = functools.partial(_or_policy, policies=policies)
    
        return auto_wrap_policy


class FSDPParameterFilter:
    def __init__(self):
        self.model_parameters_storage = set()

    def __call__(self, tensor):
        return tensor.untyped_storage().data_ptr() not in self.model_parameters_storage

    def update_model_parameters(self, model):
        new_storage = set()
        for p in model.parameters():
            new_storage.add(p.data.untyped_storage().data_ptr())
        self.model_parameters_storage = new_storage


class ActivationHandler:
    def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt):
        self._offload_ctx = offload_ctx
        self._sync_func = sync_func
        self._enable_ckpt = enable_ckpt
        self._tensor_filter = tensor_filter
        if enable_ckpt:
            self.checkpoint_fn = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=True,
            )

    def pre_forward(self, module):
        if module.training:
            self._offload_ctx.__enter__()
            self._tensor_filter.update_model_parameters(module)

    def post_forward(self, module):
        if module.training:
            self._offload_ctx.__exit__(None, None, None)

    def _pack_kwargs(self, *args, **kwargs):
        kwarg_keys = []
        flat_args = list(args)
        for k, v in kwargs.items():
            kwarg_keys.append(k)
            flat_args.append(v)

        return tuple(flat_args), tuple(kwarg_keys)

    def _unpack_kwargs(self, flat_args, kwarg_keys):
        assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
        if len(kwarg_keys) == 0:
            return flat_args, {}
        args = flat_args[: -len(kwarg_keys)]
        kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :], strict=True))
        return args, kwargs

    def _ckpt_forward(self, forward_method, *args, **kwargs):
        flat_args, kwarg_keys = self._pack_kwargs(*args, **kwargs)

        def my_function(*inputs):
            # unpack back into args and kwargs
            nonlocal forward_method, kwarg_keys
            unpacked_args, unpacked_kwargs = self._unpack_kwargs(inputs, kwarg_keys)
            # run original module
            return forward_method(*unpacked_args, **unpacked_kwargs)

        return self.checkpoint_fn(
            my_function,
            *flat_args,
        )

    def forward(self, module, forward_method, *args, **kwargs):
        if not module.training:
            return forward_method(*args, **kwargs)
        if not self._enable_ckpt:
            ret = forward_method(*args, **kwargs)
        else:
            ret = self._ckpt_forward(forward_method, *args, **kwargs)
        binded_tensor = ret
        if isinstance(ret, tuple):
            binded_tensor = ret[0]
        binded_tensor = self._sync_func(binded_tensor)
        final_ret = binded_tensor
        if isinstance(ret, tuple):
            final_ret = (final_ret,) + ret[1:]
        return final_ret

    def wrap_module_forward_method(self, module):
        orig_method = module.forward
        handler = self

        @functools.wraps(orig_method)
        def wrapped_method(model_self, *args, **kwargs):
            nonlocal handler
            handler.pre_forward(model_self)
            out = handler.forward(model_self, orig_method, *args, **kwargs)
            handler.post_forward(model_self)
            return out

        module.forward = wrapped_method.__get__(module, type(module))


class OffloadHandler:
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. Inherit this class and implement your "
            "custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        """Tensor pop."""
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. Inherit this class and implement your "
            "custom tensor_pop."
        )


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


class CpuOffloadHookWithOffloadHandler:
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: dict[str, Any] = handler_extra_kwargs
        self.inside_context = False

    def __enter__(self):
        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(self.on_save_for_backward, self.on_get_saved_tensor)

    def __exit__(self, *args: Any):
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        return tensor


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(self, num_offload_group, tensor_need_offloading_checker=(lambda _: True)) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker

        self.groupid_reset()

    def groupid_reset(self):
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        self.current_group += 1  # increment
        self.tensor_count_current_group = 0  # reset

    def on_group_commit_backward(self):
        """On group commit backward."""
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_backup.copy_(src_tensor, non_blocking=True)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after group commit
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    def __init__(
        self,
        num_offload_group,  # must be <= actual number of groups (number of commits)
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
    ) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
        )
        # Number of layers in the model
        self.num_layers = num_model_group
        # Data Structure to maintain reference to activation tensors
        self.tensor_tag_to_buf = {}
        # Tracking the number of layers offloaded
        self.offloaded_group_count = 0
        # Core data structure that decides the window for offloading
        self.layer_window_map = {}
        self.group_offload_mapping = {}

        # Logic to make offloading load balance across computation
        # for optimal CPU/GPU interconnect usage
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = ((self.num_layers // self.num_offload_group) * (i + 1)) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        # allocate streams and events for synchronization
        self.d2h_stream = get_torch_device().Stream()
        self.h2d_stream = get_torch_device().Stream()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        torch_stray_tensor = isinstance(
            tensor,
            torch._subclasses.fake_tensor.FakeTensor | torch._subclasses.functional_tensor.FunctionalTensor,
        )
        need_offload = not torch_stray_tensor
        need_offload = need_offload and self.tensor_need_offloading_checker(tensor)

        if need_offload:
            # obtain a unique tensor tag
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1

            assert tensor_tag not in self.tensor_tag_to_state
            self.tensor_tag_to_state[tensor_tag] = tensor

            if self.current_group < self.num_offload_group:
                self.tensor_tag_to_buf[tensor_tag] = tensor
        else:
            tensor_tag = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        if isinstance(tensor_tag, torch.Tensor):
            return tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)

        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        """Bulk offload group."""
        offload_mapping = {}
        offload_size = 0
        with get_torch_device().stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    key = _get_unique_tensor_key(state)
                    if key not in offload_mapping:
                        offload_mapping[key] = state
                    # if offload, return the reference to cpu copy
                    self.tensor_tag_to_state[tensor_tag] = (key, state.shape)
            for key, tensor in offload_mapping.items():
                state = SynchronizedGroupOffloadHandler.offload(tensor)
                offload_size += tensor.numel() * tensor.element_size()
                offload_mapping[key] = state

            self.group_offload_mapping[group_to_offload] = offload_mapping

    def synchronize_on_group_commit_forward(self, current_group):
        """Synchronize on group commit forward."""

        # For the first group, kickstart the offload after we have
        # the first compute completion
        if current_group == 0:
            self.d2h_stream.wait_stream(get_torch_device().current_stream())
            self.bulk_offload_group(current_group)

        # Window map data structure helps us synchronize based on number
        # of layers offloaded
        if self.layer_window_map[self.offloaded_group_count] == current_group:
            # Stream synchronization both ways
            self.d2h_stream.wait_stream(get_torch_device().current_stream())
            get_torch_device().current_stream().wait_stream(self.d2h_stream)

            # Time to free the activation memory after usage
            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_group_count:
                    self.tensor_tag_to_buf[tensor_tag] = None

            # Time to offload the next group
            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)

            # Increment the offload group count to keep track
            self.offloaded_group_count += 1

    def on_group_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)

        super().on_group_commit_forward()

    @torch.no_grad
    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        assert group_to_reload < self.num_offload_group

        with get_torch_device().stream(self.h2d_stream):
            # move back tensors
            offload_mapping = self.group_offload_mapping.pop(group_to_reload)
            assert offload_mapping is not None
            for key, state in offload_mapping.items():
                offload_mapping[key] = SynchronizedGroupOffloadHandler.reload(state)
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload and not isinstance(state, torch.Tensor):
                    assert isinstance(state, tuple), f"{group_id} {state}"
                    key, shape = state
                    recovered_tensor = offload_mapping[key].view(shape)
                    self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        self.current_group -= 1
        assert self.current_group >= 0

        # Layer window data structure helps us to reload at right times
        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:
            # Stream synchronization both ways
            self.h2d_stream.wait_stream(get_torch_device().current_stream())
            get_torch_device().current_stream().wait_stream(self.h2d_stream)

            # Time to reload the next group
            self.bulk_reload_group(self.offloaded_group_count - 1)

            # Decrease the offloading group counter
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        # Last group computation needs to wait till all the reloads complete
        if self.current_group == 0:
            get_torch_device().current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0


group_prefetch_offload_commit = GroupCommitFunction.apply


def _get_unique_tensor_key(tensor):
    key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    return key

def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        torch.cuda.empty_cache()
    return x

def get_activation_offload_context(
    num_layers: int = 1, model_layers: int = 1, tensor_need_offloading_checker=(lambda t: True)
):
    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    return (
        CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
        group_prefetch_offload_commit_async,
    )

def enable_activation_offloading(model, enable_ckpt=False):
    """
    Enable activation offloading for the model. It groups activations by TransformerLayer and offloads activation
    groups asynchronously. This means that the offloading of the i-th activation group and the computation of the i+1-th
    activation group happen at the same time, and there are at most two activation groups in GPU memory.

    Args:
        model: the model to enable activation offloading
        strategy: the training strategy of the model, such as "fsdp"
        enable_ckpt: whether activation checkpointing(also called gradient checkpointing) has been enabled for the model

    Note:
        For best efficiency, activation offloading is usually combined with activation checkpointing. However, this
        implementation of activation offloading is conflicted with the implementation of activation checkpointing in
        some training strategies. This function resolves this conflict, and therefore requires the "strategy" and
        "enable_ckpt" arguments.

    Returns:

    """

    layers = []

    def get_layers(module):
        for name, child in module.named_children():
            if not isinstance(child, FSDP):
                get_layers(child)
            else:
                wrapped_module = child
                if isinstance(child, FSDP):
                    wrapped_module = child._fsdp_wrapped_module
                # In some cases, torch.nn.Embedding is wrapped with FSDP alone. However, the activation
                # size of torch.nn.Embedding is small, so it's not necessary to offload it.
                if not isinstance(wrapped_module, torch.nn.Embedding):
                    layers.append(child)

    get_layers(model)
    if len(layers) < 3:
        logger.warning(f"Find only {len(layers)} fsdp layers, not neccessary to enable async activation offloading")
        return

    tensor_filter = FSDPParameterFilter()
    context, sync_func = get_activation_offload_context(len(layers) - 1, len(layers), tensor_filter)
    if enable_ckpt:
        # The implementation of activation checkpointing in transformers library is incompatible with
        # activation offloading,
        # so it will be disabled, but this implementation supports another version of activation checkpointing, so that
        # these two features can be enabled at the same time.
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt)
    for layer in layers:
        module = layer
        if isinstance(layer, FSDP):
            module = module._fsdp_wrapped_module
        handler.wrap_module_forward_method(module)

def create_device_mesh(world_size, fsdp_size):
    device_name = get_device_name()

    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh

def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy

def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
    
    with FSDP.state_dict_type(
        model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
    ):
        state_dict = model.state_dict()
    return state_dict