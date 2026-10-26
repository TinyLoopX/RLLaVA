import os
import warnings
from rllava.data.config import DataConfig
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from typing import Type, TypeVar, cast
from dataclasses import is_dataclass, fields, asdict



T = TypeVar("T")

def init_config(cls: Type[T]) -> T:
    from rllava.distribute.resource_manager import normalize_parallel_config

    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(cls())
    if "config" in cli_args:
        config_path = cli_args["config"]
        del cli_args["config"]
        file_config = normalize_parallel_config(OmegaConf.load(config_path))
        default_config = OmegaConf.merge(default_config, file_config)
    config = OmegaConf.merge(default_config, normalize_parallel_config(cli_args))
    obj = OmegaConf.to_object(config)
    if not isinstance(obj, cls):
        obj = cls(**obj)  # type: ignore[arg-type]
    if hasattr(obj, "deep_post_init"):
        obj.deep_post_init()
    return cast(T, obj)

def conf_as_dict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)

def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "rllava"
    """project name for logger"""
    experiment_name: str = "agent"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "tensorboard")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    outputs_dir: Optional[str] = None
    """outputs directory"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    async_training: bool = False
    """enable async-like rollout using workflow interface"""

    def post_init(self):
        if self.outputs_dir is None:
            self.outputs_dir = os.path.join(self.project_name, self.experiment_name)
        else:
            self.outputs_dir = os.path.join(self.outputs_dir, self.project_name, self.experiment_name)
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join(self.outputs_dir, "checkpoints")

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class BaseConfig:
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)


@dataclass
class FSDPConfig:
    wrap_policy: dict[str, Any] = field(default_factory=dict)
    enable_cpu_offload: bool = False
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    offload_params: bool = False
    offload_optimizer: bool = False
    reshard_after_forward: bool = True
    forward_prefetch: bool = False
    fsdp_size: int = -1
    ulysses_size: int = 1
    mixed_precision: Optional[dict[str, Any]] = None


@dataclass
class DeepSpeedConfig:
    zero_stage: int = 3
    enable_cpu_offload: bool = True
    torch_dtype: str = "bf16"
    contiguous_gradients: bool = True
    overlap_comm: bool = True
    reduce_scatter: bool = True
    stage3_prefetch_bucket_size: float = 1e8
    stage3_param_persistence_threshold: float = 1e5
