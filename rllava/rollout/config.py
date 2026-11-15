from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union
from rllava.distribute.resource_manager import ParallelConfig
from rllava.engine.inference.config import VLLMConfig, SGLangConfig


@dataclass
class RolloutConfig:
    agent: str = "single_turn"  # single_turn, multi_turn, gui
    name: str = "vllm"  # options: vllm, sglang
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    seed: Optional[Union[int, str]] = 1  # int, or "none"/"null" for random seed
    limit_images: int = 0
    ignore_eos: bool = False
    disable_tqdm: bool = False
    micro_batch_size: int = 2
    val_override_config: Dict[str, Any] = field(default_factory=dict)
    partial_rollout: bool = False # only available for SGLang
    async_mode: bool = False
    bucket_size: int = 512
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    load_format: str = "dummy"
    # multiturn
    max_turns: int = 10
    discount: float = 0.9
    tool_config_path: Optional[str] = None
    env_config_path: Optional[str] = None
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=True, init=False)
    adv_estimator: str = field(default="grpo", init=False)
    min_pixels: Optional[int] = field(default=None, init=False)
    max_pixels: Optional[int] = field(default=None, init=False)
    video_fps: float = field(default=2.0, init=False)

    def to_dict(self):
        return asdict(self)
