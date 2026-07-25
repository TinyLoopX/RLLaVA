from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from rllava.utils.config import conf_as_dict
from rllava.utils import pkg_version



@dataclass
class SGLangConfig:
    dtype: str = "bfloat16"
    mem_fraction_static: Optional[float] = 0.6
    tp_size: int = 1  
    skip_server_warmup: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class VLLMConfig:
    """Configuration for vLLM runtime. Refer to:
    https://github.com/vllm-project/vllm for detailed documentation.
    """

    model: str = ""
    seed: Optional[int] = 1
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    dtype: str = "float16"
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    enforce_eager: bool = True
    enable_chunked_prefill: bool = False
    distributed_executor_backend: Optional[str] = None
    tokenizer: Optional[str] = None
    chat_template: Optional[str] = None

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        vllm_config: "VLLMConfig",
        tp_size,
        host,
        port,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args = VLLMConfig.build_args(
            vllm_config=vllm_config,
            tp_size=tp_size,
            host=host,
            port=port,
            n_nodes=n_nodes,
            node_rank=node_rank,
        )

        # convert to flags
        flags = []
        for k, v in args.items():
            if v is None or v is False or v == "":
                continue
            if v is True:
                flags.append(f"--{k.replace('_','-')}")
            elif isinstance(v, list):
                flags.append(f"--{k.replace('_','-')} {' '.join(map(str, v))}")
            else:
                flags.append(f"--{k.replace('_','-')} {v}")
        return f"python3 -m vllm.entrypoints.openai.api_server {' '.join(flags)}"

    @staticmethod
    def build_args(
        vllm_config: "VLLMConfig",
        tp_size,
        host,
        port,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):

        # ensure vllm is installed
        if not pkg_version.is_available("vllm"):
            raise RuntimeError("vllm is not installed in the current environment.")

        cfg: Dict = conf_as_dict(vllm_config)

        args = dict(
            host=host,
            port=port,
            tensor_parallel_size=tp_size,
            # multi-node placeholders (not all backends require explicit flags)
            # Expose distributed backend if provided via config
            **cfg,
        )
        return args