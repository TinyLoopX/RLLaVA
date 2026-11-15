import os
from dataclasses import dataclass, field
from typing import Optional
from rllava.utils.config import FSDPConfig, DeepSpeedConfig
from rllava.model.config import ModelConfig, OptimConfig, CheckpointConfig
from rllava.distribute.resource_manager import ParallelConfig
from rllava.train.config import PolicyLossConfig


@dataclass
class RewardConfig:
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)
    model: ModelConfig = field(default_factory=ModelConfig)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                print(f"Reward function {self.reward_function} not found.")
                self.reward_function = None


@dataclass
class CriticConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    strategy: str = "fsdp2"
    ppo_mini_batch_size: int = 256
    """number of samples per minibatch for updating critic"""
    ppo_micro_batch_size: int = 4
    """number of samples per forward pass for updating critic"""
    log_prob_micro_batch_size: int = 16
    """number of samples per forward pass for computing values"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    cliprange_value: float = 0.5
    """clip range for value loss"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)


@dataclass
class ActorConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    strategy: str = "fsdp2"
    ppo_mini_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    ppo_micro_batch_size: int = 4
    """number of samples per forward pass for updating actor"""
    log_prob_micro_batch_size: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """clip grad norm to this value (equivalent to original's grad_clip)"""
    skip_grad_norm: float = 80.0
    """skip optimizer step entirely when pre-clip grad norm exceeds this threshold"""
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_c: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    clip_upper_bound: float = 1.0
    loss_agg_mode: str = "token-mean"
    """loss aggregate mode: `token-mean`, `seq-mean-token-sum`"""
    entropy_coeff: float = 0
    tis_imp_ratio_cap: float = -1
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    use_torch_compile: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    kl_loss_coef: float = 0.01
    kl_loss_type: str = "low_var_kl"
    # SFT&RFT configs
    loss_remove_clip: bool = False
    off_policy_reshape: str = "no_reshape"
    off_policy_reshape_weight: float = 0.1
    off_policy_reshape_pow_exp: float = 0.5
    off_policy_cliprange: float = 0.2
    off_policy_normalize: bool = False
    off_policy_max_clip: float = -1
    off_policy_min_clip: float = -1
    all_max_clip: float = -1
    on_policy_reshape: str = "no_reshape"
    on_policy_reshape_weight: float = 0.1
    on_policy_reshape_pow_exp: float = 0.5
    srft_type: str = "paper"
    hpt_hint_loss_coef: float = 0.0
    sft_loss_coef: float = 1.0


@dataclass
class RefConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
