from dataclasses import dataclass, field


@dataclass
class PolicyLossConfig:
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class KLControlConfig:
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator: `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`, etc"""
    norm_adv_by_std_in_grpo: bool = True
    """whether to normalize advantage by standard deviation in grpo"""
    use_kl_loss: bool = True
    """whether to use kl loss"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    """kl controller for reward calculation"""
    use_kl_in_reward: bool = False
    """whether to use kl penalty in reward calculation"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    # # SFT&RFT configs
    unify_strategy: str = "none"  # "none" | "switch" | "soft"
    switch_gate: int = 0
    success_reward_threshold: float = 1.0
    # --- target prefix ---
    prefix_strategy: str = "none"          # "none" | "random"
    min_prefix_ratio: float = 0.0
    max_prefix_ratio: float = 1.0
    n_prefix: int = -1
    prefix_share_across_samples: bool = False
    # UFT hint schedule (used when prefix_strategy == "hint")
    # NOTE: hint ratios are tied to min_prefix_ratio/max_prefix_ratio.
    hint_steps: int = 0
    hint_min_ratio: float = 0.0  # deprecated, kept for compatibility
    hint_max_ratio: float = 1.0  # deprecated, kept for compatibility
