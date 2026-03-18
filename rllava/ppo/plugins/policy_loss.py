import torch
from omegaconf import DictConfig
from rllava.ppo.config import AlgorithmConfig, ActorConfig
from typing import Callable, Optional
from rllava.ppo.utils.core_algos import agg_loss
from rllava.utils import torch_functional as VF



PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgorithmConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(
    name: str,
    extra_keys: Optional[list[str] | set[str]] = None,
) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        func.extra_keys = set(extra_keys or [])
        return func

    return decorator


def get_policy_loss_extra_keys(policy_loss_fn: PolicyLossFn) -> set[str]:
    return getattr(policy_loss_fn, "extra_keys", set())


def build_policy_loss_kwargs(
    policy_loss_fn: PolicyLossFn,
    model_inputs: dict,
    metrics: dict
) -> dict:
    extra_keys = getattr(policy_loss_fn, "extra_keys", set())
    if not extra_keys:
        return {}

    kwargs: dict = {}
    for key in extra_keys:
        if key in model_inputs:
            kwargs[key] = model_inputs.get(key)
    kwargs['metrics'] = metrics
    return kwargs


def get_policy_loss(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(rllava.ppo.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgorithmConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.clip_ratio_c  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    if config.loss_remove_clip:
        pg_losses2 = pg_losses1
        clip_pg_losses1 = pg_losses1
        pg_clipfrac = torch.tensor(0.0, device=pg_losses1.device)
    else:
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = VF.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = VF.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    if config.loss_remove_clip:
        pg_losses = pg_losses1
        pg_clipfrac_lower = torch.tensor(0.0, device=pg_losses1.device)
    else:
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        pg_losses = pg_losses * tis_imp_ratio

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("sft")
def compute_policy_loss_sft(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Supervised token-level NLL over valid response tokens.
    del old_log_prob, advantages, config, rollout_log_probs
    sft_losses = -log_prob
    sft_loss = agg_loss(loss_mat=sft_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    zero = torch.tensor(0.0, device=log_prob.device)
    return sft_loss, zero, zero, zero


@register_policy_loss(
    "luffy",
    extra_keys={"prefix_mask"}
)
def compute_policy_loss_luffy(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    metrics: dict = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert config is not None
    assert not isinstance(config, AlgorithmConfig)

    if prefix_mask is None or prefix_mask.numel() == 0:
        return compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_log_probs=rollout_log_probs,
        )

    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)

    # on-policy ratio
    on_ratio = torch.exp(negative_approx_kl)
    on_ratio = _reshape_ratio(
        on_ratio,
        log_prob,
        old_log_prob,
        config.on_policy_reshape,
        config.on_policy_reshape_weight,
        config.on_policy_reshape_pow_exp,
    )
    on_pg_losses = -advantages * on_ratio
    if config.loss_remove_clip:
        on_pg_clipfrac = torch.tensor(0.0, device=on_pg_losses.device)
        on_pg_loss = VF.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)
    else:
        on_pg_losses2 = -advantages * torch.clamp(
            on_ratio, 1 - config.clip_ratio, 1 + config.clip_ratio
        )
        on_pg_clipfrac = VF.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), response_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = VF.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)

    # off-policy ratio
    off_ratio = torch.exp(log_prob)
    if config.off_policy_reshape == "classic_reject_token":
        my_off_ratio = off_ratio.detach()
        random_val = torch.rand_like(my_off_ratio)
        reject_coef = torch.where(
            my_off_ratio == 0,
            torch.zeros_like(my_off_ratio),
            torch.where(random_val < (1 - my_off_ratio), torch.zeros_like(my_off_ratio), 1.0 / my_off_ratio),
        )
    else:
        reject_coef = None

    off_ratio = _reshape_ratio(
        off_ratio,
        log_prob,
        old_log_prob,
        config.off_policy_reshape,
        config.off_policy_reshape_weight,
        config.off_policy_reshape_pow_exp,
    )

    off_ratio_mean = VF.masked_mean(off_ratio, prefix_mask * response_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    if reject_coef is not None:
        off_pg_losses = -advantages * reject_coef * off_ratio
    else:
        off_pg_losses = -advantages * off_ratio
    off_pg_loss = VF.masked_mean(off_pg_losses, prefix_mask * response_mask)
    off_pg_clipfrac = torch.tensor(0.0)

    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = VF.masked_mean(off_policy_probs, prefix_mask * response_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = VF.masked_mean(on_policy_probs, (1.0-prefix_mask) * response_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)

    if metrics is not None:
        metrics['actor/pg_loss'] = pg_loss.detach().item()
        metrics['actor/off_pg_loss'] = off_pg_loss.detach().item()
        metrics['actor/on_pg_loss'] = on_pg_loss.detach().item()
        metrics['actor/on_pg_clipfrac'] = on_pg_clipfrac.detach().item()
        metrics['actor/off_pg_clipfrac'] = off_pg_clipfrac.detach().item()
        metrics['actor/ppo_kl'] = ppo_kl.detach().item()
        metrics['actor/pg_clipfrac_lower'] = torch.tensor(0.0, device=pg_loss.device).detach().item()
        metrics['actor/off_ratio_mean'] = off_ratio_mean.detach().item()
        metrics['actor/on_policy_prob'] = on_policy_prob.detach().item()
        metrics['actor/off_policy_prob'] = off_policy_prob.detach().item()

    return pg_loss, on_pg_clipfrac, ppo_kl, torch.tensor(0.0, device=pg_loss.device)


@register_policy_loss(
    "srft",
    extra_keys={"prefix_mask", "token_level_scores", "target_probs"},
)
def compute_policy_loss_srft(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    token_level_scores: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    metrics: dict = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert config is not None
    assert not isinstance(config, AlgorithmConfig)

    if prefix_mask is None or token_level_scores is None or entropy is None:
        return compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_log_probs=rollout_log_probs,
        )

    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)

    # entropy-aware coefficients
    H_coef = VF.masked_mean(entropy, response_mask, dim=-1)
    H_coef = H_coef.detach()
    sft_coef = 0.5 * torch.exp(-1 * H_coef)
    on_coef = 0.1 * torch.exp(H_coef)

    # correct/incorrect mask from reward
    correct_answer_mask = token_level_scores.sum(-1) == 1
    on_advantages = torch.where(
        correct_answer_mask.unsqueeze(-1).expand_as(advantages),
        on_coef.view(-1, 1).expand_as(advantages),
        torch.tensor(-1.0, device=advantages.device, dtype=advantages.dtype).expand_as(advantages),
    )

    # on-policy ratio (with reshape)
    ratio = _reshape_ratio(
        torch.exp(negative_approx_kl),
        log_prob,
        old_log_prob,
        config.on_policy_reshape,
        config.on_policy_reshape_weight,
        config.on_policy_reshape_pow_exp,
    )

    if config.srft_type == "exp":
        srft_ratio = torch.exp(log_prob)
    elif config.srft_type == "classic_rl":
        srft_ratio = ratio
    elif config.srft_type == "minus_old":
        srft_ratio = log_prob - old_log_prob
    else:
        srft_ratio = log_prob

    on_pg_losses = -on_advantages * srft_ratio

    clip_upper_bound = getattr(config, "clip_upper_bound", 1.0)
    upper_bound = max(clip_upper_bound, 1.0 + config.clip_ratio)
    if config.loss_remove_clip:
        on_pg_loss = VF.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)
        on_pg_clipfrac = torch.tensor(0.0, device=on_pg_losses.device)
    else:
        on_pg_losses2 = -on_advantages * torch.clamp(
            log_prob, 1.0 - config.clip_ratio, upper_bound
        )
        on_pg_clipfrac = VF.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), response_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = VF.masked_mean(on_pg_losses, (~prefix_mask) * response_mask)

    # compute off-policy ratio
    target_probs = kwargs.get("target_probs")
    if target_probs is None:
        off_ratio = torch.exp(log_prob)
        reject_coef = None
        if config.off_policy_reshape == "classic_reject_token":
            my_off_ratio = off_ratio.detach()
            random_val = torch.rand_like(my_off_ratio)
            reject_coef = torch.where(
                my_off_ratio == 0,
                torch.zeros_like(my_off_ratio),
                torch.where(random_val < (1 - my_off_ratio), torch.zeros_like(my_off_ratio), 1.0 / my_off_ratio),
            )
        elif config.off_policy_reshape == "no_reshape":
            pass
        elif config.off_policy_reshape == "logp":
            off_ratio = log_prob * config.off_policy_reshape_weight
        elif config.off_policy_reshape == "p_logp":
            off_ratio = log_prob * config.off_policy_reshape_weight + off_ratio
        elif config.off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif config.off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif config.off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif config.off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif config.off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, config.off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {config.off_policy_reshape}")
    else:
        off_ratio = torch.exp(log_prob) / (target_probs + 1e-6)
        off_ratio = off_ratio * prefix_mask
        reject_coef = None

    off_max_clip = getattr(config, "off_policy_max_clip", -1)
    if off_max_clip != -1:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
    off_min_clip = getattr(config, "off_policy_min_clip", -1)
    if off_min_clip != -1:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)

    off_ratio_mean = VF.masked_mean(off_ratio, prefix_mask * response_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    if reject_coef is not None:
        off_pg_losses = -advantages * reject_coef * off_ratio
    else:
        off_pg_losses = -advantages * off_ratio
    off_pg_loss = VF.masked_mean(off_pg_losses, prefix_mask * response_mask)

    off_pg_clipfrac = torch.tensor(0.0)

    if off_pg_loss.isnan().item():
        off_pg_loss = torch.tensor(0.0, device=off_pg_losses.device)

    # extra SFT loss on off-policy part
    sft_loss = None
    off_policy_mask = prefix_mask.any(-1)
    if off_policy_mask.any():
        off_log_prob = (log_prob * sft_coef.view(-1, 1))[off_policy_mask]
        off_response_mask = response_mask[off_policy_mask]
        if off_log_prob.numel() > 0 and off_response_mask.sum().item() > 0:
            sft_loss = VF.masked_mean(-off_log_prob, off_response_mask)

    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    # off_pg_losses_sum = off_pg_losses.sum()
    # on_pg_losses_sum = on_pg_losses.sum()

    all_max_clip = getattr(config, "all_max_clip", -1)
    if all_max_clip != -1:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        response_mask = response_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    if sft_loss is not None and not torch.isnan(sft_loss):
        pg_loss = pg_loss + sft_loss

    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = VF.masked_mean(off_policy_probs, prefix_mask * response_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = VF.masked_mean(on_policy_probs, (1.0-prefix_mask) * response_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)

    if metrics is not None:
        metrics['actor/pg_loss'] = pg_loss.detach().item()
        metrics['actor/off_pg_loss'] = off_pg_loss.detach().item()
        metrics['actor/on_pg_loss'] = on_pg_loss.detach().item()
        metrics['actor/on_pg_clipfrac'] = on_pg_clipfrac.detach().item()
        metrics['actor/off_pg_clipfrac'] = off_pg_clipfrac.detach().item()
        metrics['actor/ppo_kl'] = ppo_kl.detach().item()
        metrics['actor/pg_clipfrac_lower'] = torch.tensor(0.0, device=pg_loss.device).detach().item()
        metrics['actor/off_ratio_mean'] = off_ratio_mean.detach().item()
        metrics['actor/on_policy_prob'] = on_policy_prob.detach().item()
        metrics['actor/off_policy_prob'] = off_policy_prob.detach().item()
        metrics['actor/sft_coef'] = sft_coef.mean().item()
        metrics['actor/on_coef'] = on_coef.mean().item()
        metrics['actor/H_coef'] = H_coef.mean().item()

    return pg_loss, on_pg_clipfrac, ppo_kl, torch.tensor(0.0, device=pg_loss.device)


@register_policy_loss(
    "hpt",
    extra_keys={"prefix_mask"},
)
def compute_policy_loss_hpt(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    metrics: dict = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert config is not None
    assert not isinstance(config, AlgorithmConfig)

    if prefix_mask is None or prefix_mask.numel() == 0:
        return compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_log_probs=rollout_log_probs,
        )

    off_policy_mask = prefix_mask.any(-1)

    sft_loss = None
    if off_policy_mask.any():
        off_log_prob = log_prob[off_policy_mask]
        off_response_mask = response_mask[off_policy_mask]
        if off_log_prob.numel() > 0 and off_response_mask.sum().item() > 0:
            sft_loss = VF.masked_mean(-off_log_prob, off_response_mask)

    on_policy_mask = ~off_policy_mask
    if on_policy_mask.any():
        on_policy_log_prob = log_prob[on_policy_mask]
        on_policy_old_log_prob = old_log_prob[on_policy_mask]
        on_policy_adv = advantages[on_policy_mask]
        on_policy_resp_mask = response_mask[on_policy_mask]
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_vanilla(
            old_log_prob=on_policy_old_log_prob,
            log_prob=on_policy_log_prob,
            advantages=on_policy_adv,
            response_mask=on_policy_resp_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_log_probs=None,
        )
    else:
        pg_loss = torch.tensor(0.0, device=log_prob.device)
        pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
        ppo_kl = torch.tensor(0.0, device=log_prob.device)
        pg_clipfrac_lower = torch.tensor(0.0, device=log_prob.device)

    ori_pg_loss = pg_loss
    if sft_loss is not None and not torch.isnan(sft_loss):
        sft_coef = getattr(config, "sft_loss_coef", 1.0)
        pg_loss = pg_loss + sft_loss * sft_coef

    if metrics is not None:
        metrics['actor/ori_pg_loss'] = ori_pg_loss.detach().item()
        metrics['actor/pg_loss'] = pg_loss.detach().item()
        metrics['actor/ppo_kl'] = ppo_kl.detach().item()
        metrics['actor/sft_coef'] = getattr(config, "sft_loss_coef", 1.0)
        if sft_loss is not None and not torch.isnan(sft_loss):
            metrics['actor/sft_loss'] = sft_loss.detach().item()
        else:
            metrics['actor/sft_loss'] = torch.tensor(0.0, device=pg_loss.device).detach().item()

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower



@register_policy_loss(
    "uft",
    extra_keys={"prefix_mask"},
)
def compute_policy_loss_uft(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    metrics: dict | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """UFT policy loss: vanilla PPO + SFT loss on prefix (hint) tokens.
    
    UFT (Unifying SFT and RFT) adds supervised learning on hint tokens
    while performing RL on the rest of the response.
    
    The SFT loss is computed on prefix tokens (marked by prefix_mask),
    which are sampled from the golden CoT with cosine annealing schedule.
    """
    assert config is not None
    assert not isinstance(config, AlgorithmConfig)

    response_mask_no_hint = response_mask
    if prefix_mask is not None:
        # Exclude hint/prefix tokens from PPO loss (SFT-only on hint tokens)
        response_mask_no_hint = response_mask & (~prefix_mask)

    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask_no_hint,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_log_probs=rollout_log_probs,
    )

    ori_pg_loss = pg_loss
    sft_loss_coef = getattr(config, "sft_loss_coef", 0.0)
    if sft_loss_coef > 0 and prefix_mask is not None:
        hint_mask = prefix_mask * response_mask
        hint_sft_loss = VF.masked_mean(-log_prob, hint_mask)
        pg_loss = pg_loss + hint_sft_loss * sft_loss_coef
        
        # Log metrics
        if metrics is not None:
            metrics['actor/ori_pg_loss'] = ori_pg_loss.detach().item()
            metrics["actor/sft_coef"] = sft_loss_coef
            metrics["actor/sft_loss"] = hint_sft_loss.detach().item()
            # Log hint coverage: fraction of response that is hint
            hint_token_count = hint_mask.sum().item()
            response_token_count = response_mask.sum().item()
            if response_token_count > 0:
                metrics["actor/hint_ratio"] = hint_token_count / response_token_count

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = VF.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgorithmConfig), "passing AlgorithmConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - VF.masked_mean(advantages, response_mask)) * (
        log_prob - VF.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = VF.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgorithmConfig), "passing AlgorithmConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = VF.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgorithmConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgorithmConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio
    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = VF.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = VF.masked_mean((clipped * (advantages < 0)).float(), response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _reshape_ratio(
    ratio: torch.Tensor,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    reshape: str,
    reshape_weight: float,
    reshape_pow_exp: float,
):
    if reshape == "no_reshape":
        return ratio
    if reshape == "logp":
        return log_prob * reshape_weight
    if reshape == "p_logp":
        return log_prob * reshape_weight + ratio
    if reshape == "square_root":
        return torch.sqrt(ratio)
    if reshape == "p_div_p_0.1":
        return ratio / (ratio + 0.1)
    if reshape == "p_div_p_0.5":
        return ratio / (ratio + 0.5)
    if reshape == "p_div_p_0.3":
        return ratio / (ratio + 0.3)
    if reshape == "pow":
        return torch.pow(ratio, reshape_pow_exp)
    if reshape == "classic_reject_token":
        # handled in luffy loss to use reject_coef
        return ratio
    raise ValueError(f"Invalid reshape: {reshape}")
