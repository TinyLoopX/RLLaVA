import torch
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, Dict, Tuple
from rllava.data.protocol import DataProto
from rllava.utils import torch_functional as VF




class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


@register_adv_est(AdvantageEstimator.GAE)
@torch.no_grad()
def compute_advantage_gae(data: DataProto, config):
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        data: `(DataProto)`
            Contains the following keys:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - values: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length). The token after eos tokens have mask zero.
        config: Configuration object containing:
            - gamma: `(float)`
                discounted factor used in RL
            - lam: `(float)`
                lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    values = data.batch["values"]
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]

    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]

    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + config.gamma * nextvalues - values[:, t]
        lastgaelam_ = delta + config.gamma * config.lam * lastgaelam

        # skip values and TD-error on observation tokens
        nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
        lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
@torch.no_grad()
def compute_advantage_reinforce_pp(data: DataProto, config):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        data: `(DataProto)`
            Contains the following keys:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            config: Configuration object containing:
                - gamma: `(float)`
                    discounted factor used in RL

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0

    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + config.gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    advantages = advantages * response_mask
    return advantages, returns


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
@torch.no_grad()
def compute_advantage_reinforce_pp_b(data: DataProto, config):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        data: `(DataProto)`
            Contains the following keys:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            - uid: `(List)`
                List of unique identifiers for each sample
        config: Configuration object containing algorithm parameters

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    for i in range(bsz):
        scores[i] = scores[i] - id2mean[index[i]]

    scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    scores = VF.masked_whiten(scores, response_mask) * response_mask
    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO)
@torch.no_grad()
def compute_advantage_grpo(data: DataProto, config, epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        data: DataProto object containing:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            - uid: `(List)`
                List of unique identifiers for each sample
        config: Configuration object containing algorithm parameters
        epsilon: `(float)`
            epsilon value to avoid division by zero (default: 1e-6)
        norm_adv_by_std_in_grpo: `(bool)`
            whether to normalize advantages by standard deviation (default: True)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = getattr(config, "norm_adv_by_std_in_grpo", True)
    
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_PASSK) 
@torch.no_grad()
def compute_advantage_grpo_passk(data: DataProto, config, epsilon: float = 1e-6):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        data: DataProto object containing:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            - uid: `(List)`
                List of unique identifiers for each sample
        config: Configuration object containing algorithm parameters
        epsilon: `(float)`
            epsilon value to avoid division by zero (default: 1e-6)
        norm_adv_by_std_in_grpo: `(bool)`
            whether to normalize advantages by standard deviation (default: True)

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = getattr(config, "norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        idx = index[i]
        id2scores[idx].append(scores[i])
        id2indices[idx].append(i)

    for idx in id2scores:
        rewards = torch.stack(id2scores[idx])  # (k,)
        if rewards.numel() < 2:
            raise ValueError(
                f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
            )
        topk, topk_idx = torch.topk(rewards, 2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0].item()]
        advantage = r_max - r_second_max
        if norm_adv_by_std_in_grpo:
            std = torch.std(rewards)
            advantage = advantage / (std + epsilon)
        advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.RLOO)
@torch.no_grad()
def compute_advantage_rloo(data: DataProto, config):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        data: DataProto object containing:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            - uid: `(List)`
                List of unique identifiers for each sample
        config: Configuration object containing algorithm parameters

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    for i in range(bsz):
        response_num = len(id2score[index[i]])
        if response_num > 1:
            scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                response_num - 1
            )
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


@register_adv_est(AdvantageEstimator.REMAX)
@torch.no_grad()
def compute_advantage_remax(data: DataProto, config):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        data: (DataProto) contains input tensors:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - reward_baselines: `(torch.Tensor)`
                shape: (bs,)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
        config: Configuration object containing algorithm parameters

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    reward_baselines = data.batch["reward_baselines"]
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
    return advantages, returns


@register_adv_est(AdvantageEstimator.OPO)
@torch.no_grad()
def compute_advantage_opo(data: DataProto, config, epsilon: float = 1e-6):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        data: DataProto object containing:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            - uid: `(List)`
                List of unique identifiers for each sample
        config: Configuration object containing algorithm parameters
        epsilon: `(float)`
            epsilon value to avoid division by zero (default: 1e-6)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2len[index[i]].append(response_length[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2bsl[idx] = torch.tensor(0.0)
        elif len(id2score[idx]) > 1:
            score_tensor = torch.stack(id2score[idx])
            len_tensor = torch.stack(id2len[idx])
            id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    for i in range(bsz):
        scores[i] = scores[i] - id2bsl[index[i]]
    scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GPG)
@torch.no_grad()
def compute_advantage_gpg(data: DataProto, config, epsilon: float = 1e-6, f_norm: float = 1.0, alpha: float = 1.0):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        data: DataProto object containing:
            - token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            - response_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            - uid: `(List)`
                List of unique identifiers for each sample
        config: Configuration object containing algorithm parameters
        epsilon: `(float)`
            epsilon value to avoid division by zero (default: 1e-6)
        f_norm: `(float)`
        alpha: `(float)`

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    bsz = scores.shape[0]
    m = torch.count_nonzero(scores)
    alpha = bsz / m.clamp(min=1)

    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    for i in range(bsz):
        scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores