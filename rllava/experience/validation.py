"""Validate samples once at ingestion, and groups at assembly/queue boundaries."""
from __future__ import annotations

import torch

from rllava.data.protocol import DataProto
from .protocol import EXPERIENCE_MODE, IDENTITY_FIELDS, LEARNING_FIELDS, SCHEMA_VERSION, ExperienceGroup


def validate_sample(sample: DataProto) -> None:
    """Check one rollout record before handing it to the experience pipeline."""
    if sample.batch is None or len(sample) != 1:
        raise ValueError("Expected a single-sample DataProto")
    batch = sample.batch
    mask_fields = ("response_mask", "action_mask", "loss_mask", "rollout_logprob_mask")
    required = {"prompts", "responses", "input_ids", "attention_mask", "position_ids",
                "rollout_log_probs", "rollout_temperature", "policy_version", "turn_count",
                "token_level_scores", *mask_fields}
    if missing := required - set(batch.keys()):
        raise ValueError(f"Missing experience tensors: {sorted(missing)}")
    if LEARNING_FIELDS & set(batch.keys()):
        raise ValueError("Rollout experience contains learning-derived fields")
    if sample.meta_info.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported experience schema_version")
    if sample.meta_info.get("experience_mode") != EXPERIENCE_MODE:
        raise ValueError("Unsupported experience_mode")
    if not {"min_pixels", "max_pixels", "video_fps"} <= sample.meta_info.keys():
        raise ValueError("Missing multimodal preprocessing metadata")
    for key in IDENTITY_FIELDS:
        if key not in sample.non_tensor_batch or sample.non_tensor_batch[key].shape != (1,):
            raise ValueError(f"Missing or invalid identity field: {key}")
        if not sample.non_tensor_batch[key][0]:
            raise ValueError(f"Empty identity field: {key}")
    if sample.non_tensor_batch["sample_source"][0] != "policy":
        raise ValueError("Only policy samples are supported")

    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    for key in ("prompts", "responses", "input_ids"):
        if batch[key].ndim != 2 or batch[key].dtype not in integer_types:
            raise ValueError(f"{key} must be integer [1,L]")
    prompt_width, response_width = batch["prompts"].shape[-1], batch["responses"].shape[-1]
    if prompt_width == 0 or response_width == 0:
        raise ValueError("Prompt and response widths must be positive")
    full_shape = (1, prompt_width + response_width)
    if batch["input_ids"].shape != full_shape or batch["attention_mask"].shape != full_shape:
        raise ValueError("input_ids and attention_mask must have shape [1,P+R]")
    if (not torch.equal(batch["input_ids"][:, :prompt_width], batch["prompts"])
            or not torch.equal(batch["input_ids"][:, prompt_width:], batch["responses"])):
        raise ValueError("Prompt/response tokens are misaligned")
    positions = batch["position_ids"]
    if positions.ndim not in (2, 3) or positions.shape[-1] != full_shape[-1]:
        raise ValueError("position_ids must have shape [1,L] or [1,axes,L]")
    attention = batch["attention_mask"]
    if not bool(((attention == 0) | (attention == 1)).all()):
        raise ValueError("attention_mask must be binary")
    for key in mask_fields:
        if not torch.equal(batch[key], attention[:, prompt_width:]):
            raise ValueError(f"{key} must match the response attention mask")
    mask = batch["response_mask"].bool()
    if not bool(mask.any()):
        raise ValueError("Every response must contain at least one valid token")
    if bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError("response_mask must use right padding")
    for key in ("rollout_log_probs", "token_level_scores"):
        value = batch[key]
        if value.shape != (1, response_width) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite [1,R]")
        if not bool((value[~mask] == 0).all()):
            raise ValueError(f"{key} must be zero on padding")
    if not bool((batch["rollout_log_probs"][mask] <= 1e-5).all()):
        raise ValueError("Log probabilities must be nonpositive")
    for key, minimum in (("policy_version", 0), ("turn_count", 1)):
        value = batch[key]
        if value.shape != (1,) or value.dtype not in integer_types or value[0].item() < minimum:
            raise ValueError(f"{key} must be integer [1] with value >= {minimum}")
    temperature = batch["rollout_temperature"]
    if (temperature.shape != (1,) or not bool(torch.isfinite(temperature).all())
            or temperature[0].item() < 0):
        raise ValueError("rollout_temperature must be finite nonnegative [1]")


def validate_groups(groups: list[ExperienceGroup], *, expected_policy_version: int | None = None) -> None:
    """Check completeness, identities and sampling settings in one pass.

    Members must have passed validate_sample; this does not rescan token tensors.
    """
    seen = {key: set() for key in ("group_id", "trajectory_id", "sample_id")}
    version, temperature = None, None
    for group in groups:
        if group.expected_trajectories <= 0 or len(group.samples) != group.expected_trajectories:
            raise ValueError("Incomplete group: last_response requires one sample per trajectory")
        if group.group_id in seen["group_id"]:
            raise ValueError("Duplicate group_id")
        seen["group_id"].add(group.group_id)
        for sample in group.samples:
            if len(sample) != 1 or sample.non_tensor_batch["uid"][0] != group.group_id:
                raise ValueError("Group members must be single samples with matching uid")
            if LEARNING_FIELDS & set(sample.batch.keys()):
                raise ValueError("Rollout experience contains learning-derived fields")
            for key in ("trajectory_id", "sample_id"):
                identity = sample.non_tensor_batch[key][0]
                if identity in seen[key]:
                    raise ValueError(f"Duplicate {key}")
                seen[key].add(identity)
            current_version = sample.batch["policy_version"][0].item()
            current_temperature = sample.batch["rollout_temperature"][0].item()
            if expected_policy_version is not None and current_version != expected_policy_version:
                raise ValueError("Unexpected policy_version")
            if version is not None and current_version != version:
                raise ValueError("Mixed policy versions")
            if temperature is not None and current_temperature != temperature:
                raise ValueError("Mixed rollout temperatures")
            version, temperature = current_version, current_temperature
