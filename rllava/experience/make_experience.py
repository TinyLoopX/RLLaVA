"""Build complete groups, then pack variable prompts at the training boundary."""
from __future__ import annotations

import numpy as np
import torch

from rllava.data.protocol import DataProto
from .protocol import ExperienceGroup
from .validation import validate_groups, validate_sample


class ExperienceMaker:
    """Accept producer-assigned identities and versions; never infer them.

    ``build_groups`` does no padding. ``make_experience`` inserts masked padding
    at the end of each prompt segment and leaves response coordinates intact.
    Samples are handed off by reference. After handing them off, the producer
    must not modify them. Device placement remains the caller's responsibility.
    """

    def __init__(self, pad_token_id: int):
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError('pad_token_id must be a nonnegative integer')
        self.pad_token_id = pad_token_id

    def build_groups(self, samples: list[DataProto], *, expected_trajectories: int) -> list[ExperienceGroup]:
        if type(expected_trajectories) is not int or expected_trajectories <= 0:
            raise ValueError('expected_trajectories must be a positive integer')
        by_uid = {}
        for sample in samples:
            validate_sample(sample)
            sample.non_tensor_batch.pop("multi_modal_inputs", None)
            by_uid.setdefault(sample.non_tensor_batch["uid"][0], []).append(sample)
        groups = [ExperienceGroup(uid, expected_trajectories, tuple(members))
                  for uid, members in by_uid.items()]
        validate_groups(groups)
        return groups

    def make_experience(self, groups: list[ExperienceGroup]) -> DataProto:
        """Pack validated groups without repeating sample-level tensor checks."""
        if not groups:
            raise ValueError('Cannot assemble an empty experience batch')
        validate_groups(groups)
        samples = [sample for group in groups for sample in group.samples]
        first = samples[0]
        metadata = {k: v for k, v in first.meta_info.items() if k != "global_token_num"}
        width = first.batch["responses"].shape[-1]
        target = max(s.batch["prompts"].shape[-1] for s in samples)
        for sample in samples:
            if sample.batch["responses"].shape[-1] != width:
                raise ValueError('Different response widths are not supported')
            if set(sample.batch.keys()) != set(first.batch.keys()):
                raise ValueError('Tensor fields differ across samples')
            if sample.non_tensor_batch.keys() != first.non_tensor_batch.keys():
                raise ValueError('Non-tensor fields differ across samples')
            # Shared metadata contains ordinary configuration values, not tensors.
            current_metadata = {k: v for k, v in sample.meta_info.items() if k != "global_token_num"}
            if current_metadata != metadata:
                raise ValueError('Incompatible experience metadata')

        aligned = [self._align_prompt(sample, target) for sample in samples]
        result = DataProto.concat(aligned)
        result.meta_info = metadata
        result.meta_info["global_token_num"] = result.batch["attention_mask"].sum(-1).tolist()
        return result

    def _align_prompt(self, sample: DataProto, target: int) -> DataProto:
        tensors = dict(sample.batch.items())
        current = tensors["prompts"].shape[-1]
        padding = target - current
        if padding:
            for key, fill in (("prompts", self.pad_token_id), ("input_ids", self.pad_token_id),
                              ("attention_mask", 0), ("position_ids", 0)):
                value = tensors[key]
                shape = (*value.shape[:-1], padding)
                pad = torch.full(shape, fill, dtype=value.dtype, device=value.device)
                tensors[key] = torch.cat([value[..., :current], pad, value[..., current:]], dim=-1)
        # A [1,P] raw_prompt_ids array and a [1,Q] array cannot concatenate.
        # Treat each row as one object, retaining its original nested structure.
        non_tensors = {}
        for key, value in sample.non_tensor_batch.items():
            if key == "multi_modal_inputs":
                continue
            rows = np.empty(1, dtype=object)
            rows[0] = value[0]
            non_tensors[key] = rows
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=sample.meta_info)
