"""RL experience identities and grouping; DataProto stays the carrier."""
from __future__ import annotations

from dataclasses import dataclass

from rllava.data.protocol import DataProto

SCHEMA_VERSION = 1
EXPERIENCE_MODE = "last_response"
ExperienceBatch = DataProto
LEARNING_FIELDS = frozenset({
    "old_log_probs", "ref_log_probs", "values", "token_level_rewards", "advantages", "returns",
})
IDENTITY_FIELDS = ("uid", "trajectory_id", "sample_id", "sample_source", "termination_reason")


@dataclass
class ExperienceGroup:
    group_id: str
    expected_trajectories: int
    samples: tuple[DataProto, ...]

    @property
    def policy_version(self) -> int:
        return int(self.samples[0].batch["policy_version"][0].item())

