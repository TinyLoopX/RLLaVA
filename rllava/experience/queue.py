"""Nonblocking FIFO handoff for a single synchronous coordinator.

The producer relinquishes samples on put; the consumer owns them on pop.
Only queued identities are checked for duplicates. No payloads are copied.
"""
from __future__ import annotations

from collections import deque

from .config import ExperienceConfig
from .protocol import SCHEMA_VERSION, ExperienceGroup
from .validation import validate_groups


class QueueFullError(RuntimeError):
    """The complete submission would exceed the configured group capacity."""


class SampleQueue:
    def __init__(self, config: ExperienceConfig | None = None, *, policy_version: int = 0):
        self.config = config if config is not None else ExperienceConfig()
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        self._policy_version = policy_version
        self._groups = deque()

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def __len__(self) -> int:
        """Number of complete groups available for consumption."""
        return len(self._groups)

    @property
    def remaining_capacity(self) -> int:
        return self.config.max_groups - len(self)

    def put_groups(self, groups: list[ExperienceGroup]) -> None:
        """Accept validated rollout samples; check group/version/capacity here."""
        if len(groups) > self.remaining_capacity:
            raise QueueFullError(f"Cannot put {len(groups)} groups: {self.remaining_capacity} slots available")
        validate_groups(list(self._groups) + groups, expected_policy_version=self.policy_version)
        self._groups.extend(groups)

    def pop_groups(self, num_groups: int, *, policy_version: int) -> list[ExperienceGroup]:
        """Return exactly num_groups, or [] without consuming anything."""
        if policy_version != self.policy_version:
            raise ValueError('Unexpected policy_version')
        if num_groups <= 0:
            raise ValueError('num_groups must be positive')
        if len(self) < num_groups:
            return []
        return [self._groups.popleft() for _ in range(num_groups)]

    def discard_stale(self, policy_version: int) -> int:
        """Advance after weight synchronization; drop, never relabel, old data."""
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if policy_version < self.policy_version:
            raise ValueError('Cannot roll back policy_version')
        if policy_version == self.policy_version:
            return 0
        discarded = len(self)
        self._groups.clear()
        self._policy_version = policy_version
        return discarded

    def state_dict(self) -> dict:
        """Return live payload references; serialize at a paused iteration boundary."""
        return {
            "schema_version": SCHEMA_VERSION,
            "experience_mode": self.config.experience_mode,
            "policy_version": self.policy_version,
            "groups": list(self._groups),
        }

    def load_state_dict(self, state: dict) -> None:
        """Take ownership of deserialized groups, preserving FIFO order."""
        if state["schema_version"] != SCHEMA_VERSION:
            raise ValueError('Unsupported queue schema_version')
        if state["experience_mode"] != self.config.experience_mode:
            raise ValueError('Incompatible experience_mode')
        version, groups = state["policy_version"], state["groups"]
        if type(version) is not int or version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if len(groups) > self.config.max_groups:
            raise QueueFullError("Checkpoint exceeds queue capacity")
        validate_groups(groups, expected_policy_version=version)
        self._groups = deque(groups)
        self._policy_version = version
