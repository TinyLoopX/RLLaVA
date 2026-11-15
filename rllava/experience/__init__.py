"""Synchronous rollout experience contracts, assembly and group handoff.

Model inference and advantage computation remain in ``rllava.train.Learner``.
"""
from importlib import import_module

__all__ = [
    "ExperienceConfig", "ExperienceBatch", "ExperienceGroup", "ExperienceMaker",
    "SampleQueue", "QueueFullError", "validate_sample", "validate_groups",
]

_MODULES = {
    "ExperienceConfig": "config",
    "ExperienceBatch": "protocol", "ExperienceGroup": "protocol",
    "ExperienceMaker": "make_experience",
    "SampleQueue": "queue", "QueueFullError": "queue",
    "validate_sample": "validation", "validate_groups": "validation",
}


def __getattr__(name):
    if name in _MODULES:
        return getattr(import_module(f".{_MODULES[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
