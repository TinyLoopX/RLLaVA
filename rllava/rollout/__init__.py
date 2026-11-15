"""Rollout scheduling and per-trajectory execution."""

__all__ = ["RolloutManager", "RolloutWorker"]


def __getattr__(name):
    if name == "RolloutManager":
        from .rollout_manager import RolloutManager
        return RolloutManager
    if name == "RolloutWorker":
        from .rollout_worker import RolloutWorker
        return RolloutWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
