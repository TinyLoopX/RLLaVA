"""Learning algorithms and training orchestration."""

__all__ = ["Learner"]


def __getattr__(name):
    if name == "Learner":
        from .train import Learner
        return Learner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
