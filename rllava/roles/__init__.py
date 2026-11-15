"""Model roles, imported on demand to avoid loading unrelated backends."""

from importlib import import_module

__all__ = ["Actor", "Critic", "Ref", "Reward", "Generator"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    role = getattr(import_module(f".{name.lower()}", __name__), name)
    globals()[name] = role
    return role
