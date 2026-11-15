"""Training-loop and resource lifecycle orchestration."""

__all__ = ["Controller"]


def __getattr__(name):
    if name == "Controller":
        from .controller import Controller
        return Controller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
