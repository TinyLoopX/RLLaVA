"""Agent policies selected independently of rollout scheduling."""

from importlib import import_module

_AGENTS = {
    "single_turn": ("base", "SingleTurnAgent"),
    "multi_turn": ("multi_turn", "MultiTurnAgent"),
    "gui": ("gui", "GUIAgent"),
}


def get_agent(name):
    if name not in _AGENTS:
        raise ValueError(f"Unknown rollout agent {name!r}; choose from {tuple(_AGENTS)}")
    module, symbol = _AGENTS[name]
    return getattr(import_module(f".{module}", __name__), symbol)
