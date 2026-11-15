"""Compatibility adapter; use rollout.agent=gui for new pipelines."""

import asyncio
from types import SimpleNamespace
from rllava.rollout.agents.gui import GUIAgent


class MultiTurnWorkflow:
    agent_cls = GUIAgent

    def __init__(self, reward, tokenizer, processor, max_turns=5,
                 tool_config_path=None, env_config_path=None, trajectory_dir=None):
        if trajectory_dir is not None:
            raise NotImplementedError("Trajectory storage is not implemented; configure rollout logging separately")
        self.worker_config = SimpleNamespace(
            max_turns=max_turns, env_config_path=env_config_path,
            tool_config_path=tool_config_path,
        )

    def run_single(self, rollout, data):
        return rollout.run_workers(data, agent_cls=self.agent_cls,
                                   worker_config=self.worker_config)

    async def arun_single(self, rollout, data):
        return await asyncio.to_thread(self.run_single, rollout, data)
