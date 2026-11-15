"""Compatibility adapter for the migrated multi-turn rollout implementation."""

import asyncio
from types import SimpleNamespace
from rllava.rollout.agents.multi_turn import MultiTurnAgent


class MultiTurnWorkflow:
    agent_cls = MultiTurnAgent

    def __init__(self, reward, tokenizer, processor, max_turns=5,
                 discount=1.0, env_config_path=None):
        self.worker_config = SimpleNamespace(
            max_turns=max_turns, env_config_path=env_config_path, tool_config_path=None,
        )

    def run(self, rollout, gen_batch, source_data=None):
        return rollout.run_workers(gen_batch, source_data, agent_cls=self.agent_cls,
                                   worker_config=self.worker_config)

    async def arun(self, rollout, gen_batch, source_data=None):
        return await asyncio.to_thread(self.run, rollout, gen_batch, source_data)
