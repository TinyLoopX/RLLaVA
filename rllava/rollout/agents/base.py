"""Per-trajectory agent state. Agents never execute or close environments."""

from dataclasses import dataclass


@dataclass
class AgentAction:
    kind: str
    payload: object = None
    tool_index: int = 0


class SingleTurnAgent:
    multi_turn = False

    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = None
        self.output = None

    def start(self, data):
        self.data = data

    def reset_arguments(self):
        return (self.data,), {}

    def observe_reset(self, observation):
        pass

    def build_inputs(self):
        self.data.meta_info["n"] = 1
        return self.data

    def accept_output(self, output, env, tools):
        for key, value in self.data.non_tensor_batch.items():
            if key not in output.non_tensor_batch:
                output.non_tensor_batch[key] = value
        self.output = output
        return AgentAction("finish")

    def observe(self, observation, reward, info, *, continue_episode, action):
        pass

    def finish(self, step_reward, eval_reward):
        return self.output
