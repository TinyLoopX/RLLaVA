"""GUI/tool agent backed by the token-first context builder."""

import copy
import json
import numpy as np

from .base import AgentAction, SingleTurnAgent
from .context import ContextBuilder


class GUIAgent(SingleTurnAgent):
    multi_turn = True

    def __init__(self, tokenizer, processor=None):
        super().__init__(tokenizer, processor)
        self.context = ContextBuilder(tokenizer, processor)

    def start(self, data):
        super().start(self.context.start_from_data(data))

    def reset_arguments(self):
        task = self.data.non_tensor_batch.get("task_config", [None])[0]
        if isinstance(task, str):
            task = json.loads(task)
        return (), {"task_config": task} if task is not None else {}

    @staticmethod
    def _image(observation):
        if isinstance(observation, dict):
            image = observation.get("screenshot")
            return image if image is not None else observation.get("image")
        if hasattr(observation, "save") or isinstance(observation, (bytes, bytearray)):
            return observation
        return None

    def observe_reset(self, observation):
        image = self._image(observation)
        if image is not None:
            self.data = self.context.inject_initial_observation(self.data, image)
        else:
            self.data.non_tensor_batch.pop("initial_prompt_text", None)

    def build_inputs(self):
        self.context.materialize_generation_inputs(self.data)
        self.data.batch["position_ids"] = self.context.compute_position_ids()
        # Preserve the actual request images when subsequent observations arrive.
        if "multi_modal_data" in self.data.non_tensor_batch:
            self.data.non_tensor_batch["multi_modal_data"] = copy.deepcopy(
                self.data.non_tensor_batch["multi_modal_data"]
            )
        return super().build_inputs()

    def accept_output(self, output, env, tools):
        super().accept_output(output, env, tools)
        content = self.context.append_assistant_generation(output)
        self.tools = tools
        for index, tool in enumerate(tools):
            call = tool.extract_tool_call(content)
            if call is not None:
                return AgentAction("tool", call, index)
        if env is not None:
            action = env.extract_action(content)
            return AgentAction("environment", action) if action is not None else AgentAction("feedback")
        return AgentAction("finish")

    def observe(self, observation, reward, info, *, continue_episode, action):
        if not continue_episode:
            return
        image = self._image(observation)
        if action.kind == "tool":
            tool = self.tools[action.tool_index]
            content = observation.get("text", "") if isinstance(observation, dict) else str(observation)
            self.context.append_external_delta(
                content, role="tool", image=image,
                tool_name=getattr(tool, "name", type(tool).__name__),
            )
        else:
            content = (
                "Invalid action format. Please output a valid action."
                if action.kind == "feedback" else f"{observation if image is None else ''}\n{info}"
            )
            self.context.append_external_delta(content, role="environment", image=image)

    def finish(self, step_reward, eval_reward):
        # Keep the last generated [prompt | response] tensors and their exact
        # image snapshot; final observations are not model-generated responses.
        self.output.non_tensor_batch.update({
            "env_reward": np.array([step_reward + eval_reward], dtype=np.float32),
            "eval_reward": np.array([eval_reward], dtype=np.float32),
            "step_reward_sum": np.array([step_reward], dtype=np.float32),
        })
        return self.output
