import asyncio
import json
from typing import Any

import numpy as np
import torch

from rllava.data.protocol import DataProto
from rllava.data.trajectory_storage import TrajectoryStorage
from rllava.ppo.tool import initialize_tools_from_config
from rllava.ppo.env import initialize_env_from_config
from rllava.workflow.context import ContextBuilder
import rllava.utils.torch_functional as VF


class MultiTurnWorkflow:
    """Multi-turn workflow with tool calling and environment interaction."""
    
    def __init__(
        self,
        reward,   # Reward function
        tokenizer,
        processor,
        max_turns: int = 5,
        tool_config_path: str = None,
        env_config_path: str = None,
        trajectory_dir: str = None,
    ):
        self.reward = reward
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_turns = max_turns
        self.tool_config_path = tool_config_path
        self.env_config_path = env_config_path
        self.trajectory_dir = trajectory_dir
        self.context = ContextBuilder(self.tokenizer, self.processor)
        self.tools_list = self._initialize_tools()
        self.env = self._initialize_env()
        self.storage = TrajectoryStorage(trajectory_dir) if trajectory_dir else None

    async def arun_single(self, rollout: Any, data: DataProto) -> DataProto:
        """Async wrapper for run_single."""
        return await asyncio.to_thread(self.run_single, rollout, data)

    def create_session(self) -> "MultiTurnWorkflow":
        """Create a per-sample workflow session with isolated state."""
        return MultiTurnWorkflow(
            reward=self.reward,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_turns=self.max_turns,
            tool_config_path=self.tool_config_path,
            env_config_path=self.env_config_path,
            trajectory_dir=self.trajectory_dir,
        )

    def close(self):
        """Release resources held by this workflow session."""
        if self.env and hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
        for tool in self.tools_list or []:
            if hasattr(tool, "release"):
                try:
                    tool.release()
                except Exception:
                    pass

    def run_single(self, rollout: Any, data: DataProto) -> DataProto:
        """Run multi-turn generation for a single sample.
        
        Args:
            rollout: Rollout instance with generate_sequences method
            data: Single sample DataProto (batch_size=1)
        
        Note: For batch processing, the caller should iterate over each sample.
        """
        t = 0
        step_rewards = [0.0] * self.max_turns
        done = False

        if self.storage:
            task_id = data.non_tensor_batch.get("uid", [""])[0]
            instruction = ""
            raw_prompt_ids = data.non_tensor_batch.get("raw_prompt_ids", None)
            if raw_prompt_ids is not None and len(raw_prompt_ids) > 0:
                decoded = self.tokenizer.decode(
                    raw_prompt_ids[0], skip_special_tokens=False,
                )
                if isinstance(decoded, str):
                    instruction = decoded
            self.storage.start_trajectory(task_id=task_id, instruction=instruction)

        data = self.context.start_from_data(data)

        if self.env:
            task_config = None
            if "task_config" in data.non_tensor_batch:
                task_config = data.non_tensor_batch["task_config"][0]
                if isinstance(task_config, str):
                    try:
                        task_config = json.loads(task_config)
                    except (ValueError, TypeError):
                        task_config = None
            obs = self.env.reset(task_config=task_config) if task_config is not None else self.env.reset()
            initial_image = None
            if isinstance(obs, dict) and obs.get("screenshot") is not None:
                initial_image = obs["screenshot"]
            elif hasattr(obs, "save") or isinstance(obs, (bytes, bytearray)):
                initial_image = obs
            if initial_image is not None:
                data = self.context.inject_initial_observation(data, initial_image)
            else:
                data.non_tensor_batch.pop("initial_prompt_text", None)

        gen_output = None
        data.meta_info["n"] = 1
        while not done and t < self.max_turns:
            raw_prompt_ids = self.tokenizer.decode(
                data.non_tensor_batch['raw_prompt_ids'][0], skip_special_tokens=True
            )
            gen_output = rollout.generate_sequences(data)
            str_sequence = self.tokenizer.decode(
                gen_output.batch.get("responses")[0], skip_special_tokens=False
            )
            #print(f"str_sequence: {str_sequence}")
            content = self.context.append_assistant_generation(gen_output)
            is_processed = False
            
            if self.tools_list:
                for tool in self.tools_list:
                    tool_call_data = tool.extract_tool_call(content)
                    if tool_call_data:
                        tool_result, tool_success = tool.execute(tool_call_data)
                        step_rewards[t] = 1.0 if tool_success else 0.0
                        tool_name = getattr(tool, "name", type(tool).__name__)
                        tool_image = None
                        if isinstance(tool_result, dict) and "image" in tool_result:
                            tool_image = tool_result["image"]
                            tool_content = f"<image>\n{tool_result.get('text', '')}"
                        elif (
                            hasattr(tool_result, "save")
                            or isinstance(tool_result, (bytes, bytearray))
                            or (isinstance(tool_result, dict) and "bytes" in tool_result)
                        ):
                            tool_image = tool_result
                            tool_content = "<image>"
                        else:
                            tool_content = str(tool_result)

                        self.context.append_external_delta(
                            tool_content, role="tool",
                            image=tool_image, tool_name=tool_name,
                        )
                        if self.storage:
                            self.storage.record_tool_step(
                                step=t,
                                action=content,
                                tool_name=tool_name,
                                result=tool_result,
                                reward=step_rewards[t],
                                done=done,
                                logprob=gen_output.batch.get("old_log_probs"),
                                token_ids=gen_output.batch.get("responses"),
                            )
                        is_processed = True
                        break
            
            if not is_processed and self.env:
                action = self.env.extract_action(content)
                if action:
                    obs, step_reward, env_done, info = self.env.step(action)
                    step_rewards[t] = step_reward
                    if env_done:
                        done = True
                else:
                    obs = {"screenshot": None}
                    info = {"error": "Invalid action format. Please output a valid action."}
                    step_reward = 0.0

                obs_image = None
                if isinstance(obs, dict) and obs.get("screenshot") is not None:
                    obs_image = obs["screenshot"]
                elif hasattr(obs, "save") or isinstance(obs, (bytes, bytearray)):
                    obs_image = obs

                if obs_image is not None:
                    obs_text = f"<image>\n{info}"
                else:
                    obs_text = f"{obs}\n{info}"
                self.context.append_external_delta(
                    obs_text, role="environment", image=obs_image,
                )
                if self.storage:
                    self.storage.record_environment_step(
                        step=t,
                        action=content,
                        obs=obs,
                        info=info,
                        reward=step_reward if action else 0.0,
                        done=done,
                        logprob=gen_output.batch.get("old_log_probs"),
                        token_ids=gen_output.batch.get("responses"),
                    )

                is_processed = True
            self.context.materialize_generation_inputs(data)
            gen_output.non_tensor_batch["multi_modal_data"] = data.non_tensor_batch["multi_modal_data"]
            
            if not is_processed:
                done = True
            
            t += 1
        
        # Align last gen_output with full trajectory (same length as mRoPE).
        if gen_output is not None:
            device = gen_output.batch["input_ids"].device
            dtype = gen_output.batch["input_ids"].dtype
            full_list = self.context.input_ids
            L = len(full_list)
            full_ids = torch.tensor([full_list], dtype=dtype, device=device)
            attn = torch.ones(1, L, dtype=torch.long, device=device)
            pos = self.context.compute_position_ids().to(device)
            if pos.size(-1) != L:
                raise RuntimeError(
                    f"position_ids length {pos.size(-1)} != input_ids length {L}"
                )
            resp_mask = gen_output.batch["response_mask"]
            resp_valid = int(resp_mask[0].sum().item())
            prompt_len = max(L - resp_valid, 0) if resp_valid > 0 else L
            gen_output.batch["input_ids"] = full_ids
            gen_output.batch["attention_mask"] = attn
            gen_output.batch["position_ids"] = pos
            gen_output.batch["prompts"] = full_ids[:, :prompt_len]

        step_reward_sum = float(sum(step_rewards[:t]))
        eval_reward = 0.0
        if self.env and hasattr(self.env, "evaluate"):
            eval_reward = float(self.env.evaluate())
        total_reward = eval_reward + step_reward_sum

        if self.storage:
            self.storage.finish_trajectory(total_reward)

        if gen_output is not None:
            gen_output.non_tensor_batch["env_reward"] = np.array([total_reward], dtype=np.float32)
            gen_output.non_tensor_batch["eval_reward"] = np.array([eval_reward], dtype=np.float32)
            gen_output.non_tensor_batch["step_reward_sum"] = np.array([step_reward_sum], dtype=np.float32)
        
        return gen_output

    # ================================================================
    # Initialisation helpers
    # ================================================================

    def _initialize_tools(self):
        if self.tool_config_path is None:
            return []
        return initialize_tools_from_config(self.tool_config_path)
    
    def _initialize_env(self):
        if self.env_config_path is None:
            return None
        return initialize_env_from_config(self.env_config_path)
