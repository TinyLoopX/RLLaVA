import os
import importlib
import sys
import torch
from rllava.data.protocol import DataProto
from collections import defaultdict
from typing import Tuple
from ..config import RewardConfig
from transformers import PreTrainedTokenizer, AutoConfig
from functools import partial
from typing import Callable, TypedDict, Optional



class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]
BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]

class Reward:
    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = None
        
        # "env" mode: rewards come from environment's evaluate(), no reward_function needed
        if config.reward_type == "env":
            print("Using environment-based reward (reward_type='env'). Rewards will come from env.evaluate().")
            return
        
        # For "sequential" and "batch" modes, reward_function is required
        if config.reward_function is None or config.reward_function == "":
            raise ValueError("Reward function is not provided. Set reward.reward_function in config.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)


    def initialize(self):
        if self.config.model.model_path is not None:
            # TODO: init reward model
            pass

    def compute_rewards(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        
        # "env" mode: use pre-computed rewards from environment
        if self.config.reward_type == "env":
            if "env_reward" in data.non_tensor_batch:
                env_rewards = data.non_tensor_batch["env_reward"]
                for i in range(len(data)):
                    cur_response_length = int(response_length[i].item())
                    env_reward = float(env_rewards[i]) if i < len(env_rewards) else 0.0
                    reward_tensor[i, cur_response_length - 1] = env_reward
                    reward_metrics["overall"].append(env_reward)
                    reward_metrics["env_reward"].append(env_reward)
            else:
                # No env_reward available, return zeros
                for i in range(len(data)):
                    reward_metrics["overall"].append(0.0)
                    reward_metrics["env_reward"].append(0.0)
            return reward_tensor, reward_metrics
        
        # "sequential" and "batch" modes: use reward_function
        reward_inputs = []
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ri = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            for key in data.non_tensor_batch:
                if key not in ri:
                    ri[key] = data.non_tensor_batch[key][i]
            reward_inputs.append(ri)

        if self.config.reward_type == "sequential":
            for i, inputs in enumerate(reward_inputs):
                cur_response_length = int(response_length[i].item())
                score = self.reward_fn(inputs)
                reward_tensor[i, cur_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
        elif self.config.reward_type == "batch":
            scores = self.reward_fn(reward_inputs)
            for i, score in enumerate(scores):
                cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
                reward_tensor[i, cur_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

        return reward_tensor, reward_metrics

