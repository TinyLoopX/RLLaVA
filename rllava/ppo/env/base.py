from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple, Union


class BaseEnv:
    """Base class for environments. Uses standard RL interface.

    Subclasses implement task-specific logic (visual crop, code sandbox,
    browser, etc.) while ``MultiTurnWorkflow`` drives the generic loop.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.concurrent_workers = max(1, int(config.get("concurrent_workers", 1)))

    def reset(self, data: Optional[Any] = None) -> Any:
        """Reset environment for a new episode.

        Args:
            data: Optional sample-level data (e.g. ``DataProto`` for a single
                  sample) so the env can extract task-specific context such as
                  the original image, question text, etc.

        Returns:
            obs: Initial observation. Can be Dict (e.g. ``{"image": PIL.Image,
                 "text": str}``), a PIL Image, a string, or any format the
                 task needs.
        """
        raise NotImplementedError
    
    def extract_action(self, content: str) -> Any:
        """Extract action from model response. Return None if not match."""
        raise NotImplementedError
        
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """Execute action.
        
        Returns:
            obs: Observation – Dict with ``"image"`` and/or ``"text"`` keys,
                 a PIL Image, a plain string, etc.
            reward: Step-level reward (typically 0; final reward comes from
                    the reward function).
            done: Whether the episode is finished.
            info: Additional information.
        """
        raise NotImplementedError
    
    def close(self):
        """Release resources held by this env."""
        pass

    @staticmethod
    def _step_item(step_item):
        # step_item = (sample_idx, env, action, gen_output, content, response_ids)
        sample_idx, env, action, gen_output, content, response_ids = step_item
        obs, step_reward, env_done, info = env.step(action)
        return sample_idx, obs, step_reward, env_done, info, gen_output, content, response_ids

    def batch_step(self, step_items):
        """Execute multiple env steps, optionally in parallel.

        Each item in step_items is a tuple:
            (sample_idx, env, action, gen_output, content, response_ids)

        Returns a list of tuples:
            (sample_idx, obs, step_reward, env_done, info, gen_output, content, response_ids)
        """
        if not step_items:
            return []
        num_workers = min(self.concurrent_workers, len(step_items))
        if num_workers <= 1:
            return [self._step_item(item) for item in step_items]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(self._step_item, step_items))
