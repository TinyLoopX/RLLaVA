import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Any
from rllava.utils.dist_utils import dist_rank0, is_rank0
from rllava.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, remove_obsolete_ckpt, CHECKPOINT_TRACKER
from rllava.utils.logger import Tracking, ValidationGenerationsLogger
from rllava.utils.config import BaseConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Pipeline:

    def __init__(
        self,
        model,
        config: BaseConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        self.global_steps = 0
        # dataloader epoch counter for DistributedSampler.set_epoch
        self._dataloader_epoch = 0
                
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.model = model
        self.config = config

        self.training_steps = self.model.training_steps
        self.validation_generations_logger = ValidationGenerationsLogger()

        self.train_dataloader, self.val_dataloader = train_dataloader, val_dataloader
        
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )

        self.load_checkpoint()

    def iter_train_batches(self):
        """Re-enter the dataloader each epoch without caching its batches.

        The first traversal also respects a restored StatefulDataLoader cursor.
        Later traversals run the sampler and dataset processing again.
        """
        while True:
            yield from self.train_dataloader

    def save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_steps

        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_steps}")
        self.model.save_checkpoint(folder_path)
        
        if is_rank0():
            remove_obsolete_ckpt(
                self.config.trainer.save_checkpoint_path,
                self.global_steps,
                self.best_global_step,
                self.config.trainer.save_limit,
            )
    
            dataloader_path = os.path.join(folder_path, "dataloader.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_path)
    
            checkpointer_tracker_info = {
                "best_global_step": self.best_global_step,
                "best_val_reward_score": round(self.best_val_reward_score, 4),
                "last_global_step": self.global_steps,
                "last_actor_path": os.path.abspath(os.path.join(folder_path, "actor")),
            }
            checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
            with open(checkpointer_tracker_path, "w") as f:
                json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)
        
    def load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path = find_latest_ckpt_path(self.config.trainer.save_checkpoint_path)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_steps = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        self.model.load_checkpoint(load_checkpoint_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")
        
    @dist_rank0()
    def maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        # Note: labels are currently not logged by ValidationGenerationsLogger backends
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @dist_rank0()
    def validate(self) -> dict[str, Any]:
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
