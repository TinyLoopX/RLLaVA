import torch.distributed as dist
from contextlib import contextmanager


class TrainEngine:
    """Base class for all training engines.

    This class defines the common interface that all training engines must implement.
    Engines are responsible for managing the training process for PPO.
    """

    def __init__(self, config, *, layout):
        self.config = config
        self.layout = layout
        self.device_mesh = None

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    @property
    def rank(self) -> int:
        return self.layout.rank

    @property
    def num_processes(self) -> int:
        return self.layout.dp_size

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def wait_for_everyone(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def prepare(self, *args, **kwargs):
        raise NotImplementedError("prepare is not implemented")

    def unwrap_model(self, model):
        raise NotImplementedError("unwrap_model is not implemented")

    def unwrap_model_for_generation(self, model, is_peft_model):
        raise NotImplementedError("unwrap_model_for_generation is not implemented")

    def load_state(self, model, optimizer, lr_scheduler, checkpoint_path):
        raise NotImplementedError("load_state is not implemented")

    def save_state(self, model, optimizer, lr_scheduler, checkpoint_path):
        raise NotImplementedError("save_state is not implemented")

    def backward(self, loss, is_last_step: bool = True):
        raise NotImplementedError("backward is not implemented")

    def optimizer_step(self):
        raise NotImplementedError("optimizer_step is not implemented")

    def clip_grad_norm_(self, model, max_norm):
        raise NotImplementedError("clip_grad_norm_ is not implemented")

    @contextmanager
    def eval(self, model):
        pass

    @contextmanager
    def train(self, model, optimizer):
        pass

