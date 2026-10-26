"""Process initialization and communication primitives, independent of batches."""

import os
from functools import wraps

import torch
import torch.distributed as dist


def init_dist():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    if not dist.is_available() or dist.is_initialized():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def dist_barrier(group=None):
    if dist.is_initialized():
        dist.barrier(group=group)


def broadcast_object(obj, src=0, group=None):
    if not dist.is_initialized():
        return obj
    objects = [obj]
    dist.broadcast_object_list(objects, src=src, group=group)
    return objects[0]


class CoordinatorError(RuntimeError):
    """The coordinator failed before entering the next collective."""


def on_coordinator(func, src=0, group=None):
    """Run Python work once, notifying peers before the next collective."""
    result, error = None, None
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == src:
        try:
            result = func()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = broadcast_object(error, src=src, group=group)
    if error:
        raise CoordinatorError(error)
    return result


def dist_rank0(broadcast_result=False, return_for_others=None, barrier=False):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs) if is_rank0() else return_for_others
            if broadcast_result:
                result = broadcast_object(result)
            if barrier and dist.is_initialized():
                dist.barrier()
            return result
        return wrapped
    return decorator
