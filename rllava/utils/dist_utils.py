import os, torch
import torch.distributed as dist
from typing import Any, Callable, Dict, Tuple, List, Optional
from rllava.data.protocol import DataProto, pad_dataproto_to_divisor, all_gather_data_proto

  

def init_dist() -> None:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # Initialize torch.distributed when launched via torchrun so DistributedSampler activates
    if not dist.is_available() or dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def dist_barrier():
    """Synchronize all ranks. No-op when not distributed."""
    if _is_dist():
        dist.barrier()

def _accelerate_rank_world() -> Tuple[int, int] | None:
    try:
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        # When accelerate is active, these reflect the global process topology
        if getattr(state, "num_processes", 1) > 1:
            return int(state.process_index), int(state.num_processes)
    except Exception:
        pass
    return None

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def _rank() -> int:
    accel = _accelerate_rank_world()
    if accel is not None:
        return accel[0]
    return dist.get_rank() if _is_dist() else 0

def _world_size() -> int:
    accel = _accelerate_rank_world()
    if accel is not None:
        return accel[1]
    return dist.get_world_size() if _is_dist() else 1

def _all_reduce_mean_scalar(value: float) -> float:
    if not _is_dist():
        return value
    tensor = torch.tensor([float(value)], device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / _world_size()).item()

def _reduce_metrics_mean(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_dist():
        return metrics
    reduced: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            reduced[key] = _all_reduce_mean_scalar(float(value))
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            tensor = value.to(torch.cuda.current_device(), dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = (tensor / _world_size()).item()
        else:
            # Non-scalar metrics are passed through; customize here if needed (e.g., gather/concat)
            reduced[key] = value
    return reduced

def _dataproto_chunk(data_proto, chunks: int):
    # DataProto has chunk(chunks) -> list[DataProto]
    return data_proto.chunk(chunks)

def _dataproto_all_gather_inplace(data_proto):
    # Lazy import to avoid circular imports
    from rllava.data.protocol import all_gather_data_proto

    if not _is_dist():
        return data_proto

    group = dist.group.WORLD
    all_gather_data_proto(data_proto, size=_world_size(), group=group)
    return data_proto


# Decorators
def dist_distribute_only():
    """
    Each rank processes its local shard. No extra gather/reduce.
    """
    def decorator(func: Callable):
        def wrapped(self, data, *args, **kwargs):
            return func(self, data, *args, **kwargs)
        return wrapped
    return decorator

def dist_gather_then_scatter():
    """
    Gather full DataProto to all ranks, run func on global batch, then split evenly and return local shard.
    """
    def decorator(func: Callable):
        def wrapped(self, data, *args, **kwargs):
            if not _is_dist():
                return func(self, data, *args, **kwargs)
            # 1) Make data global (in-place)
            _dataproto_all_gather_inplace(data)
            # 2) Compute on full batch
            new_data = func(self, data, *args, **kwargs)
            # 3) Split evenly and return local shard
            parts = _dataproto_chunk(new_data, _world_size())
            local = parts[_rank()]
            return local
        return wrapped
    return decorator

def dist_distribute_then_reduce():
    """
    Each rank processes its local shard; metrics are reduced by mean across ranks.
    Supports returning dict metrics or (metrics, other_output).
    """
    def decorator(func: Callable):
        def wrapped(self, data, *args, **kwargs):
            result = func(self, data, *args, **kwargs)
            if isinstance(result, tuple):
                metrics, other = result
                if isinstance(metrics, dict):
                    metrics = _reduce_metrics_mean(metrics)
                return metrics, other
            else:
                if isinstance(result, dict):
                    return _reduce_metrics_mean(result)
                return result
        return wrapped
    return decorator

def dist_rank0(broadcast_result: bool = False, return_for_others: Any = None, barrier: bool = False):
    """
    Run the wrapped function only on rank 0.
    - If broadcast_result=True, the result from rank 0 will be broadcast to all ranks
      using torch.distributed.broadcast_object_list, and each rank returns the same value.
    - Otherwise, non-zero ranks return `return_for_others`.
    - If barrier=True, call dist.barrier() after execution/broadcast for synchronization.
    """
    def decorator(func: Callable):
        def wrapped(*args, **kwargs):
            # Not distributed -> treat as rank 0
            if not _is_dist() or _rank() == 0:
                result = func(*args, **kwargs)
                if _is_dist() and broadcast_result:
                    obj_list = [result]
                    dist.broadcast_object_list(obj_list, src=0)
                    if barrier:
                        dist.barrier()
                    return obj_list[0]
                if _is_dist() and barrier:
                    dist.barrier()
                return result
            else:
                if _is_dist() and broadcast_result:
                    obj_list = [None]
                    dist.broadcast_object_list(obj_list, src=0)
                    if barrier:
                        dist.barrier()
                    return obj_list[0]
                if _is_dist() and barrier:
                    dist.barrier()
                return return_for_others
        return wrapped
    return decorator

def dist_batch(iterator) -> DataProto:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shards = [None] * world_size
    if is_rank0():
        try:
            batch_dict = next(iterator)
        except StopIteration:
            # Broadcast termination sentinel to all ranks to avoid hang
            dist.broadcast_object_list(shards, src=0)
            return None
        batch = DataProto.from_single_dict(batch_dict)
        batch, _ = pad_dataproto_to_divisor(batch, world_size)
        parts = batch.chunk(world_size)
        for i in range(world_size):
            shards[i] = parts[i]
    dist.broadcast_object_list(shards, src=0)
    batch = shards[rank]
    return batch
    
def gather_batch(batch) -> DataProto:
    if batch is None:
        return None
    world_size = dist.get_world_size()

    local_len = torch.tensor([len(batch)], device=torch.cuda.current_device(), dtype=torch.int64)
    len_list = [torch.empty_like(local_len) for _ in range(world_size)]
    dist.all_gather(len_list, local_len)
    lengths = [int(v.item()) for v in len_list]

    if len(set(lengths)) != 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, batch, group=dist.group.WORLD)
        merged = DataProto.concat(gathered)
        if torch.cuda.is_available():
            merged = merged.to(torch.device("cuda", torch.cuda.current_device()))
        return merged

    all_gather_data_proto(batch, size=world_size, group=dist.group.WORLD)
    return batch

def broadcast_object(
    obj: Optional[Any],
    src: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    group_src: Optional[int] = None
) -> Any:

    object_list = [obj]
    dist.broadcast_object_list(
        object_list,
        src=src,
        group=_unwrap_process_group(process_group),
        group_src=group_src
    )
    return object_list[0]

def gather_and_concat_list(
    lst: List[Any], process_group: dist.ProcessGroup
) -> Optional[List[Any]]:

    lists = (
        dist.get_world_size(process_group) * [None]
        if dist.get_rank(process_group) == 0
        else None
    )
    dist.gather_object(
        lst,
        lists,
        group=_unwrap_process_group(process_group),
        group_dst=0
    )
    return (
        [item for lst in lists for item in lst]
        if dist.get_rank(process_group) == 0
        else None
    )

def _unwrap_process_group(
    process_group: dist.ProcessGroup
) -> dist.ProcessGroup:

    if hasattr(process_group, "group"):
        return process_group.group
    elif hasattr(process_group, "get_group"):
        return process_group.get_group()
    else:
        return process_group