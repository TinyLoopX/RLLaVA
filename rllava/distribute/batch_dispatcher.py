from __future__ import annotations

from copy import copy

import torch.distributed as dist

from .dist_utils import on_coordinator


class BatchDispatcher:
    """Scatter batches to role input ranks and gather results on the coordinator."""

    def __init__(self, resources):
        self.resources = resources

    @property
    def is_coordinator(self):
        return self.resources.rank == self.resources.coordinator_rank

    def scatter(self, batch, role="actor"):
        layout = self.resources.roles[role]
        def split():
            shards = [None] * self.resources.world_size
            size, remainder = divmod(len(batch), layout.dp_size)
            start = 0
            for i, rank in enumerate(layout.input_ranks):
                stop = start + size + (i < remainder)
                shard = batch[start:stop]
                shard.meta_info = dict(batch.meta_info)
                shards[rank] = shard.to("cpu")
                start = stop
            return shards

        shards = on_coordinator(split, src=self.resources.coordinator_rank, group=self.resources.control_group)
        if self.resources.world_size == 1:
            return shards[0]
        received = [None]
        dist.scatter_object_list(received, shards, src=self.resources.coordinator_rank, group=self.resources.control_group)
        return received[0]

    def gather(self, batch, role="actor"):
        layout = self.resources.roles[role]
        result = copy(batch).to("cpu") if self.resources.rank in layout.output_ranks and batch is not None else None
        if self.resources.world_size == 1:
            return result
        results = [None] * self.resources.world_size if self.is_coordinator else None
        dist.gather_object(result, results, dst=self.resources.coordinator_rank, group=self.resources.control_group)
        if self.is_coordinator:
            results = [results[rank] for rank in layout.output_ranks if results[rank] is not None]
            return type(results[0]).concat(results) if results else None
