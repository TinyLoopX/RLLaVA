from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch.distributed as dist


@dataclass
class ParallelConfig:
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    sp_mode: str = "none"


@dataclass
class RoleLayout:
    name: str
    parallel: ParallelConfig
    world_size: int
    rank: int
    devices: tuple[str, ...]
    groups: dict = field(default_factory=dict)

    @property
    def model_parallel_size(self):
        p = self.parallel
        return p.tp_size * p.pp_size * p.cp_size

    @property
    def dp_size(self):
        return self.world_size // self.model_parallel_size

    @property
    def dp_rank(self):
        return self.rank // self.model_parallel_size

    @property
    def tp_rank(self):
        return self.rank % self.parallel.tp_size

    @property
    def input_ranks(self):
        return tuple(range(0, self.world_size, self.model_parallel_size))

    @property
    def output_ranks(self):
        p = self.parallel
        return tuple(r + (p.pp_size - 1) * p.cp_size * p.tp_size for r in self.input_ranks)

    @property
    def input_rank(self):
        return self.input_ranks[self.dp_rank]

    @property
    def replica_devices(self):
        start = self.input_rank
        return self.devices[start:start + self.model_parallel_size]


def normalize_parallel_config(config):
    """Normalize one explicit config layer, before defaults and CLI merging."""
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(config, resolve=False)
    for name in ("actor", "critic", "ref", "rollout"):
        role = config.get(name)
        if not role:
            continue
        parallel = role.setdefault("parallel", {})

        def assign(key, value):
            if key in parallel and parallel[key] != value:
                raise ValueError(f"Conflicting {name}.parallel.{key} settings in the same config layer.")
            parallel[key] = value

        if name == "rollout":
            for owner, key, target in (
                (role, "tensor_parallel_size", "tp_size"),
                (role.get("sglang", {}), "tp_size", "tp_size"),
                (role.get("vllm", {}), "pipeline_parallel_size", "pp_size"),
            ):
                if key in owner:
                    assign(target, owner.pop(key))
            if "gpus_per_replica" in role:
                size = role.pop("gpus_per_replica")
                if parallel:
                    expected = parallel.get("tp_size", 1) * parallel.get("pp_size", 1) * parallel.get("cp_size", 1)
                    if size != expected:
                        raise ValueError("rollout.gpus_per_replica differs from the explicit parallel layout.")
                else:
                    parallel["tp_size"] = size
        else:
            for key in ("model", "fsdp"):
                size = role.get(key, {}).get("ulysses_size", 1)
                if size > 1:
                    assign("cp_size", size)
                    assign("sp_mode", "ulysses")
    return OmegaConf.create(config)


class ResourceManager:
    """Own role placement and control groups; engines own their compute groups."""

    def __init__(self, layouts):
        self.roles = {layout.name: layout for layout in layouts}
        first = layouts[0]
        self.world_size, self.rank = first.world_size, first.rank
        self.coordinator_rank = 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.control_group = None
        self._owned_groups = []

    @classmethod
    def from_config(cls, config):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        devices = tuple(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        if devices == ("",):
            devices = tuple(str(i) for i in range(world_size))
        roles = {"actor": config.actor, "ref": config.actor, "rollout": config.rollout}
        if config.critic is not None:
            roles["critic"] = config.critic
        layouts = []
        for name, role in roles.items():
            p = config.ref.parallel if name == "ref" and config.ref is not None else role.parallel
            sizes = (p.tp_size, p.pp_size, p.cp_size, p.ep_size)
            if any(type(size) is not int or size < 1 for size in sizes) or world_size % (p.tp_size * p.pp_size * p.cp_size):
                raise ValueError(f"{name}: parallel sizes must be positive integers and TP×PP×CP must divide {world_size} GPUs.")
            supported = p.pp_size == p.cp_size == p.ep_size == 1 and p.sp_mode == "none"
            if name != "rollout":
                supported &= p.tp_size == 1 and role.model.ulysses_size == role.fsdp.ulysses_size == 1
            elif config.rollout.name == "hf":
                supported &= p.tp_size == 1
            if not supported:
                raise NotImplementedError(f"{name}: this backend currently supports {'DP/TP inference' if name == 'rollout' and config.rollout.name != 'hf' else 'DP only'}; requested {p}.")
            layouts.append(RoleLayout(name, p, world_size, rank, devices))
        manager = cls(layouts)
        manager._check_batches(config)
        manager._create_groups()
        return manager

    def _check_batches(self, config):
        responses = config.data.train_batch_size * config.rollout.n
        for name, role, multiplier in (("actor", config.actor, config.rollout.n), ("critic", config.critic, 1)):
            if role is None:
                continue
            mini = role.ppo_mini_batch_size * multiplier
            micro = role.ppo_micro_batch_size * self.roles[name].dp_size
            if mini <= 0 or micro <= 0 or mini % micro or responses % mini:
                raise ValueError(f"{name}: global response batch ({responses}) must divide into mini-batches ({mini}), each divisible by DP × micro-batch ({micro}).")

    def _create_groups(self):
        if not dist.is_initialized():
            return
        if dist.get_backend() == "gloo":
            self.control_group = dist.group.WORLD
        else:
            self.control_group = dist.new_group(backend="gloo")
            self._owned_groups.append(self.control_group)
        for layout in self.roles.values():
            layout.groups["control"] = self.control_group
            if layout.name != "rollout":
                layout.groups["data"] = dist.group.WORLD
        rollout = self.roles["rollout"]
        for start in rollout.input_ranks:
            ranks = list(range(start, start + rollout.model_parallel_size))
            if len(ranks) == self.world_size:
                group = self.control_group
            else:
                group = dist.new_group(ranks=ranks, backend="gloo")
                if self.rank in ranks:
                    self._owned_groups.append(group)
            if self.rank in ranks:
                rollout.groups["replica"] = group

    def close(self):
        for group in reversed(self._owned_groups):
            dist.destroy_process_group(group)
        self._owned_groups.clear()
