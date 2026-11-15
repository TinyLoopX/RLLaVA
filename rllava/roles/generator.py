"""Inference role; rollout orchestration owns dispatch, workflows and rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllava.data.protocol import DataProto
    from rllava.distribute.resource_manager import RoleLayout
    from rllava.rollout.config import RolloutConfig


class Generator:
    def __init__(self, config: RolloutConfig, tokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.engine = None
        self.generation_config = None

    def initialize(self, model_path: str, layout: RoleLayout):
        from transformers import GenerationConfig
        from rllava.engine import EngineFactory
        from rllava.utils.model_utils import print_gpu_memory_usage

        self.generation_config = GenerationConfig.from_pretrained(model_path)
        print_gpu_memory_usage(f"Before {self.config.name} rollout engine init")
        self.engine = EngineFactory(self.config.name)(
            model_name_or_path=model_path,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            layout=layout,
        )
        print_gpu_memory_usage(f"After {self.config.name} rollout engine init")

    @property
    def collective_generate(self):
        """Whether every rank of an inference replica must enter generation."""
        return self.engine.collective_generate

    def prepare_prompts(self, prompts: DataProto):
        """Attach generation token IDs before rollout distributes requests."""
        config = self.generation_config
        prompts.meta_info.update({
            "eos_token_id": config.eos_token_id if config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": config.pad_token_id if config is not None else self.tokenizer.pad_token_id,
        })
        return prompts

    def prepare(self, prompts: DataProto):
        # Separate preparation from generation so rollout can agree on errors
        # across ranks before entering backend generation collectives.
        return self.engine.prepare(prompts)

    def generate(self, prepared) -> DataProto:
        """Return backend output, including sampled-token log probabilities."""
        return self.engine.generate(prepared)

    def load(self, model):
        """Load inference resources and synchronize the supplied policy weights."""
        self.engine.load(model)

    def offload(self):
        self.engine.offload()

    def close(self):
        if self.engine is not None:
            self.engine.close()
