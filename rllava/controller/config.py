import math
from dataclasses import dataclass, field
from typing import Optional
from rllava.utils.config import BaseConfig
from rllava.roles.config import ActorConfig, CriticConfig, RefConfig, RewardConfig
from rllava.rollout.config import RolloutConfig
from rllava.train.config import AlgorithmConfig


@dataclass
class RLConfig(BaseConfig):
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: Optional[CriticConfig] = None
    ref: Optional[RefConfig] = None
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.rollout.prompt_length = self.data.max_prompt_length
        self.rollout.response_length = self.data.max_response_length
        self.rollout.trust_remote_code = self.actor.model.trust_remote_code
        self.rollout.adv_estimator = self.algorithm.adv_estimator
        self.rollout.min_pixels = self.data.min_pixels
        self.rollout.max_pixels = self.data.max_pixels
        self.rollout.video_fps = self.data.video_fps
        if not math.isfinite(self.actor.tis_imp_ratio_cap):
            raise ValueError("actor.tis_imp_ratio_cap must be finite.")
        if self.actor.tis_imp_ratio_cap > 0 and self.actor.policy_loss.loss_mode != "vanilla":
            raise ValueError("TIS is currently supported only by actor.policy_loss.loss_mode=vanilla.")
        if self.algorithm.prefix_strategy != "none" or self.algorithm.unify_strategy != "none":
            raise ValueError("Rollout log probabilities cannot yet be used with prefix/demo response rewriting.")
        # normalize rollout.seed: "none"/"null" -> None, numeric string -> int
        if isinstance(self.rollout.seed, str):
            if self.rollout.seed.lower() in ("none", "null"):
                self.rollout.seed = None
            else:
                self.rollout.seed = int(self.rollout.seed)
