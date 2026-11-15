"""Configuration for synchronous, group-based experience handoff."""
from dataclasses import dataclass


@dataclass
class ExperienceConfig:
    max_groups: int = 5000
    experience_mode: str = "last_response"

    def __post_init__(self):
        self.post_init()

    def post_init(self):
        if type(self.max_groups) is not int or self.max_groups < 1:
            raise ValueError("max_groups must be a positive integer")
        if self.experience_mode != "last_response":
            raise ValueError("Only last_response experiences are supported")
