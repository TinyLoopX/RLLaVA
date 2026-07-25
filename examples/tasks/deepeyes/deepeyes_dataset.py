import re
from typing import Dict, Any, List

from rllava.data.dataset import RLHFDataset


class DeepEyesDataset(RLHFDataset):
    """Dataset for DeepEyes-style visual agentic tasks.

    Handles two DeepEyes-specific data conventions:

    1. Pre-formatted messages with ``<image>`` as plain string in content
       need to be converted to structured ``[{"type": "image"}, ...]`` format
       for Qwen2-VL / Qwen3-VL chat templates.

    2. ``reward_model`` column is a dict ``{"style": "rule", "ground_truth": "..."}``.
       Extract ``ground_truth`` as a string for downstream reward computation.
    """

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = example[self.prompt_key]

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            has_media = (self.image_key in example and example[self.image_key] is not None) or \
                        (self.video_key in example and example[self.video_key] is not None)
            if has_media:
                for message in prompt:
                    content = message.get("content", "")
                    if isinstance(content, str) and ("<image>" in content or "<video>" in content):
                        content_list = []
                        for segment in re.split(r"(<image>|<video>)", content):
                            if segment == "<image>":
                                content_list.append({"type": "image"})
                            elif segment == "<video>":
                                content_list.append({"type": "video"})
                            elif segment:
                                content_list.append({"type": "text", "text": segment})
                        message["content"] = content_list
            return prompt

        return super()._build_messages(example)

    def __getitem__(self, index):
        # Capture original structured messages before the parent tokenizes them.
        # self.dataset[index] returns a fresh dict from the parquet each time,
        # so _build_messages here does not interfere with the parent call below.
        raw_example = dict(self.dataset[index])
        messages = self._build_messages(raw_example)

        example = super().__getitem__(index)

        # Store messages for multi-turn workflow (conversation_history init).
        # The dataset's system message already embeds tool-call instructions
        # (SYSTEM_PROMPT_V2/V5 format), so no extra injection is needed.
        example["messages"] = messages

        gt = example.get("ground_truth")
        if isinstance(gt, dict):
            example["ground_truth"] = gt.get(
                "ground_truth", gt.get("answer", str(gt))
            )

        return example
