"""DeepEyes visual environment.

Implements the ``BaseEnv`` interface for DeepEyes-style visual agentic tasks:
the model can call ``image_zoom_in_tool`` (and optionally ``image_rotate_tool``)
to inspect regions of the image, then provide a final ``<answer>`` when ready.

The env manages the stateful image across turns, parses ``<tool_call>`` / ``<answer>``
from model output, and returns cropped / rotated images as observations.
"""

import json
import logging
import re
from io import BytesIO
from math import ceil, floor
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from rllava.ppo.env.base import BaseEnv

logger = logging.getLogger(__name__)


class DeepEyesEnv(BaseEnv):
    """Visual zoom-in / rotate environment ported from the DeepEyes project.

    Config keys (all optional):
        min_crop_side (int): minimum side length after crop, default 28.
        enable_rotate (bool): support ``image_rotate_tool``, default False.
    """

    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    OBSERVATION_TEXT = (
        "Think in the mind first, and then decide whether to call tools one or more times "
        "OR provide final answer. Format strictly as: "
        "<think>...</think> <tool_call>...</tool_call> (if any tools needed) "
        "OR <answer>...</answer> (if no tools needed)."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_crop_side: int = config.get("min_crop_side", 28)
        self.enable_rotate: bool = config.get("enable_rotate", False)
        self._image: Optional[Image.Image] = None
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self, data: Optional[Any] = None) -> Any:
        """Extract the original image from *data* and store it for cropping."""
        self._image = None
        self._width = 0
        self._height = 0

        if data is None:
            return None

        multi_modal_data = data.non_tensor_batch.get("multi_modal_data", [{}])
        mm = multi_modal_data[0] if multi_modal_data else {}
        if mm is None:
            mm = {}
        images = mm.get("image") or mm.get("images") or []
        if images:
            img = images[0]
            if isinstance(img, dict):
                img = Image.open(BytesIO(img["bytes"]))
            elif isinstance(img, bytes):
                img = Image.open(BytesIO(img))
            elif isinstance(img, str):
                img = Image.open(img)
            if hasattr(img, "load"):
                img.load()
            self._image = img.copy()
            self._width = self._image.width
            self._height = self._image.height

        return None

    def extract_action(self, content: str) -> Optional[str]:
        """Return the last ``<tool_call>`` body, or *None* if model output
        contains ``<answer>`` (episode done) or no tool call."""
        if self.ANSWER_PATTERN.search(content):
            return None
        matches = self.TOOL_CALL_PATTERN.findall(content)
        return matches[-1] if matches else None

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """Execute a tool call parsed from model output.

        Returns:
            obs: ``{"image": PIL.Image, "text": str}`` on success,
                 or a plain error string on failure.
            reward: Always 0.0 (final reward comes from reward function).
            done: False (the loop continues; ``<answer>`` stops it via
                  ``extract_action`` returning None).
            info: Metadata dict.
        """
        try:
            tool_call = json.loads(action.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            return f"Error: invalid JSON – {exc}", 0.0, False, {"status": "failed"}

        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        try:
            if tool_name == "image_zoom_in_tool":
                return self._zoom_in(args)
            elif tool_name == "image_rotate_tool" and self.enable_rotate:
                return self._rotate(args)
            else:
                return f"Error: unknown tool '{tool_name}'", 0.0, False, {"status": "failed"}
        except Exception as exc:
            logger.warning("DeepEyesEnv.step failed: %s", exc)
            return f"Error: {exc}", 0.0, False, {"status": "failed"}

    def close(self):
        self._image = None

    # ------------------------------------------------------------------
    # Private: tool implementations
    # ------------------------------------------------------------------

    def _zoom_in(self, args: dict) -> Tuple[Any, float, bool, Dict]:
        bbox = args.get("bbox_2d") or args.get("bbox")
        if not bbox or len(bbox) != 4:
            return "Error: bbox_2d must be [x1, y1, x2, y2]", 0.0, False, {"status": "failed"}

        bbox = self._maybe_resize_bbox(*bbox)
        if bbox is None:
            return "Error: invalid or too-small bounding box", 0.0, False, {"status": "failed"}

        cropped = self._image.crop(bbox)
        self._image = cropped
        self._width = cropped.width
        self._height = cropped.height

        obs = {"image": cropped, "text": self.OBSERVATION_TEXT}
        return obs, 1.0, False, {"status": "success", "tool": "image_zoom_in_tool"}

    def _rotate(self, args: dict) -> Tuple[Any, float, bool, Dict]:
        angle = args.get("angle", 0)
        rotated = self._image.rotate(angle, expand=True)
        self._image = rotated
        self._width = rotated.width
        self._height = rotated.height

        obs = {"image": rotated, "text": self.OBSERVATION_TEXT}
        return obs, 1.0, False, {"status": "success", "tool": "image_rotate_tool"}

    # ------------------------------------------------------------------
    # BBox validation & auto-resize (ported from DeepEyes)
    # ------------------------------------------------------------------

    def _validate_bbox(self, left, top, right, bottom) -> bool:
        if left >= right or top >= bottom:
            return False
        h, w = bottom - top, right - left
        if max(h, w) / max(min(h, w), 1) > 100:
            return False
        if min(h, w) <= self.min_crop_side:
            return False
        return True

    def _maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(self._width, int(right))
        bottom = min(self._height, int(bottom))

        if not self._validate_bbox(left, top, right, bottom):
            return None

        h, w = bottom - top, right - left
        if h < self.min_crop_side or w < self.min_crop_side:
            cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
            ratio = self.min_crop_side / min(h, w)
            half_h = ceil(h * ratio * 0.5)
            half_w = ceil(w * ratio * 0.5)
            left, right = floor(cx - half_w), ceil(cx + half_w)
            top, bottom = floor(cy - half_h), ceil(cy + half_h)
            if not self._validate_bbox(left, top, right, bottom):
                return None

        return [left, top, right, bottom]
