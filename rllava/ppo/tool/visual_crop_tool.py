import re
import json
import logging
from math import ceil, floor
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from .base import BaseTool

logger = logging.getLogger(__name__)


class VisualCropTool(BaseTool):
    """DeepEyes-style visual zoom-in / rotate tool.

    Parses ``<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1,y1,x2,y2]}} </tool_call>``
    from the model response, crops (or rotates) the current image, and returns
    the processed PIL image together with a textual observation so that
    ``MultiTurnWorkflow.add_tool_message`` can feed it back into the next turn.

    Config keys (all optional):
        min_crop_side (int): minimum side length of cropped region, default 28.
        enable_rotate (bool): whether to support image_rotate_tool, default False.
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
        self._original_image: Optional[Image.Image] = None
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # Public API required by BaseTool
    # ------------------------------------------------------------------

    def extract_tool_call(self, content: str) -> Optional[str]:
        """Return the last ``<tool_call>...</tool_call>`` body, or *None*."""
        if self.ANSWER_PATTERN.search(content):
            return None
        matches = self.TOOL_CALL_PATTERN.findall(content)
        return matches[-1] if matches else None

    def execute(self, tool_content: str) -> Tuple[Any, bool]:
        """Execute the parsed tool call.

        Returns
        -------
        result : dict | str
            On success a dict ``{"image": PIL.Image, "text": str}``;
            on failure an error string.
        success : bool
        """
        try:
            tool_call = json.loads(tool_content.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            return f"Error: invalid JSON in tool_call – {exc}", False

        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        try:
            if tool_name == "image_zoom_in_tool":
                return self._zoom_in(args)
            elif tool_name == "image_rotate_tool" and self.enable_rotate:
                return self._rotate(args)
            else:
                return f"Error: unknown tool '{tool_name}'", False
        except Exception as exc:
            logger.warning("VisualCropTool.execute failed: %s", exc)
            return f"Error: {exc}", False

    def release(self):
        self._original_image = None

    # ------------------------------------------------------------------
    # Image setter – called by MultiTurnWorkflow before the loop starts
    # ------------------------------------------------------------------

    def set_image(self, image: Image.Image):
        self._original_image = image.copy()
        self._width = image.width
        self._height = image.height

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _zoom_in(self, args: dict) -> Tuple[dict, bool]:
        bbox = args.get("bbox_2d") or args.get("bbox")
        if not bbox or len(bbox) != 4:
            return "Error: bbox_2d must be [x1, y1, x2, y2]", False

        bbox = self._maybe_resize_bbox(*bbox)
        if bbox is None:
            return "Error: invalid or too-small bounding box", False

        img = self._original_image
        cropped = img.crop(bbox)
        self._original_image = cropped
        self._width = cropped.width
        self._height = cropped.height
        return {"image": cropped, "text": self.OBSERVATION_TEXT}, True

    def _rotate(self, args: dict) -> Tuple[dict, bool]:
        angle = args.get("angle", 0)
        img = self._original_image
        rotated = img.rotate(angle, expand=True)
        self._original_image = rotated
        self._width = rotated.width
        self._height = rotated.height
        return {"image": rotated, "text": self.OBSERVATION_TEXT}, True

    # ------------------------------------------------------------------
    # BBox validation & auto-resize (ported from DeepEyes)
    # ------------------------------------------------------------------

    def _validate_bbox(self, left, top, right, bottom) -> bool:
        try:
            assert left < right and top < bottom, f"invalid shape: {left=}, {top=}, {right=}, {bottom=}"
            h, w = bottom - top, right - left
            assert max(h, w) / max(min(h, w), 1) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            assert min(h, w) > self.min_crop_side, f"{h=}, {w=} too small (min={self.min_crop_side})"
            return True
        except AssertionError:
            return False

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
