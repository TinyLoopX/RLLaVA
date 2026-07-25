"""``BaseTool`` wrapper around the OpenCV code-execution logic of
``examples/tasks/agent_code/agent_code_env.py``.

This mirrors how ``VisualCropTool`` wraps the DeepEyes zoom-in logic so that
both pipelines (multi-turn workflow via ``BaseEnv`` and standalone tool
invocation via ``BaseTool``) share *one* implementation.

The execution semantics intentionally match
``rllava/eval/agent_code/eval_mat_coding.py``:

* parse the last ``<code>...```python ... ``` ...</code>`` block
* substitute ``path_to_input_image.jpg`` / ``path_to_output_image.jpg`` with
  this tool's managed input/output paths
* for ``<problem>{'crop'}</problem>`` rewrite ``[y1:y2, x1:x2]`` slicing to use
  the bbox embedded in the question (1000-normalised by default)
* ``exec`` the code, then re-load the saved output image via PIL
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from .base import BaseTool

logger = logging.getLogger(__name__)


# Reuse regexes/helpers from the env to avoid duplication.
from importlib import import_module  # noqa: E402

try:
    _env_mod = import_module(
        "examples.tasks.agent_code.agent_code_env"
    )
except Exception:  # pragma: no cover - fallback when running from repo root
    import importlib.util
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.abspath(
        os.path.join(_here, "..", "..", "..", "examples", "tasks", "agent_code", "agent_code_env.py")
    )
    spec = importlib.util.spec_from_file_location("_agent_code_env_for_tool", _env_path)
    _env_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _env_mod
    spec.loader.exec_module(_env_mod)


CODE_BLOCK_PATTERN = _env_mod.CODE_BLOCK_PATTERN
ANSWER_BLOCK_PATTERN = _env_mod.ANSWER_BLOCK_PATTERN
PROBLEM_BLOCK_PATTERN = _env_mod.PROBLEM_BLOCK_PATTERN
BBOX_COORD_PATTERN = _env_mod.BBOX_COORD_PATTERN
SLICE_REWRITE_PATTERN = _env_mod.SLICE_REWRITE_PATTERN
_extract_problems = _env_mod._extract_problems


class CodeExecuteTool(BaseTool):
    """OpenCV code-execution tool for MAT-Coding-style trajectories."""

    DEFAULT_OBS_TEXT = _env_mod.AgentCodeEnv.DEFAULT_OBS_TEXT

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bbox_norm: int = int(config.get("bbox_norm", 1000))
        self.observation_text: str = config.get("observation_text", self.DEFAULT_OBS_TEXT)
        cache_root = config.get("cache_dir") or os.path.join(
            tempfile.gettempdir(), "code_execute_tool"
        )
        os.makedirs(cache_root, exist_ok=True)
        self._cache_dir = cache_root

        self._image: Optional[Image.Image] = None
        self._question: str = ""
        self._uid: str = uuid.uuid4().hex[:12]
        self._input_path = os.path.join(self._cache_dir, f"{self._uid}_in.jpg")
        self._output_path = os.path.join(self._cache_dir, f"{self._uid}_out.jpg")

    # ------------------------------------------------------------------
    # Setters – called by the workflow before the loop starts
    # ------------------------------------------------------------------

    def set_image(self, image: Image.Image):
        self._image = image.convert("RGB")
        self._image.save(self._input_path, format="JPEG", quality=95)

    def set_question(self, question: str):
        self._question = question or ""

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------

    def extract_tool_call(self, content: str) -> Optional[str]:
        """Return the last ``<code>`` block, or *None* if the model has
        already produced a final ``<answer>``."""
        if ANSWER_BLOCK_PATTERN.search(content):
            return None
        m = CODE_BLOCK_PATTERN.search(content)
        return m.group(0) if m else None

    def execute(self, tool_content: str) -> Tuple[Any, bool]:
        """Run the code block and return (obs, success_flag)."""
        if self._image is None:
            return "Error: no input image set on tool.", False

        # Replace the well-known placeholders with real disk paths.
        full_response = tool_content or ""
        result = full_response.replace(
            "path_to_input_image.jpg", self._input_path
        ).replace(
            "path_to_output_image.jpg", self._output_path
        )

        labels = _extract_problems(full_response)
        if labels and labels[0] == "crop":
            result = self._rewrite_crop_slicing(result)

        m = CODE_BLOCK_PATTERN.search(result)
        if not m:
            return "Error: <code>```python ... ```</code> block not found.", False

        try:
            local_vars: Dict[str, Any] = {}
            exec(m.group(1), globals(), local_vars)
        except Exception as exc:
            return f"Error: code execution raised {type(exc).__name__}: {exc}", False

        if not os.path.exists(self._output_path):
            return _env_mod.AgentCodeEnv.DEFAULT_OBS_NOOP_TEXT, True

        try:
            new_image = Image.open(self._output_path)
            new_image.load()
            new_image = new_image.convert("RGB")
        except Exception as exc:
            return f"Error: failed to read output image – {exc}", False

        self._image = new_image
        new_image.save(self._input_path, format="JPEG", quality=95)
        return {"image": new_image, "text": self.observation_text}, True

    def release(self):
        self._image = None
        for path in (self._input_path, self._output_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rewrite_crop_slicing(self, code_text: str) -> str:
        if not self._question or self._image is None:
            return code_text
        coords = BBOX_COORD_PATTERN.findall(self._question)
        if not coords:
            return code_text
        bbox = list(map(int, coords[0]))
        width, height = self._image.size
        x_min = int(bbox[0] / self.bbox_norm * width)
        y_min = int(bbox[1] / self.bbox_norm * height)
        x_max = int(bbox[2] / self.bbox_norm * width)
        y_max = int(bbox[3] / self.bbox_norm * height)
        replacement = f"[{y_min}:{y_max}, {x_min}:{x_max}]"
        return SLICE_REWRITE_PATTERN.sub(replacement, code_text)
