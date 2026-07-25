"""Agent-Code visual environment.

Implements the ``BaseEnv`` interface for MAT-Coding-style image-processing
agent tasks: the model reasons step-by-step inside ``<think>`` tags and emits
one of three actions per turn — ``<problem>...`` (diagnose the image issue),
``<code>``...```python ... ``` ...``</code>`` (run an OpenCV snippet to fix
the image) or ``<answer>...</answer>`` (final answer).

The env reproduces the *exact* tool execution loop used in
``rllava/eval/agent_code/eval_mat_coding.py``:

* When the model returns ``<problem>``, the env decides which textual ``<tips>``
  hint to feed back (crop / none / generic).
* When the model returns ``<code>``, the env replaces
  ``path_to_input_image.jpg`` / ``path_to_output_image.jpg`` placeholders with
  the env-managed input/output paths, runs the python block via ``exec``, and
  returns the processed image as the next observation.  For ``crop`` problems
  it additionally rewrites the bbox slicing pattern using the bbox embedded in
  the question (matching the eval script).
* When the model returns ``<answer>``, ``extract_action`` returns ``None`` so
  ``MultiTurnWorkflow`` stops the rollout for this trajectory.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from rllava.ppo.env.base import BaseEnv

logger = logging.getLogger(__name__)

# cv2 internal threads can deadlock when many ThreadPoolExecutor workers exec
# OpenCV pipelines concurrently. Force single-threaded cv2 to keep things stable.
try:  # pragma: no cover - cv2 is a soft dep at module import time
    import cv2 as _cv2  # noqa: F401

    _cv2.setNumThreads(1)
except Exception:  # noqa: BLE001
    pass

# Serialise model-emitted ``exec`` blocks. Multiple sample envs share this
# process-wide lock because cv2 internals (and many third-party libs the model
# may import) are not guaranteed re-entrant across Python threads.
_EXEC_LOCK = threading.Lock()

# Minimum spatial dimension we accept for an output image. Anything below this
# is rejected so it never reaches the image processor (which would otherwise
# crash the whole training step).
_MIN_OUTPUT_DIM = 8

CODE_BLOCK_PATTERN = re.compile(r"<code>\s*```python(.*?)```.*?</code>", re.DOTALL)
PROBLEM_BLOCK_PATTERN = re.compile(r"<problem>\s*\{(.*?)\}\s*</problem>", re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BBOX_COORD_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
SLICE_REWRITE_PATTERN = re.compile(
    r"\[\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*:\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)"
    r"\s*,\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*:\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*\]"
)

# Placeholder names that the model is expected to use. We replace each of them
# with the actual env-managed file path. Order matters: handle the explicit
# quoted-with-leading-slash variants first so we never produce ``//<path>``.
_INPUT_PLACEHOLDERS = (
    "path_to_input_image.jpg",
    "path_to_input_image.jpeg",
    "path_to_input_image.png",
    "path_to_input.jpg",
    "input_image.jpg",
)
_OUTPUT_PLACEHOLDERS = (
    "path_to_output_image.jpg",
    "path_to_output_image.jpeg",
    "path_to_output_image.png",
    "path_to_output.jpg",
    "output_image.jpg",
)


def _extract_problems(text: str) -> list[str]:
    """Pull problem labels (e.g. ``'crop'``, ``'overexposure'``) out of a
    ``<problem>{...}</problem>`` block."""
    match = PROBLEM_BLOCK_PATTERN.search(text)
    if not match:
        return []
    return sorted(re.findall(r"'([^']+)'", match.group(1)))


class AgentCodeEnv(BaseEnv):
    """OpenCV code-execution env for MAT-Coding style RL training.

    Config keys (all optional):
        max_code_seconds (int): soft hint for code execution timeout, default 10.
        cache_dir (str): scratch dir for input/output image files,
            default ``$TMPDIR/agent_code_env``.
        tips_default (str): tip text for non-crop / non-none problems.
        tips_crop (str): tip text appended when the diagnosed problem is ``crop``.
        tips_none (str): tip text appended when the diagnosed problem is ``none``.
        bbox_norm (int): coordinate range used in the question's bbox (default
            1000, matching the MAT prompt convention).
        observation_text (str): textual hint added on top of the image
            observation after a successful code execution.
    """

    DEFAULT_TIPS_DEFAULT = (
        "<tips> Now that we have identified the issue in the image: {problems}, "
        "please proceed to address it by outputting the python code. </tips>"
    )
    DEFAULT_TIPS_CROP = (
        "<tips> We now need to crop the image. Please provide the Python code. "
        "Use [x_min, y_min, x_max, y_max] to represent the bounding box "
        "coordinates. </tips>"
    )
    DEFAULT_TIPS_NONE = (
        "<tips> The image has no issues, so no code is needed in the next step. "
        "You can directly provide the answer. </tips>"
    )
    DEFAULT_OBS_TEXT = (
        "Above is the processed image after executing the code. "
        "Continue the reasoning chain: emit the next <think>...<code>...</code> "
        "step, or conclude with <think>...<answer>...</answer>."
    )
    DEFAULT_OBS_NOOP_TEXT = (
        "Code executed but produced no readable output image. "
        "Please continue the reasoning chain or output <answer>."
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_code_seconds: int = int(config.get("max_code_seconds", 10))
        self.bbox_norm: int = int(config.get("bbox_norm", 1000))

        cache_root = config.get("cache_dir") or os.path.join(
            tempfile.gettempdir(), "agent_code_env"
        )
        self._cache_dir = os.path.normpath(cache_root)
        os.makedirs(self._cache_dir, exist_ok=True)

        self.tips_default: str = config.get("tips_default", self.DEFAULT_TIPS_DEFAULT)
        self.tips_crop: str = config.get("tips_crop", self.DEFAULT_TIPS_CROP)
        self.tips_none: str = config.get("tips_none", self.DEFAULT_TIPS_NONE)
        self.observation_text: str = config.get("observation_text", self.DEFAULT_OBS_TEXT)
        self.observation_noop_text: str = config.get(
            "observation_noop_text", self.DEFAULT_OBS_NOOP_TEXT
        )

        # Per-episode state populated in ``reset``.
        self._image: Optional[Image.Image] = None
        self._question: str = ""
        self._uid: str = ""
        self._input_path: str = ""
        self._output_path: str = ""

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self, data: Optional[Any] = None) -> Any:
        """Cache the initial image + question for this episode."""
        self._image = None
        self._question = ""
        self._uid = uuid.uuid4().hex[:12]
        self._input_path = os.path.normpath(
            os.path.join(self._cache_dir, f"{self._uid}_in.jpg")
        )
        self._output_path = os.path.normpath(
            os.path.join(self._cache_dir, f"{self._uid}_out.jpg")
        )

        if data is None:
            return None

        # ---- load PIL image from multi_modal_data, mirroring DeepEyesEnv ----
        multi_modal_data = data.non_tensor_batch.get("multi_modal_data", [{}])
        mm = multi_modal_data[0] if multi_modal_data else {}
        if mm is None:
            mm = {}
        images = mm.get("images") or mm.get("image") or []
        if images:
            try:
                img = images[0]
                if isinstance(img, dict):
                    img = Image.open(BytesIO(img["bytes"]))
                elif isinstance(img, bytes):
                    img = Image.open(BytesIO(img))
                elif isinstance(img, str):
                    img = Image.open(img)
                if hasattr(img, "load"):
                    img.load()
                self._image = img.convert("RGB")
                self._save_current_image()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AgentCodeEnv.reset: failed to materialise initial image: %s",
                    exc,
                )
                self._image = None

        # ---- pull the raw question for crop bbox extraction ----
        for key in ("question", "problem", "prompt"):
            if key in data.non_tensor_batch:
                val = data.non_tensor_batch[key]
                try:
                    val = val[0]
                except Exception:
                    pass
                if isinstance(val, str):
                    self._question = val
                    break

        return None

    def extract_action(self, content: str) -> Optional[Dict[str, Any]]:
        """Identify which agentic action the model emitted on this turn."""
        if ANSWER_BLOCK_PATTERN.search(content):
            return None

        code_match = CODE_BLOCK_PATTERN.search(content)
        #print(f"content: {content}")
        if code_match:
            print(f"code_match: {code_match.group(0)}")
            return {"type": "code", "raw": content, "body": code_match.group(0)}

        if PROBLEM_BLOCK_PATTERN.search(content):
            #print(f"problem_match: {_extract_problems(content)}")
            return {"type": "problem", "raw": content, "labels": _extract_problems(content)}

        return None

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Execute the parsed action and return (obs, reward, done, info)."""
        if not isinstance(action, dict):
            return (
                "Error: malformed action; expected dict with 'type' field.",
                0.0,
                False,
                {"status": "failed"},
            )

        try:
            if action["type"] == "problem":
                return self._handle_problem(action)
            if action["type"] == "code":
                return self._handle_code(action)
            return (
                f"Error: unknown action type '{action.get('type')}'.",
                0.0,
                False,
                {"status": "failed"},
            )
        except Exception as exc:
            logger.warning("AgentCodeEnv.step failed: %s", exc)
            return (
                f"Error: tool execution raised {type(exc).__name__}: {exc}",
                0.0,
                False,
                {"status": "failed"},
            )

    def close(self):
        self._image = None
        for path in (self._input_path, self._output_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_problem(self, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict]:
        """Mirror eval_mat_coding.py: return a textual <tips> hint."""
        labels = action.get("labels") or []
        if labels and labels[0] == "crop":
            tip = self.tips_crop
        elif labels and labels[0] == "none":
            tip = self.tips_none
        else:
            tip = self.tips_default.format(problems=labels or ["<unspecified>"])
        return tip, 1.0, False, {"status": "success", "tool": "problem_tip", "labels": labels}

    def _handle_code(self, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict]:
        """Run the model-emitted OpenCV snippet and return the new image."""
        if self._image is None:
            return "Error: no input image available.", 0.0, False, {"status": "failed"}

        self._save_current_image()

        # Bootstrap output_path with the current input so that any model code
        # which reads ``path_to_output_image.jpg`` (a common emergent pattern)
        # gets a real file instead of triggering ``imread can't open/read file``
        # warnings and ``cv2.error: empty matrix`` failures downstream.
        try:
            shutil.copyfile(self._input_path, self._output_path)
        except OSError as exc:
            logger.warning(
                "AgentCodeEnv: failed to bootstrap output image %s -> %s: %s",
                self._input_path, self._output_path, exc,
            )

        # Stamp the bootstrap copy with an epoch-zero mtime so any subsequent
        # write by the model's code (which updates mtime to "now") is trivially
        # distinguishable, regardless of the filesystem's mtime resolution.
        try:
            os.utime(self._output_path, (0, 0))
        except OSError:
            pass

        full_response = action.get("raw", "") or action.get("body", "")
        result_replace_path = self._inject_paths(full_response)

        labels = _extract_problems(full_response)
        if labels and labels[0] == "crop":
            result_replace_path = self._rewrite_crop_slicing(result_replace_path)

        ok, payload = self._extract_and_run_code(result_replace_path)
        if not ok:
            return (
                f"Error: code execution failed – {payload}",
                0.0,
                False,
                {"status": "failed", "tool": "code_execute_tool"},
            )

        # Did the model code actually produce a new output file? Bootstrap mtime
        # is epoch zero, so any real write produces a mtime > 1 (sec).
        try:
            output_mtime_after = os.path.getmtime(self._output_path)
        except OSError:
            output_mtime_after = 0.0
        wrote_output = output_mtime_after > 1.0

        new_image = self._load_output_image() if wrote_output else None
        if new_image is None:
            return (
                self.observation_noop_text,
                1.0,
                False,
                {"status": "success", "tool": "code_execute_tool", "image": False},
            )

        self._image = new_image
        try:
            self._save_current_image()
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentCodeEnv: failed to persist new image: %s", exc)
        obs = {"image": new_image, "text": self.observation_text}
        return obs, 1.0, False, {"status": "success", "tool": "code_execute_tool", "image": True}

    # ------------------------------------------------------------------
    # Path injection
    # ------------------------------------------------------------------

    def _inject_paths(self, code_text: str) -> str:
        """Replace ``path_to_*`` placeholders with the env-managed file paths.

        Robust against three common model failure modes that we have observed
        in training logs:

        1. The model writes ``'/path_to_input_image.jpg'`` (with a leading
           slash). A naive ``.replace`` would yield ``'//tmp/agent_code_env/...'``.
        2. The model invents a slightly different alias such as
           ``'path_to_output.jpg'`` or ``'input_image.jpg'``.
        3. The model emits a path with extra surrounding ``./`` such as
           ``'./path_to_input_image.jpg'``.

        We handle (1) by replacing the leading-slash variants first with the
        already-absolute env paths, (2) via the extended placeholder list, and
        (3) by re-running ``os.path.normpath`` on every path-like fragment
        whose normalisation differs from the original (only fragments that
        contain our cache dir, to avoid touching unrelated strings).
        """
        if not code_text:
            return code_text

        for placeholder in _INPUT_PLACEHOLDERS:
            code_text = code_text.replace("/" + placeholder, self._input_path)
            code_text = code_text.replace("./" + placeholder, self._input_path)
            code_text = code_text.replace(placeholder, self._input_path)
        for placeholder in _OUTPUT_PLACEHOLDERS:
            code_text = code_text.replace("/" + placeholder, self._output_path)
            code_text = code_text.replace("./" + placeholder, self._output_path)
            code_text = code_text.replace(placeholder, self._output_path)

        # Final safety net: collapse any accidental ``//cache_dir`` produced by
        # nested placeholder substitutions back to a single slash so cv2 stops
        # emitting ``imread_('//tmp/...')`` warnings.
        cache_no_lead = self._cache_dir.lstrip("/")
        if cache_no_lead:
            code_text = code_text.replace("//" + cache_no_lead, "/" + cache_no_lead)

        return code_text

    # ------------------------------------------------------------------
    # Helpers (ported from eval_mat_coding.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_and_run_code(input_str: str) -> Tuple[bool, Any]:
        """Replicates ``eval_mat_coding.extract_and_run_code`` with two extra
        safety nets compared to the offline eval script:

        * Use a *fresh* ``globals`` dict per call so concurrent ``exec``s
          from sibling envs cannot leak state into each other (the previous
          implementation passed the env module's own ``globals()``, which is
          shared across threads and was occasionally mutated by model code).
        * Hold a process-wide lock around ``exec`` so we serialise cv2 access
          across the ``ThreadPoolExecutor`` workers spawned by ``batch_step``.
          cv2's allocator + several common imports (numpy, PIL) are not
          guaranteed re-entrant from Python threads, and a hard SIGABRT inside
          one worker would tear down the whole trainer process.
        * Catch ``BaseException`` (minus ``KeyboardInterrupt`` / ``SystemExit``)
          so a stray ``sys.exit()`` / ``raise SystemError`` in model code can
          never crash the trainer; we surface it as a regular tool failure.
        """
        match = CODE_BLOCK_PATTERN.search(input_str)
        if not match:
            return False, "code not extracted"
        code_str = match.group(1)
        exec_globals: Dict[str, Any] = {"__builtins__": __builtins__}
        local_vars: Dict[str, Any] = {}
        try:
            with _EXEC_LOCK:
                exec(code_str, exec_globals, local_vars)  # noqa: S102
            return True, local_vars
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    def _rewrite_crop_slicing(self, code_text: str) -> str:
        """Substitute the model's relative bbox slicing with the question's
        absolute pixel coordinates (same trick as in eval_mat_coding.py)."""
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

    def _save_current_image(self):
        if self._image is None:
            return
        try:
            os.makedirs(os.path.dirname(self._input_path) or ".", exist_ok=True)
            self._image.save(self._input_path, format="JPEG", quality=95)
        except OSError as exc:
            logger.warning(
                "AgentCodeEnv: failed to persist input image %s: %s",
                self._input_path, exc,
            )

    def _load_output_image(self) -> Optional[Image.Image]:
        """Load + validate the output image. Returns ``None`` (which routes the
        trajectory through the *no-op* observation branch) instead of raising
        whenever the file is missing / unreadable / degenerate, so a single bad
        OpenCV call from the model can never bring the trainer down.
        """
        if not self._output_path or not os.path.exists(self._output_path):
            return None
        try:
            if os.path.getsize(self._output_path) <= 0:
                return None
        except OSError:
            return None
        try:
            img = Image.open(self._output_path)
            img.load()
            img = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - PIL raises a zoo of types
            logger.warning("AgentCodeEnv: failed to load output image: %s", exc)
            return None
        w, h = img.size
        if w < _MIN_OUTPUT_DIM or h < _MIN_OUTPUT_DIM:
            logger.warning(
                "AgentCodeEnv: rejecting output image of size %dx%d (< %dpx).",
                w, h, _MIN_OUTPUT_DIM,
            )
            return None
        return img
