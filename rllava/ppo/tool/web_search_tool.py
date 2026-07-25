"""``BaseTool`` wrapper around the web-search logic of
``examples/tasks/agent_search/agent_search_env.py``.

Mirrors the relationship between ``DeepEyesEnv`` and ``VisualCropTool`` —
both pipelines (multi-turn workflow via ``BaseEnv`` and tool registry via
``BaseTool``) share the **same** retrieval function from
``rllava.eval.agent_search.tools.web_search``, so the training-time tool
behaviour is byte-for-byte identical to ``eval_mat_search.py``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseTool

logger = logging.getLogger(__name__)


SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _resolve_search_fn(backend: str) -> Callable[..., List[Dict[str, str]]]:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from rllava.eval.agent_search.tools import web_search as _ws  # type: ignore

    backend = (backend or "bocha").lower()
    if backend == "bocha":
        return _ws.web_search_BOCHA_API
    if backend in ("serper", "google"):
        return _ws.web_search_SERPER_API
    if backend in ("ddg", "duckduckgo"):
        return _ws.web_search_DDG
    raise ValueError(
        f"Unknown search backend '{backend}'. Supported: bocha / serper / ddg."
    )


class WebSearchTool(BaseTool):
    """Web-search tool for MAT-Search-style trajectories."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend: str = config.get("backend", "bocha")
        self.search_num: int = int(config.get("search_num", 4))
        self.search_mode: str = config.get("search_mode", "fast")
        self._search_fn: Optional[Callable[..., List[Dict[str, str]]]] = None

        bocha_key = config.get("bocha_api_key")
        if bocha_key:
            os.environ.setdefault("BOCHA_API", bocha_key)

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------

    def extract_tool_call(self, content: str) -> Optional[str]:
        if ANSWER_PATTERN.search(content):
            return None
        matches = SEARCH_PATTERN.findall(content)
        return matches[-1].strip() if matches else None

    def execute(self, tool_content: str) -> Tuple[Any, bool]:
        query = (tool_content or "").strip()
        if not query:
            return "Error: empty search query.", False
        try:
            if self._search_fn is None:
                self._search_fn = _resolve_search_fn(self.backend)
            results = self._search_fn(query, self.search_num, search_mode=self.search_mode)
        except Exception as exc:
            return f"Error: search backend raised {type(exc).__name__}: {exc}", False

        if not results:
            return "<information> No results returned. </information>", True

        chunks = ["<information>"]
        for index, item in enumerate(results, start=1):
            body = (item.get("body") or item.get("snippet") or "").strip()
            chunks.append(f"{index}. Content:{body}")
        chunks.append("</information>")
        return " ".join(chunks), True
