"""Agent-Search environment.

Implements the ``BaseEnv`` interface for MAT-Search-style multi-hop search
agent tasks: the model reasons inside ``<think>`` and emits either
``<search>...</search>`` (issue a textual web query) or
``<answer>...</answer>`` (final answer).

The env reproduces the search → ``<information> ... </information>`` formatting
loop used in ``rllava/eval/agent_search/eval_mat_search.py``:

* Calls ``web_search_BOCHA_API`` (or any pluggable backend exposed via the
  ``backend`` config key — ``ddg`` / ``serper`` are also supported via the
  same module) with ``search_num`` results.
* Concatenates each result's ``body`` into a single
  ``<information> 1. Content:... 2. Content:... </information>`` text payload
  used as the next-turn observation.
* When the model emits ``<answer>``, ``extract_action`` returns ``None`` so
  ``MultiTurnWorkflow`` terminates the rollout for this trajectory.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from rllava.ppo.env.base import BaseEnv
from rllava.eval.agent_search.tools.web_search import web_search_BOCHA_API, web_search_SERPER_API
logger = logging.getLogger(__name__)


SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _resolve_search_fn(backend: str) -> Callable[..., List[Dict[str, str]]]:
    """Lazy-import the search backend used by the original MAT-Search eval.

    Imports from ``rllava.eval.agent_search.tools.web_search`` so the training
    loop and the offline benchmark share **the same** retrieval function.
    """
    # Make sure we can import the eval package even when launched via
    # ``-m rllava.train.pipeline.agentic`` from arbitrary cwds.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    backend = (backend or "bocha").lower()
    if backend == "bocha":
        return web_search_BOCHA_API
    if backend in ("serper", "google"):
        return web_search_SERPER_API

    raise ValueError(
        f"Unknown search backend '{backend}'. Supported: bocha / serper / ddg."
    )


class AgentSearchEnv(BaseEnv):
    """Web-search env for MAT-Search style RL training.

    Config keys (all optional):
        backend (str): which retrieval function to use, one of
            ``"bocha"`` (default), ``"serper"`` or ``"ddg"``.
        search_num (int): top-k results to fetch per query, default 4.
        search_mode (str): ``"fast"`` (default) or ``"pro"`` — passed straight
            through to the underlying ``web_search_*`` helpers.
        observation_template (str): ``str.format``-style template producing the
            text observation; receives a single ``{information}`` field that is
            already wrapped in ``<information> ... </information>``.
        bocha_api_key (str): optional override of the BOCHA API key — falls back
            to the default baked into ``web_search_BOCHA_API``.
    """

    DEFAULT_OBS_TEMPLATE = "{information}"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backend: str = config.get("backend", "bocha")
        self.search_num: int = int(config.get("search_num", 4))
        self.search_mode: str = config.get("search_mode", "fast")
        self.observation_template: str = config.get(
            "observation_template", self.DEFAULT_OBS_TEMPLATE
        )

        bocha_key = config.get("bocha_api_key")
        if bocha_key:
            os.environ.setdefault("BOCHA_API", bocha_key)

        self._search_fn: Optional[Callable[..., List[Dict[str, str]]]] = None
        self._question: str = ""

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self, data: Optional[Any] = None) -> Any:
        """Cache the question text (used only for logging) and lazy-init the
        retrieval backend on first use."""
        self._question = ""
        if data is None:
            return None

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

    def extract_action(self, content: str) -> Optional[str]:
        """Return the *last* ``<search>`` query body, or ``None`` if the
        response already contains a final ``<answer>``."""
        if ANSWER_PATTERN.search(content):
            return None
        matches = SEARCH_PATTERN.findall(content)
        if not matches:
            return None
        return matches[-1].strip()

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Execute the search query and return formatted snippets as obs."""
        if not isinstance(action, str) or not action.strip():
            return (
                "Error: empty search query.",
                0.0,
                False,
                {"status": "failed", "tool": "web_search_tool"},
            )

        results = self._run_search(action.strip())


        info_text = self._format_results(results)
        obs_text = self.observation_template.format(information=info_text)
        return (
            obs_text,
            1.0,
            False,
            {
                "status": "success",
                "tool": "web_search_tool",
                "num_results": len(results) if results else 0,
                "query": action.strip(),
            },
        )

    def close(self):
        self._question = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_search(self, query: str) -> List[Dict[str, str]]:
        if self._search_fn is None:
            self._search_fn = _resolve_search_fn(self.backend)
        results = self._search_fn(query, self.search_num, search_mode=self.search_mode)
        print(len(results))
        return list(results) if results else []

    @staticmethod
    def _format_results(results: List[Dict[str, str]]) -> str:
        """Match eval_mat_search.py: ``<information> 1. Content:... 2. ... </information>``."""
        if not results:
            return "<information> No results returned. </information>"
        chunks = ["<information>"]
        for index, item in enumerate(results, start=1):
            body = (item.get("body") or item.get("snippet") or "").strip()
            chunks.append(f"{index}. Content:{body}")
        chunks.append("</information>")
        return " ".join(chunks)
