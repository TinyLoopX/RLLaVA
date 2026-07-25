"""Dataset adapter for MAT-Coding agentic RL training.

Differences from the default ``RLHFDataset``:

1. Filter the RFT-style training mix to **only the ``pre_answer`` rows** so each
   sample starts from a fresh image + question and the gold ``solution`` field
   already carries the final ``<answer>...</answer>`` (which doubles as the
   multi-turn reward target).
2. Keep the original question text in ``non_tensor_batch["question"]`` so the
   ``AgentCodeEnv`` can extract the bbox embedded inside ``<query>`` for crop
   problems (parity with ``eval_mat_coding.py``).
3. Keep ``ground_truth`` as a plain string answer extracted from the
   ``<answer>...</answer>`` solution tag.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from rllava.data.dataset import RLHFDataset


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


class AgentCodeDataset(RLHFDataset):
    """Multi-turn agentic dataset for MAT-Coding."""

    # Only keep rows whose ``solution`` already contains the final answer so we
    # have a multi-turn reward signal per trajectory. ``pre_answer`` matches
    # the type tag used in ``rft_agent_code_*k.json``.
    KEEP_TYPE = "pre_answer"

    def _load_data(self, data_path: str, data_split: str):
        """Pre-filter ``type == pre_answer`` *before* the parent's overlong
        filter runs, so the trajectory-terminal rows are guaranteed to survive
        regardless of whether the upstream length-filter cache is stale."""
        ds = super()._load_data(data_path, data_split)
        try:
            if "type" in ds.column_names:
                ds = ds.filter(
                    lambda doc: doc.get("type") == self.KEEP_TYPE,
                    desc=f"agent_code: keep '{self.KEEP_TYPE}' rows",
                )
                print(
                    f"AgentCodeDataset: pre-filter kept {len(ds)} '{self.KEEP_TYPE}' rows."
                )
        except Exception:
            pass
        return ds

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_example: Dict[str, Any] = dict(self.dataset[index])
        question_text = ""
        for key in (self.prompt_key, "problem", "question", "prompt"):
            val = raw_example.get(key)
            if isinstance(val, str):
                question_text = val
                break

        example = super().__getitem__(index)

        # Stash the raw question for the env (used for crop bbox extraction).
        example["question"] = question_text

        # ``ground_truth`` originates from ``solution``; extract the final answer
        # from the ``<answer>...</answer>`` tag for downstream reward scoring.
        gt = example.get("ground_truth", "")
        if isinstance(gt, dict):
            gt = gt.get("ground_truth", gt.get("answer", str(gt)))
        if isinstance(gt, str):
            m = _ANSWER_RE.search(gt)
            if m:
                example["ground_truth"] = m.group(1).strip()
            else:
                example["ground_truth"] = gt.strip()
        else:
            example["ground_truth"] = str(gt)

        return example
