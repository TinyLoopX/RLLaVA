"""Dataset adapter for MAT-Search agentic RL training.

Differences from the default ``RLHFDataset``:

1. Keep the original question text in ``non_tensor_batch["question"]`` so the
   ``AgentSearchEnv`` (and any logging hooks) can access the human-readable
   query without re-decoding the prompt tokens.
2. Filter the RFT-style training mix to rows whose ``solution`` already
   contains the final ``<answer>...</answer>`` (i.e. the trajectory-terminal
   step) so we get a clean multi-turn reward signal. Rows that only contain a
   gold ``<search>`` are dropped because they do not provide a final answer.
3. Extract the answer string out of the ``<answer>...</answer>`` solution and
   store it as plain text in ``ground_truth``.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from rllava.data.dataset import RLHFDataset


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


class AgentSearchDataset(RLHFDataset):
    """Multi-turn agentic dataset for MAT-Search."""

    def _load_data(self, data_path: str, data_split: str):
        """Pre-filter rows whose ``solution`` already carries a final
        ``<answer>...</answer>`` *before* the parent's overlong filter runs —
        only those rows give a multi-turn reward signal."""
        ds = super()._load_data(data_path, data_split)
        try:
            cols = ds.column_names
            sol_key = "solution" if "solution" in cols else getattr(self, "answer_key", "solution")
            if sol_key in cols:
                ds = ds.filter(
                    lambda doc: isinstance(doc.get(sol_key), str)
                    and "<answer>" in doc[sol_key],
                    desc="agent_search: keep rows containing a final <answer>",
                )
                print(
                    f"AgentSearchDataset: pre-filter kept {len(ds)} rows with a final <answer>."
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
        example["question"] = question_text

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
