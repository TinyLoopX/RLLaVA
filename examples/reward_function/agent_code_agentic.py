"""Reward function for the multi-turn agentic version of MAT-Coding.

Differs from ``examples/reward_function/agent_code.py`` (single-turn RFT) in
that the model ``response`` here is the **last turn** of a trajectory that
will normally end with ``<answer>...</answer>`` (the multi-turn workflow stops
when ``<answer>`` is emitted) — so the response should look like a clean final
answer instead of a single intermediate ``<problem>`` / ``<code>`` step.

Score = ``acc_weight * accuracy + format_weight * format + tool_weight * tool_bonus``

* ``accuracy``: F1 between the extracted ``<answer>`` and the gold answer
  (using the same chinese-aware normaliser as ``eval_mat_coding.py``).
* ``format``: 1.0 if the final turn matches ``<think>...</think><answer>...</answer>``,
  else 0.0.
* ``tool_bonus``: 1.0 if the trajectory used a real tool (``<code>`` or
  ``<problem>``) somewhere AND the final answer is correct, else 0.0.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Normalisation / metrics  (parity with rllava/eval/agent_code/eval_mat_coding.py)
# ---------------------------------------------------------------------------

def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _normalize(s: str) -> str:
    chinese_punc = "！？｡＂＃＄％＆＇（）＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～""''、。：《》【】"
    exclude = set(string.punctuation + chinese_punc)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in exclude)

    s = remove_punc(s.lower())
    if _is_chinese(s):
        return s.replace(" ", "")
    return white_space_fix(remove_articles(s))


def _f1(prediction: str, ground_truth: str) -> float:
    if not prediction:
        return 0.0
    norm_pred = _normalize(prediction)
    norm_gt = _normalize(ground_truth)
    if not norm_pred or not norm_gt:
        return 0.0
    if _is_chinese(norm_pred) or _is_chinese(norm_gt):
        pred_tokens = list(norm_pred)
        gt_tokens = list(norm_gt)
    else:
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    if not prediction:
        return 0.0
    return 1.0 if _normalize(prediction) == _normalize(ground_truth) else 0.0


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_TOOL_USE_RE = re.compile(r"<code>.*?</code>|<problem>.*?</problem>", re.DOTALL)


def _last_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _strip_answer_tags(text: str) -> str:
    """Pull out the gold answer from a ``<answer>...</answer>`` solution tag,
    or return the text untouched if no tags are present (the dataset adapter
    already strips them by default)."""
    if not isinstance(text, str):
        return ""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Sub-rewards
# ---------------------------------------------------------------------------
def _format_reward(response: str) -> float:
    if not response:
        return 0.0
    if response.count("<answer>") >= 2 or response.count("<think>") >= 2:
        return 0.0
    return 1.0 if _FORMAT_RE.search(response) else 0.0

def format_reward(response: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_code = r"<think>.*?</think>\s*<code>\s*```python(.*?)```.*?</code>"
    pattern_problem = r"^<think>.*?</think>\s*<problem>\s*\{\s*'[^']+'\s*(?:,\s*'[^']+'\s*)*\}\s*</problem>$"
        
    """
    pattern_answer = r"<answer>.*?</answer>"
    pattern_code = r"<code>\s*```python(.*?)```.*?</code>"
    pattern_problem = r"<problem>\s*\{\s*'[^']+'\s*(?:,\s*'[^']+'\s*)*\}\s*</problem>$"
    """

    if response.count("<answer>")>=2 or response.count("<code>")>=2 or response.count("<think>")>=2 or response.count("<problem>")>=2:
        return 0.0
    elif '<answer>' in response:
        match_answer = re.match(pattern_answer, response, re.DOTALL)
        if match_answer:
            return 1.0
        else:
            return 0.0
    elif '<code>' in response:
        match_code = re.match(pattern_code, response, re.DOTALL)
        if match_code:
            return 1.0
        else:
            return 0.0
    elif '<problem>' in response:
        match_problem = re.match(pattern_problem, response, re.DOTALL)
        if match_problem:
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


def _accuracy_reward(response: str, ground_truth: str) -> Dict[str, float]:
    pred = _last_answer(response)
    gt = _strip_answer_tags(ground_truth)
    if not pred or not gt:
        return {"f1": 0.0, "em": 0.0}
    em = _exact_match(pred, gt)
    f1 = _f1(pred, gt) if em < 0.5 else 1.0
    return {"f1": f1, "em": em}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _tool_reward(reward_input: Dict[str, Any]) -> Dict[str, float]:
    # Prefer workflow-recorded state; fallback to regex for non-workflow evals.
    step_reward = _as_float(reward_input.get("step_reward", 0.0))
    return 1.0 if step_reward >= 1.0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_score(
    reward_input: Dict[str, Any],
    acc_weight: float = 0.8,
    format_weight: float = 0.2,
    tool_weight: float = 1.2,
) -> Dict[str, float]:
    """Multi-turn agentic reward for MAT-Coding-style trajectories."""
    if not isinstance(reward_input, dict):
        raise ValueError("Use reward_type=sequential for agent_code_agentic reward.")
    response: str = reward_input.get("response", "") or ""
    ground_truth = reward_input.get("ground_truth", "")
    #print(f"ground_truth",ground_truth)
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get(
            "ground_truth", ground_truth.get("answer", str(ground_truth))
        )

    fmt = _format_reward(response)
    acc = _accuracy_reward(response, ground_truth)
    tool_stats = _tool_reward(reward_input)

    overall = acc_weight * acc["f1"] + format_weight * fmt + tool_weight * tool_stats

    return {
        "overall": overall,
        "accuracy": acc["f1"],
        "exact_match": acc["em"],
        "format": fmt,
        "tool": tool_stats,

    }
