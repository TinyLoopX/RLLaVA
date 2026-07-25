"""DeepEyes-style reward function for visual agentic tasks.

Supports two accuracy modes:
    1. **LLM-as-a-judge** (default): uses an OpenAI-compatible chat model to
       semantically judge whether the extracted answer matches the ground truth.
       Requires env var ``LLM_AS_A_JUDGE_BASE`` pointing to the judge endpoint.
    2. **Exact match fallback**: when no judge endpoint is available, falls back
       to case-insensitive exact match.

Score = 0.8 * accuracy + 0.2 * format + 1.2 * tool_bonus

where:
    - accuracy ∈ {0, 1}
    - format   ∈ {-1, 0}   (penalty for malformed output)
    - tool_bonus ∈ {0, 1}   (reward for using tools AND being correct)
"""

import logging
import os
import random
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LLM-as-a-judge setup
# ---------------------------------------------------------------------------
_judge_client = None
_judge_model = ""

_JUDGE_BASE = os.environ.get("LLM_AS_A_JUDGE_BASE", "")
if _JUDGE_BASE:
    try:
        from openai import OpenAI
        import requests

        _judge_client = OpenAI(api_key="EMPTY", base_url=_JUDGE_BASE)
        resp = requests.get(f"{_JUDGE_BASE}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()
        if models.get("data"):
            _judge_model = models["data"][0]["id"]
        else:
            logger.warning("LLM judge: no models found at %s", _JUDGE_BASE)
    except Exception as exc:
        logger.warning("LLM judge init failed (%s), falling back to exact match.", exc)
        _judge_client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    """Extract content from ``<answer>...</answer>`` tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _check_format(text: str) -> bool:
    """Return True if the output conforms to DeepEyes format rules."""
    if text.count("<think>") != text.count("</think>"):
        return False
    after_think = text.split("</think>")[-1].strip() if "</think>" in text else text
    if after_think.count("<answer>") != after_think.count("</answer>"):
        return False
    if "<answer>" not in after_think and "<tool_call>" not in after_think:
        return False
    return True


def _has_tool_usage(text: str) -> bool:
    return bool(re.search(r"<tool_call>.*?</tool_call>", text, re.DOTALL))

def _tool_reward(reward_input: Dict[str, Any]) -> Dict[str, float]:
    # Prefer workflow-recorded state; fallback to regex for non-workflow evals.
    step_reward = float(reward_input.get("step_reward", 0.0))
    return step_reward

def _judge_accuracy(answer: str, ground_truth: str, question: str = "") -> float:
    """Return 1.0 if *answer* matches *ground_truth*, else 0.0."""

    if not answer:
        return 0.0

    # Penalize excessively long answers (potential judge hacking)
    if len(answer) >= 1000:
        return 0.0

    # --- LLM-as-a-judge path ---
    if _judge_client and _judge_model:
        system_prompt = (
            "You are an expert evaluator. Determine if the model answer is semantically "
            "equivalent to the standard answer for the given question.\n"
            "Reply with a single word: CORRECT or INCORRECT."
        )
        user_prompt = (
            f"[Question]: {question}\n"
            f"[Standard Answer]: {ground_truth}\n"
            f"[Model Answer]: {answer}\n"
            f"[Your Judgement]:"
        )
        try:
            resp = _judge_client.chat.completions.create(
                model=_judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                seed=random.randint(0, 1_000_000),
                temperature=0.1,
                max_tokens=16,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content.strip().upper()
            if "CORRECT" in text and "INCORRECT" not in text:
                return 1.0
            return 0.0
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
            # fall through to exact match

    # --- exact match fallback ---
    return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_score(
    reward_input: dict[str, Any],
    acc_weight: float = 0.8,
    format_weight: float = 0.2,
    tool_weight: float = 1.2,
) -> dict[str, float]:
    """Compute DeepEyes-style reward for a single sample.

    ``reward_input`` must contain:
        - ``response``: the full model output string
        - ``ground_truth``: the reference answer
    Optionally:
        - ``extra_info.question``: the original question (for LLM judge)
    """
    if not isinstance(reward_input, dict):
        raise ValueError("Use reward_type=sequential for deepeyes_visual reward.")

    response: str = reward_input["response"]
    ground_truth = reward_input["ground_truth"]
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", ground_truth.get("answer", str(ground_truth)))
    ground_truth = str(ground_truth)
    question: str = ""
    extra_info = reward_input.get("extra_info")
    if isinstance(extra_info, dict):
        question = extra_info.get("question", "")

    # 1) format
    fmt_ok = _check_format(response)
    format_score = 0.0 if fmt_ok else -1.0

    # 2) accuracy
    answer_text = _extract_answer(response)
    if not answer_text:
        # fallback: try to get content after last </think>
        after_think = response.split("</think>")[-1].strip() if "</think>" in response else ""
        after_think = re.sub(r"<tool_call>.*?</tool_call>", "", after_think, flags=re.DOTALL)
        after_think = re.sub(r"<tool_response>.*?</tool_response>", "", after_think, flags=re.DOTALL)
        answer_text = after_think.strip()

    accuracy_score = _judge_accuracy(answer_text, ground_truth, question)

    # 3) tool usage bonus (only when correct)
    #used_tool = _has_tool_usage(response)
    #tool_score = 1.0 if used_tool and accuracy_score > 0.5 else 0.0
    tool_score = _tool_reward(reward_input)

    overall = acc_weight * accuracy_score + format_weight * format_score + tool_weight * tool_score

    return {
        "overall": overall,
        "accuracy": accuracy_score,
        "format": format_score,
        "tool": tool_score,
    }
