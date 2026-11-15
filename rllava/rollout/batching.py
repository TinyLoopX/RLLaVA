"""Tensor layout helpers shared by the rollout request scheduler."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class RolloutBatching:
    def __init__(self, pad_token_id):
        self._pad_id = pad_token_id

    def _align_active_samples(self, samples, active_indices):
        """Right-pad input_ids / attention_mask / position_ids of all active samples
        to a common last-dim length so DataProto.concat works. Only these three keys
        may diverge across samples via ``_extend_batch_with_turn``.
        """
        if not active_indices:
            return
        sizes = [samples[i].batch["input_ids"].size(-1) for i in active_indices]
        target = max(sizes)
        if target == min(sizes):
            return
        pad_id = self._pad_id
        for i in active_indices:
            b = samples[i].batch
            pad = target - b["input_ids"].size(-1)
            if pad <= 0:
                continue
            new_source = {k: b[k] for k in b.keys()}
            new_source["input_ids"] = self._pad_last(b["input_ids"], pad, pad_id, side="right")
            new_source["attention_mask"] = self._pad_last(b["attention_mask"], pad, 0, side="right")
            new_source["position_ids"] = self._pad_last(b["position_ids"], pad, 0, side="right")
            samples[i].batch = TensorDict(source=new_source, batch_size=b.batch_size)

    # ------------------------------------------------------------------
    # Trajectory length alignment (right-pad the prompt segment so all
    # trajectories share the same prompt length before DataProto.concat)
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_last(t: torch.Tensor, pad_len: int, value, side: str = "right") -> torch.Tensor:
        """Pad ``t`` along its last dim by ``pad_len`` slots filled with ``value``."""
        if pad_len <= 0:
            return t
        pad_shape = list(t.shape)
        pad_shape[-1] = pad_len
        pad_tensor = torch.full(pad_shape, value, dtype=t.dtype, device=t.device)
        return torch.cat([pad_tensor, t] if side == "left" else [t, pad_tensor], dim=-1)

    def _align_trajectory_lengths(self, samples):
        """Right-pad the prompt segment of every per-sample batch tensor so
        each trajectory shares the same total seq length.

        Per-sample tensor layout coming out of the rollout engine
        (``rllava/engine/inference/vllm.py``) is::

            prompts          : (1, P_i)            ← prompt-side (left-padded)
            responses        : (1, R)              ← response-side (right-padded, R is constant)
            response_mask    : (1, R)              ← response-side (right-padded, R is constant)
            input_ids        : (1, P_i + R)        ← [prompt | response]
            attention_mask   : (1, P_i + R)        ← [prompt | response]
            position_ids     : (1, [3,] P_i + R)   ← [prompt | response]   (mrope adds a leading rope-axis)

        ``P_i`` differs across samples because trajectory ``i`` may have run
        through more multi-turn iterations than another.  We right-pad the
        prompt segment of every sample to ``P_max`` (no-op when already
        aligned) so dim-0 concatenation works.
        """
        if not samples:
            return
        first_batch = samples[0].batch
        if first_batch is None or "responses" not in first_batch.keys():
            return

        # response width R is fixed by rollout config — verify once for safety.
        R = first_batch["responses"].size(-1)
        prompt_lens = []
        for s in samples:
            assert s.batch is not None and s.batch["responses"].size(-1) == R, (
                "RolloutManager expects all samples to share the same response "
                f"length, got {s.batch['responses'].size(-1)} vs {R}."
            )
            prompt_lens.append(s.batch["prompts"].size(-1))

        target_P = max(prompt_lens)
        if target_P == min(prompt_lens):
            return  # all trajectories already share the same prompt length

        pad_id = self._pad_id

        for s in samples:
            b = s.batch
            cur_P = b["prompts"].size(-1)
            pad = target_P - cur_P
            if pad <= 0:
                continue

            # 1) prompts: right-pad with pad_token_id (prompt-only buffer)
            new_prompts = self._pad_last(b["prompts"], pad, pad_id, side="right")

            # 2) input_ids = [prompt | response] → split, pad prompt, re-cat
            in_prompt = b["input_ids"][..., :cur_P]
            in_response = b["input_ids"][..., cur_P:]
            new_input_ids = torch.cat(
                [self._pad_last(in_prompt, pad, pad_id, side="right"), in_response], dim=-1
            )

            # 3) attention_mask: pad the prompt segment with 0 (masked-out)
            am_prompt = b["attention_mask"][..., :cur_P]
            am_response = b["attention_mask"][..., cur_P:]
            new_attention_mask = torch.cat(
                [self._pad_last(am_prompt, pad, 0, side="right"), am_response], dim=-1
            )

            # 4) position_ids: pad the prompt segment with 0 (masked region —
            #    its position_ids never reach the model thanks to attention_mask=0).
            pos_prompt = b["position_ids"][..., :cur_P]
            pos_response = b["position_ids"][..., cur_P:]
            new_position_ids = torch.cat(
                [self._pad_last(pos_prompt, pad, 0, side="right"), pos_response], dim=-1
            )

            # response / response_mask are already (1, R) — keep as-is.
            new_source = {
                "prompts": new_prompts,
                "responses": b["responses"],
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "response_mask": b["response_mask"],
                "position_ids": new_position_ids,
            }
            # carry through any extra tensor keys (e.g. tgt_input_ids) untouched
            for k in b.keys():
                if k not in new_source:
                    new_source[k] = b[k]

            s.batch = TensorDict(source=new_source, batch_size=b.batch_size)

