import asyncio
import logging
import numpy as np
import torch
import copy
from typing import Any
from tensordict import TensorDict

from rllava.data.protocol import DataProto
from rllava.data.data_utils import process_image
from rllava.ppo.env import initialize_env_from_config

logger = logging.getLogger(__name__)


class MultiTurnWorkflow:
    """Multi-turn workflow driven by a single env.

    The env handles all task-specific logic (action parsing, tool execution,
    observation generation) via the ``BaseEnv`` interface:
    ``reset(data)`` / ``extract_action(content)`` / ``step(action)`` / ``close()``.

    History tracking uses incremental token ID concatenation (same as the official
    DeepEyes / Mini-o3 implementations) rather than re-rendering the full
    conversation text each turn.  This avoids repeated ``apply_chat_template``
    calls and is more robust against chat-template edge cases.
    """

    def __init__(
        self,
        reward,
        tokenizer,
        processor,
        max_turns: int = 5,
        discount: float = 1.0,
        env_config_path: str = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_turns = max_turns
        self.env_config_path = env_config_path

    async def arun(self, rollout: Any, gen_batch: DataProto, source_data: DataProto | None = None) -> DataProto:
        """Async wrapper for run."""
        return await asyncio.to_thread(self.run, rollout, gen_batch, source_data)

    def run(self, rollout: Any, gen_batch: DataProto, source_data: DataProto | None = None) -> DataProto:
        """Run multi-turn generation in batched turns over active samples."""
        num_samples = len(gen_batch.batch["input_ids"])
        samples = []
        for i in range(num_samples):
            single_data = gen_batch[[i]]
            single_data.non_tensor_batch["step_reward"] = np.zeros(1, dtype=np.float32)
            if "multi_modal_data" in single_data.non_tensor_batch:
                mmd = single_data.non_tensor_batch["multi_modal_data"]
                new_mmd = np.empty(1, dtype=object)
                new_mmd[0] = copy.deepcopy(mmd[0]) if mmd[0] is not None else {}
                single_data.non_tensor_batch["multi_modal_data"] = new_mmd
            if source_data is not None:
                for key, value in source_data.non_tensor_batch.items():
                    if key not in single_data.non_tensor_batch:
                        try:
                            single_data.non_tensor_batch[key] = value[[i]]
                        except Exception:
                            single_data.non_tensor_batch[key] = np.array([value[i]], dtype=object)
            # Repair input_ids whose <|image_pad|> count was clipped by the dataset's
            # max_prompt_length truncation: a large image expanded by processor() may
            # exceed max_prompt_length, and right-truncation can drop image_pad tokens
            # at the tail, leaving fewer pads in input_ids than image_processor produces
            # at forward time ("Image features and image tokens do not match").
            # raw_prompt_ids (one pad per image, much shorter) is intact, so we rebuild
            # input_ids / attention_mask / position_ids from it whenever a mismatch is
            # detected.  No-op for samples where input_ids already matches.
            self._maybe_rebuild_initial_input_ids(single_data)
            samples.append(single_data)

        envs = [self._initialize_env(self.env_config_path) for _ in range(num_samples)]
        for i, env in enumerate(envs):
            if env is not None:
                env.reset(samples[i])

        active_indices = list(range(num_samples))
        turn = 0

        # Cache pad_token_id once for the trajectory-padding step at the end.
        self._pad_id = self.tokenizer.pad_token_id

        try:
            while active_indices and turn < self.max_turns:
                # Per-turn alignment: _add_observation may have grown each sample's
                # input_ids by a different amount (image grid varies per obs), so
                # right-pad them to a common length before DataProto.concat.
                self._align_active_samples(samples, active_indices)
                active_batch = DataProto.concat([samples[i] for i in active_indices])
                # Force n=1 per turn; the caller pre-expands gen_batch by rollout.n
                # so each row in active_batch is already an independent trajectory.
                active_batch.meta_info["n"] = 1
                active_output = rollout.generate_sequences(active_batch)
                content_list = self.tokenizer.batch_decode(
                    active_output.batch["responses"], skip_special_tokens=True
                )

                next_active = []
                pending_env_steps = []
                for j, sample_idx in enumerate(active_indices):
                    gen_output = active_output[[j]]
                    prev_data = samples[sample_idx]
                    for key, value in prev_data.non_tensor_batch.items():
                        if key not in gen_output.non_tensor_batch:
                            gen_output.non_tensor_batch[key] = value

                    content = content_list[j] if j < len(content_list) else ""
                    env = envs[sample_idx]
                    has_answer = ("<answer>" in content) and ("</answer>" in content)
                    has_tool_call = env is not None and env.extract_action(content) is not None

                    # Extract actual (non-padded) response token IDs for incremental
                    # history concatenation.  Responses are right-padded with pad_token_id.
                    response_tensor = active_output.batch["responses"][j]
                    pad_id = self.tokenizer.pad_token_id
                    nonpad = (response_tensor != pad_id).nonzero(as_tuple=False)
                    response_ids = response_tensor[:nonpad[-1].item() + 1].tolist() if len(nonpad) > 0 else []

                    if has_tool_call:
                        # Tool call: queue for batched env.step, continue loop.
                        action = env.extract_action(content)
                        pending_env_steps.append((sample_idx, env, action, gen_output, content, response_ids))
                    elif has_answer:
                        # Final answer: episode done, reward computed later by pipeline.
                        pass
                    else:
                        # Neither: give feedback and continue if turns remain.
                        if (turn + 1) < self.max_turns:
                            self._add_observation(
                                gen_output,
                                response_ids,
                                "Please either call a tool with <tool_call>...</tool_call> "
                                "or provide the final result with <answer>...</answer>.",
                                0.0,
                                {},
                            )
                            next_active.append(sample_idx)

                    samples[sample_idx] = gen_output

                if pending_env_steps:
                    step_results = pending_env_steps[0][1].batch_step(pending_env_steps)

                    for sample_idx, obs, step_reward, env_done, info, gen_output, content, response_ids in step_results:
                        # Only append obs if the trajectory will have another generation turn.
                        # Otherwise obs tokens would land at the tail of input_ids and shift
                        # the response window used by forward_batch (logits[:, -R-1:-1]).
                        if not env_done and (turn + 1) < self.max_turns:
                            self._add_observation(gen_output, response_ids, obs, step_reward, info)
                            next_active.append(sample_idx)
                        samples[sample_idx] = gen_output

                active_indices = next_active
                turn += 1
        finally:
            for env in envs:
                if env is not None:
                    env.close()

        # Different trajectories exit the loop at different turns, so each
        # samples[i].batch has shape (1, P_init + turn_i * R).  Before
        # concatenating along dim=0 we right-pad the *prompt segment* of every
        # batched tensor to the trajectory-wise max so torch.cat doesn't fail
        # on mismatched seq-dim sizes.  Layout invariant
        # ``[prompt | response]`` (assumed by metrics.py / actor.compute_*) is
        # preserved — we only insert pad between prompt-end and response-start.
        self._align_trajectory_lengths(samples)

        result = DataProto.concat(samples)
        # torch.cat on TensorDicts may produce a lazy representation
        # (e.g. LazyStackedTensorDict).  When such a TensorDict is later
        # assigned into an empty TensorDict (after pop) via union, and
        # then moved to CUDA, the internal offset bookkeeping can become
        # inconsistent, corrupting tensor data.  Rebuilding from to_dict()
        # materialises all tensors into a plain, concrete TensorDict.
        if result.batch is not None:
            result.batch = TensorDict(
                source=result.batch.to_dict(),
                batch_size=result.batch.batch_size,
            )
        return result

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    def _add_observation(self, gen_output, response_ids, obs, step_reward, info):
        """Append response tokens + observation tokens to raw_prompt_ids incrementally.

        Uses token ID concatenation (same as official DeepEyes / Mini-o3) instead of
        re-rendering the full conversation history each turn via apply_chat_template.

        Layout appended to raw_prompt_ids each turn:
            [response_token_ids ...] + [<|im_start|>user\\n{obs}<|im_end|>\\n<|im_start|>assistant\\n]

        Image observations are accumulated in multi_modal_data["images"]; vLLM
        handles expansion of the single <|image_pad|> placeholder internally.

        Args:
            gen_output:   DataProto for this sample (modified in-place).
            response_ids: List[int] — actual (non-padded) response token IDs
                          from the previous generation turn.
            obs:          Env observation.  Supported types:
                          - dict with "image"/"screenshot" (and optional "text")
                          - PIL.Image
                          - plain string
            info:         Metadata dict from env (unused, kept for future use).
        """
        # Accumulate: existing prompt tokens + new response tokens
        current_ids = list(gen_output.non_tensor_batch["raw_prompt_ids"][0])
        current_ids.extend(response_ids)

        # Parse observation into text + optional image. Image candidates are
        # validated below; if they fail validation we silently drop the image
        # and degrade this turn to text-only so a bad env observation cannot
        # crash the whole training step.
        multi_modal_data = gen_output.non_tensor_batch.get("multi_modal_data", [{}])
        images = list((multi_modal_data[0] or {}).get("images") or [])

        new_image = None
        candidate_image = None
        text_part = ""
        if isinstance(obs, dict) and ("image" in obs or "screenshot" in obs):
            candidate_image = obs.get("image") or obs.get("screenshot")
            text_part = obs.get("text", "") or ""
        elif hasattr(obs, "save"):  # PIL.Image-like
            candidate_image = obs
        else:
            text_part = str(obs)

        if candidate_image is not None and self._is_image_safe(candidate_image):
            new_image = candidate_image
            images.append(new_image)
            obs_text = f"<|vision_start|><|image_pad|><|vision_end|>\n{text_part}".rstrip()
        elif candidate_image is not None:
            # Unsafe / degenerate image — drop it but keep any accompanying text
            # so the trajectory still makes forward progress.
            logger.warning(
                "MultiTurnWorkflow: dropped unsafe image observation (type=%s).",
                type(candidate_image).__name__,
            )
            obs_text = text_part or (
                "(image observation was unavailable; please continue reasoning.)"
            )
        else:
            obs_text = text_part

        # Encode only the new user turn as a ChatML suffix (no full re-render).
        # tokenizer.encode does NOT expand image placeholders; vLLM does that.
        obs_suffix = f"<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n"
        obs_ids = self.tokenizer.encode(obs_suffix, add_special_tokens=False)
        current_ids.extend(obs_ids)

        # Update raw_prompt_ids: must be a 1-D object array (not 2-D).
        # np.array([list], dtype=object) may infer shape (1, seq_len) when the
        # list has a uniform length — always use np.empty + index assignment.
        raw_ids = np.empty(1, dtype=object)
        raw_ids[0] = current_ids
        gen_output.non_tensor_batch["raw_prompt_ids"] = raw_ids

        # Update multi_modal_data so vLLM processes all accumulated images.
        multi_modal_data[0]["images"] = images
        gen_output.non_tensor_batch["multi_modal_data"] = multi_modal_data
        gen_output.non_tensor_batch["step_reward"] += np.float32(step_reward)

        # Keep training-time tensors in sync: expand <|image_pad|> for the new image
        # and append [response_ids + expanded_obs_ids] to input_ids / attention_mask /
        # position_ids so n_image_tokens matches n_image_features at forward time.
        try:
            self._extend_batch_with_turn(
                gen_output, response_ids, obs_ids, images, new_image is not None
            )
        except Exception as exc:  # noqa: BLE001 - last-line defence
            # The image processor occasionally rejects pathological inputs even
            # after our pre-validation (rare jpeg quirks, edge-case grid_thw, ...).
            # Fall back to a text-only turn so this trajectory still proceeds.
            logger.warning(
                "MultiTurnWorkflow: image-side tensor extension failed (%s); "
                "retrying turn as text-only.", exc,
            )
            if new_image is not None:
                # Roll back the image we just appended to multi_modal_data /
                # raw_prompt_ids and re-encode the turn without an image.
                images = images[:-1]
                multi_modal_data[0]["images"] = images
                gen_output.non_tensor_batch["multi_modal_data"] = multi_modal_data

                fallback_text = text_part or (
                    "(image observation was unavailable; please continue reasoning.)"
                )
                fallback_suffix = (
                    f"<|im_start|>user\n{fallback_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                fallback_obs_ids = self.tokenizer.encode(
                    fallback_suffix, add_special_tokens=False
                )

                # Rebuild raw_prompt_ids without the image placeholder turn.
                rebuilt = list(gen_output.non_tensor_batch["raw_prompt_ids"][0])
                rebuilt = rebuilt[: len(rebuilt) - len(obs_ids)]
                rebuilt.extend(fallback_obs_ids)
                raw_ids = np.empty(1, dtype=object)
                raw_ids[0] = rebuilt
                gen_output.non_tensor_batch["raw_prompt_ids"] = raw_ids

                self._extend_batch_with_turn(
                    gen_output, response_ids, fallback_obs_ids, images, False
                )

    @staticmethod
    def _is_image_safe(image, min_dim: int = 8) -> bool:
        """Cheap pre-flight check for an env-supplied image.

        Returns ``False`` for objects that don't look like a usable PIL image
        or that are below the processor's minimum patch size; the caller
        degrades the turn to text-only when this returns ``False``.
        """
        if image is None:
            return False
        size = getattr(image, "size", None)
        if size is None or len(size) != 2:
            return False
        try:
            w, h = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return False
        if w < min_dim or h < min_dim:
            return False
        return True


    # ------------------------------------------------------------------
    # Sync training tensors with multi-turn images
    # ------------------------------------------------------------------

    def _get_rope_index_fn(self):
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from rllava.model.patch.qwen3_vl import get_rope_index
        else:
            from rllava.model.patch.qwen2_vl import get_rope_index
        return get_rope_index

    def _maybe_rebuild_initial_input_ids(self, sample):
        """If dataset-side truncation clipped <|image_pad|> tokens (long-image prompts
        exceeding max_prompt_length), rebuild input_ids from raw_prompt_ids using
        the actual per-image grid_thw, so forward-time n_image_tokens matches
        n_image_features.  No-op when the counts already agree.
        """
        if sample.batch is None or "input_ids" not in sample.batch.keys():
            return
        if "multi_modal_data" not in sample.non_tensor_batch:
            return
        mmd_arr = sample.non_tensor_batch["multi_modal_data"]
        mmd = mmd_arr[0] if len(mmd_arr) > 0 else None
        if not mmd:
            return
        images = mmd.get("images") or []
        if not images:
            return

        meta = sample.meta_info or {}
        min_pixels = meta.get("min_pixels")
        max_pixels = meta.get("max_pixels")
        processed = [process_image(img, min_pixels, max_pixels, self.processor) for img in images]
        mm_inputs = self.processor.image_processor(images=processed, return_tensors="pt")
        grid_thw = mm_inputs["image_grid_thw"].to(torch.long)  # (N, 3)
        merge_size = self.processor.image_processor.merge_size
        n_pads = [int(g.prod().item()) // (merge_size ** 2) for g in grid_thw]
        expected_pads = sum(n_pads)

        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        actual_pads = int((sample.batch["input_ids"] == image_pad_id).sum().item())
        if actual_pads == expected_pads:
            return  # already consistent

        raw_ids_arr = sample.non_tensor_batch.get("raw_prompt_ids")
        if raw_ids_arr is None or len(raw_ids_arr) == 0:
            return
        raw_ids = list(raw_ids_arr[0])

        rebuilt = []
        img_idx = 0
        for tok in raw_ids:
            if tok == image_pad_id:
                if img_idx >= len(n_pads):
                    return  # malformed; bail out without modification
                rebuilt.extend([image_pad_id] * n_pads[img_idx])
                img_idx += 1
            else:
                rebuilt.append(tok)
        if img_idx != len(n_pads):
            return  # raw_prompt_ids had fewer image_pad placeholders than images

        device = sample.batch["input_ids"].device
        ids_dtype = sample.batch["input_ids"].dtype
        attn_dtype = sample.batch["attention_mask"].dtype
        pos = sample.batch["position_ids"]

        new_ids = torch.tensor(rebuilt, dtype=ids_dtype, device=device).unsqueeze(0)
        new_attn = torch.ones_like(new_ids, dtype=attn_dtype)

        if pos.dim() == 3:  # qwen2vl mrope: (1, 4, L)
            get_rope_index = self._get_rope_index_fn()
            vision_pos = get_rope_index(
                self.processor,
                input_ids=new_ids[0],
                image_grid_thw=grid_thw,
                attention_mask=new_attn[0],
            ).to(device=device, dtype=pos.dtype)  # (3, L)
            text_pos = torch.arange(new_ids.size(-1), device=device, dtype=pos.dtype).unsqueeze(0)
            new_pos = torch.cat([text_pos, vision_pos], dim=0).unsqueeze(0)  # (1, 4, L)
        else:
            new_pos = torch.arange(
                new_ids.size(-1), device=device, dtype=pos.dtype
            ).unsqueeze(0)

        new_source = {k: sample.batch[k] for k in sample.batch.keys()}
        new_source["input_ids"] = new_ids
        new_source["attention_mask"] = new_attn
        new_source["position_ids"] = new_pos
        sample.batch = TensorDict(source=new_source, batch_size=sample.batch.batch_size)

    def _extend_batch_with_turn(self, gen_output, response_ids, obs_ids, all_images, has_new_image):
        """Append [response_ids + expanded obs_ids] to input_ids / attention_mask /
        position_ids without recomputing position_ids over the whole sequence.

        Strategy: leave the existing position_ids of the dataset prompt + vLLM-produced
        response untouched, and only compute positions for the new segment we are
        appending. This avoids relying on the initial image's <|image_pad|> count in
        input_ids matching what re-running image_processor would predict (which can
        drift by a few tokens between the dataset's processor path and a direct
        image_processor call, breaking get_rope_index's stride bookkeeping).

        For the new segment:
          - response_ids: pure text, all 4 mrope axes advance by +1 (same scheme as
            ``vllm.py`` uses for response position_ids).
          - expanded obs_ids: contains at most one new image, so we feed *only this
            obs segment* and *only the new image's grid_thw* to get_rope_index — no
            cross-segment image_pad accounting needed.
        """
        if gen_output.batch is None or "input_ids" not in gen_output.batch.keys():
            return

        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        merge_size = self.processor.image_processor.merge_size

        # Only process the *new* image to compute its grid_thw (used both to expand
        # the placeholder in obs_ids and to feed get_rope_index for the obs segment).
        new_grid_thw = None
        if has_new_image and all_images:
            meta = gen_output.meta_info or {}
            min_pixels = meta.get("min_pixels")
            max_pixels = meta.get("max_pixels")
            new_img = process_image(all_images[-1], min_pixels, max_pixels, self.processor)
            new_mm = self.processor.image_processor(images=[new_img], return_tensors="pt")
            new_grid_thw = new_mm["image_grid_thw"].to(torch.long)  # (1, 3)

        expanded_obs_ids = list(obs_ids)
        if new_grid_thw is not None and new_grid_thw.size(0) > 0:
            n_pad = int(new_grid_thw[0].prod().item()) // (merge_size ** 2)
            expanded_obs_ids = []
            for tok in obs_ids:
                if tok == image_pad_id:
                    expanded_obs_ids.extend([image_pad_id] * n_pad)
                else:
                    expanded_obs_ids.append(tok)

        extra_ids = list(response_ids) + expanded_obs_ids
        if not extra_ids:
            return

        device = gen_output.batch["input_ids"].device
        in_dtype = gen_output.batch["input_ids"].dtype
        extra_t = torch.tensor(extra_ids, dtype=in_dtype, device=device).unsqueeze(0)

        new_input_ids = torch.cat([gen_output.batch["input_ids"], extra_t], dim=-1)
        new_attn = torch.cat(
            [
                gen_output.batch["attention_mask"],
                torch.ones_like(extra_t, dtype=gen_output.batch["attention_mask"].dtype),
            ],
            dim=-1,
        )

        pos = gen_output.batch["position_ids"]
        response_len = len(response_ids)
        obs_len = len(expanded_obs_ids)

        if pos.dim() == 3:  # qwen2vl mrope: (1, 4, L)
            # response segment: all 4 axes advance by +1 (matches vllm.py's scheme).
            if response_len > 0:
                resp_delta = torch.arange(
                    1, response_len + 1, device=device, dtype=pos.dtype
                ).view(1, 1, -1)
                resp_pos = pos[..., -1:] + resp_delta  # (1, 4, response_len)
            else:
                resp_pos = pos[..., :0]

            # obs segment: local mrope over a self-contained sub-sequence with at
            # most one image, offset to start right after resp_pos.
            if obs_len > 0:
                get_rope_index = self._get_rope_index_fn()
                obs_t = torch.tensor(expanded_obs_ids, dtype=in_dtype, device=device)
                obs_attn_1d = torch.ones_like(obs_t)
                obs_vision_pos = get_rope_index(
                    self.processor,
                    input_ids=obs_t,
                    image_grid_thw=new_grid_thw,
                    attention_mask=obs_attn_1d,
                ).to(device=device, dtype=pos.dtype)  # (3, obs_len)
                obs_text_pos = torch.arange(
                    obs_len, device=device, dtype=pos.dtype
                ).unsqueeze(0)  # (1, obs_len)
                obs_local = torch.cat([obs_text_pos, obs_vision_pos], dim=0).unsqueeze(0)
                last_axis = resp_pos[..., -1:] if response_len > 0 else pos[..., -1:]
                obs_pos = obs_local + (last_axis + 1)
            else:
                obs_pos = pos[..., :0]

            new_pos = torch.cat([pos, resp_pos, obs_pos], dim=-1)
        else:
            last = pos[..., -1:].clone()
            delta = torch.arange(1, extra_t.size(-1) + 1, device=device).view(1, -1).to(pos.dtype)
            new_pos = torch.cat([pos, last + delta], dim=-1)

        new_source = {k: gen_output.batch[k] for k in gen_output.batch.keys()}
        new_source["input_ids"] = new_input_ids
        new_source["attention_mask"] = new_attn
        new_source["position_ids"] = new_pos
        gen_output.batch = TensorDict(source=new_source, batch_size=gen_output.batch.batch_size)

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
                "MultiTurnWorkflow expects all samples to share the same response "
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

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_env(env_config_path):
        if not env_config_path:
            return None
        return initialize_env_from_config(env_config_path)
