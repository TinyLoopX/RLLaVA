"""Tool/answer agent: context and action state, without an execution loop."""

from __future__ import annotations

import logging
import numpy as np
import torch
from tensordict import TensorDict
from rllava.data.data_utils import process_image
from .base import AgentAction, SingleTurnAgent

logger = logging.getLogger(__name__)


class MultiTurnAgent(SingleTurnAgent):
    multi_turn = True

    def start(self, data):
        super().start(data)
        self.data.non_tensor_batch["step_reward"] = np.zeros(1, dtype=np.float32)
        self._maybe_rebuild_initial_input_ids(self.data)

    def accept_output(self, output, env, tools):
        super().accept_output(output, env, tools)
        length = int(output.batch["response_mask"][0].sum().item())
        self.response_ids = output.batch["responses"][0][:length].tolist()
        content = self.tokenizer.decode(self.response_ids, skip_special_tokens=True)
        # Legacy task environments expose their parser on the env adapter.
        # Parsing is invoked here, never by the worker's lifecycle loop.
        for index, tool in enumerate(tools):
            action = tool.extract_tool_call(content)
            if action is not None:
                return AgentAction("tool", action, index)
        action = env.extract_action(content) if env is not None else None
        if action is not None:
            return AgentAction("environment", action)
        if "<answer>" in content and "</answer>" in content:
            return AgentAction("finish")
        return AgentAction("feedback")

    def observe(self, observation, reward, info, *, continue_episode, action):
        self.output.non_tensor_batch["step_reward"] += np.float32(reward)
        if continue_episode:
            if action.kind == "feedback":
                observation = (
                    "Please either call a tool with <tool_call>...</tool_call> "
                    "or provide the final result with <answer>...</answer>."
                )
            self._add_observation(self.output, self.response_ids, observation, reward, info)
            self.data = self.output

    def finish(self, step_reward, eval_reward):
        self.output.non_tensor_batch.update({
            "env_reward": np.array([step_reward + eval_reward], dtype=np.float32),
            "eval_reward": np.array([eval_reward], dtype=np.float32),
            "step_reward_sum": np.array([step_reward], dtype=np.float32),
        })
        return self.output

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
        if multi_modal_data[0] is None:
            multi_modal_data[0] = {}
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
                "MultiTurnAgent: dropped unsafe image observation (type=%s).",
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
                "MultiTurnAgent: image-side tensor extension failed (%s); "
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
        merge_size = self.processor.image_processor.merge_size if self.processor is not None else 1

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

        prompt_len = gen_output.batch["prompts"].size(-1)
        new_input_ids = torch.cat([gen_output.batch["input_ids"][:, :prompt_len], extra_t], dim=-1)
        new_attn = torch.cat(
            [
                gen_output.batch["attention_mask"][:, :prompt_len],
                torch.ones_like(extra_t, dtype=gen_output.batch["attention_mask"].dtype),
            ],
            dim=-1,
        )

        pos = gen_output.batch["position_ids"][..., :prompt_len]
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

