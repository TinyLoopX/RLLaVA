"""Context builder for multi-turn token-first state management.

Maintains the exact prompt token chain used by vLLM generation and by the
training-side forward pass. Assistant turns are appended by exact sampled token
IDs, while external turns (tool / environment) are appended as template deltas.

The only remaining full re-encode path is turn-0 initial observation injection,
which uses a one-shot dataset-style re-encode from preserved
``initial_prompt_text`` to compute multimodal expansion and mRoPE position IDs.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rllava.data.data_utils import process_image, process_video
from rllava.data.protocol import DataProto
import rllava.utils.torch_functional as VF


class ContextBuilder:
    """Manage per-session message/image state and materialize model inputs."""

    def __init__(self, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        self._messages: List[Dict[str, Any]] = []
        self._images: list = []
        self._meta_info: dict = {}
        self.raw_prompt_ids: List[int] = []
        self.input_ids: List[int] = []
        self.response_mask: List[int] = []
        self.multi_modal_data: Dict[str, list] = {}
        self._derived_dirty: bool = False

    # ================================================================
    # Session initialisation
    # ================================================================

    def start_from_data(self, data: DataProto) -> DataProto:
        """Initialise context from exact dataset-provided token IDs.

        Unlike the previous message-first initialisation, this method does
        **not** decode and re-encode ``raw_prompt_ids``. The dataset-provided
        IDs become the session's initial token truth, completely bypassing the
        tokeniser encode / decode asymmetry at turn 0.

        Call :meth:`inject_initial_observation` afterwards when the
        environment provides an initial screenshot at reset time.
        """
        self._meta_info = dict(data.meta_info) if data.meta_info else {}

        # ---- exact engine-side IDs ----
        raw = data.non_tensor_batch.get("raw_prompt_ids")
        self.raw_prompt_ids = (
            list(raw[0]) if raw is not None and len(raw) > 0 else []
        )

        # ---- exact train-side IDs (strip left-padding) ----
        input_ids_t = data.batch["input_ids"][0]
        attn_mask_t = data.batch["attention_mask"][0]
        valid = attn_mask_t.bool()
        self.input_ids = input_ids_t[valid].tolist()
        self.response_mask = [0] * len(self.input_ids)

        # ---- multimodal state ----
        existing_mmd = data.non_tensor_batch.get("multi_modal_data")
        if (
            existing_mmd is not None
            and len(existing_mmd) > 0
            and isinstance(existing_mmd[0], dict)
        ):
            self.multi_modal_data = {
                k: list(v) for k, v in existing_mmd[0].items()
            }
        else:
            self.multi_modal_data = {}

        self._derived_dirty = False

        # ---- semantic seed for turn-0 multimodal injection ----
        self._images = list(
            self.multi_modal_data.get("images", [])
        )
        self._messages = []
        initial_prompt_text = str(data.non_tensor_batch.get("initial_prompt_text")[0])
        self._messages.append(
            {"role": "user", "content": initial_prompt_text.strip()}
        )

        return data

    def inject_initial_observation(
        self, data: DataProto, image,
    ) -> DataProto:
        """Append an env-reset screenshot to the initial context.

        Processes the image, adds an observation user message to the
        semantic history, and re-encodes the full initial prompt through
        the processor so that image tokens and mRoPE position IDs are
        computed correctly.

        Must be called right after :meth:`start_from_data`.
        """
        initial_prompt_text = str(data.non_tensor_batch.pop("initial_prompt_text", None)[0])
        
        processed_image = process_image(
            image,
            self._meta_info.get("min_pixels"),
            self._meta_info.get("max_pixels"),
            self.processor,
        )

        content: List[Dict[str, Any]] = []
        if initial_prompt_text:
            content.append({"type": "text", "text": initial_prompt_text})
        content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]
        self._messages = messages

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        model_inputs = self.processor(
            images=[processed_image],
            text=[prompt],
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

        rope_mm_inputs = self._build_rope_mm_inputs()
        position_ids = self._get_rope_position_ids(
            input_ids, attention_mask, rope_mm_inputs,
        )

        max_prompt_length = int(data.batch["input_ids"].size(-1))
        """input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )"""
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[:max_prompt_length]

        batch_input_ids = input_ids.unsqueeze(0)
        batch_attention_mask = attention_mask.unsqueeze(0)
        batch_position_ids = position_ids.unsqueeze(0)

        data.batch["input_ids"] = batch_input_ids
        data.batch["attention_mask"] = batch_attention_mask
        data.batch["position_ids"] = batch_position_ids
        data.non_tensor_batch["raw_prompt_ids"] = np.array(
            [raw_prompt_ids], dtype=object,
        )
        data.non_tensor_batch["multi_modal_data"] = [
            {"images": [image]}
        ]
        self._images = [image]

        valid = batch_attention_mask[0].bool()
        self.input_ids = batch_input_ids[0][valid].tolist()
        self.raw_prompt_ids = list(raw_prompt_ids)
        self.response_mask = [0] * len(self.input_ids)
        self.multi_modal_data = {"images": [image]}

        self._derived_dirty = False
        return data


    # ================================================================
    # Position ID helpers
    # ================================================================

    def _get_rope_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position IDs for a single (unpadded) sequence.

        For Qwen2-VL / Qwen3-VL returns ``(n_rope_channels, seq_len)``
        (e.g. shape ``(4, L)`` = text + 3 vision channels).
        For other models returns ``(seq_len,)``.
        """
        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from rllava.model.patch.qwen3_vl import get_rope_index
            else:
                from rllava.model.patch.qwen2_vl import get_rope_index

            mm = mm_inputs or {}
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=mm.get("image_grid_thw", None),
                video_grid_thw=mm.get("video_grid_thw", None),
                second_per_grid_ts=mm.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            return torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            return torch.clip(
                attention_mask.cumsum(dim=0) - 1, min=0, max=None,
            )

    def _build_rope_mm_inputs(self) -> Dict[str, Any]:
        """Build only the multimodal metadata required by mRoPE.

        This keeps ``multi_modal_data`` as the single source of truth and
        avoids relying on a separately accumulated ``self.multi_modal_inputs``
        for final full-trajectory position id computation.
        """
        if (
            self.processor is None
            or not hasattr(self.processor, "image_processor")
            or not self.multi_modal_data
        ):
            return {}

        min_pixels = self._meta_info.get("min_pixels")
        max_pixels = self._meta_info.get("max_pixels")
        video_fps = self._meta_info.get("video_fps")

        images = [
            process_image(image, min_pixels, max_pixels, self.processor)
            for image in (self.multi_modal_data.get("images") or [])
        ]
        videos = [
            process_video(video, min_pixels, max_pixels, video_fps)
            for video in (self.multi_modal_data.get("videos") or [])
        ]

        if not images and not videos:
            return {}

        if images and videos:
            mm_inputs = dict(
                self.processor.image_processor(
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                )
            )
        elif images:
            mm_inputs = dict(
                self.processor.image_processor(
                    images=images,
                    return_tensors="pt",
                )
            )
        else:
            mm_inputs = dict(
                self.processor.image_processor(
                    images=None,
                    videos=videos,
                    return_tensors="pt",
                )
            )

        return {
            key: value
            for key, value in mm_inputs.items()
            if key in {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"}
        }

    # ================================================================
    # Turn-level state management
    # ================================================================

    def append_assistant_generation(
        self,
        gen_output: DataProto,
        decoded_text: Optional[str] = None,
    ) -> str:
        """Exact-append assistant response token IDs to the token chains.

        Uses the raw token IDs from ``gen_output.batch["responses"]``
        
        data keys:
        dict_keys(['input_ids', 'attention_mask', 'position_ids'])
        dict_keys(['raw_prompt_ids', 'task_config', 'initial_prompt_text'])
        dict_keys(['min_pixels', 'max_pixels', 'video_fps'])
        
        gen_output keys:
        dict_keys(['prompts', 'responses', 'input_ids', 'attention_mask', 'response_mask', 'position_ids'])),
        dict_keys(['multi_modal_data']),
        dict_keys(['min_pixels', 'max_pixels', 'video_fps', 'n', 'eos_token_id', 'pad_token_id'])

        Returns:
            Decoded assistant text (for parser / logging).
        """
        response_ids = gen_output.batch["responses"]
        response_length = torch.sum(gen_output.batch["response_mask"], dim=-1)
        cur_response_length = int(response_length[0].item())  # avoid tensor indexing error
        valid_response_ids = response_ids[0][:cur_response_length].tolist()
        self.raw_prompt_ids.extend(valid_response_ids)
        self.input_ids.extend(valid_response_ids)
        self.response_mask.extend([1] * len(valid_response_ids))

        response_str = self.tokenizer.decode(
            valid_response_ids, skip_special_tokens=True
        )

        # ---- keep semantic history aligned with token-first state ----
        self._messages.append({"role": "assistant", "content": response_str})
        self._derived_dirty = True

        return response_str

    def materialize_generation_inputs(self, data: DataProto) -> None:
        """Write current state into *data* for the next generation call.

        Writes the exact token chains directly — no re-encode.
        """
        if not self._derived_dirty:
            return

        seq_len = len(self.input_ids)
        data.batch["input_ids"] = torch.tensor(
            [self.input_ids], dtype=torch.long,
        )
        data.batch["attention_mask"] = torch.ones(
            1, seq_len, dtype=torch.long,
        )
        data.non_tensor_batch["raw_prompt_ids"] = np.array(
            [self.raw_prompt_ids], dtype=object,
        )
        if self.multi_modal_data:
            data.non_tensor_batch["multi_modal_data"] = [
                self.multi_modal_data
            ]
        else:
            data.non_tensor_batch.pop("multi_modal_data", None)

        self._derived_dirty = False

    def compute_position_ids(self) -> torch.Tensor:
        """Compute position IDs for the complete multi-turn trajectory.

        Called once after all turns are finished.  For Qwen2-VL / Qwen3-VL
        models this produces 3D mRoPE position IDs using the accumulated
        ``multi_modal_data``; for other models it falls back to a simple
        cumulative position sequence.

        Returns:
            position_ids tensor with batch dim 1, ready for training.
        """
        seq_len = len(self.input_ids)
        input_ids_t = torch.tensor(self.input_ids, dtype=torch.long)
        attn_mask_t = torch.ones(seq_len, dtype=torch.long)
        rope_mm_inputs = self._build_rope_mm_inputs()

        position_ids = self._get_rope_position_ids(
            input_ids_t, attn_mask_t, rope_mm_inputs,
        )
        return position_ids.unsqueeze(0)

    def _compute_template_delta(
        self,
        message: Dict[str, Any],
        processed_image=None,
    ) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """Extract new-turn tokens via synthetic base differencing.

        Encodes a minimal [user, assistant] base and [base + message] full
        conversation, returns full[base_len:] as delta.  Automatically strips
        any template preamble (system block, BOS, etc.).

        Returns:
            (engine_delta, train_delta, mm_inputs):
            engine/train deltas as int lists, mm_inputs contains
            image_grid_thw etc. from processor (empty dict if text-only).
        """
        template_fn = (
            self.processor.apply_chat_template
            if self.processor is not None
            else self.tokenizer.apply_chat_template
        )

        base_msgs = [
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
        ]
        full_msgs = base_msgs + [message]

        base_text = template_fn(
            base_msgs, add_generation_prompt=False, tokenize=False,
        )
        full_text = template_fn(
            full_msgs, add_generation_prompt=True, tokenize=False,
        )
        base_len = len(
            self.tokenizer.encode(base_text, add_special_tokens=False)
        )

        engine_delta = self.tokenizer.encode(
            full_text, add_special_tokens=False,
        )[base_len:]

        if processed_image is not None and self.processor is not None:
            model_inputs = self.processor(
                text=[full_text], images=[processed_image],
                add_special_tokens=False, return_tensors="pt",
            )
            base_inputs = self.processor(
                text=[base_text],
                add_special_tokens=False,
                return_tensors="pt",
            )
            train_base_len = len(base_inputs["input_ids"][0])
            train_delta = model_inputs["input_ids"][0][train_base_len:].tolist()

        else:
            train_delta = list(engine_delta)

        return engine_delta, train_delta

    def append_external_delta(
        self,
        content: str,
        role: str = "user",
        image=None,
        tool_name: Optional[str] = None,
    ) -> None:
        """Delta-append an external turn (tool / env observation) to the token chains.

        For text-only observations the delta is computed purely from
        template token arithmetic.  For multimodal observations a single
        processor call is made to expand image tokens.
        """
        # ---- process image ----
        processed_image = None
        if image is not None:
            processed_image = process_image(
                image,
                self._meta_info.get("min_pixels"),
                self._meta_info.get("max_pixels"),
                self.processor,
            )
            self._images.append(image)
            if "images" not in self.multi_modal_data:
                self.multi_modal_data["images"] = []
            self.multi_modal_data["images"].append(image)

        # ---- build message content ----
        content_parts: list = []
        if processed_image is not None:
            content_parts.append({"type": "image"})
        
        if role == "tool" and tool_name:
            content_parts.append({"type": "text", "text": f"[Tool: {tool_name}]\n{content}"})
        elif role == "environment":
            content_parts.append({"type": "text", "text": f"[Environment: {content}"})
        else:
            content_parts.append({"type": "text", "text": content})
        template_message = {"role": "user", "content": content_parts or content}

        # ---- compute delta via synthetic base diff ----
        engine_delta, train_delta = self._compute_template_delta(
            template_message, processed_image,
        )
        self.raw_prompt_ids.extend(engine_delta)
        self.input_ids.extend(train_delta)
        self.response_mask.extend([0] * len(train_delta))


        # ---- keep semantic history aligned (preserve real role) ----
        if role == "tool":
            self._messages.append(
                {"role": "tool", "name": tool_name or "tool", "content": content}
            )
        elif role == "environment":
            self._messages.append({"role": "environment", "content": content})
        else:
            self._messages.append({"role": "user", "content": content})

        self._derived_dirty = True

