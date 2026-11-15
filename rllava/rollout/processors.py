from __future__ import annotations

import math
import random
from collections import defaultdict
import numpy as np
import torch
from tensordict import TensorDict

from rllava.data.protocol import DataProto
from rllava.utils import torch_functional as VF
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllava.controller.config import RLConfig



class RolloutProcessor:

    def __init__(self, config: RLConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def pre_process(self, prompts: DataProto) -> DataProto:
        return prompts

    def post_process(self, generated: DataProto, prompts: DataProto) -> DataProto:
        return generated

    def post_rollout(
        self,
        batch: DataProto,
        reward,
        rollout,
    ) -> DataProto:
        return batch

    def _trim_right_pad(self, tokens: torch.Tensor, pad_token_id: int) -> list[int]:
        non_pad = torch.nonzero(tokens != pad_token_id, as_tuple=False)
        if len(non_pad) == 0:
            return []
        last = non_pad[-1].item()
        return tokens[: last + 1].tolist()

    def _expand_to_match(self, base_list: list, target_len: int) -> list | None:
        if len(base_list) == target_len:
            return base_list
        if len(base_list) == 0 or target_len % len(base_list) != 0:
            return None
        repeat_factor = target_len // len(base_list)
        return [item for item in base_list for _ in range(repeat_factor)]


class PrefixRolloutProcessor(RolloutProcessor):

    def pre_process(self, prompts: DataProto) -> DataProto:
        if "tgt_input_ids" not in prompts.batch.keys():
            return prompts
        if "raw_prompt_ids" not in prompts.non_tensor_batch:
            return prompts

        n = self.config.rollout.n
        if n > 1:
            prompts = prompts.repeat(repeat_times=n, interleave=True)
            prompts.meta_info["orig_n"] = n
            prompts.meta_info["n"] = 1

        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        target_ids = prompts.batch["tgt_input_ids"]
        prefix_tokens = [self._trim_right_pad(target_ids[i], pad_token_id) for i in range(target_ids.size(0))]
        prefix_tokens = [tokens + [eos_token_id] if len(tokens) > 0 else tokens for tokens in prefix_tokens]
        prefix_ratios = self._build_prefix_ratios(prefix_tokens, prompts)
        prompts.meta_info["prefix_ratios"] = prefix_ratios

        raw_prompt_ids = prompts.non_tensor_batch["raw_prompt_ids"]
        # sequences_str = tokenizer.decode(raw_prompt_ids[0])
        merged_prompt_ids = []
        for base_ids, prefix_ids, ratio in zip(raw_prompt_ids, prefix_tokens, prefix_ratios):
            prefix_len = int(len(prefix_ids) * ratio)
            merged_prompt_ids.append(list(base_ids) + prefix_ids[:prefix_len])
        prompts.non_tensor_batch["raw_prompt_ids"] = merged_prompt_ids
        return prompts

    def post_process(self, generated: DataProto, prompts: DataProto) -> DataProto:
        if "tgt_input_ids" not in prompts.batch.keys():
            return generated

        response_ids = generated.batch["responses"]
        prompt_ids = generated.batch["prompts"]
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        response_tokens = [self._trim_right_pad(resp, pad_token_id) for resp in response_ids]
        target_ids = prompts.batch["tgt_input_ids"]
        prefix_tokens = [self._trim_right_pad(target_ids[i], pad_token_id) for i in range(target_ids.size(0))]
        prefix_tokens = [tokens + [eos_token_id] if len(tokens) > 0 else tokens for tokens in prefix_tokens]

        prefix_tokens = self._expand_to_match(prefix_tokens, len(response_tokens))
        if prefix_tokens is None:
            return generated

        prefix_ratios = self._build_prefix_ratios(prefix_tokens, prompts)
        prefix_ratios = self._expand_to_match(prefix_ratios, len(prefix_tokens))
        if prefix_ratios is None:
            return generated

        prefix_mask = torch.zeros(
            [len(prefix_tokens), self.config.rollout.response_length],
            dtype=torch.bool,
            device=response_ids.device,
        )

        merged_responses = []
        for i, (prefix_ids, response_ids_list) in enumerate(zip(prefix_tokens, response_tokens)):
            ratio = prefix_ratios[i]
            prefix_len = min(int(len(prefix_ids) * ratio), self.config.rollout.response_length)
            if ratio < 1.0:
                response_len = int(len(response_ids_list))
            else:
                response_len = 0

            concat = prefix_ids[:prefix_len] + response_ids_list[:response_len]
            merged_responses.append(concat)
            if prefix_len > 0:
                prefix_mask[i, :prefix_len] = True

        new_response_ids = VF.pad_2d_list_to_length(
            merged_responses, pad_token_id, max_length=self.config.rollout.response_length
        ).to(response_ids.device)[:, :self.config.rollout.response_length]

        response_length = new_response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=prompt_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(prompt_ids.size(0), -1)
        prompt_position_ids = generated.batch["position_ids"][..., : prompt_ids.size(1)]
        # prompt_position_ids = prompts.batch["position_ids"][..., : prompt_ids.size(1)]
        if prompt_position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(prompt_ids.size(0), 1, -1).expand(
                prompt_ids.size(0), prompt_position_ids.size(1), -1
            )
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        # response_length = response_ids.size(1)
        # delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        # delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        # if position_ids.dim() == 3:  # qwen2vl mrope
        #     delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # response_position_ids = position_ids[..., -1:] + delta_position_id
        # position_ids = torch.cat([position_ids, response_position_ids], dim=-1)


        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        response_mask = VF.get_response_mask(
            response_ids=new_response_ids, eos_token_id=eos_token_id, dtype=generated.batch["attention_mask"].dtype
        )
        # response_mask_l = response_mask.sum(dim=-1)
        prompt_attention_mask = generated.batch["attention_mask"][:, : prompt_ids.size(1)]
        attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        generated.batch["responses"] = new_response_ids
        generated.batch["input_ids"] = torch.cat([prompt_ids, new_response_ids], dim=-1)
        generated.batch["response_mask"] = response_mask
        generated.batch["attention_mask"] = attention_mask
        generated.batch["position_ids"] = position_ids
        generated.batch["prefix_mask"] = prefix_mask
        generated.meta_info["prefix_ratios"] = prefix_ratios
        return generated

    def _sample_prefix_ratio(self, prompts) -> float:
        if self.config.algorithm.prefix_strategy == "hint":
            hint_steps = max(int(self.config.algorithm.hint_steps), 0)
            min_ratio = self.config.algorithm.min_prefix_ratio
            max_ratio_base = self.config.algorithm.max_prefix_ratio
            if hint_steps <= 0:
                max_ratio = max_ratio_base
            else:
                step = int(prompts.meta_info.get("global_step", 0))
                step = min(max(step, 0), hint_steps)
                max_ratio = 0.5 * (1 + math.cos(math.pi * step / hint_steps)) * max_ratio_base
            max_ratio = max(max_ratio, min_ratio)
            return random.uniform(min_ratio, max_ratio)
        return random.uniform(self.config.algorithm.min_prefix_ratio, self.config.algorithm.max_prefix_ratio)

    def _build_prefix_ratios(self, prefix_tokens: list, prompts) -> list:
        cached = prompts.meta_info.get("prefix_ratios")
        if cached is not None:
            return list(cached)

        total_samples = len(prefix_tokens)
        group_n = int(prompts.meta_info.get("orig_n", 1))
        if group_n <= 0:
            group_n = 1

        if group_n > 1 and total_samples % group_n == 0:
            prompt_count = total_samples // group_n
        else:
            prompt_count = None

        if self.config.algorithm.n_prefix is None:
            n_prefix = -1
        else:
            n_prefix = int(self.config.algorithm.n_prefix)

        if prompt_count is None:
            # Fallback: sample per token list entry
            return [self._sample_prefix_ratio(prompts) for _ in range(total_samples)]

        prefix_ratios = []
        for _ in range(prompt_count):
            share_across = self.config.algorithm.prefix_share_across_samples
            shared_ratio = self._sample_prefix_ratio(prompts) if share_across else None

            def _sample(count: int) -> list[float]:
                if count <= 0:
                    return []
                if share_across:
                    return [shared_ratio] * count
                return [self._sample_prefix_ratio(prompts) for _ in range(count)]

            if n_prefix >= 0:
                prefix_ratios.extend(_sample(n_prefix))
                prefix_ratios.extend([0.0] * (group_n - n_prefix))
            else:
                prefix_ratios.extend(_sample(group_n))

        return prefix_ratios

class HPTBatchRolloutProcessor(RolloutProcessor):

    def post_rollout(
        self,
        batch: DataProto,
        reward,
        rollout,
    ) -> DataProto:
        if self.config.algorithm.unify_strategy == "none":
            return batch
        if "token_level_scores" not in batch.batch or "uid" not in batch.non_tensor_batch:
            return batch
        if "tgt_input_ids" not in batch.batch:
            return batch

        threshold = self.config.algorithm.success_reward_threshold
        switch_gate = self.config.algorithm.switch_gate

        if "prefix_mask" not in batch.batch:
            batch.batch["prefix_mask"] = torch.zeros(
                (batch.batch["input_ids"].size(0), self.config.rollout.response_length),
                dtype=torch.bool,
                device=batch.batch["input_ids"].device,
            )

        scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid_to_indices[uid].append(idx)

        replace_indices: list[int] = []
        mask_indices: list[int] = []  # surplus on-policy samples to mask out
        for uid, idxs in uid_to_indices.items():
            uid_scores = scores[idxs]
            on_solve_num = int(np.sum(uid_scores >= threshold))
            if on_solve_num <= switch_gate:
                replace_indices.append(idxs[0])
                mask_indices.extend(idxs[1:])  # remaining on-policy → mask out
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> replace_targets', len(replace_indices),
        #       'mask_on_policy', len(mask_indices), '>>>>>>>>')
        if not replace_indices:
            return batch

        off_batch = self._build_off_batch(batch, replace_indices, rollout)
        if off_batch is None:
            return batch
        reward_tensor, reward_metrics = reward.compute_rewards(off_batch)
        off_batch.batch["token_level_scores"] = reward_tensor
        for k, v in reward_metrics.items():
            off_batch.batch[f"reward_{k}"] = torch.tensor(
                v, dtype=torch.float32, device=reward_tensor.device
            )

        replace_index_array = np.array(replace_indices, dtype=int)
        for key in off_batch.batch.keys():
            if key == "batch_size" or key not in batch.batch:
                continue
            batch.batch[key][replace_index_array] = off_batch.batch[key]
        for key in off_batch.non_tensor_batch.keys():
            if key not in batch.non_tensor_batch:
                continue
            batch.non_tensor_batch[key][replace_index_array] = off_batch.non_tensor_batch[key]

        # Zero out response_mask for surplus on-policy samples of replaced UIDs
        # so they contribute nothing to loss / advantage, while keeping batch
        # size AND attention_mask unchanged across GPUs.
        # NOTE: we intentionally do NOT touch attention_mask here -- zeroing
        # it would change effective sequence lengths seen by dynamic_batching
        # and padding_free, causing wildly uneven micro-batch token counts
        # and CUDA memory fragmentation that leads to OOM.
        if mask_indices:
            mask_index_array = np.array(mask_indices, dtype=int)
            batch.batch["response_mask"][mask_index_array] = 0

        return batch

    def _build_off_batch(
        self, batch: DataProto, indices: list[int], rollout
    ) -> DataProto | None:
        if not indices:
            return None
        if "tgt_input_ids" not in batch.batch.keys():
            return None
        if "prompts" not in batch.batch.keys():
            return None

        index_array = np.array(indices, dtype=int)
        input_ids = batch.batch["prompts"][index_array]
        prompt_len = input_ids.size(1)

        attention_mask = batch.batch["attention_mask"][index_array][:, :prompt_len]
        position_ids = batch.batch["position_ids"][index_array][..., :prompt_len]
        tgt_input_ids = batch.batch["tgt_input_ids"][index_array]

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        batch_size = tgt_input_ids.size(0)

        target_tokens = [self._trim_right_pad(tgt_input_ids[i], pad_id) for i in range(batch_size)]
        target_tokens = [t + [eos_id] if len(t) > 0 else t for t in target_tokens]

        prefix_mask = torch.zeros(
            [batch_size, rollout.config.response_length],
            dtype=torch.bool,
            device=input_ids.device,
        )
        response_tokens = []
        for i, tgt in enumerate(target_tokens):
            response_tokens.append(tgt[: rollout.config.response_length])
            prefix_len = min(len(tgt), rollout.config.response_length)
            if prefix_len > 0:
                prefix_mask[i, :prefix_len] = True

        response_length = rollout.config.response_length
        response_ids = VF.pad_2d_list_to_length(
            response_tokens, pad_id, max_length=response_length
        ).to(input_ids.device)[:, :response_length]

        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, position_ids.size(1), -1
            )

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        batch_dict = {
            "prompts": input_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "tgt_input_ids": tgt_input_ids,
            "prefix_mask": prefix_mask,
        }

        non_tensor_batch = {
            key: value[index_array] for key, value in batch.non_tensor_batch.items()
        }

        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        return DataProto.from_dict(
            tensors=batch_dict,
            non_tensors=non_tensor_batch,
            meta_info=meta_info,
        )
