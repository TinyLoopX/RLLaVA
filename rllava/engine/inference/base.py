import os
import re
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from torch.distributed.tensor import DTensor
from transformers import PreTrainedTokenizer, ProcessorMixin, PreTrainedModel
from rllava.data.protocol import DataProto
from rllava.data.data_utils import process_image, process_video


if TYPE_CHECKING:
    # Import for type checking only to avoid runtime circular import
    from rllava.ppo.config import RolloutConfig


def response_mask_from_lengths(response_ids, lengths, eos_token_id, *, dtype, ignore_eos=False):
    """Include the first EOS, but never include padding, even when pad == EOS."""
    lengths = torch.as_tensor(lengths, device=response_ids.device)
    mask = torch.arange(response_ids.shape[1], device=response_ids.device)[None, :] < lengths[:, None]
    if eos_token_id is not None and not ignore_eos:
        eos_ids = torch.as_tensor(eos_token_id, device=response_ids.device).reshape(-1)
        is_eos = (response_ids[..., None] == eos_ids).any(-1) & mask
        mask &= (is_eos.long().cumsum(-1) - is_eos.long()) == 0
    return mask.to(dtype)


def pack_rollout_log_probs(token_ids, token_log_probs, response_mask, temperature):
    """Pad only after checking each generated token has its own probability."""
    if len(token_ids) != response_mask.shape[0] or len(token_log_probs) != len(token_ids):
        raise ValueError("Rollout response and log-probability batch sizes differ.")
    log_probs = torch.zeros(response_mask.shape, dtype=torch.float32, device=response_mask.device)
    for i, (ids, values) in enumerate(zip(token_ids, token_log_probs)):
        if len(ids) != len(values):
            raise ValueError(f"Token and log-probability counts differ for response {i}.")
        log_probs[i, :len(values)] = torch.as_tensor(values, dtype=torch.float32, device=log_probs.device)
    log_probs = log_probs.detach().masked_fill(~response_mask.bool(), 0.0)
    if temperature == 0:
        # Backend softmax scores are not the probability of a greedy action.
        # The deterministic policy assigns probability 1 to the selected token.
        log_probs.zero_()
    return {
        "rollout_log_probs": log_probs,
        "rollout_temperature": torch.full((len(token_ids),), temperature, dtype=torch.float32, device=log_probs.device),
    }


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None
    
def _process_multi_modal_data(
    multi_modal_data: Dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float, processor: Optional[ProcessorMixin] = None
) -> Dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data and multi_modal_data["images"] is not None:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels, processor))

    if "videos" in multi_modal_data and multi_modal_data["videos"] is not None:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None

class InferenceEngine():
    """Base class for all inference engines.
    
    This class defines the common interface that all rollout engines must implement.
    Engines are responsible for managing the rollout process for PPO training.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        self.model = model
        # Keep tokenizer and processor references for engines needing decoding or multimodal preprocessing
        self.tokenizer = tokenizer
        self.processor = processor
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if (torch.distributed.is_initialized() and config.tensor_parallel_size > torch.distributed.get_world_size()):
            raise ValueError("Tensor parallelism size should be less than world size.")

        self.engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            self.engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                self.engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.tp_size = self.config.tensor_parallel_size
        self.dp_size = self.world_size // self.tp_size if self.tp_size > 0 else 1
        if self.world_size % self.tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {self.tp_size}.")

        # Avoid creating distributed device mesh here since engine runs only on rank0.
        # Creating a mesh would require collectives across all ranks and can deadlock.
        self.device_mesh = None

    def generate(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError()

    async def agenerate(self, prompts: DataProto) -> DataProto:
        """Asynchronously generate a response for the given request."""
        raise NotImplementedError()

    def load(self, model):
        """Load the engine and sync the weights."""
        raise NotImplementedError()

    def offload(self):
        """Offload the engine."""
        raise NotImplementedError()        

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights
