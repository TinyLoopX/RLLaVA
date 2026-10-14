import os
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, TYPE_CHECKING
from collections import defaultdict
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoModelForVision2Seq, AutoModelForCausalLM, AutoConfig, GenerationConfig
from .base import InferenceEngine, pack_rollout_log_probs, response_mask_from_lengths
from .. import register_engine
from rllava.utils import torch_functional as VF
from torch.distributed.tensor import DTensor
from tensordict import TensorDict
from rllava.data.protocol import DataProto
from rllava.data.data_utils import process_image, process_video
from rllava.utils.logger.aggregate_logger import print_rank_0
from rllava.utils.device import get_device_id



if TYPE_CHECKING:
    from rllava.ppo.config import RolloutConfig


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


@register_engine("hf")
class HFEngine(InferenceEngine):
    def __init__(
        self,
        model_name_or_path: str,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__(model_name_or_path, config, tokenizer, processor)
        self.model = None
        self.loaded = False

        self.model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            attn_implementation='flash_attention_2'
        )
        print_rank_0(f"Inference Model config: {self.model_config}")

        if type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM
        
        self.model = model_class.from_pretrained(
                model_name_or_path,
                config=self.model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2',
                trust_remote_code=self.config.trust_remote_code,
            )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _process_multi_modal_inputs(self, data: DataProto):
        """
        Process multi-modal data from dataloader format to model input format.
        Converts multi_modal_data (raw images/videos) to multi_modal_inputs (processed tensors).
        """
        if "multi_modal_data" not in data.non_tensor_batch:
            return
        
        if "multi_modal_inputs" in data.non_tensor_batch:
            # Already processed
            return
        
        else:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            
            for multi_modal_data in data.non_tensor_batch["multi_modal_data"]:
                images, videos = [], []
                if "images" in multi_modal_data:
                    for image in multi_modal_data["images"]:
                        images.append(process_image(image, min_pixels, max_pixels, self.processor))
                
                if "videos" in multi_modal_data:
                    for video in multi_modal_data["videos"]:
                        videos.append(process_video(video, min_pixels, max_pixels, video_fps))
                
                if len(images) != 0:      
                    multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                elif len(videos) != 0:
                    multi_modal_inputs = dict(
                        self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                    )
                else:
                    multi_modal_inputs = {}
                
                batch_multi_modal_inputs.append(multi_modal_inputs)
            
            data.non_tensor_batch["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        if temperature is None or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("HF generation temperature must be finite and non-negative (0 for greedy).")
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", getattr(self.config, "top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", getattr(self.config, "top_k", 0)))  

        input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = input_ids.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        
        # When temperature is 0.0, use greedy decoding (do_sample=False)
        # Otherwise use sampling (do_sample=True)
        do_sample = temperature > 0.0
        
        self.generation_config = GenerationConfig(
            max_new_tokens=response_length,
            do_sample=do_sample,
            num_beams=1,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            temperature=temperature if do_sample else None, 
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        # Prepare generation arguments based on task type
        prompt_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_scores": True,  # required to recover probabilities of sampled tokens
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        # Merge multi_modal_inputs into prompt_inputs
        if "multi_modal_inputs" in prompts.non_tensor_batch:
            multi_modal_inputs = defaultdict(list)
            for input_dict in prompts.non_tensor_batch["multi_modal_inputs"]:
                for key, value in input_dict.items():
                    multi_modal_inputs[key].append(value)
            
            for key, value in multi_modal_inputs.items():
                if len(value) != 0:
                    concatenated = torch.cat(value, dim=0)
                    # Move to the same device as input_ids
                    prompt_inputs[key] = concatenated.to(device=input_ids.device)
                else:
                    prompt_inputs[key] = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model.generate(**prompt_inputs, 
                                        generation_config=self.generation_config)
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences
        generated_length = seq.shape[1] - prompt_length
        token_ids = seq[:, prompt_length:].tolist()
        # Scores include the actual sampling processors and temperature.
        # Gather one step at a time to avoid another B x R x vocabulary tensor.
        selected = [
            scores.float().log_softmax(-1).gather(1, seq[:, prompt_length + i, None]).squeeze(1)
            for i, scores in enumerate(output.scores)
        ]
        token_log_probs = torch.stack(selected, dim=1).tolist() if selected else [[] for _ in token_ids]

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        
        # Handle Qwen2-VL MRoPE 3D position_ids
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(generated_batch_size, 1, -1).expand(generated_batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_mask = response_mask_from_lengths(
            response, [generated_length] * generated_batch_size, eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        batch.update(pack_rollout_log_probs(token_ids, token_log_probs, response_mask, temperature))
        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        if "multi_modal_data" in prompts.non_tensor_batch:
            non_tensor_batch = {"multi_modal_data": prompts.non_tensor_batch["multi_modal_data"]}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def generate(self, prompts: DataProto) -> DataProto:
        self._process_multi_modal_inputs(prompts)
        prompts = prompts.to(torch.cuda.current_device())
        num_return_sequences = prompts.meta_info.get("n", getattr(self.config, "n", 1))
        prompts = prompts.repeat(repeat_times=num_return_sequences, interleave=True)
        
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // getattr(self.config, "micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        
        # Add progress bar for batch processing
        batch_prompts_iter = tqdm(
            batch_prompts, 
            desc="Generating batches", 
            disable=False,
            total=len(batch_prompts)
        )
        
        output = [self._generate_minibatch(p) for p in batch_prompts_iter]
        output = DataProto.concat(output)
        return output

    def update_weights(self, model):
        weights = model.state_dict()
        weights_iter = self._make_weight_iterator(weights)
        
        processed_weights = {name: tensor for name, tensor in weights_iter}
        
        missing_keys, unexpected_keys = self.model.load_state_dict(processed_weights, strict=False)
        
        if missing_keys:
            print_rank_0(f"Warning: Missing keys when loading weights: {missing_keys[:5]}...")  
        if unexpected_keys:
            print_rank_0(f"Warning: Unexpected keys when loading weights: {unexpected_keys[:5]}...")  
        
        # 将模型移到 CUDA 设备
        self.model = self.model.to(torch.cuda.current_device())
        
        torch.cuda.empty_cache()

    def load(self, model):
        torch.cuda.empty_cache()
        assert self.loaded is False, "hf engine has already been loaded"

        self.update_weights(model)
        self.loaded = True

    def offload(self):
        assert self.loaded is True, "hf engine has not been loaded"
        
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        self.loaded = False

    def _make_weight_iterator(self, weights: Dict[str, torch.Tensor]):
        """Create an iterator over model weights.
        
        Args:
            actor_weights: Model weights dictionary
            
        Returns:
            Iterator over (name, tensor) pairs
        """
        device = get_device_id() 
        for name in sorted(weights.keys()):
            tensor = weights[name]
            # Handle DTensor for distributed training
            if hasattr(tensor, 'full_tensor'):
                yield name, tensor.to(device, non_blocking=True).full_tensor() if (self.world_size != 1 or isinstance(tensor, DTensor)) else tensor
            else:
                yield name, tensor 
