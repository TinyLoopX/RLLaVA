import json
import os
import shutil
import hashlib
import traceback
import torch
import torch.distributed as dist
import rllava.utils.torch_functional as VF
from typing import Optional, List, Any
from jinja2 import Template
from typing import Dict
from PIL import ImageFile
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from rllava.data.data_utils import process_image, process_video
from rllava.utils.constants import *
from rllava.distribute import dist_utils



ImageFile.LOAD_TRUNCATED_IMAGES = True


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        self.dataset = self._load_data(data_path, data_split)
        self.data_source = {"path": data_path, "split": data_split}

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            doc2len = self.build_filter()
            self.dataset = self.maybe_filter_out_long_prompts(
                doc2len, filter_overlong_prompts_workers
            )

    @staticmethod
    def _load_single(path: str, split: str):
        """Load a single data source (file, directory, or HF hub name)."""
        if os.path.isdir(path):
            try:
                full_dataset = load_from_disk(path)
                if hasattr(full_dataset, "keys") and split in full_dataset:
                    return full_dataset[split]
                return full_dataset
            except Exception:
                return load_dataset("arrow", data_dir=path, split=split)
        elif os.path.isfile(path):
            file_type = os.path.splitext(path)[-1][1:].replace("jsonl", "json")
            return load_dataset(file_type, data_files=path, split=split)
        else:
            return load_dataset(path, split=split)

    def _load_data(self, data_path: str, data_split: str):
        """Load one or more data sources separated by commas."""
        paths = [p.strip() for p in data_path.split(",") if p.strip()]
        if len(paths) == 1:
            return self._load_single(paths[0], data_split)
        datasets = [self._load_single(p, data_split) for p in paths]
        combined = concatenate_datasets(datasets)
        print(f"Concatenated {len(paths)} datasets: {[len(d) for d in datasets]} → {len(combined)} total")
        return combined

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = example[self.prompt_key]

        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt

        prompt_str: str = prompt
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example and example[self.image_key] is not None:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            if "<image>" not in prompt_str:
                prompt_str = "<image>" + prompt_str
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example and example[self.video_key] is not None:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return prompt_str

    def build_filter(self):
        if self.processor is not None:
            def doc2len(doc) -> int:
                try:
                    messages = self._build_messages(doc)
                    # pass tool schemas if available so the processor can format prompts
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    if self.image_key in doc and doc[self.image_key]:
                        if not isinstance(doc[self.image_key], list):
                            doc[self.image_key] = [doc[self.image_key]]
                        images = [
                            process_image(image, self.min_pixels, self.max_pixels) for image in doc[self.image_key]
                        ]
                    else:
                        images = None

                    if self.video_key in doc and doc[self.video_key]:
                        videos, video_metadata = zip(
                            *[
                                process_video(
                                    video, self.min_pixels, self.max_pixels, self.video_fps
                                )
                                for video in doc[self.video_key]
                            ],
                            strict=True,
                        )
                        videos = list(videos)
                        video_metadata = list(video_metadata)
                        videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                    else:
                        videos = None
                        videos_kwargs = {}

                    return len(
                        self.processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                            "input_ids"
                        ][0]
                    )
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    traceback.print_exc()
                    return self.max_prompt_length + 1
        else:
            def doc2len(doc) -> int:
                try:
                    return len(
                        self.tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)
                    )
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    traceback.print_exc()
                    return self.max_prompt_length + 1
        return doc2len

    def _dist_barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def maybe_filter_out_long_prompts(self, doc2len, num_workers: int):
        cache_path = self._filter_cache_path()
        cached = self._load_filtered_dataset_from_cache(cache_path)
        if cached is not None:
            return cached

        if dist_utils.is_rank0():
            filtered = self.dataset.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )
            # self.dataset = self.dataset.filter(
            #     lambda doc: len(tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
            #     num_proc=filter_overlong_prompts_workers,
            #     desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            # )
            print(f"filter dataset len: {len(filtered)}")
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path)
            filtered.save_to_disk(cache_path)
        self._dist_barrier()

        cached = self._load_filtered_dataset_from_cache(cache_path)
        if cached is None:
            raise RuntimeError(f"Failed to load filtered dataset cache from {cache_path}.")

        self._dist_barrier()
        return cached

    def _filter_cache_path(self) -> str:
        cache_root = os.environ.get(
            "RLLAVA_FILTER_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "rllava", "filtered"),
        )
        os.makedirs(cache_root, exist_ok=True)

        format_prompt_hash = None
        if self.format_prompt is not None:
            format_prompt_hash = hashlib.md5(self.format_prompt.encode("utf-8")).hexdigest()

        payload = {
            "data_path": self.data_source["path"],
            "data_split": self.data_source["split"],
            "max_prompt_length": self.max_prompt_length,
            "prompt_key": self.prompt_key,
            "answer_key": self.answer_key,
            "image_key": self.image_key,
            "video_key": self.video_key,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "video_fps": self.video_fps,
            "format_prompt_hash": format_prompt_hash,
            "processor_cls": type(self.processor).__name__ if self.processor else None,
            "tokenizer_cls": type(self.tokenizer).__name__ if self.tokenizer else None,
            "dataset_fingerprint": getattr(self.dataset, "_fingerprint", None),
        }
        cache_key = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return os.path.join(cache_root, cache_key)

    def _load_filtered_dataset_from_cache(self, cache_path: str):
        if not os.path.isdir(cache_path):
            return None
        try:
            return load_from_disk(cache_path)
        except Exception:
            traceback.print_exc()
            return None

    def __len__(self):
        return len(self.dataset)

    def remove_data(self, remove_item_list: list[int]) -> None:
        if not remove_item_list:
            return
        total = len(self.dataset)
        valid = sorted({idx for idx in remove_item_list if 0 <= idx < total})
        if not valid:
            return
        keep = [i for i in range(total) if i not in set(valid)]
        self.dataset = self.dataset.select(keep)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)

            if images is not None:
                # Handle both single image and list of images #mzh
                if not isinstance(images, list): #mzh
                    images = [images] #mzh
                if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                    images = [os.path.join(self.image_dir, image) for image in images]
    
                processed_images = [] if len(images) != 0 else None  # text-only data
                for image in images:
                    #processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels, self.processor))
    
                model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            else:
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
                
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            # image_grid_thw = model_inputs.pop("image_grid_thw")[0]
            # pixel_values = model_inputs.pop("pixel_values")
            example["multi_modal_data"] = {"images": images}
            # example['pixel_values'] = pixel_values
            # example['image_grid_thw'] = image_grid_thw
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["initial_prompt_text"] = messages[0]["content"]# for environment reset

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from rllava.model.patch.qwen3_vl import get_rope_index
            else:
                from rllava.model.patch.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
            # position_ids = vision_position_ids
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        if self.answer_key and self.answer_key in example:
            example["ground_truth"] = example.pop(self.answer_key)
        else:
            example["ground_truth"] = ""
        example["item"] = index
        return example
