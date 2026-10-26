import os
import time
import json
import socket
import asyncio
import requests
import aiohttp
import torch
import torch.distributed as dist
from collections import defaultdict
from typing import (
    Tuple,
    Literal,
    Optional,
    TYPE_CHECKING,
    Dict,
    Any,
    List,
    Sequence,
)

from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server_engine import launch_server_process
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from torch.distributed.tensor import DTensor, Replicate
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from .. import register_engine
from .base import InferenceEngine, _process_multi_modal_data, pack_rollout_log_probs, response_mask_from_lengths
from rllava.data.protocol import DataProto
from rllava.utils import torch_functional as VF
from rllava.utils.model_utils import print_gpu_memory_usage
from rllava.utils.transformers_compat import is_transformers_version_in_range
from rllava.distribute.dist_utils import dist_barrier
from rllava.utils.device import get_device_id
from rllava.data.data_utils import image2base64




class _SyncGenerateAdapter:
    """Simple synchronous adapter that issues blocking HTTP requests."""

    def __init__(self, worker_url: str):
        self.worker_url = worker_url

    def generate(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        response = sync_request(self.worker_url, "generate", json=payload, retry_delay=10)
        if isinstance(response, list):
            return response
        return [response]


class _AsyncGenerateAdapter:
    """Adapter that issues concurrent HTTP requests via aiohttp."""

    def __init__(self, worker_url: str, max_trials: int = 3, retry_delay: float = 1.0):
        self.worker_url = worker_url
        self.max_trials = max_trials
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None)
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        await self._ensure_session()
        assert self._session is not None
        url = f"{self.worker_url}/{endpoint}"

        for trial in range(self.max_trials):
            try:
                async with self._session.post(url, json=payload) as response:
                    response.raise_for_status()
                    try:
                        return await response.json(content_type=None)
                    except json.decoder.JSONDecodeError:
                        return await response.text()
            except Exception:
                if trial == self.max_trials - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

    async def generate_batch(self, payloads: List[Dict[str, Any]]) -> List[Any]:
        tasks = [self._post("generate", payload) for payload in payloads]
        return await asyncio.gather(*tasks)



if TYPE_CHECKING:
    from rllava.ppo.config import RolloutConfig


@register_engine("sglang")
class SGLangEngine(InferenceEngine):

    def __init__(
        self,
        model_name_or_path: str,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        layout,
    ):
        super().__init__(model_name_or_path, config, tokenizer, processor, layout)

        self.gloo_group = layout.groups.get("control")
        self.replica_group = layout.groups.get("replica")
        self.tp_cuda_visible_devices = ",".join(layout.replica_devices)
        self.server_process = None
        self._sync_adapter = None
        self._async_adapter = None
        monkey_patch_torch_reductions()
        if self.layout.tp_rank == 0:
            self._launch_server()
            self._sync_adapter = _SyncGenerateAdapter(self.worker_url)
            self._async_adapter = _AsyncGenerateAdapter(self.worker_url)
            self._sleep()
        self.loaded = False
        dist_barrier(self.gloo_group)

    def _build_sampling_params(self, meta_info: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "temperature": meta_info.get("temperature", self.config.temperature),
            "top_p": meta_info.get("top_p", self.config.top_p),
            "top_k": meta_info.get("top_k", self.config.top_k),
            "repetition_penalty": meta_info.get("repetition_penalty"),
            "max_new_tokens": meta_info.get("max_new_tokens", self.config.response_length),
            "stop": meta_info.get("stop"),
            "ignore_eos": meta_info.get("ignore_eos", self.config.ignore_eos),
        }
        return {k: v for k, v in params.items() if v is not None}

    def _prepare_batch(self, prompts: DataProto) -> List[Dict[str, Any]]:
        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.get("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.get("multi_modal_data", None)
        sampling_params = self._build_sampling_params(prompts.meta_info)
        print(f"SGLang Sampling params: {sampling_params}")

        engine_inputs = []
        for idx, raw_prompt_ids in enumerate(batch_raw_prompt_ids):
            payload = {
                "input_ids": list(raw_prompt_ids),
                "sampling_params": {**sampling_params},
                "return_logprob": True,
            }
            if batch_multi_modal_data is not None:
                processed_images = _process_multi_modal_data(
                    batch_multi_modal_data[idx],
                    prompts.meta_info["min_pixels"],
                    prompts.meta_info["max_pixels"],
                    prompts.meta_info["video_fps"],
                    self.processor,
                )['image']
                payload["image_data"] = image2base64(processed_images)

            engine_inputs.append(payload)

        return engine_inputs

    def _finalize_batch(self, prompts: DataProto, response_ids: List[List[int]], token_log_probs: List[List[float]]) -> DataProto:
        token_ids = response_ids
        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(prompts.batch["input_ids"].device)

        batch_size = len(response_ids)
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        batch_multi_modal_data = prompts.non_tensor_batch.pop("multi_modal_data", None)
        eos_token_id: int = prompts.meta_info["eos_token_id"]

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = response_mask_from_lengths(
            response_ids, [len(ids) for ids in token_ids], eos_token_id, dtype=attention_mask.dtype,
            ignore_eos=prompts.meta_info.get("ignore_eos", self.config.ignore_eos),
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        batch.update(pack_rollout_log_probs(token_ids, token_log_probs, response_mask, temperature))
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def generate(self, prompts: DataProto) -> DataProto:
        num_return_sequences = prompts.meta_info.get("n", getattr(self.config, "n", 1))
        if num_return_sequences > 1:
            prompts = prompts.repeat(repeat_times=num_return_sequences, interleave=True)

        prepared = self._prepare_batch(prompts)
        responses, log_probs = self._collect_sync(prepared)
        return self._finalize_batch(prompts, responses, log_probs)

    async def agenerate(self, prompts: DataProto) -> DataProto:
        num_return_sequences = prompts.meta_info.get("n", self.config.n)
        if num_return_sequences > 1:
            prompts = prompts.repeat(repeat_times=num_return_sequences, interleave=True)
        prepared = self._prepare_batch(prompts)
        responses, log_probs = await self._collect_async(prepared)
        return self._finalize_batch(prompts, responses, log_probs)

    def update_weights(self, model):
        weights = model.state_dict()
        if is_transformers_version_in_range(min_version="4.54.0"):
            weights = self._rename_weight_keys(weights, model)

        device = get_device_id() 
        dtype_to_named_tensors = defaultdict(list)
        bucket_size = 0
        for name in sorted(weights.keys()):
            tensor = weights[name]
            param_size = tensor.numel() * tensor.element_size()

            if bucket_size > 0 and bucket_size + param_size > (self.config.bucket_size << 20):
                self._update_tensor_bucket(dtype_to_named_tensors)
                dtype_to_named_tensors = defaultdict(list)
                bucket_size = 0
            
            tensor = tensor.to(device, non_blocking=True).detach()
            if isinstance(tensor, DTensor):
                # async version of `tensor.full_tensor()`
                tensor = tensor.redistribute(
                    placements=[Replicate()] * tensor.device_mesh.ndim,
                    async_op=True
                ).to_local()

            dtype_to_named_tensors[tensor.dtype].append((name, tensor))
            bucket_size += param_size

        self._update_tensor_bucket(dtype_to_named_tensors)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in vllm engine")

    def load(self, model):
        torch.cuda.empty_cache()
        assert self.loaded is False, "sglang engine has already been loaded"

        dist_barrier(self.gloo_group)
        print_gpu_memory_usage("Before sglang wake up in sglang engine")
        
        self._wake_up(tags=["weights"])
        self.update_weights(model)
        self._wake_up(tags=["kv_cache"])
        dist_barrier(self.gloo_group)

        print_gpu_memory_usage("After sglang wake up in sglang engine")
        self.loaded = True

    def offload(self):
        assert self.loaded is True, "sglang engine has not been loaded"

        dist_barrier(self.gloo_group)
        self._sleep()
        dist_barrier(self.gloo_group)

        torch.cuda.empty_cache()
        self.loaded = False

    def close(self):
        if self.server_process is not None:
            kill_process_tree(self.server_process.pid)
            self.server_process = None

    def _launch_server(self):
        if any(
            os.environ.get(key, "0").lower() in ("1", "true")
            for key in ("RETURN_ORIGINAL_LOGPROB", "SGLANG_RETURN_ORIGINAL_LOGPROB")
        ):
            raise ValueError("Unset RETURN_ORIGINAL_LOGPROB / SGLANG_RETURN_ORIGINAL_LOGPROB to collect temperature-scaled probabilities.")
        server_args = ServerArgs(
            enable_memory_saver=True,
            host=get_host(),
            port=get_available_port(),
            model_path=self.model,
            tp_size=self.tp_size,
            log_level="error",
            **self.config.sglang.to_dict()
        )

        # Only the spawned server inherits its own device/rendezvous environment.
        keys = ("CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_USE_AGENT_STORE")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            os.environ["CUDA_VISIBLE_DEVICES"] = self.tp_cuda_visible_devices
            self.server_process = launch_server_process(server_args)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.worker_url = server_args.url()

    def _sleep(self, tags=("weights", "kv_cache")):
        if self.layout.tp_rank == 0:
            sync_request(self.worker_url, "release_memory_occupation", json={"tags": list(tags)})

    def _wake_up(self, tags=("weights", "kv_cache")):
        if self.layout.tp_rank == 0:
            sync_request(self.worker_url, "resume_memory_occupation", json={"tags": list(tags)})
            self._wait_worker_ready()

    def _wait_worker_ready(self, timeout: float = 300.0, interval: float = 1.0) -> None:
        if self.layout.tp_rank != 0:
            return
        worker_url = getattr(self, "worker_url", None)
        if not worker_url:
            return

        end_time = time.time() + timeout
        last_error: Optional[Exception] = None

        while time.time() < end_time:
            try:
                # sync_request(worker_url, endpoint, method="GET", max_trials=1, retry_delay=interval)
                sync_request(self.worker_url, "health", "GET", 1, interval)
                return
            except Exception as exc:  # noqa: PERF203
                last_error = exc
            time.sleep(interval)

        raise RuntimeError(
            f"SGLang worker at {worker_url} failed health check after wake up. Last error: {last_error}"
        )

    def _collect_sync(self, engine_inputs: List[Dict[str, Any]]) -> Tuple[List[List[int]], List[List[float]]]:
        raw_responses: List[Any] = []
        print(f"sending request to {self.worker_url}")
        for payload in engine_inputs:
            raw_responses.extend(self._sync_adapter.generate(payload))
        return self._decode_responses(raw_responses)

    async def _collect_async(self, engine_inputs: List[Dict[str, Any]]) -> Tuple[List[List[int]], List[List[float]]]:
        raw_responses = await self._async_adapter.generate_batch(engine_inputs)
        return self._decode_responses(raw_responses)

    def _decode_responses(self, raw_responses: Sequence[Any]) -> Tuple[List[List[int]], List[List[float]]]:
        collected: List[List[int]] = []
        collected_log_probs = []
        for raw in raw_responses:
            responses = raw if isinstance(raw, list) else [raw]
            for resp in responses:
                # These IDs are authoritative. Re-encoding decoded text or
                # stripping role/EOS tokens would change the sampled actions.
                entries = resp["meta_info"]["output_token_logprobs"]
                tokens, values = [], []
                for entry in entries:
                    if isinstance(entry, dict):
                        token = entry.get("token_id", entry.get("token"))
                        value = entry["logprob"]
                    else:
                        value, token = entry[:2]
                    tokens.append(int(token))
                    values.append(float(value))
                output_ids = resp.get("output_ids", resp.get("meta_info", {}).get("output_ids"))
                if output_ids is not None and list(output_ids) != tokens:
                    raise ValueError("SGLang output_ids and output_token_logprobs are misaligned.")
                collected_log_probs.append(values[: self.config.response_length])
                collected.append(tokens[: self.config.response_length])
        return collected, collected_log_probs

    def _update_tensor_bucket(
            self, dtype_to_named_tensors: Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]
        ):

        torch.cuda.synchronize()
        serialized_tensors = []
        for named_tensors in dtype_to_named_tensors.values():

            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors)
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": flattened_tensor_bucket.get_metadata()
            }
            serialized_tensors.append(
                MultiprocessingSerializer.serialize(
                    flattened_tensor_data, output_str=True
                )
            )

        gathered_serialized_tensors = [
            None for _ in range(self.tp_size)
        ] if self.layout.tp_rank == 0 else None
        if dist.is_initialized():
            dist.gather_object(serialized_tensors, gathered_serialized_tensors,
                               dst=self.layout.input_rank, group=self.replica_group)
        else:
            gathered_serialized_tensors = [serialized_tensors]
        # [
        #     [tp0_bucket0, tp0_bucket1, ...],
        #     [tp1_bucket0, tp1_bucket1, ...],
        #     ...
        # ]
        if self.layout.tp_rank == 0:

            for serialized_named_tensors in zip(*gathered_serialized_tensors):
                # [
                #     (tp0_bucket0, tp1_bucket0, ...),
                #     (tp0_bucket1, tp1_bucket1, ...),
                #     ...
                # ]
                # HTTP server only sends meta data. Actual weights will be directly 
                # copied from GPUs
                sync_request(
                    self.worker_url,
                    "update_weights_from_tensor",
                    json={
                        "serialized_named_tensors": serialized_named_tensors,
                        "load_format": "flattened_bucket",
                        "flush_cache": False
                    }
                )


def get_host() -> str:
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]

def sync_request(
    url: str,
    endpoint: str,
    method: Literal["POST", "GET"] = "POST",
    max_trials: int = 3,
    retry_delay: int = 1,
    **kwargs
):
    with requests.Session() as session:
        for trial in range(max_trials):
            try:
                response = session.request(method, f"{url}/{endpoint}", **kwargs)

                response.raise_for_status()
                try:
                    return response.json()
                except json.decoder.JSONDecodeError:
                    return response.text

            except requests.RequestException:
                if trial == max_trials - 1:
                    raise
                time.sleep(retry_delay)
