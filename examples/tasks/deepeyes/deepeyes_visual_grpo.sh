#!/bin/bash
# DeepEyes-style visual agentic RL training with GRPO
#
# Prerequisites:
#   1. Launch an LLM-as-a-judge server (e.g. Qwen2.5-72B-Instruct via sglang)
#      and set LLM_AS_A_JUDGE_BASE below.
#      python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 18901 --tp-size 8
#      vllm serve Qwen/Qwen3-8B \
#          --port 18901 \
#          --gpu-memory-utilization 0.8 \
#          --max-model-len 32768 \
#          --tensor-parallel-size 1 \ 
#          --served-model-name "judge" \
#          --trust-remote-code \
#          --disable-log-requests
#   2. Prepare a parquet/HF dataset with columns:
#        - images: list of image bytes
#        - problem / question: the question text
#        - answer: ground truth answer
#   3. Adjust MODEL_PATH, TRAIN_SET, VAL_SET, GPU count.

set -x

export PYTHONUNBUFFERED=1

# ---------- Judge endpoint (for reward scoring) ----------
export LLM_AS_A_JUDGE_BASE="http://localhost:18901/v1"

# ---------- Model ----------
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # or Qwen3-VL-4B, etc.

# ---------- Data ----------
# Download from Hugging Face (run once from repo root):
#   huggingface-cli download ChenShawn/DeepEyes-Datasets-47k --repo-type dataset --local-dir Visual-Agent/DeepEyes-RL-Data
# Or: python -c "from huggingface_hub import snapshot_download; snapshot_download('ChenShawn/DeepEyes-Datasets-47k', repo_type='dataset', local_dir='Visual-Agent/DeepEyes-RL-Data')"
LOCAL_DATASET_PATH="../huggingface_cache/hub/datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots/5546681e28fa2eda9f60a9ea9dd0cf291216ded3"
TRAIN_SET="${LOCAL_DATASET_PATH}/data_0.1.2_visual_toolbox_v2.parquet,${LOCAL_DATASET_PATH}/data_v0.8_visual_toolbox_v2.parquet,${LOCAL_DATASET_PATH}/data_thinklite_reasoning_acc.parquet"
VAL_SET="${LOCAL_DATASET_PATH}/data_thinklite_reasoning_acc.parquet"

# ---------- Output ----------
OUTPUT_DIR="outputs_deepeyes"
export TENSORBOARD_DIR=$OUTPUT_DIR
NAME=deepeyes_visual_grpo_qwen3_vl_4b

# ---------- Train ----------
CUDA_VISIBLE_DEVICES=3 torchrun --master_port=29618 --nproc_per_node=1 -m rllava.train.pipeline.agentic \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_loss=false \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.prompt_key=prompt \
    data.answer_key=reward_model \
    data.format_prompt=null \
    data.dataset_class=examples/tasks/deepeyes/deepeyes_dataset.py:DeepEyesDataset \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    actor.model.model_path=${MODEL_PATH} \
    actor.ppo_mini_batch_size=1 \
    actor.ppo_micro_batch_size=1 \
    data.filter_overlong_prompts=false \
    rollout.seed=None \
    rollout.n=8 \
    rollout.temperature=1.0 \
    rollout.max_turns=2 \
    rollout.discount=1.0 \
    rollout.env_config_path=./examples/tasks/deepeyes/deepeyes_env_config.yaml \
    rollout.vllm.gpu_memory_utilization=0.8 \
    rollout.vllm.enforce_eager=false \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/deepeyes_visual.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=false \
    trainer.val_freq=-1 \
    trainer.save_freq=50 \
    trainer.val_before_train=false