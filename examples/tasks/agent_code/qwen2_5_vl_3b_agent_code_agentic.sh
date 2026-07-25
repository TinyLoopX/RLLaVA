#!/bin/bash
# Multi-turn agentic RL training for MAT-Coding (OpenCV-tool agent).
#
# Architecture mirrors examples/tasks/deepeyes/deepeyes_visual_grpo.sh:
#   * pipeline = rllava.train.pipeline.agentic   (uses MultiTurnWorkflow)
#   * env      = examples/tasks/agent_code/agent_code_env.py:AgentCodeEnv
#                wraps the OpenCV code-execution loop from
#                rllava/eval/agent_code/eval_mat_coding.py
#   * dataset  = examples/tasks/agent_code/agent_code_dataset.py:AgentCodeDataset
#   * reward   = examples/reward_function/agent_code_agentic.py:compute_score
#
# Each rollout step:
#   <think>...</think> + <problem>{...}</problem>      → env returns <tips>
#   <think>...</think> + <code>```python ... ```</code> → env runs the code,
#                                                          returns processed image
#   <think>...</think> + <answer>...</answer>          → env signals episode done

set -x

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ---------- Model ----------
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# ---------- Data ----------
LOCAL_DATASET_PATH="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_code_1_2k.json"
TRAIN_IMAGE_DIR="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_code_1_2k_images"
VAL_IMAGE_DIR="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Coding-image"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@train"

# ---------- Output ----------
OUTPUT_DIR="outputs_agent_code_agentic"
export TENSORBOARD_DIR=$OUTPUT_DIR
NAME=qwen2_5_vl_7b_agent_code_agentic

# ---------- Train ----------
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=29617 --nproc_per_node=1 -m rllava.train.pipeline.agentic \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_loss=false \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.image_key=image_path \
    data.answer_key=solution \
    data.train_image_dir=${TRAIN_IMAGE_DIR} \
    data.val_image_dir=${VAL_IMAGE_DIR} \
    data.format_prompt=./examples/format_prompt/agent_code.jinja \
    data.dataset_class=examples/tasks/agent_code/agent_code_dataset.py:AgentCodeDataset \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.max_pixels=401408 \
    data.filter_overlong_prompts=false \
    actor.model.model_path=${MODEL_PATH} \
    actor.ppo_mini_batch_size=2 \
    actor.ppo_micro_batch_size=1 \
    actor.log_prob_micro_batch_size=1 \
    actor.fsdp.enable_cpu_offload=true \
    rollout.seed=None \
    rollout.n=8 \
    rollout.temperature=1.0 \
    rollout.max_turns=5 \
    rollout.discount=1.0 \
    rollout.tensor_parallel_size=1 \
    rollout.env_config_path=./examples/tasks/agent_code/agent_code_env_config.yaml \
    rollout.vllm.gpu_memory_utilization=0.6 \
    rollout.vllm.enforce_eager=true \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/agent_code_agentic.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=false \
    trainer.val_freq=-1 \
    trainer.save_freq=20 \
    trainer.val_before_train=false
