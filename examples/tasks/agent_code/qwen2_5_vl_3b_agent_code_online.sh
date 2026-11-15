#!/bin/bash

set -x

export PYTHONUNBUFFERED=1


MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

LOCAL_DATASET_PATH="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_code_1_2k.json"
IMAGE_DIR="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_code_1_2k_images"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@train"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 -m rllava.controller \
    config=examples/config.yaml \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.image_key=image_path \
    data.max_prompt_length=2048 \
    data.max_pixels=401408 \
    data.answer_key=solution \
    data.train_image_dir="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_code_1_2k_images" \
    data.val_image_dir="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Coding-image" \
    data.format_prompt=./examples/format_prompt/agent_code.jinja \
    data.train_batch_size=4 \
    actor.ppo_mini_batch_size=4 \
    actor.ppo_micro_batch_size=1 \
    actor.log_prob_micro_batch_size=1 \
    actor.model.model_path=${MODEL_PATH} \
    actor.fsdp.enable_cpu_offload=true \
    rollout.tensor_parallel_size=1 \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/agent_code.py:compute_score \
    trainer.experiment_name=qwen2.5_vl_3b_agent_code_online \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=false \
    trainer.val_freq=-1 \
    trainer.save_freq=50 \
    trainer.val_before_train=false
