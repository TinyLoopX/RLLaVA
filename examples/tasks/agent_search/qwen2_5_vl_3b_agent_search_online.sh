#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

LOCAL_DATASET_PATH="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_search_20.json"
IMAGE_DIR="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_search_20_images"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@train"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 -m rllava.controller \
    config=examples/config.yaml \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.image_key=image_path \
    data.max_prompt_length=2048 \
    data.max_pixels=401408 \
    data.train_image_dir="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Training/rft_agent_search_20_images" \
    data.val_image_dir="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Search-image" \
    data.answer_key=solution \
    data.format_prompt=./examples/format_prompt/agent_search.jinja \
    actor.model.model_path=${MODEL_PATH} \
    rollout.tensor_parallel_size=1 \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/agent_search.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_agent_search_online \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=false \
    trainer.val_freq=-1 \
    trainer.save_freq=50 \
    trainer.total_epochs=40 \
    trainer.val_before_train=false



