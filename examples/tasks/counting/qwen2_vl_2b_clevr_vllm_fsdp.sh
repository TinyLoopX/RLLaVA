#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"

LOCAL_DATASET_PATH="BUAADreamer/clevr_count_70k"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@test"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR


torchrun --nproc_per_node=2 -m rllava.controller \
    config=examples/config.yaml \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.val_batch_size=200 \
    actor.model.model_path=${MODEL_PATH} \
    rollout.vllm.gpu_memory_utilization=0.6 \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_vl_2b_clevr \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=true
