#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"

LOCAL_DATASET_PATH="Kangheng/PR1-Datasets-Grounding"

TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@train"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m rllava.controller \
    config=examples/config_grounding.yaml \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.image_key=image \
    data.answer_key=solution \
    data.format_prompt=./examples/format_prompt/rec.jinja \
    data.train_batch_size=4 \
    actor.ppo_mini_batch_size=4 \
    actor.ppo_micro_batch_size=1 \
    actor.log_prob_micro_batch_size=1 \
    actor.model.model_path=${MODEL_PATH} \
    rollout.tensor_parallel_size=1 \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/grounding.py:compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_grounding_online \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=false \
    trainer.val_freq=-1 \
    trainer.save_freq=50
    
