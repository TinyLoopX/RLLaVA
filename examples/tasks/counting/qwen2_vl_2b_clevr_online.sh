#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH="/mnt/ve_share/zhaolei/model/Qwen__Qwen2-VL-2B-Instruct"
# 使用统一的本地数据集路径（HF cache格式）
LOCAL_DATASET_PATH="/mnt/ve_share/zhaolei/.cache/huggingface/datasets/BUAADreamer__clevr_count_70k"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@test"

OUTPUT_DIR="/mnt/ve_share/zhaolei/outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

torchrun --nproc_per_node=2 -m rllava.controller \
    config=examples/config.yaml \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.train_batch_size=4 \
    actor.ppo_mini_batch_size=4 \
    actor.ppo_micro_batch_size=1 \
    actor.log_prob_micro_batch_size=1 \
    actor.model.model_path=${MODEL_PATH} \
    rollout.tensor_parallel_size=1 \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_vl_2b_clevr_online \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=true \
    trainer.val_before_train=false
