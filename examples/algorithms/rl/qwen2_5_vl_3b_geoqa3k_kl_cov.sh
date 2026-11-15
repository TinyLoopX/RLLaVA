#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

LOCAL_DATASET_PATH="hiyouga/geometry3k"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@test"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen2_5_vl_3b_geoqa3k_klcov

torchrun --nproc_per_node=2 -m rllava.controller \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_loss=false \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.val_batch_size=1000 \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.max_prompt_length=1024 \
    actor.model.model_path=${MODEL_PATH} \
    rollout.vllm.gpu_memory_utilization=0.7 \
    actor.clip_ratio_low=0.2 \
    actor.clip_ratio_high=0.2 \
    actor.clip_ratio_c=10.0 \
    actor.policy_loss.loss_mode=kl_cov \
    actor.policy_loss.kl_cov_ratio=0.002 \
    actor.policy_loss.ppo_kl_coef=1.0 \
    reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR}
