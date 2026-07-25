#!/bin/bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="Tongyi-MAI/MAI-UI-8B"
LOCAL_DATASET_PATH="./examples/data/osworld_train.json"
ENV_CONFIG="./rllava/ppo/env/osworld_subprocess.yaml"
OUTPUT_DIR="outputs/rllava/gui_agent_osworld"
EXP_NAME="mai_ui_2b_osworld_grpo"

export TENSORBOARD_DIR="${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=5 torchrun --master_port=29619 --nproc_per_node=1 -m rllava.train.pipeline.rlvr \
    config=examples/config_gui.yaml \
    actor.model.model_path="${MODEL_PATH}" \
    data.train_files=${LOCAL_DATASET_PATH} \
    data.val_files=${LOCAL_DATASET_PATH} \
    rollout.env_config_path="${ENV_CONFIG}" \
    rollout.max_turns=15 \
    reward.reward_type=env \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.outputs_dir="${OUTPUT_DIR}" \
    trainer.val_freq=-1 \
    trainer.save_freq=50 \
    trainer.val_before_train=false

