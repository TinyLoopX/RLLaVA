#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-Math-1.5B

LOCAL_DATASET_PATH="data"
TRAIN_SET="${LOCAL_DATASET_PATH}/openr1.parquet"
VAL_SET="${LOCAL_DATASET_PATH}/AIME24/test.parquet"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR


HINT_STEPS=100  # number of steps for cosine annealing, 0 means no annealing (fixed max_ratio)

NAME=qwen2_5_math_1_5b_uft_drgrpo

torchrun --nproc_per_node=2 --master-port 29509 -m rllava.controller \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.prefix_strategy=hint \
    algorithm.hint_steps=${HINT_STEPS} \
    algorithm.min_prefix_ratio=0.0 \
    algorithm.max_prefix_ratio=1.0 \
    algorithm.n_prefix=1 \
    algorithm.prefix_share_across_samples=false \
    algorithm.use_kl_loss=false \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.val_batch_size=512 \
    data.format_prompt=null \
    data.prompt_key=prompt \
    data.answer_key=reward_model \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.dataset_class=examples/algorithms/sft-rl/custom_datasets.py:RLHFDatasetWithTarget \
    data.dataset_kwargs.target_key=target \
    data.dataset_kwargs.max_target_length=2048 \
    actor.model.model_path=${MODEL_PATH} \
    actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor.ppo_mini_batch_size=2 \
    actor.ppo_micro_batch_size=1 \
    actor.loss_remove_clip=True \
    actor.entropy_coeff=0.001 \
    actor.policy_loss.loss_mode=uft \
    actor.sft_loss_coef=1.0 \
    actor.fsdp.offload_params=False \
    reward.reward_function=./examples/algorithms/sft-rl/entropy_math_reward.py:compute_score \
    rollout.seed=none \
    rollout.vllm.dtype=float16 \
    rollout.n=8 \
    rollout.val_override_config.temperature=0.6 \
    rollout.val_override_config.top_p=0.95 \
    rollout.val_override_config.n=8 \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.val_before_train=False \
    trainer.save_freq=5
