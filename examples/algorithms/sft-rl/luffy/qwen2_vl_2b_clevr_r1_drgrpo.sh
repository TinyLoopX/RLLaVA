#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"

TRAIN_SET="MMInstruction/Clevr_CoGenT_TrainA_R1"
VAL_SET="MMInstruction/SuperClevr_Val"

OUTPUT_DIR="outputs_clevr_r1"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen2_vl_2b_luffy_drgrpo

torchrun --nproc_per_node=2 --master-port 29505 -m rllava.controller \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.prefix_strategy=random \
    algorithm.min_prefix_ratio=1.0 \
    algorithm.max_prefix_ratio=1.0 \
    algorithm.n_prefix=1 \
    algorithm.prefix_share_across_samples=false \
    algorithm.use_kl_loss=false \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.val_batch_size=200 \
    data.prompt_key=problem \
    data.answer_key=solution \
    data.image_key=image \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.dataset_class=examples/algorithms/sft-rl/custom_datasets.py:RLHFDatasetWithTargetClevr \
    data.dataset_kwargs.target_key=thinking \
    data.dataset_kwargs.max_target_length=2048 \
    actor.model.model_path=${MODEL_PATH} \
    actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor.ppo_mini_batch_size=2 \
    actor.ppo_micro_batch_size=1 \
    actor.loss_remove_clip=True \
    actor.entropy_coeff=0.001 \
    actor.policy_loss.loss_mode=luffy \
    actor.off_policy_reshape=p_div_p_0.1 \
    actor.fsdp.offload_params=False \
    reward.reward_type=sequential \
    reward.reward_function=./examples/reward_function/r1v.py:compute_score_without_format \
    rollout.seed=none \
    rollout.vllm.dtype=float16 \
    rollout.n=5 \
    rollout.val_override_config.temperature=0.6 \
    rollout.val_override_config.top_p=0.95 \
    rollout.val_override_config.n=1 \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.val_before_train=True \
    trainer.save_freq=5



