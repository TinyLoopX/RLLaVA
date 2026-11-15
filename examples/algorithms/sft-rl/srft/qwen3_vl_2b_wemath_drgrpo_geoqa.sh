#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

TRAIN_SET="DaveKevin/GeoQA-GoldenCoT-8K@train"
VAL_SET="We-Math/We-Math@testmini"

OUTPUT_DIR="outputs_wemath"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen3_vl_2b_srft_drgrpo_geoqa

torchrun --nproc_per_node=2 --master-port 29503 -m rllava.controller \
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
    data.format_prompt=./examples/format_prompt/wemath.jinja \
    data.val_batch_size=1000 \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.dataset_class=examples/algorithms/sft-rl/custom_datasets.py:RLHFDatasetWithTargetWemath \
    data.dataset_kwargs.target_key=golden_cot \
    data.dataset_kwargs.max_target_length=2048 \
    data.dataset_kwargs.val.prompt_key=question \
    data.dataset_kwargs.val.image_key=image \
    actor.model.model_path=${MODEL_PATH} \
    actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor.ppo_mini_batch_size=2 \
    actor.ppo_micro_batch_size=1 \
    actor.loss_remove_clip=True \
    actor.entropy_coeff=0.0 \
    actor.policy_loss.loss_mode=srft \
    actor.off_policy_reshape=p_div_p_0.1 \
    actor.fsdp.offload_params=False \
    reward.reward_type=batch \
    reward.reward_function=./examples/reward_function/wemath.py:compute_score_batch_with_wemath_metrics \
    rollout.vllm.gpu_memory_utilization=0.5 \
    rollout.seed=none \
    rollout.vllm.dtype=float16 \
    rollout.n=5 \
    rollout.val_override_config.temperature=1.0 \
    rollout.val_override_config.top_p=0.95 \
    rollout.val_override_config.n=1 \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.val_before_train=True \
    trainer.save_freq=5

