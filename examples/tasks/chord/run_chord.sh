#!/usr/bin/env bash
set -euo pipefail

# Example launcher for CHORD task using rllava RLVR pipeline
# Environment overrides:
#   OBS_CHORD_MODEL, OBS_CHORD_TRAIN, OBS_CHORD_VAL, OBS_CHORD_SFT, OBS_CHORD_OUT

CONFIG_DIR=$(cd "$(dirname "$0")" && pwd)
CONFIG_FILE="$CONFIG_DIR/config_chord.yaml"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NP=${NP:-4}

torchrun --nproc_per_node=$NP -m rllava.train.pipeline.rlvr \
  --config "$CONFIG_FILE" \
  trainer.n_gpus_per_node=$NP \
  actor.model.model_path=${OBS_CHORD_MODEL:-Qwen/Qwen2.5-1.5B-Instruct} \
  trainer.save_checkpoint_path=${OBS_CHORD_OUT:-./checkpoints/chord}


