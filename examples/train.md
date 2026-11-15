# RLLaVA Training Guide

## Quick Start

```bash
set -x
export PYTHONUNBUFFERED=1
MODEL_PATH=your_model_path
TRAIN_SET=your_org/dataset@train
OUTPUT_DIR=your_output_directory
export TENSORBOARD_DIR=$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 -m rllava.controller \
  config=examples/config.yaml \
  data.train_files=${TRAIN_SET} \
  data.format_prompt=./examples/format_prompt/your_task.jinja \
  data.prompt_key=prompt_key_of_your_dataset \
  data.answer_key=answer_key_of_your_dataset \
  data.image_key=image_key_of_your_dataset \
  actor.model.model_path=${MODEL_PATH} \
  reward.reward_function=./examples/reward_function/your_task.py:compute_score \
  algorithm.adv_estimator=grpo \
  trainer.outputs_dir=${OUTPUT_DIR} \
  trainer.val_freq=-1 \
  trainer.val_before_train=false
```

### Key Parameters

- **`CUDA_VISIBLE_DEVICES`**: Specifies which GPUs to use for training
- **`--nproc_per_node`**: Total number of GPUs/processes per node
- **`config`**: Base configuration file (e.g., `examples/config.yaml`)
- **`data.train_files`**: Training dataset path in format `org/dataset@split` (existing datasets are loaded from Hugging Face)
- **`data.prompt_key`**: Dataset field name containing the question/prompt text (default: `problem`)
- **`data.answer_key`**: Dataset field name containing the ground-truth answer (default: `answer`)
- **`data.image_key`**: Dataset field name containing image data or image paths (default: `images`)
- **`data.format_prompt`**: Prompt template file path using Jinja2 syntax
- **`actor.model.model_path`**: Path to the base model (local or HuggingFace)
- **`reward.reward_function`**: Reward function in format `path/to/file.py:function_name`
- **`algorithm.adv_estimator`**: RL algorithm to use (grpo, rloo, ppo, etc.)
- **`trainer.val_freq`**: Validation frequency in epochs (-1 to disable)
- **`trainer.val_before_train`**: Whether to validate before training starts

### Configuration Tips

- Set `trainer.val_freq=-1` and `trainer.val_before_train=false` if your dataset lacks a validation set.
- Override any `config.yaml` parameter by adding it as a command-line argument.
- Change the algorithm by setting `algorithm.adv_estimator` according to scripts in `examples/algorithms/`.
- Always execute commands from the `RLLaVA` directory root.

## Downstream Tasks

### Grounding
Visual referring expression comprehension - predict bounding box coordinates of the referred region.
```bash
bash examples/tasks/grounding/qwen2_vl_2b_grounding_vllm_online.sh
```

### OVD
Open-Vocabulary Detection - detect objects and output bounding boxes with confidence scores.
```bash
bash examples/tasks/OVD/qwen2_5_vl_3b_ovd_online.sh
```

### Counting
Visual object counting in complex scenes.
```bash
bash examples/tasks/counting/qwen2_5_vl_3b_clevr_online.sh
```

### Math
Geometry problem solving with mathematical reasoning.
```bash
bash examples/tasks/math/qwen2_5_vl_3b_geoqa8k_online.sh
```

### Agent-Search
Multi-step web search agent for complex queries.
```bash
bash examples/tasks/agent_search/qwen2_5_vl_3b_agent_search_online.sh
```

### Agent-Code
Code generation from visual interface designs.
```bash
bash examples/tasks/agent_code/qwen2_5_vl_3b_agent_code_online.sh
```

## Contributing

To add a new task:
1. Create prompt template in `format_prompt/`
2. Implement reward function in `reward_function/`
3. Add task-specific config (optional)
4. Create training script in `tasks/your_task/`
