# SFT-RL Fusion Algorithms

This directory contains RLLaVA implementations for representative **SFT-RL Fusion Algorithms**. 

These methods bridge the gap between Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) by unifying them under the **Unified Policy Gradient Estimator (UPGE)** framework (proposed in [UPT](https://arxiv.org/abs/2509.04419)). In RLLaVA, we implement these algorithms using a decoupled plugin system (`ExperienceMixer`, `RolloutProcessor`, and `PolicyLoss`), allowing for flexible composition and "knob-tuning" of the gradient estimator.

## Supported Algorithms

| Algorithm | Key Mechanism (The "Knob") | RLLaVA Implementation Focus | Paper |
|-----------|----------------------------|-----------------------------|-------|
| **HPT** | **Dynamic Gating**: Switching data sources/gradients based on performance feedback. | `ExperienceMixer`: Dynamically replaces low-reward on-policy rollouts with off-policy demonstrations. | [arXiv](https://arxiv.org/abs/2509.04419) |
| **SRFT** | **Entropy-Aware Weighting**: Joint optimization of SFT and RL objectives with entropy-guided balance. | `PolicyLoss`: `srft` plugin with entropy constraints in the advantage estimator. | [arXiv](https://arxiv.org/abs/2506.19767) |
| **LUFFY** | **Off-Policy Guidance**: Incorporating reasoning traces from stronger models with policy shaping. | `PolicyLoss`: `luffy` plugin with policy shaping; `RolloutProcessor` for mask construction. | [arXiv](https://arxiv.org/abs/2504.14945) |
| **UFT** | **Hint Annealing**: Injecting hints into prompts and annealing them over time to guide exploration. | `RolloutProcessor`: Handles hint injection & annealing; `PolicyLoss`: `uft` plugin adds SFT loss on hints. | [arXiv](https://arxiv.org/abs/2505.16984) |

## Directory Structure

Each subdirectory contains ready-to-run scripts for specific benchmarks (e.g., MATH, GSM8K) using these algorithms.

```text
examples/algorithms/sft-rl/
├── hpt/        # Hybrid Post-Training scripts
├── srft/       # Supervised Reinforcement Fine-Tuning scripts
├── luffy/      # Learning to Reason under Off-Policy Guidance scripts
└── uft/        # Unifying Supervised and Reinforcement Fine-Tuning scripts
```

## Implementation Highlights

The implementation is organized by sampling and learning responsibilities:

1.  **Trajectory Processing (`rllava/rollout/processors.py`)**:
    *   **`HPTBatchRolloutProcessor`**: Handles HPT batch rewriting after rollout.
    *   **`PrefixRolloutProcessor`**: Handles hint injection and prefix masking.

2.  **Policy Loss (`rllava/train/policy_loss.py`)**:
    *   Custom loss functions (plugins) that handle the specific gradient math for each method (e.g., `compute_policy_loss_srft`, `compute_policy_loss_luffy`).

3.  **Configuration**: `rllava/controller/config.py` composes role, rollout, and learning configurations. All RL examples use `rllava.controller` as their entry point.

Prefix/unify strategies remain disabled by configuration validation until response rewriting and rollout log probabilities are aligned.

## Usage

### Data Preparation

This project reuses the dataset preparation workflow from the **UPT (Unified Post-Training)** project. 

For training data, you can directly download the pre-processed `openr1.parquet` (used for LUFFY/SRFT/etc.) from [Elliott/Openr1-Math-46k-8192](https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192/tree/main) and place it in the `data` folder.

Alternatively, you can generate the data yourself using the provided script:

```bash
cd examples/algorithms/sft-rl/data
python prepare_data.py
```

### Running Experiments

To run a specific algorithm (e.g., LUFFY on Qwen2.5-Math-1.5B):

```bash
# Example: Running UFT
bash examples/algorithms/sft-rl/luffy/qwen2_5_math_1_5b_openr1_math_drgrpo.sh
```

Ensure you have configured your environment and dataset paths as described in the main [README](../../../../README.md).
