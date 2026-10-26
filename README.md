<div align="center">
  <img src="assets/icon.png" alt="RLLaVA Icon" width="200">
</div>
  
# RLLaVA: An RL-central Framework for Language and Vision Assistants 🚀

<p align="center">
<img src="./assets/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org/abs/2512.21450" target="_blank">Arxiv(RLLaVA)</a> ｜
🤗 <a href="https://huggingface.co/collections/mzh12345/rllava" target="_blank">Models(RLLaVA)</a> ｜
<a href="https://qingkeai.online/archives/RLLaVA" target="_blank">Blog(RLLaVA)</a>
<p align="center">

<h5 align="center"> If you like our project, please send us a star ⭐ on GitHub.</h5>

## ✨ What's RLLaVA?

RLLaVA is a user-friendly framework for multi-modal RL. It features an **RL-central** design that decouples algorithm logic from distributed execution, enables modular customization of algorithms, models, and engines, and is optimized for resource-constrained setups to make advanced RL research more accessible.

<div align="center">
  <img src="assets/arch.png" alt="RLLaVA Architecture" width="600">
</div>

## 📰 News
*   **[2026-02-02]** 🚀 We have released implementations for a series of representative **SFT-RL Fusion Algorithms** (including SRFT, LUFFY, UFT and HPT). These methods are unified under our plugin system, demonstrating the flexibility of RLLaVA's architecture. Check out the [examples](examples/algorithms/sft-rl/README.md)!

---
  
## ✨ Why RLLaVA?

- **🎯 RL-Centric**: Implements an algorithm-driven approach tailored for RL, decoupling logic from distributed execution so researchers can focus on innovation without distributed system complexities.
- **📦 Modular Design**: Develop, extend, and customize RL algorithms and multi-modal architectures as easily as snapping together building blocks.
- **⚡ Resource-Efficient**: Optimized for resource-constrained teams—most tasks run on a single 24GB GPU, making multi-modal RL truly accessible.
- **🛠️ User-Friendly**: Minimalist code with familiar HuggingFace & PyTorch APIs for seamless setup and extensions.


## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/TinyLoopX/RLLaVA && cd RLLaVA

conda create -n rllava python==3.12 && conda activate rllava

bash ./install.sh
```

### 2. Run Examples

We provide ready-to-run scripts for various algorithms and tasks in the `examples/` directory.

```bash
# Example: Train with GRPO
bash examples/algorithms/qwen2_5_vl_3b_geoqa3k_grpo.sh
```

You can explore more examples in the directory structure:

```bash
examples/
├── algorithms/      # Algorithm comparisons and ablations (GRPO, RLOO, DAPO, etc.)
└── tasks/           # End-to-end task scripts:
    ├── math/        # Geometry, reasoning, and equation solving
    ├── counting/    # Object counting and compositional queries
    ├── grounding/   # Visual grounding and detection-style tasks
    ├── agent_search/# Web search–augmented agents
    ├── agent_code/  # Code-generation agents with tool use
    └── ...          # More real-world multi-modal benchmarks
```

### 3. Customize Your Experiment

RLLaVA makes it easy to define custom tasks. You only need 3 files:

1. **Reward function** → `examples/reward_function/your_task.py`
2. **Prompt template** → `examples/format_prompt/your_task.jinja`  
3. **Launch script / command** → Point to dataset + reward + prompt (no need to modify YAML directly):

```bash
torchrun -m rllava.train.pipeline.rlvr \
  config=examples/config.yaml \
  data.train_files=your_org/dataset@train \
  data.format_prompt=./examples/format_prompt/your_task.jinja \
  reward.reward_function=./examples/reward_function/your_task.py:compute_score \
  algorithm.adv_estimator=grpo  # Switch algorithms here (rloo, remax, ppo, etc.)
```

For detailed usage instructions, please refer to `examples/README.md`

### Single-node multi-GPU execution

Use torchrun to select the GPU count. Training uses the selected FSDP/FSDP2/ZeRO backend; generation defaults to one replica per GPU. For vLLM or SGLang, set the number of GPUs per generation replica:

```bash
torchrun --nproc_per_node=4 -m rllava.train.pipeline.rlvr \
  config=examples/config.yaml rollout.gpus_per_replica=2
```

This assigns two GPUs to each of two generation replicas, while all four GPUs process separate training samples. HF generation uses one GPU per replica. `rollout.parallel.tp_size` is the equivalent advanced setting; the previous `rollout.tensor_parallel_size` and `rollout.sglang.tp_size` configuration keys remain accepted by the config loader.

Resource layouts and batch dispatch live in `rllava/distribute/`. Training TP/CP/SP/PP/EP are not enabled by this refactor; unsupported combinations fail before models or services start. CPU/Gloo control-plane tests can be run with `python -m unittest discover -s tests -p 'test_distribute.py' -v`. Real GPU backend validation is still required.

## 📦 Supported Scope

### Algorithms
We support a broad family of RL methods, enabled by simple config switches:
- GRPO, RLOO, REINFORCE++, OPO, REMAX, GPG, PPO, DAPO, GMPO, GSPO, DR-GRPO, CLIP-COV, KL-COV

**Models:**
- Qwen2-VL/Qwen2.5-VL/Qwen3-VL vision language models
- TinyLLaVA-style architectures with customizable vision encoders, connectors, and LLMs
- Support for LLMs (e.g., Qwen3, LLaMA) in text-only RL scenarios

**Backends:**
- **Training**: FSDP, FSDP2, DeepSpeed
- **Inference**: SGLang, vLLM, HuggingFace


## 🤝 Contributing & Community

We welcome contributions! We're especially interested in new RL algorithms, multi-modal tasks, and resource-constrained improvements. Have questions? Join our WeChat group:

<div align="center">
  <img src="assets/wechat1.jpg" alt="RLLaVA WeChat Group" width="200">
</div>

## 🙏 Acknowledgements
Our RL algorithms and distributed training implementation draw inspiration from the open-source community, particularly [veRL](https://github.com/volcengine/verl), [EasyR1](https://github.com/hiyouga/EasyR1), and [AReaL](https://github.com/inclusionAI/AReaL).

## Citation

```bibtex
@misc{zhao2025rllavarlcentralframeworklanguage,
      title={RLLaVA: An RL-central Framework for Language and Vision Assistants}, 
      author={Lei Zhao and Zihao Ma and Boyu Lin and Yuhe Liu and Wenjun Wu and Lei Huang},
      year={2025},
      eprint={2512.21450},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.21450}, 
}
```
