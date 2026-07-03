<div align="center">

# Visually Grounded Self-Reflection for Vision-Language Models via Reinforcement Learning

<p>
<a href="https://huggingface.co/fcyin/VRRL_qwen3_frozenlake"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow" alt="HuggingFace Models"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
<a href="https://arxiv.org/abs/2607.02490"><img src="https://img.shields.io/badge/paper-blue" alt="Paper"></a>
</p>

</div>

Large vision-language models can reason over multimodal inputs by generating textual chains of thought (CoT). A key capability exhibited in CoT reasoning is self-reflection: revisiting earlier decisions and correcting previous errors. However, existing LVLMs often fail to properly attend to **visual** inputs during reflection, limiting their ability to translate feedback into grounded corrections, especially for out-of-distribution images. To address this issue, we propose a novel reinforcement learning training framework **VRRL**, with two components explicitly designed to elicit visually grounded self-reflection. First, we randomly mask trajectory prefixes during training to emphasize recovery from incorrect intermediate predictions rather than making early mistakes. Second, we introduce buffered roll-ins from an experience replay buffer to expose the model to diverse failure states that it must learn to correct. We evaluate our approach on visual grounding tasks involving tables and charts, as well as spatial navigation benchmarks. While off-the-shelf and conventionally fine-tuned models degrade substantially under distribution shift, our method substantially improves average out-of-distribution accuracy over standard RL and reflection-oriented fine-tuning baselines by using self-reflection effectively.

---

## Released checkpoints

| Release | Base | Method | HuggingFace repo |
|---|---|---|---|
| Qwen2.5-VL-3B Multi-SFT | Qwen2.5-VL-3B | Multi-turn SFT | [`fcyin/VRRL_multi_sft_qwen2.5_frozenlake`](https://huggingface.co/fcyin/VRRL_multi_sft_qwen2.5_frozenlake) |
| Qwen2.5-VL-3B **VRRL**  | ↑ Multi-SFT | **VRRL** | [`fcyin/VRRL_qwen2.5_frozenlake`](https://huggingface.co/fcyin/VRRL_qwen2.5_frozenlake) |
| Qwen3-VL-4B Multi-SFT   | Qwen3-VL-4B | Multi-turn SFT | [`fcyin/VRRL_multi_sft_qwen3_frozenlake`](https://huggingface.co/fcyin/VRRL_multi_sft_qwen3_frozenlake) |
| Qwen3-VL-4B **VRRL**    | ↑ Multi-SFT | **VRRL** | [`fcyin/VRRL_qwen3_frozenlake`](https://huggingface.co/fcyin/VRRL_qwen3_frozenlake) |

Each is a standard, vLLM-loadable Hugging Face directory. The two Multi-SFT bases are released so you
can reproduce the full eval table and re-run VRRL from the same starting point.

---

## 1. Setup

Eval and training run in **one** conda environment.

- Python 3.12, PyTorch 2.8 (CUDA 12.8)
- `transformers==4.57.0` (loads both Qwen2.5-VL and Qwen3-VL)
- `vllm==0.11.0` (serves both families)
- `bespokelabs-curator`, `matplotlib`, `pillow`, `qwen-vl-utils`, `datasets`, `openai`

```bash
conda create -n frozenlake python=3.12 -y && conda activate frozenlake
pip install -r requirements.txt
```

The shell scripts resolve `python` / `vllm` from your active environment. To point at a specific env
without activating it, set `FROZENLAKE_PY` / `FROZENLAKE_VLLM` to its binaries.

## 2. Get the checkpoints

Checkpoints are hosted on the Hugging Face Hub (too large for git). Download them into
`checkpoints/<repo-name>` — the local dir names below are exactly what the eval matrix expects:

```bash
for repo in \
  VRRL_multi_sft_qwen2.5_frozenlake \
  VRRL_qwen2.5_frozenlake \
  VRRL_multi_sft_qwen3_frozenlake \
  VRRL_qwen3_frozenlake ; do
  huggingface-cli download fcyin/$repo --local-dir checkpoints/$repo
done
```

The eval and RL-training data are **vendored** in `data/` — no download needed.

## 3. Training

RL starts from the corresponding multi-turn SFT base and runs the exact VRRL recipe:

```bash
# Qwen3-VL-4B VRRL  ->  reproduces fcyin/VRRL_qwen3_frozenlake
MODEL_PATH=checkpoints/VRRL_multi_sft_qwen3_frozenlake \
DATA_ROOT=data/FrozenLake/rl_train/rl_3k \
bash EasyR1/training/train_frozenlake_vrrl_qwen3_vl_4b_4gpu.sh
```

Key pieces:

- **Launcher:** `EasyR1/training/train_frozenlake_vrrl_qwen3_vl_4b_4gpu.sh`
- **Config:** `EasyR1/training/config_frozenlake_multi_turn_reflection.yaml`.
- **Reward:** `EasyR1/training/reward_function/frozenlake_score.py`
- **Prompt template:** `EasyR1/training/format_prompt/frozenlake_reflection.jinja`.
- **Trainer:** verl-based multi-turn GRPO under `EasyR1/verl/`.

Set `WANDB_API_KEY` in your shell to enable Weights & Biases logging (optional). Checkpoints and
rollout artifacts are written under `outputs/`. The released VRRL checkpoint is the step
with the best balanced in-domain / OOD optimal rate for the run.

## 4. Evaluation

The eval loop drives an external `vllm serve` instance through the multi-turn protocol
(`api_inference_frozenlake.py`). Each turn:

1. The model sees the grid image and proposes a full action sequence (`route`).
2. The environment renders the proposed trajectory on the grid and returns the new image.
3. The model inspects the feedback image and either `terminate`s or proposes a corrected route.
4. Scoring (`frozenlake_scoring.py`) checks the final trajectory for exact optimality.

Levels **L3–L5** are in-domain (3×3–5×5 grids); **L6–L7** are out-of-distribution (6×6–7×7).

One command runs OOD (L6+L7) first across GPUs, then in-domain (L3–L5) per checkpoint:

```bash
conda activate frozenlake
bash scripts/run_matrix_eval.sh          # uses checkpoints/ + data/ by default
```

Each level's `metrics.json` reports the **optimal rate (exact match)** (`optimal_rate`); compare it against the
reference table below.

To eval a single checkpoint directly:

```bash
bash scripts/eval_ood.sh      checkpoints/VRRL_qwen3_frozenlake qwen3_vrrl 0 8900   # L6+L7
bash scripts/eval_indomain.sh checkpoints/VRRL_qwen3_frozenlake qwen3_vrrl 0 1 2    # L3-L5
```
## TO-DOs
[x] RL training and evaluation scripts for spatial navigation tasks.

[ ] RL training and evaluation scripts for visual grounding tasks.

[ ] SFT training data and llamafactory SFT training configs.


## License

Released under the **Apache License 2.0** (see [`LICENSE`](./LICENSE)). The `EasyR1/verl` trainer is
derived from [ByteDance verl](https://github.com/volcengine/verl) / EasyR1, also Apache-2.0.

## Citation

Please consider citing our paper if you find our codebase useful. 

```bibtex
@article{tang2026vrrl,
      title={Visually Grounded Self-Reflection for Vision-Language Models via Reinforcement Learning}, 
      author={Liyan Tang and Fangcong Yin and Greg Durrett},
      year={2026},
      eprint={2607.02490},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.02490}, 
}
```