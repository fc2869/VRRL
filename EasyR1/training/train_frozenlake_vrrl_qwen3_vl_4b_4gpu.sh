#!/bin/bash
# ============================================================================
# VRRL training — Qwen3-VL-4B (4 GPUs)
#
# Multi-turn reflection GRPO from the multi-turn SFT base. Low-LR EM outcome
# reward with a reflection bonus and a flat per-revision step-cost, online
# filtering, and a prefix-buffer roll-in. Produces the released
# `fcyin/VRRL_qwen3_frozenlake` checkpoint.
#
# Global batch size (32) is GPU-count independent; this runs it across 4 GPUs.
# ============================================================================
set -e
set -x

# ---- environment: activate your env first (e.g. conda activate frozenlake), or set FROZENLAKE_PY
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${FROZENLAKE_PY:-python}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# Prefer gcc-13 if present, else fall back to the system gcc (some hosts only
# ship an unversioned gcc). Override by exporting CC/CXX before launch.
export CC=${CC:-$(command -v gcc-13 || command -v gcc)}
export CXX=${CXX:-$(command -v g++-13 || command -v g++)}
# vLLM's compiled _C extension (used by the rollout engine) needs a working
# libcublas; preload one via FROZENLAKE_LD_PRELOAD (default: the fbcode-platform
# lib), skipped when the file is absent so it is a no-op on standard installs.
_FROZENLAKE_LDP="${FROZENLAKE_LD_PRELOAD:-/usr/local/fbcode/platform010/lib/libcublas.so.12}"
[ -f "$_FROZENLAKE_LDP" ] && export LD_PRELOAD="${_FROZENLAKE_LDP}${LD_PRELOAD:+:$LD_PRELOAD}"
# Set WANDB_API_KEY in your shell to enable Weights & Biases logging (optional).
export WANDB_CONSOLE=off
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vc_vrrl_q3_$$}
export RAY_ADDRESS=local
export RAY_TMPDIR=/tmp/r_vrrl_q3_$$

MODEL_PATH=${MODEL_PATH:-$REPO_ROOT/checkpoints/VRRL_multi_sft_qwen3_frozenlake}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data/FrozenLake/rl_train/rl_3k}
IMAGE_DIR=${IMAGE_DIR:-$REPO_ROOT/data/FrozenLake/rl_train/rl_3k}

BASE_EXPERIMENT_NAME=frozenlake_qwen3_vl_4b_vrrl
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FULL_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${TIMESTAMP}"
FULL_SAVE_PATH=${OUTPUT_ROOT:-$REPO_ROOT/outputs}/${FULL_EXPERIMENT_NAME}
TEMP_DIR=${OUTPUT_ROOT:-$REPO_ROOT/outputs}/rollouts/${FULL_EXPERIMENT_NAME}

mkdir -p "${FULL_SAVE_PATH}" "${TEMP_DIR}"
echo "Starting VRRL training (Qwen3-VL-4B): ${FULL_EXPERIMENT_NAME}"

cd "$REPO_ROOT"

$PY -m EasyR1.verl.trainer.main \
    config=EasyR1/training/config_frozenlake_multi_turn_reflection.yaml \
    data.train_files=${DATA_ROOT}/hf_dataset@train \
    data.val_files=${DATA_ROOT}/hf_dataset@train \
    data.image_dir=${IMAGE_DIR} \
    data.max_pixels=1254400 \
    data.max_prompt_length=2000 \
    data.max_response_length=8000 \
    data.rollout_batch_size=32 \
    data.shuffle=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=32 \
    worker.actor.optim.lr=5e-7 \
    worker.rollout.fmt=reflection_tag \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.65 \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.max_num_batched_tokens=12288 \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    worker.rollout.max_turns=8 \
    worker.rollout.num_llm_calls_available=8 \
    worker.rollout.temp_dir=${TEMP_DIR} \
    worker.rollout.per_question_mode_selection=true \
    worker.rollout.normal_mode_weight=0 \
    worker.rollout.random_start_mode_weight=0.6 \
    worker.rollout.prefix_buffer_mode_weight=0.4 \
    worker.rollout.prefix_buffer_size=500 \
    worker.rollout.prefix_buffer_min_size=32 \
    worker.rollout.prefix_buffer_wrong_ratio=0.9 \
    worker.rollout.prefix_buffer_max_staleness_steps=50 \
    worker.rollout.prefix_buffer_force_pointing=true \
    worker.rollout.prefix_buffer_max_pointing_turns=2 \
    worker.rollout.prefix_buffer_correct_replay=false \
    worker.rollout.prefix_buffer_recycle=false \
    worker.rollout.random_start_turn_min=0 \
    worker.rollout.random_start_mask_all_until_final=false \
    worker.reward.reward_function_kwargs.fmt=reflection_tag \
    worker.reward.reward_function_kwargs.outcome_reward=em \
    worker.reward.reward_function_kwargs.pr_overshoot_zero=true \
    worker.reward.reward_function_kwargs.lambda_deg=0.5 \
    worker.reward.reward_function_kwargs.reflect_weight=0.9 \
    worker.reward.reward_function_kwargs.reflect_lower_clip=0 \
    worker.reward.reward_function_kwargs.min_turns=1 \
    worker.reward.reward_function_kwargs.correct_value=1.0 \
    worker.reward.reward_function_kwargs.em_reflect_bonus_weight=0.1 \
    worker.reward.reward_function_kwargs.em_reflect_step_discount=1.0 \
    worker.reward.reward_function_kwargs.step_cost=0.01 \
    worker.reward.reward_function_kwargs.step_cost_churn_only=false \
    worker.reward.reward_function_kwargs.log_filename=${FULL_SAVE_PATH}/rewards.jsonl \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    algorithm.kl_coef=0.01 \
    algorithm.online_filtering=true \
    algorithm.filter_key=overall \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.project_name=VisualReasonFrozenLake \
    trainer.experiment_name=${FULL_EXPERIMENT_NAME} \
    trainer.logger='["console","wandb"]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=3 \
    trainer.max_steps=1200 \
    trainer.max_try_make_batch=20 \
    trainer.val_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_freq=15 \
    trainer.save_limit=20 \
    trainer.save_model_only=true \
    trainer.save_checkpoint_path=${FULL_SAVE_PATH}
