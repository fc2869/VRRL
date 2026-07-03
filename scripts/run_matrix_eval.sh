#!/bin/bash
# Reproduce the EM numbers for the released checkpoints. OOD (L6+L7) for all
# checkpoints first (across GPUs 0..N-1), then in-domain (L3-L5) per checkpoint.
#
# Place the checkpoints under $CKPT_ROOT (default: <repo>/checkpoints/<name>), e.g.
#   huggingface-cli download fcyin/VRRL_qwen3_frozenlake \
#       --local-dir checkpoints/VRRL_qwen3_frozenlake
# Activate the env (conda activate frozenlake) or set FROZENLAKE_PY / FROZENLAKE_VLLM.
set -eo pipefail
EV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EV/lib/frozenlake_eval_common.sh"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints}"

# checkpoint dir (under CKPT_ROOT) : run tag
declare -a CK=(
  "VRRL_multi_sft_qwen2.5_frozenlake:qwen25_multi_sft"
  "VRRL_qwen2.5_frozenlake:qwen25_vrrl"
  "VRRL_multi_sft_qwen3_frozenlake:qwen3_multi_sft"
  "VRRL_qwen3_frozenlake:qwen3_vrrl"
)

# --- OOD first: all checkpoints in parallel, one GPU each ---
declare -a PIDS; i=0
for entry in "${CK[@]}"; do
  CKPT="$CKPT_ROOT/${entry%%:*}"; TAG="${entry##*:}"; PORT=$((8900+i))
  [ -d "$CKPT" ] || { echo "MISSING checkpoint: $CKPT (download from HF first)"; exit 1; }
  bash "$EV/eval_ood.sh" "$CKPT" "$TAG" "$i" "$PORT" > "/tmp/${TAG}_ood.log" 2>&1 &
  PIDS+=($!); i=$((i+1))
  sleep 25   # stagger vllm serve launches (torch.compile cache race)
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo "==== OOD done for all ===="

# --- In-domain: per checkpoint, 3 GPUs each, sequential ---
for entry in "${CK[@]}"; do
  CKPT="$CKPT_ROOT/${entry%%:*}"; TAG="${entry##*:}"
  bash "$EV/eval_indomain.sh" "$CKPT" "$TAG" 0 1 2
done
echo "==== ID done for all ===="
