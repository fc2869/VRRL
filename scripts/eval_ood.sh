#!/bin/bash
# OOD FrozenLake eval (L6 + L7) for one checkpoint.
# L6 <- eval_data.json ; L7 <- eval_data/L7/eval_data.json. One server, two parallel level shards.
# Usage: eval_ood.sh <ckpt_dir> <run_tag> [gpu] [port]
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/frozenlake_eval_common.sh"

CKPT="${1:?Usage: $0 <ckpt_dir> <run_tag> [gpu] [port]}"
RUN_TAG="${2:?run_tag}"
GPU="${3:-0}"
PORT="${4:-8900}"

DATA_DIR="${FROZENLAKE_DATA:-$INT/data/FrozenLake/eval_data}"
DATA_L6="$DATA_DIR/eval_data.json"
DATA_L7="$DATA_DIR/L7/eval_data.json"
SCRIPT=$INT/api_inference_frozenlake.py
MERGE=$INT/merge_frozenlake_shards.py

read FMT PROMPT_VARIANT MAX_TURNS < <($PY $INT/scripts/frozenlake_model_family.py --recipe "$CKPT")
MERGED=$(ensure_merged "$CKPT")
TS=$(date +"%Y_%m_%d_%H_%M_%S")
OUT=$INT/eval_outputs/${RUN_TAG}_OOD_${TS}; mkdir -p "$OUT"
echo "[ood] model=$MERGED fmt=$FMT pv=$PROMPT_VARIANT mt=$MAX_TURNS gpu=$GPU port=$PORT out=$OUT"

VPID=$(serve_vllm "$MERGED" "$GPU" "$PORT" "$OUT/vllm.log")
wait_ready "$PORT" || { teardown "$MERGED" "$VPID"; exit 1; }

declare -a EPIDS
for pair in "6 $DATA_L6" "7 $DATA_L7"; do
  set -- $pair; LV=$1; DJ=$2
  CURATOR_DISABLE_CACHE=1 $PY $SCRIPT \
    --model_name "$MERGED" --vllm_model_port "$PORT" \
    --model_temperature 0.0 --max_tokens 512 --max_turns "$MAX_TURNS" \
    --max_image_pixels $MAX_PIXELS --eval_dataset "$DJ" --levels "$LV" --eval_first_k 250 \
    --fmt "$FMT" --prompt_variant "$PROMPT_VARIANT" --require_all_responses false \
    --output_dir "$OUT/level_${LV}" > "$OUT/level_${LV}.log" 2>&1 &
  EPIDS+=($!)
done
FAIL=0; for p in "${EPIDS[@]}"; do wait "$p" || FAIL=$((FAIL+1)); done
teardown "$MERGED" "$VPID"
[ $FAIL -eq 0 ] || { echo "ERROR: $FAIL OOD shard(s) failed"; exit 1; }

$PY $MERGE --shard_dirs "$OUT/level_6,$OUT/level_7" --output_dir "$OUT"
echo "$OUT" > "/tmp/eval_${RUN_TAG}_ood_outdir.txt"
$PY -c "import json;m=json.load(open('$OUT/level_6/metrics.json'));n=json.load(open('$OUT/level_7/metrics.json'));print('L6 EM=%.1f  L7 EM=%.1f'%(m['overall']['optimal_rate']*100,n['overall']['optimal_rate']*100))"
