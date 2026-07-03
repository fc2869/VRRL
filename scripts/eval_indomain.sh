#!/bin/bash
# In-domain FrozenLake eval (L3 L4 L5) for one checkpoint.
# One vLLM server per level on its own GPU; 3 clients in parallel.
# Usage: eval_indomain.sh <ckpt_dir> <run_tag> [gpu0 gpu1 gpu2]
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/frozenlake_eval_common.sh"

CKPT="${1:?Usage: $0 <ckpt_dir> <run_tag> [gpu0 gpu1 gpu2]}"
RUN_TAG="${2:?run_tag}"
GPUS=(${3:-3} ${4:-4} ${5:-5})
PORTS=(8903 8904 8905)
LEVELS=(3 4 5)
DATA="${FROZENLAKE_DATA:-$INT/data/FrozenLake/eval_data}/eval_data.json"
SCRIPT=$INT/api_inference_frozenlake.py
MERGE=$INT/merge_frozenlake_shards.py

read FMT PROMPT_VARIANT MAX_TURNS < <($PY $INT/scripts/frozenlake_model_family.py --recipe "$CKPT")
MERGED=$(ensure_merged "$CKPT")
TS=$(date +"%Y_%m_%d_%H_%M_%S")
OUT=$INT/eval_outputs/${RUN_TAG}_ID_${TS}; mkdir -p "$OUT"
echo "[id] model=$MERGED fmt=$FMT pv=$PROMPT_VARIANT mt=$MAX_TURNS gpus=${GPUS[*]} out=$OUT"

declare -a VPIDS
for i in 0 1 2; do
  VPIDS+=("$(serve_vllm "$MERGED" "${GPUS[$i]}" "${PORTS[$i]}" "$OUT/vllm_l${LEVELS[$i]}.log")")
  [ $i -lt 2 ] && sleep 30   # stagger torch.compile cache warmup
done
for i in 0 1 2; do wait_ready "${PORTS[$i]}" || { teardown "$MERGED" "${VPIDS[@]}"; exit 1; }; done

declare -a EPIDS
for i in 0 1 2; do
  LV=${LEVELS[$i]}; PORT=${PORTS[$i]}
  CURATOR_DISABLE_CACHE=1 $PY $SCRIPT \
    --model_name "$MERGED" --vllm_model_port "$PORT" \
    --model_temperature 0.0 --max_tokens 512 --max_turns "$MAX_TURNS" \
    --max_image_pixels $MAX_PIXELS --eval_dataset "$DATA" --levels "$LV" --eval_first_k 250 \
    --fmt "$FMT" --prompt_variant "$PROMPT_VARIANT" --require_all_responses false \
    --output_dir "$OUT/level_${LV}" > "$OUT/level_${LV}.log" 2>&1 &
  EPIDS+=($!)
done
FAIL=0; for p in "${EPIDS[@]}"; do wait "$p" || FAIL=$((FAIL+1)); done
teardown "$MERGED" "${VPIDS[@]}"
[ $FAIL -eq 0 ] || { echo "ERROR: $FAIL ID shard(s) failed"; exit 1; }

$PY $MERGE --shard_dirs "$OUT/level_3,$OUT/level_4,$OUT/level_5" --output_dir "$OUT"
echo "$OUT" > "/tmp/eval_${RUN_TAG}_id_outdir.txt"
$PY -c "import json;f=lambda l:json.load(open('$OUT/level_%d/metrics.json'%l))['overall']['optimal_rate']*100;print('L3 %.1f  L4 %.1f  L5 %.1f'%(f(3),f(4),f(5)))"
