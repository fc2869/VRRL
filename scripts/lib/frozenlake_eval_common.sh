# Shared FrozenLake eval orchestration helpers.
# Source this from eval_ood.sh / eval_indomain.sh.
#
# Configuration via environment (sensible defaults shown):
#   FROZENLAKE_PY     python interpreter   (default: `python` on PATH)
#   FROZENLAKE_VLLM   vllm launcher        (default: `vllm` on PATH)
#   FROZENLAKE_DATA   eval data directory  (default: <repo>/data/FrozenLake/eval_data)
# Activate your env first (e.g. `conda activate frozenlake`) so python/vllm resolve,
# or point FROZENLAKE_PY / FROZENLAKE_VLLM at the env's binaries.
REPO_ROOT="${FROZENLAKE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
INT="$REPO_ROOT"                              # alias used throughout the eval scripts
PY="${FROZENLAKE_PY:-python}"
VLLM="${FROZENLAKE_VLLM:-vllm}"
MAX_PIXELS="${FROZENLAKE_MAX_PIXELS:-1254400}"

_has_weights() {  # <dir> -> 0 if a vLLM-loadable safetensors dir
  [ -f "$1/model.safetensors.index.json" ] || [ -f "$1/model.safetensors" ]
}

_merge_actor() {  # <actor_dir>: merge FSDP shards into <actor_dir>/huggingface if not already
  local actor="$1"
  if ! _has_weights "$actor/huggingface"; then
    echo "[ensure_merged] merging FSDP shards under $actor ..." >&2
    $PY "$INT/EasyR1/scripts/model_merger.py" --local_dir "$actor" >&2
  fi
}

ensure_merged() {  # <ckpt_dir> -> prints a vLLM-loadable dir on stdout
  local d="$1"
  # (a) already a merged HF dir
  if _has_weights "$d"; then echo "$d"; return 0; fi
  # (b) d IS an FSDP actor dir (has rank shards + a huggingface/ with config+tokenizer)
  if ls "$d"/model_world_size_*_rank_*.pt >/dev/null 2>&1 || \
     { [ -d "$d/huggingface" ] && [ "$(basename "$d")" = "actor" ]; }; then
    _merge_actor "$d"
    echo "$d/huggingface"; return 0
  fi
  # (c) d is a step dir containing actor/
  if [ -d "$d/actor" ]; then
    _merge_actor "$d/actor"
    echo "$d/actor/huggingface"; return 0
  fi
  echo "ERROR: $d has no weights, no rank shards, and no actor/" >&2; return 1
}

serve_vllm() {  # <merged_dir> <gpu> <port> <logfile> -> prints PID on stdout
  local merged="$1" gpu="$2" port="$3" log="$4"
  # vLLM's compiled _C extension needs a working libcublas. Preload one via
  # FROZENLAKE_LD_PRELOAD (default: the fbcode-platform lib); skipped when the
  # file is absent, so this is a no-op on standard CUDA installs.
  # (serve_vllm is always called via $(...) command substitution, so this export
  # is scoped to that subshell + the vLLM child and does not leak to the caller.)
  local _ldp="${FROZENLAKE_LD_PRELOAD:-/usr/local/fbcode/platform010/lib/libcublas.so.12}"
  [ -f "$_ldp" ] && export LD_PRELOAD="${_ldp}${LD_PRELOAD:+:$LD_PRELOAD}"
  # --mm-processor-cache-gb 0 disables the vLLM 0.11 multimodal-processor cache,
  # which otherwise hits an intermittent mm_hash AssertionError under the
  # multi-turn image feedback loop (crashes EngineCore on OOD levels).
  VLLM_CACHE_ROOT="$(dirname "$log")/.vllm_cache_p${port}" \
  CUDA_VISIBLE_DEVICES="$gpu" "$VLLM" serve "$merged" \
    --trust-remote-code --host localhost --port "$port" \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --max-model-len 32000 --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
    --mm-processor-cache-gb 0 \
    > "$log" 2>&1 &
  echo $!
}

wait_ready() {  # <port> -> 0 when /v1/models responds, else 1 after ~20 min
  local port="$1" a
  for a in $(seq 1 240); do
    curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "ERROR: vLLM port $port never became ready" >&2
  return 1
}

teardown() {  # <merged_dir> <pid...>
  local merged="$1"; shift
  local pid
  for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
  sleep 5
  pkill -9 -f "vllm serve $merged" 2>/dev/null || true
}
