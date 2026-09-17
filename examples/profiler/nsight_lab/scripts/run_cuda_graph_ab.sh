#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: bash scripts/run_cuda_graph_ab.sh MODEL_PATH" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

OUT="${OUTPUT_ROOT:-$ROOT/results/cuda-graph-ab}"
REPEATS="${REPEATS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUT"
rm -f "$OUT/disabled.jsonl" "$OUT/full.jsonl" "$OUT/summary.json" "$OUT/summary.md"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --batch-size "${BATCH_SIZE:-1}"
  --input-len "${INPUT_LEN:-256}"
  --output-len "${OUTPUT_LEN:-128}"
  --cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}"
  --dtype "${DTYPE:-bfloat16}"
  --context-length "${CONTEXT_LENGTH:-512}"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-512}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.30}"
  --disable-radix-cache
)

run_once() {
  local backend="$1"
  local run_name="$2"
  local result_file="$3"
  local log_file="$4"
  local graph_args=(--cuda-graph-backend-decode "$backend")
  if [[ "$backend" != "disabled" ]]; then
    graph_args+=(--cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE:-1}")
  fi

  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    "$PYTHON_BIN" -m sglang.benchmark.one_batch \
      "${COMMON_ARGS[@]}" \
      "${graph_args[@]}" \
      --run-name "$run_name" \
      --result-filename "$result_file" 2>&1 | tee "$log_file"
}

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader >"$OUT/gpu_before.csv"
fi

for ((i = 1; i <= WARMUP_RUNS; i++)); do
  run_once disabled "warmup-disabled-r${i}" "" "$OUT/warmup-disabled-r${i}.log"
  run_once full "warmup-full-r${i}" "" "$OUT/warmup-full-r${i}.log"
done

for ((i = 1; i <= REPEATS; i++)); do
  if ((i % 2 == 1)); then
    ORDER=(disabled full)
  else
    ORDER=(full disabled)
  fi
  for backend in "${ORDER[@]}"; do
    run_once "$backend" "${backend}-r${i}" "$OUT/${backend}.jsonl" \
      "$OUT/${backend}-r${i}.log"
  done
done

"$PYTHON_BIN" "$ROOT/scripts/summarize_ab.py" \
  --disabled "$OUT/disabled.jsonl" \
  --full "$OUT/full.jsonl" \
  --markdown "$OUT/summary.md" \
  --json "$OUT/summary.json"

cat "$OUT/summary.md"
