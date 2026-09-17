#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: bash scripts/run_offline_sweep.sh MODEL_PATH" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

OUT="${OUTPUT_ROOT:-$ROOT/results/offline-sweep}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
read -r -a BATCH_VALUES <<<"${BATCH_SIZES:-1 4 16}"
read -r -a INPUT_VALUES <<<"${INPUT_LENS:-128 512 2048}"
read -r -a OUTPUT_VALUES <<<"${OUTPUT_LENS:-32 128}"

max_value() {
  local maximum=0 value
  for value in "$@"; do
    if ((value > maximum)); then
      maximum="$value"
    fi
  done
  echo "$maximum"
}

MAX_BATCH="$(max_value "${BATCH_VALUES[@]}")"
MAX_INPUT="$(max_value "${INPUT_VALUES[@]}")"
MAX_OUTPUT="$(max_value "${OUTPUT_VALUES[@]}")"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-$((MAX_INPUT + MAX_OUTPUT))}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-$((MAX_BATCH * (MAX_INPUT + MAX_OUTPUT) + 256))}"
DECODE_BACKEND="${CUDA_GRAPH_BACKEND_DECODE:-full}"

mkdir -p "$OUT"
rm -f "$OUT/results.jsonl" "$OUT/summary.json" "$OUT/summary.md" "$OUT/run.log"

GRAPH_ARGS=(--cuda-graph-backend-decode "$DECODE_BACKEND")
if [[ "$DECODE_BACKEND" != "disabled" ]]; then
  if [[ -n "${CUDA_GRAPH_BS_DECODE:-}" ]]; then
    GRAPH_ARGS+=(--cuda-graph-bs-decode "$CUDA_GRAPH_BS_DECODE")
  else
    GRAPH_ARGS+=(--cuda-graph-max-bs-decode "$MAX_BATCH")
  fi
fi

env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  "$PYTHON_BIN" -m sglang.benchmark.one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --batch-size "${BATCH_VALUES[@]}" \
    --input-len "${INPUT_VALUES[@]}" \
    --output-len "${OUTPUT_VALUES[@]}" \
    "${GRAPH_ARGS[@]}" \
    --cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}" \
    --dtype "${DTYPE:-bfloat16}" \
    --context-length "$CONTEXT_LENGTH" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.60}" \
    --disable-radix-cache \
    --run-name "${RUN_NAME:-offline-shape-sweep}" \
    --result-filename "$OUT/results.jsonl" 2>&1 | tee "$OUT/run.log"

"$PYTHON_BIN" "$ROOT/scripts/summarize_offline_sweep.py" \
  --input "$OUT/results.jsonl" \
  --json "$OUT/summary.json" \
  --markdown "$OUT/summary.md"

cat "$OUT/summary.md"
