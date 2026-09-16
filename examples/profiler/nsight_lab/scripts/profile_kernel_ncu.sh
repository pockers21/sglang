#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: bash scripts/profile_kernel_ncu.sh MODEL_PATH" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results}"
OUT="$OUTPUT_ROOT/sglang-one-batch-ncu"
mkdir -p "$OUT"

NCU="${NCU:-$(command -v ncu || true)}"
if [[ -z "$NCU" || ! -x "$NCU" ]]; then
  echo "Cannot find ncu. Put it on PATH or set NCU=/path/to/ncu" >&2
  exit 1
fi

KERNEL_FILTER_ARGS=()
if [[ -n "${KERNEL_NAME:-}" ]]; then
  KERNEL_FILTER_ARGS+=(--kernel-name "$KERNEL_NAME")
  KERNEL_FILTER_ARGS+=(--kernel-name-base "${KERNEL_NAME_BASE:-function}")
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT="$OUT/sglang_one_batch_ncu"

"$NCU" \
  --section "${NCU_SECTION:-SpeedOfLight}" \
  --target-processes all \
  --replay-mode kernel \
  "${KERNEL_FILTER_ARGS[@]}" \
  --launch-skip "${LAUNCH_SKIP:-40}" \
  --launch-count "${LAUNCH_COUNT:-1}" \
  --force-overwrite \
  -o "$REPORT" \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PYTHON_BIN" -m sglang.benchmark.one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --batch-size "${BATCH_SIZE:-1}" \
    --input-len "${INPUT_LEN:-16}" \
    --output-len "${OUTPUT_LEN:-4}" \
    --cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE:-disabled}" \
    --cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}" \
    --dtype "${DTYPE:-bfloat16}" \
    --context-length "${CONTEXT_LENGTH:-128}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS:-128}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.36}" \
    --disable-radix-cache

"$NCU" --import "$REPORT.ncu-rep" --page details >"$OUT/ncu_details.txt" || true

echo "Nsight Compute report:"
echo "$REPORT.ncu-rep"
