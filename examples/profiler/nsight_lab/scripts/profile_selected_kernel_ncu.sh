#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
KERNEL_CSV="${2:-${KERNEL_CSV:-}}"
if [[ -z "$MODEL_PATH" || -z "$KERNEL_CSV" ]]; then
  echo "usage: bash scripts/profile_selected_kernel_ncu.sh MODEL_PATH CUDA_GPU_KERN_SUM.csv" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/ncu}"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" "$ROOT/scripts/select_ncu_target.py" \
  --input "$KERNEL_CSV" \
  --json "$OUTPUT_ROOT/target.json"
KERNEL_FILTER="$("$PYTHON_BIN" "$ROOT/scripts/select_ncu_target.py" \
  --input "$KERNEL_CSV" --print-filter)"

OUTPUT_ROOT="$OUTPUT_ROOT" \
RUN_NAME="${RUN_NAME:-selected-hot-kernel}" \
KERNEL_NAME="$KERNEL_FILTER" \
KERNEL_NAME_BASE="${KERNEL_NAME_BASE:-function}" \
LAUNCH_SKIP="${LAUNCH_SKIP:-0}" \
LAUNCH_COUNT="${LAUNCH_COUNT:-1}" \
  CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-disabled}" \
  bash "$ROOT/scripts/profile_kernel_ncu.sh" "$MODEL_PATH"

"$PYTHON_BIN" "$ROOT/scripts/summarize_ncu.py" \
  --details "$OUTPUT_ROOT/${RUN_NAME:-selected-hot-kernel}/ncu_details.txt" \
  --target-json "$OUTPUT_ROOT/target.json" \
  --json "$OUTPUT_ROOT/summary.json" \
  --markdown "$OUTPUT_ROOT/summary.md"

cat "$OUTPUT_ROOT/summary.md"
