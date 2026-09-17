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
RUN_NAME="${RUN_NAME:-sglang-one-batch-ncu}"
OUT="$OUTPUT_ROOT/$RUN_NAME"
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

if [[ -n "${NCU_SECTIONS:-}" ]]; then
  read -r -a NCU_SECTION_VALUES <<<"$NCU_SECTIONS"
elif [[ -n "${NCU_SECTION:-}" ]]; then
  NCU_SECTION_VALUES=("$NCU_SECTION")
else
  NCU_SECTION_VALUES=(
    SpeedOfLight
    MemoryWorkloadAnalysis
    SchedulerStats
    WarpStateStats
    SourceCounters
    Occupancy
    InstructionStats
  )
fi
NCU_SECTION_ARGS=()
for section in "${NCU_SECTION_VALUES[@]}"; do
  NCU_SECTION_ARGS+=(--section "$section")
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORT="$OUT/sglang_one_batch_ncu"
DECODE_BACKEND="${CUDA_GRAPH_BACKEND_DECODE:-disabled}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-1}"
INPUT_LEN_VALUE="${INPUT_LEN:-16}"
OUTPUT_LEN_VALUE="${OUTPUT_LEN:-4}"
MIN_CONTEXT_LENGTH=$((INPUT_LEN_VALUE + OUTPUT_LEN_VALUE))
MIN_TOTAL_TOKENS=$((BATCH_SIZE_VALUE * MIN_CONTEXT_LENGTH))
CONTEXT_LENGTH_VALUE="${CONTEXT_LENGTH:-$MIN_CONTEXT_LENGTH}"
MAX_TOTAL_TOKENS_VALUE="${MAX_TOTAL_TOKENS:-$MIN_TOTAL_TOKENS}"

if ((CONTEXT_LENGTH_VALUE < MIN_CONTEXT_LENGTH)); then
  echo "CONTEXT_LENGTH=$CONTEXT_LENGTH_VALUE is too small; need at least $MIN_CONTEXT_LENGTH for input=$INPUT_LEN_VALUE output=$OUTPUT_LEN_VALUE" >&2
  exit 2
fi
if ((MAX_TOTAL_TOKENS_VALUE < MIN_TOTAL_TOKENS)); then
  echo "MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS_VALUE is too small; need at least $MIN_TOTAL_TOKENS for batch=$BATCH_SIZE_VALUE input=$INPUT_LEN_VALUE output=$OUTPUT_LEN_VALUE" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "$DECODE_BACKEND" != "disabled" ]]; then
  EXTRA_ARGS+=(--cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE:-1}")
fi

"$NCU" \
  "${NCU_SECTION_ARGS[@]}" \
  --target-processes all \
  --replay-mode kernel \
  "${KERNEL_FILTER_ARGS[@]+"${KERNEL_FILTER_ARGS[@]}"}" \
  --launch-skip "${LAUNCH_SKIP:-40}" \
  --launch-count "${LAUNCH_COUNT:-1}" \
  --force-overwrite \
  -o "$REPORT" \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  "$PYTHON_BIN" -m sglang.benchmark.one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --batch-size "$BATCH_SIZE_VALUE" \
    --input-len "$INPUT_LEN_VALUE" \
    --output-len "$OUTPUT_LEN_VALUE" \
    --cuda-graph-backend-decode "$DECODE_BACKEND" \
    --cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}" \
    --dtype "${DTYPE:-bfloat16}" \
    --context-length "$CONTEXT_LENGTH_VALUE" \
    --max-total-tokens "$MAX_TOTAL_TOKENS_VALUE" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.36}" \
    --disable-radix-cache \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

"$NCU" --import "$REPORT.ncu-rep" --page details >"$OUT/ncu_details.txt" || true

echo "Nsight Compute report:"
echo "$REPORT.ncu-rep"
