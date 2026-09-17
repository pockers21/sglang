#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
MODEL_PATH="${2:-${MODEL_PATH:-}}"
if [[ "$STAGE" != "prefill" && "$STAGE" != "decode" ]]; then
  echo "usage: bash scripts/profile_stage_nsys.sh prefill|decode MODEL_PATH" >&2
  exit 2
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required as the second argument or an environment variable" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results}"
DECODE_BACKEND="${CUDA_GRAPH_BACKEND_DECODE:-disabled}"
RUN_NAME="${RUN_NAME:-${STAGE}-${DECODE_BACKEND}}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
fi
OUT="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUT"

NSYS="${NSYS:-$(command -v nsys || true)}"
if [[ -z "$NSYS" || ! -x "$NSYS" ]]; then
  echo "Cannot find nsys. Put it on PATH or set NSYS=/path/to/nsys" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_PREFIX="$OUT/profile"
RESULT_FILE="$OUT/result.jsonl"
EXTRA_ARGS=()
if [[ "$DECODE_BACKEND" != "disabled" ]]; then
  EXTRA_ARGS+=(--cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE:-1}")
fi
if [[ "${ENABLE_LAYERWISE_NVTX_MARKER:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-layerwise-nvtx-marker)
fi
rm -f "$OUT/clean_${STAGE}.nsys-rep" \
  "$OUT/clean_${STAGE}.sqlite" \
  "$OUT/clean_${STAGE}.qdstrm" \
  "$RESULT_FILE"

"$NSYS" profile \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --flush-on-cudaprofilerstop=true \
  -o "$OUT/clean_${STAGE}" \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  "$PYTHON_BIN" -m sglang.benchmark.one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --batch-size "${BATCH_SIZE:-1}" \
    --input-len "${INPUT_LEN:-256}" \
    --output-len "${OUTPUT_LEN:-64}" \
    --cuda-graph-backend-decode "$DECODE_BACKEND" \
    --cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}" \
    --dtype "${DTYPE:-bfloat16}" \
    --context-length "${CONTEXT_LENGTH:-512}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS:-512}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.36}" \
    --disable-radix-cache \
    --profile \
    --profile-activities CUDA_PROFILER \
    --profile-stage "$STAGE" \
    --profile-prefix "$PROFILE_PREFIX" \
    --profile-start-step "${PROFILE_START_STEP:-16}" \
    --profile-steps "${PROFILE_STEPS:-8}" \
    --result-filename "$RESULT_FILE" \
    "${EXTRA_ARGS[@]}"

"$NSYS" stats --force-export=true --report cuda_gpu_kern_sum \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_gpu_kern_sum.txt" || true
"$NSYS" stats --force-export=true --report cuda_api_sum \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_api_sum.txt" || true
"$NSYS" stats --force-export=true --report cuda_gpu_mem_time_sum \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_gpu_mem_time_sum.txt" || true
"$NSYS" stats --force-export=true --report cuda_kern_exec_trace:nvtx-name \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_kern_exec_trace_nvtx.txt" || true
"$NSYS" stats --force-export=true --format csv --report cuda_gpu_kern_sum \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_gpu_kern_sum.csv" || true
"$NSYS" stats --force-export=true --format csv --report cuda_api_sum \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_api_sum.csv" || true
"$NSYS" stats --force-export=true --format csv --report cuda_gpu_mem_time_sum \
  "$OUT/clean_${STAGE}.nsys-rep" >"$OUT/cuda_gpu_mem_time_sum.csv" || true

echo "Clean ${STAGE} Nsight Systems report:"
echo "$OUT/clean_${STAGE}.nsys-rep"
echo "$OUT/cuda_kern_exec_trace_nvtx.txt"
