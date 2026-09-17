#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: bash scripts/run_full_study.sh MODEL_PATH" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_NAME="${STUDY_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
STUDY_ROOT="${STUDY_ROOT:-$ROOT/results/study-$STUDY_NAME}"
mkdir -p "$STUDY_ROOT"

# Keep the unprofiled A/B and the diagnostic traces on one workload shape.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export INPUT_LEN="${INPUT_LEN:-256}"
export OUTPUT_LEN="${OUTPUT_LEN:-128}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-512}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.30}"
export DTYPE="${DTYPE:-bfloat16}"
export CUDA_GRAPH_BACKEND_PREFILL="${CUDA_GRAPH_BACKEND_PREFILL:-disabled}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-1}"
export PROFILE_START_STEP="${PROFILE_START_STEP:-32}"
export PROFILE_STEPS="${PROFILE_STEPS:-16}"
export REPEATS="${REPEATS:-5}"
export WARMUP_RUNS="${WARMUP_RUNS:-1}"

{
  echo "study_name=$STUDY_NAME"
  echo "model_path=$MODEL_PATH"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}"
  echo "batch_size=$BATCH_SIZE"
  echo "input_len=$INPUT_LEN"
  echo "output_len=$OUTPUT_LEN"
  echo "context_length=$CONTEXT_LENGTH"
  echo "max_total_tokens=$MAX_TOTAL_TOKENS"
  echo "mem_fraction_static=$MEM_FRACTION_STATIC"
  echo "dtype=$DTYPE"
  echo "cuda_graph_backend_prefill=$CUDA_GRAPH_BACKEND_PREFILL"
  echo "cuda_graph_bs_decode=$CUDA_GRAPH_BS_DECODE"
  echo "profile_start_step=$PROFILE_START_STEP"
  echo "profile_steps=$PROFILE_STEPS"
  echo "repeats=$REPEATS"
  echo "warmup_runs=$WARMUP_RUNS"
} >"$STUDY_ROOT/configuration.txt"

bash "$ROOT/scripts/collect_environment.sh" "$MODEL_PATH" "$STUDY_ROOT/environment.txt"

OUTPUT_ROOT="$STUDY_ROOT/ab" \
  bash "$ROOT/scripts/run_cuda_graph_ab.sh" "$MODEL_PATH"

OUTPUT_ROOT="$STUDY_ROOT/nsys" \
  RUN_NAME="prefill-disabled" \
  CUDA_GRAPH_BACKEND_DECODE=disabled \
  bash "$ROOT/scripts/profile_stage_nsys.sh" prefill "$MODEL_PATH"

for backend in disabled full; do
  OUTPUT_ROOT="$STUDY_ROOT/nsys" \
    RUN_NAME="decode-$backend" \
    CUDA_GRAPH_BACKEND_DECODE="$backend" \
    bash "$ROOT/scripts/profile_stage_nsys.sh" decode "$MODEL_PATH"
done

"${PYTHON_BIN:-python3}" "$ROOT/scripts/summarize_nsys.py" \
  --nsys-root "$STUDY_ROOT/nsys" \
  --markdown "$STUDY_ROOT/nsys_summary.md" \
  --json "$STUDY_ROOT/nsys_summary.json"

echo "Full study completed: $STUDY_ROOT"
