#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: bash scripts/run_complete_lab.sh MODEL_PATH" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LAB_NAME="${LAB_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
LAB_RUN_ROOT="${LAB_RUN_ROOT:-$ROOT/results/complete-$LAB_NAME}"
RESUME="${RESUME:-0}"
STATUS_FILE="$LAB_RUN_ROOT/status.json"
CURRENT_STAGE="initializing"
COMPLETE=0
mkdir -p "$LAB_RUN_ROOT"

write_status() {
  "$PYTHON_BIN" - "$STATUS_FILE" "$1" "$2" <<'PY'
import datetime
import json
import sys
from pathlib import Path

path, status, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path.write_text(
    json.dumps(
        {
            "status": status,
            "stage": stage,
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

summary_is_valid() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
json.loads(path.read_text(encoding="utf-8"))
PY
}

finish_status() {
  local exit_code=$?
  if ((COMPLETE == 0)); then
    write_status FAILED "$CURRENT_STAGE"
  fi
  exit "$exit_code"
}
trap finish_status EXIT INT TERM
write_status RUNNING "$CURRENT_STAGE"

CURRENT_STAGE="cuda-graph-and-nsys"
write_status RUNNING "$CURRENT_STAGE"
if [[ "$RESUME" == "1" ]] && summary_is_valid "$LAB_RUN_ROOT/core-study/nsys_summary.json"; then
  echo "Resume: keeping completed CUDA Graph and Nsight Systems study"
else
  STUDY_ROOT="$LAB_RUN_ROOT/core-study" \
    bash "$ROOT/scripts/run_full_study.sh" "$MODEL_PATH"
fi

CURRENT_STAGE="offline-shape-sweep"
write_status RUNNING "$CURRENT_STAGE"
if [[ "$RESUME" == "1" ]] && summary_is_valid "$LAB_RUN_ROOT/offline/summary.json"; then
  echo "Resume: keeping completed offline shape sweep"
else
  OUTPUT_ROOT="$LAB_RUN_ROOT/offline" \
    bash "$ROOT/scripts/run_offline_sweep.sh" "$MODEL_PATH"
fi

CURRENT_STAGE="online-serving-sweep"
write_status RUNNING "$CURRENT_STAGE"
if [[ "$RESUME" == "1" ]] && summary_is_valid "$LAB_RUN_ROOT/serving/summary.json"; then
  echo "Resume: keeping completed online serving sweep"
else
  OUTPUT_ROOT="$LAB_RUN_ROOT/serving" \
    bash "$ROOT/scripts/run_serving_sweep.sh" "$MODEL_PATH"
fi

if [[ "${RUN_NCU:-1}" == "1" ]]; then
  CURRENT_STAGE="targeted-ncu"
  write_status RUNNING "$CURRENT_STAGE"
  if [[ "$RESUME" == "1" ]] && summary_is_valid "$LAB_RUN_ROOT/ncu/summary.json"; then
    echo "Resume: keeping completed targeted Nsight Compute profile"
  else
    OUTPUT_ROOT="$LAB_RUN_ROOT/ncu" \
    INPUT_LEN="${INPUT_LEN:-256}" \
    OUTPUT_LEN="${NCU_OUTPUT_LEN:-4}" \
      bash "$ROOT/scripts/profile_selected_kernel_ncu.sh" \
        "$MODEL_PATH" \
        "$LAB_RUN_ROOT/core-study/nsys/decode-disabled/cuda_gpu_kern_sum.csv"
  fi
fi

CURRENT_STAGE="diagnosis"
write_status RUNNING "$CURRENT_STAGE"
DIAGNOSIS_ARGS=(
  --ab-json "$LAB_RUN_ROOT/core-study/ab/summary.json"
  --nsys-json "$LAB_RUN_ROOT/core-study/nsys_summary.json"
  --offline-json "$LAB_RUN_ROOT/offline/summary.json"
  --serving-json "$LAB_RUN_ROOT/serving/summary.json"
  --json "$LAB_RUN_ROOT/diagnosis.json"
  --markdown "$LAB_RUN_ROOT/diagnosis.md"
)
if [[ -f "$LAB_RUN_ROOT/ncu/summary.json" ]]; then
  DIAGNOSIS_ARGS+=(--ncu-summary-json "$LAB_RUN_ROOT/ncu/summary.json")
elif [[ -f "$LAB_RUN_ROOT/ncu/target.json" ]]; then
  DIAGNOSIS_ARGS+=(--ncu-target-json "$LAB_RUN_ROOT/ncu/target.json")
fi
"$PYTHON_BIN" "$ROOT/scripts/diagnose.py" "${DIAGNOSIS_ARGS[@]}"

CURRENT_STAGE="complete"
COMPLETE=1
write_status COMPLETE "$CURRENT_STAGE"
trap - EXIT INT TERM
echo "Complete lab finished: $LAB_RUN_ROOT"
cat "$LAB_RUN_ROOT/diagnosis.md"
