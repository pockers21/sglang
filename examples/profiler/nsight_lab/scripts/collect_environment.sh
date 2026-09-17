#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
OUTPUT_FILE="${2:-}"
if [[ -z "$MODEL_PATH" || -z "$OUTPUT_FILE" ]]; then
  echo "usage: bash scripts/collect_environment.sh MODEL_PATH OUTPUT_FILE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "sglang_repo=$SGLANG_REPO_ROOT"
  echo "sglang_commit=$(git -C "$SGLANG_REPO_ROOT" rev-parse HEAD)"
  echo "sglang_branch=$(git -C "$SGLANG_REPO_ROOT" branch --show-current)"
  echo "model_path=$MODEL_PATH"
  if [[ -f "$MODEL_PATH/config.json" ]]; then
    echo "model_config_sha256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')"
  fi
  "$PYTHON_BIN" - <<'PY'
import platform
import sglang
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"sglang={getattr(sglang, '__version__', 'unknown')}")
print(f"sglang_module={sglang.__file__}")
PY
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader || true
  "${NSYS:-nsys}" --version 2>&1 || true
  "${NCU:-ncu}" --version 2>&1 | head -6 || true
} >"$OUTPUT_FILE"

cat "$OUTPUT_FILE"
