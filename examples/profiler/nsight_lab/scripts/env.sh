#!/usr/bin/env bash

# Source this file from the other lab scripts. It keeps machine-specific paths
# outside the repository and imports SGLang from the current source checkout.

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$LAB_ROOT/../../.." && pwd)}"

if [[ -n "${VENV_PATH:-}" ]]; then
  if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "VENV_PATH does not contain bin/activate: $VENV_PATH" >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

if [[ ! -d "$SGLANG_REPO_ROOT/python/sglang" ]]; then
  echo "Cannot find SGLang sources under: $SGLANG_REPO_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

export LAB_ROOT SGLANG_REPO_ROOT
export PYTHONPATH="$SGLANG_REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${CUDA_COMPAT_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
