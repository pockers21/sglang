#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while IFS= read -r script; do
  bash -n "$script"
done < <(find "$ROOT/scripts" -maxdepth 1 -name '*.sh' -type f | sort)

"$PYTHON_BIN" -m py_compile "$ROOT"/scripts/*.py
"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v

echo "Lab checks passed."
