#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "usage: bash scripts/run_serving_sweep.sh MODEL_PATH" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

OUT="${OUTPUT_ROOT:-$ROOT/results/serving-sweep}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${SERVER_PORT:-30000}"
BASE_URL="http://$HOST:$PORT"
read -r -a CONCURRENCY_VALUES <<<"${CONCURRENCIES:-1 8 32}"
read -r -a EXTRA_SERVER_ARGS <<<"${SGLANG_SERVER_EXTRA_ARGS:-}"
MAX_CONCURRENCY=0
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
  if ((concurrency > MAX_CONCURRENCY)); then
    MAX_CONCURRENCY="$concurrency"
  fi
done
TOKENIZE_ARGS=()
if [[ "${TOKENIZE_PROMPT:-1}" == "1" ]]; then
  TOKENIZE_ARGS+=(--tokenize-prompt)
fi
SERVER_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

mkdir -p "$OUT"
rm -f "$OUT/results.jsonl" "$OUT/summary.json" "$OUT/summary.md" "$OUT/server.log"

"$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket() as sock:
    try:
        sock.bind((host, port))
    except OSError as error:
        raise SystemExit(f"Server port {host}:{port} is already in use: {error}")
PY

setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  "$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype "${DTYPE:-bfloat16}" \
    --context-length "${CONTEXT_LENGTH:-4096}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.70}" \
    --cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE:-full}" \
    --cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}" \
    --cuda-graph-max-bs-decode "$MAX_CONCURRENCY" \
    "${EXTRA_SERVER_ARGS[@]+"${EXTRA_SERVER_ARGS[@]}"}" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"

"$PYTHON_BIN" - "$BASE_URL/health" "${SERVER_READY_TIMEOUT:-600}" "$SERVER_PID" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

url, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit("SGLang server exited before becoming healthy")
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                print(f"SGLang server ready at {url}")
                break
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(2)
else:
    raise SystemExit(f"Timed out waiting for {url}")
PY

for concurrency in "${CONCURRENCY_VALUES[@]}"; do
  "$PYTHON_BIN" -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL_PATH" \
    --dataset-name "${SERVING_DATASET:-random-ids}" \
    --num-prompts "${NUM_PROMPTS:-64}" \
    --random-input-len "${RANDOM_INPUT_LEN:-256}" \
    --random-output-len "${RANDOM_OUTPUT_LEN:-64}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO:-0}" \
    --request-rate "${REQUEST_RATE:-inf}" \
    --max-concurrency "$concurrency" \
    --warmup-requests "${WARMUP_REQUESTS:-1}" \
    --flush-cache \
    --disable-tqdm \
    "${TOKENIZE_ARGS[@]+"${TOKENIZE_ARGS[@]}"}" \
    --tag "concurrency-$concurrency" \
    --output-file "$OUT/results.jsonl" 2>&1 | tee "$OUT/client-concurrency-$concurrency.log"
done

"$PYTHON_BIN" "$ROOT/scripts/summarize_serving.py" \
  --input "$OUT/results.jsonl" \
  --json "$OUT/summary.json" \
  --markdown "$OUT/summary.md" \
  --slo-p99-ttft-ms "${SLO_P99_TTFT_MS:-1000}" \
  --slo-p99-tpot-ms "${SLO_P99_TPOT_MS:-100}"

cat "$OUT/summary.md"
