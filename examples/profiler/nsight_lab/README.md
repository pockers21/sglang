# SGLang Performance Engineering Lab

This repository-local lab automates an end-to-end SGLang performance
investigation. It starts with reproducible latency and serving measurements,
narrows the problem with Nsight Systems, selects one important kernel for
Nsight Compute, validates an optimization with an A/B experiment, and produces
a diagnosis whose claims are bounded by the evidence collected.

The lab is model- and machine-agnostic. The checked-in case study uses Qwen3-8B
BF16 on an NVIDIA H200, but machine paths and workload choices are runtime
configuration.

## What Complete Means

| Layer | Question | Tool/output |
|---|---|---|
| Environment | What code, model, CUDA stack, and GPU produced this result? | `environment.txt`, `configuration.txt` |
| Offline baseline | How do prefill/decode change with batch and sequence shape? | `one_batch`, `offline/summary.md` |
| Online serving | Where do throughput and tail latency saturate? | SGLang HTTP server, `serving/summary.md` |
| Stage diagnosis | Which stages, kernels, APIs, and launches consume time? | bounded NSYS captures |
| Kernel diagnosis | Is a selected hot kernel compute-, memory-, or scheduler-limited? | targeted NCU report |

The CUDA Graph case study closes the optimization loop: measure without a
profiler, explain with paired traces, and only then state why it improved.

## Start Here

Read these in order:

1. `docs/learning_path.md`
2. `docs/metrics.md`
3. `docs/decision_tree.md`
4. `reports/006_qwen3_8b_cuda_graph_ab.md`
5. `reports/007_qwen3_8b_complete_perf_lab.md`
6. `docs/interview_playbook.md`

Run repository checks without a GPU:

```bash
cd examples/profiler/nsight_lab
make check
```

## Complete Run

Load a machine-local configuration based on
`configs/qwen3_8b_h200.env.example`, then run:

```bash
source /path/to/local-lab.env
cd examples/profiler/nsight_lab
LAB_RUN_ROOT=/path/to/results/qwen3-8b-h200 \
  make complete MODEL_PATH="$MODEL_PATH"
```

`CUDA_VISIBLE_DEVICES` is resolved where the process starts. If a container is
launched with physical GPU 2 as its only visible device, use
`CUDA_VISIBLE_DEVICES=0` inside that container.

The complete run performs:

1. Unprofiled, interleaved CUDA Graph disabled/full A/B trials.
2. Bounded prefill, eager-decode, and graph-decode NSYS captures.
3. A Cartesian offline batch/input/output sweep in one model-loading process.
4. A real SGLang server lifecycle and HTTP concurrency sweep.
5. Hot-kernel selection from NSYS, a one-launch NCU capture, and a compact
   SpeedOfLight summary. The report also collects memory workload, scheduler,
   warp-state, source-counter, occupancy, and instruction sections for deeper
   inspection.
6. Evidence-based diagnosis and a machine-readable completion status.

Expected output layout:

```text
complete-<timestamp>/
  status.json
  core-study/
    environment.txt
    configuration.txt
    ab/summary.{json,md}
    nsys_summary.{json,md}
    nsys/{prefill-disabled,decode-disabled,decode-full}/
  offline/{results.jsonl,summary.json,summary.md}
  serving/{server.log,results.jsonl,summary.json,summary.md}
  ncu/{target.json,summary.json,summary.md,selected-hot-kernel/sglang_one_batch_ncu.ncu-rep}
  diagnosis.{json,md}
```

`status.json` is `RUNNING`, `FAILED`, or `COMPLETE` and records the last stage.
Large profiler binaries and raw generated results are ignored by Git.

Resume an interrupted run without repeating stages that already produced a
valid summary:

```bash
RESUME=1 LAB_RUN_ROOT=/path/to/existing/results \
  make complete MODEL_PATH="$MODEL_PATH"
```

The serving runner defaults to SGLang's `random-ids` dataset with tokenized
prompts. It therefore needs no ShareGPT download and remains reproducible on an
offline benchmark host.

## Focused Runs

Environment check:

```bash
VENV_PATH=/path/to/venv bash scripts/check_environment.sh
```

Offline shape sweep:

```bash
BATCH_SIZES="1 4 16" INPUT_LENS="128 512 2048" OUTPUT_LENS="32 128" \
  bash scripts/run_offline_sweep.sh /path/to/model
```

Real serving sweep:

```bash
CONCURRENCIES="1 8 32" NUM_PROMPTS=64 \
RANDOM_INPUT_LEN=256 RANDOM_OUTPUT_LEN=64 \
  bash scripts/run_serving_sweep.sh /path/to/model
```

CUDA Graph A/B and NSYS only:

```bash
STUDY_ROOT=/path/to/results/core \
INPUT_LEN=256 OUTPUT_LEN=128 PROFILE_START_STEP=32 PROFILE_STEPS=16 \
  bash scripts/run_full_study.sh /path/to/model
```

Target the hottest kernel from a prior eager-decode capture:

```bash
OUTPUT_ROOT=/path/to/results/ncu \
  bash scripts/profile_selected_kernel_ncu.sh /path/to/model \
  /path/to/decode-disabled/cuda_gpu_kern_sum.csv
```

Set `RUN_NCU=0` when NCU is unavailable. The final diagnosis will explicitly
report compute-versus-memory classification as unknown instead of guessing.

## Measurement Rules

- Report latency from unprofiled runs. Profiler traces explain results; they do
  not replace timing runs.
- Capture a bounded, named stage. Whole-process traces mix loading, warmup,
  allocation, inference, and shutdown.
- Use NSYS before NCU. NCU replay is expensive and should target one kernel
  selected from stage-level evidence.
- Quote tail latency with throughput. Maximum throughput alone is not a serving
  capacity target.
- Treat every result as specific to its model, revision, shape, hardware, and
  software stack.

## Configuration

The primary variables are documented in
`configs/qwen3_8b_h200.env.example`. Useful overrides include:

| Variable | Default | Purpose |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU visible to the process |
| `VENV_PATH` | unset | Python environment containing SGLang dependencies |
| `NSYS`, `NCU` | from `PATH` | Profiler executables |
| `NCU_SECTIONS` | five focused sections | Sections collected for the selected launch |
| `BATCH_SIZES` | `1 4 16` | Offline sweep batch values |
| `INPUT_LENS` | `128 512 2048` | Offline sweep input lengths |
| `OUTPUT_LENS` | `32 128` | Offline sweep output lengths |
| `CONCURRENCIES` | `1 8 32` | Online maximum concurrency values |
| `NUM_PROMPTS` | `64` | Requests per online point |
| `SLO_P99_TTFT_MS` | `1000` | TTFT threshold for capacity classification |
| `SLO_P99_TPOT_MS` | `100` | TPOT threshold for capacity classification |
| `RUN_NCU` | `1` | Run the targeted NCU stage |

## Proven Case Studies

The checked-in Qwen3-8B/H200 case study measured batch-1 decode at 10.888 ms
with ordinary launches and 5.215 ms with full CUDA Graph replay: 2.088x faster.
Paired NSYS captures showed 8,368 ordinary launches in the eager window versus
144 ordinary plus 16 graph launches in the graph window, while total GPU kernel
time remained nearly unchanged. That combination supports a host-launch
overhead diagnosis; the speedup is not attributed to doing less model math.

See `reports/006_qwen3_8b_cuda_graph_ab.md` for the full claim and its limits.

The end-to-end `007` case study adds the 18-point offline sweep, a real serving
concurrency sweep, hot-kernel selection, and a successful targeted NCU replay.
Its formal run measured a 2.682x batch-1 decode speedup, 34,406.8 tok/s at the
best measured offline point, 2,295.1 output tok/s at the highest tested
SLO-compliant serving point, and a memory-throughput-dominant signal for one
selected `nvjet` launch. See `reports/007_qwen3_8b_complete_perf_lab.md`.
