# SGLang Nsight Profiling Lab

This lab provides a reproducible workflow for profiling SGLang inference with
NVIDIA Nsight Systems and Nsight Compute. It is designed as a compact
performance case study: isolate a meaningful inference window, identify hot
kernels, attribute overhead, form an optimization hypothesis, and validate it
with an A/B experiment.

The lab lives inside the SGLang source tree at:

```text
examples/profiler/nsight_lab
```

Generated profiler reports and locally installed tools are intentionally not
tracked by Git.

## What To Read First

```text
reports/006_qwen3_8b_cuda_graph_ab.md
reports/005_clean_metrics_dashboard.md
scripts/profile_stage_nsys.sh
scripts/profile_kernel_ncu.sh
scripts/run_full_study.sh
python/sglang/benchmark/one_batch.py
```

The case study shows why capture boundaries matter:

1. Whole-process profiling included model loading, KV-cache initialization,
   warmup, and shutdown, which obscured steady-state inference behavior.
2. Clean prefill and decode captures use SGLang's CUDA profiler controls with
   the Nsight Systems `cudaProfilerApi` capture range.
3. The clean batch-1 decode trace contains many short kernels and visible
   launch/API overhead, while explicit CUDA memory operations are small.
4. The complete runner performs an unprofiled CUDA Graph on/off A/B test,
   followed by paired Nsight Systems captures that explain the timing result.

## Requirements

- A CUDA-capable system supported by SGLang.
- A working SGLang development environment for this checkout.
- `nsys` for stage-level analysis.
- `ncu` for optional single-kernel analysis.
- A local model checkpoint or a model identifier accepted by SGLang.

Set `VENV_PATH` if the environment is not already active. Set `NSYS` or `NCU`
when the tools are not available on `PATH`. If a driver compatibility package
is required in a container, set `CUDA_COMPAT_DIR` to its library directory.

## Quick Start

From the repository root:

```bash
cd examples/profiler/nsight_lab
VENV_PATH=/path/to/venv bash scripts/check_environment.sh
```

Profile a bounded decode window:

```bash
CUDA_VISIBLE_DEVICES=0 \
INPUT_LEN=256 OUTPUT_LEN=64 PROFILE_START_STEP=16 PROFILE_STEPS=8 \
bash scripts/profile_stage_nsys.sh decode /path/to/model
```

Profile prefill:

```bash
CUDA_VISIBLE_DEVICES=0 INPUT_LEN=256 OUTPUT_LEN=64 \
bash scripts/profile_stage_nsys.sh prefill /path/to/model
```

The generated files are written under `results/` by default:

```text
results/decode-disabled/clean_decode.nsys-rep
results/decode-disabled/cuda_gpu_kern_sum.txt
results/decode-disabled/cuda_api_sum.txt
results/decode-disabled/cuda_gpu_mem_time_sum.txt
results/decode-disabled/cuda_kern_exec_trace_nvtx.txt
```

Set `OUTPUT_ROOT` to place the generated files elsewhere.

## Complete CUDA Graph Study

Run the full workflow on an otherwise idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
VENV_PATH=/path/to/venv \
SGLANG_REPO_ROOT=/path/to/sglang \
NSYS=/path/to/nsys NCU=/path/to/ncu \
STUDY_ROOT=/path/to/output/qwen3-8b-cuda-graph \
REPEATS=5 WARMUP_RUNS=1 \
INPUT_LEN=256 OUTPUT_LEN=128 \
PROFILE_START_STEP=32 PROFILE_STEPS=16 \
bash scripts/run_full_study.sh /path/to/model
```

The workflow deliberately separates measurement from diagnosis:

1. `run_cuda_graph_ab.sh` runs unprofiled, interleaved `disabled` and `full`
   decode trials. It reports the median, mean, standard deviation, throughput,
   speedup, and latency reduction.
2. `profile_stage_nsys.sh` captures representative prefill, eager decode, and
   CUDA Graph decode windows. Profiler-instrumented latency is not used as the
   performance result.
3. `profile_kernel_ncu.sh` remains an optional targeted follow-up after a hot
   kernel has been selected from the Nsight Systems evidence.

The study directory contains:

```text
environment.txt
configuration.txt
ab/disabled.jsonl
ab/full.jsonl
ab/summary.json
ab/summary.md
nsys/prefill-disabled/
nsys/decode-disabled/
nsys/decode-full/
nsys_summary.json
nsys_summary.md
```

Each Nsight directory contains the binary report plus text and CSV exports for
CUDA kernels and API calls. Keep the large binary reports outside Git; commit
only compact, reviewed evidence needed by a written report.

## Kernel Deep Dive

Use Nsight Compute only after Nsight Systems has identified a kernel worth
investigating:

```bash
CUDA_VISIBLE_DEVICES=0 \
KERNEL_NAME='regex:nvjet_sm90_tst_256x8.*' \
LAUNCH_SKIP=40 LAUNCH_COUNT=1 \
bash scripts/profile_kernel_ncu.sh /path/to/model
```

The output includes an `.ncu-rep` file and a text export of the details page.
The exact kernel name and launch index depend on the model, batch size, SGLang
revision, and CUDA stack.

## Configuration

The scripts accept configuration through environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU visible to the benchmark |
| `BATCH_SIZE` | `1` | Benchmark batch size |
| `INPUT_LEN` | `256` for NSYS | Input sequence length |
| `OUTPUT_LEN` | `64` for NSYS | Generated token count |
| `PROFILE_START_STEP` | `16` | First decode step to capture |
| `PROFILE_STEPS` | `8` | Number of decode steps to capture |
| `CUDA_GRAPH_BACKEND_DECODE` | `disabled` | Decode CUDA Graph backend |
| `CUDA_GRAPH_BACKEND_PREFILL` | `disabled` | Prefill CUDA Graph backend |
| `CUDA_GRAPH_BS_DECODE` | `1` | Batch size captured by decode CUDA Graph |
| `REPEATS` | `5` | Timed trials per A/B mode |
| `WARMUP_RUNS` | `1` | Untimed process-level warmup pairs |
| `VENV_PATH` | unset | Optional Python virtual environment |
| `NSYS` | `nsys` from `PATH` | Nsight Systems executable |
| `NCU` | `ncu` from `PATH` | Nsight Compute executable |
| `OUTPUT_ROOT` | `results/` | Generated report directory |
| `RUN_NAME` | stage and graph mode | Per-capture output directory name |
| `STUDY_ROOT` | timestamped directory | Full-study output directory |
| `CUDA_PYTHON_LIB_DIR` | auto-detected | CUDA wheel runtime library directory |

## Reports

The historical reports record one H200/Qwen2-style 3B model investigation.
Measured numbers are evidence for that environment, not universal SGLang
performance claims. Re-run the complete workflow on the target model, SGLang
revision, CUDA stack, and otherwise idle hardware before drawing conclusions.

The latest result is in `reports/006_qwen3_8b_cuda_graph_ab.md`. It closes the
experiment proposed by `reports/005_clean_metrics_dashboard.md`: CUDA Graph
replay cuts batch-1 decode latency while leaving the amount of GPU kernel work
nearly unchanged, because thousands of ordinary host launch calls are folded
into one graph replay per decode step.

## Profiling Method

Use Nsight Systems first to answer broad questions:

- Which stage is slow?
- Which kernel families dominate GPU time?
- How many launches occur?
- Is explicit copy/memset time material?
- Is launch overhead worth investigating?

Use Nsight Compute second, on one selected kernel, to inspect compute, memory,
cache, occupancy, scheduler, and warp-level metrics. A single NCU result must
not be generalized into a whole-model bottleneck without stage-level evidence.
