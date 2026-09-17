# Qwen3-8B CUDA Graph A/B on H200

## Question

The previous clean batch-1 decode trace showed hundreds of short kernels per
token and substantial CUDA launch activity. The falsifiable hypothesis was:

> If batch-1 decode is materially launch-bound, full CUDA Graph replay should
> reduce decode latency even though the GPU executes nearly the same kernels.

## Environment

- GPU: one otherwise idle NVIDIA H200, 143771 MiB, driver 580.105.08.
- Model: Qwen3-8B BF16, dense `Qwen3ForCausalLM`.
- SGLang: 0.5.14 at commit `ba764eff04c7262fc71ea7a85a6d90844fa2ab5f`.
- PyTorch: 2.11.0+cu130; CUDA runtime: 13.0.
- Nsight Systems: 2024.6.2; Nsight Compute: 2025.1.1.
- Workload: batch 1, input 256, output 128, max tokens 512.
- Decode modes: `disabled` versus `full`, with graph batch size 1.
- Prefill graph: disabled in both modes.

The physical host GPU was device 2 and appeared as device 0 in a dedicated
container. It reported 0 MiB used immediately before and after the study.

## Method

The latency result uses five unprofiled trials per mode after one untimed
warmup pair. Trial order is interleaved to reduce time-order bias. Each
`sglang.benchmark.one_batch` process also performs its built-in warmup.

Nsight Systems then captures one prefill and 16 steady-state decode steps for
each decode mode. Profiler-instrumented latency is intentionally excluded from
the A/B result. Nsight Compute remains a targeted follow-up and was not needed
to answer this launch-overhead question.

## Timing Result

| Decode mode | Runs | Median | Mean | Stddev | Median throughput | Median total latency |
|---|---:|---:|---:|---:|---:|---:|
| Eager (`disabled`) | 5 | 10.888 ms | 11.741 ms | 1.824 ms | 91.84 tok/s | 1403.721 ms |
| CUDA Graph (`full`) | 5 | 5.215 ms | 5.232 ms | 0.059 ms | 191.74 tok/s | 678.960 ms |

Full CUDA Graph gives a **2.088x decode speedup** and a **52.10% median decode
latency reduction** for this batch-1 shape.

## Nsight Explanation

| Capture | GPU kernel time | Kernel instances | Ordinary launch calls | Graph launches |
|---|---:|---:|---:|---:|
| Prefill eager, one step | 7.199 ms | 524 | 524 | 0 |
| Decode eager, 16 steps | 76.959 ms | 8368 | 8368 | 0 |
| Decode graph, 16 steps | 78.542 ms | 8272 | 144 | 16 |

The two decode captures spend almost the same aggregate time in GPU kernels:
CUDA Graph is actually **2.06% higher** in this profiled sample. They also run
roughly the same number of kernel instances, about 520 per token. The dominant
GEMM and attention kernels remain the same families with similar durations.

The large change is on the host submission side. Eager decode makes 8368
ordinary launch calls across 16 steps, exactly 523 per step. Full graph replay
reduces that to 144 ordinary calls plus 16 `cudaGraphLaunch` calls: one graph
replay per captured decode step. That is a **98.28% reduction in ordinary
launch calls** while preserving the model's GPU work.

Explicit GPU memory operations are negligible in both decode captures: 54.4
microseconds eager and 59.4 microseconds with graph. The speedup is therefore
not explained by fewer copies or faster GEMMs. It is explained by removing
per-kernel host launch work and the corresponding gaps between short kernels.

## Conclusion

The hypothesis is supported for this workload. Batch-1 Qwen3-8B decode on H200
is strongly sensitive to host launch overhead. CUDA Graph replay approximately
halves token latency without changing the underlying arithmetic workload.

This is not a universal SGLang number. It is an offline, fixed-shape,
single-GPU, batch-1 result on one software stack. Continuous batching, variable
batch shapes, tensor parallelism, quantization, and production scheduling can
change both graph coverage and the achievable gain.

Compact evidence is stored in `reports/data/006_qwen3_8b_cuda_graph_ab/`.
Large `.nsys-rep` and SQLite files remain on the profiling host and are not
tracked by Git.
