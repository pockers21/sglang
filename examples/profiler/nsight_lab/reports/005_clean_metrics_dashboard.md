# 005 Clean SGLang Metrics Dashboard

## Scope

This dashboard summarizes the clean profiling runs, not the earlier whole-process
run. The clean runs use:

```text
nsys --capture-range=cudaProfilerApi --capture-range-end=stop
sglang.benchmark.one_batch --profile --profile-activities CUDA_PROFILER
```

So the reports capture only benchmark `prefill` or benchmark `decode`, excluding:

```text
Python startup
model loading
KV cache allocation
warmup
cleanup
```

## Config

```text
model: local VibeThinker-3B checkpoint
accelerator: one H200 GPU
batch_size: 1
input_len: 256
output_len: 64
context_length: 512
max_total_tokens: 512
dtype: bfloat16
cuda graph: disabled
radix cache: disabled
decode profile window: steps 16-23, 8 decode steps
```

## Report Files

| Stage | Report | Size |
|---|---:|---:|
| clean prefill | `results/sglang-clean-prefill-nsys/clean_prefill.nsys-rep` | 253 KB |
| clean decode | `results/sglang-clean-decode-nsys/clean_decode.nsys-rep` | 2.0 MB |
| clean decode NCU top kernel | `results/sglang-one-batch-ncu/sglang_one_batch_ncu.ncu-rep` | 89 KB |

## Benchmark Timing

| Stage Report | Prefill Latency | Prefill Tput | Median Decode Latency | Median Decode Tput | Total Latency | Overall Tput |
|---|---:|---:|---:|---:|---:|---:|
| clean prefill run | 19.40 ms | 13,198 tok/s | 14.91 ms | 67.08 tok/s | 962.66 ms | 332.41 tok/s |
| clean decode run | 14.41 ms | 17,765 tok/s | 16.37 ms | 61.07 tok/s | 1002.72 ms | 319.13 tok/s |

Notes:

- Timing is from `sglang.benchmark.one_batch`.
- The clean prefill report captures only the benchmark prefill region.
- The clean decode report captures only decode steps 16-23.

## NSYS Stage Totals

| Stage | GPU Kernel Time | Kernel Launches | GPU MemOp Time | MemOps | CUDA API Time | CUDA API Calls |
|---|---:|---:|---:|---:|---:|---:|
| clean prefill | 3.949 ms | 488 | 12.512 us | 16 | 2.723 ms | 1,355 |
| clean decode | 125.898 ms | 21,197 | 164.992 us | 188 | 105.647 ms | 59,600 |

Interpretation:

- The earlier `Memory 97.4%` view was from the whole-process report and is not representative of steady-state inference.
- In the clean captures, explicit CUDA MemOps are tiny compared with kernel time.
- CUDA API time is CPU-side launch/driver overhead and should not be added directly to GPU kernel time as if they were serialized wall-clock time.

## Clean Prefill Top Kernels

| Share | Total | Count | Avg | Kernel |
|---:|---:|---:|---:|---|
| 30.2% | 1.197 ms | 36 | 33.25 us | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` |
| 20.1% | 0.797 ms | 36 | 22.13 us | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_splitK_TNT` |
| 9.6% | 0.380 ms | 36 | 10.54 us | `FlashAttnFwdSm90 CUTLASS kernel` |
| 7.6% | 0.302 ms | 36 | 8.39 us | `nvjet_sm90_tst_80x64_64x11_1x2_h_bz_bias_TNN` |
| 6.3% | 0.249 ms | 36 | 6.92 us | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT` |
| 5.1% | 0.204 ms | 72 | 2.83 us | `flashinfer fused_add_rmsnorm` |
| 4.2% | 0.167 ms | 36 | 4.64 us | `act_and_mul_kernel` |
| 3.8% | 0.151 ms | 1 | 151.23 us | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` |

## Clean Decode Top Kernels

| Share | Total | Count | Avg | Kernel |
|---:|---:|---:|---:|---|
| 32.7% | 41.165 ms | 1,692 | 24.33 us | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` |
| 19.1% | 24.070 ms | 1,692 | 14.23 us | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 11.8% | 14.901 ms | 1,692 | 8.81 us | `FlashAttnFwdSm90 CUTLASS kernel` |
| 8.4% | 10.641 ms | 1,692 | 6.29 us | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` |
| 7.6% | 9.545 ms | 1,692 | 5.64 us | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` |
| 5.5% | 6.963 ms | 47 | 148.15 us | `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` |
| 4.8% | 6.070 ms | 3,384 | 1.79 us | `flashinfer fused_add_rmsnorm` |
| 2.3% | 2.877 ms | 1,692 | 1.70 us | `cublasLt splitKreduce` |

## Launch Overhead View

`TAvg` is end-to-end launch-to-completion time for the kernel launch record.
`AAvg` is CPU-side CUDA API time. `QAvg` is queue delay. `KAvg` is GPU kernel time.

| Stage | Count | TAvg | AAvg | QAvg | KAvg | Kernel |
|---|---:|---:|---:|---:|---:|---|
| prefill | 36 | 39.22 us | 4.19 us | 1.78 us | 33.25 us | `nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN` |
| prefill | 36 | 28.16 us | 4.40 us | 1.80 us | 22.13 us | `nvjet_sm90_tst_128x64_64x8_1x2_h_bz_splitK_TNT` |
| decode | 1,692 | 30.46 us | 4.29 us | 1.87 us | 24.33 us | `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` |
| decode | 1,692 | 20.14 us | 4.12 us | 1.82 us | 14.23 us | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| decode | 1,692 | 12.28 us | 4.16 us | 1.87 us | 6.29 us | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` |

This suggests launch/API overhead is non-trivial for these very small batch-1
kernels. CUDA graph comparison is therefore a useful next experiment.

## Memory Ops

| Stage | Operation | Total Time | Count | Total Size |
|---|---|---:|---:|---:|
| prefill | Host-to-Device memcpy | 9.952 us | 13 | 0.002 MB |
| prefill | memset | 1.536 us | 2 | ~0 MB |
| prefill | Device-to-Device memcpy | 1.024 us | 1 | ~0 MB |
| decode | Device-to-Device memcpy | 87.008 us | 94 | 0.001 MB |
| decode | memset | 77.984 us | 94 | ~0 MB |

The clean captures do not support the earlier idea that steady-state inference is
dominated by explicit CUDA memory copy/memset operations.

## NCU: Decode Top Kernel

Kernel:

```text
nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT
```

SpeedOfLight:

| Metric | Value |
|---|---:|
| Duration | 27.33 us |
| DRAM Frequency | 3.19 GHz |
| SM Frequency | 1.36 GHz |
| Memory Throughput | 69.34% |
| DRAM Throughput | 69.34% |
| L2 Cache Throughput | 72.41% |
| L1/TEX Cache Throughput | 31.12% |
| Compute Throughput | 6.41% |

NCU says this specific decode GEMM kernel is more memory-utilized than
compute-utilized. That is a single-kernel statement, not a whole-model claim.

## Practical Takeaways

1. The clean decode stage is the best current baseline.
2. The top decode kernel is `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT`.
3. Explicit MemOps are not the main issue in the clean captures.
4. Batch-1 launch/API overhead is visible; CUDA graph on/off should be compared.
5. Top-kernel NCU indicates memory-side pressure for the sampled decode GEMM.

## Next Runs

Recommended next matrix:

```text
decode, batch_size=1, input_len=256, output_len=64, cuda_graph=disabled
decode, batch_size=1, input_len=256, output_len=64, cuda_graph=enabled
decode, batch_size=1, input_len=256, output_len=256, cuda_graph=disabled
decode, batch_size=2 or 4 if a cleaner GPU is available
```
