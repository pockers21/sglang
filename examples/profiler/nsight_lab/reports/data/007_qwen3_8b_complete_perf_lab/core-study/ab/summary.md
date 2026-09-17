# CUDA Graph A/B Summary

| Mode | Runs | Decode median | Decode mean | Decode stdev | Decode throughput | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| `disabled` | 5 | 14.178 ms | 12.920 ms | 2.227 ms | 70.53 tok/s | 1827.512 ms |
| `full` | 5 | 5.287 ms | 5.264 ms | 0.069 ms | 189.16 tok/s | 689.542 ms |

Decode latency speedup: **2.682x**.

Decode latency reduction: **62.71%**.

Timing comes from unprofiled `sglang.benchmark.one_batch` runs. Use the paired Nsight Systems traces to explain the result, not to report latency.
