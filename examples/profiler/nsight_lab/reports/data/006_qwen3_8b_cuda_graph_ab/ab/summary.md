# CUDA Graph A/B Summary

| Mode | Runs | Decode median | Decode mean | Decode stdev | Decode throughput | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| `disabled` | 5 | 10.888 ms | 11.741 ms | 1.824 ms | 91.84 tok/s | 1403.721 ms |
| `full` | 5 | 5.215 ms | 5.232 ms | 0.059 ms | 191.74 tok/s | 678.960 ms |

Decode latency speedup: **2.088x**.

Decode latency reduction: **52.10%**.

Timing comes from unprofiled `sglang.benchmark.one_batch` runs. Use the paired Nsight Systems traces to explain the result, not to report latency.
