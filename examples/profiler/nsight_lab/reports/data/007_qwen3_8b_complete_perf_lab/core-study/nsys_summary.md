# Nsight Systems Summary

| Capture | GPU kernel time | Explicit GPU memory ops | Kernel instances | Ordinary launches | Graph launches |
|---|---:|---:|---:|---:|---:|
| `prefill-disabled` | 7.208 ms | 0.013 ms | 524 | 524 | 0 |
| `decode-disabled` | 76.950 ms | 0.054 ms | 8368 | 8368 | 0 |
| `decode-full` | 78.579 ms | 0.060 ms | 8272 | 144 | 16 |

Decode GPU kernel time change with CUDA Graph: **+2.12%**.

Ordinary decode launch-call reduction with CUDA Graph: **98.28%**.

The CUDA Graph path still executes nearly the same GPU work. Its main benefit is replaying that work without one host launch per kernel.
