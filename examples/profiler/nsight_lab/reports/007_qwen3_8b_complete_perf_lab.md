# Qwen3-8B Complete Performance Lab on H200

## Scope

This run validates the complete lab workflow rather than one isolated
microbenchmark. It combines unprofiled A/B timing, bounded Nsight Systems
captures, an offline shape sweep, a real HTTP serving sweep, and a targeted
Nsight Compute capture selected from the eager-decode trace.

The run completed successfully. Its machine-readable status is `COMPLETE`.

## Environment

- Physical GPU: NVIDIA H200 device 2, exposed as device 0 in the container.
- GPU memory: 143771 MiB; the card was idle before and after the run.
- Model: Qwen3-8B BF16, dense `Qwen3ForCausalLM`.
- SGLang: 0.5.14 at `ba764eff04c7262fc71ea7a85a6d90844fa2ab5f`.
- PyTorch: 2.11.0+cu130; CUDA runtime: 13.0; driver: 580.105.08.
- Nsight Systems: 2024.6.2; Nsight Compute: 2025.1.1.
- Core A/B shape: batch 1, input 256, output 128.

## CUDA Graph A/B

Five unprofiled trials were collected per mode after warmup. Prefill graph was
disabled in both modes, while decode compared eager execution with full CUDA
Graph replay.

| Decode mode | Runs | Median | Mean | Stddev | Median throughput |
|---|---:|---:|---:|---:|---:|
| Eager (`disabled`) | 5 | 14.178 ms | 12.920 ms | 2.227 ms | 70.53 tok/s |
| CUDA Graph (`full`) | 5 | 5.287 ms | 5.264 ms | 0.069 ms | 189.16 tok/s |

Full CUDA Graph produced a **2.682x decode speedup** and a **62.71% median
decode-latency reduction** for this fixed batch-1 workload.

## Nsight Systems Explanation

| Capture | GPU kernel time | Kernel instances | Ordinary launches | Graph launches |
|---|---:|---:|---:|---:|
| Prefill eager, one step | 7.208 ms | 524 | 524 | 0 |
| Decode eager, 16 steps | 76.950 ms | 8368 | 8368 | 0 |
| Decode graph, 16 steps | 78.579 ms | 8272 | 144 | 16 |

The graph capture did not make the GPU arithmetic cheaper: aggregate profiled
kernel time increased by 2.12%. The structural change was a **98.28% reduction
in ordinary launch calls**, with one graph launch for each captured decode
step. Together with the unprofiled 2.682x result, this supports host launch
overhead as the cause of the batch-1 speedup.

## Offline Shape Sweep

The sweep covered all 18 combinations of batch `{1,4,16}`, input
`{128,512,2048}`, and output `{32,128}`.

- Highest prefill throughput: 48,367.6 tok/s at batch 4, input 512, output 128.
- Highest decode throughput: 2,932.9 tok/s at batch 16, input 128, output 128.
- Highest overall throughput: 34,406.8 tok/s at batch 16, input 2048, output 32.

These are different operating points. A single peak number does not describe
prefill, decode, and end-to-end behavior simultaneously.

## Online Serving Sweep

A real `sglang.launch_server` process served 64 tokenized random-ID requests at
each concurrency. The capacity rule used p99 TTFT <= 1000 ms and p99 TPOT <=
100 ms.

| Max concurrency | Request throughput | Output throughput | p99 TTFT | p99 TPOT | SLO |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.52 req/s | 178.4 tok/s | 70.6 ms | 5.06 ms | pass |
| 8 | 30.94 req/s | 1000.3 tok/s | 48.5 ms | 8.57 ms | pass |
| 32 | 70.99 req/s | 2295.1 tok/s | 122.7 ms | 12.67 ms | pass |

No SLO saturation point was observed within the tested range. Concurrency 32
was both the highest-throughput and highest tested SLO-compliant point; it is
not evidence that 32 is the host's absolute saturation limit.

## Targeted Nsight Compute

The eager-decode NSYS trace selected
`nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT`, which accounted for 36.9% of the
captured GPU kernel time across 576 instances. One matching launch was replayed
with NCU's SpeedOfLight, Memory Workload Analysis, Scheduler Statistics,
Warp State Statistics, Source Counters, Occupancy, and Instruction Statistics
sections.

| Metric | Value |
|---|---:|
| Duration | 53.02 us |
| Compute (SM) throughput | 7.41% |
| Memory throughput | 78.37% |
| DRAM throughput | 78.37% |
| L1/TEX throughput | 22.02% |
| L2 throughput | 83.89% |
| Memory throughput | 3.85 TB/s |
| L2 hit rate | 1.91% |
| Scheduler cycles with no eligible warp | 91.09% |
| Issued warps per scheduler | 0.09 |
| Theoretical occupancy | 18.75% |
| Achieved occupancy | 14.12% |
| Scoreboard-dependency stall | 21.5 cycles, 84.8% of issue interval |
| Branch efficiency | 99.59% |

This selected launch has a **memory-throughput-dominant signal**. That is not a
claim that the whole model or all decode work is memory-bound. A stronger root
cause claim must interpret the collected sections together. In this launch,
high DRAM use coincides with weak latency hiding: 91.09% of scheduler cycles
have no eligible warp, and occupancy is limited by registers and shared memory.
The dominant reported stall is a scoreboard dependency on L1TEX data, while
99.59% branch efficiency makes divergence an unlikely primary cause. Those
observations narrow the next investigation, but they do not by themselves prove
that a particular source-level change will improve end-to-end latency.

## Conclusion

The complete workflow closes the loop from symptom to bounded diagnosis:

1. Unprofiled A/B establishes a real 2.682x fixed-shape decode improvement.
2. NSYS attributes that improvement to launch consolidation rather than less
   GPU arithmetic.
3. Offline and online sweeps prevent the batch-1 conclusion from being
   mistaken for a serving-capacity result.
4. Targeted NCU identifies a memory-throughput signal for one important
   kernel launch without overgeneralizing it to the model.

All compact evidence is stored under
`reports/data/007_qwen3_8b_complete_perf_lab/`. Large `.nsys-rep`, SQLite, and
`.ncu-rep` files remain on the profiling host and are intentionally not tracked
by Git. The compact NCU details export is included at
`reports/data/007_qwen3_8b_complete_perf_lab/ncu/selected-hot-kernel/ncu_details.txt`
for auditability.
