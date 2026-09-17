# Targeted Nsight Compute Summary

Kernel: `nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT`

Classification: **memory-throughput-dominant**.

Collected sections: GPU Speed Of Light Throughput, Memory Workload Analysis, Scheduler Statistics, Warp State Statistics, Instruction Statistics, Occupancy, Source Counters.

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
| Scoreboard-dependency stall | 21.50 cycles |
| Scoreboard-dependency share | 84.80% |
| Branch efficiency | 99.59% |

This is a kernel- and launch-specific signal. Interpret memory, scheduler, instruction, and occupancy evidence together before claiming the full model is memory- or compute-bound.
