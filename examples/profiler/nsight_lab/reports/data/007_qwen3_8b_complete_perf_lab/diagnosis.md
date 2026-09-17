# Evidence-Based Diagnosis

## Host launch overhead: supported

CUDA Graph speedup=2.682x, GPU kernel time change=+2.12%, ordinary launch reduction=98.28%.

Next step: Keep CUDA Graph enabled for matching steady-state shapes.

## Offline shape sensitivity: measured

Best measured point is batch=16, input=2048, output=32 at 34406.8 tok/s.

Next step: Compare the production request shape to the full sweep instead of quoting one batch-1 number.

## Serving saturation: measured

Peak output throughput=2295.1 tok/s at concurrency 32. Best SLO-compliant concurrency=32 at 2295.1 tok/s.

Next step: Use the SLO-compliant point for capacity planning; peak throughput alone is not a serving target.

## Kernel deep dive: memory-throughput-dominant

Kernel nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT measured memory throughput 78.37% and compute throughput 7.41% in the selected launch. Scheduler cycles with no eligible warp=91.09%, achieved occupancy=14.12%, and scoreboard-dependency share=84.80%.

Next step: Map the long-scoreboard stalls back to source counters and memory access patterns, then validate any kernel change with end-to-end A/B timing.

## Claim boundary

Every conclusion above is scoped to the recorded model, SGLang revision, workload, GPU, and runtime stack. A missing profiler layer is reported as unknown rather than guessed.
