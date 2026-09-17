# Bottleneck Decision Tree

Use this as a sequence of tests, not as a list of labels.

## Latency Regressed

1. Re-run the same unprofiled shape several times.
2. Confirm model revision, clocks, GPU occupancy by other jobs, and cache mode.
3. Separate prefill, decode, and HTTP serving behavior.
4. Compare a bounded NSYS trace only after the timing regression is real.

## Many Short Decode Kernels and CPU Gaps

Run CUDA Graph disabled/full A/B.

- A meaningful wall-time speedup, large ordinary-launch reduction, and nearly
  unchanged GPU kernel time support host launch overhead.
- A speedup with much less GPU kernel time needs another explanation, such as a
  changed execution path or workload.
- No speedup means launches were not the limiting critical path for that shape.

## One or a Few Kernels Dominate GPU Time

Select by total stage time, not merely the longest single invocation. Capture
one representative launch in NCU.

- High compute utilization plus suitable instruction mix suggests a compute
  limit.
- High DRAM/L2 traffic and memory utilization with low arithmetic intensity
  suggests a memory limit.
- Low compute and memory utilization requires scheduler, dependency, occupancy,
  and launch-shape analysis before drawing a conclusion.

## Throughput Plateaus as Concurrency Rises

Check p99 TTFT and TPOT at the same points.

- Plateau plus rapidly growing tail latency is a saturation boundary.
- Throughput still rising but SLO failing means the capacity point was passed
  earlier.
- Poor low-concurrency throughput may be launch/scheduling overhead rather than
  device saturation.

## Evidence Is Missing

Say `unknown`. NSYS cannot prove a kernel is memory-bound, NCU cannot explain an
entire request timeline, and offline `one_batch` cannot represent queueing or
HTTP service quality.
