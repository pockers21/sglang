# Metrics Guide

## Offline Metrics

- **Prefill latency**: time to process all prompt tokens and produce the first
  sampled token in the low-level runner.
- **Prefill throughput**: `batch * input_tokens / prefill_time`.
- **Median decode latency**: median synchronized time for one autoregressive
  decode step. It is not request end-to-end latency.
- **Decode throughput**: `batch / median_step_time` in tokens per second.
- **Overall throughput**: prompt and output tokens divided by total low-level
  execution time.

## Serving Metrics

- **TTFT**: request arrival to first output token. It includes queueing,
  scheduling, prefill, and transport overhead.
- **TPOT**: average time per output token after the first token.
- **ITL**: observed interval between streamed tokens. Its distribution exposes
  jitter hidden by average TPOT.
- **E2E latency**: complete request latency.
- **Output throughput**: generated tokens per second across completed requests.
- **Request throughput**: completed requests per second; meaningful only with a
  stated request shape.

Always report throughput with p99 TTFT and p99 TPOT. Concurrency can raise
throughput while making the service unusable.

## NSYS Metrics

- **GPU kernel time** is the sum of captured kernel durations. It can remain
  constant even when wall time improves through lower launch overhead.
- **Kernel instances** describe launch count, not work efficiency.
- **CUDA API time** is host-side time inside CUDA calls. It is not the complete
  CPU critical path.
- **Explicit GPU memory operation time** covers reported memcpy/memset activity;
  it does not represent all memory traffic performed inside kernels.

## NCU Metrics

- **Compute throughput** and **memory throughput** are percentages of relevant
  device limits, not percentages of runtime.
- **Occupancy** is resident warps relative to the architectural maximum. High
  occupancy does not guarantee useful issue rate; low occupancy can still be
  sufficient.
- **Warp stall reasons** are scheduler observations. Interpret them together
  with instruction mix, dependencies, and memory behavior.
- **Roofline position** applies to the captured kernel and launch shape. It is
  not automatically a model-wide label.
