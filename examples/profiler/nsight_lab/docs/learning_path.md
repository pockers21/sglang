# Learning Path

The lab follows the order used in a real performance investigation. Do not
start by staring at a single kernel.

## 1. Establish the Workload

Record the model, revision, dtype, GPU, batch, input length, output length,
cache policy, and graph mode. A performance number without this tuple is not
reproducible.

Run the offline sweep first. It separates prefill from per-token decode and
shows whether a conclusion survives changes in shape.

## 2. Measure the User-Facing System

Start a real SGLang HTTP server and sweep concurrency. Track output throughput,
p99 TTFT, p99 TPOT, and p99 end-to-end latency together. The useful operating
point is normally the highest throughput that still satisfies the latency SLO,
not the unconstrained peak.

## 3. Locate the Expensive Stage

Use bounded NSYS captures for prefill and decode. Inspect:

- GPU kernel-time distribution;
- kernel instances and launch API calls;
- CPU gaps between kernels;
- CUDA Graph replay;
- explicit copy and memset time;
- NVTX stage/layer context.

This tells you where to investigate, not yet whether a kernel is compute- or
memory-limited.

## 4. Form One Falsifiable Hypothesis

Example: "Batch-1 decode is launch-bound. CUDA Graph should reduce wall time
without materially reducing total GPU kernel work."

A useful hypothesis predicts what both the timing run and profiler evidence
will look like. If only the final latency is predicted, the explanation is too
weak.

## 5. Inspect One Kernel

Select a high-total-time kernel from the relevant NSYS stage, then capture one
launch with NCU. Read Speed of Light, memory workload, scheduler, occupancy, and
warp-stall sections together. High utilization in one metric does not prove the
entire model has the same bottleneck.

## 6. Validate and State the Boundary

Repeat unprofiled A/B trials in interleaved order. Use paired traces to explain
the result. State the model, shape, hardware, revision, and untested cases.

The final artifact should let another engineer reproduce the measurement,
challenge the diagnosis, and choose the next experiment.
