# Interview Playbook

Use the case study as an engineering investigation, not as a list of profiler
screenshots.

## Two-Minute Version

1. **Problem:** batch-1 decode latency was high despite a fast H200.
2. **Measurement:** unprofiled, interleaved A/B trials showed 14.178 ms eager
   decode versus 5.287 ms with CUDA Graph, a 2.682x speedup.
3. **Diagnosis:** paired bounded NSYS captures showed ordinary launch calls fell
   from 8,368 to 144 plus 16 graph launches, while GPU kernel time changed only
   slightly.
4. **Conclusion:** for that model and shape, host launch overhead was on the
   critical path; graph replay removed much of it without reducing model math.
5. **Boundary:** this does not prove all batches, models, or serving workloads
   are launch-bound. The same run therefore included offline shape, online
   concurrency, and targeted-kernel sweeps.

## Questions to Expect

**Why not use profiler latency?**

Instrumentation perturbs execution. Timing comes from unprofiled repeated runs;
profiler output explains the difference.

**Why NSYS before NCU?**

NSYS shows the whole stage and ranks kernels by total contribution. NCU replays
individual kernels and is too expensive and too narrow for initial discovery.

**How do you know it was launch overhead?**

The optimization reduced wall time and launch count while preserving nearly the
same GPU kernel work. Those observations match the hypothesis simultaneously.

**How would you productionize the result?**

Sweep realistic input/output distributions and concurrency, pick the highest
SLO-compliant point, monitor graph fallbacks and shape coverage, and rerun the
suite after SGLang, model, driver, or kernel changes.

**What did the NCU pass add?**

NSYS selected a `nvjet` kernel responsible for 36.9% of captured eager-decode
kernel time. One launch showed 78.37% DRAM throughput, 91.09% scheduler cycles
without an eligible warp, and an 84.8% scoreboard-dependency share. That
narrows the kernel investigation to memory access and latency hiding, but it is
not a model-wide memory-bound claim.

**What is the measured serving capacity?**

The highest tested SLO-compliant point was concurrency 32 at 2,295.1 output
tokens/s. It is a tested lower bound on the saturation point, not proof that
the absolute maximum occurs at concurrency 32.

## What Not to Claim

- Do not call the whole model memory-bound from one NCU kernel.
- Do not call peak throughput service capacity without tail latency.
- Do not generalize one batch-1 result to MoE, long-context, or multi-GPU runs.
- Do not present CUDA Graph as reducing FLOPs; it primarily changes launch and
  replay behavior in this case.
