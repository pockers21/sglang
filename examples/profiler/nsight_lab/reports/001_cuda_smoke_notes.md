# 001 CUDA Smoke Test

The standalone CUDA program in `cuda/nsight_smoke.cu` verifies the profiler
workflow before a full SGLang run. It contains SAXPY and reduction kernels plus
NVTX ranges for host-to-device copies, warmup, the measured loop, and
device-to-host copies.

Build it with the CUDA compiler and NVTX library available in the target
environment. For example:

```bash
nvcc -O3 -lineinfo cuda/nsight_smoke.cu -o nsight_smoke -lnvToolsExt
```

## Nsight Systems Checklist

- Find the `h2d_copy`, `warmup`, `profile_loop`, `saxpy`, `reduce`, and
  `d2h_copy` NVTX ranges.
- Check that kernel launches and synchronization calls appear on the CUDA API
  timeline.
- Compare kernel count, total GPU time, and average duration.
- Confirm that the warmup range is excluded from conclusions about the steady
  measured loop.

## Nsight Compute Checklist

- GPU Speed Of Light
- Occupancy
- Memory Workload Analysis
- Scheduler Statistics
- Warp State Statistics

Once this path works, use `scripts/profile_stage_nsys.sh` to capture bounded
SGLang prefill or decode work.
