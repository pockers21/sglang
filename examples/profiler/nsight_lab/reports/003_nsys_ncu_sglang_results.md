# 003 First SGLang Nsight Results

## Tooling Lesson

The first investigation exposed an important container-profiling constraint:
the profiler must be compatible with the active CUDA runtime and must execute
where it can trace the target process. Wrapping a container command with a host
profiler did not provide the expected CUDA trace, while running compatible
Nsight binaries inside the target environment did.

The original whole-process Nsight Systems trace contained real CUDA kernel and
API data. Its top kernel families included:

```text
nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT
nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT
nvjet_sm90_tst_192x16_64x8_4x1_v_bz_TNT
FlashAttnFwdSm90 CUTLASS kernels
flashinfer fused_add_rmsnorm
fused_rope_kernel
store_kvcache
act_and_mul_kernel
```

This established that timeline analysis was working, but it did not yet isolate
steady-state prefill or decode. The cleaner method is documented in
`004_clean_prefill_decode_nsys.md` and implemented by
`scripts/profile_stage_nsys.sh`.

## Nsight Compute Sample

After Nsight Systems identified a frequently launched decode GEMM, Nsight
Compute captured:

```text
nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT
```

The first SpeedOfLight sample reported:

```text
Duration:              26.43 us
Memory Throughput:     71.62%
DRAM Throughput:       71.62%
L2 Cache Throughput:   75.27%
L1/TEX Throughput:     30.99%
Compute Throughput:     6.64%
```

This kernel was more memory-utilized than compute-utilized in that sample. It
does not establish that the entire model is memory-bound. The stage-level trace
and the per-kernel NCU result answer different questions and should be reported
separately.

An equivalent NCU invocation is:

```bash
KERNEL_NAME='regex:nvjet_sm90_tst_256x8.*' \
LAUNCH_SKIP=0 LAUNCH_COUNT=1 \
bash scripts/profile_kernel_ncu.sh /path/to/model
```
