# 004 Clean Prefill/Decode Nsight Systems Results

## Goal

The previous SGLang Nsight Systems report wrapped the whole process:

```text
Python startup
model loading
KV cache allocation
warmup
benchmark
cleanup
```

That made the GUI summary show a misleading large Memory share. This pass uses
SGLang's built-in CUDA profiler controls and Nsight Systems capture ranges so
only the benchmark prefill or benchmark decode section is recorded.

## Script

```text
scripts/profile_stage_nsys.sh
```

It uses:

```text
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop
python3 -m sglang.benchmark.one_batch --profile --profile-activities CUDA_PROFILER
```

SGLang starts/stops the CUDA profiler around the selected benchmark stage:

```text
--profile-stage prefill
--profile-stage decode
```

For decode, it profiles the middle decode steps:

```text
--profile-start-step 16
--profile-steps 8
```

## Common Config

```text
model: local VibeThinker-3B checkpoint
accelerator: one H200 GPU
batch_size: 1
input_len: 256
output_len: 64
context_length: 512
max_total_tokens: 512
dtype: bfloat16
cuda graph: disabled
radix cache: disabled
```

## Clean Prefill

Command:

```bash
INPUT_LEN=256 OUTPUT_LEN=64 PROFILE_START_STEP=16 PROFILE_STEPS=8 \
bash scripts/profile_stage_nsys.sh prefill /path/to/model
```

Report:

```text
results/sglang-clean-prefill-nsys/clean_prefill.nsys-rep
```

Benchmark timing:

```text
prefill_latency:       0.01940 s
prefill_throughput:    13198 token/s
median_decode_latency: 0.01491 s
overall_throughput:      332 token/s
```

Top kernels:

```text
30.3% nvjet_sm90_tst_168x128_64x5_1x2_h_bz_TNN
20.2% nvjet_sm90_tst_128x64_64x8_1x2_h_bz_splitK_TNT
 9.6% FlashAttnFwdSm90 CUTLASS kernel
 7.6% nvjet_sm90_tst_80x64_64x11_1x2_h_bz_bias_TNN
 6.3% nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT
```

Memory ops inside the captured prefill stage are tiny:

```text
Host-to-Device memcpy: 9.95 us total
memset:                1.54 us total
Device-to-Device copy: 1.02 us total
```

## Clean Decode

Command:

```bash
INPUT_LEN=256 OUTPUT_LEN=64 PROFILE_START_STEP=16 PROFILE_STEPS=8 \
bash scripts/profile_stage_nsys.sh decode /path/to/model
```

Report:

```text
results/sglang-clean-decode-nsys/clean_decode.nsys-rep
```

Benchmark timing:

```text
prefill_latency:       0.01441 s
median_decode_latency: 0.01637 s
median_decode_tput:    61.07 token/s
overall_throughput:      319 token/s
```

Top kernels:

```text
32.7% nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT
19.1% nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT
11.8% FlashAttnFwdSm90 CUTLASS kernel
 8.5% nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT
 7.6% nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT
 5.5% nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT
```

Memory ops inside the captured decode stage are also tiny relative to kernel
time:

```text
Device-to-Device copy: 87.0 us total
memset:                78.0 us total
```

## Conclusion

The earlier "Memory 97.4%" GUI summary was not a reliable inference about the
steady-state inference path. It was mostly caused by profiling too much of the
whole process, including loading and initialization.

After capture-range profiling:

```text
prefill is dominated by nvjet_sm90 GEMM kernels and FlashAttention kernels
decode is dominated by nvjet_sm90 GEMM kernels and FlashAttention kernels
CUDA memory ops inside the clean captured regions are small
```

The next profiling step should use the clean decode report as the baseline and
compare:

```text
CUDA graph disabled vs enabled
longer decode length
larger batch size if an idle GPU is available
Nsight Compute on the top decode kernel from this clean run
```
