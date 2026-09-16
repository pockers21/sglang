# 002 Initial SGLang Model Smoke Test

## Model And Hardware

The initial run used a local VibeThinker-3B checkpoint, a Qwen2-style causal
language model, on one H200 GPU. The model was chosen because it was small
enough to coexist with other workloads while validating the profiling path.

The following is an equivalent portable command. Replace `/path/to/model` and
the GPU index for the target machine:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m sglang.benchmark.one_batch \
  --model-path /path/to/model \
  --trust-remote-code \
  --batch-size 1 \
  --input-len 16 \
  --output-len 4 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --dtype bfloat16 \
  --context-length 128 \
  --max-total-tokens 128 \
  --mem-fraction-static 0.36 \
  --disable-radix-cache
```

## Observed Result

The original smoke test completed with approximately:

```text
Prefill latency:       0.01455 s
Decode median latency: 0.01274 s
Total latency:         0.053 s
```

These values are not a performance claim. They only establish that model
loading, SGLang execution, and the CUDA runtime path worked before profiler
overhead and capture boundaries were investigated.
