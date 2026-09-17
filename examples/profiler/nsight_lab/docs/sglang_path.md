# SGLang Execution Path Used by the Lab

## Offline Path

`scripts/run_offline_sweep.sh` invokes:

```text
python/sglang/benchmark/one_batch.py
  latency_test()
    load_model()
    latency_test_run_once()
      extend() -> ModelRunner.forward()   # prefill
      decode() -> ModelRunner.forward()   # one token step
```

The Cartesian product of batch size, input length, and output length is run
after one model load. CUDA synchronization around each phase makes the reported
latency suitable for controlled microbenchmarking but more intrusive than an
asynchronous production scheduler.

## Online Path

`scripts/run_serving_sweep.sh` starts:

```text
python -m sglang.launch_server
  sglang/srt/entrypoints/http_server.py::launch_server()
```

It waits for `/health`, then invokes:

```text
python/sglang/benchmark/serving.py
  run_benchmark()
    benchmark()
      asynchronous HTTP generation requests
```

This path includes tokenization, HTTP, scheduling, batching, queueing, model
execution, and streaming. That is why online TTFT/TPOT cannot be substituted
with `one_batch` prefill/decode latency.

## Profiler Boundaries

`scripts/profile_stage_nsys.sh` enables `one_batch.py`'s CUDA profiler controls.
NSYS is configured with `--capture-range=cudaProfilerApi`, so model loading and
warmup remain outside the saved stage trace. Prefill and decode are captured
separately, and decode can be captured with ordinary launches or CUDA Graph
replay.
