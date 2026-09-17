# Online Serving Sweep

SLO used for classification: p99 TTFT <= 1000 ms and p99 TPOT <= 100 ms.

| Max concurrency | Completed | Requests/s | Output tok/s | Total tok/s | p99 TTFT | p99 TPOT | p99 E2E | SLO |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 64 | 5.52 | 178.4 | 909.2 | 70.6 ms | 5.1 ms | 331.7 ms | pass |
| 8 | 64 | 30.94 | 1000.3 | 5097.9 | 48.5 ms | 8.6 ms | 485.3 ms | pass |
| 32 | 64 | 70.99 | 2295.1 | 11696.2 | 122.7 ms | 12.7 ms | 696.1 ms | pass |

Peak measured output throughput: **2295.1 tok/s** at max concurrency **32**.

Best SLO-compliant point: concurrency **32**, output throughput **2295.1 tok/s**.
