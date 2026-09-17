# Offline Shape Sweep

| Batch | Input | Output | Prefill | Prefill throughput | Decode/token | Decode throughput | Overall throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 32 | 12.30 ms | 10409.2 tok/s | 5.144 ms | 194.4 tok/s | 928.4 tok/s |
| 1 | 128 | 128 | 12.68 ms | 10094.1 tok/s | 5.136 ms | 194.7 tok/s | 384.9 tok/s |
| 1 | 512 | 32 | 19.45 ms | 26328.0 tok/s | 5.156 ms | 194.0 tok/s | 3030.9 tok/s |
| 1 | 512 | 128 | 13.29 ms | 38529.0 tok/s | 5.161 ms | 193.7 tok/s | 954.9 tok/s |
| 1 | 2048 | 32 | 45.30 ms | 45205.3 tok/s | 5.282 ms | 189.3 tok/s | 9859.5 tok/s |
| 1 | 2048 | 128 | 44.85 ms | 45660.2 tok/s | 5.255 ms | 190.3 tok/s | 3048.4 tok/s |
| 4 | 128 | 32 | 13.38 ms | 38276.7 tok/s | 5.340 ms | 749.1 tok/s | 3554.3 tok/s |
| 4 | 128 | 128 | 13.20 ms | 38775.5 tok/s | 5.318 ms | 752.2 tok/s | 1486.2 tok/s |
| 4 | 512 | 32 | 42.64 ms | 48024.8 tok/s | 5.416 ms | 738.5 tok/s | 10314.7 tok/s |
| 4 | 512 | 128 | 42.34 ms | 48367.6 tok/s | 5.397 ms | 741.1 tok/s | 3509.0 tok/s |
| 4 | 2048 | 32 | 245.78 ms | 33331.2 tok/s | 5.646 ms | 708.5 tok/s | 19630.9 tok/s |
| 4 | 2048 | 128 | 184.41 ms | 44423.5 tok/s | 5.620 ms | 711.7 tok/s | 9640.3 tok/s |
| 16 | 128 | 32 | 42.95 ms | 47681.2 tok/s | 5.480 ms | 2920.0 tok/s | 11931.6 tok/s |
| 16 | 128 | 128 | 42.90 ms | 47738.9 tok/s | 5.455 ms | 2932.9 tok/s | 5554.4 tok/s |
| 16 | 512 | 32 | 178.25 ms | 45958.5 tok/s | 5.742 ms | 2786.4 tok/s | 24381.2 tok/s |
| 16 | 512 | 128 | 178.32 ms | 45938.9 tok/s | 5.771 ms | 2772.6 tok/s | 11213.9 tok/s |
| 16 | 2048 | 32 | 759.38 ms | 43151.1 tok/s | 6.612 ms | 2419.8 tok/s | 34406.8 tok/s |
| 16 | 2048 | 128 | 764.31 ms | 42872.8 tok/s | 6.589 ms | 2428.3 tok/s | 21628.3 tok/s |

Best measured overall-throughput point: **batch=16, input=2048, output=32**, at **34406.8 tok/s**.

This sweep describes the low-level offline runner. It does not include HTTP, scheduling, queueing, or multi-request serving overhead.
