# Qwen3-8B 在 H200 上的完整性能实验

## 实验范围

本次实验验证的是一套完整的性能分析流程，而不是某个孤立的微基准测试。整个流程包括：不启用性能分析器的 A/B 计时、限定范围的 Nsight Systems 采集、离线输入形状扫描、真实 HTTP 服务并发扫描，以及根据 eager decode 时间线选择热点 kernel 后进行的定向 Nsight Compute 分析。

实验已成功完成，机器可读的状态为 `COMPLETE`。

## 实验环境

- 物理 GPU：NVIDIA H200 的第 2 张卡，在容器内映射为设备 0。
- GPU 显存：143771 MiB；实验前后该卡均处于空闲状态。
- 模型：Qwen3-8B BF16，稠密模型 `Qwen3ForCausalLM`。
- SGLang：0.5.14，源码提交为 `ba764eff04c7262fc71ea7a85a6d90844fa2ab5f`。
- PyTorch：2.11.0+cu130；CUDA Runtime：13.0；驱动版本：580.105.08。
- Nsight Systems：2024.6.2；Nsight Compute：2025.1.1。
- A/B 实验的核心输入形状：batch 1、输入 256 tokens、输出 128 tokens。

## CUDA Graph A/B 实验

预热后，每种模式分别进行了 5 次未启用性能分析器的计时。两种模式均关闭 prefill graph；decode 阶段对比 eager execution 与完整 CUDA Graph replay。

| Decode 模式 | 运行次数 | 中位延迟 | 平均延迟 | 标准差 | 中位吞吐 |
|---|---:|---:|---:|---:|---:|
| Eager（`disabled`） | 5 | 14.178 ms | 12.920 ms | 2.227 ms | 70.53 tok/s |
| CUDA Graph（`full`） | 5 | 5.287 ms | 5.264 ms | 0.069 ms | 189.16 tok/s |

对于这个固定的 batch 1 工作负载，完整 CUDA Graph 带来了 **2.682 倍 decode 加速**，decode 中位延迟降低了 **62.71%**。

## Nsight Systems 原因分析

| 采集对象 | GPU kernel 总时间 | Kernel 实例数 | 普通 launch 次数 | Graph launch 次数 |
|---|---:|---:|---:|---:|
| Prefill eager，1 个 step | 7.208 ms | 524 | 524 | 0 |
| Decode eager，16 个 step | 76.950 ms | 8368 | 8368 | 0 |
| Decode graph，16 个 step | 78.579 ms | 8272 | 144 | 16 |

CUDA Graph 并没有降低 GPU 算术计算本身的成本：采集到的 kernel 总时间反而增加了 2.12%。真正的结构性变化是，**普通 kernel launch 调用减少了 98.28%**，每个被捕获的 decode step 只需要进行一次 graph launch。

结合未启用性能分析器时测得的 2.682 倍加速，可以判断：这个 batch 1 工作负载的加速主要来自 host 端 kernel launch 开销的大幅减少，而不是模型执行了更少的 GPU 计算。

## 离线输入形状扫描

扫描覆盖了以下参数的全部 18 种组合：batch `{1,4,16}`、输入长度 `{128,512,2048}`、输出长度 `{32,128}`。

- 最高 prefill 吞吐：48,367.6 tok/s，对应 batch 4、输入 512、输出 128。
- 最高 decode 吞吐：2,932.9 tok/s，对应 batch 16、输入 128、输出 128。
- 最高总体吞吐：34,406.8 tok/s，对应 batch 16、输入 2048、输出 32。

这三个峰值对应不同的运行点。单一峰值数字无法同时描述 prefill、decode 和端到端行为。

## 在线服务并发扫描

实验启动了真实的 `sglang.launch_server` 进程，并在每个并发度下发送 64 个由随机 token ID 构成的请求。容量判定条件为：p99 TTFT 不超过 1000 ms，且 p99 TPOT 不超过 100 ms。

| 最大并发数 | 请求吞吐 | 输出吞吐 | p99 TTFT | p99 TPOT | SLO |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.52 req/s | 178.4 tok/s | 70.6 ms | 5.06 ms | 通过 |
| 8 | 30.94 req/s | 1000.3 tok/s | 48.5 ms | 8.57 ms | 通过 |
| 32 | 70.99 req/s | 2295.1 tok/s | 122.7 ms | 12.67 ms | 通过 |

在本次测试范围内没有观察到违反 SLO 的饱和点。并发 32 同时是测试范围内吞吐最高、并且满足 SLO 的最高并发点；但这不能证明 32 就是该主机的绝对饱和上限。

## 定向 Nsight Compute 分析

根据 eager decode 的 NSYS 时间线，实验选择了 kernel：

```text
nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT
```

它的 576 个实例占采集范围内 GPU kernel 总时间的 36.9%。随后使用 NCU 重放其中一次匹配的 launch，并采集以下 section：SpeedOfLight、Memory Workload Analysis、Scheduler Statistics、Warp State Statistics、Source Counters、Occupancy 和 Instruction Statistics。

| 指标 | 数值 |
|---|---:|
| 单次执行时间 | 53.02 us |
| 计算单元（SM）吞吐利用率 | 7.41% |
| Memory throughput 利用率 | 78.37% |
| DRAM throughput 利用率 | 78.37% |
| L1/TEX throughput 利用率 | 22.02% |
| L2 throughput 利用率 | 83.89% |
| 显存吞吐 | 3.85 TB/s |
| L2 命中率 | 1.91% |
| 没有 eligible warp 的调度周期占比 | 91.09% |
| 每个 scheduler 发射的 warp 数 | 0.09 |
| 理论 occupancy | 18.75% |
| 实际 occupancy | 14.12% |
| Scoreboard 依赖停顿 | 21.5 cycles，占发射间隔的 84.8% |
| 分支效率 | 99.59% |

这个被选中的单次 kernel launch 呈现出明显的 **显存吞吐主导信号**。这不等于整个模型或者全部 decode 工作都是内存受限的。要提出更强的根因结论，必须把多个 section 的指标结合起来解释。

在这次 launch 中，高 DRAM 利用率同时伴随着较弱的延迟隐藏能力：91.09% 的 scheduler cycle 没有可执行的 eligible warp，occupancy 又受到寄存器和共享内存资源的限制。报告中占主导地位的停顿原因，是等待 L1TEX 数据的 scoreboard dependency；与此同时，99.59% 的分支效率说明 warp divergence 不太可能是首要原因。

这些现象缩小了后续调查范围，但仅凭它们还不能证明某一项源码级修改一定能够降低端到端延迟。任何优化方案仍然需要端到端 A/B 实验验证。

## 结论

这套完整流程建立了从性能现象到有限范围诊断的闭环：

1. 未启用性能分析器的 A/B 实验，确认固定输入形状下存在真实的 2.682 倍 decode 加速。
2. NSYS 证明该加速主要来自 kernel launch 合并，而不是 GPU 算术计算量减少。
3. 离线形状扫描和在线服务扫描，避免把 batch 1 的结论错误推广为服务容量结论。
4. 定向 NCU 分析在不过度推广到整个模型的前提下，识别出一个重要 kernel launch 的显存吞吐主导信号和延迟隐藏不足问题。

所有经过压缩整理、适合 Git 跟踪的证据都保存在：

```text
reports/data/007_qwen3_8b_complete_perf_lab/
```

体积较大的 `.nsys-rep`、SQLite 和 `.ncu-rep` 文件仍保留在性能分析主机上，并有意不纳入 Git。为了便于审核，精简后的 NCU 详细导出文件位于：

```text
reports/data/007_qwen3_8b_complete_perf_lab/ncu/selected-hot-kernel/ncu_details.txt
```
