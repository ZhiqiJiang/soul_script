# Nsight Compute 计算利用率指标解读：Compute Throughput、SM Busy 与昇腾指标对标

> 分析对象：CUTLASS GEMM kernel（`cutlass_profiler`，969 - Kernel2），NVIDIA L20，Grid (128,4,1) × Block (256,1,1)

## 结论速览

- 这个 kernel 是 **计算瓶颈（compute bound）**，瓶颈在 **Tensor Core 流水线**：Compute (SM) Throughput 93.04%，Memory Throughput 只有 21.14%。
- 只要 SM 在工作，Tensor 流水线几乎是满的（**SM Busy 99.17%**）。剩下约 6% 的差距来自 **SM 空闲时间**（SM Active ÷ Elapsed ≈ 93.7%），主要原因是尾波效应。
- 三个指标之间的关系（单位均为百分比）：

```math
\underbrace{93.04}_{\text{Compute Throughput}} \approx \underbrace{93.7}_{\text{SM Active / Elapsed}} \times \underbrace{99.17}_{\text{SM Busy}}
```

- 昇腾侧对标：`cube_utilization` ↔ SM Active ÷ Elapsed；`aic_mac_ratio` ↔ SM Busy（严格来说对标的是 Tensor 流水线利用率，见第 3 节）。

---

## 1. Compute (SM) Throughput

![GPU Speed Of Light Throughput](figs/ncu_gpu_speed_of_light.png)

### 1.1 计算方式

Compute (SM) Throughput 不是 SM 各单元的平均利用率。NCU 会计算 SM 内部一组子指标（各条执行流水线的利用率、指令发射率、执行指令数等），取其中的**最大值**。所以它表示的是 SM 里**最忙的那个单元**有多忙，也就是瓶颈单元的利用率。

每个子指标都是"实际值 ÷ 理论峰值"，分母按 **Elapsed Cycles**（kernel 从开始到结束的总周期）计算。

### 1.2 Compute Throughput Breakdown

| 子指标 | 值 | 含义 |
|---|---|---|
| SM: Pipe Tensor Cycles Active V2 | **93.04** | Tensor Core 流水线处于忙碌状态的周期占比，Compute Throughput 取的就是这一项 |
| SM: Pipe Tensor Cycles Active | 46.52 | 旧版统计口径，见下面说明 |
| SM: Inst Executed Pipe Lsu | 9.63 | Load/Store 流水线（共享内存、全局内存访问、ldmatrix 等）的利用率 |
| SM: Issue Active | 6.35 | warp 调度器在多少比例的周期里发射了指令 |
| SM: Inst Executed | 6.35 | 实际执行的指令数（IPC）相对峰值的比例 |

### 1.3 容易误解的地方

- **V2 和非 V2 刚好差 2 倍。** 两者是对同一条 Tensor 流水线的不同统计口径。旧版按"执行的指令数 × 名义吞吐率"计算；V2 按流水线实际被占用的周期计算，会考虑当前架构下具体 MMA 指令的精度和形状。这个 kernel 用的 16-bit MMA 指令在 L20 上每条实际占用的周期是名义值的 2 倍，所以旧版口径低估了一半。应当以 V2 的 **93%** 为准。
- **Issue Active 只有 6% 并不代表 SM 闲着。** 一条 HMMA 指令会占用 Tensor 流水线很多个周期，所以 GEMM kernel 本来就是"发射的指令少，但每条指令很重"。这不是延迟（latency）瓶颈。
- **Roofline 显示 FP32 达到峰值的 0% 是正常的。** Tensor Core 运算不计入 FP32/FP64 roofline，这一行对这个 kernel 没有参考意义。

### 1.4 关于 "GPU frequencies not fixed" 警告

这次 profiling 没有锁定 GPU 频率（SM 实际跑在约 2.5 GHz 的 boost 频率上）。百分比类指标是按周期算的，影响不大；但 Duration（耗时）在不同次运行之间可能有波动，做对比时要注意。

---

## 2. SM Busy 与 SM Active Cycles ÷ Elapsed Cycles

![Compute Workload Analysis](figs/ncu_compute_workload_analysis.png)

一句话区分：**SM Active ÷ Elapsed 衡量 SM 有多少时间在干活；SM Busy 衡量干活的时候最忙的流水线有多满。**

### 2.1 SM Active Cycles ÷ Elapsed Cycles：时间维度的利用率

- **Elapsed Cycles**：kernel 从开始到结束经过的总周期数。
- **SM Active Cycles**：SM 上**至少有一个 warp 驻留**的周期数，对所有 SM 取平均。

本例：4,964,833 ÷ 5,300,438 ≈ **93.7%**。

这个比例**不关心活干得多饱满**：哪怕 SM 上只有一个 warp 卡在等访存，这个周期也算 active。少掉的约 6.3% 是 SM 上一个 warp 都没有的时间，常见原因：

- **尾波（wave quantization / tail effect）**：512 个 CTA 分到 92 个 SM 上，最后一波分配不满，提前做完的 SM 只能空等。
- CTA 启动和退出的间隙。
- 负载不均：有的 SM 分到的 CTA 更多。

### 2.2 SM Busy：干活时流水线的饱满程度

SM Busy 对应 `sm__instruction_throughput.avg.pct_of_peak_sustained_active`，要点有两个：

- **分母是 Active Cycles**，不是 Elapsed Cycles，SM 空闲的时间不计入。
- **取各条流水线利用率的最大值**（Tensor、FMA、ALU、LSU 等）。

本例 99.17% 的意思是：只要 SM 上有 warp 在跑，**Tensor 流水线几乎每个周期都被占满**。截图里的说明文字 "Tensor is the highest-utilized pipeline (99.2%) based on active cycles" 讲的就是这个。

### 2.3 按 Elapsed 算和按 Active 算的指标对照

截图里的指标都能用同一个比例 0.937 对上：

| 按 Elapsed 算 | 按 Active 算 | 比值 |
|---|---|---|
| Compute Throughput 93.04% | SM Busy 99.17% | ≈0.937 |
| Executed IPC Elapsed 0.25 | Executed IPC Active 0.27 | ≈0.937 |
| Pipe Tensor Cycles Active 46.52% | Tensor (FP) 49.6%（按指令数算） | ≈0.937 |

另外两点：

- **Issue Slots Busy 6.77% 也是按 Active 算的。** 每个 SM 有 4 个 warp 调度器，每周期最多发射 4 条指令，所以 0.27 ÷ 4 ≈ 6.75%。它和 SM Busy 不矛盾：指令发得少，但每条 HMMA 指令占用流水线很多个周期。
- **说明文字里出现了 99.2% 和 49.6% 两个 Tensor 利用率。** 49.6% 按"执行的指令数 × 名义吞吐率"计算，99.2% 按流水线实际被占用的周期计算，差 2 倍的原因和 1.3 节中 V2 与旧版指标的差别相同。应当以 99.2% 为准。

---

## 3. 与昇腾（Ascend）指标对标

昇腾侧的 `cube_utilization` 定义为：

```math
\text{cube\_utilization} = \frac{\text{aic\_total\_cycles}}{\text{aic\_frequency} \times \text{ai\_core\_num} \times \text{Task Duration}}
```

### 3.1 cube_utilization ↔ SM Active Cycles ÷ Elapsed Cycles（完全对应）

两者都是**时间维度**的利用率，衡量核在多少比例的墙上时间里有活干：

- 分子 `aic_total_cycles`：所有 AI Core 实际活跃的周期总和，对应 NCU 的 **SM Active Cycles**。NCU 对 SM 取平均，这里是求和，分母里的 `ai_core_num` 把这个差异抵消了。
- 分母 `aic_frequency × ai_core_num × Duration`：所有核的理论可用总周期，对应 NCU 的 **Elapsed Cycles × SM 数**。

它同样不关心活干得饱不饱，少掉的部分是核空闲的时间（尾波、启动/退出间隙、负载不均）。

### 3.2 aic_mac_ratio ↔ SM Busy（方向对，但范围更窄）

`aic_mac_ratio` 的分母是 `aic_total_cycles`（核**活跃**的周期），衡量核在干活时 Cube/MAC 单元被占满的程度，这和 SM Busy 的定义一致。但统计范围不同：

| 指标 | 统计范围 |
|---|---|
| SM Busy | 所有流水线（Tensor、FMA、ALU、LSU 等）利用率的**最大值** |
| aic_mac_ratio | **只统计 Cube/MAC 这一条流水线** |

昇腾侧还有 `aic_vec_ratio`、`aic_scalar_ratio`、`aic_mte2_ratio`/`aic_mte3_ratio`（数据搬运）等并列指标。严格对标 SM Busy 的应当是**这些 ratio 中的最大值**；`aic_mac_ratio` 单独对标的是 NCU 里按 active 归一化的 **Tensor 流水线利用率**。

只有当 Cube 是瓶颈流水线时（类比本例 Tensor 主导），`aic_mac_ratio ≈ SM Busy`。如果瓶颈是向量单元或搬运单元，SM Busy 由那条流水线决定，`aic_mac_ratio` 会偏低，这时不能直接划等号。

### 3.3 两者相乘对应 Compute Throughput

```math
\underbrace{\text{cube\_utilization}}_{\text{核有多少时间在干活}} \times \underbrace{\text{aic\_mac\_ratio}}_{\text{干活时 Cube 有多满}} \approx \underbrace{\text{Cube 忙碌时间占总时间的比例}}_{\text{对标 Compute (SM) Throughput}}
```

### 3.4 指标对照总表

| 含义 | NVIDIA（Nsight Compute） | 昇腾 |
|---|---|---|
| 核有多少时间在干活（时间维度） | SM Active Cycles ÷ Elapsed Cycles | cube_utilization |
| 干活时最忙流水线的饱满程度 | SM Busy | max(aic_mac_ratio, aic_vec_ratio, aic_scalar_ratio, aic_mte*_ratio, ...) |
| 干活时矩阵单元的饱满程度 | Tensor 流水线利用率（按 Active 归一化） | aic_mac_ratio |
| 矩阵单元忙碌时间占总时间的比例 | Pipe Tensor Cycles Active V2（按 Elapsed 归一化） | cube_utilization × aic_mac_ratio |

---

## 4. 优化方向

- **SM Busy ≈ 99%**（昇腾侧对应 `aic_mac_ratio` 很高）：干活时矩阵单元已经跑满，指令层面没有优化空间了。
- **Active 比例 93.7%**（昇腾侧对应 `cube_utilization`）：剩下约 6% 的提升空间全部在核空闲时间上，应处理尾波和负载均衡：
  - 调整 tile 大小，让 CTA 数和 SM 数更匹配；
  - 改用 stream-K、split-K 或 persistent kernel。
- 进一步提升峰值算力的方向：
  - **降低精度**：Ada 架构支持 FP8 Tensor Core，峰值约为 FP16 的 2 倍；
  - **2:4 结构化稀疏**：理论上再获得约 2 倍峰值；
  - **算子融合**：把 GEMM 前后的操作融合进来，减少 kernel 数量和额外开销。
