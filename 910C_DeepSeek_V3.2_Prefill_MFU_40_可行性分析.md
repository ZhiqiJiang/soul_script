# Ascend 910C 上 DeepSeek V3.2 Compute-Bound Prefill MFU 优化到 40% 的可行性分析

**文档性质**：技术论证备忘录（汇总公开文献与硬件规格对比）  
**结论摘要**：将 `148/752 ≈ 20%` 直接等同于 910C 的 MFU 上限，在概念上不成立；华为自己在 DeepSeek-R1 prefill 上已公开达到约 **33% MFU**（换算自 CloudMatrix-Infer）。目标 **40%** 是相对已发表结果约 **1.2×** 的激进但可争取的工程目标，成败取决于 V3.2 indexer/gather 与 MoE 负载均衡等 glue 开销，而非「华为只能对标 H20」这一市场话术。

---

## 1. 问题陈述

内部常见说法：

1. 华为称 910C 只能对标 NVIDIA H20；
2. H20 Peak FP16 Tensor = **148 TFLOPS**；
3. 910C Peak FP16 Tensor = **752 TFLOPS**；
4. 因此 910C 上 MFU上限是 `148/752 ≈ 20%`；
5. 据此质疑：凭什么个人/小团队能把 DeepSeek V3.2 的 compute-bound prefill MFU 做到 **40%**？

本文逐条拆解上述推理，并给出可复算的公开数据与风险清单。

---

## 2. 硬件峰值与「对标 H20」语义

### 2.1 Ascend 910C vs NVIDIA H200


| 指标                      | NVIDIA H200 | Huawei Ascend 910C | 910C / H200 |
| ----------------------- | ----------- | ------------------ | ----------- |
| Peak FP16 Tensor TFLOPS | 989         | **752**            | 76%         |
| 显存带宽                    | 4.8 TB/s    | 3.2 TB/s           | 67%         |
| P2P 带宽                  | 900 GB/s    | 784 GB/s           | 87%         |
| 显存容量                    | 141 GB      | 128 GB             | 91%         |
| Peak FP16（非 Tensor）     | 134         | 47                 | 35%         |
| INT8                    | 支持          | 支持                 | —           |
| FP8                     | 支持          | **不支持**            | —           |


数据来源：

- 910C官方文档：[https://support.huawei.com/enterprise/zh/doc/EDOC1100461253/8eaff0ba?idPath=23710424|251366513|22892968|252309113|261716443](https://support.huawei.com/enterprise/zh/doc/EDOC1100461253/8eaff0ba?idPath=23710424|251366513|22892968|252309113|261716443)
- H100白皮书：[https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
- H200 Datasheet: [https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23](https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23)

### 2.2 NVIDIA H20 关键规格（「对标」对象）


| 指标                      | 数值                  | 来源                                                                             |
| ----------------------- | ------------------- | ------------------------------------------------------------------------------ |
| Peak FP16 / BF16 Tensor | **148 TFLOPS**      | [Flopper.io H20 Spec Sheet](https://flopper.io/gpu/nvidia-h20-96gb/spec-sheet) |
| Peak INT8 / FP8 Tensor  | **296** TOPS/TFLOPS | 同上                                                                             |
| 显存带宽                    | **4.0 TB/s**        | 同上                                                                             |
| 显存容量                    | 96 GB HBM3          | 同上                                                                             |


### 2.3 910C 相对 H20 的强弱轴（为何「对标」不能套到 prefill）


| 维度               | 910C       | H20        | 比值（910C/H20） |
| ---------------- | ---------- | ---------- | ------------ |
| Peak FP16 Tensor | 752 TFLOPS | 148 TFLOPS | **≈ 5.1×**   |
| 显存带宽             | 3.2 TB/s   | 4.0 TB/s   | **≈ 0.8×**   |


**来源**：910C 峰值见 §2.1；H20 见 §2.2。

**推论**：

- **Decode** 通常更偏带宽瓶颈 → H20 带宽不低于 910C，「整机服务/对标 H20」在此轴上说得通。
- **Prefill** 在足够大的 batch / 合适精度下通常更偏算力瓶颈 → 910C 峰值算力约为 H20 的 5 倍，「对标 H20」**不应**直接当作 prefill 算力天花板。

---

## 3. 「对标 H20」的性质

「对标 H20」更多是产品/市场定位（整机 TCO、decode、合规语境），**不等于** Ascend 软硬件栈在 compute-bound prefill 上的物理极限。反例见 §4（华为自己公开的 DeepSeek prefill 效率已远高于「20% MFU」叙事）。

---

## 4. 正面证据：910C 上 DeepSeek Prefill 已可达约 33% MFU

### 4.1 CloudMatrix-Infer 端到端数字（DeepSeek-R1，INT8）

论文：**Serving Large Language Models on Huawei CloudMatrix384**（CloudMatrix-Infer），arXiv:[2506.12708](https://arxiv.org/abs/2506.12708)。


| 配置                             | Prefill 吞吐                 | Compute efficiency       | 来源（v3）             |
| ------------------------------ | -------------------------- | ------------------------ | ------------------ |
| CloudMatrix-Infer Default      | **5,655** tokens/s per NPU | **3.76** tokens/s/TFLOPS | Table 2 / §5.2     |
| CloudMatrix-Infer Perfect EPLB | **6,688** tokens/s per NPU | **4.45** tokens/s/TFLOPS | Table 2 / Abstract |
| SGLang on H100（Perfect EPLB）   | 7,417 tokens/s per GPU     | **3.75** tokens/s/TFLOPS | Table 2（对比）        |
| DeepSeek on H800（Profile）      | 7,839 tokens/s per GPU     | **3.96** tokens/s/TFLOPS | Table 2            |


**实验条件要点**（同文）：

- 模型：DeepSeek-R1，**671B** 总参数，每 token 激活约 **37B**（文中 §3.5.1）；  
- Prefill：Input Length **4,096**；Batch Size **16,384** tokens/accelerator（Table 2）；  
- 精度：Ascend 910 上 **INT8** 量化推理（§5.1）；  
- Perfect EPLB 为理想专家负载均衡假设下的投影（Table 2 说明文字）。

论文结论：prefill 的 tokens/s/TFLOPS **超过** SGLang@H100 与 DeepSeek@H800 的已发表效率（Abstract / Table 2 / §5.2）。

### 4.2 将 6,688 tokens/s 换算为近似 MFU

**假设**（推理前向、以激活参数为主的粗算，未单独加 attention FLOPs）：

- **分子（实际有用 FLOP/s）**：每 token 前向约 `2 × 激活参数` 次运算，再乘 tokens/s  
`2 × 37×10⁹ × 6688 ≈ 495` TFLOP/s  
  - 「2×参数」为 dense GEMM 前向常见近似；37B 激活见 CloudMatrix-Infer §3.5.1 / DeepSeek-V3 技术报告。
- **分母（硬件峰值 FLOP/s）**：由论文效率反推 `6688 / 4.45 ≈ 1503` TFLOPS，与 INT8 峰值 ≈ `2 × 752 = 1504` TOPS 一致（910C FP16 752 见 §2.1；INT8 常取 FP16 Tensor 的 2 倍）。此处必须用 **INT8 峰值**，因为实测跑的是 INT8。

```text
MFU ≈ 实际有用 FLOP/s ÷ 硬件峰值 FLOP/s
    ≈ 495 / 1504
    ≈ 33%
```

**说明**：

1. 若分子再计入 attention FLOPs，估算 MFU 会略高；
2. 若用 INT8 达成吞吐去除 **FP16** 峰值 752，会把数字虚高到约 66%，**口径错误**（精度必须分子分母一致）；
3. Default 配置 5,655 tokens/s → 同口径约 **28% MFU**（按比例缩放）。

**要点**：华为自己公开的 DeepSeek-R1 prefill 已约 **28%–33% MFU**，远高于「20% 天花板」叙事。

### 4.3 算子级利用率（说明 gap 不在大矩阵乘）

同文 CloudMatrix-Infer（arXiv:2506.12708）§5.5：


| 算子                                | 指标                  | 数值                                              | 来源               |
| --------------------------------- | ------------------- | ----------------------------------------------- | ---------------- |
| INT8 GEMM（单 die）                  | Compute utilization | **77.4%–82.7%**                                 | Table 10         |
| MLA（compute-intensive, BF16/FP16） | TFLOPS utilization  | **65.4%**（910C die） vs **66.7%**（H800 FlashMLA） | Table 8          |
| MLA（memory-intensive）             | 带宽利用率               | **84.1%**（1,346 / 1,600 GB/s） vs H800 **89.6%** | Table 9          |
| Prefill microbatch pipeline       | 端到端吞吐提升             | **+23% ~ +31%**                                 | Figure 21 / §5.4 |
| Prefill microbatch                | 每层 overall latency  | 约 **-24%**                                      | Figure 21(b)     |


**解读**：大 GEMM 已接近峰值 80%；端到端约 33% 与算子约 80% 的落差，主要来自 attention 尾处理、量化/反量化、MoE permute/routing、通信暴露与气泡等 **glue**，而非 Cube 算不动。

### 4.4 训练侧基准率（Ascend 栈可吃到的利用率）


| 工作                 | 规模                           | MFU                     | 来源                                                           |
| ------------------ | ---------------------------- | ----------------------- | ------------------------------------------------------------ |
| Pangu Ultra（dense） | 135B，8,192 Ascend NPU        | **>50% / >52%**         | arXiv:[2504.07866](https://arxiv.org/abs/2504.07866)（摘要与 §4） |
| Pangu Ultra MoE    | 718B（约 39B 激活），6K Ascend NPU | **30.0%**（基线 **18.9%**） | arXiv:[2505.04519](https://arxiv.org/abs/2505.04519)         |


Pangu Ultra MoE 架构与 DeepSeek 高度同构（MoE + MLA + 大规模 EP）；文中称相对基线 MFU 提升约 **58.7%**（相对增幅）。说明 **约 19% 是未优化起点，不是硬件终点**。

---

## 5. 架构差异：为何「算子融合 / overlap」决定 MFU

内部对比材料中的「差异 1」：


| 平台          | Cube/Tensor ↔ Vector/CUDA 数据通路                                   |
| ----------- | ---------------------------------------------------------------- |
| Ascend 910C | Cube 用 L0A/L0B/L0C；Vector 用 UB；跨核常需 **L0C → Global Memory → UB** |
| NVIDIA GPU  | Tensor Core 与 CUDA Core 经 **寄存器** 直接传递                           |


**来源**：内部 910C vs H200 架构对比材料。

该路径使 **向量密集、不规则访存** 的算子（indexer、top-k、gather、部分量化）更容易暴露为串行开销，压低端到端 MFU。这与 §4.3「GEMM 高、端到端中等」一致。

---

## 6. DeepSeek V3.2 对目标的影响

DeepSeek-V3.2 引入稀疏注意力（DSA：lightning indexer + top-k 选择），长上下文下将 attention 复杂度从 `O(L²)` 降为约 `O(L·k)`（见 DeepSeek-V3.2 技术报告 / 官方发布说明）。

对 **compute-bound prefill MFU** 的含义：

- **利好**：context 变长时，cube 友好的 FFN/MoE GEMM 占比上升（GEMM 已可到约 80%，见 §4.3）；  
- **风险**：indexer / top-k / gather 增加向量与不规则访存；在中等 context 下稀疏收益不足时，可能成为新瓶颈（见 §5）。

因此 V3.2 **不是**自动保证 40%，但在长序列 compute-bound 场景下对冲架构短板是有利条件。

---

## 7. 40% 目标是否可达：判断与前提

### 7.1 相对位置


| 锚点                       | 约略 MFU  | 依据                   |
| ------------------------ | ------- | -------------------- |
| 错误叙事「对标 H20」             | 20%     | 148/752，非 MFU        |
| CloudMatrix Default      | 约 28%   | 5,655 tok/s，§4.2 同口径 |
| CloudMatrix Perfect EPLB | 约 33%   | 6,688 tok/s，§4.2     |
| **本目标**                  | **40%** | 相对 33% 约 **1.21×**   |
| INT8 GEMM 算子上限           | 约 80%   | Table 10             |


从已发表的约 33% 到 40%，需要的是进一步压缩 glue、做实 EPLB、以及把 V3.2 稀疏路径藏进 Cube 流水，**不是**突破 GEMM 80% 物理上限。

### 7.2 主要风险（按优先级）

1. **V3.2 indexer + top-k + gather** 无法与 AIC 充分 overlap（§5）；
2. **大 EP 下专家负载不均**（CloudMatrix Default 3.76 vs Perfect EPLB 4.45 tokens/s/TFLOPS，差距即证据）；
3. **MFU 口径不一致**（INT8 峰值 vs FP16 峰值；indexer FLOPs 是否计入「有效」）；
4. CANN/融合算子成熟度与 AIC–AIV–通信三级 overlap 工程量。

### 7.3 建议的口径钉死方式（避免自欺）

报告 40% 时建议同时声明：

1. **分子**：模型有效计算量（或由其导出的有用 FLOP/s）；indexer 等辅助算子是否计入「有效」必须写明；
2. **分母**：与真实运行精度一致的峰值算力——910C 无原生 FP8、GEMM 走 INT8 时应用 **INT8 峰值（约 1504 TOPS/包）**，而非 FP16 752；若用「峰值 × 时间」形式，时间取同一次测量的墙钟时间；
3. **场景**：context 长度区间、batch tokens/NPU、是否 PD 分离、EP 度、是否 Ideal EPLB。

---

## 8. 对「华为都只能对标 H20，凭什么你能做到」的回应

重新 framing：


| 质疑版本                | 更正后的问题                                                    |
| ------------------- | --------------------------------------------------------- |
| 超越华为做不到的物理极限        | 把华为已公开的 DeepSeek prefill ~**33% MFU** 再推到 **40%**（约 1.2×） |
| 20% 是天花板            | 20% 来自峰值比误用；公开文献已给出约 28%–33%                              |
| 对标 H20 ⇒ prefill 不行 | 「对标」主战场是带宽/decode；prefill 恰是 910C 算力轴优势区间                 |


**凭什么可能做到**：不是「比华为更懂芯片」，而是目标落在已发表算子利用率（GEMM 约 80%、MLA 约 65%）与端到端（约 33%）之间的 **工程缝隙**，且 V3.2 长序列稀疏对冲部分架构短板。失败模式清晰（indexer/EPLB），可用 profiling 证伪，而非不可证伪的「权威天花板」。

---

## 9. 一句话结论

**40% compute-bound prefill MFU 在 910C 上不是「违背华为只能对标 H20」的神话，而是相对 CloudMatrix-Infer 已发表约 33% 的约 1.2× 工程冲刺；真正需要证明的是 V3.2 稀疏注意力路径与 MoE 负载能否把 glue 从端到端里再挤掉一截，而不是去推翻一个量纲错误的 20%。**