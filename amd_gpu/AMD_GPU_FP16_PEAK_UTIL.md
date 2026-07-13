# AMD GPU 理论 FP16 峰值利用率测试

本文记录在本机 AMD GPU 上，用现有 ROCm 库测理论 FP16（matrix）峰值利用率的方法与结果，回答：**能否达到 95% 以上？**

环境：`conda activate vllm-rocm`，工作目录 `/root/jiangzhiqi/repo/vllm`。

---

## 1. 结论

**达不到 95%。** 在当前硬件与 ROCm 7.2 库下，大尺寸 FP16 GEMM 稳定约在 **91%–92%**（相对标称峰值）；按实测运行频率折算约 **94.2%**，仍未到 95%。


| 指标                          | 数值                                          |
| --------------------------- | ------------------------------------------- |
| GPU                         | AMD Radeon PRO W7900D（gfx1100，96 CU）        |
| 标称峰值频率                      | 1760 MHz                                    |
| 理论 FP16 matrix 峰值 @1760 MHz | **86.508 TFLOPS**                           |
| 最佳短测吞吐                      | 79.430 TFLOPS → **91.82%**（6144×12288×4096） |
| 持续测吞吐（2000 iter）            | 79.009 TFLOPS → **91.33%**                  |
| 持续测实测 GFX 时钟                | ~1706 MHz，功耗 ~229 W / 241 W cap             |
| 相对实测时钟折算                    | ~**94.22%**                                 |
| 达到 95% 所需吞吐                 | ≥ 82.182 TFLOPS                             |


说明：`amd-smi` 上 GFX% = 100% 只表示引擎忙碌，**不等于**算力达到理论 FP16 峰值。

---

## 2. 理论峰值怎么算

W7900D：96 CU × 64 ALU/CU = **6144** 个 FP32 ALU。

Navi31 类 matrix FP16 相对 FP32 可按 **×2** 计（dual-issue / matrix 路径）：

```text
FP32_peak  = 6144 × 4 × (f_GHz)          → @1760 MHz ≈ 43.254 TFLOPS
FP16_peak  = FP32_peak × 2               → @1760 MHz ≈ 86.508 TFLOPS
```

利用率：

```text
util = achieved_TFLOPS / FP16_peak × 100%
achieved = 2 × M × N × K / time_sec / 1e12
```

---

## 3. 测法（尽量用现成库，不自写 kernel）

本机可用、且适合测 FP16 GEMM 墙钟吞吐的工具：


| 工具                                   | 状态                           | 用途                                                |
| ------------------------------------ | ---------------------------- | ------------------------------------------------- |
| `**MIOpenDriver gemmfp16**`          | 已安装（`/usr/bin/MIOpenDriver`） | 官方 FP16 GEMM 计时，本次主测                              |
| `rocblas-bench` / `hipblaslt-bench`  | 本机未装 clients 包               | 更标准的 BLAS 峰值扫参；可选后续安装                             |
| PyTorch `bench_fp16_peak_pytorch.py` | 脚本见同目录                       | `torch.matmul` + CUDA Event；NN 布局可接近 MIOpenDriver |


### 3.1 推荐命令

```bash
conda activate vllm-rocm
cd /root/jiangzhiqi/repo

# 选空闲卡，拉高性能档
export HIP_VISIBLE_DEVICES=1
rocm-smi -d 1 --setperflevel high

# 短测：计时单次 kernel
MIOpenDriver gemmfp16 -m 8192 -n 8192 -k 4096 -i 300 -t 1 -V 0 -C 1

# 持续测：拉长迭代，方便看功率/时钟稳态
MIOpenDriver gemmfp16 -m 8192 -n 8192 -k 4096 -i 2000 -t 1 -V 0 -C 1
```

关键参数：

- `-t 1`：输出 GPU kernel 耗时（ms）
- `-V 0`：关闭校验，避免干扰计时
- `-C 1`：列主序
- `-u` / `-v`：转置 A/B（本次默认 `u=0,v=0` 最快）

### 3.2 边测边看功耗与时钟

```bash
amd-smi monitor -g 1 -p -u -w 1 -i 20
```

关注：`GFX_CLK`、`GFX%`、`POWER` / `PWR_CAP`。

---

## 4. 实测数据摘要

硬件：GPU1，`HIP_VISIBLE_DEVICES=1`，perflevel=high。

### 4.1 Shape / layout 扫参（`-i 300`）


| Shape / layout             | Elapsed (ms) | TFLOPS    | vs @1760   |
| -------------------------- | ------------ | --------- | ---------- |
| 8192×8192×4096 NN          | 6.944        | 79.17     | 91.5%      |
| 8192×8192×4096 NT (`-v 1`) | 7.255        | 75.77     | 87.6%      |
| 8192×8192×4096 TN (`-u 1`) | 10.107       | 54.40     | 62.9%      |
| 8192×8192×4096 TT          | 7.089        | 77.55     | 89.6%      |
| 7680×7680×4096 NN          | 6.134        | 78.63     | 90.9%      |
| 12288×6144×4096 NN         | 7.797        | 79.33     | 91.70%     |
| **6144×12288×4096 NN**     | **7.786**    | **79.43** | **91.82%** |


最佳短测取 **6144×12288×4096 ≈ 79.430 TFLOPS / 91.82%**（同 flops 下比 12288×6144×4096 略快）。

历史更全的 shape 扫参（同机、同公式）里，相对 1760 MHz 的最优档也多在 **90%–92%**（如 8192×8192×4096、6144³、4608³ 等），未见 ≥95%。

### 4.2 持续跑（`-i 2000`）+ 监控

```text
Elapsed: 6.958098 ms
→ 79.009 TFLOPS → 91.33% @1760 MHz
```

稳态监控（摘录）：

```text
POWER ~228–229 W / 241 W
GFX_CLK ~1705–1706 MHz
GFX% 100%
```

按 1706 MHz 折算峰值 ≈ 83.85 TFLOPS，则利用率 ≈ **94.22%**。

### 4.3 与 PyTorch 对照（实测）

脚本：`/root/jiangzhiqi/repo/vllm/bench_fp16_peak_pytorch.py`

```bash
conda activate vllm-rocm
export HIP_VISIBLE_DEVICES=1
rocm-smi -d 1 --setperflevel high
python /root/jiangzhiqi/repo/vllm/bench_fp16_peak_pytorch.py            # 默认 NN（不转置 B）
python /root/jiangzhiqi/repo/vllm/bench_fp16_peak_pytorch.py --transpose-b  # B^T，更慢
```

条件：GPU1，PyTorch 2.11.0+gitd0c8b1f / HIP 7.2.53211，warmup=30，iters=100，相对峰值 86.508 TFLOPS @1760 MHz。

**布局影响很大：** 默认 `A @ B`（NN）接近 MIOpenDriver；`A @ B.T`（`--transpose-b`）明显掉速。

#### NN（默认，`transpose_b=False`）


| Shape                    | TFLOPS    | util       |
| ------------------------ | --------- | ---------- |
| fp16 3840³               | 78.49     | 90.7%      |
| fp16 4096³               | 76.95     | 88.9%      |
| fp16 5120³               | 77.95     | 90.1%      |
| fp16 6144³               | 78.56     | 90.8%      |
| fp16 8192³               | 75.34     | 87.1%      |
| **fp16 6144×12288×4096** | **79.45** | **91.84%** |
| fp16 12288×6144×4096     | 79.42     | 91.8%      |
| fp16 8192×8192×4096      | 79.34     | 91.7%      |
| bf16 6144³               | 78.12     | 90.3%      |


PyTorch 最佳：**79.45 TFLOPS / 91.84%**，与 MIOpenDriver 最佳短测（79.43 TFLOPS / 91.82%）基本一致。

#### `A @ B.T`（`--transpose-b`）


| Shape                | TFLOPS | util  |
| -------------------- | ------ | ----- |
| bf16 3840³           | 66.47  | 76.8% |
| fp16 3840³           | 65.25  | 75.4% |
| fp16 7680³           | 57.65  | 66.6% |
| fp16 6144³           | 54.08  | 62.5% |
| fp16 6144×12288×4096 | 54.98  | 63.6% |


该布局最佳约 **76.8%**，明显低于 NN。

测「理论峰值利用率」时，PyTorch 用 **NN（默认）** 即可；结果与 `MIOpenDriver gemmfp16` 同级。

---

## 5. 为何难到 95%

1. **功耗墙**：socket 功耗顶在 ~229/241 W，GFX 稳在 ~1706 MHz，上不到标称 1760 MHz；相对标称峰值会再扣约 3%。
2. **库与调度开销**：即便 kernel busy 100%，仍有访存、同步、tile 边界等非理想部分。
3. **本机缺 `rocblas-bench` / `hipblaslt-bench`**：当前已用现成 `MIOpenDriver`；换更激进的 hipBLASLt 解未必能跨过 95%，但值得作为后续对比。

NVIDIA 侧大 GEMM 常可冲到 95%+；本机 W7900D + ROCm 7.2 实测 **稳定天花板约 91%–92%（标称）/ ~94%（实测时钟）**。

---

## 6. 复现检查清单

```bash
conda activate vllm-rocm
export HIP_VISIBLE_DEVICES=<空闲卡>
rocm-smi -d <id> --setperflevel high

# 确认工具
command -v MIOpenDriver

# 跑最佳附近 shape
MIOpenDriver gemmfp16 -m 6144  -n 12288 -k 4096 -i 300 -t 1 -V 0 -C 1
MIOpenDriver gemmfp16 -m 8192  -n 8192 -k 4096 -i 2000 -t 1 -V 0 -C 1

# PyTorch 对照（默认 NN）
python /root/jiangzhiqi/repo/vllm/bench_fp16_peak_pytorch.py

# 另开终端
amd-smi monitor -g <id> -p -u -w 1
```

换算示例（8192×8192×4096，elapsed=6.958098 ms）：

```text
TFLOPS = 2*8192*8192*4096 / (6.958098e-3) / 1e12 ≈ 79.009
util   = 79.009 / 86.508 ≈ 91.33%
```

---

## 7. 一句话

用现成 `MIOpenDriver gemmfp16` / PyTorch NN `matmul` 在 W7900D 上测 FP16 matrix GEMM：**最高约 91.8%（标称 1760 MHz）/ ~94.2%（实测 ~1706 MHz），达不到 95%。** 两者最佳吞吐接近（~79.4–79.5 TFLOPS）。