# AMD GPU 气泡分析指南（对标 NVIDIA nsys）

本文说明在 AMD ROCm 上如何用现有工具分析 vLLM 的 GPU 气泡（GPU idle / host-bound gap），对应 NVIDIA 侧常用的 `nsys` + NVTX 工作流。

产物目录：`/root/jiangzhiqi/repo/vllm/amd_bubble_profile/`

---

## 1. 工具对标

| NVIDIA | AMD（本机可用） | 用途 |
|--------|-----------------|------|
| **nsys** | **rocprofv3** | 时间线：CPU API / GPU kernel / memcpy / 用户 marker |
| **NVTX** | **ROCTx (`roctx`)** | 在时间线上标注业务区间（如 `step: N`） |
| Nsight Systems GUI | **Perfetto**（[ui.perfetto.dev](https://ui.perfetto.dev)） | 打开 `.pftrace` 查看气泡 |

结论：分析 GPU 气泡优先用 **`rocprofv3` + Perfetto**，无需自写脚本。

工作流对应关系：

```text
nsys profile -t cuda,nvtx,osrt ... app
  →  rocprofv3 --runtime-trace -f pftrace -- app

nsys-ui xxx.nsys-rep
  →  Perfetto 打开 xxx.pftrace

NVTX rangePush / rangePop
  →  roctx.rangePush / roctx.rangePop
```

---

## 2. 代码侧 ROCTx 埋点

文件：`vllm/v1/engine/core_client.py`（`InprocClient`）

为对齐 nsys 里的 NVTX step 区间，在 `get_output()` 中埋了 ROCTx：

- `roctx.rangePush("run_busy_loop")`：整段 busy loop（shutdown 时 Pop）
- `roctx.rangePush(f"step: {self._step}")` … `roctx.rangePop()`：每一步边界
- `torch.cuda.synchronize()`：强制等待 GPU，便于把 step 墙钟时间与 GPU 活动对齐（会放大同步气泡，适合诊断）

依赖：`import roctx`（ROCm 自带；若 conda 环境找不到，可软链到 `/opt/rocm-*/lib/python3.*/site-packages/roctx`）。

**必须**设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`：默认多进程下 EngineCore / GPU kernel 在子进程中执行，`rocprofv3` 挂在父进程上时 Perfetto 里看不到 kernel 轨，且 InprocClient 的 ROCTx 埋点也不会覆盖真正执行路径。

---

## 3. 采集命令

```bash
conda activate vllm-rocm
cd /root/jiangzhiqi/repo/vllm
export HIP_VISIBLE_DEVICES=1
# 必须：否则 GPU kernel 在子进程，Perfetto 无 kernel 信息
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH=/root/jiangzhiqi/repo/vllm:${PYTHONPATH}

rocprofv3 --runtime-trace --stats \
  -d /root/jiangzhiqi/repo/vllm/amd_bubble_profile \
  -o vllm_basic_bubble \
  -f pftrace csv \
  -- python examples/basic/offline_inference/basic.py
```

关键参数：

| 参数 | 作用 |
|------|------|
| `--runtime-trace` | HIP runtime + kernel + memcpy + **Marker (ROCTx)** 等（接近 `nsys profile`） |
| `--stats` | 生成各 domain 的统计 CSV |
| `-f pftrace csv` | Perfetto 时间线 + CSV 汇总 |
| `-d` / `-o` | 输出目录与文件前缀 |

说明：

- `--runtime-trace` **已包含 ROCTx**，不必再加 `--marker-trace`（加了也只是冗余）。
- `--group-by-queue` 可选，**只影响 Perfetto 里 kernel/memcpy 的归轨方式，采集内容不变**（按截图实测）：
  - **去掉（默认）**：按 **HIP stream** 分组，轨道为 `STREAM [0]` / `STREAM [1]` / `STREAM [2]`
  - **加上**：按 **HSA 硬件队列** 分组，轨道为 `COMPUTE Agent [N] QUEUE [0/1/2] GPU`
  - 注意：rocprofv3 help 中 "displaying the HIP streams ... rather than HSA queues" 的措辞与实际行为相反，以实测为准。
- 更重的全量 trace 可用 `--sys-trace`（额外含 HSA / HIP compiler API，文件更大）。

---

## 4. 如何在时间线上判断 GPU 气泡

打开：

`amd_bubble_profile/vllm_basic_bubble_results.pftrace`

→ 浏览器访问 [https://ui.perfetto.dev](https://ui.perfetto.dev) → Open trace file。

对齐三条轨：

1. **MARKER（ROCTx）**：`step: 0/1/2...` —— CPU 侧一步边界  
2. **HIP API**：`hipLaunchKernel` / `hipDeviceSynchronize` / `hipMemcpy*` —— host 提交与同步  
3. **KERNEL_DISPATCH**：真实 GPU 占用  

气泡判据：某个 `step: N` 区间内，kernel 轨出现明显空隙，而 CPU 仍在跑 HIP/调度 → GPU idle / host-bound bubble。

建议：

- 不要用 `step: 0` 下结论（含 warmup / JIT / 首次编译）
- 看稳态 decode step（如 `step: 5/6/9/10`）里 kernel 是否连续
- 注意代码里每步 `torch.cuda.synchronize()` 会把异步重叠压掉，气泡更容易暴露在 step 边界

---

## 5. 本次采集结果摘要

运行日志中的 step 墙钟时间（与 marker 一致）：

| step | 墙钟时间 | 说明 |
|------|----------|------|
| 0 | ~2308 ms | 首步，含 warmup/JIT |
| 1 / 2 | ~20 ms | decode |
| 3 | ~0.5 ms | 收尾 |
| 4 / 8 | ~240 ms | 后续 generate 的首步（仍偏长） |
| 5/6/9/10 | ~18–22 ms | 稳态 decode |
| 7/11 | ~0.5 ms | 收尾 |

`domain_stats` 占比（整段 profile）：

| Domain | Percentage | 含义 |
|--------|------------|------|
| HIP_API | ~52% | Host API 很重 |
| KERNEL_DISPATCH | ~24% | 真实 GPU 计算 |
| MARKER_API | ~21% | ROCTx 区间（含 sync 后的 step 墙钟） |
| MEMORY_COPY 等 | <2% | 拷贝不是主因 |

HIP API 侧：`hipDeviceSynchronize` 约占 HIP 时间 ~29%，与每步 `torch.cuda.synchronize()` 一致。

Marker 已落盘：`vllm_basic_bubble_marker_api_*.csv` 中可见 12 个 `step: N`，可与 kernel timeline 对齐。

---

## 6. 产物清单

目录：`/root/jiangzhiqi/repo/vllm/amd_bubble_profile/`

| 文件 | 用途 |
|------|------|
| `vllm_basic_bubble_results.pftrace` | **主文件**：Perfetto 看气泡 |
| `*_marker_api_trace.csv` / `*_marker_api_stats.csv` | ROCTx `step: N` 区间 |
| `*_kernel_trace.csv` / `*_kernel_stats.csv` | GPU kernel 时间线与汇总 |
| `*_hip_api_trace.csv` / `*_hip_api_stats.csv` | HIP API 时间线与汇总 |
| `*_domain_stats.csv` | 各 domain 总占比 |
| `*_memory_copy_*.csv` / `*_memory_allocation_*.csv` | 拷贝 / 分配 |
| `*_agent_info.csv` | GPU/Agent 信息 |

---

## 7. 环境注意

1. **必须单进程**：`export VLLM_ENABLE_V1_MULTIPROCESSING=0`。否则 GPU kernel 在子进程生成，Perfetto 中没有 kernel 信息；同时 InprocClient 的 ROCTx 埋点也不生效。  
2. **ROCTx Python 包**：`python -c "import roctx"` 必须成功，否则 marker 轨为空。  
3. **设备**：`HIP_VISIBLE_DEVICES=1` 与采集时一致。  
4. **可视化**：本机无 Nsight Systems GUI；用 Perfetto 打开 `.pftrace` 即可。  
5. **不要自写解析脚本**：优先用 `rocprofv3` 自带的 pftrace + stats CSV + Perfetto。
