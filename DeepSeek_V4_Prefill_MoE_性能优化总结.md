# DeepSeek-V4-Flash Prefill MoE 性能优化总结

对象：`DeepSeek-V4-Flash-0731-w8a8`，DP=8 × TP=2、EP=16，256 个 routed expert、topk=6、hidden 4096、专家 intermediate 2048、43 层全 MoE，W8A8 量化，`enable_fused_mc2=1`。

负载：单条 2266 token 的 prompt，`max_tokens=1`，只看首 token 延迟。计时取 `vllm-ascend/vllm_ascend/worker/worker.py` 里 dp rank 0 的 `step N: X ms`（一次 prefill 前向加采样），丢掉第一次图捕获。

**结论先行：首 token 从 465.6 ms 降到 422.8 ms，快了 42.8 ms（9.2%）。收益全部来自修复 EPLB 录表的一个 bug。Cube tiling 方向尝试过，没有收益。**

---

## 一、瓶颈定位

Profiling：`vllm/examples/basic/offline_inference/prof_ds_bs1_prefill/`，rank 0，一次 prefill 共 471.8 ms device 时间。

`DispatchFFNCombine` 这一个融合算子占 **25.8%**（43 次合计 122.1 ms），但其中真正的 MAC 只有 28.5 ms（设备时间的 6.0%）。

以 `op_summary` 第 342 行那次调用为例：

| 区间 | 时间 | 占 kernel |
|---|---|---|
| Task Duration | 2321.4 µs | 100% |
| AIV 在位 | 2318.2 µs | 99.9% |
| AIC 活跃 | 761.0 µs | 32.8% |
| 其中 MAC | 347.0 µs | **14.9%** |

AIV 虽然占满全程，但它四条流水线加起来只有 675.5 µs（vec 140 + scalar 302 + mte2 117 + mte3 116），即 **至少 70.9% 的时间在纯等待**。同一时段 Cube 空闲 1560 µs。两种核同时闲着 1.5 ms，说明等的是外部——EP=16 的 all-to-all dispatch/combine 被吃进了 kernel 内部。

**关键证据：耗时与本卡算量无关。** 43 次调用输入 shape 完全一致，但 `aicore_time` 在 460–3197 µs 之间（6.9 倍），而 duration 与 aicore_time 的相关系数只有 **0.14**。kernel 什么时候结束由 EP 组里最慢的卡决定，这是**专家路由负载不均 + 全员同步**的典型特征。

源码上对应两处全员等待：开头 `tokenPerExpert` 的跨卡 all-gather（在 `hccl_shmem.hpp` 的 `gm_signal_wait_until_ne` 上自旋），和结尾的 `shmem.CrossRankSync()`。

由此得到的优化方向排序：
1. 通信与计算 overlap
2. **专家负载均衡（EPLB）** ← 本次做的
3. 摊薄权重搬运
4. Cube 内部 tiling ← 本次尝试，无收益

---

## 二、专家负载均衡（已完成，-9.2%）

### 2.1 接入

`vllm/examples/basic/offline_inference/basic_ds.py` 增加 `EPLB_MODE` 开关，默认 `off`，不影响原有脚本。

| 模式 | 作用 |
|---|---|
| `off` | 按专家编号固定放置，每卡 16 个专家 |
| `record` | 按 prefill 热度重排并写出专家表（走 eager） |
| `on` | 加载专家表，从第一步就用均衡后的放置 |

```bash
EPLB_MODE=record MAX_TOKENS=1 NUM_ITERS=64 python basic_ds.py   # 录表，约 8 分钟
EPLB_MODE=on     MAX_TOKENS=1 NUM_ITERS=6  python basic_ds.py   # 看 step 那一行
```

冗余副本数 `EPLB_REDUNDANT_EXPERTS` 默认 16，`256 + 16 = 272`，每卡从 16 个专家变成 17 个，显存多约 1 GiB（每层 1 个专家权重 ≈ 24 MiB × 43 层）。

### 2.2 第一次测量：没有收益

开了 EPLB 之后首 token 反而慢了 5.2 ms。卡上数据显示 `aicore_time` 仍在 474–3296 µs 摆动（7.0 倍），离散度从 0.47 升到 0.53。**负载只是被打乱，没有被拉平**（层 18 从 569 涨到 3296，层 5 从 3198 降到 1032）。

### 2.3 根因：录表模式下录出来的专家表是废的

录表模式（`expert_map_record_path` 不为空）下，`forward_end` 永远不会调 `update_expert_map_and_weight`：

```python
# vllm_ascend/eplb/eplb_updator.py
if self.update_expert_weight_flag() and self.expert_map_record_path is None:
    self.eplb_loader.update_expert_map_and_weight(self.reqs)
```

所以卡上的放置从头到尾没变过。但 `EplbWorker.do_update` 每轮结束都会把自己算出来的放置写进 `old_expert_maps` 当成"现状"。下一轮拿到的 `load_info` 仍是**旧放置**下测出的每槽位 token 数，却按**新放置**反推逻辑专家热度——等于把实测负载记到了别的专家头上。

那次录表跑了 **8 轮**再均衡，热度被搅了 8 次，导出的是第 8 轮的结果。

用真实负载离线验证，三者的峰均比（每层 max rank load / mean）：

| 放置 | 峰均比 mean | max |
|---|---|---|
| 默认放置 | 2.359 | 4.166 |
| **坏表** | **2.329** | 3.731 |
| 修复后的表 | **1.115** | 1.244 |

坏表相对默认几乎没有改善。对照组：均衡器自己算出的 `new_placement` 在同一热度下是 1.117——**导出链路本身没问题（离线 round-trip 完全一致），错的是喂给它的热度**。最直观的佐证：layer 0 最热的 5 个专家，均衡器给了 3 份副本，坏表里只有 1 份。

### 2.4 修复

给 `EplbWorker` 增加 `record_only`，只录表时不推进 `old_expert_maps`：

```python
# vllm_ascend/eplb/core/eplb_worker.py
update_info = self.compose_expert_update_info_greedy(new_expert_maps, self.old_expert_maps)
if not self.record_only:
    self.old_expert_maps = new_expert_maps
```

标志由 `model_runner_v1.py` 按 `eplb_config.expert_map_record_path is not None` 传入 `EplbProcess` → `EplbWorker`。

改完后 8 轮再均衡稳定报告同一个真实现状（`current: mean=2.362`），不再漂移。

同时新增 `VLLM_ASCEND_EPLB_DUMP`，把每卡每物理槽位真实收到的 token 数落盘。**这是判断均衡有没有生效的唯一依据**——日志里原有的 imbalance 指标假设流量在副本间均分，坏表照样能报出 1.1。

### 2.5 结果

首 token（后 5 次，取中位数）：

| 配置 | step 中位数 | 对比基线 |
|---|---|---|
| 不开 EPLB | 465.6 ms | — |
| 开 EPLB（修复前） | 470.8 ms | +5.2 ms |
| **开 EPLB（修复后）** | **422.8 ms** | **−42.8 ms（−9.2%）** |

省下来的时间全部出自 MoE 融合算子：

| | device 总时间 | DispatchFFNCombine | rank 0 aicore_time | 算子平均耗时 |
|---|---|---|---|---|
| 不开 | 473.4 ms | 122.1 ms | 441–3198 µs，cv 0.47 | 2840 µs |
| 修复前 | 477.3 ms | 124.4 ms | 474–3296 µs，cv 0.53 | 2894 µs |
| **修复后** | **430.0 ms** | **79.0 ms** | **876–1989 µs，cv 0.15** | **1838 µs** |

算子省 43.1 ms，端到端省 42.8 ms，对得上。离散度 0.47 → 0.15 说明负载是真的被拉平了。两边首 token 输出都是 `如何`，数值一致。

### 2.6 遗留：log2phy 副本选择有缺陷（已知，未改）

`generate_log2phy_map` 用 `(ep_rank % tp + ep_rank // tp + 专家号) % 副本数` 选副本。EP=16、TP=2 时这个 shift 只有 0–8 共 **9 种**取值，**副本数超过 9 的专家，多出来的副本永远收不到 token**。当前表最多 3 份副本，用不到这个上限；实测影响是峰均比 1.115 → 1.125，换成按 `ep_rank` 轮转是 1.121，收益在噪声内，因此没动。

**如果以后把 `EPLB_REDUNDANT_EXPERTS` 调大到让热门专家拿到 10 份以上副本，必须先修这里。**

---

## 三、Cube 内部 tiling（已尝试，无收益）

目标是把 `aic_mac_ratio` 往上推，让 mte2 的权重预取和 mac 真正重叠。依据是 `csrc/third_party/catlass/docs/1_Practice/10_matmul_optimization.md`。

### 3.1 约束

Catlass 模板限制 `L0::M == L1::M`、`L0::N == L1::N`、推荐 `L0::K ≈ 1/4 L1::K`。L0C 128 KB 要求 `L1::M × L1::N × 4 ≤ 131072`，即 **`M × N ≤ 32768`**。当前 `L1<128,256,512>` 正好把 L0C 占满，M 和 N 不能独立调。

### 3.2 PRE 基线

单算子 bench（`dfc_bench/bench_dfc.py`，均匀路由，每专家约 386 token），device 端均值：

| 指标 | PRE `L1<128,256,512>` `L0<128,256,128>` |
|---|---|
| Duration | 1325.5 µs |
| aicore_time | 1047.1 µs |
| aic_mac_time | 605.0 µs |
| aic_mte1_time | 515.0 µs |
| **aic_mte2_time** | **706.5 µs** |
| aic_scalar_time | 558.4 µs |
| aiv_time | 1322.7 µs |

**mte2（706.5）> mac（605.0）**，Cube 被权重搬运卡住。假设是 M 方向切块导致权重重复读：M_e ≈ 386、`L1::M=128` → `CeilDiv(386,128)=4` 个 M 块，权重被完整读 4 遍，且第 4 块只有 2 行有效。

分核是跨专家连续排的（`startCoreIdx = (startCoreIdx + coreLoops) % coreNum`），所以核间负载不均衡不是主因。

### 3.3 变体 A：回退 14.9%

`L1<256,128,256>` `L0<256,128,64>`，M 块数 4 → 2。golden 16 卡 bit-exact。

| 指标 | PRE | A | Δ |
|---|---|---|---|
| Duration | 1325.5 | 1523.6 | **+14.9%** |
| aicore_time | 1047.1 | 1192.4 | +13.9% |
| aic_mac_time | 605.0 | 597.0 | −1.3% |
| aic_mte1_time | 515.0 | 544.1 | +5.7% |
| aic_mte2_time | 706.5 | 643.5 | **−8.9%** |
| aic_scalar_time | 558.4 | 439.0 | −21.4% |
| aic_fixpipe_time | 80.4 | 58.4 | −27.4% |

假设被验证了一半：**mte2 确实降了 8.9%**，scalar、fixpipe 也明显下降。但 `L0::K` 从 128 降到 64 让 mmad 粒度减半，流水重叠变差，aicore_time 反而涨 13.9%，总时长跟着涨。按指南回滚。

### 3.4 变体 B：编不出来

`L1<256,128,512>` `L0<256,128,128>`，想保持 PRE 的 K 粒度、只动 M/N。L0A 需要 `256×128×1×2 = 64 KB`，正好是 L0A 全部容量，设备侧 kernel 编译未产出二进制（`bin/dispatch_ffn_combine` 为空目录），bench 直接崩。未继续。

### 3.5 踩的坑：旧 kernel 冒充新结果

变体 A 第一次测出来和 PRE **逐项相同**（mte1 515.0 → 515.1）。原因是 AscendC 不跟踪 `dispatch_ffn_combine.h` 的头依赖，ninja 认为设备侧 kernel 是最新的，没有重编——两次测的是同一个二进制。

必须先删掉 `csrc/build/binary/ascend910_93/bin/dispatch_ffn_combine*` 和 `gen/DispatchFFNCombine_*` 再 `rebuild.sh`，并核对 `.o` 的时间戳。

> 此坑后续已由 `csrc/cmake/func.cmake` 补上真实文件依赖解决，`dfc_bench/rebuild.sh` 结束时会 diff 源码与暂存目录并打印 `staged kernel sources match csrc`，不一致直接非零退出。

### 3.6 结论

在这个 shape 下，单纯调 L1/L0 TileShape 没有找到正收益配置。M×N ≤ 32768 的 L0C 约束让"增大 M 以减少权重重复读"必然要牺牲 N 或 K 粒度，而 K 粒度对流水重叠更敏感。

> 补充（后续结论）：`aic_mac_time / aicore_time` 并不等于 cube 流水效率，`aicore_time` 把 cube 等 AIV 通信的时间也算作忙。所以以"把 mac_ratio 从 0.456 推到 0.8"为目标本身就偏了，这个比值在通信变长时会自然下降。

---

## 四、改动清单

### vllm-ascend

| 文件 | 改动 |
|---|---|
| `vllm_ascend/eplb/core/eplb_worker.py` | 新增 `record_only`，只录表时不推进 `old_expert_maps`（**核心修复**）；新增 `VLLM_ASCEND_EPLB_DUMP` 落盘真实负载/热度/放置 |
| `vllm_ascend/worker/model_runner_v1.py` | 按 `expert_map_record_path` 把 `record_only` 传给 `EplbProcess` |
| `vllm_ascend/worker/worker.py` | 每步打印 `step N: X ms`，用于首 token 计时 |
| `csrc/common/include/cann_compat.h` | CANN 8.5.2 兼容：`ASCEND950` → `ASCEND910_95` 别名；显式包含 `platform/soc_spec.h` 提供 `NpuArch` |
| `csrc/cmake/func.cmake` | 补 kernel 暂存与代码生成的真实文件依赖 |

### vllm

| 文件 | 改动 |
|---|---|
| `examples/basic/offline_inference/basic_ds.py` | `EPLB_MODE` 三模式开关、`MAX_TOKENS`、`EPLB_REDUNDANT_EXPERTS` |
| `examples/basic/offline_inference/eplb_expert_map.json` | 录好的专家表，43 层 × 16 卡 × 17 专家 |

### 工具

| 路径 | 作用 |
|---|---|
| `dfc_bench/` | DFC 单算子 bench（16 卡直调、golden 校验、profiling），用法见其 `README.md` |
| `vllm/examples/basic/offline_inference/eplb_analyze.py` | 用真实负载核对装箱假设与运行期实际分流的差距 |
| `vllm/examples/basic/offline_inference/eplb_check_log2phy.py` | 重放 log2phy 副本选择规则，统计每个副本实际收到几个源 rank |

---

## 五、下一步建议

按预期收益排序：

1. **通信与计算 overlap。** 修复 EPLB 后 Cube 空窗仍在，这是剩下最大的一块。需确认 mc2 融合算子是否按 chunk 流水（边收边算），而不是 dispatch 全收完才启动 GroupedMatmul。
2. **摊薄权重搬运。** 每次调用要读 16 个专家的 402.7 MB int8 权重（mte2 实测约 1.22 TB/s），43 层合计约 17 GB HBM 读。增大 chunked prefill 的 chunk size 提高每专家的 M，或走 INT4 权重。
3. **kernel 内并行度。** dispatch/combine 循环的并行维度是目的 rank（`dstEpIdx < params.EP = 16`），而 `coreNum = 40`，core 16~39 直接落到 `SyncAll` 空转。把并行维度改成 `dstEpIdx × row-chunk` 理论上这段能快 2.5 倍。
4. **Cube tiling 已验证无收益**，不建议在固定 shape 上继续扫 TileShape；若要继续，应先解决 L0C 的 M×N ≤ 32768 约束（例如改 `l0CStages` 或 epilogue 策略）。
