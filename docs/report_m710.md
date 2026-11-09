# 补报单 · Mission 710: GPU Cell List Native Spatial Search for DEM Contact Detection

> **补报性质**：本单为 m7988 补报单。m710 已成功交付并合并（PR #710, merge commit `5d05a1a3`），但当时未发布论文级进展报告。本报告基于 m710 交付时留存的工件（commit `d3b6c914` 的 `artifacts/dem3d_neighbor_scale/` 与 `docs/dem3d_neighbor_scale_benchmark.md`）程序化提取数字，零手抄。

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

### 1.1 物理背景

离散元方法（DEM）中，接触检测是计算瓶颈。传统 O(N²) 暴力搜索在粒子数超过数千后迅速不可行。Cell list（均匀网格空间哈希）是经典 O(N) 替代方案：将空间划分为边长等于截断距离 d_c 的立方格子，仅需检查 14 个半模板偏移（half stencil）内的邻居对。

本节点（IN14-D2）为冰区航行战役 DAG 节点（主计划 m611 播种），目标是将 DEM 接触检测的空间搜索从 Python dict 实现迁移至向量化 cell list，验证加速比与精度。

### 1.2 几何与参数

| 参数 | 值 |
|---|---|
| 粒子半径 | 0.025 m |
| 粒子密度 | 917 kg/m³（冰） |
| 格子间距 | 0.05 m (= 2r) |
| 接触截断 d_c | 0.05 m |
| 预应变 | 0.985 × spacing（确保接触活跃） |
| 接触刚度 | 5000 N/m |
| 接触阻尼 | 5.0 N·s/m |
| 时间步 | 0.1 × recommended_dt |
| 线程数 | 1（torch 默认 128 线程在此规模反而更慢） |
| 平台 | loongarch64, Linux 6.6.52, PyTorch 2.7.1 |
| 样本 | n³ 粒子立方阵列，n = 10, 15, 20, 22, 28（10³ ~ 2.2×10⁴） |

### 1.3 范围声明（未声称的内容）

- **无 GPU/CUDA 内核**：向量化搜索是 tensor 程序，理论上可跑在 `device="cuda"` 上，但所有报告数字均为 CPU（loongarch64）。CUDA 为路线图项目。
- **无 10⁵ 粒子生产运行**：承诺规模为 10⁴；2.2×10⁴ 作为余量报告，10⁵ 为路线图。

---

## 2. 方法（怎么做 / 流程 / 关键设置 / 工具链）

### 2.1 向量化 cell list 算法（`tensordem.neighbor_search`）

1. **格子分配**：`floor((x - x_min) / h)` 整体张量化，h = d_c
2. **计数排序**：`torch.sort` 对线性格子键 `(ix*ny + iy)*nz + iz` 排序，每个格子变为连续切片
3. **半模板**：13 个偏移 + 自身格子，每对无序对恰好生成一次
4. **向量化展开**：`repeat_interleave` + gather，无逐粒子 Python 循环
5. **平方距离过滤**：`sum(dx²) <= d_c²`（无 sqrt、无除法）
6. **一次规范排序**：pair codes `i*N + j` 排序一次

复杂度：固定密度下 O(N)，工作常数为 13 × 占据数。

### 2.2 逐粒子 Oracle

`brute_force_pairs` 执行分块 O(N²) 扫描，无空间假设，应用相同平方距离规则和相同规范排序。两数组可逐元素比较，使 "diff = 0" 成为有意义的测试。

### 2.3 求解器集成

`DEM3DConfig.neighbor_backend` 选择 `"python"`（默认，逐位兼容旧版）/ `"cell_list"` / `"brute_force"`。两个向量化后端发射相同排序的对数组，因此轨迹逐位相同。

### 2.4 工具链

- `src/tensordem/neighbor_search.py`（新增，344 行）
- `src/tensordem/dem3d.py`（后端切换，+32 行）
- `scripts/benchmark_dem3d_neighbor_scale.py`（420 行）
- `scripts/gen_dem3d_neighbor_scale_doc.py`（357 行）
- `tests/test_dem3d_neighbor_scale.py`（280 行，16 用例）
- `artifacts/dem3d_neighbor_scale/`（results.json, 2 CSV, 3 PNG）

---

## 3. 计算结果（数字表，程序化提取）

### 3.1 强缩放（单搜索与完整时间步）

| 算例 | N | 对数 | 搜索 cell list [ms] | 搜索 Python dict [ms] | 搜索 Oracle O(N²) [ms] | 加速比 vs Python | us/粒子 (cell list) | us/粒子 (Python) | 步 cell list [ms] | 步 Python [ms] |
|---|---|---|---|---|---|---|---|---|---|---|
| N1e3 | 1,000 | 2,700 | 17.19 | 12.95 | 99.20 | 0.75× | 17.19 | 12.95 | 40.46 | 43.70 |
| N3_4e3 | 3,375 | 9,450 | 28.61 | 60.32 | 1,074.90 | 2.11× | 8.48 | 17.87 | 93.16 | 148.82 |
| N8e3 | 8,000 | 22,800 | 49.71 | 124.82 | 6,366.89 | 2.51× | 6.21 | 15.60 | 213.02 | 299.14 |
| N1e4 | 10,648 | 30,492 | 63.24 | 175.25 | 10,940.17 | **2.77×** | 5.94 | 16.46 | 274.79 | 410.92 |
| N2_2e4 | 21,952 | 63,504 | 129.71 | 398.91 | n/a | 3.08× | 5.91 | 18.17 | 582.62 | 911.14 |

**关键观察**：
- 向量化搜索在 ~2×10³ 粒子处超越 Python dict；低于此规模时 14 次向量模板遍历的开销占主导（N=1000 时仅 0.75×）
- 在承诺规模 N=10,648 处，搜索从 175.25 ms 降至 63.24 ms（**2.77× 加速**）
- Oracle 需 10,940 ms，即 cell list 比 O(N²) 参考快 **173×**（同答案）
- 每粒子搜索成本从 ~8×10³ 粒子起趋于平稳（10⁴ 时 5.94 us，2.2×10⁴ 时 5.91 us），而 Python 路径持续上升（12.95 → 18.17 us）
- **完整时间步**加速比在 10⁴ 处为 1.50×（小于搜索加速比），因为接触内核、切向历史与断裂簿记现在占主导——搜索不再是瓶颈，这正是本次改动的目的

### 3.2 正确性：对数组与力，diff = 0

| 量（N = 10,648） | cell list | 逐粒子 Oracle | 差异 |
|---|---|---|---|
| 对数 | 30,492 | 30,492 | **0** |
| 数组元素差异数 | — | — | **0** |
| 壁钟 [ms] | 74.72 | 11,313 | — |
| 力向量元素差异数（单步，两个向量化后端） | — | — | **0** |
| 最大 |F| [N] | 13.500 | 13.500 | 0 |

### 3.3 弱缩放

| 标签 | N | 搜索 [ms] | 步 [ms] | ms/粒子（搜索） | 弱效率 vs 1e3 |
|---|---|---|---|---|---|
| N1e3 | 1,000 | 17.19 | 40.46 | 0.0172 | 1.00 |
| N1e4 | 10,648 | 63.24 | 274.79 | 0.00594 | **2.89** |

弱效率 > 1 表示每粒子搜索成本在规模增长 10.6× 时反而下降（更好-than-linear）。

### 3.4 可复现性

相同配置运行两次，5 步，位置/速度/时间的 SHA-256 摘要完全相同。

---

## 4. 结果图

原始工件包含三张 PNG 图（commit `d3b6c914` 的 `artifacts/dem3d_neighbor_scale/`）：

1. **`neighbor_search_scaling.png`** — 三种后端的壁钟 vs N 对数-对数图，含理想 O(N) 参考线；Oracle 遵循 O(N²)，向量化搜索跟踪 O(N) 参考线
2. **`neighbor_search_speedup.png`** — 相对 Python cell list 的加速比 vs N，显示 ~2×10³ 处的交叉点与单调增长至 10⁴ 处的 2.77×
3. **`neighbor_search_weak_scaling.png`** — 10³ 与 10⁴ 处的每粒子搜索成本

> **补报说明**：本补报单未重新生成图片（工件已存在于 git 历史中）。如需内嵌 base64 图片，可从 `git show d3b6c914:artifacts/dem3d_neighbor_scale/neighbor_search_scaling.png` 提取。

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 线程数是协议的一部分

loongarch64 上 torch 默认 128 线程使这些小张量操作比单线程**更慢**——并行屏障成本超过算术成本。所有数字均为 `torch.set_num_threads(1)`。跨机器比较必须固定线程数。

### 5.2 格子宽度必须等于截断距离

早期版本为"安全"将格子宽度填充 `1 + 1e-9`。当格点恰好位于格子边界时，每个格点被推入前一格子，占据数翻倍，"安全"填充使吞吐量减半。`h = d_c` 是精确的。

### 5.3 截断平局是浮点决策

恰好 `d = d_c` 的粒子位于 `overlap = 0`；是否保留取决于最后一位。基准测试将样本预应变至 `0.985 × spacing` 使邻居数明确。

### 5.4 排序影响舍入，不影响物理

旧版 Python 后端按 `dict` 插入顺序发射对；向量化后端按规范 `i*N + j` 顺序。力组装（`index_add`）在最后一位级别对顺序敏感，因此两族可在 ~1e-16 相对差异内不同（测试： agreement to 1e-12）。所有逐位相同声明均在规范顺序族内。

### 5.5 默认后端未变

`neighbor_backend` 默认为 `"python"`：本节点添加可选快速路径，不默默重排任何现有轨迹。

### 5.6 未声称的内容

- 无 GPU 吞吐量声明
- 无 10⁵ 粒子声明
- 无接触内核成本声明（仅观察到它现在占主导）

---

## 6. 误差判定

### 6.1 原单验收判据 vs 实际交付

| 原单判据 | 要求 | 实际 | 判定 |
|---|---|---|---|
| `gpu_speedup_vs_cpu >= 10.0` | ≥ 10.0× | **未留档 / 未实现** | ❌ **未满足** |
| `force_consistency` (隐含 diff=0) | 0 | **0** | ✅ 满足 |
| 加速比（CPU cell list vs CPU Python dict） | 未明确 | **2.77× @ 10⁴** | ✅ 部分满足 |
| 加速比（CPU cell list vs CPU O(N²) oracle） | 未明确 | **173× @ 10⁴** | ✅ 满足 |

### 6.2 关键量误差 vs 基准

| 关键量 | 基准 | 测量值 | 误差 | 是否 < 3% |
|---|---|---|---|---|
| 对数组（vs Oracle） | 逐位相同 | diff = 0 | **0%** | ✅ 是 |
| 力向量（两向量化后端） | 逐位相同 | diff = 0 | **0%** | ✅ 是 |
| 15 步轨迹（两向量化后端） | 逐位相同 | SHA-256 相同 | **0%** | ✅ 是 |
| GPU 加速比 | ≥ 10.0× | **未实现** | N/A | ❌ **不适用** |

**说明**：物理声明是**精确的**，非近似的——对数组和单步力向量差异为 **0** 元素，DEM3D 轨迹在两个向量化后端间逐位相同——偏差 0%，远在 3% 带内。壁钟数字是性能测量，非物理量；它们仅携带测量不确定度（空闲 128 核 loongarch64 节点上的 best-of-5 计时，单线程）。

**无 GPU 实现**：原单目标包含 GPU 加速，但实际交付为 CPU 向量化 cell list。GPU/CUDA 内核为路线图项目（§6 of original doc）。这是**部分交付**，非完整满足原单 goal。

### 6.3 无物理参考值

本节点不涉及实验/文献物理参考值，因此无 15% 级比较。参考为逐粒子 O(N²) Oracle，按构造精确。

---

## 7. 审计轨迹

| 步骤 | 命令 | 结果 |
|---|---|---|
| 单元/回归测试 | `python -m pytest tests/test_dem3d_neighbor_scale.py -q` | 16 passed |
| 旧版路径回归 | `python -m pytest tests/test_dem3d.py tests/test_dem3d_energy_audit.py tests/test_dem3d_contact_invariants.py -q` | 39 passed, 56 subtests passed |
| 基准测试 | `python scripts/benchmark_dem3d_neighbor_scale.py --output artifacts/dem3d_neighbor_scale` | results.json, 2 CSV, 3 figures |
| 文档生成 | `python scripts/gen_dem3d_neighbor_scale_doc.py` | docs/dem3d_neighbor_scale_benchmark.md |
| 合并 | PR #710, merge commit `5d05a1a3` | 已合并至 main |

---

## 8. 结论

Mission 710 交付了 **CPU 向量化 cell list 空间搜索**，在 10⁴ 粒子规模实现：
- **2.77×** 搜索加速（vs Python dict）
- **173×** 搜索加速（vs O(N²) Oracle）
- **逐位相同**的正确性（diff = 0）
- **O(N)** 缩放（每粒子成本在 8×10³ ~ 2.2×10⁴ 趋于平稳）

**未满足原单 goal 中的 GPU 加速要求**（`gpu_speedup_vs_cpu >= 10.0`）。GPU/CUDA 实现为路线图项目。

本节点为冰区航行战役 DEM 接触检测提供了性能基础，搜索不再是瓶颈；后续优化应聚焦接触内核与 10⁵ 粒子规模。

---

**补报人**：m7988 补报智能体  
**日期**：2026-11-09  
**数据来源**：commit `d3b6c914` 的 `artifacts/dem3d_neighbor_scale/results.json` 与 `docs/dem3d_neighbor_scale_benchmark.md`，程序化提取，零手抄。
