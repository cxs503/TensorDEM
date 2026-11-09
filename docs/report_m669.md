# Mission 669 补报：3D 船体曲面接触（STL/OBJ 三角网格）

> **补报说明**：本任务（m669）已于 2026-10-31 成功交付并合并（PR #72, commit 0f22cf9），但当时未发布论文级进展报告。本报告基于原交付工件（`runs/m669_acceptance.json`、`docs/m669_hull_mesh_contact.md`、`src/tensordem/hull_mesh_contact.py`、`tests/test_hull_mesh_contact.py`、`scripts/gen_figures_m669.py` 及 `artifacts/fig*.png`）程序化提取数据，零手抄。

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

### 1.1 物理背景

冰区航行中，船舶 hull 与冰块/冰层的接触是 DEM 离散元仿真的核心场景。TensorDEM 此前仅支持 2D polyline（`hull_contact.py`）进行船体-颗粒接触检测，无法表达真实三维船体曲面的几何特征。Mission 669 的目标是在 TensorDEM 中实现 **三角网格曲面接触检测**，支持 STL/OBJ 格式的船体网格输入，为冰区航行战役 DAG 节点（主计划 m611）提供 L1 基础层能力。

### 1.2 几何与参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 球体半径 R | 0.05 m | 测试用 DEM 颗粒 |
| 接触刚度 k | 10000 N/m | 线性惩罚弹簧 |
| 阻尼系数 c | 10 N·s/m | 粘性阻尼 |
| 空间哈希单元宽度 | 4 × max(R) | 宽相裁剪网格 |
| 网格格式 | STL (ASCII/binary), OBJ | 支持两种主流格式 |
| 测试球体数 N | 100 – 5000 | 复杂度标定范围 |
| 测试高度 h | 5 个值 (0.01–0.04 m) | 力精度验证 |

### 1.3 验收判据（SIR-PC2 / F04）

| 判据 | 要求 | 达成值 | 状态 |
|------|------|--------|------|
| `sphere_contact_force_error` | < 0.1 (ratio) | **0.0**（机器精度） | ✅ PASS |
| `contact_detection_complexity` | ≤ O(N) | **O(N) 确认** | ✅ PASS |
| `pytest_test_hull_mesh_contact` | PASS | **17/17 通过** | ✅ PASS |

---

## 2. 方法（算法 / 流程 / 关键设置 / 工具链）

### 2.1 算法架构

实现采用经典的两阶段碰撞检测流水线：

**宽相（Broad Phase）—— 均匀空间哈希**
- 网格单元宽度 = 4 × max(sphere_radius)
- 每个三角形按其 AABB 插入重叠单元
- 每球查询候选三角形：O(1) 均摊
- 总构建+查询复杂度：O(N_tri + N_spheres)
- 参考文献：Teschner et al. (2003), "Optimized Spatial Hashing for Collision Detection of Deformable Objects", VMV 2003

**窄相（Narrow Phase）—— Ericson 最近点算法**
- 向量化 Ericson 算法（Real-Time Collision Detection, §5.1.5）
- 处理全部 7 个 Voronoi 区域：3 顶点 + 3 边 + 1 内部
- 返回精确最近点与距离

**力模型**
- 线性惩罚：F = k · δ（δ = 穿透深度）
- 粘性阻尼：F_d = -c · v_n（v_n = 法向相对速度）
- 力仅排斥（分离时无吸引）
- 作用-反作用由构造保证

### 2.2 工具链

| 组件 | 版本/说明 |
|------|-----------|
| Python | 3.x |
| PyTorch | 张量运算后端 |
| pytest | 测试框架 |
| matplotlib | 图表生成（`scripts/gen_figures_m669.py`） |

### 2.3 代码规模

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/tensordem/hull_mesh_contact.py` | 447 | 核心模块（I/O + 空间哈希 + 窄相 + 力计算） |
| `tests/test_hull_mesh_contact.py` | 410 | 17 个 pytest 测试用例 |
| `scripts/gen_figures_m669.py` | 155 | 三张图表生成脚本 |

---

## 3. 计算结果（数字表，程序化提取）

### 3.1 力精度验证

球体在 z=0 平面上方高度 h 处，解析解为 F_analytical = k × (R - h)。

| 测试方法 | 对比基准 | 误差 |
|----------|----------|------|
| 5 个高度球体-平面接触 | 解析惩罚解 F=k(R-h) | **0.0**（机器精度） |

> 数据来源：`runs/m669_acceptance.json` → `acceptance_criteria.sphere_contact_force_error.achieved = 0.0`

### 3.2 复杂度标定

| 球体数 N | 方法 | 结果 |
|----------|------|------|
| 100 – 5000 | 时间比 N2/N1 与线性预期对比 | **O(N) 确认**（时间比 < 2× 预期比） |

> 数据来源：`runs/m669_acceptance.json` → `acceptance_criteria.contact_detection_complexity.achieved = "O(N) confirmed"`

### 3.3 测试套件

| 指标 | 值 |
|------|-----|
| 总测试数 | 17 |
| 通过 | 17 |
| 失败 | 0 |
| 耗时 | 11.65 s |

> 数据来源：`runs/m669_acceptance.json` → `test_results`

---

## 4. 结果图

原交付包含三张图表（由 `scripts/gen_figures_m669.py` 生成），已从 m669 提交（0f22cf9）提取：

### 图 1：接触力 vs 解析解（`fig1_force_error.png`）

![Force Error](artifacts/fig1_force_error.png)

*球体在 5 个不同高度处的接触力与解析惩罚解 F=k(R-h) 的对比。误差在机器精度范围内（~1e-15），验证了力计算的正确性。*

### 图 2：复杂度标定（`fig2_complexity.png`）

![Complexity](artifacts/fig2_complexity.png)

*N=100 至 N=5000 球体的检测耗时随球体数的变化。线性拟合确认 O(N) 复杂度，空间哈希的宽相裁剪有效避免了 O(N²) 暴力搜索。*

### 图 3：三维接触可视化（`fig3_3d_contact.png`）

![3D Contact](artifacts/fig3_3d_contact.png)

*球体与三角网格船体曲面的三维接触场景可视化。展示了空间哈希候选三角形（浅色）与最终接触三角形（深色）的筛选过程。*

---

## 5. 注意点（踩坑 / 局限 / 适用范围）

### 5.1 当前局限

1. **仅 CPU**：每球循环为顺序执行；GPU 加速需重构窄相
2. **无法向摩擦**：当前仅计算法向力；切向摩擦尚未实现
3. **单接触/球**：每球仅与最近三角形交互；角落多接触场景可能低估力
4. **刚体网格**：船体视为刚体；耦合 FEM-DEM 需额外基础设施

### 5.2 适用范围

- ✅ 凸/凹三角网格船体（STL/OBJ）
- ✅ 球体-网格接触（DEM 颗粒-船体）
- ✅ 大规模颗粒系统（O(N) 复杂度保证）
- ⚠️ 非球体颗粒（需扩展窄相）
- ⚠️ 大变形网格（需耦合网格更新）

### 5.3 证据等级

本模块证据等级为 **REAL**——最近点算法与力公式均为标准计算接触力学方法，精度经解析解验证。

---

## 6. 误差判定

| 关键量 | 基准 | 计算值 | 误差 | 是否 <3% |
|--------|------|--------|------|----------|
| 接触力 F | 解析解 k(R-h) | 与解析解一致 | **0.0**（机器精度） | ✅ **是**（远优于 3%） |
| 复杂度 | O(N) | O(N) 确认 | 定性符合 | ✅ **是** |
| 测试通过率 | 100% | 17/17 = 100% | 0% | ✅ **是** |

> **说明**：力误差为 0.0（机器精度级别），因为线性惩罚模型 F=kδ 在数值实现上与解析解完全一致（无迭代/近似）。这不是"拟合好"而是"实现正确"——算法直接计算 δ 并乘以 k，与解析公式等价。

### 无基准声明

- **摩擦系数**：未实现，无基准可比较
- **多接触场景**：未测试，无基准可比较
- **真实船体网格**：未用实际 STL 船体模型验证（仅用合成平面/曲面），缺乏实验基准

---

## 附录：工件清单

| 路径 | 说明 |
|------|------|
| `src/tensordem/hull_mesh_contact.py` | 核心模块（447 行） |
| `tests/test_hull_mesh_contact.py` | 测试套件（410 行，17 用例） |
| `scripts/gen_figures_m669.py` | 图表生成（155 行） |
| `docs/m669_hull_mesh_contact.md` | 原始技术文档 |
| `runs/m669_acceptance.json` | 验收数据 JSON |
| `artifacts/fig1_force_error.png` | 力精度对比图 |
| `artifacts/fig2_complexity.png` | 复杂度标定图 |
| `artifacts/fig3_3d_contact.png` | 三维接触可视化 |

---

*报告生成时间：2026-11-09（补报）*
*原始交付时间：2026-10-31*
*原始分支：agent/m669，commit 0f22cf9*
*合并 PR：#72*
