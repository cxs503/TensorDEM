# TensorDEM 上游差集巡更报告 (m15706)

**日期**: 2026-11-12  
**仓库**: TensorDEM  
**本地 HEAD**: `3cfff476` (origin/main)  
**上游基准**: `c60dd3e1` (github/main, PR #324)  
**分支**: agent/m15706  

---

## 1. 差集概览

| 指标 | 数值 |
|---|---|
| 提交总数（ahead） | 1,181 |
| 非合并提交 | 971 |
| 真合并提交（多父） | 210 |
| 独立任务(mission)数 | 424 |
| PR 编号范围 | #325 – #1300（757 个唯一 PR） |
| 文件变更数 | 2,753 |
| 新增代码行 | 964,038 |
| 删除代码行 | 3,637 |
| 净增行数 | ~960,401 |

**结论**: `origin/main` 大幅领先 `github/main`（约 1,181 提交、424 个独立任务），两者自 PR #324 后分叉。`github/main` 停留在 2025 年中的状态，而 `origin/main` 已推进到 PR #1301。

---

## 2. 变更主题分类（按 src/ 子目录演化）

### 2.1 核心求解器与接触力学（高风险·持续演进）

| 子目录 | 变更文件数 | 主题 | 风险标注 |
|---|---|---|---|
| `tensordem/dem3d.py` | 11 | 3D DEM 核心引擎迭代 | 🔴 核心路径，任何回归影响全局 |
| `tensordem/granular2d/engine.py` | 6 | 2D 颗粒引擎 | 🟡 稳定但持续优化 |
| `tensordem/granular3d_core.py` | 4 | 3D 颗粒核心 | 🟡 与 dem3d.py 有重叠风险 |
| `tensordem/granular_column/` | 9 | 颗粒柱坍塌基准 | 🟢 新增验证面 |
| `tensordem/biaxial_shear_band.py` | 1 | 双轴剪切带 | 🟢 新能力 |
| `tensordem/shear_cell.py` | 4 | 剪切池 | 🟡 持续改进 |

### 2.2 冰区物理域（中高风险·多模块并行）

| 子目录 | 变更文件数 | 主题 | 风险标注 |
|---|---|---|---|
| `tensordem/ice_pressure_band/` | 8 | 冰压带统计与标定 | 🟡 新模块，需验证 |
| `tensordem/ice_thermal_stress/` | 10 | 冰热应力耦合 | 🟡 多物理场耦合 |
| `tensordem/ice_ridge_keel_porosity/` | 10 | 冰脊龙骨孔隙率 | 🟡 新参数化 |
| `tensordem/ice_friction_thermo/` | 8 | 冰摩擦热力学 | 🟡 新模块 |
| `tensordem/ice_contact/` | 8 | 冰接触律 | 🔴 核心接触律变更 |
| `tensordem/ice_evolution/` | 9 | 冰演化 | 🟡 新模块 |
| `tensordem/ice_twin/` | 8 | 冰孪生 | 🟡 新模块 |
| `tensordem/broken_ice_resistance/` | 9 | 碎冰阻力 | 🟡 关键应用 |
| `tensordem/mixed_ice/` | 10 | 混合冰 | 🟡 新模块 |
| `tensordem/hull_collision/` | 8 | 船体碰撞 | 🟡 应用层 |
| `tensordem/freeze_bond/` | 13 | 冻结黏结 | 🔴 核心物理 |
| `tensordem/freeze_bond.py` | 3 | 冻结黏结（顶层） | 🔴 核心物理 |
| `tensordem/ice_ridge_keel_freezebond.py` | 4 | 冰脊冻结黏结 | 🟡 耦合模块 |
| `tensordem/adfreeze_bond.py` | - | 附着冻结 | 🟡 新模块 |

### 2.3 颗粒形状与表征（中风险）

| 子目录 | 变更文件数 | 主题 | 风险标注 |
|---|---|---|---|
| `tensordem/shape_representation/` | 10 | 形状表征 | 🟡 新模块 |
| `tensordem/superquadric.py` | 3 | 超二次曲面 | 🟡 新粒子类型 |
| `tensordem/polyhedron_floe3d/` | 8 | 多面体浮冰3D | 🟡 新模块 |
| `tensordem/particle/clump_dem.py` | 3 | 团簇DEM | 🟡 新粒子类型 |

### 2.4 标定与验证（低风险·支撑性）

| 子目录 | 变更文件数 | 主题 | 风险标注 |
|---|---|---|---|
| `tensordem/dem_calibration/` | 10 | DEM 自动标定 | 🟢 工具链 |
| `tensordem/auto_calibration.py` | - | 自动标定 | 🟢 工具链 |
| `tensordem/sensitivity/` | 9 | 敏感性分析 | 🟢 工具链 |
| `tensordem/mu_i_rheology.py` | 3 | μ(I) 流变 | 🟢 验证面 |

### 2.5 新物理模块（中风险·需验收）

| 子目录 | 变更文件数 | 主题 | 风险标注 |
|---|---|---|---|
| `tensordem/brash_ice_rheology/` | 11 | 碎冰流变学 | 🟡 新模块 |
| `tensordem/erosion_dem/` | 9 | DEM 冲蚀 | 🟡 新模块 |
| `tensordem/erosion_pbm/` | 9 | PBM 冲蚀 | 🟡 新模块 |
| `tensordem/segregation/` | 8 | 偏析分析 | 🟢 验证中 |
| `tensordem/seepage/` | 9 | 渗流 | 🟡 新模块 |
| `tensordem/fsd_evolution/` | 9 | FSD 演化 | 🟡 新模块 |
| `tensordem/floe_population/` | 8 | 浮冰群 | 🟡 新模块 |
| `tensordem/rotary_dryer/` | 10 | 旋转干燥器 | 🟢 基准案例 |
| `tensordem/wet_ice_cohesion/` | 8 | 湿冰黏聚力 | 🟡 新模块 |
| `tensordem/whipping/` | 11 | 鞭击响应 | 🟡 新模块 |
| `tensordem/shm/` | 8 | 结构健康监测 | 🟡 新模块 |
| `tensordem/acoustics/` | - | 声学 | 🟡 新模块 |
| `tensordem/ae/` | - | 声发射 | 🟡 新模块 |
| `tensordem/bedcoupling/` | - | 床层耦合 | 🟡 新模块 |
| `tensordem/interface_operators/` | 9 | 界面算子 | 🟡 新模块 |
| `tensordem/friction_surface/` | 9 | 摩擦面 | 🟡 新模块 |

### 2.6 非任务提交（基础设施）

| 提交 | 主题 | 风险标注 |
|---|---|---|
| `f6967247` | docs: document repository license and asset scope | 🟢 文档 |
| `77bc7a93` | chore(packaging): declare Apache-2.0 license metadata | 🟢 元数据 |
| `91a50c39` | docs(license): clarify repository license scope | 🟢 文档 |
| `f163da47` | chore(license): add Apache-2.0 license | 🟢 许可证 |
| `a91163b3` | ci: self-hosted ci-loong64 同池运行 | 🟡 CI 基础设施 |

---

## 3. 风险标注汇总

### 🔴 高风险（核心路径变更）
- **`dem3d.py`** (11 次变更): 3D DEM 核心引擎，任何回归影响所有下游模块
- **`freeze_bond/`** (13 次变更): 冻结黏结物理，冰区航行的核心机制
- **`ice_contact/`** (8 次变更): 冰接触律，直接影响力学精度

### 🟡 中风险（新模块/多物理场耦合）
- 冰区物理域 12+ 个子目录并行开发，模块间接口需验证
- 颗粒形状表征（超二次曲面、多面体、团簇）引入新接触检测逻辑
- 热-力耦合（ice_thermal_stress, ice_friction_thermo）增加数值稳定性风险
- 冲蚀模块（erosion_dem, erosion_pbm）双实现需一致性验证

### 🟢 低风险（工具链/验证面/文档）
- 标定工具链（dem_calibration, auto_calibration, sensitivity）
- 基准案例（rotary_dryer, granular_column, biaxial_shear_band）
- 许可证与 CI 基础设施变更

---

## 4. 最近 15 个非合并提交主题演化

| 序号 | 提交 | 任务ID | 主题 | 文件过滤 |
|---|---|---|---|---|
| 1 | `905272cc` | m15486 | 成熟任务交付物 | 15 文件被门禁滤除 |
| 2 | `c2693034` | m15418 | 成熟任务交付物 | Beverloo 基准验证 |
| 3 | `b94a5d87` | m15397 | 成熟任务交付物 | 6 文件被门禁滤除 |
| 4 | `cc5ee10b` | m15367 | 成熟任务交付物 | 球体穿透阻力基准 |
| 5 | `f5ab3eba` | m15324 | 成熟任务交付物 | 14 文件被门禁滤除 |
| 6 | `08e840e5` | m15321 | 成熟任务交付物 | 8 文件被门禁滤除 |
| 7 | `720dacd5` | m15315 | 成熟任务交付物 | 双轴剪切带估计器偏差 |
| 8 | `397e38af` | m15279 | 成熟任务交付物 | 9 文件被门禁滤除 |
| 9 | `cb570b8b` | m15265 | 成熟任务交付物 | 11 文件被门禁滤除 |
| 10 | `2465ecd0` | m15259 | 成熟任务交付物 | 2 文件被门禁滤除 |
| 11 | `af16e0d2` | m15235 | 成熟任务交付物 | 11 文件被门禁滤除 |
| 12 | `1c48100d` | m15192 | 成熟任务交付物 | 11 文件被门禁滤除 |
| 13 | `002159ba` | m15189 | 成熟任务交付物 | 18 文件被门禁滤除 |
| 14 | `d9ca9000` | m15166 | 成熟任务交付物 | 83 文件被门禁滤除 |
| 15 | `e59b51c2` | m15187 | 成熟任务交付物 | Gap-DEM-Clump 审计 |

**模式观察**: 最近提交全部为"成熟任务交付物"模式，通过 PR 合并到 main。门禁系统持续过滤文件（平均 ~12 文件/任务被滤除），说明平台有严格的领域语义隔离机制。

---

## 5. 测试覆盖演化

| 测试文件 | 变更次数 | 覆盖领域 |
|---|---|---|
| `test_tutorial_first_verified_run.py` | 3 | 教程验证 |
| `test_segregation.py` | 3 | 偏析 |
| `test_ice_ridge_keel_freezebond.py` | 3 | 冰脊冻结黏结 |
| `test_dem2d_energy_fracture_ledger_negative.py` | 3 | 能量账本负控 |
| `test_coarse_graining_scaling.py` | 3 | 粗粒化缩放 |

测试文件共 406 个变更，表明测试覆盖与代码开发同步推进。

---

## 6. 基准测试演化

| 基准 | 变更次数 | 状态 |
|---|---|---|
| `unsteady_drag_validation.py` | 2 | 非定常阻力验证 |
| `superquadric_repose/run_benchmark.py` | 2 | 超二次曲面休止角 |
| `mts_contact_integration.py` | 2 | 多时间步接触积分 |
| `ice_thermal_stress/run_validation.py` | 2 | 冰热应力验证 |
| `h1_broken_ice_dem/run_validation.py` | 2 | H1 碎冰 DEM 验证 |
| `case_pair_collision_hertz/` | 4 | Hertz 接触对验证 |
| `verified/lbm_dem_drag_calibration/` | 5 | LBM-DEM 阻力标定（已验证） |

---

## 7. 结论与建议

### 7.1 上游同步状态
- `github/main` 严重落后（~1,181 提交），建议评估是否需要将 `github/main` 推进到与 `origin/main` 同步
- 分叉点为 PR #324 (`c60dd3e1`)，此后所有开发均在 `origin/main` 上进行

### 7.2 代码健康度
- **正面**: 测试覆盖与开发同步（406 测试变更），基准测试体系完善（245 基准变更）
- **关注**: 核心模块（dem3d.py, freeze_bond）高频变更，需关注回归风险
- **关注**: 12+ 冰区物理子目录并行开发，模块间接口一致性需验证

### 7.3 门禁系统
- 门禁持续过滤领域文件（平均 ~12 文件/任务），说明平台领域隔离机制运行正常
- 被过滤文件归入领域仓（ship-performance-platform），符合生态架构原则

### 7.4 建议行动项
1. **高优先级**: 对 `dem3d.py` 和 `freeze_bond/` 运行完整回归测试
2. **中优先级**: 验证冰区物理模块间接口一致性（ice_contact ↔ freeze_bond ↔ ice_thermal_stress）
3. **低优先级**: 评估 `github/main` 同步策略

---

*报告生成: 2026-11-12 | 工具: git diff/log 分析 | 方法: 提交主题分类 + 文件变更统计 + 风险标注*
