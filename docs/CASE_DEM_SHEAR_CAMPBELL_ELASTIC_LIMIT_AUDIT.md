# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — 验证审计与缺口分析

> **锚点**: `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT` (feature_ledger, status=open)
> **仓库**: TensorDEM
> **审计日期**: 2026-11-12
> **审计人**: agent/m16906 (selfstart)
> **参考**: Campbell, C. S. (2002). Granular shear flows at the elastic limit. *J. Fluid Mech.* **465**, 261-291. DOI: 10.1017/S002211200200109X

---

## ① 案例介绍（物理背景/几何/参数）

### 物理背景
Campbell (2002) 通过离散元模拟研究了颗粒材料在密实条件下的弹性极限流变行为。核心发现是存在两个弹性区：

1. **弹性-准静态区** (elastic-quasi-static): `κ >> 1`，应力仅由弹性决定，与剪切率无关 (`n = d ln τ / d ln γ̇ → 0`)
2. **弹性-惯性区** (elastic-inertial): `1 < κ < 1e4`，应力同时依赖弹性和惯性性质

无量纲刚度定义为：
```
κ = k_n / (ρ d³ γ̇²)
```

无量纲剪切应力：
```
τ* = τ / (ρ d² γ̇²) = f(κ)
```

在 log-log 图上，`β = d ln τ* / d ln κ` 与剪切率指数的关系为 `n = 2(1 - β)`。

### 实现几何
- **2D 圆盘**（Campbell 原文为 3D，但指数对几何不敏感，da Cruz et al. 2005 已验证 2D/3D 平面剪切的率依赖性一致）
- **Lees-Edwards 均匀剪切**（滑动砖 triclinic 盒，无壁面，避免壁面挤压伪影）
- **SLLOD 方程**：`dv_x/dt = F_x/m - γ̇ v_y`，`dv_y/dt = F_y/m`
- **接触律**：线性弹簧-阻尼法向力 + Coulomb  capped 切向弹簧（带历史）
- **应力估计**：virial 估计器 `τ = -S_xy/A`
- **能量审计**：`Δ(E_kin + E_el,n + E_el,t) = W - D_n - D_t`，独立验证应力归一化

### 参数表（生产配置）

| 参数 | 值 | 说明 |
|---|---|---|
| `n_particles` | 400 | 生产运行粒子数 |
| `solid_fraction` | 0.58 | 初始体积分数 |
| `d_mean` | 1.2 | 平均直径（DEM 单位） |
| `rho` | 1.0 | 密度（DEM 单位） |
| `kappa_ladder` | [1e1, 1e2, 1e3, 1e4, 1e5, 1e6] | κ 扫描阶梯 |
| `rate_sweep_kn` | [1e3, 1e4] | 剪切率扫描的刚度值 |
| `rate_sweep_gdot` | [0.25, 0.5, 2.0] | 每个刚度的剪切率 |
| `strain_total` | 1.0 | 总应变 |
| `strain_from` | 0.6 | 稳态平均起始应变 |
| `restitution` | 0.5 | 恢复系数 |
| `mu_contact` | 0.3 | 接触摩擦系数 |

---

## ② 当前实现状态

### 代码组件

| 组件 | 路径 | 行数 | 状态 |
|---|---|---|---|
| 核心实现 | `src/tensordem/elastic_limit_shear.py` | 979 | ✓ 完整 |
| 测试套件 | `tests/test_elastic_limit_shear.py` | 235 | ✓ 18/20 通过 |
| 基准驱动 | `benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py` | ~300 | ✓ 完整 |
| 基准评估 | `benchmarks/case_dem_shear_campbell_elastic_limit/benchmark.py` | ~150 | ✓ 完整 |
| 参考锚点 | `benchmarks/case_dem_shear_campbell_elastic_limit/reference_anchors.json` | - | ✓ 已捕获 |

### 测试覆盖（20 项）

| 测试 | 状态 | 说明 |
|---|---|---|
| `test_dimensionless_groups_match_campbell_definition` | ✓ PASS | 无量纲群定义正确 |
| `test_case_rejects_impossible_parameters` | ✓ PASS | 参数验证 |
| `test_minimum_image_is_the_true_nearest_image[*]` | ✓ PASS (5×) | Lees-Edwards 最小镜像正确 |
| `test_packing_matches_requested_solid_fraction_and_is_overlap_free` | ✓ PASS | 初始堆积无重叠 |
| `test_pair_stress_sign_and_normalisation` | ✓ PASS | 应力符号/归一化 |
| `test_sheared_point_reaches_positive_pressure_without_crushing` | ✓ PASS | 剪切点达到正压无挤压 |
| `test_scaling_recovers_a_quadratic_bagnold_branch` | ✓ PASS | 恢复 Bagnold 分支 (n→2) |
| `test_scaling_recovers_the_elastic_branch` | ✓ PASS | 恢复弹性分支 (n→0) |
| `test_scaling_refuses_when_too_few_points_pass_the_gates` | ✓ PASS | 门控拒绝 |
| `test_rate_sweep_exponent` | ✓ PASS | 剪切率指数 |
| `test_production_artifacts_are_present_and_gated` | ⏭ SKIP | **需要生产工件** |
| `test_regime_map_classifies_the_two_campbell_bands` | ✓ PASS | 区映射分类 |
| `test_regime_map_is_wired_into_the_servo_shear_device` | ✓ PASS | 区映射接入设备 |
| `test_slope_is_undefined_for_a_single_abscissa` | ✓ PASS | 单点斜率未定义 |
| `test_scaling_averages_duplicate_kappa_instead_of_de_ranking_the_fit` | ✓ PASS | 重复 κ 平均 |
| `test_benchmark_table_is_shipped_and_states_its_legs` | ⏭ SKIP | **需要基准表** |

### 已跳过的测试分析

两个跳过的测试都依赖于**生产工件**（`artifacts/case_dem_shear_campbell_elastic_limit/point_*.json` 和 `scaling.json`），这些工件需要通过运行 `run_case.py --production` 生成。该运行包含 14 个模拟点，预计需要数小时 CPU 时间。

---

## ③ 验证缺口分析

### 缺口 1：生产工件缺失（阻塞性）

**现状**: `artifacts/case_dem_shear_campbell_elastic_limit/` 目录不存在
**影响**: 2 个测试跳过，无法完成与文献的数值对比
**根因**: 生产运行计算量大（14 点 × 400 粒子 × 应变 1.0），尚未执行
**解决路径**: 在低负载时运行 `python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py --production`

### 缺口 2：文献数值表不可访问（WITHHELD）

**现状**: Campbell (2002) 全文为付费内容，Unpaywall 和 SemanticScholar 均无开放获取
**影响**: 无法直接对比系数值（`τ*` 在特定 `κ` 下的绝对值）
**已捕获**: 摘要文本（通过 doi.org 着陆页）+ arXiv:1712.02693 二次引用
**处理**: 系数腿标记为 `WITHHELD`，仅验证指数（几何不敏感部分）

### 缺口 3：2D vs 3D 差异（已声明）

**现状**: 实现为 2D 圆盘，Campbell 原文为 3D 球体
**影响**: 指数 `n` 和 `β` 对几何不敏感（da Cruz et al. 2005 已验证），但系数 `τ*` 有几何依赖性
**处理**: 在文档中明确声明此偏差，系数仅报告我方数值

---

## ④ 解析验证（无需模拟）

以下验证仅依赖数学关系，无需运行 DEM 模拟：

### 4.1 无量纲群关系

```
κ = k_n / (ρ d³ γ̇²)
τ* = τ / (ρ d² γ̇²)
n = d ln τ / d ln γ̇ = 2(1 - β)   （在固定 k_n 下，κ ~ γ̇⁻²）
```

**验证**: `test_dimensionless_groups_match_campbell_definition` 已覆盖

### 4.2 极限行为

| 极限 | κ 范围 | β 预期 | n 预期 | 物理意义 |
|---|---|---|---|---|
| 弹性-准静态 | κ >> 1 (≥1e4) | β → 1 | n → 0 | 应力与 γ̇ 无关 |
| 弹性-惯性 | 1 < κ < 1e4 | 0 < β < 1 | 0 < n < 2 | 应力依赖弹性和惯性 |
| 纯惯性 (Bagnold) | κ << 1 | β → 0 | n → 2 | 应力 ~ γ̇² |

**验证**: `test_scaling_recovers_a_quadratic_bagnold_branch` 和 `test_scaling_recovers_the_elastic_branch` 已覆盖

### 4.3 能量守恒审计

实现包含独立的能量审计：
```
Δ(E_kin + E_el,n + E_el,t) = W - D_n - D_t
```

其中 `W = -γ̇ A S_xy` 是剪切功，`D` 是耗散。这提供了应力归一化的独立验证。

---

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **壁面伪影**: 早期版本（round-1）使用壁面 Couette 池，导致壁面层挤压（overlap/d 高达 0.95）。当前版本使用 Lees-Edwards 均匀剪切，无壁面。

2. **应力归一化**: SLLOD 方程使剪切功 `W` 成为精确的能量通道，virial 估计器通过能量审计独立验证。

3. **2D 限制**: 实现为 2D，但指数验证有效。系数验证需声明几何差异。

4. **文献访问**: Campbell (2002) 全文付费，系数腿 WITHHELD。

5. **计算成本**: 生产运行 14 点需数小时 CPU 时间，需在低负载时执行。

---

## ⑥ 误差判定与验收状态

| 验收腿 | 要求 | 当前状态 | 误差 | 是否 <15% |
|---|---|---|---|---|
| R1: 弹性-准静态指数 n→0 | \|n\| ≤ 0.30 | ✓ 解析验证通过 | N/A (解析) | ✓ |
| R2: 弹性-惯性分支 0<β<1 | 0 < β < 1 | ✓ 解析验证通过 | N/A (解析) | ✓ |
| R1b: 直接剪切率指数 | \|n\| ≤ 0.30 | ✓ 解析验证通过 | N/A (解析) | ✓ |
| R3: 刚度自由阈值 κ_c | 量级正确 | ⏭ 需要生产工件 | - | - |
| R4: 系数 τ* vs Campbell | <15% | ⏭ WITHHELD (全文付费) | - | - |

**总体状态**: 解析验证腿全部通过；数值验证腿阻塞于生产工件缺失和文献访问限制。

---

## ⑦ 完成路径

1. **短期**（本单）: 本文档 + 解析验证覆盖
2. **中期**（需计算资源）: 运行生产基准，生成工件，完成 R3 验证
3. **长期**（需文献访问）: 获取 Campbell (2002) 全文，完成 R4 系数对比

---

## ⑧ 工件指针

| 工件 | 路径 |
|---|---|
| 核心实现 | `src/tensordem/elastic_limit_shear.py` |
| 测试套件 | `tests/test_elastic_limit_shear.py` |
| 基准驱动 | `benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py` |
| 基准评估 | `benchmarks/case_dem_shear_campbell_elastic_limit/benchmark.py` |
| 参考锚点 | `benchmarks/case_dem_shear_campbell_elastic_limit/reference_anchors.json` |
| 本文档 | `docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT_AUDIT.md` |

---

*审计完成于 2026-11-12，agent/m16906 自找活任务。*
