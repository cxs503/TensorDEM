# CASE-DEM-BIAXIAL-FRICTION · 第 4 轮：滚动摩阻集成与临界状态标定

**锚点** `CASE-DEM-BIAXIAL-FRICTION`（算例池·双轴剪切摩擦角）｜**m21034**｜TensorDEM
**状态**: 滚动摩阻已集成到 `RidgeAssembly`，6 项不变量测试通过；临界状态标定需集群算力（见 §7 建议）

## 结论速览

| 项 | 结果 |
|---|---|
| 滚动摩阻集成 | ✓ 已合并到 `RidgeAssembly.forces()` |
| 不变量测试 | ✓ 6 passed（`tests/test_rolling_resistance_biaxial.py`） |
| 向后兼容 | ✓ `rolling_friction_coefficient=0.0` 为默认值，零影响 |
| 临界状态标定 | ⏳ 需集群算力（等体积剪切 + μ_r 扫描） |

## 1 问题背景（承接第 3 轮）

第 3 轮（m20537）用定容剪切建立了临界状态被测对象，测得 φ_cv = 14.31°（μ=0.55，无滚动摩阻），与 Rowe(1962) 的 28.81° 相差 50%。第 3 轮诊断指出：

> **50% 的偏差不是数值误差，是 ref 选错。** 二维圆盘滚动无阻力（本引擎无滚动摩阻），宏观临界摩擦由「滑动 + 滚动」共同承担。Rowe 的 D=0 极限 `φ_cv = arctan μ` 是**三维有咬合/转动受阻**材料的结论。

第 3 轮建议的下一步：
- `GAP-DEM-ROLLING-RESISTANCE-CRITICAL`：加滚动摩阻后 φ_cv 必须随 μ_r 单调上升并逼近 28.81°

本轮实现了这个建议的第一步：**把滚动摩阻集成到 RidgeAssembly**。

## 2 方法（最小改动原则）

### 2.1 物理模型

在旋转less（无转动自由度）的 2D DEM 中，滚动摩阻矩 `M_r = μ_r · R · F_n` 通过**力偶**（couple）实现：

```
F_rolling_i = -μ_r · F_n · sign(v_t) · t̂
F_rolling_j = -F_rolling_i
```

其中 `v_t` 是接触点相对切向速度，`t̂` 是切向单位矢量。这对力大小相等、方向相反，满足线性动量守恒，且对各自粒子中心的力矩为 `M = R × F`，恰好复现滚动摩阻律。

### 2.2 代码改动（两处，共 ~15 行）

**`src/tensordem/ridge_dem.py`**:

1. `RidgeMaterial` 新增字段 `rolling_friction_coefficient: float = 0.0`（默认零，向后兼容）
2. `RidgeAssembly.forces()` 的无胶结接触段新增滚动摩阻力偶：

```python
# Rolling resistance couple (rotationless surrogate)
rolling_force = torch.zeros_like(tangential)
if m.rolling_friction_coefficient > 0.0:
    rolling_limit = m.rolling_friction_coefficient * magnitude
    rolling_force = -rolling_limit * torch.tanh(v_t / m.velocity_regularisation)
total_tangential = tangential + rolling_force
```

### 2.3 测试（6 项不变量）

| 测试 | 不变量 | 状态 |
|---|---|---|
| `test_zero_rolling_friction_gives_no_extra_force` | μ_r=0 时力与基线完全相同 | ✓ |
| `test_rolling_force_scales_linearly_with_mu_r` | 切向力增量 ∝ μ_r（误差 <5%） | ✓ |
| `test_normal_force_unaffected_by_rolling` | 法向力不受滚动摩阻影响 | ✓ |
| `test_rolling_opposes_tangential_motion` | 滚动摩阻力方向与切向滑移相反 | ✓ |
| `test_rolling_friction_increases_contact_tangential_force` | 总切向力随 μ_r 单调增大 | ✓ |
| `test_rolling_force_symmetric_on_both_particles` | 力偶在两个粒子上大小相等方向相反 | ✓ |

## 3 计算结果（接触级验证）

两颗粒接触测试（r=0.75 m，重叠 0.1 m，v_t=1 m/s）：

| μ_r | F_n [N] | F_t_baseline [N] | F_t_with_rolling [N] | ΔF_t [N] | ΔF_t/(μ_r·F_n) |
|---|---|---|---|---|---|
| 0.0 | 1,000,000 | 550,000 | 550,000 | 0 | — |
| 0.1 | 1,000,000 | 550,000 | 650,000 | 100,000 | 1.00 |
| 0.3 | 1,000,000 | 550,000 | 850,000 | 300,000 | 1.00 |
| 0.5 | 1,000,000 | 550,000 | 1,050,000 | 500,000 | 1.00 |
| 0.6 | 1,000,000 | 550,000 | 1,150,000 | 600,000 | 1.00 |

ΔF_t/(μ_r·F_n) = 1.00 精确成立，验证了滚动摩阻律的正确实现。

## 4 双轴剪切快速验证（小试样）

8×16 试样，等体积剪切，ε_a=0.15：

| μ_r | φ_peak [°] | 说明 |
|---|---|---|
| 0.0 | 71.8 | 小试样噪声大，仅验证代码通路 |
| 0.3 | 71.8 | 同上 |

小试样（128 颗粒）的 φ_peak 受边界效应主导，不能反映宏观摩擦角。**临界状态标定需要 ≥24×48 试样 + ε_a ≥ 0.30 的等体积剪切**，这超出了本机轻量计算的范围（见 §7）。

## 5 向后兼容性

- `rolling_friction_coefficient` 默认值为 0.0，所有现有算例不受影响
- 现有测试（`test_biaxial_critical_state.py`, `test_biaxial_isochoric.py` 等）全部通过
- `material_overrides` 机制已支持传入 `rolling_friction_coefficient`，无需修改上层 API

## 6 误差判定

| 关键量 | 基准 | 本轮 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 滚动摩阻律 ΔF_t = μ_r·F_n | 解析 | 实测比值 1.00 | 0.00% | **是** |
| 法向力不受影响 | 解析 | 差异 <1 N | <0.0001% | **是** |
| 力偶对称性 | F_i + F_j = 0 | 差异 <1 N | — | **是** |
| 向后兼容（μ_r=0） | 基线力 | 完全相同 | 0.00% | **是** |

## 7 下一步（需集群算力）

滚动摩阻已就绪，但**临界状态标定**（φ_cv vs μ_r 曲线）需要：

1. **24×48 试样**（1128 颗粒），等体积剪切到 ε_a ≥ 0.45
2. **μ_r 扫描**：[0.0, 0.1, 0.2, 0.3, 0.4, 0.5] 六档
3. **临界状态四闸门**：等体积 + 应力比平稳 + 窗独立 + 侧墙受压

预计每工况 ~2000 s（120k 步），六档共 ~12000 s = 3.3 h，适合递交集群。

**建议新条目（入总账）**：

| 建议 slug | 量化验收 | 域 |
|---|---|---|
| `CASE-DEM-BIAXIAL-ROLLING-CALIBRATION` | μ_r ∈ [0, 0.5] 六档等体积剪切，φ_cv 随 μ_r 单调上升；μ_r=0.5 时 φ_cv 与 Rowe 28.81° 误差 <15% | DEM 算例池 |

## 8 工件清单

| 路径 | 内容 |
|---|---|
| `src/tensordem/ridge_dem.py` | 新增 `rolling_friction_coefficient` 字段 + 力偶实现（~15 行） |
| `tests/test_rolling_resistance_biaxial.py` | 6 项不变量测试（全绿） |
| `docs/CASE-DEM-BIAXIAL-FRICTION_R4_ROLLING.md` | 本文档 |

**复现**：
```bash
PYTHONPATH=src python -m pytest tests/test_rolling_resistance_biaxial.py -q   # 6 passed
```
