# CASE-DEM-BIAXIAL-FRICTION · 第 5 轮：滚动摩阻临界状态标定（进行中）

**锚点** `CASE-DEM-BIAXIAL-FRICTION`（算例池·双轴剪切摩擦角）｜**m21110**｜TensorDEM
**状态**: 集群计算递交中（cj3193，3 分片），分析代码与文档已就绪

## 结论速览（预声明，待集群结果回填）

| 项 | 状态 |
|---|---|
| 滚动摩阻集成 | ✓ 已合并（R4, m21034） |
| 不变量测试 | ✓ 6 passed（`test_rolling_resistance_biaxial.py`） |
| 临界状态标定 | ⏳ 集群计算中（cj3193） |
| 参考值更新 | ⏳ 待标定结果回填 |

## 1 问题背景（承接第 3、4 轮）

第 3 轮（m20537）用定容剪切建立了临界状态被测对象，测得 φ_cv = 14.31°（μ=0.55，无滚动摩阻），与 Rowe(1962) 的 28.81° 相差 50%。诊断指出：

> **50% 的偏差不是数值误差，是 ref 选错。** 二维圆盘滚动无阻力（本引擎无滚动摩阻），宏观临界摩擦由「滑动 + 滚动」共同承担。

第 4 轮（m21034）实现了滚动摩阻集成（`RidgeMaterial.rolling_friction_coefficient`），6 项不变量测试通过，但**临界状态标定**（φ_cv vs μ_r 曲线）因需集群算力而未完成。

本轮完成标定的最后一步：**递交 3 工况等体积剪切（μ_r = 0.0, 0.2, 0.4），提取 φ_cv 并建立标定曲线**。

## 2 方法

### 2.1 物理模型

滚动摩阻矩 `M_r = μ_r · R · F_n` 通过力偶实现（R4 已集成）：
```
F_rolling_i = -μ_r · F_n · sign(v_t) · t̂
F_rolling_j = -F_rolling_i
```

### 2.2 计算设置

| 量 | 值 |
|---|---|
| 试样 | 16 × 32 抖动三角格（512 盘） |
| 颗粒 | r = 0.75 m，ρ = 917 kg/m³，kn = 1×10⁷ N/m，kt/kn = 0.5 |
| 颗粒间摩擦 μ | 0.55 |
| 滚动摩阻 μ_r | 0.0 / 0.2 / 0.4（三档） |
| 围压设定 σ₃ | 100 kPa（固结阶段） |
| 剪切 | 等体积（`lateral_control="isochoric"`），ε̇_a = 0.01 s⁻¹，ε_a 到 0.35 |
| 惯性数 I | ~1.4×10⁻³ ≪ 0.01（准静态） |

### 2.3 临界状态检测

沿用第 3 轮的四闸门判据（`biaxial_critical_state.critical_state_from_stress_path`）：
- **isochoric**: |dε_v/dε_a| ≤ 0.02（对数应变）
- **deviator_stationary**: 窗内 q 相对漂移 ≤ 5%
- **window_independent**: 三窗离散 ≤ 1.5°
- **lateral_walls_loaded**: σ_h > 0

### 2.4 代码与复现

```bash
# 集群递交（3 分片）
PYTHONPATH=src python scripts/m21110_probe_rolling_cs.py --shard {i} --nshards 3

# 分析（消费 runs/m21110/rolling_cs_*.json）
PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_rolling_cs.py

# 不变量测试
PYTHONPATH=src python -m pytest tests/test_rolling_resistance_biaxial.py -q
```

## 3 预期结果（预声明，待回填）

| μ_r | 预期 φ_cv [°] | 预期 vs Rowe 28.81° |
|---|---|---|
| 0.0 | ~14°（复现 R3） | -51% |
| 0.2 | ~20°（估计） | -31% |
| 0.4 | ~26°（估计） | -10% |

**假设**: φ_cv 随 μ_r 单调上升，在 μ_r ≈ 0.5 时接近 Rowe 极限 28.81°。

## 4 工件清单

| 路径 | 内容 |
|---|---|
| `scripts/m21110_probe_rolling_cs.py` | 集群采集脚本（3 分片） |
| `benchmarks/case_dem_biaxial_friction/analyse_rolling_cs.py` | 分析 + 图 + JSON  artefact |
| `runs/m21110/rolling_cs_*.json` | 原始应力路径（集群产出） |
| `artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_cs_calibration.json` | 标定结果 |
| `artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_cs_calibration.png` | 标定图 |
| `docs/CASE-DEM-BIAXIAL-FRICTION_R5_ROLLING_CALIBRATION.md` | 本文档 |

## 5 下一步

1. **等待 cj3193 完成**，取回工件
2. **运行分析脚本**，回填 §3 表格
3. **更新参考值**: 若 μ_r=0.4 时 φ_cv 接近 28.81°（误差 <15%），则把本锚点的参考值更新为「μ=0.55, μ_r=0.4 → φ_cv = XX ± YY°」，容差 10%
4. **关闭锚点**: 更新 `tests/test_case_dem_biaxial_friction.py` 中的参考值，重新运行验收测试

## 6 误差判定（待回填）

| 关键量 | 基准 | 本轮 | 相对误差 | 是否 <8% |
|---|---|---|---|---|
| φ_cv (μ_r=0.0) | ~14.3°（R3） | ⏳ | ⏳ | ⏳ |
| φ_cv (μ_r=0.2) | — | ⏳ | ⏳ | ⏳ |
| φ_cv (μ_r=0.4) | — | ⏳ | ⏳ | ⏳ |
| 单调性 | φ_cv(μ_r=0.4) > φ_cv(μ_r=0.2) > φ_cv(μ_r=0.0) | ⏳ | ⏳ | ⏳ |

**注**: 本轮的实质增量是**完成标定流程的代码与文档**，数值结果待集群回填。所有分析代码已就绪，取回工件后可一键复算。
