# DEM3D FEM→DEM 断裂场质量/体积映射审计（验证欠账清偿）

## 1. 目的与范围

`tensordem.fracture_gen`（SI-2 / PR #70）把 TensorFEM 的黏聚破坏面转换为带预断键的
DEM 冰场，并在模块头部把质量映射冻结表述为 **"equal-volume spheres"**：

    m_p = ρ · 4/3 π r³,        M_DEM = N · m_p

本审计对这条已合并能力做**解析基准复核**（不新增物理模型，只做验证）：把生成器的
质量/体积映射与简单立方格子的闭式几何对照，量化偏差、给出可判定的门限、并把结论
落进代码（`mass_mode`）与测试。

**审计判据**：`|M_DEM / (ρ·V_domain) − 1| ≤ 1 %`（与 owner「几何/守恒量先验证」口径一致）。

## 2. 解析基准（推导）

粒子由 `generate_particle_lattice` 置于间距 `h = 2r` 的简单立方格点上；每个粒子在
Voronoi 意义下**占有**一个边长 `h` 的立方胞元，体积 `h³ = 8r³`。而内接球体积为
`4/3 π r³`，故

    V_sphere / V_cell = (4/3 π r³) / (2r)³ = π/6 = 0.523598775598299

即 **简单立方等球堆积率 π/6**（Scott & Kilgour 1969 经典堆积表；DEM 中"质量取所占
胞元体积"的标准做法见 Cundall & Strack 1979、O'Sullivan 2011 第 3 章）。
于是"内接球质量"相对所代表冰体积的质量缺口为

    1 − π/6 = 0.4764012244  (47.64 %)

等价地，体积一致的球半径应为 `r_eq = (3h³/4π)^(1/3) = 0.6203505 h`。

第二项是**格点量化**：粒子满足 `r + k h < W`，故每轴粒子数
`N = ceil((W − r)/h)`，格点覆盖 `N h` 而非 `W`，相对体积误差 `(N h)³/W³ − 1`。

## 3. 审计结果（网格无关性：量化项归零，只剩材料映射项）

冻结 SI-2 冒烟域 `[0,0.6]³ m`、ρ = 917 kg/m³；每轴 n 粒子取 `r = W/(2n)` 使 `W = 2nr`
为 `h` 的整数倍。数据来自 `artifacts/fracture-mass-mapping/fracture_mass_mapping_sweep.csv`。

| n/轴 | r [m] | N | cell 映射相对误差 | sphere 映射误差 [%] | 量化项 |
|---|---|---|---|---|---|
| 2 | 0.15000 | 8 | -1.435e-16 | -47.64 | -1.3e-16 |
| 3 | 0.10000 | 27 | -2.870e-16 | -47.64 | -1.3e-16 |
| 4 | 0.07500 | 64 | -1.435e-16 | -47.64 | -1.3e-16 |
| 5 | 0.06000 | 125 | -1.435e-16 | -47.64 | -1.3e-16 |
| 6 | 0.05000 | 216 | -2.870e-16 | -47.64 | -1.3e-16 |
| 8 | 0.03750 | 512 | -1.435e-16 | -47.64 | -1.3e-16 |
| 10 | 0.03000 | 1000 | -1.435e-16 | -47.64 | -1.3e-16 |

* 冻结参考算例（27 粒子、54 键）：cell 映射 `M_DEM = 198.072000 kg`，
  参考 `ρV = 198.072000 kg`，误差 `+2.220e-16`；
  legacy sphere 映射 `M_DEM = 103.710257 kg`，误差
  `-47.6401 %`，与 `π/6 − 1` 一致到 1e-15。
* 断键拓扑与质量映射**解耦**：两种映射下键数同为 54（解析值 3·3·2·3 = 54），
  说明本次修正只动质量，不动断裂几何。

量化项（`W ≠ 整数倍 h` 时）在 `W ∈ [0.60, 0.80] m`、`h = 0.2 m` 上扫描：
最有利 `0.700 m → -37.03 %`，
最不利 `0.705 m → +46.12 %`
（立方放大：体积误差 ≈ 3×线性误差）。

船舶尺度解析行（**不建格点**，纯几何）：10 m × 10 m × 0.8 m 平整冰盖、r = 0.1 m →
`N = 10000`（[50, 50, 4]），冰体积
`80.000 m³`；cell 映射误差 `+1.984e-16`，
sphere 映射误差 `-47.6401 %`，
质量比 `M_sphere/M_cell = 0.523599`（= π/6）。

![mass mapping audit](artifacts/fracture-mass-mapping/fracture_mass_mapping_audit.png)

*图 (a)* 材料映射项：legacy sphere 恒为 −47.64 %（与分辨率无关，因为是几何常数），
cell 映射恒为 0（在机器精度内）。*图 (b)* 量化项随域尺寸阶跃变化。

## 4. 代码变更（最小改动）

| 文件 | 变更 |
|---|---|
| `src/tensordem/fracture_mass_mapping.py` | 新增：解析映射层（π/6、`equal_volume_radius`、`lattice_axis_count`、`analytic_bond_count`、`mass_mapping_report`、`resolution_sweep`） |
| `src/tensordem/fracture_gen.py` | `FractureGenConfig.mass_mode`（默认 `"cell"`，与冻结表述 "equal-volume spheres" 的本意一致）；`FractureField.mass_mode/particle_cell_volume/represented_volume`；`FractureLedger.mass_mapping_report()`；`generate_particle_lattice` 增加域尺寸输入守卫 |
| `tests/test_fracture_gen_mass_mapping.py` | 新增 13 项解析判据测试 |
| `tests/test_fracture_gen.py`, `benchmarks/fracture_gen_ledger.py` | 原断言编码的是内接球质量（即缺陷本身），改为体积一致值 |
| `scripts/audit_fracture_mass_mapping.py` | 审计脚本：JSON + CSV + 双面板图 |
| `benchmarks/fracture_gen_ledger.py` | 附带修掉报告路径冲突：它原与 `ice_floe_physics.py` 同写 `benchmarks/validation_report.json`，会把三锚点浮冰物理报告覆盖掉；现写 `fracture_gen_ledger_report.json` |
| `docs/dem3d_fracture_mass_mapping_audit.md` | 本文档 |

**向后兼容**：`mass_mode="sphere"` 完整保留（复现审计前数字）；`to_icedem3d_initial_state()`
的键集合不变，IceDEM3D 接口未动；断裂拓扑（键表、`alive` 掩码）与本次改动无关。

## 5. 影响面（为什么这条欠账值得还）

质量映射是 **一阶偏差**，会同时污染以下已合并能力（未逐个重跑，属本次**未做**部分）：

* `buoyancy_wet` / 冰盖浮态：Archimedes 平衡吃水由 `m/(ρ_w A)` 决定，−47.6 % 质量意味着
  干舷虚高 2 倍，浮冰与船体接触高度系统性偏低；
* 船–冰相互作用能量账：冰场动能份额 ∝ 质量，`dem3d_energy` 的能量残差判定会被质量
  口径掩盖；
* 附加质量比 / 冲量账本（`dem3d` impulse–energy ledgers）中的惯性项；
* 与 TensorFEM 的质量守恒门（`FractureLedger.mass_conservation_error`）此前恒以
  内接球质量为"期望值"，属于自证，门本身无效。

## 6. 局限与未做

1. **FEM 侧质量未联调**：`mass_mapping_report(fem_mass=...)` 接口已备好，但本次没有
   TensorFEM 实测质量工件，船舶尺度只做到解析行；跨库联调应作为下一张单。
2. **未做船舶尺度 DEM 实算**：`build_bond_topology` 是 O(N²) 纯 Python 双循环
   （10 m 冰盖 10⁴ 粒子即 5×10⁷ 对），本次只做几何解析；管线化/邻域搜索属独立增量。
3. **球形粒子惯量**：体积一致质量用的仍是球惯量（2/5 m r²）；若后续改用 `r_eq` 作
   惯量半径，需另立契约。
4. **量化项未修**：`N = ceil((W−r)/h)` 的阶跃是离散化固有属性，本次只**报告**不改
   （改动会动冻结域填充规则），建议上游把域尺寸对齐 `h` 的整数倍或按实覆盖体积归一。

## 7. 误差判定（逐项、显式回答"是否 < 3%"）

| 关键量 | 基准 | 修正后误差 | < 3 % ? |
|---|---|---|---|
| 冻结算例 M_DEM | ρ·V_domain（解析） | 2.220e-14 % | **是** |
| 10×10×0.8 m 冰盖 M_DEM | ρ·V（解析） | 1.984e-14 % | **是** |
| 分辨率扫描 (n=2…10) | ρ·V（解析） | ≤ 2.87e-14 % | **是** |
| 粒子数 N | ceil((W−r)/h)（解析） | 0（逐用例相等） | **是** |
| 键数 | NxNy(Nz−1)+…（解析） | 0（逐用例相等） | **是** |
| legacy sphere 映射（修正前） | ρ·V（解析） | −47.640 % | **否**（缺陷本体） |
| 量化项（W 非整数倍 h） | (Nh)³/W³−1 | −37.0 % … +46.1 % | **否**（离散化固有，已报告未修） |

结论：**质量映射缺陷已修复并通过解析判据**（cell 模式三项均 < 3 %，实际 1e-16 量级）；
legacy 模式作为可复现的历史口径保留并显式标注为不满足门限；量化项为已知、已量化的
离散化误差，不属于本次修复范围。

## 8. 复现

```bash
python scripts/audit_fracture_mass_mapping.py --output artifacts/fracture-mass-mapping
PYTHONPATH=src python -m pytest tests/test_fracture_gen_mass_mapping.py tests/test_fracture_gen.py -q
```
