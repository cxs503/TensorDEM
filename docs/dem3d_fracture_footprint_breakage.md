# FEM 破坏面 → DEM 断裂场：足印判据与质量映射（验证欠账清偿，m2935）

**状态**：`tensordem.fracture_gen` 的三项已发布判据（足印断裂 / 层重合平面 / 质量-半径闭式）
此前**只有文档与测试、没有可执行实现**（`tests/test_fracture_gen_mass_footprint.py` 在
`main` 上 collection error，全仓 `pytest tests/` 因此中断）。本单把缺失的实现补齐，使已合并
的审计契约可复现，并保持已合并的「诚实 FAIL」结论（π/6 − 1 = −47.64 %）不变。

## 1. 案例介绍（物理背景 / 几何 / 参数）

TensorFEM 给出黏聚破坏面（面积 A、法向 n、形心 c、释放能 E），`tensordem.fracture_gen`
把它转换成带**预断键**的 DEM 冰场，直接喂给 `IceDEM3D.step(external_weights=...)`。
海冰与船体/冰脊接触时，破坏面是一次局部事件；若把它的能量按「整个剖面」断键，
断裂能密度会被稀释、冰盖刚度下降，属于一阶偏差。

| 量 | 值 |
|---|---|
| 格点 | 简单立方，间距 h = 2r = 0.25 m |
| 粒子半径 r | 0.125 m（接触半径） |
| 域 | [0, 2]³ m（每轴 n = 8 层） |
| 粒子数 N | 512（解析 8³） |
| 最近邻键数 | 1344（解析 3n²(n−1)） |
| 冰密度 ρ | 917 kg/m³ |
| 连续体参考体积 | 8.000 m³（= (n h)³，无量化误差） |
| 连续体参考质量 | 7336.000 kg |
| 破坏面 | A = 4.0 m²、z = 0.5 m（层间）、R = √(A/π) = 1.128 m；足印扫描 R ∈ [0.25, 1.25] m |
| 平台判据 | 材料映射误差 ≤ 15 %；精度目标 ≤ 3 % |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **欠账定位**（只读）：`pytest --collect-only` → 1070 tests, 1 error（`test_fracture_gen_mass_footprint.py`，
   缺 `MASS_RADIUS_FACTOR`、`face_footprint_radius`、`extent_radius`、`footprint_aware`、
   `on_plane_tolerance`、`mass_mapping`、`mass_radius`、`FractureLedger.mass_accounting`）。
   文档 `docs/dem3d_fracture_mass_mapping_audit.md` 与 `benchmarks/fracture_gen_mass_footprint_audit.py`
   已合并，故这是「已发布但不可执行」的验证欠账，而非新功能提案。
2. **最小实现**（`src/tensordem/fracture_gen.py`，+~200 行，默认路径不变）：
   * `MASS_RADIUS_FACTOR = 2(3/4π)^{1/3} = 1.240701`、`MASS_MAPPING_ALIASES`、
     `normalise_mass_mapping()`；
   * `FailureFace.extent_radius`、`face_footprint_radius()`（显式半径优先，否则等面积圆盘 √(A/π)）；
   * `FractureGenConfig.mass_mapping / footprint_aware / on_plane_tolerance` 与守卫
     （负 tolerance、未知映射名 → `ValueError`）；
   * `identify_bonds_crossing_face(..., on_plane_tolerance, footprint_aware)`：
     严格符号判别 → 平面容差（恰一端在平面上则断，共面键不断）→ 足印圆盘限制
     （键中点的面内投影半径 ≤ R）；`generate_fracture_field`、`FractureField`、
     `FractureLedger.check_bonds_at_faces` 全链路透传同一判据；
   * `FractureField.mass_radius`（承载 `particle_mass` 的球半径）、
     `FractureLedger.mass_accounting(continuum_volume, tolerance=0.15)`。
3. **对标基准**（独立于实现推导）：π/6 简单立方堆积率；`r_eq = (3h³/4π)^{1/3}`；
   全平面断键数 `A/h²`；层重合平面 `2n²`；足印断键数 `πR²/h²` **且**与格点逐列枚举逐例对拍。
4. **工具链**：`python -m pytest`；`benchmarks/fracture_gen_mass_footprint_audit.py`
   （生成 `artifacts/fracture_gen_audit/{audit.json, audit_tables.md, *.png}`，本文件表格即该脚本
   `render_markdown()` 的原样输出，零手抄）；matplotlib 出图。

**契约冲突的处理（关键决定，非「改测试变绿」）**：`main` 的默认质量映射是审计后修正的
体积一致映射 `mass_mode="cell"`（`tests/test_fracture_gen.py` 断言默认 `M = ρNh³`），
而并行的验证欠账单把「冻结的球映射」当作默认来断言 π/6 − 1 的赤字。两者不能同时为真。
处理：默认保持 `cell`（物理正确、已合并测试与文档的依据），
`equal_volume_sphere`/`sphere` 作为**显式可选**的冻结契约；
断言 π/6 − 1 的两处测试改为显式请求冻结映射，**所断言的数字一字未改**（诚实 FAIL 保留），
并把「默认映射体积一致」作为新契约固定在新测试里。

## 3. 计算结果（程序化输出）

### 3.1 质量映射（连续体参考 ρV = 7336.000 kg）

| mapping | DEM 质量 (kg) | 连续体质量 (kg) | 相对误差 | <15 % ? |
|---|---|---|---|---|
| equal_volume_sphere (frozen) | 3841.120618 | 7336.000000 | −47.6401 % | **no** |
| lattice_cell (trunk default) | 7336.000000 | 7336.000000 | +0.000000 % | yes |

质量半径闭式：`r_m/r = 1.240701`；`4/3 π r_m³ ρ / m_p − 1 < 1e-15`（两种映射）。

### 3.2 全平面与层重合平面断键数（解析 A/h² = 64）

| 算例 | 实测断键 | 解析 | 相对误差 |
|---|---|---|---|
| 层间平面 z = 0.5，严格判据 | 64 | 64 | +0.00e+00 |
| 层重合平面 z = 0.625，严格判据 | 0 | 128 | −100.0 % |
| 层重合平面 z = 0.625，on_plane_tolerance = 1e-12 | 128 | 128 | +0.0 % |

### 3.3 有限足印（n = 8, h = 0.25 m）

| R (m) | A = πR² (m²) | 解析 πR²/h² | 逐列枚举 | 冻结（无限平面） | footprint_aware | aware 误差 | 过断倍数 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.1963 | 3.14 | 4 | 64 | 4 | +27.3 % | 16.00× |
| 0.50 | 0.7854 | 12.57 | 12 | 64 | 12 | −4.5 % | 5.33× |
| 0.75 | 1.7671 | 28.27 | 32 | 64 | 32 | +13.2 % | 2.00× |
| 1.00 | 3.1416 | 50.27 | 52 | 64 | 52 | +3.5 % | 1.23× |
| 1.25 | 4.9087 | 78.54 | 64 | 64 | 64 | −18.5 % | 1.00× |

判定（脚本 `verdicts`）：frozen 质量映射 **FAIL (>15 %)**；lattice_cell **PASS (<3 %)**；
全平面断键 **PASS**；冻结足印过断 **FAIL（最大 ~16×）**；层重合平面 **严格判据 FAIL /
加容差 PASS**。

## 4. 结果图

![footprint field map](artifacts/fracture_gen_audit/footprint_field_map.png)

*图 1* 断裂键空间分布（俯视，z = 0.5 m 平面，R = 0.50 m 时）：灰点为格点，红叉/蓝叉为断键中点
投影，黑虚线为破坏面足印圆盘。左：冻结判据把**整个 4 m² 剖面**全断（64 键，含圆盘外 52 键）；
右：footprint_aware 只断圆盘内 12 键，与逐列枚举完全一致。第三栏：断键数 vs 足印半径，
解析 πR²/h²（黑线）/逐列枚举（方块）/aware（圆点）/冻结（红叉，不随 R 变化）。

![mass and footprint audit](artifacts/fracture_gen_audit/fracture_gen_audit.png)

*图 2* 审计总图：(a) 断键数 vs 足印半径（同图 1 右栏）；(b) 质量映射相对连续体质量误差，
绿色带为 ±15 % 验收带，`π/6 − 1 = −47.6401 %` 虚线为冻结映射的解析赤字。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

* **判据用键中点投影，不是精确交点**。二者在轴向键上等价；对倾斜面的斜键差 O(h) 量级，
  因此 `footprint_aware` 的计数在与「精确枚举」对拍时仍有 0–1 键的边界差异，
  是本实现的有意简化（几何一致、可复现）。
* **圆盘-格点量化**：R ≲ 2h 时一列就是 ~1/h² 的台阶（R = 0.25 m 时 +27.3 %）；R 超过剖面半宽
  （1.25 m 时 πR²/h² = 78.5 而剖面只有 64 列）时解析上界大于可用列数，误差带符号翻转。
  这两行**不是**模型误差，而是「解析圆盘 vs 离散格点」的量化项；正确锚是逐列枚列（0 %）。
* **45° 平面陷阱**：过 (0.5, 0.5, 0.5)、法向 (1,0,1) 的平面在 n = 4 格点上**每根跨面键都恰好
  端点落在平面上**，严格判据给出 0 断键（与层重合平面同型退化）。新增测试因此把形心移到
  非对称位置 (0.5, 0.5, 0.5625) 才真正验证「面内投影」判据。
* **`on_plane_tolerance = 0` 保持冻结语义**（共面/端点落面不算断）；只有 > 0 才启用
  「恰一端在平面上」。若默认为正容差，`main` 上 z = 0.625 层重合算例会从 0 跳到 128 键，
  属行为变更，故不做。
* **适用范围**：足印半径由 FEM 侧提供 `extent_radius` 或按等面积圆盘推断；若破坏面强烈
  非圆（长条形裂纹），等面积圆盘是高估面积的保守近似，应显式给 `extent_radius`。

## 6. 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 质量半径 r_m（cell 映射） | (3h³/4π)^{1/3} = 0.155088 m | 0.155088 m | <1e-15 | **是** |
| DEM 质量（cell 映射） | ρV = 7336.000 kg | 7336.000 kg | 0.0 | **是** |
| DEM 质量（冻结球映射） | ρV | 3841.121 kg | −47.64 % | **否**（已知诚实 FAIL，保留为审计结论） |
| 全平面断键数 | A/h² = 64 | 64 | 0.0 | **是** |
| 层重合平面断键数（容差 1e-12） | 2n² = 128 | 128 | 0.0 | **是** |
| 层重合平面断键数（严格判据） | 2n² = 128 | 0 | −100 % | **否**（冻结判据缺陷，已由 `on_plane_tolerance` 修复，默认不启用） |
| 足印断键数 vs 逐列枚举（R = 0.25…1.25 m，5 例） | 枚举值 | 4/12/32/52/64 | **0.0** | **是** |
| 足印断键数 vs 解析 πR²/h²（R = 0.5, 0.75, 1.0 m） | πR²/h² | 12/32/52 | −4.5 % / +13.2 % / +3.5 % | **否**（量化项，见 §5；R = 0.75 m 的 +13.2 % 仍在平台 15 % 带内） |
| 足印断键数 vs 解析（R = 0.25 / 1.25 m） | πR²/h² | 4 / 64 | +27.3 % / −18.5 % | **否**（量化项 + 剖面截断，见 §5） |

结论：**实现量（质量半径、质量、全平面计数、容差后层重合计数）全部与闭式解一致到机器精度
（< 1e-12）**；足印计数与**精确离散枚举 0 % 误差**；相对**连续体解析圆盘**的 3–27 % 差异
是可预测的格点量化项，随分辨率 O(h/R) 收敛，不是模型偏差。冻结质量映射的 −47.64 % 与
层重合平面的 −100 % 是**该冻结契约自身的缺陷**，本单如实保留并给出显式修复开关。

## 7. 复现

```
python -m pip install -e ".[test]"
python -m pytest tests/test_fracture_gen_mass_footprint.py \
                 tests/test_fracture_gen_footprint_breakage.py \
                 tests/test_fracture_gen.py tests/test_fracture_gen_mass_mapping.py -q
python benchmarks/fracture_gen_mass_footprint_audit.py     # -> artifacts/fracture_gen_audit/
```

* `tests/test_fracture_gen_mass_footprint.py`（已合并，16 项）：冻结契约审计，2 处显式选择冻结映射。
* `tests/test_fracture_gen_footprint_breakage.py`（本单新增，17 项）：替代契约的独立锚
  （质量半径 ↔ 质量、别名等价、默认体积一致、容差作用域、足印为**限制**而非扩展、
  倾斜面面内投影、断裂能与断键判据解耦）。