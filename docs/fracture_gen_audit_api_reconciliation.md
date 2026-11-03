# 验证欠账清偿：SI-2/DEM 断裂场生成器三套审计口径的冲突归一（m2633）

## 1. 问题：一个被"收集错误"掩盖的红套件

`tensordem.fracture_gen`（FEM 失效面 → DEM 冰场）在仓库里先后被三次独立审计，每次
只合入了**验证侧**（文档 + 测试 + 基准脚本），`src` 侧的配套 API 被门禁滤除或后续
重构覆盖：

| 代 | 测试文件 | 文档 | 期望的 src API | HEAD 状态 |
|---|---|---|---|---|
| m2185 | `tests/test_fracture_gen_mass_footprint.py` | `docs/fracture_gen_mass_footprint_audit.md` | `MASS_RADIUS_FACTOR`、`face_footprint_radius`、`mass_mapping=(equal_volume_sphere\|lattice_cell)`、`footprint_aware`、`on_plane_tolerance`、`FractureLedger.mass_accounting(V)` | **ImportError（集合期）** |
| m2287 | `tests/test_fracture_gen_audit.py` | `docs/dem3d_fracture_gen_audit.md` | `mass_mapping=(equal_volume_spheres\|cell_volume)`、`face_extent=(plane\|disc)`、`mass_accounting()` 无参、`volume_fraction`、`energy_accounting()` 的可判伪键 | **14 项 FAILED** |
| m2477 | `tests/test_fracture_gen_mass_mapping.py` | `docs/dem3d_fracture_mass_mapping_audit.md` | `mass_mode=(cell\|sphere)` + `fracture_mass_mapping` 模块 | 通过（并入 HEAD） |

后果比"若干红测"更严重：`tests/test_fracture_gen_mass_footprint.py` 的 ImportError 发生在
**集合期**，`pytest tests` 整体被中断 —— 全套件不可运行，第三代的 14 项失败因此长期
不可见。本单把三套口径归一到**一个** `src` API（向后兼容、单一语义），并让三套审计
同时转绿。

## 2. 做法（最小改动、只动 `fracture_gen.py` 与其验证面）

1. **词表归一**：`MASS_MAPPING_TO_MODE = {equal_volume_sphere(s)→sphere, lattice_cell/cell_volume→cell}`，
   `VALID_FACE_EXTENTS = (plane, disc)`；显式 `mass_mode` 优先于 `mass_mapping`，显式
   `face_extent` 优先于布尔 `footprint_aware`。未知值一律 `ValueError`。
2. **补上缺的 API**：`MASS_RADIUS_FACTOR=(6/π)^(1/3)=1.240701`、`FractureField.mass_radius/
   volume_fraction/domain_volume/mass_mapping/face_extent`、`face_footprint_radius()`、
   `FractureLedger.mass_accounting(domain_volume=None)`、`.lattice_volume()`、
   `identify_bonds_crossing_face(..., footprint_aware=, on_plane_tolerance=, face_extent=)`。
3. **能量/键数判据可判伪**：`energy_accounting()` 增加 `analytic_bond_count=ΣA/s²`、
   `analytic_unit_bond_energy=G s²`、`unit_bond_energy_relative_error`、
   `bond_count_relative_error`、`input_consistency_error=Σ|released_energy−G·A|`
   —— 原来的 c2 判据是代数恒等式，这些键才让它可被证伪。
4. **不变量保持**：`FractureLedger.check_bonds_at_faces()` 用生成时同一套断裂选项重识别，
   "所有过面键都被断"判据在 disc/tolerance 模式下仍有意义。
5. **两处验证面的显式化**（唯一的测试改动，均加注释）：
   * 冻结映射（sphere）在本代已成"历史口径"，第三代默认已改正为 cell，因此
     `test_fracture_gen_mass_footprint.py` 的 2 个"frozen"用例改为**显式**请求
     `mass_mapping="equal_volume_sphere"`（数值不变，只是不再依赖默认值）；
   * `mass_accounting()["relative_error"]` 统一为**带符号**（DEM−连续体）/连续体，
     `test_fracture_gen_audit.py` 中的量化误差用例改用 `abs(acc["relative_error"])`，
     判据（量化误差幅值）不变。

## 3. 结果（数字由模块 API 程序化产出，`artifacts/fracture_gen_audit/reconciliation.json`）

| 量 | 基准 | 实测 | 误差 |
|---|---|---|---|
| DEM 质量，冻结 sphere 映射 | ρV = 7336.00 kg | 3841.121 kg | -47.640 % |
| DEM 质量，cell 映射（默认） | ρV = 7336.00 kg | 7336.000 kg | +0.0e+00 % |
| 过面断键数（面内全平面） | A/s² = 64 | 64 | 0 % |
| 层重合切面（严格符号检验） | 2n² = 128 | 0 | −100 %（默认，如实记 FAIL） |
| 层重合切面（on_plane_tolerance=1e-12） | 2n² = 128 | 128 | 0 % |
| 有限面 R=0.5 m，冻结无限平面 | πR²/s² ≈ 12.57（精确枚举 12） | 64 | +433 % |
| 有限面 R=0.5 m，footprint disc | 精确枚举 12 | 12 | 0 % |

三套审计执行结果（`pytest -q`，2026-11-03）：

* 修前：`tests/test_fracture_gen_mass_footprint.py` **集合期 ImportError（全套件不可运行）**；
  将集合错误绕开后 `tests/test_fracture_gen_audit.py` **14 failed**。
* 修后：`tests/test_fracture_gen_audit.py + test_fracture_gen_mass_footprint.py +
  test_fracture_gen_mass_mapping.py + test_fracture_gen.py` → **64 passed, 0 failed**（425 s）。

## 4. 误差判定（平台门：>15 % 不可接受；关键量 <3 %）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| cell 映射的 DEM 质量 | ρV（连续体） | 0（浮点精确） | **是** |
| 冻结 sphere 映射的 DEM 质量 | ρV | −47.64 % | **否**（如实记 FAIL，故默认已改为 cell） |
| 全平面过面断键数 | A/s² | 0 % | **是** |
| footprint disc 断键数（R ≤ 1.0 m） | πR²/s² | −4.5 %…+13.2 %；对精确格点枚举为 0 % | 是（对枚举精确；对解析近似 <15 % 但 >3 %） |
| footprint disc 断键数（R=0.25 m） | πR²/s² = 3.14 | 4（+27.3 % vs 解析，0 % vs 枚举） | 否（vs 解析）／是（vs 枚举）——小面时解析近似失效，见 §5 |
| 层重合切面断键数（默认严格检验） | 2n² | −100 % | **否**（默认如实 FAIL，`on_plane_tolerance` 可选修复） |

## 5. 注意点与适用范围

* **默认值不是最优值**：`face_extent="plane"`、`on_plane_tolerance=0` 与 sphere 语义仍是
  冻结契约的默认，保持向后兼容；要物理正确的定位必须显式 opt-in（`face_extent="disc"`、
  小 tolerance）。这与"冻结 prereg 决策不擅自改"的纪律一致。
* **πR²/s² 只在足迹不越出平面范围时成立**：R=1.25 m 在 2 m 截面上会裁掉角部，
  解析值(78.5)超过可用列数(64)，此时只有精确枚举可用（审计脚本两者都报）。
* **足迹形状是圆盘**：非圆（狭长）黏聚面用等面积圆盘替代，偏差 O(周长)；FEM 侧若输出
  狭长面需重新引入 `extent_radius` 之外的方向信息。
* **cell 映射意味着质量半径 ≠ 接触半径**（`r_mass = 1.2407 r`）：粒子不再"内接于胞元"，
  这是格点 DEM 的标准做法（O'Sullivan 2011 ch.3），但会与按接触半径算的转动惯量/浮力
  体积口径不同，下游引用时须显式声明。
* 断键判据仍是几何零水平集：曲面、亚粒子裂纹、真正节点分离都未建模；层重合切面两侧
  共享面上粒子，仍通过接触传力，切口动量记账是近似的。
* 本单只动 `src/tensordem/fracture_gen.py`、两个审计测试的两处语义显式化、一个基准脚本
  的冻结映射显式化，以及上述文档与工件；求解器、LBM 耦合与冻结 JSON 契约未触碰。

## 6. 复现

```bash
python3 benchmarks/fracture_gen_mass_footprint_audit.py   # artifacts/fracture_gen_audit/*
python3 benchmarks/fracture_gen_audit.py                  # docs/dem3d_fracture_gen_audit.md 数字
python3 -m pytest tests/test_fracture_gen_audit.py tests/test_fracture_gen_mass_footprint.py \
                 tests/test_fracture_gen_mass_mapping.py tests/test_fracture_gen.py -q
```
