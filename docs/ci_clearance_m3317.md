# TensorDEM main CI 清红（m3317）：19 条红项的逐条归因与处置

范围：`main`（f5de971）上 `python -m pytest -q` 的收集期/运行期红项。本增量不改物理模型，
只按「源改了测试未跟 → 按现 API 修测试」「源丢了参数 → 修源」「账本过期 → 重生成」三类归因处置，
并对每类给出可复算的前后证据。

## 1. 前置：scipy 依赖声明（PR#197）

`pyproject.toml` 的 `[project].dependencies` 缺 `scipy>=1.10`，而 `src/tensordem` 多个模块
`import scipy`，导致 CI 在**收集期**就有 6 个测试文件 `ModuleNotFoundError`。
本条按指令直接并入既有修复分支 `ci/declare-scipy-runtime-dep`（cb89a1d0，单行声明），未重做。
并入后收集期恢复：**1234 tests collected**。

## 2. 归因总表（19 条）

| # | 红项 | 现象 | 归属 | 处置 |
|---|---|---|---|---|
| 1–3 | `tests/test_dem3d_moving_wedge_bow.py` | `TypeError: MovingWedgeBowIceDEM3D.forces() got an unexpected keyword argument 'update_history'` | 源落后：`dem3d.py` 的 `forces()` 新增了 `update_history`，子类覆盖未跟 | 修源：`src/tensordem/dem3d_wedge.py` 覆盖签名加参数并透传 |
| 4–6 | `tests/test_dem3d_restitution_verification.py` | 同上（`update_history`） | 同上 | 同上（同一根因，由 1–3 的修复一并消除） |
| 7 | `tests/test_dem3d_restitution_verification.py::test_solver_reproduces_clamped_reference_model` 等 4 项 | `AssertionError`：`gamma_n=1.0` 时求解器 `e=1.0001`，参考值 `0.9148`（相对误差 ~9%…100%） | 源缺配置：`measure_dem3d_impact` 传 `contact_damping=gamma_n`，但 `DEM3DConfig.contact_damping_model` 默认 `"restitution"`，该分支**忽略** `contact_damping`，改用 `restitution_coefficient`（默认 1.0）重建 `gamma_n` | 修源：`src/tensordem/contact_restitution.py` 显式声明 `contact_damping_model="absolute"` |
| 8 | `tests/test_dem3d_wedge_resistance_sensitivity.py` | 同 1–3 的 `update_history` | 同 1–3 | 同 1–3 |
| 9–12 | `tests/test_doc_evidence_pointers.py`（4 项） | 账本过期 1≠0；2 个 `code:MISSING` 指针；3 个 ORPHAN；未提交产物缺 producer | 文档侧指针漂移 + 账本未重生成 | 修文档 + 补 producer + 重生成账本（见 §4） |
| 13–16 | `tests/test_fracture_gen_audit.py::MassMappingAudit` | `ValueError: mass_mapping/mass_mode must be one of [...], got 'cell_volume' / 'equal_volume_spheres'` | 源改了测试未跟：审计层（测试/基准/绘图脚本）用的拼写未进别名表 | 修源：`MASS_MAPPING_ALIASES` 补两个拼写（`cell_volume→cell`、`equal_volume_spheres→sphere`） |
| 17–19 | `tests/test_fracture_gen_audit.py::FaceExtentAudit` | `TypeError: FractureGenConfig.__init__() got an unexpected keyword argument 'face_extent'` | 源丢了参数：`face_extent`（`"plane"`/`"disc"`）被 `footprint_aware: bool` 取代，但测试与全部 producer 仍在用旧名 | 修源：恢复 `face_extent` 作为 `footprint_aware` 的显式别名并校验矛盾组合；`identify_bonds_crossing_face` 同步接受该别名 |

## 3. 恢复审计层时补齐的两处真实缺口

审计键不是等价改名可以绕过的，必须真有实现：

* `FractureField.volume_fraction`（= `particle_mass / (rho h^3)`：`cell` 映射恰为 1.0，
  `sphere` 映射恰为 π/6）——原本缺失，审计的 c1 无法区分两种映射。
* `FractureLedger.energy_accounting()` 的分析解键：`analytic_bond_count`（= `ΣA/h²`）、
  `analytic_unit_bond_energy`（= `G h²`）、`unit_bond_energy_relative_error`、
  `bond_count_relative_error`、`input_consistency_error`（= `Σ|released − G·A|`）、
  `face_extent`。缺少它们时 c2 只能拿 `unit_bond_energy` 乘回去，是代数恒等式（不可伪证）。
* `FractureLedger.mass_accounting()` 增加 `domain_volume` 别名、`num_particles`、
  `lattice_volume_covered`、`fem_continuum_mass`、`volume_fraction` 与
  `relative_error_magnitude`；`relative_error` 保持主干的有符号约定（π/6 − 1 = −0.4764，
  由 `tests/test_fracture_gen_mass_footprint.py` 断言），量值键供 15% 判据与量化审计使用。

同时 `generate_fracture_field` 之前的 `build_bond_topology` 是逐对 Python 循环（1000 粒子
= 22.5 s，主导了整个审计套件的耗时）；改为分块 `cdist` 后 **0.38 s**，键序保持不变
（`tests/test_fracture_gen_audit.py::test_vectorised_topology_matches_brute_force` 逐对验证）。

## 4. 文档证据链（4 条）

| 指针 | 处置 |
|---|---|
| `docs/dem3d_contact_impact_dashpot_convention.md` 引用了一个从未提交的临时探针脚本（`scripts` 下划线前缀的 scratch 文件） | 改为引用已提交的同构装置 `src/tensordem/contact_restitution.py` 的 `measure_dem3d_impact`，并新增 §7 记录该装置的同源默认值坑与修复证据 |
| `docs/wake_lbm_analytic_verification.md` 引用 `threefield` 子包路径时漏掉 `src/tensordem/` 前缀 | 改为真实路径 `src/tensordem/threefield/coupling.py` |
| `docs/dem3d_fracture_footprint_breakage.md` 引用 `artifacts/fracture_gen_audit/footprint_field_map.png` 无 producer（ORPHAN） | 在 `benchmarks/fracture_gen_mass_footprint_audit.py` 补 `plot_footprint_field_map()`，产出该图；R = 0.50 m 时冻结判据 64 键 / footprint_aware 12 键，与文档表述一致 |
| `docs/doc_evidence_ledger.json` 过期 | 重生成（`scripts/audit_doc_evidence.py`），在**无未跟踪产物**的树上生成，使账本与全新 clone 的扫描一致 |

## 5. 顺带消除的 1 条同因红项（不在 19 条清单内）

`tests/test_wake_lbm_analytic_verification.py::test_smagorinsky_closure_matches_independent_evaluation`：
脚本侧的手算 `|Π|` 用「六分量不加权」求和（在纯剪切下比真 Frobenius 范数小 √2），
而内核（`WakeLBM._effective_tau`，m2949）已改为「对角 1 / 非对角 2」的加权六分量求和，
并显式除以 `rho`。脚本未跟 → 改为与内核同一约定（含 `/rho`），误差从 1.8e-3 降到 **<1e-12**。

## 6. 未完成项（诚实登记）

`tests/test_ice_floe_physics.py`（16 项）在本分支仍红，且**在本增量之前就已红**
（在 f5de971 的干净副本上同样 16 failed，与本次改动无因果关系）。原因是该测试覆盖的
`benchmarks/ice_floe_physics.py` v2 五锚点接口（`analytic_submerged_*`、
`anchor_hydrostatic_equilibrium`、`anchor_archimedes_static`、`anchor_heave_period`、
`anchor_friction_slip`、`anchor_restitution_mapping`、`run_all`，schema
`tensordem.ice-floe-physics/2`）从未进入主干，现存脚本仍是三锚点旧版。
补齐它需要重写该基准模块（5 个锚点 + 每个力律的负控制），属于独立增量，本单不做，
不以改测试断言的方式洗红。

## 7. 复算命令

```
python -m pytest -q tests/test_dem3d_moving_wedge_bow.py tests/test_dem3d_restitution_verification.py \
    tests/test_dem3d_wedge_resistance_sensitivity.py tests/test_doc_evidence_pointers.py \
    tests/test_fracture_gen_audit.py tests/test_fracture_gen_footprint_breakage.py \
    tests/test_fracture_gen_mass_mapping.py tests/test_fracture_gen_mass_footprint.py \
    tests/test_wake_lbm_analytic_verification.py
python scripts/audit_doc_evidence.py --check
python benchmarks/fracture_gen_audit.py           # 重生成的报告与已提交账本逐叶一致
python benchmarks/fracture_gen_mass_footprint_audit.py
```