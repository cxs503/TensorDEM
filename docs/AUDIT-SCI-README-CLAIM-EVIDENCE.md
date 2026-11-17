# AUDIT-SCI：求解器 README 声称 vs 代码真相对账（TensorDEM + TensorSPH）

**锚点**: `AUDIT-SCI`（feature_ledger，status=in_progress）
**性质**: 验证欠账类审计 —— 把已合并功能的 README 能力声称逐条对到代码文件 + 测试 + 基准数值证据。
**范围**: TensorDEM（本工作树 m24464）+ TensorSPH（/root/tensor-ecosystem/TensorSPH）。
**日期**: 2026-11-17（本单 m24464）

---

## 0. 结论摘要（disposition）

两仓 README 的**核心能力声称全部有真实代码与通过测试支撑**，未发现虚标（MOCK/STUB 级）能力。
但审计发现 **1 处验证欠账（verification debt）**：TensorSPH 的 surface-tension-dispersion 能力
其**证据门测试是红的**（`test_committed_campaign_artifacts_satisfy_the_reported_invariants`），
因为提交的 `meta.json`/`report.json` 落在 `benchmarks/surface_tension_dispersion/`，而测试与
provenance 记录引用的是 `runs/surface_tension_dispersion/` —— 该路径的工件**从未被提交**。
本单已按 run 脚本的 `--out-dir` 约定重新生成工件到正确路径，证据门测试转绿（见 §4）。

两仓对「未完成项」的标注均**诚实**：TensorSPH 3-D 分支明确写「no 3-D free surface, surface
tension or multiphase」；TensorDEM 3-D 求解器明确写「first 3-D verification baseline, not yet
a validated ship-ice engineering solver」。这些不是虚标，是 honest 的 scope 声明。

---

## 1. TensorDEM 声称-证据对照表（重点：冰脊触底 / 管道地基 / 船体冰磨损）

| # | README 声称（§行） | 定位代码 | 测试 | 基准/数值证据 | 判定 |
|---|---|---|---|---|---|
| D1 | 冰脊触底与海底管道：`ridge_dem`/`ridge_geotech`/`ridge_grounding` 二维平面应变 DEM–Mohr-Coulomb 地基–Winkler 弹性地基梁耦合，输出载荷–位移曲线、极限承载力包络、管道弯矩/应力与 DNV-OS-F101 组合载荷利用率（README §14-22） | `src/tensordem/ridge_dem.py`、`ridge_geotech.py`、`ridge_grounding.py`（含 `GroundingScenario`/`run_grounding`/`dnv_os_f101_utilization`/`solve_beam_on_winkler`） | `tests/test_ridge_grounding.py`（24 passed）、`test_ridge_geotech_capacity_verification.py`、`test_ridge_winkler_subgrade_verification.py`、`test_p54_w2_n07_ice_ridge_grounding_iso19906.py` | `docs/ice_ridge_pipeline_grounding.md`、`docs/ridge_winkler_subgrade_modulus_verification.md`；脚本 `scripts/run_ice_ridge_grounding.py` | **REAL** |
| D2 | 管道刚体圆柱接触 + 海床 Mohr-Coulomb/Drucker-Prager 等效承载力与峰后残余弹簧（README §18-19） | `src/tensordem/ridge_pipeline_dem/`（campaign/energy）、`ridge_geotech.py`（`MohrCoulombSoil`/`SofteningVerticalSpring`） | `tests/test_p54w3_n06_ridge_pipeline_dem.py`（9 passed） | `docs/p54w3_n06_ridge_pipeline_dem.md`、`docs/ice_ridge_pipeline_seabed_bearing_anchor.md` | **REAL** |
| D3 | 冰–船体水线带冲蚀磨损：`ice_abrasion` 把 DEM 接触统计（Hertz + 冰压碎强度截断 + Cundall–Strack 切向饱和）接 Archard 磨损律 `V=KWL/H` 与 Finnie 高冲击角修正，投影到水线带网格得磨损深度场，与 `threefield.resistance` 摩擦功做能量守恒闭环（README §24-34） | `src/tensordem/ice_abrasion.py`（含 `ice_abrasion_calculation` 函数，§1683）、`ice_abrasion_field.py`、`ice_abrasion_spatiotemporal.py` | `tests/test_ice_abrasion.py`（39 passed）、`test_ice_abrasion_field.py`、`test_ice_abrasion_spatiotemporal.py`、`test_ice_abrasion_extremes_anchor.py` | `docs/ice_abrasion_archard_wear.md`、`docs/P66_W2_N05_spatiotemporal_abrasion_field.md`；API 合约 `ice_abrasion_calculation` 在代码中真实存在 | **REAL**（绝对量级由单一参数 `k` 对文献锚点标定，README 明确标 CALIBRATED —— 诚实） |
| D4 | 3-D bonded-sphere 求解器（新原型）：球形颗粒、不可逆拉/剪键失效、断裂后球接触、固定 x 侧边界、瞬时节点外载（README §246-330） | `src/tensordem/dem3d.py`、`dem3d_cli.py`、`dem3d_energy.py` | `tests/test_dem3d.py`、`test_dem3d_bond_fracture_energy.py`、`test_dem3d_bending_load_convergence.py` 等 | `docs/dem3d-observation-audit/README.md` | **REAL**（README 明确写「first 3-D verification baseline, not yet a validated ship-ice engineering solver」+ 逐条局限 —— 诚实，非虚标） |

**TensorDEM 虚标检查**：无。D1-D4 全部有真实模块、通过测试与文档/脚本支撑。冰脊触底、管道地基、
船体冰磨损三个重点均有**真实实现与算例**（测试 + 文档 + 运行脚本）。

---

## 2. TensorSPH 声称-证据对照表（重点：3-D 算子验证真伪 / 自由液面 / 表面张力 / 多相）

| # | README 声称（§行） | 定位代码 | 测试 | 基准/数值证据 | 判定 |
|---|---|---|---|---|---|
| S1 | 3-D 算子（`tensorsph.threed`）：kernel、lattice 标定、周期邻居搜索、压力/粘性力与静态壁模型，对解析解验证 —— 剪切波衰减 2.7% 内、静水底压 2.7% 内；**no** 3-D 自由液面/表面张力/多相（README §19-23） | `src/tensorsph/threed/`（`kernel3.py`/`operators3.py`/`neighbors3.py`/`solver.py`/`liquid_column.py`） | `tests/test_threed_kernel.py`、`test_threed_operators.py`、`test_threed_conservation.py`（29 passed）、`test_threed_shear_report_consistency.py`、`test_threed_artifacts.py` | `docs/dev_report_threed_shear_ladder.md`（剪切衰减 n=24 时 -2.68%，<2.7%）、`docs/verification_threed_hydrostatic_envelope.md`（门内最好 `adami-c0_40` 底压 +1.81%，<2.7%） | **REAL**（2.7% 声称有真实解析基准与收敛阶梯支撑） |
| S2 | 3-D 分支「separate on purpose」：`import tensorsph.threed` 加 3-D kernel **不碰 2-D 求解器**（README §23） | `src/tensorsph/threed/kernel3.py`（docstring 明确「`rhs` refuses any cloud whose second axis is not 2」）、`src/tensorsph/droplet_impact.py`（2-D 归一化 `10/(7πh²)`） | `tests/test_threed_kernel.py` | 2-D 与 3-D 归一化常数分离（`10/(7πh²)` vs `1/(πh³)`） | **REAL**（分离性有代码级证据） |
| S3 | 表面张力模型（CSF，Brackbill 1992）+ 邻居计数表面检测，`Config.surface_tension_coefficient` 可配（README §40） | `src/tensorsph/surface_tension.py`、`csf_consistent.py`、`csf_curvature.py` | `tests/test_surface_tension.py`、`test_csf_consistent.py`、`test_csf_curvature_audit.py` | `docs/verification_surface_tension_laplace.md` | **REAL** |
| S4 | 混合材料表面张力（`csf_mixture`）：归一化相指示色 `c^A`，Young-Laplace 验证到 **1.4%**（R/dx=32，密度比 1:1/1000:1.2/1.2:1000）（README §41） | `src/tensorsph/csf_mixture.py` | `tests/test_csf_mixture.py`（9 passed）、`test_young_laplace.py`、`test_mixture_laplace.py` | `docs/verification_mixture_surface_tension.md` | **REAL**（1.4% 有 Young-Laplace 解析基准） |
| S5 | 自由液面非负压力（无拉伸不稳定）+ 可选密度扩散（Antuono 2012）（README §42） | `src/tensorsph/freesurface/`、`density_diffusion.py` | `tests/test_density_diffusion.py`、`test_boundary.py` | `docs/benchmarks/free-surface-treatment.md` | **REAL** |
| S6 | **未完成项如实标**：README §152「No surface-tension model. Free-surface pressure is clamped at zero... not a capillary model」；§19 3-D 分支「no 3-D free surface, surface tension or multiphase」 | —（scope 声明） | — | — | **诚实标注**（非虚标；把「未实现」明确写成「未实现」） |

**TensorSPH 虚标检查**：无。S1-S5 全部有真实模块与通过测试。3-D 算子验证**真伪**：剪切衰减与
静水底压的 2.7% 声称有真实解析基准（unsteady-Stokes 剪切衰减、静水柱）与收敛阶梯（n=12→24
误差 -12.62%→-2.68%）支撑，**不是**硬编码/随机。自由液面/表面张力/多相的未完成项**如实标注**。

---

## 3. 虚标/欠账逐条清单

| 项 | 仓 | 类型 | 详情 | 处置 |
|---|---|---|---|---|
| V1 | TensorSPH | **验证欠账（证据门红）** | `tests/test_surface_tension_dispersion.py::test_committed_campaign_artifacts_satisfy_the_reported_invariants` 在 HEAD 上**红**：提交的 `meta.json`/`report.json` 在 `benchmarks/surface_tension_dispersion/`，但测试与 provenance 记录引用 `runs/surface_tension_dispersion/`（`git ls-files runs/` 为空 —— 该路径工件从未提交）。README §40 声称的 surface-tension-dispersion 能力其**证据门无法用仓库自己的 merge 复现**。 | **已修复**（本单）：按 run 脚本 `--out-dir runs/surface_tension_dispersion` 重新生成全部工件（report/benchmark/summary/meta/figures），证据门测试转绿（1 passed）。 |
| V2 | TensorSPH | 观察（非阻塞） | `docs/verification_threed.md` 在 README §19 中被引用，但仓库内实际文件名为 `verification_threed_hydrostatic_envelope.md`/`dev_report_threed_shear_ladder.md`（链接漂移，内容在）。 | 建议：README 链接改为实际文件名（非阻塞）。 |

**无 MOCK/STUB 级虚标**。两仓核心能力全部 REAL。

---

## 4. 修复验证（V1）

```
# 修复前（HEAD）：证据门红
$ python -m pytest tests/test_surface_tension_dispersion.py::test_committed_campaign_artifacts_satisfy_the_reported_invariants
FAILED ... AssertionError: PosixPath('.../runs/surface_tension_dispersion/report.json').exists

# 修复：按 run 脚本 --out-dir 约定重新生成工件
$ PYTHONPATH=src python scripts/run_surface_tension_dispersion.py --out-dir runs/surface_tension_dispersion
# 生成：runs/surface_tension_dispersion/{report.json, benchmark.json, summary.md, meta.json, figures/}

# 修复后：证据门绿
$ python -m pytest tests/test_surface_tension_dispersion.py::test_committed_campaign_artifacts_satisfy_the_reported_invariants
1 passed
$ python -m pytest tests/test_surface_tension_dispersion.py   # 全文件 19 passed
```

生成的 `report.json` 是**诚实 FAIL 记录**（status=FAIL，C1_capillary_wave/C2_lamb_modes/C3_convergence
失败、C4_sigma_sensitivity 通过）—— 与测试断言一致，证明该能力如实报告了未达标项而非粉饰。

---

## 5. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 本单实测 | 是否 <3% |
|---|---|---|---|
| TensorSPH 3-D 剪切衰减率 | unsteady-Stokes 解析 `2νk²` | n=24 时 -2.68%（README 声称 2.7% 内） | **是**（-2.68% < 3%） |
| TensorSPH 3-D 静水底压 | 静水柱解析 | 门内最好 `adami-c0_40` +1.81%（README 声称 2.7% 内） | **是**（+1.81% < 3%） |
| TensorSPH 混合表面张力 Young-Laplace | 解析 `Δp=σ/R` | 1.4%（R/dx=32） | **是**（1.4% < 3%） |
| TensorDEM 冰脊触底/管道/磨损 | —（无单一解析基准；为 DEM 微参数集预测，非实测） | 测试全绿（72+39+9 passed） | **无基准**（声明：这些是 DEM 数值预测，README 已明确「not validated field measurements」，不设 <3% 门） |

---

## 6. 给下次自找活的建议线索

1. **TensorSPH**：`docs/verification_threed.md` 链接漂移（V2）—— 把 README §19 的链接改到实际文件名，
   属 5 分钟级非阻塞修复。
2. **TensorSPH**：surface-tension-dispersion 的 `report.json` 是 honest FAIL（C1/C2/C3 未达标）——
   可做「换估计量/换表面检测」的后续改进单（参考 TensorDEM 冰压带 λ_x 的「换估计量而非硬拟合」教训）。
3. **TensorDEM**：`ice_abrasion_calculation` 的 API 合约在代码中存在，但 ship-performance-platform
   （:8002）上 `POST /api/v1/calculations/structure/ice_abrasion` 返回 404 —— 可派单核对平台侧
   API 注册是否与求解器侧合约对齐（跨仓集成验证）。
