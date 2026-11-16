# AUDIT-SCI — 求解器 README 声称 vs 代码真相对账（TensorDEM + TensorSPH）

单号 `m22981` · 分支 `agent/m22981` · 锚点 `AUDIT-SCI`
（`feature_ledger`：求解器 README 声称 vs 代码真相对账，验收=两仓逐条对账、声称→代码+测试+基准数值、
虚标逐条列）

**方法可执行化**：本单不写「人工读了一遍」，而是把对账做成可执行合同
`scripts/readme_claim_audit.py`（15 条注册声称 × 7 类检查），产物
`artifacts/readme_claim_audit/{claims.json,CLAIM_EVIDENCE_TABLE.md}`，
契约测试 `tests/test_readme_claim_audit.py`。**README 改一句、代码删一个符号、门禁值漂一点，
审计就会变色**——这正是这类对账唯一可能不腐烂的形态。

```
$ python scripts/readme_claim_audit.py --check-entrypoints --strict
REAL 8 · REAL_REGENERABLE 4 · SELF_CONTRADICTION 1 · WITHHELD 1 · (3 SPH …)
```

| 仓库 | 审计快照 | 注册声称 | REAL | REAL_REGENERABLE | STALE_NUMBER | WITHHELD | SELF_CONTRADICTION | NOT_SUPPORTED |
|---|---|---|---|---|---|---|---|---|
| TensorDEM | `agent/m22981`（7af13456 派生） | 11 | 7 | 4 | 0（已修，见 D2） | 0 | 0 | 0 |
| TensorSPH | `cde14cf5`（只读外部检出） | 4 | 2 | 0 | 0 | 1 | 1 | 0 |

判定口径（`README` 里每条声称落到哪一格）：

* **REAL**：代码文件存在 + `path::符号`在 AST 里真的被*定义*（不是 import、不是注释）+ 测试文件存在且
  引用了被测模块 + 引用的**受版本管理**证据（`docs/`、`schemas/`、`benchmarks/.../gates.json`）存在 +
  引用的数字在指定来源里能找到第二份。
* **REAL_REGENERABLE**：同上，但数值证据是 `artifacts/` 下的**生成物**。本仓惯例是生成物不入 VCS
  （`scripts/readme_api_audit.py:652` 明示：「生成物不入 VCS … 干净检出里这类断言会让 pytest -q 固定红」），
  所以缺 `artifacts/` 只有在**同时缺生成脚本**时才算缺陷；本单进一步用 `--check-entrypoints` 实跑
  `--help` 证明脚本活着（结果：`help-ok`）。
* **WITHHELD / SELF_CONTRADICTION / NOT_SUPPORTED / STALE_NUMBER / NO_TEST / GATE_DRIFT**：见文末表。

---

## 一、TensorDEM：锚点三条能力到底有没有实现与算例

锚点要求「DEM 重点=冰脊触底/管道地基/船体冰磨损是否真有实现与算例」。逐条核到位：

| 能力 | 实现（定位到文件） | 测试 | 可读数值证据 | 判定 |
|---|---|---|---|---|
| 冰脊触底 + 海底管道地基 | `src/tensordem/ridge_dem.py`(1104 行) / `ridge_geotech.py`(770) / `ridge_grounding.py`(706) | `test_ridge_grounding.py`、`test_ridge_geotech_capacity_verification.py`、`test_ice_ridge_grounding_rederivation.py` | `docs/ice_ridge_pipeline_grounding.md`（DNV-OS-F101 利用率、Mohr-Coulomb 承载力）；`artifacts/ridge_grounding/` **不在 VCS**，由 `scripts/run_ice_ridge_grounding.py` 重生成（`--help` OK） | **REAL_REGENERABLE** |
| 船体水线带冰磨损 | `src/tensordem/ice_abrasion.py`(1820 行)：Hertz 接触 + Cundall–Strack 切向饱和 → Archard `V=KWL/H` + Finnie 高冲击角 | `test_ice_abrasion*.py`（8 个文件引用） | `docs/ice_abrasion_archard_wear.md`（`V = K W L / H`、`CALIBRATED` 单参数标定声明） | **REAL_REGENERABLE** |
| 稳态碎冰直航 + 接触级能量账本 | `src/tensordem/ice_transit/{transit,laws,report}.py` | `tests/test_ice_transit.py` | `docs/ICE_TRANSIT_RESISTANCE_SURFACE.md`（`W_thrust` 账本恒等式、`ShipForm.signed_distance` 缺陷修复记录）；`artifacts/ice_transit/` 由 `scripts/run_ice_transit_campaign.py` 重生成 | **REAL_REGENERABLE** |

**结论（DEM 半场）**：三条能力**都是真实现**，不是 MOCK/STUB；README 对校准状态的自我标注也诚实
（磨损绝对量级只有一个参数 `k` 对文献锚点标定，README 明确写 `CALIBRATED`，没有把标定值说成预测值）。
唯一系统性弱点是 **数值证据不在版本库里**，见 D1。

### D1（中，非缺陷·口径风险）三条能力的基准数值只能重跑得到，不能从干净检出读到
`artifacts/{ridge_grounding,ice_abrasion,ice_transit,pressure_band_lambda*}/` 全部不存在于 git。
这不是虚标（有文档 + 脚本 + 测试，启动器 `--help` 全部 OK），但**审稿人无法在检出里核对任何一条
载荷–位移曲线或磨损深度场**，只能重跑整场 campaign。建议：把每场 campaign 的**判据 JSON**
（不是整份 PNG/轨迹）纳入版本管理——`pressure_band` 已经是这个模式（`schemas/lambda_x_estimate.v1.json`
在库内、`artifacts/.../criteria.json` 不在），把 `criteria`/`gates` 级的 JSON 一并入库即可，
体积小、可 diff、可直接被本审计当 `schemas` 检查。

### D2（低，**本单已修**）README 测试规模快照比代码树小一半
README 原话「2026-11 巡检快照：343 个测试文件、2714 个模块级 `test_*` 函数、319 个
`unittest.TestCase` 类」，用仓库自己的计数定义（`scripts/readme_api_audit.py:_test_style`：模块级
`test_*` + `TestCase` 子类）现场复算为 **698 / 6506 / 346**，差约 2 倍。
已改为实测值，并把该句纳入审计（`DEM-README-TEST-CENSUS`，带日期戳的快照漂移只记 info，
无日期戳的数字不匹配才升级为 `STALE_NUMBER`）——避免下一轮又腐烂成一句假话。

### D3（信息）悬空指针：DEM 侧只剩「生成物」类
README 里所有**受版本管理**的路径（`docs/`、`examples/hull_profile.csv`、`schemas/`、
`benchmarks/verified/lbm_dem_drag_calibration/{result.json,gates.json}`）**全部解析成功**；
悬空的 5 条全是 `artifacts/`，属 D1 的同一件事。

### D4（正面证据）LBMDEM 的 honest FAIL 是真 FAIL，不是修辞
README 自陈「`gates.json.all_recomputed_pass = false`，曳力闭合只在 Stokes 极限定量可用」。
机器核对：`benchmarks/verified/lbm_dem_drag_calibration/gates.json` 内
`stokes_drag_analytic.measured = 0.009757`（= +0.98 %，门 ≤2 %，pass=true），顶层
`all_recomputed_pass = false`。即 README **没有**把一条通过的门说成整体通过，也没有隐瞒未通过项
（中低 Re 的 C_D、MEM/直接强迫一致性、二球串联、沉降终速都逐条列出）。这是本仓文档纪律的正面样本。

---

## 二、TensorSPH（只读外部检出 `cde14cf5`）：3-D 算子与自由液面/表面张力/多相

### S1（**阻断级，文档缺陷**）README 把 3-D 的全部量化验证指向一个不存在的文件
README 两处写「see `docs/verification_threed.md`」「the numbers, the criteria and the limitations are in
`docs/verification_threed.md`」，该文件在 `cde14cf5` 里**不存在**（`git ls-files | grep threed` 无此 doc）。
真实证据在 `benchmarks/threed_shear/report.json` 与 `benchmarks/threed_hydrostatic/report.json`
（机器可读，存在）。修法只有两条：写这份 doc，或把 README 的指针改到两个 report.json。

### S2（**阻断级，自相矛盾**）Surface tension 同时被声称与否认
同一份 README：`## Implemented` 说 CSF 表面张力 + mixture 版本「Young–Laplace verified to **1.4 %**」；
`## Current limitations` 第一条说「**No surface-tension model.** Free-surface pressure is clamped at zero,
which removes tensile instability but is not a capillary model」。
代码侧为真（`src/tensorsph/surface_tension.py`、`csf_mixture.py`，`docs/verification_mixture_surface_tension.md`
里 `-1.4 %` 在控制算例列）；即**限制段是没随实现更新的残留**。读者按哪一段决策都可能错。

### S3（中，数字不可复现）3-D 静水底压「within 2.7 %」在它自己的 report 里找不到
README：「hydrostatic bottom pressure within 2.7 %」。`benchmarks/threed_hydrostatic/report.json` 的四个
gated 算例底压相对误差为：

| 算例 | 底压相对误差 | 核心层最大误差 | 15 % 内层数 |
|---|---|---|---|
| `adami-c0_40-nz16`（best_case） | **−1.21 %** | +17.3 % | 11 |
| `adami-c0_20-nz16` | −2.49 % | +24.3 % | 10 |
| `adami-c0_10-nz12` | −4.39 % | +24.9 % | 6 |
| `adami-c0_10-nz16` | −4.21 % | +25.8 % | 10 |

**没有任何 gated 算例是 2.7 %**；落在 2.7 % 以内的只有 best_case（1.21 %），其余三个都超出。
而 2.7 % 恰好等于同句里 shear-decay 的数（`resolution_ladder.-1.decay.rel_error = −2.68 %`），
**同一句里两个 2.7 % 极可能是抄了一个**。此外「底压」单点指标之外的**核心层最大误差 17–26 %** 完全没
出现在 README——这才是这套算例的真实精度面。建议把该句改成「底压 best-case −1.2 %（四个 gated 算例
−1.2 %…−4.4 %），核心层最大误差 ≤26 %」。

### S4（中，结论被隐去）shear 报告的顶层 verdict 是 `fail`
同一份 `threed_shear/report.json`：`verdict = "fail"`，触发项
`fourth_order_improves_on_second_order_every_rung = false`；且 `missing_cases = [h1.3, h1.5, h1.8,
order4-n16, order4-n20, order4-n24]`（6 个算例没跑）。README 用 "verified against analytic solutions …
shear-wave decay within 2.7 %" 一概而论，未提这条 FAIL 与 6 个缺测。2.7 % 本身是对的（n=24 档 −2.68 %），
问题在**只报通过项**。

### S5（待 SPH 侧确认，本单不下结论）3-D 表面张力可能被**低**估
`src/tensorsph/threed/csf3.py` 实现了 3-D CSF（明确为 Rayleigh–Plateau 液柱不稳定性），而 README 与
`src/tensorsph/threed/__init__.py` 的 docstring 都写「Nothing here is a surface-tension, free-surface or
multiphase model」。本单**无法**判定 csf3 是否已验收（`tests/` 下无 `test_csf3*`，本席未跑其 benchmark），
故在审计里登记为 **unknown/not_run**，不给 pass。

---

## 三、逐条判定表（摘，完整表见 `artifacts/readme_claim_audit/CLAIM_EVIDENCE_TABLE.md`）

| id | 仓库 | 判定 | 依据 |
|---|---|---|---|
| `DEM-CORE-2D` | DEM | REAL | `dem.py::IceDEM/DEMConfig` 定义存在；教程被 `test_tutorial_first_verified_run.py` 原样执行 |
| `DEM-EXTERNAL-LOAD` | DEM | REAL | `external_forces` 由 `tests/test_load_interface.py` 引用 |
| `DEM-LBMDEM-DRAG` | DEM | REAL | 2/2 数字有第二份；门 `stokes_drag_analytic`=0.009757 与 README 的 +0.98 % 一致 |
| `DEM-SENSITIVITY-SCAN` | DEM | REAL | `sensitivity.py` + 测试；README 已否认其为收敛阶估计 |
| `DEM-POLYLINE-HULL` | DEM | REAL | `examples/hull_profile.csv` 在库内，`test_hull_profile_cli.py` 引用 |
| `DEM-DEM3D` | DEM | REAL | `dem3d.py::IceDEM3D/DEM3DConfig` 定义存在；`test_dem3d.py` |
| `DEM-RIDGE-GROUNDING` | DEM | REAL_REGENERABLE | 代码+3 测试+doc；`artifacts/` 由 `run_ice_ridge_grounding.py` 重生成 |
| `DEM-ICE-ABRASION` | DEM | REAL_REGENERABLE | 同上；`CALIBRATED` 标注真实存在 |
| `DEM-ICE-TRANSIT` | DEM | REAL_REGENERABLE | 同上；`laws.py` 内 `signed_distance` 缺陷修复记录可查 |
| `DEM-PRESSURE-BAND` | DEM | REAL_REGENERABLE | README 的 honest FAIL（0.058 vs 1.5–3）两份来源一致 |
| `DEM-README-TEST-CENSUS` | DEM | REAL（本单已修 D2） | 698/6506/346 现场复算一致 |
| `SPH-2D-CASES` | SPH | REAL | 观测阶 0.96 在 `docs/benchmarks/dam-break-2d.md` |
| `SPH-3D-OPERATORS` | SPH | **WITHHELD** | 缺 `docs/verification_threed.md`；静水 2.7 % 无算例支持；report `verdict=fail` 未披露 |
| `SPH-SURFACE-TENSION` | SPH | **SELF_CONTRADICTION** | "1.4 %" 与 "No surface-tension model." 同文并存 |
| `SPH-WALL-MODEL` | SPH | REAL | B2 密度门 8.6 % 可在 `dam-break-2d.md` 复核 |

**虚标结论**：DEM 仓**未发现**能力虚标（三条锚点能力全部真实现，honest FAIL 也是真的）；
虚标集中在 **SPH 仓**：1 条自相矛盾（S2）、2 条只报通过项/数字不可复现（S3/S4）、1 条证据指针悬空（S1）。

---

## 四、注意点与适用范围（本审计自己的局限）

1. **注册表≠全文**：审计覆盖 15 条**能力级**声称，不是 README 每一句话。散文性描述（背景、动机）不查。
2. **"存在"≠"正确"**：`code`/`tests` 检查的是*被定义、被引用*，不是*物理正确*。数值正确性由各能力自己的
   benchmark 门负责（本单对 LBMDEM 与 SPH 3-D 做了门级复核，其余沿用仓内既有门）。
3. **SPH 侧只读**：外部检出 `/root/tensor-ecosystem/TensorSPH@cde14cf5`，本单不改 SPH 仓（白名单外，
   且未经该仓验收），S1–S4 以问题单形式移交。
4. **`--check-entrypoints` 只证明启动器能 import 并解析参数**（`--help` 返回 0），不证明它能跑完一整场
   campaign。真正的重生成验证需要重算（本机 load15≈91，按容量纪律本单未起）。
5. **未跑**：`pressure_band` 的合成标定扫描、ice_transit campaign、SPH `threed_shear`（约 30 min）与
   `threed_hydrostatic`（约 10 min）benchmark 均**未在本单重跑**，故只引用仓内既有 report.json 的原始字段值，
   不新增测量。这些记为 `not_run`，不作为 pass。

## 五、复现

```bash
python scripts/readme_claim_audit.py                    # 生成表 + JSON
python scripts/readme_claim_audit.py --check-entrypoints --strict   # 含生成器可用性 + 回归门
python -m pytest -q tests/test_readme_claim_audit.py    # 契约测试（含负例）
```