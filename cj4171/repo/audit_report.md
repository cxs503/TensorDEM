# AUDIT-ASSET-12REPO · TensorDEM 质量审计报告（A 资产盘点 + B 功能真实性）

**任务** mission_id=14487 · **仓** cxs503/TensorDEM · **审计对象** main @ `20a77f0a` + 近期已并 agent PR
**审计窗口** 最近 25 个 push-merge（PR #1151 → #1209，2026-11-10 → 2026-11-11 14:46 +0800）
**审计人** audit2-ab（独立复核；不采用 PR 自述作为证据）
**判级依据** owner 2026-11-11 六级定义（L0 文档 / L1 数学与单元验证 / L2 独立功能实现 / L3 系统集成（含工作流协同运行）/ L4 基准验证（含收敛·守恒·误差·可复现）/ L5 科研工程成果）

**结论一句话** 窗口内新增 54,546 行（195 文件）：核心代码 43 %、测试 13 %、文档 14 %、脚本 7 %、其它 22 %（含 4.3 k 行被误并的 `.tmpwork/` 草稿）。核心功能多数真实可运行（L4=3、L3=2、L2=15），但发现 **2 起「验收测量由测试自己合成/判据恒等不可能失败」的伪验证**、**6 处模块/包同名遮蔽导致 2 个测试模块彻底无法收集**、**40/77 新模块无生产调用方**。L0/L1 共 5 个。

---

## 0. 审计方法与局限（honest）

| 环节 | 做法 | 工件 |
|---|---|---|
| 资产盘点 | 逐 merge commit `git diff --numstat <sha>^1 <sha>`，按路径分类（core/test/doc/script/other）+ 静态体检（stub 标记、顶层函数、调用方） | `artifacts/audit_m14487/inventory.json`；工具 `tools/audit_asset_inventory.py` |
| 调用方判定 | 全仓 import 索引（`src/tensordem/a/b.py → tensordem.a.b`），区分「无调用方 / 仅测试 / 生产调用」 | `artifacts/audit_m14487/wiring.json` |
| 真跑 | 30 个新增测试文件逐文件 pytest；代表性 benchmark/study 脚本真跑 | `artifacts/audit_m14487/logs/*.log` |
| 结构体检 | 复跑仓内已有 import 合约扫描器（#1202 交付） | `artifacts/import_contract/broken_imports.json` |
| 同名遮蔽扫描 | 自写 `X.py` vs `X/` 冲突扫描 | 本报告 §4-F3 |

**局限（未验证即未验证，不计为通过）**
1. 5 个新增测试文件本地单次调用 120 s 预算内跑不完（`test_polyhedra_repose_angle`、`test_polyhedra_repose_fast`、`test_friction_polyhedron_real_verification` 等），已递交集群作业 `cj850`/`cj881`（递交时排队 252/274，截至本报告落笔仍 queued）→ 记为 `not_run`。
2. `freeze_bond` 全量 benchmark、`pid_mdof` benchmark 本地超时未跑完；其判级只依据「单测真跑通过 + PR 已入库结果 CSV/JSON」。
3. 逐行复核 14 个功能；其余按静态 + 单测证据判定，已在 §2 表末列标注复核方式。

---

## 1. 资产盘点（A）

### 1.1 分类清单与行数（25 个 merge / 195 文件 / +54,546 行）

| 分类 | 文件数 | 新增行 | 说明 |
|---|---|---|---|
| 核心算法模块 `src/tensordem/**` | 77 | 23,669（当前 29,560 LOC） | 9 个新包（ice_thermal_stress / erosion_dem / seepage / fem_dem_seeding / rotary_dryer / ice_anisotropy / wave_ice / shape_representation / ice_friction_meltfilm）+ 15 个单文件模块 |
| 测试代码 `tests/**` | 30 | 7,291 | 全部新增测试文件 |
| 文档报告 `docs/**` + `*.md` | 39 | 7,714 | 28 篇 md（5,293 行）+ 2 个 JSON 报告 + 9 张 png |
| 运行器 `scripts/**` | 20 | 3,993 | 含 3 个 fem_dem_seeding 交接脚本、post 报告脚本 |
| 其它（benchmark 运行器/结果、工件、过程垃圾） | 29 | 11,879 | 其中 **4,331 行是 `.tmpwork/` 草稿**（#1157 误并，后被 #1174 删除） |

### 1.2 逐 PR 盘点表

| PR | 核心模块文件(新增行) | 测试 | 文档 | 脚本 | 其它 | 合计 | 结果工件入库 | 判级 |
|---|---|---|---|---|---|---|---|---|
| #1151 freeze_bond/thermal_profile | 3 (815) | 1 | 1 | 0 | 5 | 4,487 | ✅ csv+json | **L4** |
| #1157 pid_mdof | 1 (930) | 1 | 1 | 0 | 11 | 7,323 | ❌（含 4.3k 垃圾） | L2 |
| #1162 polyhedra_repose | 1 (478) | 3 | 1 | 0 | 0 | 1,674 | ❌ | L2 |
| #1168 wave_ice + bowden_hughes | 5 (1,909) | 1 | 1 | 1 | 0 | 2,878 | ❌ | L2 |
| #1170 random_ice_strength | 1 (23) | 0 | 1 | 0 | 0 | 60 | ❌ | L1 |
| #1171 ice_ridge_formation | 2 (241) | 0 | 2 | 1 | 1 | 1,047 | ❌ | L1 |
| #1173 rotary_dryer | 4 (1,486) | 1 | 1 | 1 | 0 | 2,928 | ❌ | L2 |
| #1175 ice_friction_regime_map | 1 (695) | 1 | 1 | 0 | 0 | 1,313 | ❌ | L2 |
| #1176 seepage | 6 (1,469) | 1 | 1 | 1 | 0 | 2,285 | ❌ | **L1** |
| #1178 fem_dem_seeding | 6 (1,929) | 1 | 1 | 3 | 0 | 3,154 | ❌ | L2 |
| #1179 ice_friction_meltfilm | 4 (919) | 1 | 1 | 1 | 0 | 1,915 | ❌ | L2 |
| #1180 gen_energy_audit_figures | 0 | 0 | 0 | 1 | 0 | 211 | ❌ | **L0** |
| #1181 contact/friction_law | 3 (1,046) | 1 | 0 | 2 | 0 | 1,742 | ❌ | L2 |
| #1182 IN157_N03 文档 | 0 | 0 | 1 | 0 | 0 | 505 | ❌ | **L0** |
| #1188 ice_vertical_structure_pileup | 1 (851) | 1 | 7 | 0 | 3 | 4,439 | ✅ report.json+5 图 | **L3** |
| #1191 ice_thermal_stress | 8 (3,017) | 4 | 1 | 0 | 2 | 4,422 | 运行生成（未入库） | **L4** |
| #1195 erosion_dem | 6 (1,649) | 1 | 1 | 0 | 1 | 3,090 | 运行生成（未入库） | **L4** |
| #1196 breakage_pbm | 5 (1,063) | 1 | 1 | 4 | 0 | 1,717 | ❌ | L2 |
| #1202 shape_representation | 4 (922) | 1 | 1 | 0 | 2 | 1,415 | 运行生成（未入库） | L2（含结构缺陷） |
| #1203 friction_polyhedron 真验证 | 3 (637) | 1 | 2 | 0 | 0 | 966 | ❌ | **L3** |
| #1204 shear_cell | 1 (419) | 1 | 2 | 1 | 1 | 651 | ❌ | L2 |
| #1205 dem3d_energy | 2 (154) | 1 | 2 | 0 | 0 | 438 | ❌ | L2 |
| #1206 friction_surface repose v2 | 2 (200) | 1 | 2 | 2 | 1 | 850 | ❌ | L2 |
| #1208 ice_anisotropy | 7 (1,878) | 5 | 6 | 2 | 0 | 3,116 | ✅ 4 图 | L2 |
| #1209 thermal_fracture_dem | 1 (939) | 1 | 1 | 0 | 2 | 1,920 | ✅ results.json+report.md | L2（判据自证） |

### 1.3 占位实现 / 重复实现 / 未接入模块

**占位实现**（全窗仅 4 处显式标记，均为声明式护栏而非空壳）

| 位置 | 内容 | 定性 |
|---|---|---|
| `src/tensordem/breakage_pbm/daughter_distribution.py:240` | `raise NotImplementedError("quantile form is available for n_daughters == 2")` | **部分实现**（n≠2 分支未做；3 例 xfail 钉住） |
| `src/tensordem/shape_representation/pile2d.py:95` | `raise NotImplementedError` 拒绝 3D 通道 | 设计护栏 |
| `friction_polyhedron_real_verification.py` / `wave_ice/floe_rao.py` | 各 1 处 stub 词命中 | 文档串误报 |

**重复实现（功能重叠）**

| 重复对 | 证据 | 风险 |
|---|---|---|
| `friction_polyhedron.py` (745) / `friction_polyhedron_verification.py` (324) / `friction_polyhedron_real_verification.py` (549) | 三者同题；#1203 已证明中间者 C1–C3 自证 | 高：旧自证模块仍在树内，`scripts/verify_friction_polyhedron.py:26` 仍调用 |
| `shape_representation.py` (1,080) / `shape_representation/`（包） | 6 个公共名（`SurfaceDeviation`/`ellipticity_closure`/`sample_superquadric_surface`/`sq_signed_distance_estimate`/`superellipse_curve_radius`/`union_balls_signed_distance`）只在旧模块，包未导出 | 高：旧模块被遮蔽成死代码（§4-F3） |
| `ice_structure_contact.py` / `ice_structure_contact/`（包） | 15 个符号只在旧模块，测试已无法收集 | 高：整块验收面消失 |
| `ice_thermal_stress/validation.py` / `ice_friction_thermo/validation.py` | `acceptance_report`/`canonical_digest`/`convergence_table` 两处各写一遍 | 低（复制粘贴） |
| `erosion.py` / `erosion_dem/` / `erosion_pbm/` | PR 文档显式声明三段分工并共用材料库 | 低（有声明） |

**未接入模块**：77 个新核心文件里 **40 个（52 %）无生产调用方**

- 完全无人 import（21 个）：`friction_polyhedron_real_verification`、`shape_representation/{clump_inscribed,elliptic_hertz,pile2d}`、`erosion_dem/{breakage_bridge,report,stream}`、`ice_thermal_stress/validation`、`fem_dem_seeding/{dynamics,kinematics,raster,study}`、`seepage/{chain,suffusion}`、`rotary_dryer/psychrometrics`、`ice_ridge_formation/{dynamics,floes}`、`wave_ice/{breaking,floe_rao,transmission}`、`pid_mdof`
- 仅被测试 import（19 个）：`ice_anisotropy/{fabric,slip_systems,verification,virtual_test,yield_surface}`、`breakage_pbm/{assembly_compression,daughter_distribution,lagrangian}`、`ice_thermal_stress/{plastic_heat,thermal_stress}`、`ice_friction_meltfilm/{contact,physics,sweep}`、`fem_dem_seeding/clump`、`seepage/critical_gradient`、`ice_friction_regime_map`、`random_ice_strength/study`、`polyhedra_repose`、`freeze_bond/thermal_profile`
- 统计：77 个新核心文件中 **生产调用方 37 个 / 仅测试 19 个 / 完全无 import 21 个**（图见 `artifacts/audit_m14487/figures/fig3_wiring.png`）。
- 结论：本仓已是「结点式研究仓」——每节点自带模块+单测+benchmark+文档，但节点之间、与主求解器 `tensordem.dem.IceDEM`/`cli.py` 之间几乎无边。故 **L3（集成）在窗口内基本未达成**，L4 全靠「自建 benchmark」，两种证据强度不同。

---

## 2. 功能真实性审查（B）六问表

Q1 代码入口 · Q2 真执行计算 · Q3 接入主程序 · Q4 实际输入输出 · Q5 完整运行案例 · Q6 独立基准验证

| 功能（PR） | Q1 入口 | Q2 真算 | Q3 接入 | Q4 IO | Q5 可运行案例 | Q6 独立基准 | 复核 | 判级 |
|---|---|---|---|---|---|---|---|---|
| **ice_thermal_stress**（#1191） | `ice_thermal_stress/__init__.py`（8 模块 3,017 行） | ✅ 1D FV θ 格式 + 独立节点 FEM 交叉校验 + Hunt–Crossley RK4 + 能量账本 | ❌ 仅被自身 benchmark 使用 | ✅ dataclass + `validation.json`/`sweeps.json`/7 图 | ✅ **真跑通**（12 判据全 PASS，2 项 WITHHELD，<115 s） | ✅ C1 精确半无限解、C2 带源 slab、C3 FV-vs-FEM、C13 中接触源解；含离散收敛表 | pytest 45 例全绿 | **L4** |
| **erosion_dem**（#1195） | `erosion_dem/__init__.py`（6 模块 1,649 行） | ✅ 弹道 + 塑性/摩擦恢复多颗粒流积分、球冠坑反解、通量口径 E | ❌ 无生产调用方 | ✅ `artifacts/p176w2-n11/results.json` | ✅ **真跑通**（70 s，C1–C7 全 PASS） | ✅ DNV-RP-O501 逐点对拍（最大偏差 6.33 %，**单锚标定**并标 CALIBRATED）；Oka 常数 WITHHELD | pytest 45 例全绿 | **L4** |
| **freeze_bond/thermal_profile**（#1151） | `freeze_bond/thermal_profile.py`（791 行） | ✅ 焓法凝固前沿 + 强度耦合 | ❌ 无生产调用方 | ✅ csv/json 已入库 | ⚠️ benchmark 本地 110 s 未跑完（集群排队） | ✅ **Neumann 相似解**对拍 0.11 %–2.6 %；能量闭合 4.4e-12；格数 200/400/800 收敛；记载前版 125 % 误差 | pytest 32 例全绿 | **L4** |
| **ice_vertical_structure_pileup**（#1188） | `ice_vertical_structure_pileup.py`（851 行） | ✅ 平面 DEM 挤压堆积 + SDF 方柱接触 + 载荷时历 | ❌ 无生产调用方 | ✅ report.json + 5 图入库 | ✅ **真跑通** | ⚠️ ISO 19906 名义载荷对照：C1 同量级 PASS、**C1b 3 倍带 FAIL 如实报告**；无收敛研究 | pytest 16 例全绿 | **L3** |
| **friction_polyhedron 真验证**（#1203） | `friction_polyhedron_real_verification.py`（549 行，有 `__main__`） | ✅ 经 `PolyhedronDEMSystem` 真跑 DEM 测量；查出并修 3 个内核缺陷 | ✅ 修的是共享内核（6 个调用方） | ✅ 报告 JSON | ⚠️ 模块主程序与测试本地 110 s 未跑完（集群排队）；PR 自报 17 例/1 xfail | ⚠️ 锚点=解析/几何恒等式 + 负控 + 已知局限钉住（强于前版，仍非文献基准） | ⚠️ not_run | **L3** |
| **shape_representation**（#1202） | `shape_representation/`（包 922 行） | ✅ 椭圆 Hertz 闭式 + 独立 Boussinesq 求积残差 + clump 拟合收敛 | ❌ 无生产调用方，且**遮蔽同名旧模块** | ✅ verdict.json | ✅ `benchmarks/p40_n10_shape_restoration_audit.py` | ⚠️ 闭式 vs 求积（独立数值法）；无文献实验对拍 | pytest 27 例全绿 | L2 |
| **thermal_fracture_dem**（#1209） | `thermal_fracture_dem.py`（939 行） | ⚠️ 一维键链 + 闭式热应力，**无动力学/接触/时间积分** | ❌ 无生产调用方 | ✅ results.json | ✅ **真跑通**（全判据 PASS） | ❌ **判据 (d)(f) 为恒等式**（§4-F2）；(a) 自承分辨率受限 | pytest 33 例全绿 | L2 |
| **seepage**（#1176） | `seepage/__init__.py`（6 模块 1,469 行） | ⚠️ 只有解析公式（Terzaghi/Kozeny–Carman/有效应力），无渗流 DEM | ❌ 无生产调用方 | ✅ verdict dict | ⚠️ 无运行案例；「DEM 测量」由测试合成 | ❌ **伪测量**（§4-F1） | pytest 28 例全绿（绿在合成数据上） | **L1** |
| **wave_ice**（#1168） | `wave_ice/`（5 模块 1,909 行，含 647 行圆盘 RAO） | ✅ 线性势流圆盘解 + 透射率 + 破碎判据 | ❌ 仅出图脚本 | ✅ dataclass/数组 | ⚠️ 无 benchmark 数值表 | ❌ 单测只断言峰值存在/有限/`<100`/单调（`tests/test_wave_ice_verification.py:49-120`），文档写 `PASS (29/29 tests)` | pytest 29 例全绿 | L2 |
| **ice_anisotropy**（#1208） | `ice_anisotropy/`（7 模块 1,878 行） | ✅ 组构张量 + 滑移系 + Hill 型屈服面 + 虚拟试验 | ❌ 无生产调用方 | ✅ 账本 record/query 回放 | ✅ `scripts/run_ice_anisotropy_study.py`（未跑） | ⚠️ 45° held-out 外推 +2.67 % 对「自家 mean-field」；von Mises τ/σ=0.5774 对文献 +3.76 % | pytest 45 例全绿 | L2 |
| **rotary_dryer**（#1173） | `rotary_dryer/api.py`（4 模块 1,486 行） | ✅ 床层运动学 + 传热 + RTD + 湿分传递 | ❌ 无生产调用方 | ✅ `calculate_rotary_dryer` 返回含证据分级的结果 | ✅ 单测 43 例 | ⚠️ 文献关联式最小二乘 R²/RMSE（基准出处本轮未独立复核） | pytest 43 例（1 skip） | L2 |
| **fem_dem_seeding**（#1178） | `fem_dem_seeding/`（6 模块 1,929 行） | ✅ 栅格 EDT + 内切圆/球播种 + 碎块划分 | ❌ 无生产调用方（仅交接脚本） | ✅ `as_payload()` 交接负载 | ⚠️ 测试全绿，无独立算例 | ⚠️ 与 TensorFEM 碎块面积守恒由单测钉 | pytest 33 例全绿（96 s） | L2 |
| **ice_friction_meltfilm**（#1179） | `ice_friction_meltfilm/`（4 模块 919 行） | ✅ 融膜厚度/摩擦系数/能量分配 | ❌ 无生产调用方 | ✅ sweep 表 | ✅ 扫掠 20×3 点 | ❌ C1 干分支恒等返回 μ_asp（0.0 误差）、C2 `q_in=q_solid+q_latent+q_conv` 是定义式（残差 2.2e-16）→ 不可失败（§4-F7） | pytest 15 例全绿 | L2 |
| **pid_mdof**（#1157） | `pid_mdof.py`（930 行） | ✅ 梁/板模态 + ζ_eff 曲面 → MDOF 插入损失 | ❌ 无生产调用方 | ✅ benchmark 报告 | ⚠️ benchmark 本地未跑完 | ⚠️ 文档 §6 自报两个 >3 % 项；模型 B 判据如实 FAIL | pytest 27 例（1 skip）全绿 | L2 |

### 2.1 「PR 声明 vs 代码实况」差异表

| PR/文档声明 | 代码实况 | 差异性质 |
|---|---|---|
| `docs/seepage_piping_verification.md:53/62/64`：C1「DEM-measured 5.85 %」、C2「DEM-simulated 1.04，偏差 5.0 %」、C3「9.09 %」 | `tests/test_seepage_piping.py:95` 用 `u_analytic*(1+5 % 噪声)`；`:168` 用 `i_c_analytic*1.10`；`:245` 用 `k*1.15`。文档 §5.2 自承「当前只有解析参考框架，完整 DEM 耦合是下一步」 | **声称验证完成，实为解析公式代入 + 预设数据** |
| `benchmarks/verified/dem_thermal_fracture/results.json`：`thermal_stress_relative_error=1.386e-16`、`relative_residual=0.0`；report「All acceptance criteria: PASS」 | `thermal_fracture_dem.py:271-306`（场）与 `:338+`（解析）是同一公式 `EαΔT/(1−ν)`；`:428-448` 用同一 `½σ²V/E` 算断裂能再从初值减掉 → 残差恒 0 | **判据不可能失败**；且「DEM」名不符实（一维键链） |
| `docs/wave_ice_verification_report.md`：`Status: ✅ PASS (29/29 tests)` | 测试仅断言峰值存在/有限/`<100`/单调/轴对称为 1e-10 | **PASS 被当作验证**（实为 sanity） |
| `docs/p273w3_n02_friction_polyhedron_verification.md`：`Status: ✓ PASS (all criteria met)` | 顶部有 CORRECTION 横幅（#1203），正文仍旧结论；`scripts/verify_friction_polyhedron.py:26` 仍调用旧自证模块 | **矛盾证据共存**（横幅缓解，未下架） |
| `docs/ICE_THERMAL_STRESS_COUPLING.md:1,130`：digest 作为可复核指纹 | `ice_thermal_stress/validation.py:481` 把 `generated_utc` 纳入摘要 payload，`:512-516` 算 SHA-256 → 每次跑都变（入库 `a1197a8c…` vs 本次 `3daede6c…`） | **digest 非可复现指纹**（只能查手改） |
| `docs/SHAPE_REPRESENTATION_MODULE_RESTORATION.md`：恢复 `shape_representation` 契约 | 用法是**新包遮蔽既有同名模块**，其 6 个公共名不再可 import；同模式在 `ice_structure_contact` 上仍是坏账（15 符号） | **修复引入新的同名遮蔽** |

---

## 3. 成熟度判级与分布（C）

| 级别 | 数量 | PR |
|---|---|---|
| **L5** | 0 | —— |
| **L4** | 3 | #1151、#1191、#1195 |
| **L3** | 2 | #1188、#1203 |
| **L2** | 15 | #1157、#1162、#1168、#1173、#1175、#1178、#1179、#1181、#1196、#1202、#1204、#1205、#1206、#1208、#1209 |
| **L1** | 3 | #1170、#1171、#1176 |
| **L0** | 2 | #1180、#1182 |

**分布统计**：L≥4 = 12 %（3/25）、L≥3 = 20 %、L2 = 60 %、L≤1 = 20 %。
**判级纪律**
1. L≥4 的三个都有真跑证据 + 独立参考 + 守恒/收敛中至少之一，且工件可重生成（本审计已重算 #1191/#1195，与文档数字一致）。
2. 报告规模不进判级：#1188 附 7 篇文档 + 5 张图仍只到 L3（缺收敛/守恒 + 未集成）；#1191 一篇文档到 L4。
3. L2 ≠ 无用：15 个 L2 里 7 个「真算法 + 好文档 + 单测绿」，缺的只是外部基准或集成，§2 表逐行标出缺哪一轴。

---

## 4. 主要问题（按严重度）

### F1【严重·验证诚信】seepage 的「DEM 测量」由测试自己合成（#1176）
- 证据：`tests/test_seepage_piping.py:95`、`:168`、`:245`；文档 `docs/seepage_piping_verification.md:53/62/64` 把这三处写成「DEM-measured」并给出 5.85 %/5.00 %/9.09 % 误差。
- 影响：C1–C3 的 PASS 由常数决定（改成 1.14 即 FAIL），对 `seepage/` 代码零信息量；文档 §5.2 已自承无 DEM 耦合，但正文判据表未降级。
- 与 #1203 自查出的 `atan(mu)` vs `atan(mu)` 同病 → 纠正只做个案、无机制拦截。
- 建议：C1–C3 标 `WITHHELD`（无 DEM 源）；或在 `tools/` 加静态检查「`measured_*` 不得来自被测模型的解析式」。

### F2【严重·验证诚信】thermal_fracture_dem 判据 (d)(f) 是恒等式，且不是 DEM（#1209）
- `src/tensordem/thermal_fracture_dem.py:271-306` 与 `:338+` 同一公式；能量链路 `:428-448` + `:881-930` 使残差恒 0（无断裂表面能、无位移松弛）。
- 模块内无接触力/速度/加速度/时间积分，`ParticleChain.displacements` 从未参与应力计算 → 「DEM」名不符实。
- 建议：(d) 换不同路径（热应变场积分求等效应力）；(f) 加断裂表面能与松弛步，否则标 `NO_INFORMATION`。

### F3【严重·结构】6 处模块/包同名遮蔽，2 个测试模块彻底无法收集
- 扫描：`freeze_bond`、`ice_structure_contact`、`segregation`、`sensitivity`、`shape_representation`、`verification` 共 **6 处** `X.py` 与 `X/` 并存；包优先 → 旧 `X.py` 成不可 import 死代码。
- 机检：复跑 `tools/import_contract_scan.py` → `unresolved test imports: 36 in 2 test modules`（`freeze_bond` 21、`ice_structure_contact` 15），工件 `artifacts/import_contract/broken_imports.json`。
- 复现：`python -m pytest tests/test_freeze_bond.py tests/test_ice_structure_contact.py -q` → `Interrupted: 2 errors during collection`。
- 成因（#1202 文档自述）：合并门禁「N 个文件被门禁滤除」滤掉模块、留下验收测试 → 测试不红反而消失；随后用新包补符号，撞上既有同名模块。
- 建议：`git mv X.py legacy_X.py` 并由包 `__init__` re-export，或把缺失符号补进包；把「同名遮蔽」加入 `import_contract_scan` 作门。

### F4【中·验证诚信】旧的自证验证模块仍在使用面（#1203 遗留）
- `friction_polyhedron_verification.py`（C1–C3 = `atan(mu)` vs `atan(mu)`）仍在树内，`scripts/verify_friction_polyhedron.py:26` 仍 import；对应文档正文仍 `Status: ✓ PASS`。
- 建议：删脚本或改指向 `*_real_verification`；文档正文标 WITHDRAWN。

### F5【中·证据链】22/25 个 PR 未把结果工件入库
- 仅 #1151（csv+json）、#1188（report.json+5 图）、#1209（results.json+report.md）入库结果；其余文档引用的 `artifacts/...` 在 checkout 里不存在（`artifacts/` 仅剩 `pr166/`）。
- 建议：benchmark 结果入 `benchmarks/<node>/results/`（#1188 已示范）。

### F6【中·可复现】「证据 digest」不可复现
- `ice_thermal_stress/validation.py:481` 把 `generated_utc` 纳入摘要 payload，`:512-516` 计算 → 每次变；跨架构更不同。
- 建议：`canonical_digest` 先剔除 `generated_utc`（或分 `digest_results`/`digest_all`）。

### F7【中·方法论】一批「不可能失败」的判据（同类）
| 位置 | 判据 | 为何不可能失败 |
|---|---|---|
| `ice_friction_meltfilm/physics.py:6-7` | C1 误差 0.0、C2 残差 2.2e-16 | 干分支按定义返回 `μ_asp`；`q_in=q_solid+q_latent+q_conv` 即求解 h 的方程 |
| `benchmarks/erosion_dem_verification.py:354-415` | C4 几何往返/质量-体积一致 1e-15 | 球冠体积与半径同一公式反解；质量=体积×密度 |
| `ice_thermal_stress` C5b | 线性梯度锚误差 0 | 两侧同为解析式 |
| `thermal_fracture_dem` (d)(f) | 见 F2 | 恒等式 |
| `seepage` C1–C3 | 见 F1 | 合成测量 |
- 说明：erosion_dem 的 C4 仍有「delivered vs contact mass」「efficiency ≤ 1」是真检查，故整体仍判 L4；此表按判据逐条登记无信息量项。

### F8【轻·卫生】过程垃圾入库
- #1157 并入 9 个 `.tmpwork/` 草稿（`acc_prev.json` 2,450 行、`mpl/fontlist-v390.json` 1,854 行等，共 4,331 行），后被 #1174 删除。
- 仓根仍留 18 个过程文件：`.scratch/bench*.py`、`post_report*.py`、`probe2.py`、`probe3.py`、`report_*.json`、`reports_m13798.json`（#1204）、`results-dem3d-scale/`、`IMPLEMENTATION_SUMMARY.md`（#1171）。
- 建议：`.gitignore` 增 `.tmpwork/`、`probe*.py`、`post_report*.py`、`report_*.json`；仓根文件迁 `docs/process/` 或删除。

### F9【轻】#1180 只并了一个出图脚本（211 行，无模块/测试/文档）→ L0；计入「PR 数」会虚高产能。

---

## 5. 「曾声称为功能、实为文档/测试/自证」膨胀清单

| PR | 声称为 | 实为 | 膨胀 |
|---|---|---|---|
| #1176 seepage | 「DEM 渗流-管涌闭环验证 C1–C3 PASS」 | 解析公式 + 测试合成的「测量值」 | 1,469 行 → L1 |
| #1209 thermal_fracture_dem | 「DEM thermal fracture verification，All criteria PASS」 | 一维键链 + 2 条恒等判据 | 939 行 → L2 |
| #1168 wave_ice | 「Status: ✅ PASS (29/29 tests)」 | sanity 断言（存在/有限/单调） | 1,909 行 → L2（无基准） |
| #1180 gen_energy_audit_figures | 「energy audit」 | 纯出图脚本 | 211 行 → L0 |
| #1182 IN157_N03 | 「DEM broken-ice validation」 | 文档 505 行 | L0 |
| #1170 random_ice_strength | 节点交付 | 23 行 study 增量 + 1 篇文档 | L1 |
| #1171 ice_ridge_formation | 节点交付（2 个新模块） | 无测试、无运行验证、模块无调用方 | 241 行 → L1 |
| 旧 `friction_polyhedron_verification.py`（仍在树内） | C1–C4 全 PASS | 3 条自证 + 1 条零基准能量闭合（#1203 已揭穿） | 324 行 → 应下架 |

合计约 **4,700 行（新增核心代码 16 %）被高估至少一级**；其中 2,408 行（seepage + thermal_fracture）属「验证诚信」问题而非单纯成熟度问题。

---

## 6. 值得保留的做法（审计的另一半）

1. **#1203 自我纠错**：把已合并的「全部 PASS」拆开，指出 3/4 判据无信息量，撤回并实测出 3 个内核缺陷（包围半径当穿透深度 → 自重 1.7×10⁴ 倍假接触力）。本生态最该复制的 adversarial self-review。
2. **#1188 诚实 FAIL**：C1b（3 倍带）FAIL 与 α=20.1 病态拟合写进标题级结论，并显式回答「是否 <3 %？否」。
3. **#1195 证据分级**：REAL / WITHHELD / CALIBRATED 逐项标注，标定常数出处入 `CASE_ANCHOR_PROVENANCE`。
4. **#1196/#1205/#1206 负结果登记**：breakage_pbm 记录 C2/C3/C4 FAIL；dem3d 记录 `not_run`；P152 记录被否定的测量协议。
5. **#1151 前版错误追认**：`baseline_before` 记录前版 x_f≈2× 解析值（125 % 误差）的根因（潜热积分到 L/2 + 前沿定位用单元掩码）。

---

## 7. 复现命令与证据索引

```bash
# 盘点（分类 + 调用方 + stub）
python3 tools/audit_asset_inventory.py -n 25        # → artifacts/audit_m14487/inventory.json

# 结构体检（36 个断掉符号）
python3 tools/import_contract_scan.py               # → artifacts/import_contract/broken_imports.json
python3 -m pytest tests/test_freeze_bond.py tests/test_ice_structure_contact.py -q   # 2 collection errors

# 真跑：benchmark
python3 benchmarks/ice_thermal_stress/run_validation.py          # 12/12 PASS, 2 WITHHELD
python3 benchmarks/erosion_dem_verification.py                   # C1–C7 全 PASS, 70 s
python3 benchmarks/ice_pileup_iso19906/run_benchmark.py          # C1 PASS / C1b FAIL（如实）
python3 benchmarks/verified/dem_thermal_fracture/run_verification.py   # 全 PASS（判据 (d)(f) 恒等）

# 真跑：新增测试（27/30 文件本地跑完）
python3 -m pytest tests/test_ice_thermal_stress_{properties,conduction,coupling,evidence}.py -q  # 45
python3 -m pytest tests/test_erosion_dem.py -q                    # 45
python3 -m pytest tests/test_ice_anisotropy_{fabric,slip_reference,verification,virtual_test,yield_surface}.py -q  # 45
python3 -m pytest tests/test_breakage_pbm.py -q                   # 38 passed, 3 xfailed
python3 -m pytest tests/test_fem_dem_seeding_handoff.py -q        # 33
python3 -m pytest tests/test_wave_ice_verification.py -q          # 29
python3 -m pytest tests/test_seepage_piping.py -q                 # 28（绿在合成数据上）
python3 -m pytest tests/test_thermal_profile.py -q                # 32
```

**本轮测试执行汇总**（逐文件日志见 `artifacts/audit_m14487/logs/`）：27/30 个新增测试文件本地跑完并全绿（**591 例通过、0 失败**，另有 3 xfail 与若干 skip，集中在 `test_breakage_pbm`/`test_moist_drying`/`test_pid_mdof`）；3 个文件（`polyhedra_repose_angle`、`polyhedra_repose_fast`、`friction_polyhedron_real_verification`）本地超时，转集群 `cj850`/`cj881`，**记为 not_run**。

---

## 8. 最小修复建议（按性价比排序）

1. **降级两处伪验证的证据等级**（文档 + verdicts）：seepage C1–C3 → WITHHELD；thermal_fracture (d)(f) → NO_INFORMATION。改动仅文档/元数据，零风险。
2. **修 2 个断掉的测试模块**（`freeze_bond`、`ice_structure_contact`）：补 36 个符号的 re-export，或 `git mv` 旧模块为 legacy。验收 = `pytest --collect-only` 零错误。
3. **把「同名遮蔽」与「measured_* 不得来自解析式」写进 `tools/import_contract_scan.py`**（#1202 已打开这个口子），使其成为合并门前置。
4. **digest 剔除时间戳**（`validation.py:481/512`），恢复「同结果同指纹」语义。
5. **benchmark 结果入库**：给 `benchmarks/*/results/` 白名单（#1188/#1151/#1209 已示范），使 22 个 PR 的声明可离线复核。
6. **下架旧自证模块**：`friction_polyhedron_verification.py` + `scripts/verify_friction_polyhedron.py`，文档正文标 WITHDRAWN。
7. **仓根清洁**：18 个过程文件迁 `docs/process/` 或删除，`.gitignore` 增 `.tmpwork/`、`probe*.py`、`post_report*.py`。
---

## 9. 交付索引（可审计性）

| 交付物 | 路径 |
|---|---|
| 本报告 | `audit_report.md` |
| 盘点工件（分类/调用方/stub） | `artifacts/audit_m14487/inventory.json`、`wiring.json` |
| 盘点与统计工具 | `tools/audit_asset_inventory.py`、`tools/audit_m14487_figures.py` |
| 判级落表脚本（27 条，只 INSERT） | `tools/audit_m14487_verdicts.py`（verdict_by='audit2-ab'，mission_id=14487） |
| 报告发布脚本 | `tools/audit_m14487_post.py`（平台报告 id=13629） |
| 图 | `artifacts/audit_m14487/figures/fig1..3*.png` |
| 逐文件测试日志 | `artifacts/audit_m14487/logs/*.log`、`logs/summary.tsv` |
| 真跑生成的模块工件 | `artifacts/ice_thermal_stress/`、`artifacts/p176w2-n11/`、`artifacts/dem_thermal_fracture/`、`benchmarks/verified/dem_thermal_fracture/` |
| 结构体检工件 | `artifacts/import_contract/broken_imports.json` |

**verdicts 落表结果（27 条）**：L4=3（#1151/#1191/#1195）、L3=2（#1188/#1203）、L2=15、L1=5（含 2 条 repo 级结构发现）、L0=2。
`SELECT scope, level FROM tensor_maturity_verdicts WHERE repo='TensorDEM' AND verdict_by='audit2-ab' ORDER BY level DESC;`

**未验证项（honest）**：`tests/test_polyhedra_repose_angle.py`、`tests/test_polyhedra_repose_fast.py`、
`tests/test_friction_polyhedron_real_verification.py` 三个文件本地超出单次 120 s 预算，已递交集群 `cj850`/`cj881`；
截至本报告落笔两个作业仍在排队（递交时排队 252/274），**记为 not_run，未计为通过**，对应模块判级不因之抬高。
