# AUDIT — m17451 / XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE

**锚点**: `XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE`
**审计对象**: `benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py` +
`docs/XVAL_DEM_INTERFACE_SLIP_PSEUDO_STEADY_STATE.md`（PR #1453 关闭后残留在 main 的部分）
**审计动机**: 原 PR #1453 被 owner 裁定关闭（2026-11-05），但两个文件已在 main 上，
而 mission 声明的另两个产物（`docs/AUDIT_m17451_XVAL_DEM_INTERFACE_SLIP.md`、
`post_report_m17493.py`）缺失 → 该锚点处于**半落地**状态：有脚本、有报告，没有自动化测试、
没有审计、没有报告产物，报告中的部分表格**没有生成代码**。
**审计日期**: 2026-11-13 · **审计者**: 建设公民智能体（m18659，分支 `agent/m18659`）
**判据**（先于改动声明）: ①每个报告数字都必须由脚本可复现；②对标必须有对照臂且对照臂
必须能**否定**目标（否则"全绿"无意义）；③仓库内要有 pytest 覆盖；④证据分级不得把
"文献声明"当成"本机测量"。

---

## 1. 审计前的证据分级

| 能力 | 审计前等级 | 依据 |
|---|---|---|
| `mindlin_partial_slip` 闭式模块（加载支/反转支/循环能量/粘着圆） | **REAL** | 与参考闭式对比 0~7.8e-8 |
| `contact_tangential` 引擎，`unloading="mindlin"` | **REAL** | 逐点 2.5e-15、面积 7.7e-6 对闭式 |
| `contact_tangential` 引擎，`unloading="linear"`（Cundall–Strack） | **REAL**（测量真实，但**判级错误**：脚本把它算成 FAIL，见 F3） | 面积 −23.07 % |
| 对标独立性（"闭式对闭式"） | **WITHHELD** | 两边来自同一闭式，无法排除共享错误（F2） |
| 报告 §4.1 时间步收敛表 | **WITHHELD** | `run_xval.py` 中**没有任何生成该表的代码**（F1） |
| OSS 对标（LIGGGHTS / YADE / MercuryDPM） | **WITHHELD**（文献声明，未执行） | 三个 OSS 码在本机均未安装/未运行（F6） |
| 自动化测试覆盖 | **STUB** | `benchmarks/` 下的函数名为 `test_*` 但无断言、不被 pytest 收集（F5） |
| 任意包络内多反转路径 | **NOT_SUPPORTED** | 模块只覆盖单调加载 + 从 gross-slip 反转（F7，属已声明范围） |

**审计结论（审计前）**: 核心物理（M&D 1953 伪稳态滞回）实现为 **REAL**，但**证据链不闭合**——
缺独立预言机、缺对照臂判据、缺测试、以及报告中有一张表无生成代码。

---

## 2. 审计发现与处置

| # | 严重度 | 发现 | 处置 | 验证 |
|---|---|---|---|---|
| **F1** | 高 | 报告 §4.1「时间步收敛」表的数字（7.40e-16 / 3.70e-16 / 3.70e-16）**在脚本里没有任何生成代码**，属于手写数字，违反"零手抄"纪律 | 新增 `timestep_convergence()`：固定每段位移 = δ_s，令 dt×0.5/1/2 对应 800/400/200 个增量（真正的**时间离散**加密，而非改采样密度），输出面积误差与逐点误差；报告表改为机器生成值 | `test_dt_refinement_converges`：面积误差随 dt 单调、每次加倍放大约 4 倍（一阶） |
| **F2** | 高 | 所谓"闭式对解析"是**循环论证**：解析式就写在同一函数里，与模块用的是同一个公式 | 新增独立预言机 `oracle_stick_circle_ode()`：只用 Mindlin–Deresiewicz 的**粘着圆律** `c/a=(1−F_t/μF_n)^{1/3}` 与增量刚度 `dF_t/dδ = k_t·c/a` 做 RK4 积分，**不含**任何 `F_t(δ)` 闭式 | 模块 vs ODE：加载支 **1.44e-9**、反转支 **2.26e-8**（占 μF_n），公差 5e-3；`test_oracle_ode_reproduces_both_branches`、`test_three_way_agreement_with_test_local_closed_form` |
| **F3** | 高 | 脚本把 `linear` 臂的 29.3 % 偏差写成 `FAIL`，于是总判定恒为 `SOME FAILURES`：**对照臂失败**被误读成**目标失败**，且"全绿"与"全红"都无法区分"量具失效" | 引入**预声明**设计：`mindlin`=target 臂（逐点 1e-6、面积 1e-3），`linear`=negative control（必须突破 owner 15 % 闸门）。`verdict()` 只有在 target 通过**且**对照臂确实失败时才给 `ALL PASS`；对照臂通过 → `INCONCLUSIVE` | `test_verdict_is_control_aware`（构造"量具失灵"输入，断言不得报 ALL PASS）、`test_negative_control_breaches_the_owner_gate` |
| **F4** | 中 | 模块级 `OUT_DIR.mkdir()` 副作用：**import 即建目录**；且脚本写到仓库根 `artifacts/`，而 main 上受版本控制的工件在 `benchmarks/artifacts/...`，报告工件表又写 `artifacts/...` —— 三处不一致，提交的 results.json 事实上不再由脚本再生成 | 输出目录惰性化：`resolve_out_dir()`/`ensure_dirs()`，默认 `benchmarks/artifacts/xval_interface_slip_pseudo_steady_state`，可用 `XVAL_OUT_DIR` 或 `--out-dir` 覆盖；工件内记录**仓库相对路径**，不再嵌入 `$HOME` | `test_module_level_has_no_side_effects`（AST 检查模块级无 mkdir）、`test_import_does_not_create_the_artefact_dir`；提交的 results.json 由脚本重生成 |
| **F5** | 中 | 该锚点**零自动化测试**：`benchmarks/` 目录不在 pytest 收集范围内，函数名虽为 `test_*` 却只 `print`，任何回归都不报警 | 新增 `tests/test_xval_interface_slip_pseudo_steady_state.py`（14 个用例）：ODE 预言机、三方一致、能量两恒等式、target/control 两臂、dt 收敛、三条标度律、判定逻辑、**提交工件防漂移闸门**（committed `results.json` 必须与一次全新运行一致） | `pytest tests/test_xval_interface_slip_pseudo_steady_state.py -q` → **14 passed** |
| **F6** | 中 | 报告 §3.3 把 LIGGGHTS/YADE/MercuryDPM 的"误差 0 % / ~23 %"与**本机测量行**并列成同一张对标表，读者会误认为本机跑过这三个 OSS 码 | 结果 JSON 新增 `oss_evidence_grade` 并对每条 OSS 记录加 `evidence` 字段；报告 §3.3 标题与表注显式标注"文献声明，未在本 harness 执行"，并把"可执行的对照"改为本仓 `linear` 臂（它复现了 LIGGGHTS 风格的能量亏欠） | 报告表注 + `results.json["oss_evidence_grade"]` |
| **F7** | 低（范围） | 包络内部任意点的多反转需要带运动学锚点的状态机；模块只覆盖单调加载与从 gross-slip 的反转，引擎在包络内反转退化为线性预测器 | **不修**（明确降级登记为 `NOT_SUPPORTED`），在报告 §6 与本文登记，避免下游误用 | 报告 §6 局限 1 |
| **F8** | 低（卫生） | `json.dump(..., default=str)` 会把 `numpy.bool_` 静默写成字符串 `"True"`，机读工件失去布尔类型 | 数值/布尔显式 `float()`/`bool()` 转换，新增 ODE 腿不再产生 numpy 标量 | `test_committed_artefact_is_not_stale` 直接断言 `old["oracle_ode"]["pass"] is True` |

---

## 3. 审计后的证据分级（本次交付）

| 能力 | 等级 | 证据 |
|---|---|---|
| 加载支 `F_t = μF_n[1−(1−δ/δ_s)^{3/2}]` | **REAL** | 闭式 1e-12；独立 ODE **1.44e-9** |
| 反转支 `F_t = F*−2μF_n[1−(1−Δ/2δ_s)^{3/2}]` | **REAL** | 闭式 1e-12；独立 ODE **2.26e-8** |
| 稳态循环能量 `W = 0.8μF_nδ_s = 1.2(μF_n)²/k_t` | **REAL** | 两恒等式互等 <1e-12；求积 7.8e-8；引擎 mindlin **7.7e-6** |
| 粘着圆 `c/a = (1−F_t/μF_n)^{1/3}` | **REAL** | 1e-12 |
| 引擎 `unloading="mindlin"` 整循环 | **REAL** | 逐点 **2.5e-15** μF_n、面积 **7.7e-6** |
| 引擎 `unloading="linear"`（negative control） | **REAL（对照臂）** | 面积 **−23.07 %**（文献声明 ≈ −23 %，本机复现） |
| dt 收敛 | **REAL** | 面积误差 1.94e-6 / 7.73e-6 / 3.08e-5（dt×0.5/1/2，一阶） |
| 标度律 δ_s ∝ μ¹、F_n^{2/3}、E^{−2/3} | **REAL** | 1.9e-16 / 2.0e-16 / 2.5e-16；对数斜率反拟合 <1e-9 |
| OSS 三方对标 | **WITHHELD（文献声明，未执行）** | `oss_evidence_grade` 字段显式标注 |
| 包络内任意多反转 | **NOT_SUPPORTED** | 已登记局限 |

**总结论**: 交付后该锚点的**可执行证据链闭合**——目标臂过闸、对照臂确实否定目标（量具有效）、
独立预言机（不同方程）复核、报告每个数字都有生成代码且有防漂移闸门。

---

## 4. 复现

```bash
cd <repo>
PYTHONPATH=src python benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py
#   工件: benchmarks/artifacts/xval_interface_slip_pseudo_steady_state/results.json + figures/*.png
#   退出码: 0 = ALL PASS（含"对照臂确实失败"），1 = 否则
pytest tests/test_xval_interface_slip_pseudo_steady_state.py -q
```

**环境**: python 3.11.6 / torch 2.7.1 / numpy 2.4.6 / CPU 单进程（见 `results.json["environment"]`）。

---

## 5. 审计残留（未解决，如实登记）

1. **1953 原文表格未数字化**：对标对象是**闭式公式**（Johnson 1985 §4.3–4.4 转载），
   不是原文图表数字化值；对原文图表的独立复核仍缺（无网络/无原文扫描件归仓）。
2. **三维/多球链未覆盖**：本算例是单接触对（冰类 E=9 GPa 球 vs 刚性平面），
   未验证多接触网络中的累积效应。
3. **OSS 未执行**：三个 OSS 码仍无本机可执行对标（需 HPC 环境装 LIGGGHTS/YADE），
   当前只能引用文献值并用 `linear` 臂间接复现其能量特征。
4. **本轮受环境限制**：审计过程中宿主磁盘一度 100 % 满（`/tmp` 不可用），本地测试改用
   `TMPDIR=/dev/shm/...` 运行；仓库全量测试递交集群作业（`cj2004`）执行。