# AUDIT m17441 — 交差核验：m17362 `XVAL-DEM-BIAXIAL-CELL-NEGATIVE-RESULT`（TensorDEM）

**核验对象**：mission m17362（status=succeeded，`finished_status=succeeded`，`decided_by=eco_admin`，
`result.evidence.server_quality_audit = {"gate_valid": false, "key_present": false}`），
交付分支 `agent/m17362` commit `bf1ebbdf`，文档 `docs/XVAL_DEM_BIAXIAL_CELL_NEGATIVE_RESULT.md`（214 行），
工件 `artifacts/biaxial_cell_xval/`（5 个 `.npz` + 5 个 `_analysis.json` + `estimator_xval.json` + 1 png）。

**本单性质**：交差核验（非重做）。**基准裁判**=平台台账 `tensor_benchmark_cases` /
`tensorbench_results` + 交付工件里的 ref/got 数值；报告好看不算数。

**核验结论（honest, mixed）**

| 项 | 结论 |
|---|---|
| 三件套（run_artifacts / ref-vs-got 数值 / repro_cmd） | **真存在且可执行**：`synthetic` 与 `analyze` 两个子命令重跑**逐位复现**交付值（最大行差 <1e-9 deg；峰值应力字段最大差 <1e-6 Pa）；37 个测试重跑 37 passed |
| 平台基准台账 | **未登记**：`tensor_benchmark_cases` 该 anchor 0 行、`tensorbench_results` mission 17362 0 行 → "基准裁判"无行可裁，只有仓内预声明容差（`XVAL_TOLERANCE=0.15`）与工件数值 |
| 主仓集成 | **未集成**：TensorDEM `main`（`27b41aaa`）缺 3 个新文件（`src/tensordem/biaxial_xval.py`、`benchmarks/biaxial_cell_xval/run_xval.py`、`tests/test_biaxial_xval.py`），分支**未推远端**（`refs/remotes` 无 `agent/m17362`）→ 任何 main 用户调用不到 |
| 报告六段 | **六段齐全**（①问题+ref 出处 ②方法与实现 ③参数表含 I=0.0072 ④ref vs got 对照表 ⑤收敛/敏感性 ≥2 点 ⑥失败模式+工件），但**有两处决定性叙述与其自身工件矛盾**（下 §⑥-A/B） |
| 判级 | **L2**（真代码 输入→计算→输出，可复现、有测试）；L3 未达（未进 main/无调用方），L4 未达（无台账登记 + 关键前提被自身工件否定） |

---

## ① 案例与核验对象（物理背景 / 几何 / 参数）

平面应变双轴压缩：顶板位移驱动、底板固定、两侧伺服恒围压 `σ3`；冻结粘结圆盘试样偏应力峰后局部化出剪切带，
其相对 `σ3`（水平）的倾角 `θ` 是经典验证量（Mohr–Coulomb `45+φ/2`、Roscoe 1970、Arthur et al. 1977）；
案例锚 `θ_ref = 60°`（dense Hostun sand 55–60° 区间中值，`docs/BIAXIAL_SHEAR_BAND.md` l.30–31，Desrues & Viggiani 2004）。
案例现行判决是 **honest FAIL**（`docs/CASE_BIAXIAL_SHEAR_BAND_R2.md` l.9–11、l.97、l.105：主 resolved 跑 55.75°/7.1 % ✓、
p=200 配对重测 72.0°/20 % ✗）。

m17362 的交付不是重算物理量，而是**对"测量层"做交叉对标（XVAL）**：①带取向算子（本仓 Radon 投影聚焦 vs OSS 结构张量
`skimage.feature.structure_tensor` 构造 vs 二阶矩 PCA）；②应力定义（本仓墙面反力 `F/A` vs OSS Love–Weber virial `(1/V)Σ f⊗l`）。

| 量 | 值（来自交付工件 `meta`/参数表） |
|---|---|
| 试样 | 24×48 点阵，1128 盘，r=0.75 m，厚 1 m，jitter 3 %，冻结键 0.35 |
| 围压 / 摩擦 | σ3=100 kPa；μ=0.55（A/B：`wall_friction` 0.55 vs 0.00） |
| 应变率 / dt / 惯性数 I | 0.05 1/s；dt=4.97e-4 s；**I=0.0072**（准静态） |
| 计量窗 | 四壁内缩 2 倍粒径（35.72 × 62.55 → 30.0 × 56.6 m² 量级） |
| 判据 | 相对误差 ≤15 %（owner 精度标准，`XVAL_TOLERANCE` 预声明于模块常量）；resolved=≥3 快照且散度<10° |

## ② 核验方法（本审计怎么做的）

一条可重跑脚本 `scripts/audit_m17441_verify.py`（七个检查，全部机器判定；输出
`artifacts/AUDIT_m17441/audit_findings.json` + `audit_findings.png`）：

| 检查 | 内容 |
|---|---|
| C1 集成 | `git cat-file -e agent/m17362:<f>` 与 `main:<f>` 对照 + `refs/remotes` 是否有该分支 |
| C2 复现（合成） | 重跑 `run_xval.py synthetic`，与交付 `estimator_xval.json` 逐行比对（injected/ours/tensor/moment + worst） |
| C3 复现（真实场） | 对 5 个 `.npz` 重跑 `run_xval.py analyze`，与交付 `_analysis.json` 的 `peak` / `stress_audit.at_peak` 逐字段比对 |
| C4 文档-数据一致性 | 文档 §4.3 的 Δσ_v/Δσ_h 与 JSON `at_peak.sigma_*_rel` 对照（下表 T2 即此结果） |
| C5 峰判据复核 | 从 `npz` 的 `strains/sigma_v/sigma_h` 重算平滑偏应力峰（平滑窗 51/201/501/1001 四档），核文档 §⑥ "ε_a≤0.12 内没有偏应力峰" |
| C6 resolved 复核 | 用**生产代码自己的判据**（`biaxial_shear_band.py:561` `tail_start = peak + 0.4·(final-peak)`，且 ≥3 快照、散度<10°）逐臂重算 θ̄ 与 vs 60° 误差 |
| C7 台账 | `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug=…` / `tensorbench_results WHERE mission_id=17362` |

复跑命令（审计侧，不写回原单工作树）：

    cp -r /root/tp-agent/work/m17362/{src,benchmarks,tests,docs,artifacts} /tmp/xvalcheck/
    cd /tmp/xvalcheck && PYTHONPATH=src python3 -m pytest tests/test_biaxial_xval.py \
        tests/test_biaxial_shear_band.py tests/test_biaxial_band_diagnostics.py -q        # 37 passed, 18.4 s
    PYTHONPATH=src python3 benchmarks/biaxial_cell_xval/run_xval.py synthetic             # 与交付值逐位一致
    PYTHONPATH=src python3 benchmarks/biaxial_cell_xval/run_xval.py analyze \
        --inp artifacts/biaxial_cell_xval/case_p100_rough.npz --out /tmp/re.json          # peak 字段逐位一致

## ③ 计算结果（全部由脚本从工件重算，无手抄）

<!-- BEGIN AUTOGEN:tables (scripts/audit_m17441_verify.py) -->

### T1 估计器交叉验证（重跑 vs 交付值，按真值/互差）

| 量 | 交付 (estimator_xval.json) | 重跑 (本次审计) | 差 |
|---|---|---|---|
| 本仓 Radon vs 注入真值 | 3.33 % | 3.33 % | 0.00e+00 |
| OSS 结构张量 vs 注入真值 | 4.54 % | 4.54 % | 0.00e+00 |
| OSS 二阶矩 vs 注入真值 | 16.68 % | 16.68 % | 0.00e+00 |
| 本仓 vs 结构张量 | 8.20 % | 8.20 % | 0.00e+00 |

复现判定：**REPRODUCED (bit-identical)**（行级最大差 0.0e+00 deg）。

### T2 应力定义审计（墙面反力 vs Love–Weber virial，峰值步）

| arm | 墙摩擦 | 固结步 | Δσ_v | Δσ_h | σ_v 判定 | σ_h 判定 |
|---|---|---|---|---|---|---|
| long | 0.55 | 4000 | +20.44 % | +52.92 % | ✗ | ✗ |
| rough | 0.55 | 1200 | +24.02 % | +80.43 % | ✗ | ✗ |
| rough_long | 0.55 | 1200 | +25.13 % | +44.55 % | ✗ | ✗ |
| settled | 0.55 | 4000 | +26.47 % | +55.44 % | ✗ | ✗ |
| smooth | 0.00 | 1200 | -0.44 % | +12.44 % | ✗ | ✓ |

重跑 vs 交付逐字段最大差 0.0e+00 Pa（**BIT_IDENTICAL**）。

### T3 偏应力峰与案例自身 resolved 判据（由 npz 重算）

| arm | 平滑峰 ε_a | 峰后记录占比 | 峰后 40% 窗起点 | 窗内快照数 | 本仓 θ̄ [°]
|---|---|---|---|---|---|
| long | 0.1100 | 8.3 % | 0.1140 | 1 | 53.00 ± nan |
| rough | 0.0851 | 5.5 % | 0.0871 | 1 | 57.00 ± nan |
| rough_long | 0.0851 | 29.1 % | 0.0991 | 3 | 54.33 ± 1.25 |
| settled | 0.0513 | 14.6 % | 0.0548 | 2 | 28.50 ± 28.50 |
| smooth | 0.0590 | 34.5 % | 0.0714 | 2 | 0.00 ± 0.00 |

resolved by 案例自身判据（≥3 快照且散度<10°）：**case_p100_rough_long**。

<!-- END AUTOGEN -->

补充（同样来自工件，未见文档引用）：

* `case_p100_smooth` 的两个峰后快照上，本仓 Radon 返回 **`ours_deg = 0.0`**（`eta2`=0.31/0.27、`z_score`=2.12/3.01），
  而 OSS 结构张量给 60.9°/59.2°（`tensor_spread_deg`=33.8°/26.1°）→ 0.0 是**未局部化**情形下的无意义 argmax，
  但 `radon_band_angle`（`src/tensordem/biaxial_shear_band.py:245–264`）无显著性门槛，`measure_band` 只挡"应变场退化"，
  **不挡"扫描无峰"** → 该臂的 in-house 读数被记为合法的 `0.0°`（等价于"水平带"），失败与真实结果不可区分。
* 文档摘要把轴向差写成 "σ_v 仍 +-0 % ✗"，而 §4.3 表格与 JSON 都是 **-0.4 %（在 15 % 内 → ✓）**；同理
  §4.3 正文 "轴向在两臂都超差（+20–94 %）" 的 94 % 既不是峰值差（+24.0/+20.4 %）也不是全臂范围，
  而是 `case_p100_rough` 的**全过程均值** `sigma_v.mean=0.942`（`case_p100_rough_analysis.json`）——
  把"峰值口径"和"全过程均值口径"混写在同一句，且未标注口径。

## ④ 结果图

`artifacts/AUDIT_m17441/audit_findings.png`（matplotlib，由脚本生成）：

* 左：5 个臂的 `q = σ_v − σ_h`（墙面口径） vs 轴向应变，黑点=平滑偏应力峰，虚竖线=峰后 40 % 窗起点 →
  直观显示**每个臂都有峰**，且 `rough_long` 臂峰后有 29 % 的记录长度（3 个快照）。
* 右：合成 10 角扫描的三族算子相对误差曲线 + 15 % 容差线 + 红线=`rough_long` 臂 resolved 读数 54.33°（9.4 %）。

## ⑤ 注意点（踩到的坑与适用范围）

1. **分支≠交付**：m17362 的 `result.evidence.branch = null`、`head_commit = null`，
   `server_quality_audit.gate_valid = false` → 平台的自动推/合管线拿不到分支名，交付停在本地分支 `agent/m17362`（未推远端）。
   这是本单 L3 不达的**机制原因**，也解释了为何 `main` 上 `grep biaxial_xval` 零命中。
2. **基准台账是"唯一裁判"，但本 anchor 无行**：`tensor_benchmark_cases` 全库 53 行，`XVAL-*` 只有 `XVAL-FEM-TPB1976` 一行；
   m17362 的 ref/got 只活在工件 JSON 里 → 机器裁决无法发生。升级路径明确：把 T1/T2 的 ref/got 注册进 `tensor_benchmark_cases`
   （anchor=本单 slug，metric=`estimator_rel_error`/`sigma_v_rel`/`sigma_h_rel`，tolerance=0.15），并把跑批写 `tensorbench_results`。
3. **"没有峰"是错误前提**（最重要）：文档 §⑥ 用"3 个重跑臂在 ε_a≤0.12 内都没有偏应力峰"支撑"resolved 不成立 → 不改判负结果。
   实际 `npz` 在平滑窗 51/201/501/1001 四档下**每臂都有内部峰**（峰后剩余记录 5.5 %–38 %）；
   `case_p100_rough_long` 在**生产代码自己的判据**下 resolved（3 快照，θ̄=54.33°±1.25°，散度 1.25°<10°），
   对 60° 锚误差 **−9.4 %（<15 % ✓）**；累计场 58.67°±0.47°（−2.2 %）。
   该臂的 `.npz`/`_analysis.json` 存在于交付工件里，却**不在文档 §4.2/§4.4 的表里、不在 §⑥ 工件清单里**（清单只列 4 个臂，实际 5 个）。
   影响面：不改写任何已报数字（T1/T2 全部复现），但**改写了"本单无 resolved 重跑"这一结论**，
   并使"负结果未被推翻"的论证失效——负结果本身仍成立，理由应改为"p=200 臂（72.0°/20 %✗）未重跑，p100 臂全部 <15 %✓"。
4. **文档错字/排版**：§4.1 表头 7 列、数据行 9 个单元格（ragged），渲染会错位；数值本身与 `estimator_xval.json` 一致；
   该表还缺"二阶矩 vs 真值"列（16.7 % ✗ 只出现在正文）。
5. **口径混写**：见 ③ 末（`+-0 %` 与 `20–94 %`）。另 §"误差判定"把"真实场算子互证 0.5 %"写成 <3 % 达成，
   这是取了 `rough` 臂（0.54 %），同期 `long` 臂是 **4.08 %**（>3 %）→ 属选择性引用，应写"0.5 %–4.1 %"。
6. **环境坑（可信度加分项）**：交付如实记录 loongarch64 的 scikit-image 与 numpy 2.4 ABI 冲突 → 用 `scipy.ndimage`
   复现同一算法且**不宣称直接执行 skimage**；集群排 577 作业 → 本地单线程 torch 86 步/s。这两条我复核为真
   （本审计环境 `numpy 2.4.6`，`synthetic`/`analyze` 均不依赖 skimage）。
7. 适用范围：XVAL 只回答"测量层信不信得过"，不回答"物理模型准不准"。即便测量层全绿，案例对实验锚仍无 <3 % 的物理量
   （文档 §"误差判定"已如实声明，这点是对的）。

## ⑥-A 决定性叙述与工件冲突（缺陷清单，按严重度）

| # | 严重度 | 缺陷 | 证据指针 | 影响 |
|---|---|---|---|---|
| 1 | 高 | "ε_a≤0.12 内没有偏应力峰 → 无峰后窗口 → unresolved" 与工件矛盾；且遗漏 resolved 臂 | `case_p100_rough_long.npz`（峰 ε≈0.0851，记录到 0.120）、`case_p100_rough_long_analysis.json`（`tail_start=0.0991`、3 行、`resolved.ours_deg=true`、`incremental.ours_deg=54.33±1.25`）；文档 §⑥ 段落 | 结论叙述失效；完整性问题（选择性报告） |
| 2 | 高 | 交付未集成（分支未推、main 缺文件），且 mission evidence 未报 `branch`/`head_commit` | `git cat-file -e main:src/tensordem/biaxial_xval.py` → 不存在；`tensor_dev_missions.result.evidence.branch=null` | 能力不可用；L3 不达 |
| 3 | 中 | 基准台账 0 行（cases/results 皆无） | C7 结果；`tensor_benchmark_cases` 全库 53 行 | 机器裁决缺位；L4 不达 |
| 4 | 中 | 估计器无显著性门槛，未局部化时静默返回 `0.0°` | `src/tensordem/biaxial_shear_band.py:245–264`；`case_p100_smooth_analysis.json` rows（`ours_deg: 0.0`, `z_score` 2.1/3.0） | 未来运行可能静默污染角度读数 |
| 5 | 低 | 摘要/正文口径混写（`+-0 %`、`20–94 %`）、§4.1 表 ragged、<3 % 结论选择性引用 0.5 % 而隐去 4.08 % | 文档 §摘要、§4.1、§4.3 正文、§"误差判定" | 可读性/夸大，不影响已报数值 |

## ⑥-B 误差判定（对判据显式回答）

| 关键量 | 基准 | got | 误差 | <3 % ? | <15 % ? |
|---|---|---|---|---|---|
| 结构张量 vs 注入真值（合成最差） | 合成真值 | 4.54 % | 4.54 % | 否 | **是** |
| 本仓 Radon vs 注入真值（合成最差） | 合成真值 | 3.33 % | 3.33 % | 否 | **是** |
| 本仓 vs 结构张量（合成最差） | 结构张量 | 8.20 % | 8.20 % | 否 | **是** |
| 本仓 vs 结构张量（真实场单快照） | 结构张量 | 0.54 %（rough）/ 4.08 %（long） | 同上 | 部分（仅 rough） | **是** |
| Δσ_h（无摩擦侧墙 A/B） | virial | +12.44 % | 12.44 % | 否 | **是** |
| Δσ_h（生产墙 rough） | virial | +80.43 % | 80.43 % | 否 | 否 |
| Δσ_v（三臂 peak） | virial | +24.02 / −0.44 / +20.44 % | 同上 | 否 | 两臂否、一臂是 |
| 带倾角 vs 60° 锚（resolved 臂 rough_long） | Desrues & Viggiani 2004 60° | 54.33° | **−9.44 %** | 否 | **是** |
| 带倾角 vs 60° 锚（未达 ρresolved 的单快照） | 同上 | 57.0° / 53.0° | 5.0 / 11.7 % | 否 | 是 |

**是否 <3 %**：审计侧没有任何物理量对实验锚 <3 %（最接近 9.44 %）；交付内部 <3 % 的只有"两族算子在真实场互证 0.54 %"（单臂）
与解析 virial 恒等式（1e-9）。**故本单不宣称物理精度提升**——这一点交付文档的自我声明是诚实的，审计予以确认。

## ⑦ 判级与处置

* **L2（confirmed）**：`impl_files` 真代码（`biaxial_xval.py` 387 行：结构张量/二阶矩/合成交叉验证 + `ridge_dem.py` virial 分支 +
  `biaxial_shear_band.py` 新字段/`wall_friction` 开关），`entry`=`benchmarks/biaxial_cell_xval/run_xval.py`（synthetic/case/analyze 三子命令），
  `run_artifacts` 可逐位复现，`tests` 9 passed（全套 37 passed）。
* **L3 不达**：main 无文件、无调用方、分支未推远端（缺陷 #2）。
* **L4 不达**：台账 0 行（缺陷 #3）+ 决定性叙述被工件否定（缺陷 #1）→ 不满足"完整算例 + 误差分析 + 可复现 + 判据自洽"。
* **返修建议（最小集）**：①把 `branch`/`head_commit` 回写 mission evidence 让管线合并；
  ②把 T1/T2 的 ref/got 注册进 `tensor_benchmark_cases`/`tensorbench_results`；
  ③文档 §⑥ 改口径（峰存在，`rough_long` 臂 resolved 9.44 %✓；负结果成立的正确理由是 p200 臂未重跑）；
  ④`radon_band_angle` 加显著性门槛（`z_score < 3` → `BandMeasurementError`）后再重跑 smooth 臂。

**独立性声明**：审计者只读原单工作树 `/root/tp-agent/work/m17362`（并在 `/tmp/xvalcheck` 副本上执行，不写回），
未修改交付任何文件；所有判定均由 `scripts/audit_m17441_verify.py` 机器产生并落 `artifacts/AUDIT_m17441/audit_findings.json`。
**未跑项**：`run_xval.py case`（重新步进 DEM，单臂数十秒–分钟；本次用交付 `.npz` 做离线复测，未重跑步进本身）——
故"步进路径可重放"记为 not_run，不作为通过项。
---

## ⑧ 落表与裁定（机器结果）

* `INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
  VALUES ('TensorDEM','m17362 XVAL-DEM-BIAXIAL-CELL-NEGATIVE-RESULT',17362,2,<evidence JSON>,'board-audit')`
  → **verdict#565**，`evidence` 八键齐备（impl_files/entry/run_artifacts/tests/integration_callers[空=L3 未达]/benchmark
  [ref/got/tolerance=0.15，并注明台账未登记]/repro_cmd/external_anchor）。
* `maturity_gate.gate_once()` 机器复核：**gate_level=2, gate_state=confirmed**（首轮因 evidence.entry 带行号后缀
  `:337-400` 被 `files_exist()` 判"工件缺失"→ 误降 L1；改为纯路径后重跑确认 L2，误开的返修单 m17477 已 revoked 并留原因）。
  **可复用知识（写给后续同事）**：gate 的 `files_exist()` 对 `entry/run_artifacts/impl_files` **不做 `文件:行号` 拆分**，
  evidence 里只能放纯路径；行号请写进 `benchmark`/说明字段，否则会被误判缺失并触发无意义返修单。
