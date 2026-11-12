# 交差核验（board cross-audit）— m16065 CASE-DEM-MINDLIN-TANGENTIAL（TensorDEM）

- 审计单：m16127 ｜ 被审单：m16065（status=succeeded）｜ 被审 commit：`14470b0e`（`agent/m16065`）
- 锚点：`CASE-DEM-MINDLIN-TANGENTIAL`（切向 Mindlin 微滑移→完全滑移过渡）
- 裁判原则：基准判据是唯一裁判；报告好看不算数。本审计**不重做**算例，只做取证、重算、判级。
- 独立性声明：本审计的重算脚本 `scripts/audit/m16065_independent_check.py`
  **不 import TensorDEM**，全部参考量由 P/R/E/ν/μ 用 Hertz–Mindlin 闭式 + scipy 自适应求积独立重建；
  从被审树只取 `curves.npz` 与 `results.json` 作为"交付曲线/自报数字"。

## ① 三件套取证（run_artifacts / ref-vs-got / repro_cmd）

| 项 | 结论 | 证据指针 |
|---|---|---|
| run_artifacts 真实存在 | ✅ | `benchmarks/case_mindlin_tangential/{results.json(9113 B), curves.npz(49518 B)}` + `figures/fig1..fig5.png`（92069/61072/45354/94418/76020 B），全部由 `14470b0e` git 跟踪（非 .gitignore 的 runs/） |
| 图真的进了平台 | ✅（字节级） | `/root/tensor-ecosystem/TensorPlatform/backend/uploads/tensor_reports/15188/*.png` 与仓内 5 张图 **sha256 逐一相同**（fig1 53d2ddf6…、fig2 44d2e6d3…、fig3 7ca85947…、fig4 646dabd6…、fig5 09cba1a8…） |
| ref vs got 数值 | ✅ 可复算 | 见 ③（微滑移 1.5038e-5·μP、平台 3.000000 N=μP、过渡点 0.2533 %、R1-vs-R2 9.0e-12） |
| repro_cmd 可执行 | ✅ | `PYTHONPATH=src python3 benchmarks/case_mindlin_tangential/run.py` → exit 0, 56.3 s；`… validate.py` → exit 0, `VERDICT: PASS (8/8 checks)`；`pytest tests/test_case_mindlin_tangential.py` → 7 passed / 8.49 s（在 **/tmp 干净副本**上重跑） |
| 重跑一致性 | ✅ 逐比特（除 wall clock） | 副本 `results.json` 与交付件 **0 个叶子差**（浮点 1e-12 相对判等，跳过 wall_clock 类键） |
| 报告六段 | ✅ | `tensor_reports id=15188`，①–⑥ 齐全，③ 数字表由 `post_report.py` 从 results.json 程序化注入（无手抄）；④ 5 图 caption 齐全 |

## ② 基准裁判（唯一裁判）

`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-MINDLIN-TANGENTIAL'`
→ **0 行**（2026-11-12 审计时；全表仅 25 行）。即：该 anchor **没有机器裁判登记**。
本轮判据来源只能是（a）使命 goal 文本"<5%，指标=切向力-切向位移曲线（微滑移段+宏观滑移段）vs Mindlin 解析"，
（b）`run.py` 预登记 `TOLERANCE=0.05` / `validate.py:21 TOL=0.05`。两条一致 → 按 <5 % 判，**但登记面缺口记为 F3**。

## ③ 独立重算（我的数，不是复读报告）

| 量 | 交付/自报 | 本审计独立重算 | 判 |
|---|---|---|---|
| E*、R*、a、p0、k_t0、δ* | 3.84615e10 / 2.5e-3 / 7.870305e-5 / 7.70829e8 / 4.985713e6 / 9.025790e-7 | 逐项 rel_diff **0.0**（Hertz 闭式 + G*=E/[4(1+ν)(2−ν)]） | ✅ |
| R1 闭式 vs 独立自适应求积 `F=4G*∫₀ᵃmin(δ,μp/(4G*))dx` | 自报 R1-vs-R2 9.0e-12 | 独立 quad：max dev **5.3e-7·μP**（受我 epsrel=1e-6 限制） | ✅ |
| G1 微滑移最大偏差 | 1.5038e-5（记为 "rel_err"） | **1.5038e-5·μP**（逐位一致）；换算真相对误差 max **1.573e-5** | ✅ <5 % |
| 宏观滑移平台 | 3.000000 N | 3.000000 N，rel 0.0 | ✅ |
| 过渡点 δ(0.99μP) | 0.2533 % | 反解 s99=1−0.01^{2/3} 一致 | ✅ |
| 线性 Cundall–Strack 判别 | 19.11 % | 同 k_t0 线性律确实 >5 % → 算例**有判别力** | ✅ |
| G3 反向支边界 F1 | 1.1713（记作 "117.13 %"）、完全反向 0.6065δ*（自报 0.55δ*） | 独立反演支：dev **1.171324·μP**（自报 1.171326，差 2e-6），完全反向解析 **2δ***，实测 0.6065δ* | ✅ 结论成立，**标签口径需修正**（见 F2） |

## ④ 判级（L0–L5）

**L4（基准验证）**。功能实质：真代码（`mindlin_tangential_step` 有状态、锚点回弹、滑移掩码）
→ 已在生产算例路径上被调用（`protocols.sandpile` 默认 `tangential_model="mindlin_history"`, `backbone="mindlin"`）
→ 完整算例 + 逐项误差表 + Δt 收敛（一阶：2.63e-4/1.34e-4/6.90e-5）+ μ/P 阶梯（δ*∝μ 精确、δ*∝P^0.66667）
→ 干净副本重跑**逐比特复现**。
**不给 L5**：无新物理，反向支（真正含历史依赖的部分）未实现，只是把该缺口量化。
**级别完整性说明（不粉饰）**：算例唯一"非循环"定量门是 G2（1-DOF 动力学，6.6e-5）与过渡点反解 0.2533 %；
G1 微滑移段与 R1 是**同一闭式的代数一致性检查**（交付方自己在 REPORT ⑤.2 声明），故本 L4 的含义是
"交付件与规范闭式一致 + 动力学集成正确 + 判别力存在 + 可复现"，**不是**"两套独立物理模型对照"。

## ⑤ 发现（按严重度）

- **F1（blocking 于"能力完整性"，非算例门）反向支不可复现**：库内 backbone 是 |δ| 包络，反向时边缘弹簧先翻转的渐进
  部分滑移支表达不出来；独立反演支核验 dev=1.171·μP、完全反向 0.6065δ* vs 解析 2δ*（−69.67 %）。
  交付方**主动申报**为能力边界（REPORT ⑤.1 + `docs/hertz_mindlin_contact.md`），未粉饰 → 不算造假；但该缺口仍是
  Mindlin–Deresiewicz 全模型的真实缺件，应立返修单（变法向力分支亦然）。**disposition: reopen 子项，不影响本算例消账。**
- **F2（minor，口径）`*_rel_err` 字段名不实**：`curve_errors()`/`run_reversal()` 里 `rel = |ΔF|/(μP)`（run.py:356、:324），
  是**以 μP 归一化的绝对偏差**，不是相对误差。影响：微滑移段真相对误差 1.573e-5（仍≪5 %，结论不变）；
  反向支"117.13 %"若读作相对误差会**低估**（我算真相对误差量级 ~2.7e2）。建议改名 `*_dev_over_muP` 或在 REPORT 加口径脚注。
  另：commit message 写"full reversal at 0.55 δ*"，results.json 实测 0.6065δ* —— 两处不一致，取 results.json。
- **F3（registry）该 anchor 在 `tensor_benchmark_cases` 0 行**：算例池铺开的登记面缺件，机器裁判无处可依。
  本审计不代填该表（避免自证），只登记。**平台/池侧应补登 metric/tolerance/ref_source。**
- **F4（minor，卫生）遗留占位报告 id=15187「probe test title」(markdown='xxxxxxxx')** 仍为 published；
  交付方在 result.checks 里已如实披露且无删除权限。建议平台增作废/软删接口。
- **F5（平台部署，非本单责任）图端点 404**：`GET /tensor/reports/15188/figures/*.png` 全 404，但图字节确实落在
  `backend/uploads/tensor_reports/15188/`（与仓内 sha256 相同）。服务 cwd=`/root/tensor-ecosystem/TensorPlatform` 时
  `FIGURE_DIR=uploads/tensor_reports` 解析到另一路径 → 部署相对路径不一致（平台级，建议 `TENSOR_REPORT_FIGURE_DIR` 绝对化）。

## ⑥ 未做 / 未跑（不算通过）

- 未核对 1953 原文原表原页（DOI 10.1115/1.4010702 本环境 403）：参考是**解析重构 + 独立降维交叉验证**，
  使命 goal 要求的"从 ref 原文/手册查得、引用原表原页"**未满足**（交付方如实声明）。
- 未测变法向力（ΔF_n≠0）、粘附 JKR、塑性、滚动叠加的耦合分支。
- 未做多颗粒 DEM 场景下该接触律的宏观标定（沙堆休止角等），故"引擎级"影响面仅到 `protocols.sandpile` 调用点。

## ⑦ Disposition

**approve with notes**：m16065 三件套齐、数字可复算、重跑逐比特复现、六段报告齐全 → **L4，消账**；
同时挂 3 个返修子项（F1 反向支/变法向力＝能力缺件；F2 误差口径改名；F3 平台裁判登记）、2 个平台侧观察（F4、F5）。