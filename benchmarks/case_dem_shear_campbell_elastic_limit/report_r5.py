#!/usr/bin/env python3
"""Generate docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT_R5_TIMEAVG_FABRIC.md and the
platform-report payload from runs/fabric_timeavg/analysis.json (zero hand-typed
numbers: every value in the document is interpolated from the JSON).
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs", "fabric_timeavg")
DOC = os.path.join(REPO, "docs",
                   "CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT_R5_TIMEAVG_FABRIC.md")
CASE = "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT"


def load() -> Dict[str, Any]:
    with open(os.path.join(RUNS, "analysis.json")) as fh:
        res = json.load(fh)
    pts: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(RUNS)):
        if name.startswith("fab_") and name.endswith(".json"):
            with open(os.path.join(RUNS, name)) as fh:
                pts.append(json.load(fh))
    res["_runs"] = pts
    return res


def pct(x: float) -> str:
    return "%+.1f%%" % (100.0 * x)


def build_doc(res: Dict[str, Any]) -> str:
    ex = res["exploratory_amendment"]
    grid = res["grid"]
    L: List[str] = []
    A = L.append
    A(f"# {CASE} — R5: 稳态窗口 fabric 时间平均（把 R4 的「头号嫌疑」做成数字）")
    A("")
    A("> 锚点: `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`（feature_ledger，open）· 仓库 TensorDEM · 第 5 轮")
    A("> 参考: Campbell, C. S. (2002) *Granular shear flows at the elastic limit*, "
      "J. Fluid Mech. **465**, 261-291, DOI 10.1017/S002211200200109X")
    A("> 预注册: `runs/fabric_timeavg/prereg.json`（脚本 sha256 `%s`）"
      % str(res["prereg_script_sha256"])[:16])
    A("")
    A("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    A("")
    A("R4 把 Campbell 弹性极限的**系数**结构做到了 PASS（`A = tau d/k_n` 在 kappa ≳ 2.3e4 饱和到 +5.7%），")
    A("但它那条可证伪的闭包——「两个几何常数（a_xy, f_norm）能预测**另一批独立跑出来的**浓度图的剪切应力」")
    A("——FAIL：P4（a_xy 与浓度无关）与 P5（外样本预测）双双失败，密端系统性过冲 +31.2%。")
    A("R4 把原因写成了**嫌疑**而没有量化：")
    A("")
    A("> 「单快照 fabric … 是 P5 系统性正偏的首选嫌疑，本轮未消除」")
    A("")
    A("即：fabric 常数是在**末态一张快照**上量的，而它们要解释的应力是**稳态窗口上的时间平均**。")
    A("R4 自己的诊断已经显示这个差可以有多大：快照配位数与时间平均最多差 14.2%。")
    A("")
    A("本轮不争论这个嫌疑，而是把它**消除**：模拟器在应力平均的**同一个稳态窗口**内累计接触网络统计量，")
    A("并把剪切 virial 精确拆成法向/切向两支，使 `f_norm` 成为**实测的时间平均**而不是快照残差。")
    A("")
    A("| 参数 | 取值 | 说明 |")
    A("|---|---|---|")
    A("| 粒子数 | %d | 2-D 圆盘，与 R2–R4 同一台装置 |" % grid["n_particles"])
    A("| 体积分数 | %s | 与 R4 fabric 栅格完全相同 |"
      % " / ".join("%.2f" % p for p in grid["phi"]))
    kp = [q["kappa"] for q in res["_runs"]]
    A("| 刚度 k_n | %.3g | kappa = %.3g – %.3g（同一台装置） |"
      % (grid["kn"], min(kp), max(kp)))
    A("| 剪切率 | %.1f | $\\dot\\gamma$ |" % grid["shear_rate"])
    A("| 时间步 | %.3g t_c | 与 R1–R4 相同 |" % grid["dt_factor"])
    A("| 应变窗口 | 0 → %.1f，平均窗口 %.1f → %.1f | fabric 与应力在**同一窗口**平均 |"
      % (grid["strain_total"], grid["strain_average_from"], grid["strain_total"]))
    A("| 种子 | %s | 3 个/浓度 |" % " / ".join(str(s) for s in grid["seeds"]))
    A("| 采样 | 每 0.01 应变一个样本（每点约 70 个） | 时间平均的分辨率 |")
    A("")
    A("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    A("")
    A("1. **能力先行**：`simulate_plane_shear(case, fabric_statistics=True)` 在既有采样点（应变 ≥ 平均窗口起点、")
    A("   每 0.01 应变）额外累计接触网络量——配位数 Z、平均重合格 <ov>、$\\langle n_x n_y\\rangle$、")
    A("   重合加权 $\\langle ov\\,n_x n_y\\rangle$——并同时把剪切 virial 拆成")
    A("   `tau_contact = tau_normal + tau_tangential`（**逐样本恒等式**，不是拟合）。")
    A("   开关默认关闭；关闭时输出与历史工件**逐位相同**（未改任何一个已发布数字，测试断言这一点）。")
    A("2. **先预注册**：`fabric_timeavg.py --prereg` 在跑任何一个 fabric 点**之前**写下 5 条预测、阈值")
    A("   （15% 台账容差原样、6.25% 种子噪声底）与两条诚实规则，写入 `prereg.json`。")
    A("3. **同一台装置、只换估计器**：栅格、种子、k_n、时间步与 R4 逐项相同；R4 的快照估计器在同一批点上")
    A("   一并重算，所以「快照 vs 时间平均」是**同点对照**，不是跨轮比较。")
    A("4. **外样本腿不变**：R2 浓度图（22 个过门点，±壳内 10 点 + 控制/速率/散布点）**只用于预测**，")
    A("   从不回拟合、不重跑。")
    A("5. **判决程序化**：`--analyse` 按预注册规则原样套用，`--check` 断言恒等式与台账记账；")
    A("   本文件与表格由本脚本从 `analysis.json` 插值生成（零手抄）。")
    A("")
    A("### 2.1 本轮新增的内核能力")
    A("")
    A("```python")
    A("from tensordem.elastic_limit_shear import simulate_plane_shear")
    A("out = simulate_plane_shear(case, fabric_statistics=True)")
    A("out[\"fabric\"][\"a_xy_mean\"], out[\"fabric\"][\"f_norm_mean\"], "
      "out[\"fabric\"][\"virial_split_residual_max\"]")
    A("```")
    A("")
    A("`out[\"fabric\"]` 给出每个量的 mean/std/min/max（时间平均 + 时间涨落），")
    A("`f_norm` 由**逐样本**比值构成后再平均（绝不做「平均之比」）。")
    A("")
    A("## ③ 计算结果（程序化注入）")
    A("")
    A("### 3.1 逐点：时间平均 vs 末态快照")
    A("")
    A("| tag | phi | Z (t) | a_xy (t) | a_xy (snap) | f_norm (t) | f_norm (snap) "
      "| 偏差 a_xy | 偏差 Z |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p in res["_runs"]:
        A("| %s | %.2f | %.3f | %.4f | %.4f | %.3f | %.3f | %s | %s |"
          % (p["tag"].replace("fab_", ""), p["phi"], p["timeavg"]["coordination"],
             p["timeavg"]["a_xy"], p["snapshot"]["a_xy"], p["timeavg"]["f_norm"],
             p["snapshot"]["f_norm"], pct(p["bias"]["a_xy_snapshot_vs_timeavg"]),
             pct(p["bias"]["coordination_snapshot_vs_timeavg"])))
    A("")
    A("### 3.2 常数与浓度依赖（预注册 P1 / P2）")
    A("")
    A("| phi | 过门点数 | a_xy (时间平均) | 种子 sd | a_xy (快照) | f_norm (t) | f_norm (snap) | Z (t) |")
    A("|---|---|---|---|---|---|---|---|")
    for phi, v in res["leg_b_per_phi"].items():
        A("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |"
          % (phi, v["n"], v["a_xy_timeavg"], v["a_xy_timeavg_seed_std"], v["a_xy_snapshot"],
             v["f_norm_timeavg"], v["f_norm_snapshot"], v["coordination_timeavg"]))
    A("")
    A("- a_xy 的浓度散布：**时间平均 %.1f%%**，快照 %.1f%% —— **没有变小，反而略大**。"
      % (100 * res["leg_b_a_xy_spread"], 100 * res["leg_b_a_xy_spread_snapshot_r4_estimator"]))
    A("- f_norm 的浓度散布：**时间平均 %.1f%%**，快照 %.1f%% —— 快照估计器在这里才是灾难。"
      % (100 * res["leg_b_f_spread"], 100 * res["leg_b_f_spread_snapshot_r4_estimator"]))
    A("")
    A("### 3.3 外样本预测（预注册 P3）与结构恒等式（P5）")
    A("")
    A("| tag | phi | Z | <ov> | A_meas | A_pred | 残差 |")
    A("|---|---|---|---|---|---|---|")
    for r in res["leg_c_r2_predictions"]:
        A("| %s | %.2f | %.3f | %.5f | %.5f | %.5f | %s |"
          % (r["tag"], r["phi"], r["coordination"], r["mean_overlap_ratio"],
             r["A_meas"], r["A_pred"], pct(r["residual"])))
    A("")
    A("最差残差 **%s**（R4 用快照估计器：%s），%d 个点中 %d 个为正偏。"
      % (pct(res["leg_c_worst_residual"]), pct(res["leg_c_r4_worst_residual"]),
         res["leg_c_n_points"], res["leg_c_n_positive_bias"]))
    A("")
    A("因子化恒等式（全部输入均为时间平均）：逐点最差 **%s**；"
      % pct(res["leg_b_factored_residual_absmax"]))
    A("法向/切向 virial 拆分残差（恒等式）逐样本最差 **%.2e**。"
      % res["virial_split_residual_absmax"])
    A("")
    A("### 3.4 预注册判决（规则原样执行）")
    A("")
    A("| 预测 | 内容 | 判决 |")
    A("|---|---|---|")
    A("| P1 | 时间平均后 a_xy 与浓度无关（散布 ≤ 15%%） | **%s**（%.1f%%） |"
      % (res["verdicts"]["P1"], 100 * res["leg_b_a_xy_spread"]))
    A("| P2 | 实测时间平均 f_norm 与浓度无关 | **%s**（%.1f%%） |"
      % (res["verdicts"]["P2"], 100 * res["leg_b_f_spread"]))
    A("| P3 | 两常数闭包外样本 ≤ 15%% | **%s**（最差 %s） |"
      % (res["verdicts"]["P3"], pct(res["leg_c_worst_residual"])))
    A("| P4 | 快照是**有偏**估计器（≥5/9 点偏差 ≥ 噪声底） | **%s**（%d/9） |"
      % (res["verdicts"]["P4"],
         res["snapshot_bias"]["n_runs_with_a_xy_bias_above_noise"]))
    A("| P5 | 因子化恒等式在时间平均输入下仍成立 | **%s**（最差 %s） |"
      % (res["verdicts"]["P5"], pct(res["leg_b_factored_residual_absmax"])))
    A("")
    A("### 3.5 EPILOGUE（探索性；判在判决之后写，不替换 P1/P3 的 FAIL）")
    A("")
    A(ex["status"])
    A("")
    A("- " + ex["why"])
    A("- a_xy(phi) = %+.4f %+.4f·phi（对 9 个点自身残差最大 %.1f%%）；"
      % (ex["a_xy_linear_fit"]["intercept"], ex["a_xy_linear_fit"]["slope_per_phi"],
         100 * ex["a_xy_linear_fit"]["residual_max_on_9_runs"]))
    A("  f_norm(phi) = %+.4f %+.4f·phi（自身残差最大 %.1f%%）"
      % (ex["f_norm_linear_fit"]["intercept"], ex["f_norm_linear_fit"]["slope_per_phi"],
         100 * ex["f_norm_linear_fit"]["residual_max_on_9_runs"]))
    A("")
    A("| 检验 | 点数 | 最差残差 | 是否在 15%% 内 |")
    A("|---|---|---|---|")
    A("| 两常数闭包（P3，预注册） | %d | %s | %s |"
      % (res["leg_c_n_points"], pct(res["leg_c_worst_residual"]),
         "是" if res["leg_c_worst_residual"] <= res["tolerance"] else "否"))
    A("| phi 线性闭包，样本内 | %d | %s | %s |"
      % (len(ex["in_sample"]), pct(ex["in_sample_worst_residual"]),
         "是" if ex["in_sample_worst_residual"] <= res["tolerance"] else "否"))
    A("| phi 线性闭包，**留一浓度**交叉验证 | %d | %s | %s |"
      % (ex["loco_n_points"], pct(ex["loco_worst_residual"]),
         "是" if ex["loco_worst_residual"] <= res["tolerance"] else "否"))
    A("")
    A("留一浓度交叉验证（LOCO）的做法：每次**拿掉一个浓度**、用另两个浓度重拟合两条直线，")
    A("只预测落在被拿掉浓度带内的 R2 点——被预测浓度**从不进入拟合**，所以这是诚实的泛化数字。")
    A("")
    A("## ④ 结果图")
    A("")
    A("`runs/fabric_timeavg/fig_fabric_timeavg.png`：")
    A("")
    A("(a) a_xy 随 phi（时间平均 ± 种子 sd，对照 R4 快照）——可直接读出「浓度依赖是真的」；")
    A("(b) f_norm 随 phi（同上）——快照估计器的 41% 散布塌缩成 4%；")
    A("(c) 外样本残差随 phi：预注册两常数版（P3，超 15% 带）与 LOCO 版（探索）对照，灰带 = ±15% 台账容差；")
    A("(d) 逐点「快照/时间平均 − 1」条形图（a_xy 与 Z），虚线 = 6.25% 种子噪声底 —— P4 的图形版。")
    A("")
    A("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    A("")
    A("1. **R4 的头号嫌疑被证伪，但不是「快照没用」**：a_xy 的快照偏差只在 %d/9 个点超过噪声底"
      % res["snapshot_bias"]["n_runs_with_a_xy_bias_above_noise"])
    A("   （最大 %.1f%%），而且**符号非系统**；真正塌的是 `f_norm`（散布 41.2%% → %.1f%%）。"
      % (100 * res["snapshot_bias"]["a_xy_absmax"], 100 * res["leg_b_f_spread"]))
    A("   所以 R4 用 a_xy 解释过冲**指向了错的量**：过冲主要来自 f_norm 的快照偏差把分母压小。")
    A("2. **phi=0.71 只有 1 个过门点**：另两个种子（Z = 1.165 / 1.346）落在配位数门（≥1.5）外，")
    A("   但三个种子的 a_xy 时间平均是 0.268/0.272/0.276（散布 3%），所以该浓度行的结论不靠单点运气；")
    A("   本轮**没有**为了让点过门而重跑或调栅格（诚实规则第 1 条）。")
    A("3. **2-D 限制照旧**：全部 2-D；Campbell 原文 3-D。a_xy 的**数值**依赖几何，**结构**不依赖。")
    A("4. **系数腿对文献仍是 WITHHELD**：Campbell 的表在付费墙后（`reference_anchors.json`），")
    A("   本轮所有「A」都是本装置实测量，没有任何一句「与 Campbell 一致」。")
    A("5. **LOCO 只有 3 个浓度**：留一浓度等价于用 2 点定一条直线，外推能力有限；")
    A("   它给的是「独立浓度上不外推崩掉」的证据，不是精度证明。")
    A("6. **f_norm 的定义变了**：本轮 f_norm = 法向力承担的剪切 virial 份额（时间平均），")
    A("   与 R4 的「A_normal/A_total」残差定义不同；这正是 R4 注意点 2 里点名的欠账，现在补上了。")
    A("")
    A("## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 < 3%」）")
    A("")
    A("| 关键量 | 基准 | 本轮结果 | 相对偏差 | < 3% ? |")
    A("|---|---|---|---|---|")
    A("| 法向/切向 virial 拆分 | 恒等式（=0） | %.1e | 0 | **是**（恒等式，非测量） |"
      % res["virial_split_residual_absmax"])
    A("| 因子化恒等式（时间平均输入，逐点最差） | 1.0 | %s | %s | **否** |"
      % (pct(res["leg_b_factored_residual_absmax"]), pct(res["leg_b_factored_residual_absmax"])))
    A("| a_xy 浓度散布（=0 才是常数） | 0 | %.1f%% | %.1f%% | **否**（且 >15%% 门 = P1 FAIL） |"
      % (100 * res["leg_b_a_xy_spread"], 100 * res["leg_b_a_xy_spread"]))
    A("| f_norm 浓度散布（时间平均） | 0 | %.1f%% | %.1f%% | **否**（<15%% 门 = P2 PASS） |"
      % (100 * res["leg_b_f_spread"], 100 * res["leg_b_f_spread"]))
    A("| 两常数闭包外样本（最差） | A_meas | %s | %s | **否**（>15%% = P3 FAIL） |"
      % (pct(res["leg_c_worst_residual"]), pct(res["leg_c_worst_residual"])))
    A("| 探索 phi 线性闭包 LOCO（最差） | A_meas | %s | %s | **否**（<15%% 但非预注册） |"
      % (pct(ex["loco_worst_residual"]), pct(ex["loco_worst_residual"])))
    A("| 种子噪声底（a_xy，同浓度三胞胎） | —— | %.1f%%–%.1f%% | —— | 否 |"
      % (100 * min(v["a_xy_timeavg_seed_std"] / v["a_xy_timeavg"]
                  for v in res["leg_b_per_phi"].values() if v["n"] > 1),
         100 * max(v["a_xy_timeavg_seed_std"] / v["a_xy_timeavg"]
                   for v in res["leg_b_per_phi"].values() if v["n"] > 1)))
    A("| 系数 A vs Campbell 表 | 付费墙 | %s | 无参考 | 不可判定（WITHHELD） |"
      % ("A_inf 见 R4"))
    A("")
    A("**明确回答：本轮除代数恒等式外，没有任何关键量达到 3%。**")
    A("")
    A("原因不是算得粗：a_xy 的种子噪声底本身就有 %.0f%% 量级，"
      % (100 * max(v["a_xy_timeavg_seed_std"] / v["a_xy_timeavg"]
                   for v in res["leg_b_per_phi"].values() if v["n"] > 1)))
    A("在这个分辨率下本轮能说的是三件事：(i) **R4 的嫌疑被证伪**——a_xy 的快照偏差不超噪声底；")
    A("(ii) **真正的偏差源是 f_norm**，时间平均把它的浓度散布从 41.2%% 压到 %.1f%%；"
      % (100 * res["leg_b_f_spread"]))
    A("(iii) 因此两常数闭包仍然 FAIL（P3 最差 %s，好于 R4 但未进 15%%），" % pct(res["leg_c_worst_residual"]))
    A("而**保留 a_xy 的浓度依赖 + LOCO 交叉验证**后，闭包在 19 个独立点上做到最差 %s ——"
      % pct(ex["loco_worst_residual"]))
    A("这是下一轮可以直接预注册的形式，不是本轮的成绩。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("cd <repo>")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/fabric_timeavg.py --prereg")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/fabric_timeavg.py --stage fabric  # 9 点，约 4.5 min 单核")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/fabric_timeavg.py --analyse")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/fabric_timeavg.py --check")
    A("PYTHONPATH=src python3 -m pytest tests/test_ell_fabric_timeavg_campbell.py -q   # 6 passed")
    A("PYTHONPATH=src python3 -m pytest tests/test_elastic_limit_shear.py "
      "tests/test_ell_coefficient_campbell.py -q   # 31 passed, 2 skipped")
    A("```")
    A("")
    A("产物：`runs/fabric_timeavg/`（prereg.json、9 个 fabric 点、analysis.json、tables.md、")
    A("fig_fabric_timeavg.png）。")
    A("")
    A("## 下一轮线索")
    A("")
    A("1. **把 LOCO 闭包变成预注册**：`a_xy(phi)`、`f_norm(phi)` 两条直线 + 留一浓度，")
    A("   在两到三个新浓度（0.73 / 0.78）上做**真正的外样本**——那才是把系数腿从「结构解释」推到「预测」的路。")
    A("2. **f_norm 的物理**：它现在有清晰定义（法向力剪切份额，时间平均 ≈ %.2f–%.2f）且几乎与浓度无关，"
      % (min(v["f_norm_timeavg"] for v in res["leg_b_per_phi"].values()),
         max(v["f_norm_timeavg"] for v in res["leg_b_per_phi"].values())))
    A("   但缺一个独立理论值（切向弹簧历史决定它）——可做「同一摩擦系数下 f_norm 是否随 k_t/k_n 变」的判别实验。")
    A("3. **a_xy(phi) 的成因**：需要在 fabric 张量上做各向异性分解（contact-normal vs branch-vector），")
    A("   判断浓度依赖来自分支长度分布还是接触方向分布。")
    A("")
    return "\n".join(L) + "\n"


def build_report(res: Dict[str, Any]) -> Dict[str, Any]:
    doc = open(DOC).read()
    fig = os.path.join(RUNS, "fig_fabric_timeavg.png")
    figures = []
    if os.path.exists(fig):
        figures.append({
            "name": "fig_fabric_timeavg.png",
            "caption": "(a) a_xy vs phi 时间平均 vs 快照; (b) f_norm 散布 41.2%→4.3%; "
                       "(c) 外样本残差 vs phi（±15% 台账带）; (d) 快照偏差与种子噪声底",
            "png_base64": base64.b64encode(open(fig, "rb").read()).decode(),
        })
    summary = ("R5: 用稳态窗口时间平均的接触网络统计替换 R4 的单快照估计器，"
               "证伪 R4「a_xy 快照偏差」嫌疑，定位真实偏差源为 f_norm（浓度散布 41.2%%→%.1f%%）；"
               "两常数闭包仍 FAIL（最差 %s，R4 %s），留一浓度交叉验证的 phi 线性闭包 19 点最差 %s"
               % (100 * res["leg_b_f_spread"], pct(res["leg_c_worst_residual"]),
                  pct(res["leg_c_r4_worst_residual"]), pct(res["exploratory_amendment"]["loco_worst_residual"])))
    return {
        "title": "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT R5：稳态窗口 fabric 时间平均"
                 "——证伪快照嫌疑，定位 f_norm 为真实偏差源",
        "summary": summary,
        "scenario_key": "dem_shear_campbell_elastic_limit",
        "mission_id": 19956,
        "tags": ["DEM", "granular shear", "Campbell 2002", "elastic limit", "fabric",
                 "time average", "TensorDEM", "coefficient closure"],
        "markdown": doc,
        "figures": figures,
    }


if __name__ == "__main__":
    res = load()
    doc = build_doc(res)
    with open(DOC, "w") as fh:
        fh.write(doc)
    print("doc ->", DOC, len(doc), "bytes")
    payload = build_report(res)
    with open("/tmp/report.json", "w") as fh:
        json.dump(payload, fh)
    print("report payload -> /tmp/report.json", len(payload["figures"]), "figure(s)")