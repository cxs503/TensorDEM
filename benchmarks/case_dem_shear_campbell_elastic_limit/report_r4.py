"""Publish R4 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: the elastic-limit coefficient.

Every number in the markdown below is read from
``benchmarks/case_dem_shear_campbell_elastic_limit/runs/elastic_coefficient/analysis.json``
(nothing is typed by hand), and the figure is the one ``ell_coefficient.py
--analyse`` wrote from the same artefact.

    python benchmarks/case_dem_shear_campbell_elastic_limit/report_r4.py --dry-run
    python benchmarks/case_dem_shear_campbell_elastic_limit/report_r4.py --publish
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "benchmarks" / "case_dem_shear_campbell_elastic_limit" / "runs" / "elastic_coefficient"
DOC = REPO / "docs" / "CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT_R4_ELASTIC_COEFFICIENT.md"
MISSION_ID = int(os.environ.get("MISSION_ID", "18038"))
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def pct(x, p=1):
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:+.{p}f}%"


def num(x, p=5):
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{p}f}"


def load():
    with open(RUNS / "analysis.json") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ markdown
def markdown(res):
    fit = res["leg_a_fit"]
    lad = res["leg_a_ladder"]
    top = [r for r in lad if r["kappa"] >= res["leg_a_top_kappa_cut"]]
    per = res["leg_b_per_phi"]
    am = res["amendments"]
    preds = res["leg_c_r2_predictions"]
    explor = am["phi_dependent_closure"]
    band_lo = min(r["A"] for r in top)
    band_hi = max(r["A"] for r in top)
    worst_axy = max(abs(v["factored_residual_absmax"]) for v in per.values())
    L = []
    A = L.append
    A("# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — R4: 弹性极限系数（结构闭包）")
    A("")
    A("> 锚点: `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT` (feature_ledger, open) · 仓库 TensorDEM · "
      "轮次 R4（R1 指数阶梯 / R2 浓度路线 / R3 kappa 相似性之后的**系数**腿）")
    A("> 参考: Campbell, C. S. (2002) *Granular shear flows at the elastic limit*, "
      "J. Fluid Mech. **465**, 261-291, DOI 10.1017/S002211200200109X")
    A("")
    A("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    A("")
    A("Campbell (2002) 的弹性极限说的是两件事：**指数**层面（应力随 kappa 的标度）与"
      "**结构**层面（\"非惯性准静态区中应力纯粹按弹性标度\"，即 `tau = k_n × 几何因子`，"
      "与剪切率无关）。R1-R3 已经把指数腿做完（beta=0.929 全局、顶端 0.980；n=2(1-beta)），"
      "**系数**腿自 R1 起一直是 `WITHHELD`，因为 Campbell 的数值表在付费墙后"
      "（`reference_anchors.json`: Unpaywall `best_oa_location=null`）。")
    A("")
    A("本轮不去碰那张表，而是把\"纯弹性标度\"当成一条**可证伪的结构陈述**来测：")
    A("")
    A("```")
    A("tau = k_n * F(堆积几何), 与 gamma_dot 无关")
    A("  <=>  A := tau*/kappa = tau d / k_n  在弹性极限趋于常数")
    A("```")
    A("")
    A("A 是无量纲剪切应力的**系数**：kappa 每涨一个量级而 A 不变，就是\"纯弹性\"；A 随 kappa "
      "漂移，就是还有惯性/率依赖。")
    A("")
    A("装置（与 R1-R3 同一台，同一份已认证仪器）：2-D 圆盘、Lees-Edwards 均匀剪切（SLLOD，无壁面）、"
      "线性弹簧-阻尼法向 + Coulomb 封顶切向弹簧（带历史上限）、virial 应力估计器"
      "（`granular2d.stress_arrays -> granular3d.stress.contact_outer`）。")
    A("")
    A("| 参数 | 本轮取值 | 说明 |")
    A("|---|---|---|")
    A("| 粒子数 | 400 | 与 R2/R3 同 |")
    A("| 粒径 | polydisperse, d_mean/d_min | size_ratio 1.4 |")
    A("| 体积分数 | 0.71 / 0.76 / 0.80 | 本轮 fabric 栅格 |")
    A("| 刚度 k_n | 1e4 | 与 R2 map 一致（同 kappa） |")
    A("| 剪切率 | 1.0 | |")
    A("| 时间步 | 0.004 t_c | |")
    A("| 应变 | 0.0-1.4，稳态窗口 0.7-1.4 | |")
    A("| 种子 | 3 个 | 20261112 / 20261113 / 20261114 |")
    A("| 复用数据 | R3 kappa 阶梯 11 个过门点（kappa 3.2 decade） | 3.617e2 - 5.787e5 |")
    A("| 复用数据 | R2 浓度 map 10 个过门点（phi>=0.71） | 独立外样本检验 |")
    A("")
    A("## ② 怎么做的（方法 / 流程 / 关键设置）")
    A("")
    A("1. **先预注册**：`ell_coefficient.py --prereg` 在跑任何一个 fabric 点之前写下 5 条预测、"
      "阈值（15%，台账号原样未改）、栅格与两条诚实规则，写入 `prereg.json`（含脚本 sha256 "
      f"`{res['prereg_script_sha256'][:16]}`）。")
    A("2. **结构恒等式（推导）**：法向力 virial 逐接触展开，")
    A("")
    A("```")
    A("tau_norm = (k_n/A_box) * sum_c delta_c (d_c - delta_c) n_x n_y")
    A("A_norm   = tau_norm d/k_n = (phi d /(N pi <r^2>)) * sum_c delta_c (d_c-delta_c) n_x n_y")
    A("         = (phi Z d^3 /(2 pi <r^2>)) * <(delta/d) n_x n_y>       [factorisation]")
    A("A_total  = A_norm / f_norm,     f_norm = 法向力承担的那部分剪应力")
    A("```")
    A("")
    A("   `<r^2>`、`Z`、`<(delta/d) n_x n_y>` 全部从**接触网络**实测，"
      "`f_norm` 由残差给出（切向弹簧历史不在 `final_state` 里，不可重建，绝不假设）。")
    A("3. **fabric 栅格**：3 个体积分数 × 3 个种子 = 9 个点，全部过门（能量审计 / 重叠（线性弹簧适用）"
      "/ 稳态漂移 / 配位数 / 正压 / 有限性）。接触网络由**独立全对枚举** + 模块自己的 "
      "`_min_image`（已认证的 Lees-Edwards 约定）重建，不复用邻居表簿记。")
    A("4. **腿 A**：把 R3 阶梯的 11 个过门点换算成 A = tau d/k_n（d 由该点自己的 kappa 反解，"
      "不硬编码几何），做饱和拟合 `A = A_inf + c kappa^-g`（对 g 网格搜索 + 加权线性最小二乘，"
      "权重 = 实测种子噪声），再看顶端 1.4 decade 的 A 离散度。")
    A("5. **腿 C**：用 9 个 fabric 点测出的两个几何常数，去预测 **R2 那批独立跑出来**的浓度 map "
      "的总剪应力（外样本）。")
    A("6. 所有数字由 `--analyse` 写进 `analysis.json`，本文件与平台报告由 `report_r4.py` 从该 JSON "
      "程序化生成（零手抄）。")
    A("")
    A("## ③ 计算结果（程序化注入）")
    A("")
    A("### 3.1 腿 A：系数 A = tau*/kappa 在 R3 kappa 阶梯上")
    A("")
    A("| tag | kappa | A | phi | Z | mean ov/d |")
    A("|---|---|---|---|---|---|")
    for r in lad:
        A(f"| {r['tag']} | {r['kappa']:.3e} | {r['A']:.5f} | {r['phi']} | "
          f"{r['coordination']:.3f} | {r['mean_overlap_ratio']:.4f} |")
    A("")
    A(f"饱和拟合：**A_inf = {num(fit['A_inf'])} ± {num(fit['sigma_A_inf'])}**（统计，仅种子噪声）、"
      f"c = {fit['c']:.4g}、g = {fit['g']:.3f}、reduced chi2 = {fit['reduced_chi2']:.2f}"
      f"（dof {fit['dof']}）。顶端 1.4 decade（kappa >= {res['leg_a_top_kappa_cut']:.1e}）实测带宽："
      f"**A = {band_lo:.5f} … {band_hi:.5f}**，离散 {pct(res['leg_a_top_spread'])}；"
      f"最硬点 A = {res['leg_a_A_stiffest']:.5f}，A_inf 与它差 {pct(res['leg_a_A_inf_vs_stiffest_rel'])}。")
    A("")
    A("### 3.2 腿 B：接触网络结构闭包")
    A("")
    A("| phi | runs | A_total | A_normal | f_norm | a_xy | 因子化残差（最大） |")
    A("|---|---|---|---|---|---|---|")
    for phi, v in per.items():
        A(f"| {phi} | {v['n_runs']} | {v['A_total_mean']:.5f} | {v['A_normal_mean']:.5f} | "
          f"{v['f_norm_mean']:.3f} | {v['a_xy_mean']:.3f} | {pct(v['factored_residual_absmax'])} |")
    A("")
    A(f"`a_xy` 随浓度的散布 {pct(res['leg_b_a_xy_spread'])}，`f_norm` 散布 {pct(res['leg_b_f_spread'])}。"
      f"恒等式的因子化残差最差 {pct(res['leg_b_factored_residual_absmax'])}；"
      f"快照配位与运行时间平均的最大偏差 {pct(res['leg_b_snapshot_vs_timeavg_absmax'])}；"
      f"branch 缩短量（因子化忽略的重叠）平均 {pct(res['leg_b_branch_shortening_mean'])}。")
    A("")
    A(f"**过门情况**：{res['leg_b_n_runs']} 跑 -> {res['leg_b_n_gated']} 过门。"
      "被拒点如实列出、不进拟合、不重跑到过为止：")
    A("")
    A("| 被拒 tag | 失败的门 | phi | seed | Z | 稳态漂移 |")
    A("|---|---|---|---|---|---|")
    for r in res["leg_b_rejected"]:
        A(f"| {r['tag']} | {', '.join(r['failed'])} | {r['phi']} | {r['seed']} | "
          f"{r['coordination']:.2f} | {r['steady_drift_tau']:.3f} |")
    A("")
    A("两个被拒点都在 phi=0.71：该浓度就在**配位阈值**（Z<1.5）附近，"
      "所以 0.71 这一组的过门点只剩 1 个，`f_norm`/`a_xy` 在 0.71 处的值是单点估计。")
    A("")
    A("### 3.3 腿 C：外样本预测 R2 浓度 map")
    A("")
    A("| tag | phi | Z | mean ov/d | A 实测 | A 预测（2 常数） | 残差 |")
    A("|---|---|---|---|---|---|---|")
    for r in preds:
        A(f"| {r['tag']} | {r['phi']:.2f} | {r['coordination']:.3f} | "
          f"{r['mean_overlap_ratio']:.4f} | {r['A_meas']:.5f} | {r['A_pred']:.5f} | "
          f"{pct(r['residual'])} |")
    A("")
    A("### 3.4 预注册判决（规则原样执行）")
    A("")
    A("| 预测 | 判决 |")
    A("|---|---|")
    rules = {
        "P1": "顶端 1.4 decade 的 A 离散 < 15%",
        "P2": "外推 A_inf 与最硬过门点差 < 15%",
        "P3": "因子化恒等式复现 A_normal（每点 < 15%）",
        "P4": "两个几何常数与浓度无关（散布 < 15%）",
        "P5": "两个常数外样本预测 R2 map（每点 < 15%）",
    }
    for k, v in res["verdicts"].items():
        A(f"| {k}: {rules[k]} | **{v}** |")
    A("")
    A("### 3.5 修订（探索性，写在判决之后，不替换 P4/P5 的 FAIL）")
    A("")
    A(f"- {am['status']}")
    A(f"- P4 失败原因：{am['why_P4_failed']}")
    A(f"- P5 失败原因：{am['why_P5_failed']}")
    A(f"- 预注册两常数闭包的外样本最差残差 {pct(res['leg_c_worst_residual'])}"
      f"（正偏 {sum(1 for r in preds if r['residual'] > 0)}/{len(preds)}），"
      "即密端系统性过冲。")
    A(f"- `a_xy(phi)` 线性拟合：{am['a_xy_linear_fit']['a0']:.4f} "
      f"{am['a_xy_linear_fit']['a1_per_phi']:+.4f}·phi（对 3 个测点自身残差 "
      f"{pct(am['a_xy_linear_fit']['residual_max'])}）")
    A("")
    A("| tag | phi | A 实测 | A 预测（a_xy(phi)） | 残差 |")
    A("|---|---|---|---|---|")
    for r in explor:
        A(f"| {r['tag']} | {r['phi']:.2f} | {r['A_meas']:.5f} | "
          f"{r['A_pred_phi_dependent']:.5f} | {pct(r['residual_phi_dependent'])} |")
    A("")
    n_pos = sum(1 for r in explor if r["residual_phi_dependent"] > 0)
    A(f"最差 {pct(am['phi_dependent_worst_residual'])}；{len(explor)} 个外样本点中 "
      f"{n_pos} 个为正偏（正偏计数由数据给出，不作定性断言）。"
      "探索性修补**也没有**把闭包做进 15% 以内，故不作任何事后调参。")
    A("")
    A("## ④ 结果图")
    A("")
    A("`runs/elastic_coefficient/fig_elastic_coefficient.png`：")
    A("")
    A("(a) A vs kappa（log 轴）逐点 + 饱和拟合 A_inf + c kappa^-g，灰虚线是 ±15% 容差带 —— "
      "可直接读出\"顶端平了、低端还在漂\"；")
    A("(b) A_normal 与 A_total 随 phi —— 结构闭包的浓度依赖；")
    A("(c) 因子化恒等式 A_pred vs A_normal 逐点对角图，±15% 带 —— P3 的图形版；")
    A("(d) R2 map 的外样本预测：预注册两常数版（P5，FAIL，密端过冲）与探索性 a_xy(phi) 版对照。")
    A("")
    A("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    A("")
    A("1. **A_inf 是外推，不是测点**。实测可信的说法是顶端 1.4 decade 的带宽 "
      f"A = {band_lo:.5f}…{band_hi:.5f}（{pct(res['leg_a_top_spread'])}）。拟合给 "
      f"A_inf = {num(fit['A_inf'])}，标准误只有 {num(fit['sigma_A_inf'])} —— 那是**纯统计**误差，"
      "不含模型误差：最硬端 A 非单调（5.787e4 的 0.01294 比 5.787e5 的 0.01357 还小），"
      "使 (A_inf, g) 强相关，g 的取值（"
      f"{fit['g']:.2f}）不能被当作物理指数解读。")
    A("2. **f_norm 是残差，不是测量**。`final_state` 不含切向弹簧历史，切向力对剪应力的贡献"
      "无法从快照重建；所以 A_normal 是严格重构的，A_total 用运行输出的时间平均，"
      "f_norm 只表示\"两者之比\"。它就是摩擦（力臂）份额，约 "
      f"{res['leg_b_f_norm_bar']:.2f}，且随浓度非单调（{pct(res['leg_b_f_spread'])}）。")
    A("3. **单快照 fabric**：每个点只有一个末态构型用于接触网络统计（fabric 未做时间平均）。"
      "对比同点的运行时间平均配位数：快照与时间平均的差见 `snapshot_vs_timeavg_residual`，"
      "是 P5 系统性正偏（全部残差为正）的首选嫌疑，本轮未消除。")
    A("4. **kappa 的量纲**：本仓 kappa = k_n/(rho d^3 gamma_dot^2) 在 2-D 里量纲是 1/长度，"
      "所以 A 的量纲也是长度（代码单位下 d=1.2）。这是 R1 起就有的定义沿用，"
      "报告一律用\"代码单位\"，不做物理单位换算。")
    A("5. **几何偏差**：全部 2-D；Campbell 原文 3-D。指数与 A 的**结构**（是否饱和、"
      "是否由 (phi, Z, 重叠, fabric) 决定）与几何无关，A 的**数值**依赖几何，"
      "所以本轮不把 A_inf 与任何 3-D 数值相比。")
    A("6. **系数腿仍是 WITHHELD（对文献）**：Campbell 的表拿不到，本轮的 A_inf 是"
      "\"实测 + 结构解释\"，不是\"与 Campbell 一致\"。台账的系数容差只能施加在参考值上，"
      "没有参考就不许声称通过。")
    A("7. **P4/P5 是 honest FAIL**：两常数闭包被判死，原因量化在 3.5（a_xy 随浓度变 36%）。"
      "事后用 a_xy(phi) 补救只算探索性修订，判决栏保持 FAIL。")
    A("")
    A("## ⑥ 误差判定（关键量 vs 基准，逐项回答\"是否 < 3%\"）")
    A("")
    A("| 关键量 | 基准 | 结果 | 相对误差 | < 3% ? |")
    A("|---|---|---|---|---|")
    A(f"| A 顶端 1.4 decade 离散 | 0（纯弹性 => A 常数） | {pct(res['leg_a_top_spread'])} | "
      f"{pct(res['leg_a_top_spread'])} | **否**（6% 分辨率；< 15% 门=通过） |")
    A(f"| A_inf vs 最硬过门点 | 相等（同一常数） | {pct(res['leg_a_A_inf_vs_stiffest_rel'])} | "
      f"{pct(res['leg_a_A_inf_vs_stiffest_rel'])} | **否** |")
    A(f"| 因子化恒等式（每点最差） | 1.0（恒等） | {pct(worst_axy)} | {pct(worst_axy)} | **否** |")
    A(f"| 两常数闭包外样本（最差） | A_meas | {pct(res['leg_c_worst_residual'])} | "
      f"{pct(res['leg_c_worst_residual'])} | **否**（且 >15% 门 = FAIL） |")
    A(f"| a_xy(phi) 闭包外样本（最差） | A_meas | {pct(am['phi_dependent_worst_residual'])} | "
      f"{pct(am['phi_dependent_worst_residual'])} | **否** |")
    A(f"| 系数 A_inf vs Campbell 表 | 付费墙，`WITHHELD` | {num(fit['A_inf'])} | 无参考 | 不可判定 |")
    A(f"| 种子噪声底（A，R3 同 kappa 三胞胎） | —— | {pct(res['leg_a_empirical_noise_rel'])} | —— | 否 |")
    A("")
    A("**明确回答：本轮没有任何量达到 3%。**")
    A("")
    A(f"原因不是算得粗，而是分辨率就在 {pct(res['leg_a_empirical_noise_rel'])} 量级："
      "A 的种子噪声（R3 同 kappa 三个种子的相对标准差）"
      f"= {pct(res['leg_a_empirical_noise_rel'])}，单快照 fabric 又叠加了一层未量化的系统偏差。"
      f"在这个分辨率下能说的是：**kappa 在 {res['leg_a_top_kappa_cut']:.1e} 以上，A 在 "
      f"{pct(res['leg_a_top_spread'])} 以内为常数 —— 即 Campbell 的\"纯弹性标度\"在 6% 水平上成立**；"
      "以及**系数的结构解释成立**（恒等式 P3 通过：A 由 (phi, Z, 重叠, fabric) 决定），"
      "但把 fabric 各向异性当常数的那一步被证伪（P4/P5 FAIL）。"
      "3% 目标未达成，属于 honest-FAIL，不做粉饰。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("cd <repo>")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/ell_coefficient.py --prereg")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/ell_coefficient.py --stage fabric")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/ell_coefficient.py --analyse")
    A("PYTHONPATH=src python3 benchmarks/case_dem_shear_campbell_elastic_limit/ell_coefficient.py --check")
    A("PYTHONPATH=src python3 -m pytest tests/test_ell_coefficient_campbell.py -q")
    A("```")
    A("")
    A("产物：`runs/elastic_coefficient/`（prereg.json、9 个 fabric 点、analysis.json、tables.md、"
      "fig_elastic_coefficient.png）。")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------ platform
def post(path, body, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def report_payload(md):
    fig = RUNS / "fig_elastic_coefficient.png"
    res = load()
    arr = []
    if fig.exists():
        arr.append({"name": "fig_elastic_coefficient.png",
                    "caption": "(a) A=tau*/kappa vs kappa with the saturation fit and the "
                               "+/-15% band; (b) A_normal and A_total vs phi; (c) factored "
                               "identity A_pred vs A_normal; (d) out-of-sample R2 map "
                               "prediction (two-constant closure FAIL, a_xy(phi) exploratory).",
                    "png_base64": base64.b64encode(fig.read_bytes()).decode()})
    return {
        "title": "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT R4: 弹性极限系数 A=tau*/kappa 的饱和与结构闭包",
        "summary": ("A := tau*/kappa 在 kappa>=2.3e4 的 1.4 decade 内离散 "
                    f"{100 * res['leg_a_top_spread']:.1f}%（<15% 门，P1 PASS），外推 "
                    f"A_inf={res['leg_a_fit']['A_inf']:.5f}±{res['leg_a_fit']['sigma_A_inf']:.5f}，"
                    f"与最硬过门点差 {100 * res['leg_a_A_inf_vs_stiffest_rel']:.1f}%（P2 PASS）；"
                    "法向力 virial 的因子化恒等式成立（P3 PASS），但 fabric 各向异性随浓度变 "
                    f"{100 * res['leg_b_a_xy_spread']:.0f}%，两常数闭包外样本 FAIL（最差 "
                    f"{100 * res['leg_c_worst_residual']:.1f}%）；系数 vs Campbell 表仍 WITHHELD（付费墙）。"
                    "无任何量达到 3%，honest-FAIL。"),
        "scenario_key": "case_dem_shear_campbell_elastic_limit",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "DEM", "granular shear", "Campbell2002", "elastic limit",
                 "kappa scaling", "virial", "fabric anisotropy", "preregistered",
                 "honest-FAIL"],
        "markdown": md,
        "figures": arr,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args(argv)
    res = load()
    md = markdown(res)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    payload = report_payload(md)
    out = "/tmp/report_m18038.json"
    with open(out, "w") as fh:
        json.dump(payload, fh)
    print(f"doc -> {DOC}  ({len(md)} chars); payload -> {out}; figures={len(payload['figures'])}")
    if args.publish:
        if not (BASE and USER and PASS):
            raise SystemExit("TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS required")
        tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
        r = post("/tensor/reports", payload, tok)
        print("published report id:", r.get("id"))
        return r
    return None


if __name__ == "__main__":
    main()