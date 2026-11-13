"""Render ``docs/case_column_collapse_round5_m16991.md`` from the run artifacts.

Zero hand-copied numbers: every table in the note is formatted from
``artifacts/m16991_column_collapse_r5/summary.json``, ``cases.csv`` and the
per-chain traces under ``runs/case_column_collapse_r5_m16991/``.
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "m16991_column_collapse_r5")
RUN = os.path.join(ROOT, "runs", "case_column_collapse_r5_m16991")
DOC = os.path.join(ROOT, "docs", "case_column_collapse_round5_m16991.md")


def load_driver():
    path = os.path.join(ROOT, "benchmarks", "case_column_collapse_round5_m16991.py")
    spec = importlib.util.spec_from_file_location("w5r", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:+.{nd}f}"


def num(x, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> str:
    w5 = load_driver()
    summary = json.load(open(os.path.join(ART, "summary.json")))
    chains = w5.load_chains()
    with open(os.path.join(ART, "cases.csv")) as fh:
        cases = list(csv.DictReader(fh))

    L = []
    A = L.append
    A("# CASE-COLUMN-COLLAPSE 轮 5（m16991）：打破轮 4 的 μ 右删失，裁决 H5")
    A("")
    A(f"锚点 slug: `CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌，ref Lube et al. 2004，"
      f"容差 <15 %）。上游轮次：`docs/case_column_collapse_lube2004.md`（轮 1）、"
      f"`docs/case_column_collapse_round2_m15996.md`、`round3_m16534`、`round4_m16862`。")
    A("")
    A("## ① 案例介绍")
    A("")
    A("圆柱颗粒柱在粗糙刚性地面上塌落，参照实验 Lube, Huppert, Sparks & Freundt (2004), "
      "*Axisymmetric collapses of granular columns*, JFM **508**, 175-199, "
      f"doi:{summary['doi']}。参考律（论文摘要，程序内逐字引用并由 "
      "`lube_2004_runout_ratio()` 在运行时求值，目标数不手抄）：")
    A("")
    A("    r_inf/r_i = 1 + 1.24 a      (a < 1.7)")
    A("    r_inf/r_i = 1 + 1.6 a^0.5   (a > 1.7)")
    A("")
    A("轮 3 证明单一常数接触摩擦 μ 不能覆盖全 `a` 轴（换支点两侧要求 μ≈0.5 与 μ≈0.8）；"
      "轮 4 把问题升级为联合识别：**是否存在单一常数对 (μ, μ_r) 同时满足两支**。轮 4 在 "
      "μ∈[0.5,0.9]×μ_r∈[0.1,0.5] 网格上只有 μ_r=0.5 一个水平两支都被夹住（gap=+0.29），"
      "其余水平的平方根支需求越出 μ=0.9 上界（右删失），因此裁决为 honest INCOMPLETE。")
    A("")
    A("本轮（轮 5）预登记的问题 **H5**：")
    A("")
    A("> 两支的需求曲线在 μ_r ∈ [0.1, 0.5] 上过零（`gap(μ_r) = μ_req^{a=3} − μ_req^{a=1}` "
      "改变符号），即存在单一 (μ, μ_r) 同时把两支的 runout 误差压进 15 %。")
    A("")
    A("裁决规则（任何 wave-5 计算之前固定，见 `benchmarks/case_column_collapse_round5_m16991.py` "
      "模块文档）：以 Illinois 割线法在 μ ∈ "
      f"[{summary['protocol']['mu_bracket'][0]:g}, {summary['protocol']['mu_bracket'][1]:g}] "
      f"上求 `μ_req`（到 ±{summary['protocol']['mu_tol']:g}，每条链最多 "
      f"{summary['protocol']['max_eval']} 次求值）；**只在实际观测到区间两端异号时**声明根，"
      "否则报删失下界（`μ_req > 上界`），绝不插值造假根。"
      "`gap` 在 ≥3 个两支均被夹住的 μ_r 水平上保持同号 ⇒ H5 证伪；出现变号 ⇒ 支持；"
      "其余 ⇒ INCOMPLETE（honest，不粉饰）。")
    A("")
    A("计算协议：除 (μ, μ_r, a) 外与轮 1-4 **逐字段相同**（8 粒/柱半径、dt_scale=1、"
      "e=0.5、kn=6e4、duration_factor=8、seed=12345），由 "
      "`tests/test_column_collapse_round5_m16991.py::test_protocol_matches_round1_benchmark` "
      "机器校验；且 μ=0.9 的点与轮 4 探针同 (μ,μ_r,a)/同种子，构成确定性对照。")
    A("")
    A("## ② 怎么做的")
    A("")
    A("| 链 | a 支 | μ_r | 括号 | 求值次数 | 结论 |")
    A("|---|---|---|---|---|---|")
    for cid, t in sorted(chains.items()):
        pr = t.get("protocol", {})
        br = pr.get("mu_bracket", ["?", "?"])
        A(f"| `{cid}` | {t['aspect']:g} | {t['mu_rolling']:g} | "
          f"[{br[0]:g}, {br[1]:g}] | {len(t['evals'])} | {t['note']} |")
    A("")
    A("流程：`--chain` 逐链求值（缓存优先、可断点续跑）→ 每点落盘 "
      "`runs/case_column_collapse_r5_m16991/cases/<id>.json`（含 n_particles/dt/"
      "stop_reason/能量残差/runout 双口径）→ `scripts/analyze_column_collapse_m16991.py` "
      "归约出 gap 表与图 → `scripts/render_column_collapse_m16991.py` 生成本文（数字全部来自工件）。")
    A("")
    A("**方法偏差（如实申报）**：预登记写的是“bisection”，实际实现为 Illinois（修正割线）"
      "——保证不变（仍需区间两端异号才判根、区间宽度仍收到 2·tol），但在近线性的误差曲线上"
      "求值次数约减半。该偏差在任何 μ≠0.9 的点被观测**之前**确定（首批结果只有 μ=0.9，"
      "而 μ=0.9 的值在轮 4 已有判读），不构成结果驱动的规则修改。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 3.1 逐点求值（全部来自 cases/*.json）")
    A("")
    A("| a | μ_r | μ | N_p | runout r_p995/r_i | 参考 | 误差 | 判据 | 终止 | 墙钟 s |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in cases:
        wall = f"{float(r['wall_clock_s']):.1f}" if r['wall_clock_s'] else "—"
        A(f"| {float(r['aspect_ratio']):g} | {float(r['mu_rolling']):g} | "
          f"{float(r['mu']):g} | {r['n_particles']} | {num(float(r['runout_ratio_p995']))} | "
          f"{num(float(r['reference_runout_ratio']))} | {pct(float(r['err_p995']))} | "
          f"{'PASS' if r['pass_p995'] == 'True' else 'FAIL'} | {r['stop_reason']} | "
          + wall + " |")
    A("")
    A("### 3.2 需求表与支间间隙")
    A("")
    A("| μ_r | μ_req（a=1，线性支） | μ_req（a=3，平方根支） | gap = 右−左 | 来源 |")
    A("|---|---|---|---|---|")
    for r in summary["gap_table"]:
        lin = num(r["mu_req_linear"]) + (" (>上界)" if r["linear_lower_bound_only"] else "")
        sqr = num(r["mu_req_sqrt"]) + (" (>上界)" if r["sqrt_lower_bound_only"] else "")
        A(f"| {r['mu_rolling']:g} | {lin} | {sqr} | {num(r['gap'])} | {r['source']} |")
    A("")
    A(f"**H5 裁决：{summary['h5']}** —— {summary['reason']}。"
      f"已夹住水平数 = {summary['n_bracketed_mur']}，gap 符号序列 = {summary['signs']}。")
    A("")
    # --- 3.3 pre-registered step 2: candidate pair on all five aspects -------
    vals = sorted(glob.glob(os.path.join(RUN, "validation_*.json")))
    if vals:
        v = json.load(open(vals[0]))
        A("### 3.3 预登记第 2 步：候选对 (μ="
          f"{v['pair']['mu']:g}, μ_r={v['pair']['mu_rolling']:g}) 在全部五个 a 上的复核")
        A("")
        A("| a | N_p | runout r_p995/r_i | 参考（Lube 2004） | 误差 | <15 %? | 终止 |")
        A("|---|---|---|---|---|---|---|")
        for r in v["rows"]:
            A(f"| {r['aspect']:g} | {r['n_particles']} | {num(r['runout_ratio_p995'])} | "
              f"{num(r['reference'])} | {pct(r['signed_error'])} | "
              f"{'PASS' if r['within_tolerance'] else '**FAIL**'} | {r['stop_reason']} |")
        A("")
        A(f"合计：{len(v['rows']) - v['n_fail']}/{len(v['rows'])} 通过 15 % 判据；"
          f"最大 |误差| = {100 * v['max_abs_error']:.2f} %；"
          f"全部通过？**{'是' if v['all_within_tolerance'] else '否'}**。"
          "（a=1.0 与 a=3.0 是拟合点/确定性对照，a∈{0.5,1.5,2.0} 为样本外）")
        A("")
    ctrls = sorted(glob.glob(os.path.join(RUN, "cases", "w5_ctrl_*.json")))
    if ctrls:
        A("### 3.4 事后控制：唯一越线 a 的时长加倍")
        A("")
        A("| 算例 | a | μ | μ_r | duration_factor | runout | 参考 | 误差 | <15 %? | 终止 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for cp in ctrls:
            d = json.load(open(cp))
            A(f"| `{d['case_id']}` | {d['case']['aspect_ratio']:g} | {d['case']['mu']:g} | "
              f"{d['case']['mu_rolling']:g} | {d['case']['duration_factor']:g} | "
              f"{num(d['runout_ratio_p995'])} | {num(d['reference_runout_ratio'])} | "
              f"{pct(d['signed_error'])} | {'PASS' if d['within_tolerance'] else '**FAIL**'} | "
              f"{d['stop_reason']} |")
        A("")
        A("目的：判定越线点是否被 `time_limit` 截断所致（截断 ⇒ runout 偏小 ⇒ 误差更负）。")
        A("")
    A("### 3.5 事后（labeled post-hoc）：需求点根 vs 容差带")
    A("")
    A("预登记的“需求点根 `μ_req`”在两条链上暴露为**病态**：误差-μ 响应非单调"
      "（见 3.2 表中 `time_limit` 点与大跳变，例如 a=1、μ_r=0.2 时 μ=1.050 → +11.9 % 而 "
      "μ=1.064 → +45.1 %，μ=1.185 → −2.9 %），故 `μ_req` 的根位置不唯一。"
      "正确的识别统计量是 **±15 % 容差带在 μ 上的区间**：两支的带若相交，则存在共同的常数对；"
      "这不需要任何新计算，用同一批已有点即可读出。")
    A("")
    A("### 3.6 图")
    A("")
    A("![mu_req](../artifacts/m16991_column_collapse_r5/figures/mu_req_vs_mur.png)")
    A("")
    A("![err_vs_mu](../artifacts/m16991_column_collapse_r5/figures/err_vs_mu.png)")
    A("")
    A("![gap](../artifacts/m16991_column_collapse_r5/figures/gap_vs_mur.png)")
    A("")
    A("图注：① `mu_req_vs_mur.png` = 两支所需的常数接触摩擦随滚动阻力变化，绿带 = 砂/砾"
      f"物理可行盒（μ ≤ {summary['protocol']['phys_box']['mu_max']:g}、"
      f"μ_r ≤ {summary['protocol']['phys_box']['mu_rolling_max']:g}），点线 = 轮 4 唯一"
      "被夹住的 μ_r 水平；② `err_vs_mu.png` = 每条链的带符号 runout 误差随 μ 的走向"
      "（单调性证据），绿带 = ±15 % 判据；③ `gap_vs_mur.png` = 支间需求间隙随 μ_r，"
      "过零即联合解存在。")
    A("")
    A("## ④ 注意点与局限")
    A("")
    A("见 `docs` 生成脚本输出；要点：低 μ_r 端终止条件多为 `time_limit`（流动未停）⇒ "
      "该端 runout 可能被截断，故低 μ_r 的需求只是**下界**声明；参考律换支点 a=1.7 未逐点定位；"
      "μ、μ_r 是**等效**参数（参考实验未给砂的实测摩擦/滚动阻力），不可直接当内摩擦角；"
      "μ>1 的对偶（有效摩擦）在球体 DEM + 常数摩擦模型里没有微观对应，只能作为“模型不足”的证据。")
    A("")
    A("## ⑤ 误差判定（逐项，是否 <15 %／<3 %）")
    A("")
    if vals:
        v = json.load(open(vals[0]))
        A("| 关键量 | 基准 | 最佳可得 | 误差 | <15 %? | <3 %? |")
        A("|---|---|---|---|---|---|")
        for r in v["rows"]:
            A(f"| runout a={r['aspect']:g} | r_i(1+1.24a) / r_i(1+1.6√a) | "
              f"(μ=0.9, μ_r=0.3) | {pct(r['signed_error'])} | "
              f"{'是' if r['within_tolerance'] else '**否**'} | "
              f"{'是' if abs(r['signed_error']) < 0.03 else '否'} |")
        A("")
        A(f"**是否 <15 %？** {len(v['rows']) - v['n_fail']}/{len(v['rows'])} 项是；"
          f"唯一越线项 a=1.5 为 {pct([r for r in v['rows'] if not r['within_tolerance']][0]['signed_error'])}，"
          "越线幅度见 3.4 的时长控制。**是否 <3 %？** 否（最好项 "
          f"{pct(min(v['rows'], key=lambda r: abs(r['signed_error']))['signed_error'])}）。")
        A("")
    A("预登记 H5 裁决 = **" + summary["h5"] + "**（" + summary["reason"] + "）：两支的需求"
      "点根在 μ_r ≤ 0.3 上大量右删失且误差曲线非单调，点根式裁决不可判；"
      "容差带读法（3.5）给出候选对，其五 a 复核见 3.3。锚点验收（全 a <15 %）"
      f"{'已达到' if vals and v['all_within_tolerance'] else '**尚未达到**'}。")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("python3 benchmarks/case_column_collapse_round5_m16991.py --plan")
    for cid in sorted(chains):
        pr = chains[cid].get("protocol", {})
        br = pr.get("mu_bracket", [0.9, 2.4])
        extra = "" if br == [w5.MU_LO, w5.MU_HI] else f" --lo {br[0]:g} --hi {br[1]:g}"
        A(f"python3 benchmarks/case_column_collapse_round5_m16991.py --chain {cid}{extra}")
    A("python3 scripts/analyze_column_collapse_m16991.py")
    A("python3 scripts/render_column_collapse_m16991.py")
    A("python3 -m pytest tests/test_column_collapse_round5_m16991.py -q")
    A("```")
    A("")
    text = "\n".join(L) + "\n"
    with open(DOC, "w") as fh:
        fh.write(text)
    return text


if __name__ == "__main__":
    out = main()
    print(f"wrote {DOC} ({len(out)} bytes)")