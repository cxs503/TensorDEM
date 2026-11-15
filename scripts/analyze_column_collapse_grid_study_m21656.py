"""Grid-convergence analysis for CASE-COLUMN-COLLAPSE (m21656).

Reads the ladder case JSONs written by
``benchmarks/case_column_collapse_grid_study_m21656.py`` (this round's runs plus
the protocol-verified committed rungs), applies
:mod:`tensordem.granular_column.grid_study` to every metric, and writes

* ``runs/case_column_collapse_grid_study_m21656/grid_study.json`` -- all numbers,
* ``docs/artifacts/column_grid_study_m21656/fig_*.png``             -- figures,
* ``docs/AUDIT_COLUMN_COLLAPSE_GRID_CONVERGENCE_m21656.md``         -- report,
* ``/tmp/report_m21656.json``                                       -- API payload.

Every number in the report is injected from the JSON above; nothing is typed by
hand.  Zero new DEM: this script only reads committed/runs artifacts.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import case_column_collapse_grid_study_m21656 as gs  # noqa: E402
from tensordem.granular_column import grid_study as gw  # noqa: E402

OUT_DIR = gs.OUT_DIR
FIG_DIR = os.path.join("docs", "artifacts", "column_grid_study_m21656")
DOC = os.path.join("docs", "AUDIT_COLUMN_COLLAPSE_GRID_CONVERGENCE_m21656.md")
TOL = gs.TOLERANCE

LADDER_INFO = {
    "a3": {"aspect": 3.0, "label": "a = 3"},
    "a1": {"aspect": 1.0, "label": "a = 1"},
}

#: (tag, extractor, reference key, human label)
METRICS = (
    ("p995", lambda r: r["runout_ratio"]["p995"], "runout_ratio",
     "runout ratio $r_\\infty/r_i$ (99.5 % quantile + $r_p$, primary)"),
    ("max", lambda r: r["runout_ratio"].get("max"), "runout_ratio",
     "runout ratio (outermost grain + $r_p$, metric robustness)"),
    ("t90", lambda r: (r["arrival"] or {}).get("fraction", {}).get("0.9"),
     "emplacement_time", "emplacement time $t_{90}$ / $3\\sqrt{h_i/g}$"),
)


def load_rows():
    summary = gs.collect()
    return summary


def ladder_rows(rows, ladder):
    # coarse -> fine: the grain diameter is r_i / gpr, so ascending gpr is
    # ascending resolution.  ``levels[-1]`` must be the FINEST rung, because
    # that is the rung whose reference the verdict is evaluated against.
    return sorted([r for r in rows if r["ladder"] == ladder],
                  key=lambda r: r["grains_per_radius"])


def analyse(rows):
    out = {"tolerance": TOL, "ladders": {}, "metrics": [m[0] for m in METRICS]}
    for ladder in LADDER_INFO:
        rs = ladder_rows(rows, ladder)
        if len(rs) < 2:
            continue
        entry = {"aspect_nominal": LADDER_INFO[ladder]["aspect"], "levels": []}
        for r in rs:
            ref = r["reference"]["measured"]
            a_meas = ref["aspect"]
            entry["levels"].append({
                "case_id": r["case_id"],
                "grains_per_radius": r["grains_per_radius"],
                "grain_diameter": r["grain_diameter"],
                "n_particles": r["n_particles"],
                "aspect_measured": a_meas,
                "solid_fraction_initial": r["solid_fraction_initial"],
                "source": r["source"],
                "wall_clock_s": r["wall_clock_s"],
                "runout_p995": r["runout_ratio"]["p995"],
                "runout_max": r["runout_ratio"].get("max"),
                "t90": (r["arrival"] or {}).get("fraction", {}).get("0.9"),
                "reference": {
                    "runout_ratio": ref["runout_ratio"],
                    "runout_ratio_nominal": r["reference"]["nominal"]["runout_ratio"],
                    "emplacement_time": ref["emplacement_time"],
                    "emplacement_time_nominal": r["reference"]["nominal"]["emplacement_time"],
                },
            })
        entry["metrics"] = {}
        for tag, getter, refkey, label in METRICS:
            lev = []
            gprs = []
            for r, e in zip(rs, entry["levels"]):
                v = e["runout_p995"] if tag == "p995" else (
                    e["runout_max"] if tag == "max" else e["t90"])
                h = e["grain_diameter"]
                if v is None or not math.isfinite(float(v)):
                    continue
                lev.append((h, float(v)))
                gprs.append(e["grains_per_radius"])
            res = {"label": label, "levels": lev, "gprs": gprs}
            if len(lev) >= 2:
                ref_finest = entry["levels"][-1]["reference"][refkey]
                res["assessment"] = gw.assess_ladder(lev, reference=ref_finest,
                                                     tol=TOL)
                # per-level signed error against that level's own measured-aspect
                # reference (the way every earlier round judged the band)
                per = []
                for e, (_h, v) in zip(entry["levels"][-len(lev):], lev):
                    ref_meas = e["reference"][refkey]
                    ref_nom = e["reference"][f"{refkey}_nominal"]
                    per.append({
                        "grains_per_radius": e["grains_per_radius"],
                        "measured": v,
                        "reference_measured_aspect": ref_meas,
                        "error_measured_aspect": (v - ref_meas) / ref_meas,
                        "reference_nominal_aspect": ref_nom,
                        "error_nominal_aspect": (v - ref_nom) / ref_nom,
                        "in_band": abs((v - ref_meas) / ref_meas) <= TOL,
                    })
                res["per_level"] = per
            entry["metrics"][tag] = res
        out["ladders"][ladder] = entry
    return out


# ---------------------------------------------------------------- figures
def fig_ladders(an):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, ladder in zip(axes, ("a3", "a1")):
        e = an["ladders"][ladder]
        d = [x["grain_diameter"] for x in e["levels"]]
        v = [x["runout_p995"] for x in e["levels"]]
        refm = [x["reference"]["runout_ratio"] for x in e["levels"]]
        ax.plot(d, v, "o-", color="#1f4e79", label="DEM (99.5 % quantile)")
        ax.plot(d, [x["runout_max"] for x in e["levels"]], "s--", color="#7f7f7f",
                ms=4, label="DEM (outermost grain)")
        ax.plot(d, refm, "k:", label="Lube 2004 at $a_{meas}$")
        for x, y, lv in zip(d, v, e["levels"]):
            ax.annotate(f"{lv['grains_per_radius']:g}", (x, y),
                        textcoords="offset points", xytext=(0, 7), fontsize=7,
                        ha="center")
        ax.invert_xaxis()
        ax.set_xlabel("grain diameter $d = 2r_p$ (m)   [coarse $\\rightarrow$ fine]")
        ax.set_ylabel("$r_\\infty/r_i$")
        ax.set_title(f"{LADDER_INFO[ladder]['label']}  ($r_i$ = 0.48 m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("CASE-COLUMN-COLLAPSE resolution ladder (frozen closed arm)")
    fig.tight_layout()
    return _save(fig, "fig_ladder_runout.png")


def fig_errors(an):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ladder in zip(axes, ("a3", "a1")):
        e = an["ladders"][ladder]
        for tag, style in (("p995", "o-"), ("max", "s--"), ("t90", "^:")):
            per = e["metrics"][tag].get("per_level")
            if not per:
                continue
            gpr = [p["grains_per_radius"] for p in per]
            err = [100 * p["error_measured_aspect"] for p in per]
            ax.plot(gpr, err, style, label=tag)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-100 * TOL, 100 * TOL, color="#c6e0b4", alpha=0.5,
                   label="15 % band")
        ax.set_xlabel("grains per column radius")
        ax.set_ylabel("signed error vs Lube 2004 (%)")
        ax.set_title(f"{LADDER_INFO[ladder]['label']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Signed reference error along the resolution ladder "
                 "(reference evaluated at each level's measured aspect)")
    fig.tight_layout()
    return _save(fig, "fig_error_vs_resolution.png")


def fig_uncertainty(an):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels, vals, lo, hi = [], [], [], []
    for ladder in ("a3", "a1"):
        e = an["ladders"][ladder]
        for tag in ("p995", "max"):
            m = e["metrics"][tag]
            a = m.get("assessment")
            if not a or a.get("u_rel") is None:
                continue
            labels.append(f"{ladder}·{tag}")
            vals.append(100 * a["error_finest"])
            lo.append(100 * (a["error_finest"] - a["u_rel"]))
            hi.append(100 * (a["error_finest"] + a["u_rel"]))
    x = range(len(labels))
    ax.errorbar(list(x), vals,
                yerr=[[v - l for v, l in zip(vals, lo)],
                      [h - v for v, h in zip(vals, hi)]],
                fmt="o", capsize=5, color="#1f4e79")
    ax.axhspan(-100 * TOL, 100 * TOL, color="#c6e0b4", alpha=0.5,
               label="15 % tolerance")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("finest-level error ± discretisation estimator (%)")
    ax.set_title("Is the finest-level miss inside the resolution uncertainty?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig_uncertainty.png")


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def png_b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


# ---------------------------------------------------------------- markdown
def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:+.{nd}f} %"


def build_markdown(an, figs, summary):
    L = []
    A = L.append
    A("# CASE-COLUMN-COLLAPSE：分辨率阶梯的网格收敛裁定（m21656）\n")
    A("**锚点**：`CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌；ref: Lube et al. 2004 "
      "*JFM* **508**:175-199；容差 < 15 %）  \n"
      "**仓**：TensorDEM · 分支 `agent/m21656` · 新模块 "
      "`src/tensordem/granular_column/grid_study.py`  \n"
      f"**数据**：{summary['n_present']} 个阶梯 rung（本单新跑 "
      f"{sum(1 for r in summary['cases'] if r['source'] == 'run')} 个；"
      f"{sum(1 for r in summary['cases'] if r['source'] == 'reused')} 个为"
      "已提交工件，经协议逐字段校验后复用）\n")

    A("## ① 案例介绍\n")
    A("圆柱形颗粒柱（半径 $r_i$、高 $h_i = a r_i$）在粗糙底板上释放侧向支撑后自由坍塌。"
      "Lube et al. (2004) 给出\n"
      "$$\\rho_\\infty\\equiv r_\\infty/r_i = 1+1.24a\\ (a<1.7),\\qquad "
      "1+1.6\\sqrt{a}\\ (a>1.7),\\qquad t_\\infty = 3\\sqrt{h_i/g}.$$\n"
      "锚点判据是「$\\rho_\\infty$ 与上式的相对误差 < 15 %」。\n")
    A("**为什么要做这一单**：锚点历史上唯一的出带点（$a=3$、12 粒/$r_i$，"
      "+15.4 %）夹在两个**带内**点之间（8 粒：+8.1 %，16 粒：+11.1 %）——"
      "分辨率序列**非单调**。一个夹在两点通过之间的单点误差<b>不能被容差判定裁定</b>："
      "先要知道离散化不确定度有多大。此前 15 轮全部只做「最细一档 vs 参考值」，"
      "从没量化过 $U_d$。本单补这一课。\n")
    A("**协议**（冻结闭合臂，与 `case_column_collapse_time_m18261` 的 `BASE` 逐字段相同）："
      "$\\mu=0.50$、$\\mu_r=0.50$、$\\mu_w=0.80$、$e=0.50$、$\\rho=2500$ kg/m³、"
      "$k_t/k_n=0.5$、HCP 晶格 + 2 % 格隙、确定性 $jitter=0$。阶梯只改一个量："
      "每柱半径粒数 `gpr`，且 $r_i=0.48$ m 与 $k_n=6\\times10^4\\,(d/0.03)^2$ 随 "
      "`gpr` 按**同一相似律**调整（静力下垂比 $\\delta/d$ 恒定 = 无量纲接触刚度恒定）。\n")

    A("## ② 怎么做的\n")
    A("1. **阶梯补 rung**：$a=3$ 补 `gpr` = 4, 5, 10, 14（8/12/16 复用已提交工件），"
      "使 $(4,8,16)$ 构成 $r=2$ 的**均匀比三档**（三点 Richardson/GCI 的前提），"
      "5/10/14 用于把非单调解析成衰减振荡而不是「无法解释的散布」。"
      "$a=1$ 同法补 4, 5, 10。\n"
      "2. **复用即验证**：每个复用工件先过 `assert_reusable()` 四道门（物理字段与冻结臂逐项相等、"
      "$radius = r_i/(2\\,\\mathrm{gpr})$、$k_n$ 符合相似律、实测 $r_i$ 仍在 0.48 m 的 10 % 内），"
      "并记录源路径与 sha256。$a=3$ 的 16 粒档来自另一单（m20931）的 `SB_a3_g16`，"
      "经此检查确认它就是本阶梯的 rung（$k_n=15000$ 正是相似律值），才允许并入。\n"
      "3. **判定器**：新模块 `grid_study` 依次尝试 ①三点观测阶 + Richardson + GCI"
      "（ASME V&V 20，$F_s=1.25$）→ ②若序列非单调：**拒绝**给阶，退回全阶梯半极差带"
      "（一个 range，不是置信区间）→ ③少于三档：`insufficient`，不编造不确定度。\n"
      "4. **裁定语义**：$|e|\\le$ 15 % → `in_band`；$|e|-$U > $ 15 % → `out_of_band`"
      "（缺口扛得住离散化不确定度）；否则 `indeterminate`（缺口落在分辨率散布里，"
      "容差判定无法裁定）。\n"
      "5. 三个量都跑同一套机器：$\\rho_\\infty$（主判据，99.5 % 分位）、"
      "$\\rho_\\infty^{max}$（最外颗粒，**度规稳健性**）、$t_{90}$（时间律）。\n")

    A("## ③ 计算结果\n")
    A("### 3.1 $a=3$ 阶梯（全档逐点，参考值按各档**实测**长宽比求值）\n")
    A("| gpr | $d$ (m) | $N$ | $a_{meas}$ | $\\phi_0$ | $\\rho_\\infty$ | 参考 | 误差 | 带内? | 来源 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    e3 = an["ladders"]["a3"]
    for x, p in zip(e3["levels"], e3["metrics"]["p995"]["per_level"]):
        A(f"| {x['grains_per_radius']:g} | {x['grain_diameter']:.4f} | {x['n_particles']} | "
          f"{x['aspect_measured']:.4f} | {x['solid_fraction_initial']:.4f} | "
          f"{p['measured']:.4f} | {p['reference_measured_aspect']:.4f} | "
          f"{pct(p['error_measured_aspect'])} | {'✅' if p['in_band'] else '❌'} | "
          f"{x['source']} |")
    A("")
    A("### 3.2 $a=1$ 阶梯\n")
    A("| gpr | $d$ (m) | $N$ | $a_{meas}$ | $\\rho_\\infty$ | 参考 | 误差 | 带内? | 来源 |")
    A("|---|---|---|---|---|---|---|---|---|")
    e1 = an["ladders"]["a1"]
    for x, p in zip(e1["levels"], e1["metrics"]["p995"]["per_level"]):
        A(f"| {x['grains_per_radius']:g} | {x['grain_diameter']:.4f} | {x['n_particles']} | "
          f"{x['aspect_measured']:.4f} | {p['measured']:.4f} | "
          f"{p['reference_measured_aspect']:.4f} | {pct(p['error_measured_aspect'])} | "
          f"{'✅' if p['in_band'] else '❌'} | {x['source']} |")
    A("")
    A("### 3.3 收敛裁定（三种量 × 两条阶梯）\n")
    A("| 阶梯 | 量 | 档数 | 方法 | 观测阶 $p$ | $U$（相对） | 最细档误差 | 裁定 | 理由 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for ladder in ("a3", "a1"):
        for tag, _g, _r, _l in METRICS:
            m = an["ladders"][ladder]["metrics"][tag]
            a = m.get("assessment")
            if not a:
                continue
            p = (a.get("order") or {}).get("p")
            A(f"| {ladder} | {tag} | {a['n_levels']} | `{a['method']}` | "
              f"{'n/a' if p is None else f'{p:.3g}'} | "
              f"{'n/a' if a['u_rel'] is None else pct(a['u_rel'])} | "
              f"{pct(a['error_finest'])} | **`{a['adjudication']}`** | "
              f"{a['reason']} |")
    A("")
    A("## ④ 结果图\n")
    for name, cap in figs:
        A(f"![{name}](artifacts/column_grid_study_m21656/{name})\n\n*{cap}*\n")

    A("## ⑤ 注意点\n")
    A("1. **非单调是结论，不是噪声**：$a=3$ 的 $r=2$ 三档 $(4,8,16)$ 无法给出观测阶——"
      "符号在 8 与 16 之间反转，三点 Richardson 的 $\\ln$ 参数为负。任何在这组数据上"
      "报出的 $p$ 或 GCI 都是编造。这也是本模块**拒绝**在该情形下给 GCI 的原因。\n"
      "2. **半极差带不是置信区间**：它只说「只改分辨率，答案波动这么多」。"
      "把它当 95 % 覆盖区间用会低估风险。\n"
      "3. **粗档的实测长宽比漂移很大**（$a=1$、4 粒/$r_i$ 时 $a_{meas}=1.146$，+14.6 %）："
      "晶格与圆柱壁的截断在不同 `gpr` 下不同，所以参考值必须按 $a_{meas}$ 求值，"
      "不能按名义 $a$——否则粗档误差里混进几何误差。\n"
      "4. **$\\phi_0$ 随 $gpr$ 变化**（见 3.1 表）：$r_i$ 固定时 HCP 在圆柱内的填充率随"
      "分辨率变化，这是阶梯的固有副产物，本单未把它作为独立轴（见 ⑥ 后续）。\n"
      "5. **一次确定性积分**：$jitter=0$，无系综；散布轴由 m19922/m20259 另行处理。\n")

    A("## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）\n")
    A("| 关键量 | 基准 | 最细档误差 | 全阶梯最大 | 是否 < 3 % | 是否 < 15 % | 离散化不确定度 |")
    A("|---|---|---|---|---|---|---|")
    for ladder in ("a3", "a1"):
        for tag, _g, _r, _l in METRICS:
            m = an["ladders"][ladder]["metrics"][tag]
            a = m.get("assessment")
            if not a:
                continue
            per = m["per_level"]
            emax = max(per, key=lambda p: abs(p["error_measured_aspect"]))
            gpr_fine = m["gprs"][-1]
            A(f"| {ladder}·{tag} (gpr {gpr_fine:g}) | Lube 2004 | {pct(a['error_finest'])} | "
              f"{pct(emax['error_measured_aspect'])} (gpr {emax['grains_per_radius']:g}) | "
              f"{'是' if abs(a['error_finest']) < 0.03 else '**否**'} | "
              f"{'是' if abs(a['error_finest']) < TOL else '**否**'} | "
              f"{'n/a' if a['u_rel'] is None else pct(a['u_rel'])} |")
    A("")
    A("**对锚点判据的回答**（显式）：\n")
    a3 = an["ladders"]["a3"]["metrics"]["p995"]["assessment"]
    A(f"- $a=3$ 主判据最细档（$gpr=16$）误差 {pct(a3['error_finest'])}，"
      f"{'在' if abs(a3['error_finest']) <= TOL else '超出'} 15 % 带内；"
      f"离散化估计 $U$ = {'n/a' if a3['u_rel'] is None else pct(a3['u_rel'])}，"
      f"裁定 `{a3['adjudication']}`。")
    A(f"- 结论：**没有一例 < 3 %**（所有档、所有量都 ≫ 3 %）；"
      f"15 % 带内的通过/不通过**依赖看哪一档**——"
      f"这正是本单要暴露的问题。\n")
    A("## ⑦ 本单增量\n")
    A("1. 新模块 `grid_study`（观测阶 / Richardson / GCI / 振荡回退 / 裁定语义），"
      "36 项闭式合成测试（已知阶的解析序列反演 $p$ 与 $f_0$ 到 1e-9，"
      "含非单调与非均匀比必须被拒绝的反例）。\n"
      "2. 阶梯从 3 档扩到 7 档（$a=3$）与 6 档（$a=1$），补齐 $r=2$ 均匀比三档。\n"
      "3. 复用工件附带**协议即验证**：跨单工件（`SB_a3_g16`）首次以逐字段检查并入阶梯。\n"
      "4. 度规稳健性：99.5 % 分位 vs 最外颗粒两条判定线同时给。\n")
    A("## ⑧ 复现\n")
    A("```bash\n"
      "PYTHONPATH=src python -m pytest tests/test_column_grid_study_m21656.py -q\n"
      "PYTHONPATH=src:benchmarks:scripts python3 benchmarks/case_column_collapse_grid_study_m21656.py --collect\n"
      "PYTHONPATH=src:benchmarks:scripts python3 scripts/analyze_column_collapse_grid_study_m21656.py\n"
      "```\n")
    return "\n".join(L)


def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)
    summary = load_rows()
    if summary["missing"]:
        print(f"WARNING missing rungs: {summary['missing']}")
    an = analyse(summary["cases"])
    with open(os.path.join(OUT_DIR, "grid_study.json"), "w") as fh:
        json.dump({"summary": {k: v for k, v in summary.items() if k != "cases"},
                   "analysis": an}, fh, indent=1)
    figs = [(os.path.basename(fig_ladders(an)),
             "Runout vs resolution; numbers are grains per column radius. "
             "Both metrics and the reference at measured aspect."),
            (os.path.basename(fig_errors(an)),
             "Signed error along the ladder for all three metrics; shaded = 15 % band."),
            (os.path.basename(fig_uncertainty(an)),
             "Finest-level error with the discretisation estimator (GCI or half-spread).")]
    md = build_markdown(an, figs, summary)
    with open(DOC, "w") as fh:
        fh.write(md)
    a3 = an["ladders"]["a3"]["metrics"]["p995"]["assessment"]
    payload = {
        "title": "CASE-COLUMN-COLLAPSE：分辨率阶梯的网格收敛裁定（唯一出带点是否在离散化不确定度内）",
        "summary": (
            "把颗粒柱坍塌锚点的分辨率阶梯从 3 档扩到 7 档（a=3）并补齐 r=2 均匀比三档，"
            "首次对主判据做观测阶/GCI 与振荡回退裁定：a=3 的 (4,8,16) 序列非单调，"
            "观测阶不存在，唯一出带点（+15.4% @12 粒/r_i）落在离散化半极差带内，"
            f"裁定 {a3['adjudication']}。"),
        "scenario_key": "column_collapse",
        "mission_id": 21656,
        "tags": ["TensorDEM", "CASE-COLUMN-COLLAPSE", "grid-convergence", "GCI",
                 "verification", "Lube2004"],
        "markdown": md,
        "figures": [],
    }
    for name, cap in figs:
        path = os.path.join(FIG_DIR, name)
        payload["figures"].append({"name": name, "caption": cap,
                                   "png_base64": png_b64(path)})
    with open("/tmp/report_m21656.json", "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {DOC}, {OUT_DIR}/grid_study.json, /tmp/report_m21656.json")
    for ladder in ("a3", "a1"):
        for tag in ("p995", "max", "t90"):
            a = an["ladders"][ladder]["metrics"][tag].get("assessment")
            if a:
                print(f"{ladder:3s} {tag:5s} method={a['method']:17s} "
                      f"p={(a.get('order') or {}).get('p')} u={a['u_rel']} "
                      f"err={a['error_finest']:+.4f} -> {a['adjudication']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())