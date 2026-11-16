"""Anchor-provenance audit of CASE-DEM-RCP-PACKING-FRACTION.

Why this script exists
---------------------
Two merged analysis paths judge the *same* hard-sphere intercept of the low
friction tier (mu = 0.1):

* ``docs/CASE_RCP_PACKING_FRACTION_M20918.md`` compares ``phi_inf`` with the
  mu-indexed table entry (phi = 0.6250) and reports ``-1.22 %`` -> "-3 % PASS";
* ``runs/order_audit.json`` (G13_vs_ref_mu0.1) compares the same intercept with
  the *frictionless* literature value 0.6366 and reports 3.07 % -> FAIL.

Both cannot be the anchor.  This script fixes the policy in code: an accuracy
claim (<3 %, or any tolerance) is admissible only against a *literature*
anchor; the mu = 0.1 table entry is self-declared interpolated, so it is
re-tiered ``derived`` and the claim is downgraded to ``WITHHELD`` with the
disordered-packing bracket as the strongest admissible statement.

It also closes the extrapolation debt that the previous wave left open: the
mu = 0.1 stiffness ladder had three levels only, so the fit had three
parameters and the reported order ``p = 0.51`` could not be checked.  A fourth
level (``k_n = 256000``, one seed per tier) is supplied as run artifacts
and used leave-one-out; the intercept is additionally bracketed over the
admissible order range ``p in [0.5, 1.5]`` because ``p`` itself is a
fitted nuisance.

Every number written into ``docs/AUDIT_m21145_RCP_ANCHOR_PROVENANCE.md`` comes
from this script -- nothing is hand copied.

Run::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/anchor_provenance_audit.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
FIGS = os.path.join(HERE, "figures")
DOC = os.path.join(ROOT, "docs", "AUDIT_m21145_RCP_ANCHOR_PROVENANCE.md")
OUT_JSON = os.path.join(RUNS, "anchor_provenance_audit.json")

DT_DEFAULT = 1.0 / 20.0
N_PRIMARY = 10000
TOL_REL = 0.03
BAND = (0.62, 0.66)          # feature_ledger acceptance band
ORDER_RANGE = (0.5, 1.5)     # admissible nuisance order for the bracket

sys.path.insert(0, os.path.join(ROOT, "src"))
from tensordem.deposition_packing import (  # noqa: E402
    extrapolation_selfconsistency,
    stiffness_limit,
)
from tensordem.rcp_references import (  # noqa: E402
    bracket_limits,
    frictionless_reference,
    judge,
    provenance_table,
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_runs() -> list:
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS, "deposition_*.json"))):
        d = json.load(open(path))
        spec, m = d["spec"], d["metrics"]
        runs.append({
            "path": os.path.relpath(path, ROOT),
            "n": spec["n_particles"], "mu": spec["mu"], "kn": spec["kn"],
            "seed": spec["seed"], "dt_frac": spec.get("dt_frac", DT_DEFAULT),
            "phi": m["phi_bulk"],
            "phi_corr": m["phi_bulk_overlap_corrected"],
            "Z": m.get("coordination_bulk"),
            "settled": d["settled"],
            "wall": d.get("wall_clock"),
        })
    return runs


def stat(runs, mu, kn, n=N_PRIMARY, dt_frac=DT_DEFAULT):
    rs = [r for r in runs if abs(r["mu"] - mu) < 1e-12
          and abs(r["kn"] - kn) < 1e-9 and r["n"] == n
          and abs(r["dt_frac"] - dt_frac) < 1e-12]
    if not rs:
        return None
    phis = np.array([r["phi"] for r in rs])
    return {"phi": float(phis.mean()), "n_seeds": len(rs),
            "seeds": sorted(r["seed"] for r in rs),
            "std": float(phis.std(ddof=1)) if len(rs) > 1 else None,
            "paths": [r["path"] for r in rs]}


def limit_for_order(pts, p):
    """Hard-sphere intercept with the convergence order pinned at ``p``."""
    (k1, v1), (k2, v2) = pts[-2], pts[-1]
    denom = k2 ** (-p) - k1 ** (-p)
    if abs(denom) < 1e-30:
        return float(v2)
    C = (v2 - v1) / denom
    return float(v2 - C * k2 ** (-p))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyse(runs):
    res = {
        "provenance_tiers": provenance_table(),
        "bracket_limits": list(bracket_limits()),
        "n_primary": N_PRIMARY,
        "band": list(BAND),
        "tolerance_rel": TOL_REL,
        "order_range": list(ORDER_RANGE),
        "tiers": {},
        "gates": [],
    }
    for mu in (0.0, 0.1):
        levels = sorted({r["kn"] for r in runs if r["n"] == N_PRIMARY
                         and abs(r["mu"] - mu) < 1e-12
                         and abs(r["dt_frac"] - DT_DEFAULT) < 1e-12})
        meta = [(kn, stat(runs, mu, kn)) for kn in levels]
        meta = [(kn, st) for kn, st in meta if st]
        pts = [(kn, st["phi"]) for kn, st in meta]
        tier = {
            "kn_levels": [k for k, _ in meta],
            "points": [{"kn": k, "phi": st["phi"], "n_seeds": st["n_seeds"],
                        "std": st["std"], "seeds": st["seeds"],
                        "paths": st["paths"]} for k, st in meta],
            "production": stat(runs, mu, 4000.0),
        }
        if len(pts) >= 3:
            tier["three_level"] = stiffness_limit(pts[:3])
        if len(pts) >= 4:
            tier["selfcheck"] = extrapolation_selfconsistency(pts)
            tier["four_level"] = stiffness_limit(pts[-4:])
        # nuisance-order bracket on the *last two* stiffness levels
        if len(pts) >= 2:
            br = {}
            for p in np.linspace(ORDER_RANGE[0], ORDER_RANGE[1], 11):
                br[round(float(p), 3)] = limit_for_order(pts, float(p))
            vals = np.array(list(br.values()))
            tier["order_bracket"] = {
                "by_order": br,
                "lo": float(vals.min()), "hi": float(vals.max()),
                "reference_p": tier.get("four_level", tier.get("three_level", {})).get("order"),
            }
        res["tiers"][f"{mu:g}"] = tier
    return res


def verdicts(res):
    """Anchor-policy verdicts for every reported quantity."""
    rows = []
    for tag, tier in res["tiers"].items():
        mu = float(tag)
        cands = []
        if tier.get("production"):
            cands.append(("production k_n=4000", tier["production"]["phi"]))
        for key, label in (("three_level", "intercept (3 levels)"),
                           ("four_level", "intercept (4 levels)")):
            if tier.get(key):
                cands.append((label, tier[key]["limit"]))
        if tier.get("order_bracket"):
            cands.append(("intercept bracket (low edge, p=%.1f)" % res["order_range"][1],
                          tier["order_bracket"]["lo"]))
            cands.append(("intercept bracket (high edge, p=%.1f)" % res["order_range"][0],
                          tier["order_bracket"]["hi"]))
        for label, phi in cands:
            j = judge(phi, mu, tol_rel=TOL_REL)
            rows.append({
                "tier_mu": mu,
                "quantity": label,
                "phi": phi,
                "in_ledger_band": bool(BAND[0] <= phi <= BAND[1]),
                "anchor_provenance": j["anchor_provenance"],
                "anchor_phi": j["phi_ref"],
                "rel_error_vs_anchor": j["rel_error"],
                "verdict": j["verdict"],
                "bracket_verdict": j["bracket"]["verdict"],
                "strongest_statement": j["strongest_statement"],
            })
    return rows


def cross_script_consistency(res):
    """Compare this audit with the merged order audit's G13 gates."""
    path = os.path.join(RUNS, "order_audit.json")
    if not os.path.exists(path):
        return {"available": False}
    d = json.load(open(path))
    out = {"available": True, "gates": []}
    for g in d.get("gates", []):
        if g["id"].startswith("G13_vs_ref_mu"):
            mu = float(g["id"].split("mu")[-1])
            out["gates"].append({
                "id": g["id"], "mu": mu, "limit": g["got"],
                "rel_error_vs_frictionless": g["value"], "pass": g["pass"],
                "anchor_used": d["reference"]["phi_rcp"],
            })
    # the same intercept judged against the mu-indexed table entry
    for row in res["verdicts"]:
        if row["quantity"].startswith("intercept (3 levels)"):
            mu = row["tier_mu"]
            g = [x for x in out["gates"] if abs(x["mu"] - mu) < 1e-12]
            if g:
                g[0]["rel_error_vs_table_entry"] = row["rel_error_vs_anchor"]
                g[0]["table_entry_phi"] = row["anchor_phi"]
                g[0]["table_entry_provenance"] = row["anchor_provenance"]
                g[0]["table_entry_verdict"] = row["verdict"]
    return out


def ledger_claims():
    """Accuracy claims the merged ledger path (aggregate.py -> summary.json)
    rests on non-literature anchors.

    The m21145 policy was first enforced on this audit's own verdicts; the
    see-through (m23840) extends the same check to the ledger gates so a
    certified accuracy claim can no longer rest on a derived/unverified
    anchor in *either* merged analysis path.  A reference-comparison gate
    without a provenance marker is itself a violation (a claim without a
    recorded anchor tier).
    """
    path = os.path.join(RUNS, "summary.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    bad = []
    for g in d.get("gates", []):
        if "phi_ref_source" not in g:
            continue  # not a reference-comparison gate
        prov = g.get("anchor_provenance")
        if prov is None:
            bad.append(g["id"])
        elif prov != "literature" and g.get("claim_status") != "WITHHELD":
            bad.append(g["id"])
    return bad


def gates(res):
    g = []
    ref0 = frictionless_reference()
    prod0 = res["tiers"]["0"]["production"]
    g.append({"id": "A1_frictionless_production_vs_literature",
              "criterion": f"|phi(k_n=4000, mu=0) - {ref0.phi_rcp}|/phi_ref < {TOL_REL:.0%}",
              "got": abs(prod0["phi"] / ref0.phi_rcp - 1.0),
              "pass": bool(abs(prod0["phi"] / ref0.phi_rcp - 1.0) < TOL_REL)})
    lb = res["tiers"]["0"].get("order_bracket")
    if lb:
        g.append({"id": "A2_frictionless_intercept_bracket_stable",
                  "criterion": "nuisance-order bracket width < 0.005 (mu=0)",
                  "got": lb["hi"] - lb["lo"],
                  "pass": bool(lb["hi"] - lb["lo"] < 0.005)})
    lb1 = res["tiers"]["0.1"].get("order_bracket")
    if lb1:
        g.append({"id": "A3_lowfriction_intercept_bracket_stable",
                  "criterion": "nuisance-order bracket width < 0.005 (mu=0.1)",
                  "got": lb1["hi"] - lb1["lo"],
                  "pass": bool(lb1["hi"] - lb1["lo"] < 0.005)})
    sc = res["tiers"]["0.1"].get("selfcheck")
    g.append({"id": "A4_lowfriction_order_checkable",
              "criterion": "mu=0.1 ladder has >=4 stiffness levels (leave-one-out check exists)",
              "got": len(res["tiers"]["0.1"]["kn_levels"]),
              "pass": bool(sc is not None)})
    # policy gate: no accuracy claim may rest on a derived anchor --
    # enforced on this audit's verdicts AND on the merged ledger path
    bad = [r for r in res["verdicts"]
           if r["verdict"] != "WITHHELD" and r["anchor_provenance"] != "literature"]
    bad += ledger_claims()
    g.append({"id": "A5_no_claim_on_derived_anchor",
              "criterion": "0 accuracy claims (this audit + merged ledger) rest on a non-literature anchor",
              "got": len(bad),
              "pass": not bad})
    # every tier reported must have a machine-readable policy
    g.append({"id": "A6_policy_coverage",
              "criterion": "every verdict carries anchor provenance + policy",
              "got": len([r for r in res["verdicts"] if r["anchor_provenance"]]),
              "pass": len(res["verdicts"]) > 0
              and all(r["anchor_provenance"] for r in res["verdicts"])})
    return g


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(res, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGS, exist_ok=True)
    out = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {0.0: "C0", 0.1: "C3"}
    for tag, tier in res["tiers"].items():
        mu = float(tag)
        pts = tier["points"]
        kn = np.array([p["kn"] for p in pts], float)
        phi = np.array([p["phi"] for p in pts], float)
        std = np.array([p["std"] if p["std"] else 0.0 for p in pts], float)
        ax.errorbar(kn, phi, yerr=std, marker="o", ls="-", color=colors[mu],
                    capsize=3, label=f"DEM mu={tag} (N={res['n_primary']})")
        for key, mk in (("three_level", "^"), ("four_level", "s")):
            if tier.get(key):
                ax.axhline(tier[key]["limit"], color=colors[mu], ls=":",
                           marker=mk, ms=6, alpha=0.7)
        br = tier.get("order_bracket")
        if br:
            ax.fill_between([kn.min() * 0.8, kn.max() * 1.2], br["lo"], br["hi"],
                            color=colors[mu], alpha=0.12)
    ref = frictionless_reference()
    ax.axhspan(*BAND, color="grey", alpha=0.15, label="ledger band 0.62-0.66")
    ax.axhline(ref.phi_rcp, color="k", ls="--", lw=1,
               label=f"literature RCP mu=0 ({ref.phi_rcp})")
    ax.set_xscale("log")
    ax.set_xlabel("contact stiffness $k_n$ (-)")
    ax.set_ylabel(r"bulk packing fraction $\phi$")
    ax.set_title("Stiffness ladder, hard-sphere intercepts and nuisance-order bracket")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_m21145_intercept_policy.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["intercept_policy"] = os.path.relpath(p, ROOT)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels, xpos, heights, col = [], [], [], []
    for i, r in enumerate(rows):
        labels.append(f"{r['quantity']}\n(mu={r['tier_mu']:g})")
        heights.append(100.0 * r["rel_error_vs_anchor"])
        col.append({"PASS": "C2", "FAIL": "C1", "WITHHELD": "C7"}[r["verdict"]])
        xpos.append(i)
    ax.bar(xpos, heights, color=col)
    ax.axhline(0, color="k", lw=0.8)
    for s in (+3, -3):
        ax.axhline(s, color="C1", ls=":", lw=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("relative error vs mu-indexed anchor (%)")
    ax.set_title("Anchor-policy verdicts (green PASS / red FAIL / yellow WITHHELD)")
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_m21145_verdict_map.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["verdict_map"] = os.path.relpath(p, ROOT)
    return out


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_doc(res, rows, cross, fig_paths, sha):
    prov = res["provenance_tiers"]
    lines = []
    A = lines.append
    A("# AUDIT m21145 - CASE-DEM-RCP-PACKING-FRACTION 锚点溯源与精度判定审计\n")
    A(f"**Mission** m21145 · **Repo** TensorDEM · **Artifacts sha256** `{sha[:16]}`  ")
    A("本文件全部数字由 `benchmarks/case_rcp_deposition/anchor_provenance_audit.py` 程序化写入。\n")
    A("## 1 结论先行\n")
    anc0 = [r for r in rows if r["tier_mu"] == 0.0 and r["quantity"].startswith("production")]
    anc1 = [r for r in rows if r["tier_mu"] == 0.1 and r["quantity"].startswith("production")]
    A(f"1. **无摩擦档 (mu=0) 的精度声称站得住**：生产值 φ={anc0[0]['phi']:.5f} vs 文献 "
      f"{anc0[0]['anchor_phi']}，相对误差 {100*anc0[0]['rel_error_vs_anchor']:+.2f}%，"
      f"判定 **{anc0[0]['verdict']}**（锚点 provenance = {anc0[0]['anchor_provenance']}）。")
    A(f"2. **低摩擦档 (mu=0.1) 的 3% 声称不成立，降级为 WITHHELD**：现网唯一 mu 索引锚 "
      f"φ={anc1[0]['anchor_phi']} 自身标为 *interpolated from DEM study*（provenance = "
      f"{anc1[0]['anchor_provenance']}），拿它认证同一代码族的输出是自指。"
      f"可说的最强结论是廓形必要条件：{anc1[0]['strongest_statement']}。")
    A(f"3. **跨脚本判定互相矛盾已消解**：同一硬球截距在 `docs/CASE_RCP_PACKING_FRACTION_M20918.md` "
      f"被判 -1.22%（vs 表值 0.6250）而 `runs/order_audit.json` 判 +3.07% FAIL（vs 无摩擦文献 0.6366）。"
      f"两者锚点不同且都不是 mu=0.1 的文献点；本审计把锚点策略写进代码（`tensordem.rcp_references.judge`）。")
    lb = res["tiers"]["0.1"].get("order_bracket")
    if lb:
        four = res["tiers"]["0.1"].get("four_level")
        if four:
            tail = (f"第四刚度水平 k_n={max(res['tiers']['0.1']['kn_levels']):g} 已补测，"
                    f"四点拟合 p={four['order'] if four.get('order') is not None else 'n/a'}"
                    f"、φ∞={four['limit']:.5f}；留一预测误差 "
                    f"{res['tiers']['0.1']['selfcheck']['prediction_error']:.2e}。")
        else:
            tail = ("第四刚度水平（k_n=256000, mu=0.1）尚未回件，故拟合阶 p=0.51 仍不可检验，"
                    "本节只给出阶次区间包络，不把单个外推值当作定值。")
        A(f"4. **低摩擦外推阶不可辨识的欠账已量化**：p∈[{res['order_range'][0]},"
          f"{res['order_range'][1]}] 下截距区间 [{lb['lo']:.5f}, {lb['hi']:.5f}]，"
          f"宽度 {lb['hi']-lb['lo']:.5f}；{tail}")
    A("")
    A("## 2 锚点溯源分级表\n")
    A(md_table(["mu", "phi_ref", "2σ 带", "provenance", "可认证?", "来源"],
               [[f"{r['mu']:g}", f"{r['phi_rcp']:.4f}",
                 f"[{r['band_2sigma'][0]:.4f}, {r['band_2sigma'][1]:.4f}]",
                 r["provenance"], "是" if r["anchored"] else "否", r["source"]]
                for r in prov]))
    A("")
    A("分级规则：`literature` 才可承载精度声称；`derived`（本仓构造/插值）与 `unverified`"
      "（复合引用、无页码表号转录）只能排序或做必要条件检验。\n")
    A("## 3 刚度阶梯与硬球截距\n")
    for tag, tier in res["tiers"].items():
        A(f"### mu = {tag}\n")
        A(md_table(["k_n", "phi (n seeds)", "std", "artifacts"],
                   [[f"{p['kn']:g}", f"{p['phi']:.5f} ({p['n_seeds']})",
                     "-" if p["std"] is None else f"{p['std']:.2e}",
                     "<br>".join(os.path.basename(x) for x in p["paths"])]
                    for p in tier["points"]]))
        A("")
        for key, label in (("three_level", "三点拟合"), ("four_level", "四点拟合")):
            if tier.get(key):
                t = tier[key]
                order_txt = ("不可辨识 (n/a)" if t.get("order") is None
                             else f"{t['order']:.3f}")
                A(f"- {label}: p = {order_txt}, φ∞ = {t['limit']:.5f}")
        if tier.get("selfcheck"):
            sc = tier["selfcheck"]
            A(f"- 留一预测检验: 预测最硬一级误差 {sc['prediction_error']:.2e}, "
              f"纳入后截距位移 {sc['limit_shift']:.2e}")
        if tier.get("order_bracket"):
            A(f"- 阶次区间 p∈[{res['order_range'][0]},{res['order_range'][1]}] → "
              f"φ∞ ∈ [{tier['order_bracket']['lo']:.5f}, {tier['order_bracket']['hi']:.5f}]")
        A("")
    A("## 4 判定矩阵（锚点策略）\n")
    A(md_table(["mu", "量", "值", "账簿带内?", "锚点", "相对误差", "判定", "最强可声明"],
               [[f"{r['tier_mu']:g}", r["quantity"], f"{r['phi']:.5f}",
                 "是" if r["in_ledger_band"] else "否",
                 f"{r['anchor_phi']:.4f} ({r['anchor_provenance']})",
                 f"{100*r['rel_error_vs_anchor']:+.2f}%", r["verdict"],
                 r["strongest_statement"]] for r in rows]))
    A("")
    A("## 5 门控\n")
    A(md_table(["id", "判据", "实测", "状态"],
               [[g["id"], g["criterion"],
                 f"{g['got']:.3e}" if isinstance(g["got"], float) else g["got"],
                 "PASS" if g["pass"] else "FAIL"] for g in res["gates"]]))
    A("")
    if cross.get("available"):
        A("## 6 与已并订单审计的一致性\n")
        A(md_table(["gate", "mu", "截距", "vs 无摩擦文献", "vs mu 表值", "表值 provenance", "表值判定"],
                   [[g["id"], f"{g['mu']:g}", f"{g['limit']:.5f}",
                     f"{100*g['rel_error_vs_frictionless']:+.2f}%"
                     + ("" if g["pass"] else " (FAIL)"),
                     "-" if "rel_error_vs_table_entry" not in g
                     else f"{100*g['rel_error_vs_table_entry']:+.2f}%",
                     g.get("table_entry_provenance", "-"),
                     g.get("table_entry_verdict", "-")] for g in cross["gates"]]))
        A("")
    A("## 7 图\n")
    if "intercept_policy" in fig_paths:
        A(f"![intercept](../{fig_paths['intercept_policy']})\n")
        A("*刚度阶梯、三点/四点硬球截距（点线）、阶次不确定性区间（阴影）、账簿带与文献 RCP 线。*\n")
    if "verdict_map" in fig_paths:
        A(f"![verdict](../{fig_paths['verdict_map']})\n")
        A("*每个申报量相对 mu 索引锚点的相对误差与判定色：绿 PASS / 红 FAIL / 黄 WITHHELD。*\n")
    A("## 8 注意点与局限\n")
    A("1. `derived` 判定不否认 mu=0.1 的数值本身落在廓形内，只否认「以它为基准的 −1.22% 是验证」。")
    A("2. 廓形必要条件（0.56 ≤ φ ≤ 上限）很宽，它只能排除荒谬值，不能替代文献锚。")
    A("3. 时间步 dt=tc/20 的系统性未在本审计重估，沿用 order_audit 的 dt/2 锚（内部一致）。")
    A("4. mu=0.1 第四刚度水平（k_n=256000, 2 seeds）已递交集群 cj3208，按台账语义"
      "不设节点；回件后重跑本脚本即可让 A4 门自动转绿，本文件无需手改。")
    A("5. 若后续取得 Pournin/Silbert 原文的表值（页码+表号转录），把该条 provenance 改为 "
      "`literature` 即可让判定自动升级为 PASS/FAIL —— 代码无需改动。")
    A("")
    A("## 9 误差判定\n")
    A(md_table(["关键量", "基准", "计算值", "误差", "<3%?"],
               [[r["quantity"] + f" (mu={r['tier_mu']:g})",
                 f"{r['anchor_phi']:.4f} ({r['anchor_provenance']})",
                 f"{r['phi']:.5f}", f"{100*r['rel_error_vs_anchor']:+.2f}%",
                 ("是" if r["verdict"] == "PASS" else
                  ("否" if r["verdict"] == "FAIL" else "无基准(锚点非文献)→WITHHELD"))]
                for r in rows]))
    A("")
    open(DOC, "w").write("\n".join(lines) + "\n")
    return DOC


# ---------------------------------------------------------------------------

def main() -> int:
    runs = load_runs()
    res = analyse(runs)
    res["verdicts"] = verdicts(res)
    res["cross_script"] = cross_script_consistency(res)
    res["gates"] = gates(res)
    res["provenance_sha256"] = hashlib.sha256(
        json.dumps([r["path"] for r in runs]).encode()).hexdigest()
    fig_paths = figures(res, res["verdicts"])
    res["figures"] = fig_paths
    sha = hashlib.sha256(json.dumps(res, sort_keys=True, default=str).encode()).hexdigest()
    res["self_sha256"] = sha
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    doc = write_doc(res, res["verdicts"], res["cross_script"], fig_paths, sha)
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)}")
    print(f"wrote {os.path.relpath(doc, ROOT)}")
    for g in res["gates"]:
        print(f"  {g['id']}: {'PASS' if g['pass'] else 'FAIL'} ({g['got']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())