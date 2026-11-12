"""Round-5 analysis: is the mu(I)/phi(I) reading of the servo shear cell stationary?

Reads the strain-ladder records written by :mod:`run_stationarity` /
:mod:`run_stationarity_sweep` and answers three questions the round-1..3
reports could not:

1. **Window convergence.**  Does the reported ``mu`` level off as the total
   accumulated strain grows (0.2 -> 1 -> 4 -> 8/16)?  For every ``I`` the
   relative change between consecutive rungs is tabulated; a rung pair within
   ``STATIONARY_TOL`` is called stationary.
2. **Converged error.**  Using the longest (most stationary) rung per ``I``,
   what is the error against the da Cruz (2005) ``mu(I)`` law and the
   phi(I) reference, i.e. does the ledger criterion (<10 %) actually hold?
3. **Estimator bias.**  ``mu_tail`` is the mean of per-sample ``tau/P``;
   ``mu_ratio_of_means`` is ``<tau>/<P>``.  Their difference bounds the
   estimator choice, which rounds 1-3 never reported.

Outputs (all inside the repo):
  * ``runs/stationarity/analysis_r5.json``      -- machine-readable summary
  * ``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R5_tables.md`` -- programmatic tables
  * ``docs/figures/plane_shear_r5_*.png``       -- figures
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs" / "stationarity"
DOCS = REPO / "docs"
FIGS = DOCS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_C, PHI_J = 0.58, 0.64
TOL = 0.10          # ledger acceptance tolerance
STATIONARY_TOL = 3.0  # % : consecutive-rung change regarded as levelled off


def ref_mu(I):
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi(I):
    return PHI_C + (PHI_J - PHI_C) / (I_0 / I + 1.0)


def load_runs(arm: str = "thick") -> dict[float, list[dict]]:
    by_I: dict[float, list[dict]] = {}
    for path in sorted(RUNS.glob(f"{arm}__I*__s*.json")):
        rec = json.loads(path.read_text())
        by_I.setdefault(float(rec["I_target"]), []).append(rec)
    for v in by_I.values():
        v.sort(key=lambda r: r["strain_target"])
    return dict(sorted(by_I.items()))


def rung_table(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        rows.append({
            "strain_target": r["strain_target"],
            "strain_sampled": (r["strain_sampled_lo"], r["strain_sampled_hi"]),
            "I_measured": r["I_measured"],
            "mu_tail": r["mu_tail"],
            "mu_ratio": r["mu_ratio_of_means"],
            "bias_pct": r["mu_ratio_bias_pct"],
            "drift_pct": r["mu_block_drift_pct"],
            "phi": r["phi_bed_steady"],
            "mu_err_pct": r["mu_err_pct"],
            "phi_err_pct": r["phi_err_pct"],
            "p_err_pct": 100.0 * r["pressure_control_error"],
            "travel_d": r["plate_travel_in_d"],
            "steps": r["shear_steps"],
            "wall_s": r["wall_clock_s"],
        })
    return rows


def convergence(rows: list[dict]) -> dict:
    """Consecutive-rung change of mu, and the first rung that is stationary."""
    deltas = []
    for a, b in zip(rows, rows[1:]):
        deltas.append({
            "from_strain": a["strain_target"],
            "to_strain": b["strain_target"],
            "dmu_pct": 100.0 * (b["mu_tail"] / a["mu_tail"] - 1.0),
        })
    stationary_at = next((d["to_strain"] for d in deltas
                          if abs(d["dmu_pct"]) <= STATIONARY_TOL), None)
    best = rows[-1]
    return {"deltas": deltas, "stationary_from_strain": stationary_at,
            "longest_rung": best,
            "mu_spread_pct": 100.0 * (max(r["mu_tail"] for r in rows)
                                      / min(r["mu_tail"] for r in rows) - 1.0)}


def build(arm: str = "thick") -> dict:
    by_I = load_runs(arm)
    if not by_I:
        raise SystemExit(f"no stationarity runs found in {RUNS}")
    out = {"arm": arm, "tolerance": TOL, "stationary_tol": STATIONARY_TOL,
           "points": {}}
    for I, runs in by_I.items():
        rows = rung_table(runs)
        conv = convergence(rows)
        out["points"][f"{I:g}"] = {"rows": rows, "convergence": conv,
                                   "series": [(r["series"]) for r in runs]}
    return out


# ---------------------------------------------------------------- figures ---
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures(analysis: dict) -> list[dict]:
    plt = _mpl()
    figs = []
    pts = analysis["points"]

    # fig 1: mu vs achieved sampled strain, one line per I target
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    cmap = plt.get_cmap("viridis")
    for k, (key, p) in enumerate(pts.items()):
        xs = [0.5 * (r["strain_sampled"][0] + r["strain_sampled"][1]) for r in p["rows"]]
        ys = [r["mu_tail"] for r in p["rows"]]
        ax.plot(xs, ys, "o-", color=cmap(k / max(1, len(pts) - 1)),
                label=f"I_target={key}")
    ax.axhline(MU_S, ls=":", c="0.4", label=r"$\mu_s=0.36$")
    ax.set_xscale("log")
    ax.set_xlabel("mean accumulated shear strain of the reading window [-]")
    ax.set_ylabel(r"$\mu=\langle\tau/P\rangle$ [-]")
    ax.set_title("R5 strain-stationarity ladder (thick bed, thick-arm device)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    p1 = FIGS / "plane_shear_r5_ladder.png"
    fig.tight_layout(); fig.savefig(p1); plt.close(fig)
    figs.append({"name": p1.name,
                 "caption": "R5 应变阶梯：每个 I 目标把同一剪切点重跑到总应变 "
                            "0.2/1/4/(8 或 16)，纵轴为读数窗内的 mu；读数随应变"
                            "不再移动的档位即定常档。"})

    # fig 2: converged mu(I) vs reference +-10%
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    Igrid = np.logspace(-3.3, -0.3, 200)
    ax.plot(Igrid, [ref_mu(i) for i in Igrid], "k-", label="da Cruz 2005 / GDR MiDi mu(I)")
    ax.plot(Igrid, [ref_mu(i) * (1 + TOL) for i in Igrid], "k--", lw=0.8)
    ax.plot(Igrid, [ref_mu(i) * (1 - TOL) for i in Igrid], "k--", lw=0.8,
            label="±10 % acceptance band")
    got = [(p["convergence"]["longest_rung"]["I_measured"],
            p["convergence"]["longest_rung"]["mu_tail"], key) for key, p in pts.items()]
    ax.plot([g[0] for g in got], [g[1] for g in got], "rs", ms=8,
            label="longest strain rung")
    early = []
    for key, p in pts.items():
        r = p["rows"][0]
        early.append((r["I_measured"], r["mu_tail"]))
    ax.plot([e[0] for e in early], [e[1] for e in early], "b^", ms=6, alpha=0.7,
            label="round-3 protocol (strain target 0.2)")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I (measured) [-]")
    ax.set_ylabel(r"$\mu$ [-]")
    ax.set_title("R5 longest-strain reading vs the reference law")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    p2 = FIGS / "plane_shear_r5_muI.png"
    fig.tight_layout(); fig.savefig(p2); plt.close(fig)
    figs.append({"name": p2.name,
                 "caption": "R5 最长应变档读数（红方块）与 R3 短应变协议的读数"
                            "（蓝三角）对照 da Cruz(2005) mu(I) 律与 ±10% 接受带。"})

    # fig 3: phi(I)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    ax.plot(Igrid, [ref_phi(i) for i in Igrid], "k-", label="reference phi(I)")
    xs = [p["convergence"]["longest_rung"]["I_measured"] for p in pts.values()]
    ys = [p["convergence"]["longest_rung"]["phi"] for p in pts.values()]
    ax.plot(xs, ys, "gs", ms=8, label="phi_bed_steady, longest rung (thick bed)")
    ax.set_xscale("log")
    ax.set_xlabel("inertial number I [-]")
    ax.set_ylabel(r"$\phi$ [-]")
    ax.set_title("R5 bed solid fraction vs reference")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    p3 = FIGS / "plane_shear_r5_phiI.png"
    fig.tight_layout(); fig.savefig(p3); plt.close(fig)
    figs.append({"name": p3.name,
                 "caption": "R5 厚床稳态体积分数 phi(I) 与参考 phi 律对照。"})

    # fig 4: estimator bias + servo pressure control error vs strain rung
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=150)
    for k, (key, p) in enumerate(pts.items()):
        c = cmap(k / max(1, len(pts) - 1))
        axes[0].plot([r["strain_target"] for r in p["rows"]],
                     [r["bias_pct"] for r in p["rows"]], "o-", color=c, label=key)
        axes[1].plot([r["strain_target"] for r in p["rows"]],
                     [r["p_err_pct"] for r in p["rows"]], "o-", color=c, label=key)
    for ax, t, yl in ((axes[0], "estimator bias: <tau/P> vs <tau>/<P>", "%"),
                      (axes[1], "servo pressure control error", "%")):
        ax.set_xscale("log"); ax.set_xlabel("strain target [-]")
        ax.set_ylabel(yl); ax.set_title(t, fontsize=9); ax.grid(alpha=0.3)
        ax.axhline(0.0, c="0.6", lw=0.8)
    axes[0].legend(fontsize=6, ncol=2)
    p4 = FIGS / "plane_shear_r5_bias.png"
    fig.tight_layout(); fig.savefig(p4); plt.close(fig)
    figs.append({"name": p4.name,
                 "caption": "左：估计量选择（逐样本 mu 均值 vs <tau>/<P>）随应变档的偏差；"
                            "右：伺服法向压力控制误差。"})
    return figs


def tables_md(analysis: dict) -> str:
    pts = analysis["points"]
    L = ["# CASE-DEM-PLANE-SHEAR-MUIDI R5 — 程序化表格",
         "",
         "> 本文件由 `benchmarks/plane_shear_mu_i/analyse_stationarity.py` 生成，"
         "数字零手抄（全部来自 `runs/stationarity/*.json`）。",
         ""]
    L += ["## ① 应变阶梯原始读数", "",
          "| I_target | strain target | 采样窗 strain | I measured | mu_tail | mu(<t>/<P>) "
          "| 块漂移 % | phi_bed_steady | mu 误差 % | phi 误差 % | p 控误差 % | 板行程 d | shear steps |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, p in pts.items():
        for r in p["rows"]:
            L.append(f"| {key} | {r['strain_target']:g} | "
                     f"{r['strain_sampled'][0]:.3f}–{r['strain_sampled'][1]:.3f} | "
                     f"{r['I_measured']:.4g} | {r['mu_tail']:.4f} | {r['mu_ratio']:.4f} | "
                     f"{r['drift_pct']:+.2f} | {r['phi']:.4f} | {r['mu_err_pct']:+.1f} | "
                     f"{r['phi_err_pct']:+.1f} | {r['p_err_pct']:+.2f} | "
                     f"{r['travel_d']:.1f} | {r['steps']} |")
    L += ["", "## ② 相邻档位收敛性（判据：|Δmu| ≤ "
          f"{analysis['stationary_tol']:g} % 视为定常）", "",
          "| I_target | 档位对 | Δmu % | 首个定常档 | 全长程 mu 极差 % | 最长档 mu 误差 % |",
          "|---|---|---|---|---|---|"]
    for key, p in pts.items():
        c = p["convergence"]
        pairs = "; ".join(f"{d['from_strain']:g}→{d['to_strain']:g}: "
                          f"{d['dmu_pct']:+.2f}" for d in c["deltas"])
        L.append(f"| {key} | {pairs} | — | "
                 f"{c['stationary_from_strain'] if c['stationary_from_strain'] else '未定常'} | "
                 f"{c['mu_spread_pct']:.1f} | {c['longest_rung']['mu_err_pct']:+.1f} |")
    L += ["", "## ③ 判据核对（最长应变档 vs 参考，容差 ±10 %）", "",
          "| I_target | mu got | mu ref | 误差 % | 判定 | phi got | phi ref | 误差 % |",
          "|---|---|---|---|---|---|---|---|"]
    for key, p in pts.items():
        r = p["convergence"]["longest_rung"]
        verdict = "PASS" if abs(r["mu_err_pct"]) <= 10.0 else "FAIL"
        L.append(f"| {key} | {r['mu_tail']:.4f} | {ref_mu(r['I_measured']):.4f} | "
                 f"{r['mu_err_pct']:+.1f} | {verdict} | {r['phi']:.4f} | "
                 f"{ref_phi(r['I_measured']):.4f} | {r['phi_err_pct']:+.1f} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    analysis = build()
    # strip the bulky series before serialising the analysis summary
    slim = json.loads(json.dumps(analysis, default=float))
    for p in slim["points"].values():
        p.pop("series", None)
    (RUNS / "analysis_r5.json").write_text(json.dumps(slim, indent=1))
    figs = make_figures(analysis)
    md = tables_md(analysis)
    (DOCS / "CASE_DEM_PLANE_SHEAR_MUIDI_R5_tables.md").write_text(md)
    (RUNS / "analysis_r5_figures.json").write_text(json.dumps(figs, indent=1))
    print(md)
    print("figures:", [f["name"] for f in figs])


if __name__ == "__main__":
    main()