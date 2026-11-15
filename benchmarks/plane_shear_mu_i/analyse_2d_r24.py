"""Round-24 assembly: read the 2-D plane-shear runs, judge them, tabulate, plot.

Reads ``runs/plane_shear_2d/point_I<I>_s<seed>.json`` written by
``plane_shear_2d.py``, forms the seed ensemble per inertial number, and judges it
against the **archived** da Cruz 2005 law (``reference_archive.archived_law``) -
no literature constant is retyped in this file.  The pre-registered decision
rule is in ``prereg_r24.json``.

Usage::

    python benchmarks/plane_shear_mu_i/analyse_2d_r24.py            # judge + tables + figures
    python benchmarks/plane_shear_mu_i/analyse_2d_r24.py --probe    # the probe series only
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import reference_archive  # noqa: E402

TOL = 0.10
ORDINATES = ("mu_wall", "mu_bulk", "phi_gap", "phi_cell", "p_wall", "p_bulk",
             "sigma_yy_over_P", "I_bulk")


def load_runs(run_dir: Path, pattern: str = "point_I*_s*.json") -> list[dict]:
    runs = []
    for path in sorted(run_dir.glob(pattern)):
        data = json.loads(path.read_text())
        data["_file"] = path.name
        runs.append(data)
    return runs


def _stats(values: list[float]) -> dict:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "cv": float("nan"),
                "se": float("nan"), "ci95": (float("nan"), float("nan"))}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
    se = sd / math.sqrt(len(vals)) if len(vals) > 1 else float("nan")
    if len(vals) > 1:
        # two-sided 95 % Student-t with n-1 dof (small-sample, as round 23)
        from scipy import stats as st
        tcrit = float(st.t.ppf(0.975, len(vals) - 1))
        ci = (mean - tcrit * se, mean + tcrit * se)
        cv = sd / mean if mean else float("nan")
    else:
        ci = (float("nan"), float("nan"))
        cv = float("nan")
    return {"n": len(vals), "mean": mean, "sd": sd, "cv": cv, "se": se, "ci95": ci}


def assemble(run_dir: Path, law, *, probe: bool = False) -> dict:
    runs = load_runs(run_dir, "probe_I*.json" if probe else "point_I*_s*.json")
    if not runs:
        raise SystemExit(f"no runs in {run_dir}")
    by_I: dict[float, list[dict]] = {}
    for r in runs:
        by_I.setdefault(round(float(r["I_target"]), 6), []).append(r)

    points = []
    for I_target in sorted(by_I):
        rows = by_I[I_target]
        point = {"I_target": I_target, "n_seeds": len(rows), "runs": [r["_file"] for r in rows]}
        for key in ORDINATES:
            point[key] = _stats([r.get(key) for r in rows])
        point["ke_settle"] = _stats([r.get("ke_settle") for r in rows])
        point["drift_z_mu_wall"] = _stats([r.get("drift_z_mu_wall") for r in rows])
        point["I_measured"] = point["I_bulk"]["mean"]
        judged = law.mu(point["I_measured"]) is not None and I_target <= law.i_saturation
        point["judged"] = bool(judged)
        if law.mu(point["I_measured"]) is not None:
            mu_ref = law.mu(point["I_measured"])
            phi_ref = law.phi(point["I_measured"])
            point["mu_ref"] = mu_ref
            point["mu_err_pct"] = 100.0 * (point["mu_wall"]["mean"] / mu_ref - 1.0)
            point["phi_ref"] = phi_ref
            point["phi_err_pct"] = 100.0 * (point["phi_gap"]["mean"] / phi_ref - 1.0)
            point["mu_bulk_err_pct"] = 100.0 * (point["mu_bulk"]["mean"] / mu_ref - 1.0)
            point["within_10pct_mu"] = abs(point["mu_err_pct"]) <= 100.0 * TOL
            point["within_10pct_phi"] = abs(point["phi_err_pct"]) <= 100.0 * TOL
            point["P5_stationary"] = abs(point["drift_z_mu_wall"]["mean"]) < 2.0
        else:
            point["mu_ref"] = law.mu_extrapolated(point["I_measured"])
            point["phi_ref"] = law.phi_extrapolated(point["I_measured"])
            point["mu_err_pct"] = 100.0 * (point["mu_wall"]["mean"] / point["mu_ref"] - 1.0)
            point["phi_err_pct"] = 100.0 * (point["phi_gap"]["mean"] / point["phi_ref"] - 1.0)
            point["within_10pct_mu"] = None
            point["within_10pct_phi"] = None
            point["P5_stationary"] = None
        points.append(point)

    judged = [p for p in points if p["judged"]]
    verdict = {
        "reference": {"mu_min": law.mu_min, "b": law.b, "nu_max": law.nu_max,
                      "a": law.a, "i_saturation": law.i_saturation},
        "tolerance_pct": 100.0 * TOL,
        "n_points": len(points),
        "n_judged": len(judged),
        "P1_calibration": all(abs(p["p_wall"]["mean"] / _P(p) - 1.0) <= TOL
                              for p in judged) if judged else None,
        "P2_mu_in_band": all(p["within_10pct_mu"] for p in judged) if judged else None,
        "P4_phi_in_band": all(p["within_10pct_phi"] for p in judged) if judged else None,
        "P5_stationary": all(p["P5_stationary"] for p in judged) if judged else None,
        "mu_pass_count": sum(1 for p in judged if p["within_10pct_mu"]),
        "phi_pass_count": sum(1 for p in judged if p["within_10pct_phi"]),
        "max_abs_mu_err_pct": max((abs(p["mu_err_pct"]) for p in judged), default=float("nan")),
        "max_abs_phi_err_pct": max((abs(p["phi_err_pct"]) for p in judged), default=float("nan")),
        "probe": probe,
    }
    return {"points": points, "verdict": verdict}


def _P(point: dict) -> float:
    """Imposed confining stress read back from the run case block."""
    return float(point.get("_P", 1.0e4))


def _pick(node: dict, path: tuple[str, ...]) -> float | None:
    """Follow a key path and return the scalar mean, or ``None``."""
    cur = node
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, dict):
        cur = cur.get("mean")
    return float(cur) if isinstance(cur, (int, float)) else None


def compare_with_3d(points_2d: list[dict], path: Path) -> list[dict]:
    """Pair every 2-D point with the 3-D ensemble mean at the closest I."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    pts3d = data.get("points", {})
    if isinstance(pts3d, dict):            # round-23 file is keyed by I_target
        pts3d = list(pts3d.values())
    out = []
    for p in points_2d:
        I2 = p["I_measured"]
        cands = []
        for q in pts3d:
            I3 = _pick(q, ("I_bulk_mean",))
            if I3 is None:
                I3 = _pick(q, ("I_bulk",))
            mu3 = _pick(q, ("mu", "mu_full_zz_final"))
            if mu3 is None:
                mu3 = _pick(q, ("mu_full_zz_final",))
            if I3 is not None and mu3 is not None:
                cands.append((q, I3, mu3))
        if not cands:
            continue
        q, I3, mu3 = min(cands, key=lambda t: abs(t[1] - I2))
        out.append({"I_2d": I2, "I_3d": I3,
                    "mu_2d": p["mu_wall"]["mean"], "mu_3d": mu3,
                    "I_3d_target": q.get("I_target"),
                    "delta_pct": 100.0 * (p["mu_wall"]["mean"] / mu3 - 1.0) if mu3 else float("nan")})
    return out


def tables(result: dict, cmp3d: list[dict], out: Path) -> None:
    law = result["verdict"]["reference"]
    lines = [f"# Round 24 tables - 2-D disk plane shear vs the archived law "
             f"`mu = {law['mu_min']:.2f} + {law['b']:.2f} I`, "
             f"`phi = {law['nu_max']:.2f} - {law['a']:.2f} I` (I <= {law['i_saturation']})", ""]
    lines.append("## Table A - mu(I) and phi(I), seed ensemble per inertial number")
    lines.append("| I_target | seeds | I_measured | mu_wall | sd | CV % | 95% CI | mu_law | err % | phi_gap | phi_law | err % | judged |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in result["points"]:
        mu, phi = p["mu_wall"], p["phi_gap"]
        lines.append(
            f"| {p['I_target']:.4g} | {p['n_seeds']} | {p['I_measured']:.4f} | {mu['mean']:.4f} | "
            f"{mu['sd']:.4f} | {100 * mu['cv']:.1f} | [{mu['ci95'][0]:.3f}, {mu['ci95'][1]:.3f}] | "
            f"{p['mu_ref']:.4f} | {p['mu_err_pct']:+.1f} | {phi['mean']:.3f} | {p['phi_ref']:.3f} | "
            f"{p['phi_err_pct']:+.1f} | {'yes' if p['judged'] else 'NO (extrapolation)'} |")
    lines += ["", "## Table B - estimator calibration (P1): the servo and the virial"]
    lines.append("| I_target | p_wall | p_bulk | sigma_yy/P | p_wall/P | ke_settle | drift_z(mu_wall) |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in result["points"]:
        lines.append(f"| {p['I_target']:.4g} | {p['p_wall']['mean']:.0f} | {p['p_bulk']['mean']:.0f} | "
                     f"{p['sigma_yy_over_P']['mean']:.3f} | "
                     f"{(p['p_wall']['mean'] / 1.0e4):.3f} | {p['ke_settle']['mean']:.2e} | "
                     f"{p['drift_z_mu_wall']['mean']:+.2f} |")
    if cmp3d:
        lines += ["", "## Table C - 2-D device against the 3-D device (P3): dimensionality"]
        lines.append("| I (2-D) | I (3-D, nearest) | mu 2-D | mu 3-D | 2-D/3-D - 1 % |")
        lines.append("|---|---|---|---|---|")
        for c in cmp3d:
            lines.append(f"| {c['I_2d']:.4f} | {c['I_3d']:.4f} | {c['mu_2d']:.4f} | "
                         f"{c['mu_3d']:.4f} | {c['delta_pct']:+.1f} |")
    v = result["verdict"]
    lines += ["", "## Verdict", "",
              f"* judged points: {v['n_judged']} (I <= {law['i_saturation']})",
              f"* P1 calibration (|p_wall/P-1| <= 10 %): {v['P1_calibration']}",
              f"* P2 mu inside 10 %: {v['P2_mu_in_band']} ({v['mu_pass_count']}/{v['n_judged']}), "
              f"max |err| {v['max_abs_mu_err_pct']:.1f} %",
              f"* P4 phi inside 10 %: {v['P4_phi_in_band']} ({v['phi_pass_count']}/{v['n_judged']}), "
              f"max |err| {v['max_abs_phi_err_pct']:.1f} %",
              f"* P5 stationarity (|drift_z| < 2): {v['P5_stationary']}"]
    out.write_text("\n".join(lines) + "\n")


def figures(result: dict, cmp3d: list[dict], outdir: Path, law, tag: str = ""
            ) -> list[tuple[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    pts = result["points"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    I_grid = np.linspace(1e-3, law.i_saturation, 200)
    ax.plot(I_grid, [law.mu(I) for I in I_grid], "k-", label="da Cruz 2005 (archived fit)")
    ax.fill_between(I_grid, [(1 - TOL) * law.mu(I) for I in I_grid],
                    [(1 + TOL) * law.mu(I) for I in I_grid], color="k", alpha=0.10,
                    label="+-10 % band")
    for p in pts:
        x = p["I_measured"]
        m = p["mu_wall"]
        ax.errorbar([x], [m["mean"]], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                    fmt="o", color="tab:red" if p["judged"] else "tab:gray", capsize=3)
        ax.annotate(f"{p['I_target']:.3g}", (x, m["mean"]), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
    ax.set_xlabel("inertial number I"), ax.set_ylabel("effective friction mu")
    ax.set_title("2-D disk plane shear: mu(I) with seed 95 % CI")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    path = outdir / f"fig_r24_muI_2d{tag}.png"
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)
    made.append((path.name, "mu(I) of the 2-D disk cell (seed mean with 95 % Student-t interval) "
                            "against the archived 2-D law and its +-10 % band."))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(I_grid, [law.phi(I) for I in I_grid], "k-", label="archived phi(I)")
    ax.fill_between(I_grid, [(1 - TOL) * law.phi(I) for I in I_grid],
                    [(1 + TOL) * law.phi(I) for I in I_grid], color="k", alpha=0.10)
    for p in pts:
        ax.errorbar([p["I_measured"]], [p["phi_gap"]["mean"]], fmt="s", color="tab:blue", capsize=3)
        ax.errorbar([p["I_measured"]], [p["phi_cell"]["mean"]], fmt="^", color="tab:green",
                    alpha=0.6, capsize=3)
    ax.set_xlabel("inertial number I"), ax.set_ylabel("solid fraction phi")
    ax.set_title("phi(I): area fraction (primary) and cell fraction (secondary)")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    path = outdir / f"fig_r24_phiI_2d{tag}.png"
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)
    made.append((path.name, "phi_gap (primary, surface-to-surface control volume) and phi_cell "
                            "(secondary) against the archived dilatancy law."))

    if cmp3d:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        xs = [c["I_3d"] for c in cmp3d]
        ax.plot(xs, [c["mu_3d"] for c in cmp3d], "s--", color="tab:orange", label="3-D sphere cell (r23)")
        ax.plot([c["I_2d"] for c in cmp3d], [c["mu_2d"] for c in cmp3d], "o-", color="tab:red",
                label="2-D disk cell (r24)")
        ax.plot(I_grid, [law.mu(I) for I in I_grid], "k-", label="archived 2-D law")
        ax.set_xlabel("inertial number I"), ax.set_ylabel("mu")
        ax.set_title("dimensionality: 2-D device vs 3-D device vs the 2-D law")
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
        path = outdir / f"fig_r24_dimensionality{tag}.png"
        fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)
        made.append((path.name, "the same quantity measured on both devices: does building the "
                                "law's own dimension close the offset?"))
    return made


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(HERE / "runs" / "plane_shear_2d"))
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    law = reference_archive.archived_law()
    result = assemble(run_dir, law, probe=args.probe)
    cmp3d = compare_with_3d(result["points"], HERE / "runs" / "seed_ensemble" / "analysis_r23.json")
    tag = "_probe" if args.probe else ""
    (run_dir / f"analysis_r24{tag}.json").write_text(json.dumps(
        {"points": result["points"], "verdict": result["verdict"], "vs_3d": cmp3d}, indent=1))
    tables(result, cmp3d, run_dir / f"tables_r24{tag}.md")
    made = figures(result, cmp3d, run_dir, law, tag=tag)
    print(json.dumps(result["verdict"], indent=1))
    for name, cap in made:
        print("figure", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())