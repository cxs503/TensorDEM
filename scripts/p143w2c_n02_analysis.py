"""Merge the P143W2C-N02 shard JSONs, evaluate every acceptance criterion and
write the report artefacts (numbers JSON + figures).

    python scripts/p143w2c_n02_analysis.py --indir artifacts/p143w2c_n02 \
        --outdir artifacts/p143w2c_n02

Everything the report quotes is produced here (no hand-copied numbers).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import fields

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem import bpm_identifiability as bi  # noqa: E402


def _load(indir: str) -> dict:
    records = []
    friction = {}
    # ``runs/run_*.json`` is the canonical, per-run cache written by the campaign
    # (atomic, one record per file); the shard files are only a fallback for
    # archives that predate it.  Reading the cache keeps the analysis free of
    # duplicates when several campaigns reused the same shard numbering.
    run_files = sorted(glob.glob(os.path.join(indir, "runs", "run_*.json")))
    files = sorted(glob.glob(os.path.join(indir, "shard_*.json")))
    if run_files:
        for path in run_files:
            with open(path) as fh:
                records.append(json.load(fh))
        files = []
    for path in files:
        with open(path) as fh:
            payload = json.load(fh)
        records.extend(payload.get("records", []))
        if "friction" in payload:
            friction.update(payload["friction"])
    for name, key in (("friction_cell.json", "shear_cell"),
                      ("friction_residual.json", "residual_specimen")):
        path = os.path.join(indir, name)
        if os.path.exists(path):
            with open(path) as fh:
                payload = json.load(fh)
            friction[key] = payload["rows"]
            friction.setdefault("mu_s", payload.get("mu_s"))
            friction.setdefault("mu_d", payload.get("mu_d"))
    return {"records": records, "friction": friction, "n_files": len(files)}


def _as_result(rec: dict) -> bi.SpecimenResult:
    """Rebuild a ``SpecimenResult`` from its JSON record (None -> nan)."""
    data = dict(rec["result"])
    for key, value in list(data.items()):
        if value is None:
            data[key] = float("nan")
    keep = {f.name for f in fields(bi.SpecimenResult)}
    data = {k: v for k, v in data.items() if k in keep}
    for key in ("strain", "stress", "kinetic"):
        data[key] = np.asarray(data.get(key, []), dtype=np.float64)
    return bi.SpecimenResult(**data)


def _families(records, kind):
    out: dict[float, list[float]] = {}
    for rec in records:
        if rec["kind"] != kind:
            continue
        t = round(float(rec["spec"]["thickness"]), 6)
        out.setdefault(t, []).append(float(rec["result"]["sigma_peak"]))
    return {t: np.asarray(v, dtype=np.float64) for t, v in sorted(out.items())}


def _design_curve(m_true=6.0, n_bootstrap=400):
    """Synthetic CI width of m versus specimen count and number of families."""
    rows = []
    designs = [(0.1, 0.2, 0.4), (0.05, 0.1, 0.2, 0.4), (0.05, 0.08, 0.1, 0.2, 0.4)]
    for thicknesses in designs:
        for n in (8, 16, 32, 64, 96):
            ctrl = bi.positive_control(m_true=m_true, n_specimens=n,
                                       thicknesses=thicknesses,
                                       n_bootstrap=n_bootstrap, seed=11)
            rows.append({
                "thicknesses": list(thicknesses),
                "n_per_family": n,
                "m_recovered": ctrl["m_recovered"],
                "m_ci95": ctrl["m_ci95"],
                "m_ci_width": ctrl["m_ci95"][1] - ctrl["m_ci95"][0],
                "relative_error": ctrl["m_relative_error"],
                "identifiable": ctrl["identifiable"],
            })
    return rows


def analyse(payload: dict) -> dict:
    records = payload["records"]
    results = [_as_result(rec) for rec in records]
    out = {"n_runs": len(records),
           "kinds": {k: sum(1 for r in records if r["kind"] == k)
                     for k in sorted({r["kind"] for r in records})}}

    # ---------------------------------------------------------- (a) beta gate
    beta_out = {}
    for mode in ("tension", "compression"):
        # only the runs of the beta grid itself (the size-effect design reuses
        # beta = 1600 on other thicknesses and must not leak into the drift)
        grid = [_as_result(rec) for rec in records
                if rec["kind"] == f"beta_{mode}"]
        if not grid:
            continue
        gate = bi.beta_drift_gate(grid)
        gate["n_results"] = len(grid)
        beta_out[mode] = gate
    # Fixed-physical-rate arm: the honest inertia probe (the staged arm above is
    # exactly beta-invariant by construction, see the module docstring).
    fixed = [_as_result(rec) for rec in records if rec["kind"] == "beta_fixedrate"]
    if fixed:
        gate = bi.beta_drift_gate(fixed)
        gate["n_results"] = len(fixed)
        gate["note"] = ("physical strain rate held fixed (0.01/0.02 s ramp "
                        "durations); only inertia changes with beta")
        beta_out["fixedrate"] = gate
    out["beta_gate"] = beta_out

    # ------------------------------------------------------- (b) size effect
    size_min = _families(records, "size_min")
    size_ext = _families(records, "size_ext")
    size_out = {}
    for name, fam in (("min", size_min), ("ext", size_ext)):
        if len(fam) < 2:
            continue
        per_family = {}
        for t, vals in fam.items():
            m_fam, s0 = (bi.weibull_mle(vals) if len(np.unique(vals)) >= 3
                         else (float("inf"), float(np.mean(vals))))
            per_family[f"{t:g}"] = {
                "thickness_m": float(t),
                "n_specimens": int(vals.size),
                "mean_strength_Pa": float(vals.mean()),
                "std_strength_Pa": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "cov": (float(vals.std(ddof=1) / vals.mean()) if vals.size > 1 else 0.0),
                "m_per_family": float(m_fam),
                "sigma_0_per_family_Pa": float(s0),
                "strengths_Pa": [float(v) for v in vals],
            }
        fit = bi.bootstrap_size_effect_m(fam, n_bootstrap=1000, seed=2, exponent=2.0)
        fit_vol = bi.bootstrap_size_effect_m(fam, n_bootstrap=1000, seed=2, exponent=1.0)
        verdict = bi.identifiability_verdict(fit)
        size_out[name] = {"per_family": per_family, "fit_area_law": fit,
                          "fit_volume_law": fit_vol, "identifiability": verdict}
    out["size_effect"] = size_out

    # ------------------------------------------------ (c) UCS vs literature
    compression = [r for r in results if r.mode == "compression"]
    ucs = float(np.median([r.sigma_peak for r in compression])) if compression else float("nan")
    out["ucs"] = bi.ucs_literature_cross_check(ucs) if math.isfinite(ucs) else {}
    tension = [r for r in results if r.mode == "tension"]
    out["tensile"] = {
        "sigma_t_median_Pa": float(np.median([r.sigma_peak for r in tension]))
        if tension else float("nan"),
        "E_median_Pa": float(np.median([r.youngs_modulus for r in tension
                                        if math.isfinite(r.youngs_modulus)]))
        if tension else float("nan"),
        "ucs_over_sigma_t": (float(ucs / np.median([r.sigma_peak for r in tension]))
                             if tension and math.isfinite(ucs) else float("nan")),
    }

    # ------------------------------------------------------- (d)/(e) controls
    out["positive_control"] = bi.positive_control(n_specimens=96,
                                                  thicknesses=(0.05, 0.1, 0.2, 0.4),
                                                  n_bootstrap=1000)
    out["negative_control_synthetic"] = bi.negative_control_strengths(n_bootstrap=1000)
    neg_dem = _families(records, "neg_homogeneous")
    if len(neg_dem) >= 2:
        fit = bi.bootstrap_size_effect_m(neg_dem, n_bootstrap=1000, seed=5, exponent=2.0)
        verdict = bi.identifiability_verdict(fit)
        out["negative_control_dem"] = {
            "strengths_Pa": {f"{t:g}": [float(v) for v in vals]
                             for t, vals in neg_dem.items()},
            "m_point": fit["m_point"],
            "m_ci95": fit["m_ci95"],
            "identifiable": verdict["identifiable"],
            "pass": bool(not verdict["identifiable"]),
        }
    out["design_curve"] = _design_curve()

    # ------------------------------------------------------------- friction
    if payload.get("friction"):
        out["friction"] = analyse_friction(payload["friction"])
    return out


def analyse_friction(friction: dict) -> dict:
    """Criterion (d): contact-level residual vs the analytic cap, plus the
    specimen-level residual ordering.

    The contact cell is the analytically anchored measurement (the measured
    platen normal reaction is the ``N`` of ``mu N``), so it carries the verdict
    ``pass_cell``.  The specimen-level ordering is reported as evidence but not
    used as a pass/fail gate, because the residual there is carried by the
    surviving bond skeleton as well as by friction.
    """
    cell = friction.get("shear_cell", {})
    residual = friction.get("residual_specimen", {})
    mu_s = friction.get("mu_s") or 0.6
    mu_d = friction.get("mu_d") or 0.4
    cell_rows = {}
    worst_over = {}
    for law, per_speed in cell.items():
        rows = []
        for key, row in per_speed.items():
            rows.append({
                "slip_speed": float(key),
                "mu_measured": row["mu_measured"],
                "mu_reference": row["mu_reference"],
                "bias_pct": row["bias_pct"],
                "over_prediction_vs_dynamic_pct": row.get(
                    "residual_over_prediction_vs_dynamic_pct"),
                "normal_load_measured_N": row.get("normal_load_measured_N"),
            })
        rows.sort(key=lambda r: r["slip_speed"])
        cell_rows[law] = rows
        worst_over[law] = max(r["over_prediction_vs_dynamic_pct"] for r in rows)
    res_rows = {}
    for law, row in residual.items():
        cap = row["coulomb_cap_dynamic_Pa"]
        res_rows[law] = {
            "tau_peak_Pa": row["tau_peak_Pa"],
            "tau_residual_Pa": row["tau_residual_Pa"],
            "normal_load_N": row["normal_load_N"],
            "coulomb_cap_dynamic_Pa": cap,
            "coulomb_cap_static_Pa": row["coulomb_cap_static_Pa"],
            "mu_slip_measured": row["mu_slip_measured"],
            "bond_damage_fraction_end": row["bond_damage_fraction_end"],
            "residual_over_dynamic_cap": (float(row["tau_residual_Pa"] / cap)
                                          if cap else float("nan")),
        }
    comparison = {}
    if "dual" in res_rows:
        d = res_rows["dual"]["tau_residual_Pa"]
        for law, row in res_rows.items():
            if law != "dual":
                comparison[f"{law}_vs_dual_pct"] = float(
                    (row["tau_residual_Pa"] - d) / d * 100.0)
    return {
        "mu_s": mu_s,
        "mu_d": mu_d,
        "shear_cell": cell_rows,
        "residual_specimen": res_rows,
        "worst_over_prediction_vs_dynamic_pct": worst_over,
        "residual_comparison_vs_dual": comparison,
        "pass_cell": bool(all(abs(row["bias_pct"]) <= 2.0
                              for rows in cell_rows.values() for row in rows)),
        "pass_conservative": bool(worst_over.get("dual", 1.0) <= 1.0),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figures(payload: dict, analysis: dict, outdir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    made = []
    records = payload["records"]

    # 1. sigma-eps traces: one per family (size effect), tension
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    seen = set()
    for rec in records:
        if rec["kind"] != "size_min":
            continue
        key = round(float(rec["spec"]["thickness"]), 6)
        if key in seen:
            continue
        seen.add(key)
        r = _as_result(rec)
        ax.plot(r.strain, np.abs(r.stress) / 1e6, lw=1.2,
                label=f"t = {key:g} m (seed {r.seed})")
    ax.set_xlabel("axial strain [-]")
    ax.set_ylabel(r"nominal tensile stress $\sigma_t$ [MPa]")
    ax.set_title("Bonded specimen, uniaxial tension: one trace per family")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "fig1_sigma_epsilon_families.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    # 2. beta gate: strength and E per beta (median +- spread)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for mode, colour in (("tension", "C0"), ("compression", "C3")):
        rows = analysis["beta_gate"].get(mode, {}).get("quantities", {}).get(
            "sigma_peak", {}).get("per_beta", {})
        if not rows:
            continue
        xs = [float(k) for k in rows]
        ys = [rows[k]["median"] / 1e6 for k in rows]
        es = [rows[k]["std"] / 1e6 for k in rows]
        axes[0].errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=colour,
                         label=mode)
        axes[1].plot(xs, [rows[k]["drift_pct_vs_ref"] for k in rows], marker="s",
                     color=colour, label=mode)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"mass scaling $\beta$ [-]")
    axes[0].set_ylabel(r"peak nominal stress [MPa]")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[1].axhline(bi.BETA_DRIFT_MAX_PCT, ls="--", color="k", lw=1,
                    label="5 % criterion")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"mass scaling $\beta$ [-]")
    axes[1].set_ylabel("drift vs max-$\\beta$ median [%]")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.suptitle("Criterion (a): mass-scaling drift of the specimen strength")
    path = os.path.join(outdir, "fig2_beta_gate.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    # 3. size effect: mean strength vs thickness on log-log with the fit
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, marker in (("min", "o"), ("ext", "s")):
        block = analysis["size_effect"].get(name)
        if not block:
            continue
        fam = block["per_family"]
        ts = np.array([fam[k]["thickness_m"] for k in fam])
        ms = np.array([fam[k]["mean_strength_Pa"] for k in fam])
        ss = np.array([fam[k]["std_strength_Pa"] for k in fam])
        ax.errorbar(ts, ms / 1e6, yerr=ss / 1e6, marker=marker, capsize=3, lw=1.2,
                    label=f"{name} design (fit m = {block['fit_area_law']['m_point']:.2f},"
                          f" CI {block['fit_area_law']['m_ci95'][0]:.2f}-"
                          f"{block['fit_area_law']['m_ci95'][1]:.2f})")
        slope = block["fit_area_law"].get("m_point")
        if math.isfinite(slope) and slope > 0:
            tt = np.linspace(ts.min(), ts.max(), 50)
            s0 = block["fit_area_law"]["sigma_ref_Pa"]
            ax.plot(tt, s0 * tt ** (-2.0 / slope) / 1e6, ls=":", lw=1,
                    color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("specimen thickness t [m]")
    ax.set_ylabel(r"mean tensile strength $\bar\sigma_t$ [MPa]")
    ax.set_title(r"Criterion (b): size effect $\sigma_t = \sigma_0 t^{-2/m}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = os.path.join(outdir, "fig3_size_effect.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    # 4. design curve: CI width of m vs n
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    rows = analysis["design_curve"]
    for thicknesses in sorted({tuple(r["thicknesses"]) for r in rows}):
        sub = [r for r in rows if tuple(r["thicknesses"]) == thicknesses]
        ax.plot([r["n_per_family"] for r in sub], [r["m_ci_width"] for r in sub],
                marker="o", lw=1.2,
                label=f"{len(thicknesses)} families: " + ", ".join(f"{t:g}" for t in thicknesses))
    ax.axhline(bi.M_CI_WIDTH_MAX, ls="--", color="k", lw=1, label="width <= 1.0 criterion")
    ax.set_xscale("log")
    ax.set_xlabel("specimens per family n [-]")
    ax.set_ylabel("95 % bootstrap CI width of m [-]")
    ax.set_title("Identifiability design curve (synthetic, known m = 6)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = os.path.join(outdir, "fig4_design_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    # 5. friction: mobilised coefficient vs slip speed for the three laws
    fr = analysis.get("friction", {})
    if fr:
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        for law, rows in fr["shear_cell"].items():
            speeds = [r["slip_speed"] for r in rows]
            ax.plot(speeds, [r["mu_measured"] for r in rows], marker="o", lw=1.2,
                    label=f"{law}: measured")
            ax.plot(speeds, [r["mu_reference"] for r in rows], ls=":", lw=1.0,
                    color=ax.lines[-1].get_color())
        ax.axhline(fr["mu_d"], ls="--", color="k", lw=1,
                   label=r"dynamic cap $\mu_d$")
        ax.set_xscale("log")
        ax.set_xlabel("slip speed |v| [m/s]")
        ax.set_ylabel(r"mobilised coefficient $\mu$ [-]")
        ax.set_title("Criterion (d): residual friction of the three laws")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        path = os.path.join(outdir, "fig5_friction.png")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(path)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="artifacts/p143w2c_n02")
    ap.add_argument("--outdir", default="artifacts/p143w2c_n02")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    payload = _load(args.indir)
    analysis = analyse(payload)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "numbers.json"), "w") as fh:
        json.dump({"analysis": analysis, "n_files": payload["n_files"],
                   "n_records": len(payload["records"])}, fh, indent=1,
                  default=lambda o: None)
    print(json.dumps({k: analysis[k] for k in
                      ("n_runs", "kinds") if k in analysis}, indent=1))
    if args.figures:
        made = figures(payload, analysis, args.outdir)
        print("figures:", made)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())