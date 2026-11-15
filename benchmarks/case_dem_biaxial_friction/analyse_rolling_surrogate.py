#!/usr/bin/env python
"""Analyse the rolling-surrogate additivity probe (m21803).

Reads ``runs/m21803/rolling_surrogate_*.json`` (produced by
``scripts/m21803_rolling_surrogate.py``), extracts the critical-state friction
angle with the four-gate detector in *isochoric* mode, and answers one question:

    Is phi_cv(mu=0.55, mu_r=b) the same as phi_cv(mu=0.55+b, mu_r=0)?

If yes, the "rolling resistance" operator merged in round 4 is an additive
sliding-friction limit (``tests/test_rolling_surrogate_additivity.py`` proves the
operator identity at contact level), and a phi_cv(mu_r) curve is a
re-parameterisation of mu rather than evidence about rolling resistance physics.

Outputs
-------
``artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.json``
``artifacts/CASE-DEM-BIAXIAL-FRICTION/rolling_surrogate_audit.png``

Usage
-----
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_rolling_surrogate.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensordem.biaxial_critical_state import critical_state_from_stress_path  # noqa: E402

RUNS_DIR = Path(os.environ.get("M21803_RUNS_DIR", "runs/m21803"))
ARTIFACTS_DIR = Path("artifacts/CASE-DEM-BIAXIAL-FRICTION")
TOL_ADDITIVITY_DEG = 1.5   # pre-declared: pairs agree if |dphi_cv| <= 1.5 deg


def _tail_phi(stress_path: dict, from_strain: float = 0.25) -> tuple[float, float]:
    eps = np.asarray(stress_path.get("axial_strain", []), dtype=float)
    sv = np.asarray(stress_path.get("sigma_v", []), dtype=float)
    sh = np.asarray(stress_path.get("sigma_h", []), dtype=float)
    m = eps >= from_strain
    if m.sum() < 10:
        return float("nan"), float("nan")
    phi = np.degrees(np.arcsin(np.clip((sv[m] - sh[m]) / np.maximum(sv[m] + sh[m], 1e-30),
                                       -0.95, 0.95)))
    return float(phi.mean()), float(phi.std())


def analyse_record(rec: dict) -> dict:
    cfg = rec["config"]
    sp = rec["stress_path"]
    cs = critical_state_from_stress_path(
        sp, strain_measure="log", boundary_control="isochoric", min_strain=0.15)
    phi_tail, phi_tail_std = _tail_phi(sp)
    return {
        "tag": rec["tag"],
        "mu": float(cfg["mu"]),
        "mu_r": float(cfg["mu_r"]),
        "mu_eff": float(cfg["mu"]) + float(cfg["mu_r"]),
        "n_particles": int(rec["specimen"]["n_particles"]),
        "final_strain": float(rec["loading"]["final_axial_strain"]),
        "inertial_number": float(rec["loading"]["inertial_number"]),
        "peak_phi_deg": float(rec["peak"]["friction_angle_deg"]),
        "phi_cv_deg": float(cs.phi_cv_deg),
        "phi_cv_converged": bool(cs.converged),
        "phi_cv_flags": {k: bool(v) for k, v in cs.flags.items()},
        "q_drift": float(cs.q_drift),
        "window_spread_deg": float(cs.window_spread_deg),
        "d_eps_v_d_eps_a": float(cs.d_eps_v_d_eps_a),
        "phi_tail_mean_deg": phi_tail,
        "phi_tail_std_deg": phi_tail_std,
        "phi_rowe_atan_mu_deg": math.degrees(math.atan(float(cfg["mu"]))),
        "phi_rowe_atan_mu_eff_deg": math.degrees(
            math.atan(float(cfg["mu"]) + float(cfg["mu_r"]))),
    }


def additivity_rows(rows: list[dict]) -> list[dict]:
    """Compare each (mu, mu_r>0) run with the pure-friction run at mu + mu_r."""
    index = {(r["mu"], r["mu_r"]): r for r in rows}
    out = []
    for r in rows:
        if r["mu_r"] <= 0.0:
            continue
        base = index.get((round(r["mu"] + r["mu_r"], 10), 0.0))
        if base is None:
            out.append({"split_tag": r["tag"], "merged_tag": None,
                        "measured": False})
            continue
        d_phi = r["phi_cv_deg"] - base["phi_cv_deg"]
        out.append({
            "split_tag": r["tag"],
            "merged_tag": base["tag"],
            "mu_eff": r["mu_eff"],
            "phi_cv_split_deg": r["phi_cv_deg"],
            "phi_cv_merged_deg": base["phi_cv_deg"],
            "delta_phi_cv_deg": d_phi,
            "delta_phi_cv_pct": 100.0 * abs(d_phi) / max(base["phi_cv_deg"], 1e-9),
            "tolerance_deg": TOL_ADDITIVITY_DEG,
            "additive_within_tolerance": bool(abs(d_phi) <= TOL_ADDITIVITY_DEG),
            "measured": True,
        })
    return out


def make_figure(rows: list[dict], additivity: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[analyse] matplotlib unavailable ({exc}); skipping figure")
        return False

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    mu_grid = np.linspace(0.0, 1.1, 200)
    ax[0].plot(mu_grid, np.degrees(np.arctan(mu_grid)), "k--", lw=1,
               label=r"$\varphi=\arctan(\mu_{\rm eff})$")
    roll = [r for r in rows if r["mu_r"] > 0]
    pure = [r for r in rows if r["mu_r"] == 0]
    if pure:
        ax[0].plot([r["mu"] for r in pure], [r["phi_cv_deg"] for r in pure], "s",
                   ms=7, color="tab:blue", label=r"pure friction ($\mu_r=0$)")
    if roll:
        ax[0].plot([r["mu_eff"] for r in roll], [r["phi_cv_deg"] for r in roll], "o",
                   ms=7, mfc="none", color="tab:red",
                   label=r"rolling surrogate, plotted at $\mu+\mu_r$")
    ax[0].set_xlabel(r"$\mu_{\rm eff}$")
    ax[0].set_ylabel(r"$\varphi_{cv}$ [deg]")
    ax[0].set_title("phi_cv: is the rolling surrogate just added friction?")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    if additivity:
        x = np.arange(len(additivity))
        ax[1].bar(x, [a["delta_phi_cv_deg"] for a in additivity], color="tab:purple")
        ax[1].axhline(TOL_ADDITIVITY_DEG, color="r", ls=":", lw=1)
        ax[1].axhline(-TOL_ADDITIVITY_DEG, color="r", ls=":", lw=1,
                      label=f"pre-declared +/-{TOL_ADDITIVITY_DEG} deg")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([a["split_tag"].replace("_", "\n") for a in additivity],
                              fontsize=7)
        ax[1].set_ylabel(r"$\Delta\varphi_{cv}$ split - merged [deg]")
        ax[1].set_title("Additivity residual")
        ax[1].axhline(0.0, color="k", lw=0.8)
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    files = sorted(RUNS_DIR.glob("rolling_surrogate_*.json"))
    if not files:
        print(f"[analyse] no runs in {RUNS_DIR} -- nothing to do")
        return 1

    rows = []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        row = analyse_record(rec)
        rows.append(row)
        print(f"{row['tag']:14s} mu={row['mu']:.2f} mu_r={row['mu_r']:.2f} "
              f"phi_cv={row['phi_cv_deg']:6.2f} conv={row['phi_cv_converged']} "
              f"tail={row['phi_tail_mean_deg']:6.2f}+-{row['phi_tail_std_deg']:.2f} "
              f"atan(mu)={row['phi_rowe_atan_mu_deg']:.2f}")

    additivity = additivity_rows(rows)
    for a in additivity:
        if not a["measured"]:
            print(f"[additivity] {a['split_tag']}: no merged control run")
            continue
        print(f"[additivity] {a['split_tag']} vs {a['merged_tag']}: "
              f"dphi={a['delta_phi_cv_deg']:+.2f} deg "
              f"({a['delta_phi_cv_pct']:.1f}%) within_tol={a['additive_within_tolerance']}")

    measured = [a for a in additivity if a["measured"]]
    verdict = None
    if measured:
        verdict = {
            "statements": (
                "phi_cv(mu, mu_r) equals phi_cv(mu+mu_r, 0) within the pre-declared "
                "{:.1f} deg for all measured pairs".format(TOL_ADDITIVITY_DEG)
                if all(a["additive_within_tolerance"] for a in measured)
                else "at least one pair disagrees beyond the pre-declared tolerance"),
            "all_pairs_additive": bool(all(a["additive_within_tolerance"] for a in measured)),
        }

    payload = {
        "probe": "m21803 rolling-surrogate additivity",
        "sample": "16x32 discs, isochoric, sigma3=100 kPa, eps_dot=0.01 1/s, eps_a<=0.35",
        "tolerance_additivity_deg": TOL_ADDITIVITY_DEG,
        "runs": rows,
        "additivity": additivity,
        "verdict": verdict,
        "note": ("Contact-level operator identity is proven exactly by "
                 "tests/test_rolling_surrogate_additivity.py; this artefact tests "
                 "whether the macroscopic phi_cv follows the same additive law."),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS_DIR / "rolling_surrogate_audit.json"
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[analyse] wrote {out_json}")

    png = ARTIFACTS_DIR / "rolling_surrogate_audit.png"
    if make_figure(rows, measured, png):
        print(f"[analyse] wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())