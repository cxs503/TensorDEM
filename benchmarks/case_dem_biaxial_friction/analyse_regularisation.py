#!/usr/bin/env python
"""m21958 analysis: is the contact-friction regularisation velocity a hidden input to phi_cv?

Consumes ``runs/m21958/reg_<tag>.json`` (see
``scripts/m21958_friction_regularisation_probe.py``) and writes
``artifacts/CASE-DEM-BIAXIAL-FRICTION/regularisation.json`` plus two figures.

The pre-registered decision rules are read from
``prereg/m21958_friction_regularisation_prereg.md`` and hard-coded here as
constants *before* the runs are looked at:

* H1  phi_cv invariant under ``v_reg`` in {2.5e-3, 1e-2, 4e-2, 1e-1}: spread <= 1.0 deg
* H2  phi_cv invariant under axial strain rate 0.01 -> 0.05 1/s:       <= 1.0 deg
* H3  phi_cv ≈ arctan(mu * <|tanh(v_t/v_reg)|>_Fn):                    <= 2.0 deg

Every number in the emitted JSON is computed from the run records; nothing is
hand-typed.

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_regularisation.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
    phi_deg_from_stresses,
)

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "m21958"
OUT_DIR = REPO / "artifacts" / "CASE-DEM-BIAXIAL-FRICTION"
FIG_DIR = REPO / "docs" / "figures"

MU = 0.55
TOL_VREG_DEG = 1.0          # H1, pre-registered
TOL_RATE_DEG = 1.0          # H2, pre-registered
TOL_MECHANISM_DEG = 2.0     # H3, pre-registered
PHI_ROWE_DEG = math.degrees(math.atan(MU))      # 28.81 deg (Rowe 1962, D = 0)
PHI_BOLTON_DEG = 38.0                          # Bolton 1986 dense Hostun, plane strain
ANCHOR_TOLERANCE = 0.08                        # case-pool acceptance
BASE_TAG = "A_base_rate0p01_vreg_default"
RATE_TAG = "E_rate0p05_vreg_default"
MIN_STRAIN = 0.15


def _load() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(RUN_DIR.glob("reg_*.json")):
        raw = json.loads(path.read_text())
        out[raw["tag"]] = raw
    if not out:
        raise SystemExit(f"no run records in {RUN_DIR}; run the probe first")
    return out


def _analyse_arm(raw: dict) -> dict:
    cfg = raw["config"]
    sp = {k: np.asarray(v, dtype=float) for k, v in raw["stress_path"].items()}
    cs = critical_state_from_stress_path(
        sp, sigma3_target=float(cfg["confining_stress"]),
        strain_measure="log", boundary_control="isochoric", min_strain=MIN_STRAIN)
    lo, hi = cs.window
    eps = sp["axial_strain_log"]
    mask = (eps >= lo) & (eps <= hi)

    mob = raw.get("mobilisation") or []
    in_window = [m for m in mob if lo <= m["axial_strain"] and m["axial_strain"] <= hi]
    if not in_window:                                  # fall back to the whole tail
        in_window = [m for m in mob if m["axial_strain"] >= MIN_STRAIN]
    fn_w = float(np.nanmean([m["fn_weighted_mean_abs_tanh"] for m in in_window])) \
        if in_window else float("nan")
    plain = float(np.nanmean([m["mean_abs_tanh"] for m in in_window])) \
        if in_window else float("nan")
    sat = float(np.nanmean([m["saturated_fraction"] for m in in_window])) \
        if in_window else float("nan")
    vt = float(np.nanmean([m["mean_abs_vt"] for m in in_window])) \
        if in_window else float("nan")

    phi_wall = float(np.nanmean(phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"])[mask]))
    phi_int = float("nan")
    if "sigma_v_internal" in sp and "sigma_h_internal" in sp:
        phi_int = float(np.nanmean(phi_deg_from_stresses(
            sp["sigma_v_internal"], sp["sigma_h_internal"])[mask]))

    mu_eff = MU * fn_w
    return {
        "tag": raw["tag"],
        "strain_rate": float(cfg["strain_rate"]),
        "velocity_regularisation": float(cfg["velocity_regularisation"]),
        "velocity_regularisation_is_default": bool(
            cfg.get("velocity_regularisation_is_default", False)),
        "steps": int(raw["loading"]["steps"]),
        "inertial_number": float(raw["loading"]["inertial_number"]),
        "elapsed_s": float(raw.get("elapsed_s", float("nan"))),
        "phi_cv_deg": float(cs.phi_cv_deg),
        "phi_cv_wall_window_deg": phi_wall,
        "phi_cv_internal_window_deg": phi_int,
        "q_cv_kPa": float(cs.q_cv_Pa) / 1.0e3,
        "sigma_h_cv_kPa": float(cs.sigma_h_cv_Pa) / 1.0e3,
        "d_eps_v_d_eps_a": float(cs.d_eps_v_d_eps_a),
        "q_drift": float(cs.q_drift),
        "window": [float(lo), float(hi)],
        "window_spread_deg": float(cs.window_spread_deg),
        "converged": bool(cs.converged),
        "flags": {k: bool(v) for k, v in cs.flags.items()},
        "thresholds": {k: float(v) for k, v in cs.thresholds.items()},
        "peak_phi_deg": float(cs.peak_phi_deg),
        "mobilisation": {
            "fn_weighted_mean_abs_tanh": fn_w,
            "mean_abs_tanh": plain,
            "saturated_fraction": sat,
            "mean_abs_vt": vt,
            "samples": len(in_window),
        },
        "mu_eff": mu_eff,
        "phi_from_contact_law_deg": math.degrees(math.atan(mu_eff)) if mu_eff else float("nan"),
        "phi_anchor_relative_error": abs(float(cs.phi_cv_deg) - PHI_BOLTON_DEG) / PHI_BOLTON_DEG,
        "phi_rowe_relative_error": abs(float(cs.phi_cv_deg) - PHI_ROWE_DEG) / PHI_ROWE_DEG,
    }


def main() -> int:
    runs = _load()
    arms = [_analyse_arm(raw) for raw in runs.values()]
    arms.sort(key=lambda a: (a["velocity_regularisation"], a["strain_rate"]))
    by_tag = {a["tag"]: a for a in arms}

    # --- H1: v_reg sensitivity at fixed strain rate -----------------------
    vreg_arms = [a for a in arms if abs(a["strain_rate"] - 0.01) < 1e-12]
    phi_vreg = [a["phi_cv_deg"] for a in vreg_arms]
    spread_vreg = max(phi_vreg) - min(phi_vreg) if phi_vreg else float("nan")
    h1_holds = bool(spread_vreg <= TOL_VREG_DEG) if phi_vreg else False

    # --- H2: strain-rate sensitivity at default v_reg ---------------------
    rate_base = by_tag.get(BASE_TAG)
    rate_fast = by_tag.get(RATE_TAG)
    delta_rate = (abs(rate_fast["phi_cv_deg"] - rate_base["phi_cv_deg"])
                  if rate_base and rate_fast else float("nan"))
    h2_holds = bool(delta_rate <= TOL_RATE_DEG) if rate_base and rate_fast else False

    # --- H3: does the contact law explain the measured angle? ------------
    mech = [a for a in arms if np.isfinite(a["phi_from_contact_law_deg"])]
    mech_delta = [abs(a["phi_cv_deg"] - a["phi_from_contact_law_deg"]) for a in mech]
    worst_mech = max(mech_delta) if mech_delta else float("nan")
    h3_holds = bool(worst_mech <= TOL_MECHANISM_DEG) if mech_delta else False

    baseline = rate_base or arms[0]
    payload = {
        "mission": 21958,
        "anchor": "CASE-DEM-BIAXIAL-FRICTION",
        "mu": MU,
        "references": {
            "phi_rowe_deg": PHI_ROWE_DEG,
            "phi_bolton_deg": PHI_BOLTON_DEG,
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "rowe_citation": ("Rowe (1962) Proc. R. Soc. Lond. A 269:500-527, "
                              "stress-dilatancy at D=0: phi_cv = arctan(mu)"),
            "bolton_citation": ("Bolton (1986) Geotechnique 36(1):65-78, dense Hostun "
                                "sand plane strain phi_peak = 38 deg (case-pool "
                                "acceptance reference)"),
        },
        "preregistered_tolerances_deg": {
            "H1_vreg_spread": TOL_VREG_DEG,
            "H2_rate_difference": TOL_RATE_DEG,
            "H3_contact_law_mapping": TOL_MECHANISM_DEG,
        },
        "runs": arms,
        "hypotheses": {
            "H1_vreg_invariance": {
                "holds": h1_holds, "spread_deg": spread_vreg,
                "tolerance_deg": TOL_VREG_DEG,
                "arms": [a["tag"] for a in vreg_arms],
                "phi_deg": phi_vreg,
            },
            "H2_rate_invariance": {
                "holds": h2_holds, "difference_deg": delta_rate,
                "tolerance_deg": TOL_RATE_DEG,
                "arms": [BASE_TAG, RATE_TAG],
                "phi_deg": [rate_base["phi_cv_deg"] if rate_base else float("nan"),
                            rate_fast["phi_cv_deg"] if rate_fast else float("nan")],
            },
            "H3_contact_law_explains_phi": {
                "holds": h3_holds, "worst_difference_deg": worst_mech,
                "tolerance_deg": TOL_MECHANISM_DEG,
                "per_arm": {a["tag"]: {"phi_cv_deg": a["phi_cv_deg"],
                                       "phi_from_contact_law_deg":
                                           a["phi_from_contact_law_deg"],
                                       "difference_deg":
                                           a["phi_cv_deg"] - a["phi_from_contact_law_deg"]}
                            for a in mech},
            },
        },
        "anchor_verdict": {
            "phi_cv_deg": baseline["phi_cv_deg"],
            "reference_deg": PHI_BOLTON_DEG,
            "relative_error": baseline["phi_anchor_relative_error"],
            "tolerance": ANCHOR_TOLERANCE,
            "passes_anchor_acceptance": bool(
                baseline["phi_anchor_relative_error"] <= ANCHOR_TOLERANCE),
            "under_3_percent": bool(baseline["phi_anchor_relative_error"] <= 0.03),
            "note": ("anchor acceptance is unchanged by this study; the arm closest to "
                     "the reference is reported separately"),
            "best_arm": min(arms, key=lambda a: a["phi_anchor_relative_error"])["tag"],
            "best_arm_relative_error": min(
                a["phi_anchor_relative_error"] for a in arms),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "regularisation.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"wrote {json_path}")

    make_figures(payload)
    print(json.dumps(payload["hypotheses"], indent=2))
    print(json.dumps(payload["anchor_verdict"], indent=2))
    return 0


def make_figures(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = payload["runs"]
    rowe = payload["references"]["phi_rowe_deg"]
    bolton = payload["references"]["phi_bolton_deg"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    sub = [r for r in runs if abs(r["strain_rate"] - 0.01) < 1e-12]
    ax.semilogx([r["velocity_regularisation"] for r in sub],
                [r["phi_cv_deg"] for r in sub], "o-", color="C0", label=r"$\phi_{cv}$ measured")
    ax.semilogx([r["velocity_regularisation"] for r in sub],
                [r["phi_from_contact_law_deg"] for r in sub], "s--", color="C1",
                label=r"$\arctan(\mu\langle|\tanh(v_t/v_{reg})|\rangle)$")
    ax.axhline(rowe, color="k", ls=":", label=r"Rowe 1962 $\arctan\mu$ = %.2f°" % rowe)
    ax.axhline(bolton, color="C3", ls="-.", label="Bolton 1986 dense = %.0f°" % bolton)
    ax.set_xlabel(r"regularisation velocity $v_{reg}$ [m/s]")
    ax.set_ylabel(r"friction angle [deg]")
    ax.set_title(r"(a) $\phi_{cv}$ vs numerical parameter $v_{reg}$ ($\dot\varepsilon_a$=0.01/s)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in runs:
        ax.plot(r["mobilisation"]["saturated_fraction"] * 100.0, r["phi_cv_deg"], "o")
        ax.annotate(r"$v_{reg}$=%.3g" % r["velocity_regularisation"],
                    (r["mobilisation"]["saturated_fraction"] * 100.0, r["phi_cv_deg"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(rowe, color="k", ls=":")
    ax.axhline(bolton, color="C3", ls="-.")
    ax.set_xlabel("contacts at Coulomb capacity [%]")
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title(r"(b) $\phi_{cv}$ vs saturated contact fraction")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax2 = ax.twinx()
    for r, style in zip(runs, ["-", "--", "-.", ":", "-"]):
        raw_path = RUN_DIR / f"reg_{r['tag']}.json"
        if not raw_path.exists():
            continue
        raw = json.loads(raw_path.read_text())
        sp = raw["stress_path"]
        eps = np.asarray(sp["axial_strain"], dtype=float)
        ratio = np.asarray(sp["stress_ratio"], dtype=float)
        ax.plot(eps, ratio, style, lw=1.0,
                label=r"$v_{reg}$=%.3g, $\dot\varepsilon$=%.2g" % (
                    r["velocity_regularisation"], r["strain_rate"]))
        mob = raw.get("mobilisation") or []
        if mob:
            ax2.plot([m["axial_strain"] for m in mob],
                     [m["fn_weighted_mean_abs_tanh"] for m in mob],
                     style, lw=1.0, alpha=0.55)
    ax.set_xlabel(r"axial strain $\varepsilon_a$")
    ax.set_ylabel(r"stress ratio $q/p$")
    ax2.set_ylabel(r"$\langle|\tanh(v_t/v_{reg})|\rangle_{F_n}$", color="0.4")
    ax.set_title("(c) stress ratio (left) and contact mobilisation (right)")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "CASE-DEM-BIAXIAL-FRICTION_regularisation.png"
    fig.savefig(out, dpi=140)
    art = OUT_DIR / "regularisation.png"
    fig.savefig(art, dpi=140)
    plt.close(fig)
    print(f"wrote {out} and {art}")


if __name__ == "__main__":
    raise SystemExit(main())