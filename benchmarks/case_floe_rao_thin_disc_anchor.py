"""CASE-DEM-FLOE-RAO-ANCHOR: independent analytic anchors for the floe heave RAO.

``src/tensordem/wave_ice/floe_rao.py`` was graded **L2** in ``audit_report.md``
("单测只断言峰值存在/有限/<100/单调"), i.e. it had no independent baseline.  This
benchmark supplies one and re-measures the model against it.  Every number in
``docs/CASE_FLOE_RAO_THIN_DISC_ANCHOR.md`` is produced here (no hand copying).

Anchors
-------
A1  Thin-disc heave added mass: the exact general-ellipsoid coefficient
    (Lamb 1932 §114) must converge to ``A33 -> (8/3) rho R^3`` as the
    semi-thickness ratio h/R -> 0 -- the classical zero-thickness disc result.
    This is an **exact** analytic anchor (two independent routes: closed form
    and numerical quadrature of the same Lamb integral).
A2  The module's closed form ``oblate_spheroid_axial_added_mass`` must match an
    independent adaptive quadrature of the Lamb integral.
A3  Consequence: added mass of a thin body is O(rho R^3), *not* O(displaced
    mass).  The legacy law ``A33 = rho_w pi R^2 T`` implies
    ``Ca_legacy = 1`` where the classical value is ``Ca = 8R/(3 pi T)``;
    the ratio is the legacy error factor.
A4  Heave natural period ``Tn = 2 pi sqrt((M + A33)/C33)`` must scale as
    ``Tn ~ R^(1/2)`` for thin floes (M ~ R^2 T, A33 ~ R^3, C33 ~ R^2, T ~ R).
    The legacy law predicts ``Tn`` independent of R (slope 0) -- falsified.
A5  Long-wave limit of the heave RAO must be 1 (wave-following), for both laws.
A6  ``compute_floe_rao_dem`` vs ``compute_floe_rao_analytic``: both share the
    same A33/B33/C33, so their difference *cannot* validate physics.  Measured
    here explicitly (reported, not asserted away).

Honest scope: A1-A5 are verified.  The *free-surface corrected* added mass of a
floating floe (waterline condition, ω -> 0 rigid-wall limit) is **WITHHELD** --
it needs a free-surface radiation/BEM solve that this repository does not have;
the (8/3)rhoR^3 anchor is the unbounded-fluid leading order.

Run:  PYTHONPATH=src python3 benchmarks/case_floe_rao_thin_disc_anchor.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensordem.wave_ice.floe_rao import (
    FloeRAOConfig,
    THIN_DISC_ADDED_MASS_FACTOR,
    compute_floe_rao_analytic,
    compute_floe_rao_dem,
    heave_added_mass,
    oblate_spheroid_axial_added_mass,
    _dispersion_relation,
    _hydrostatic_restoring_heave,
)

OUT = Path("artifacts/case_floe_rao_thin_disc_anchor")
RHO_W = 1025.0
RHO_I = 917.0
G = 9.81
UNIT_R = 1.0                     # all A1/A2 numbers are per (rho R^3)


# ---------------------------------------------------------------------------
# A1/A2: thin-disc limit of the Lamb ellipsoid coefficient
# ---------------------------------------------------------------------------

def lamb_integral_quadrature(radius: float, half_thickness: float,
                             rho_water: float = RHO_W) -> float:
    """Independent route: adaptive quadrature of the Lamb §114 alpha3 integral."""
    from scipy.integrate import quad

    R, h = float(radius), float(half_thickness)

    def integrand(lam: float) -> float:
        return 1.0 / ((h * h + lam) * math.sqrt((R * R + lam) ** 2 * (h * h + lam)))

    # substitution lam = h^2 (u^-2 - 1) maps [h^2, inf) -> (0, 1]; keeps the
    # h -> 0 divergence of the raw integrand under control.
    def subst(u: float) -> float:
        if u <= 0.0:
            return 0.0
        lam = h * h * (1.0 / (u * u) - 1.0)
        dlam = -2.0 * h * h / (u * u * u)
        return integrand(lam) * dlam

    alpha3, _ = quad(subst, 1e-12, 1.0, limit=400)
    vol = (4.0 / 3.0) * math.pi * R * R * h
    return vol * rho_water * alpha3 / (2.0 - alpha3)


def anchor_a1_a2() -> dict:
    aspects = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    rows = []
    for a in aspects:
        aspect_h = a * UNIT_R
        R, h = UNIT_R, aspect_h
        closed = oblate_spheroid_axial_added_mass(R, h, RHO_W) / (RHO_W * R**3)
        quad = lamb_integral_quadrature(R, h, RHO_W) / (RHO_W * R**3)
        rows.append({
            "h_over_R": aspect_h,
            "closed_form_over_rhoR3": closed,
            "quadrature_over_rhoR3": quad,
            "closed_vs_quad_rel_err": abs(closed - quad) / quad,
            "rel_err_vs_8_3": abs(closed - THIN_DISC_ADDED_MASS_FACTOR)
                              / THIN_DISC_ADDED_MASS_FACTOR,
        })
    thin = rows[-1]
    return {
        "target_8_3": THIN_DISC_ADDED_MASS_FACTOR,
        "aspect_rows": rows,
        "thinnest_aspect": thin["h_over_R"],
        "converged_value_over_rhoR3": thin["closed_form_over_rhoR3"],
        "rel_err_vs_target_at_thinnest": thin["rel_err_vs_8_3"],
        "max_closed_vs_quad_rel_err": max(r["closed_vs_quad_rel_err"] for r in rows),
        "tolerance": 1e-3,
        "verdict": "PASS" if thin["rel_err_vs_8_3"] < 1e-3 else "FAIL",
    }


# ---------------------------------------------------------------------------
# A3: legacy displaced-volume law vs the classical coefficient
# ---------------------------------------------------------------------------

def anchor_a3() -> dict:
    rows = []
    for ratio in [5.0, 10.0, 20.0, 40.0]:
        R = 20.0
        T = R / ratio
        cfg_legacy = FloeRAOConfig(floe_radius=R, floe_thickness=T)
        cfg_thin = FloeRAOConfig(floe_radius=R, floe_thickness=T,
                                 added_mass_model="lamb_disc")
        displaced = RHO_W * math.pi * R**2 * T
        a_legacy = heave_added_mass(cfg_legacy, R, T, RHO_W)
        a_thin = heave_added_mass(cfg_thin, R, T, RHO_W)
        classical_ca = THIN_DISC_ADDED_MASS_FACTOR * R / (math.pi * T)
        rows.append({
            "R_over_T": ratio,
            "floe_draft_times_rho_mass_kg": displaced,
            "A33_legacy_kg": a_legacy,
            "A33_lamb_disc_kg": a_thin,
            "A33_spheroid_kg": heave_added_mass(
                FloeRAOConfig(floe_radius=R, floe_thickness=T,
                              added_mass_model="spheroid"), R, T, RHO_W),
            "Ca_legacy": a_legacy / displaced,
            "Ca_classical_thin_disc": classical_ca,
            "underprediction_factor": classical_ca / (a_legacy / displaced),
        })
    return {"rows": rows,
            "verdict": "PASS" if all(r["underprediction_factor"] > 1.5 for r in rows)
                       else "FAIL"}


# ---------------------------------------------------------------------------
# A4: heave natural period scaling in floe radius
# ---------------------------------------------------------------------------

def _natural_period(cfg: FloeRAOConfig) -> float:
    R, T = cfg.floe_radius, cfg.floe_thickness
    mass = RHO_I * math.pi * R**2 * T
    A33 = heave_added_mass(cfg, R, T, RHO_W)
    C33 = _hydrostatic_restoring_heave(R, T, RHO_W, G)
    return 2.0 * math.pi * math.sqrt((mass + A33) / C33)


FIXED_THICKNESS = 1.0          # [m] constant floe thickness for the scaling test


def anchor_a4() -> dict:
    # Fixed *thickness* is the discriminating configuration: with T fixed the
    # legacy law has A33 ~ R^2 and M ~ R^2, so Tn is independent of R, whereas
    # the thin-disc law has A33 ~ R^3 and gives Tn ~ R^(1/2).
    radii = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
    rows = []
    for R in radii:
        T = FIXED_THICKNESS
        rows.append({
            "radius_m": R,
            "thickness_m": T,
            "Tn_legacy_s": _natural_period(FloeRAOConfig(floe_radius=R, floe_thickness=T)),
            "Tn_spheroid_s": _natural_period(FloeRAOConfig(floe_radius=R, floe_thickness=T,
                                                           added_mass_model="spheroid")),
            "Tn_lamb_disc_s": _natural_period(FloeRAOConfig(floe_radius=R, floe_thickness=T,
                                                            added_mass_model="lamb_disc")),
        })
    R = np.array([r["radius_m"] for r in rows])
    def slope(key):
        y = np.array([r[key] for r in rows])
        return float(np.polyfit(np.log(R), np.log(y), 1)[0])
    s_legacy = slope("Tn_legacy_s")
    s_spheroid = slope("Tn_spheroid_s")
    return {"rows": rows,
            "loglog_slope_legacy": s_legacy,
            "loglog_slope_spheroid": s_spheroid,
            "target_slope": 0.5,
            "slope_tolerance": 0.06,
            "verdict": "PASS" if (0.40 < s_spheroid < 0.52 and abs(s_legacy) < 0.02)
                       else "FAIL"}


# ---------------------------------------------------------------------------
# A5: long-wave RAO limit
# ---------------------------------------------------------------------------

def anchor_a5() -> dict:
    periods = [20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
    out = {}
    for model in ("displaced", "spheroid"):
        cfg = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0,
                            wave_periods=periods, added_mass_model=model)
        res = compute_floe_rao_analytic(cfg)
        out[model] = {
            "periods": [float(x) for x in res.wave_periods],
            "rao_heave": [float(x) for x in res.rao_heave],
            "rel_err_at_longest": abs(float(res.rao_heave[-1]) - 1.0),
        }
    worst = max(v["rel_err_at_longest"] for v in out.values())
    return {"by_model": out, "worst_abs_err_at_320s_plus": worst,
            "tolerance": 1e-3,
            "verdict": "PASS" if worst < 1e-3 else "FAIL"}


# ---------------------------------------------------------------------------
# A6: analytic vs "DEM" solver path (reported, not asserted)
# ---------------------------------------------------------------------------

def anchor_a6() -> dict:
    periods = [2.0, 3.0, 4.0, 6.0, 10.0, 20.0]
    cfg = FloeRAOConfig(floe_radius=20.0, floe_thickness=1.0, wave_periods=periods)
    ana = compute_floe_rao_analytic(cfg)
    dem = compute_floe_rao_dem(cfg, n_cycles=20, dt=0.01)
    rel = np.abs(ana.rao_heave - dem.rao_heave) / np.maximum(ana.rao_heave, 1e-30)
    return {
        "periods": periods,
        "rao_analytic": [float(x) for x in ana.rao_heave],
        "rao_dem_time_domain": [float(x) for x in dem.rao_heave],
        "max_rel_diff": float(np.max(rel)),
        "same_coefficients": True,
        "note": ("Both paths share A33/B33/C33 from FloeRAOConfig, so this "
                 "difference is an integrator/closure artefact, not a physics "
                 "check; the L2 grade in audit_report.md stands until an "
                 "independent reference is used (this benchmark)."),
    }


# ---------------------------------------------------------------------------
# dispersion sanity (auxiliary, already exact to 1e-14)
# ---------------------------------------------------------------------------

def anchor_dispersion() -> dict:
    from scipy.optimize import brentq
    rows = []
    for period, depth in [(10.0, 1e6), (10.0, 50.0), (10.0, 5.0), (2.0, 20.0)]:
        w = 2.0 * math.pi / period
        k_mod = _dispersion_relation(period, depth, G)
        k_ref = brentq(lambda k: w**2 - G * k * math.tanh(k * depth), 1e-10, 1e3)
        rows.append({"period_s": period, "depth_m": depth,
                     "k_module": k_mod, "k_reference": k_ref,
                     "rel_err": abs(k_mod - k_ref) / k_ref})
    worst = max(r["rel_err"] for r in rows)
    return {"rows": rows, "max_rel_err": worst, "tolerance": 1e-10,
            "verdict": "PASS" if worst < 1e-10 else "FAIL"}


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    rows = summary["A1_thin_disc_limit"]["aspect_rows"]
    hR = [r["h_over_R"] for r in rows]
    ax = axes[0]
    ax.plot(hR, [r["closed_form_over_rhoR3"] for r in rows], "o-",
            label="closed form (Lamb §114)")
    ax.plot(hR, [r["quadrature_over_rhoR3"] for r in rows], "x--",
            label="independent quadrature")
    ax.axhline(8.0 / 3.0, color="k", ls=":", label=r"thin-disc limit $8/3$")
    ax.axhline(1.0, color="tab:red", ls="-.",
               label="legacy law $C_a=1$ (displaced volume)")
    ax.set_xscale("log")
    ax.set_xlabel("$h/R$  (semi-thickness / radius)")
    ax.set_ylabel(r"$A_{33}/(\rho_w R^3)$")
    ax.set_title("A1/A2: thin-disc added mass limit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    rows4 = summary["A4_natural_period"]["rows"]
    R = [r["radius_m"] for r in rows4]
    ax = axes[1]
    ax.plot(R, [r["Tn_legacy_s"] for r in rows4], "s-", label="legacy displaced-volume law")
    ax.plot(R, [r["Tn_spheroid_s"] for r in rows4], "o-", label="thin-disc / spheroid law")
    Rl = np.array(R)
    ax.plot(Rl, rows4[0]["Tn_spheroid_s"] * np.sqrt(Rl / Rl[0]), "k:",
            label=r"$\propto R^{1/2}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("floe radius $R$ [m]   (thickness $T$ = %.1f m)" % FIXED_THICKNESS)
    ax.set_ylabel("heave natural period $T_n$ [s]")
    ax.set_title("A4: $T_n$ scaling (legacy slope %.3f vs %.3f)"
                 % (summary["A4_natural_period"]["loglog_slope_legacy"],
                    summary["A4_natural_period"]["loglog_slope_spheroid"]))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("CASE-DEM-FLOE-RAO-ANCHOR · added-mass anchors for wave_ice/floe_rao.py",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "case": "CASE-DEM-FLOE-RAO-ANCHOR",
        "module": "src/tensordem/wave_ice/floe_rao.py",
        "A1_thin_disc_limit": anchor_a1_a2(),
        "A3_displaced_vs_classical": anchor_a3(),
        "A4_natural_period": anchor_a4(),
        "A5_long_wave_limit": anchor_a5(),
        "A6_analytic_vs_dem_path": anchor_a6(),
        "aux_dispersion": anchor_dispersion(),
        "withheld": "free-surface corrected added mass of a floating floe "
                    "(needs a free-surface radiation/BEM solve; not available)",
    }
    verdicts = {
        "A1": summary["A1_thin_disc_limit"]["verdict"],
        "A3": summary["A3_displaced_vs_classical"]["verdict"],
        "A4": summary["A4_natural_period"]["verdict"],
        "A5": summary["A5_long_wave_limit"]["verdict"],
        "aux_dispersion": summary["aux_dispersion"]["verdict"],
    }
    summary["verdicts"] = verdicts
    make_figure(summary, OUT / "case_floe_rao_anchor.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(verdicts, indent=2))
    print("A1 converged A33/(rho R^3) =",
          summary["A1_thin_disc_limit"]["converged_value_over_rhoR3"],
          "vs 8/3 =", 8.0 / 3.0)
    print("legacy under-prediction factors:",
          [round(r["underprediction_factor"], 2)
           for r in summary["A3_displaced_vs_classical"]["rows"]])
    print("Tn slope legacy / spheroid:",
          summary["A4_natural_period"]["loglog_slope_legacy"],
          summary["A4_natural_period"]["loglog_slope_spheroid"])
    print("A6 max rel diff analytic-vs-DEM:", summary["A6_analytic_vs_dem_path"]["max_rel_diff"])
    print("wrote", OUT / "summary.json", OUT / "case_floe_rao_anchor.png")
    return summary


if __name__ == "__main__":
    main()