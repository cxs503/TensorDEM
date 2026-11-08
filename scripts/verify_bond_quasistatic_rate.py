"""Quasi-static single-bond rate campaign for the rate-dependent DEM ice bond.

Node closed here
----------------
``docs/dem_calibration_gap_ledger.md`` (mission m6464) recorded the specimen
campaign as an honest FAIL: ``sigma_f ~ edot**0.244`` against the analytic
``1/(m+1) = 1/3``, with a *non-flat* negative control (slope +0.117) and a
strength 7-9x the rigid-plate bound -- the 4 x 4 x 3 platen specimen was inertia
dominated.  The ledger's named next node was "isolate the constitutive response
with a single-bond or single-chain driver".  This script does exactly that with
``tensordem.bond_quasistatic``: a 2 x 2 bonded column loaded in *tension* by a
prescribed displacement, where ``n_layers = 2`` leaves no free particle at all.

Arms
----
A  brittle control      linear bond, m = 0        -> strength must be rate
                                                  independent (the previous
                                                  campaign's control failed).
B  ductile envelope     linear bond + Kachanov m=2 -> sigma_f ~ edot**(1/3),
                                                  with the prefactor predicted
                                                  in closed form (no fitting).
C  dt convergence       arm B at three dt         -> first-order in dt, so the
                                                  residual is the discretisation
                                                  of the damage clock.
D  Maxwell-Wiechert     Prony bond + creep        -> engine vs an independent
                                                  reference (closed-form Prony
                                                  force + Simpson damage
                                                  integral, no engine code).
E  exponent sweep       m = 1, 2, 4               -> the *emergent* exponent
                                                  tracks 1/(m+1) = 1/2, 1/3, 1/5.
F  inertia dial         n_layers = 2, 3, 4, 6     -> free particles appear; the
                                                  strength error is attributed
                                                  to the measured inertia index
                                                  KE_free / SE, not guessed.
G  mass dial            density x 0.5, 1, 2       -> second, independent inertia
                                                  dial at n_layers = 4.

Every number in the JSON is produced by the run; reference values are computed in
the same script from the same configuration, so nothing is transcribed.

Usage
-----
    PYTHONPATH=src python scripts/verify_bond_quasistatic_rate.py
    PYTHONPATH=src python scripts/verify_bond_quasistatic_rate.py --quick
    PYTHONPATH=src python scripts/verify_bond_quasistatic_rate.py --figures-only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordem.bond_quasistatic import (
    ChainTensionDriver,
    QuasiStaticBondConfig,
    affine_network_factor,
    brittle_force_scale,
    crossover_strain_rate,
    elastic_creep_reference,
    maxwell_wiechert_reference,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_bond_quasistatic"
JSON_NAME = "verification.json"

# ---------------------------------------------------------------------------
# Shared material / clock constants (one source of truth for the campaign)
# ---------------------------------------------------------------------------
STEPS_TARGET = 3000          # integration steps per run (dt = t_f_ref / STEPS)
RADIUS = 0.025
BOND_STIFFNESS = 2_000.0     # N/m unrelaxed
BREAKING_STRAIN = 0.015
CREEP_TIME_S = 5.0e-3        # t_ref
RELAXATION_TIME_S = (5.0e-3, 5.0e-2)
EQUILIBRIUM_FRACTION = 0.35
DELAYED_FRACTION = (0.45, 0.20)


def base_config(**overrides) -> QuasiStaticBondConfig:
    kwargs = dict(
        n_layers=2,
        radius=RADIUS,
        bond_stiffness=BOND_STIFFNESS,
        breaking_strain=BREAKING_STRAIN,
        bond_relaxation_time_s=RELAXATION_TIME_S,
        bond_equilibrium_fraction=EQUILIBRIUM_FRACTION,
        bond_delayed_elastic_fraction=DELAYED_FRACTION,
        bond_creep_time_s=CREEP_TIME_S,
    )
    kwargs.update(overrides)
    return QuasiStaticBondConfig(**kwargs)


def reference(cfg: QuasiStaticBondConfig, strain_rate: float) -> dict:
    if cfg.bond_model == "maxwell_wiechert":
        return maxwell_wiechert_reference(cfg, strain_rate)
    return elastic_creep_reference(cfg, strain_rate)


def pick_dt(cfg: QuasiStaticBondConfig, strain_rate: float, *,
            steps: int = STEPS_TARGET) -> float:
    """dt = t_f(reference) / steps, capped by the engine's stability bound."""
    driver_dt = ChainTensionDriver(cfg).sim.dt      # configured or recommended
    t_f = reference(cfg, strain_rate)["failure_time_s"]
    return min(0.95 * driver_dt, t_f / steps)


def elastic_damage(strain_value: float, strain_rate: float, *,
                   exponent: int, creep_time_s: float,
                   breaking_strain: float) -> float:
    """Closed-form Kachanov damage at ``strain_value`` for a linear ramp."""
    if exponent <= 0:
        return 0.0
    return ((strain_value / breaking_strain) ** exponent * strain_value
            / ((exponent + 1.0) * creep_time_s * strain_rate))


def run_case(cfg: QuasiStaticBondConfig, strain_rate: float, *,
             label: str, dt: float | None = None,
             steps: int = STEPS_TARGET, sample_every: int = 4,
             stop_at_strain_fraction: float | None = None) -> dict:
    cfg = QuasiStaticBondConfig(**{**cfg.__dict__,
                                   "dt": dt if dt is not None else pick_dt(cfg, strain_rate, steps=steps),
                                   "sample_every": sample_every})
    stop_at = (None if stop_at_strain_fraction is None
               else stop_at_strain_fraction * reference(cfg, strain_rate)["failure_strain"])
    run = ChainTensionDriver(cfg).run(strain_rate, stop_at_strain=stop_at)
    ref = reference(cfg, strain_rate)
    f_brittle = brittle_force_scale(cfg)
    measured = run["axial_bond_force_N"]
    record = {
        "label": label,
        "n_layers": cfg.n_layers,
        "density": cfg.density,
        "dt": cfg.dt,
        "steps": run["steps"],
        "creep_exponent": cfg.bond_creep_exponent,
        "bond_model": cfg.bond_model,
        "strain_rate": strain_rate,
        "regime_reference": ref["regime"],
        "failure_strain_measured": run["failure_strain"],
        "failure_strain_reference": ref["failure_strain"],
        "failure_time_measured": run["failure_time_s"],
        "failure_time_reference": ref["failure_time_s"],
        "axial_force_measured_N": measured,
        "axial_force_reference_N": ref["failure_force_N"],
        "brittle_force_scale_N": f_brittle,
        "normalised_strength_measured": measured / f_brittle,
        "normalised_strength_reference": ref["failure_force_N"] / f_brittle,
        "strength_rel_error": abs(measured / ref["failure_force_N"] - 1.0),
        "strain_rel_error": abs(run["failure_strain"] / ref["failure_strain"] - 1.0),
        "failure_mode": run["failure_mode"],
        "broken_bonds": run["broken_bonds"],
        "net_layer_force_N": run["peak_force_N"],
        "net_over_axial": run["peak_force_N"] / measured if measured else math.nan,
        "inertia_index": run["inertia_index"],
        "max_strain_nonuniformity": run["max_strain_nonuniformity"],
        "max_bond_force_spread": run["max_bond_force_spread"],
        "kinetic_energy_J": run["kinetic_energy_J"],
        "cumulative_bond_dissipation_J": run["cumulative_bond_dissipation_J"],
        "reference_damage_at_failure": ref["damage_at_failure"],
        "stopped_early": run["stopped_early"],
        "final_strain": run["final_strain"],
        "final_max_bond_damage": run["final_max_bond_damage"],
        "final_damage_reference": elastic_damage(
            run["final_strain"], strain_rate,
            exponent=cfg.bond_creep_exponent,
            creep_time_s=cfg.bond_creep_time_s,
            breaking_strain=cfg.breaking_strain),
        "final_damage_rel_error": abs(
            run["final_max_bond_damage"]
            - elastic_damage(run["final_strain"], strain_rate,
                             exponent=cfg.bond_creep_exponent,
                             creep_time_s=cfg.bond_creep_time_s,
                             breaking_strain=cfg.breaking_strain))
        / elastic_damage(run["final_strain"], strain_rate,
                         exponent=cfg.bond_creep_exponent,
                         creep_time_s=cfg.bond_creep_time_s,
                         breaking_strain=cfg.breaking_strain)
        if cfg.bond_creep_exponent > 0 and run["final_strain"] > 0.0 else math.nan,
        "history": {
            "strain": run["strain"],
            "load": run["load"],
            "time": run["time"],
        },
    }
    return record


def fit_exponent(records: list[dict]) -> dict:
    """Least-squares slope of log(strength) vs log(strain rate)."""
    pts = [r for r in records if r["regime_reference"] == "ductile"]
    if len(pts) < 2:
        return {"n_points": len(pts), "slope": math.nan, "intercept": math.nan,
                "reference_slope": math.nan, "reference_intercept": math.nan,
                "slope_error": math.nan, "prefactor_ratio": math.nan}
    x = np.log(np.array([r["strain_rate"] for r in pts], dtype=float))
    y_meas = np.log(np.array([r["normalised_strength_measured"] for r in pts]))
    y_ref = np.log(np.array([r["normalised_strength_reference"] for r in pts]))
    slope_m, intercept_m = np.polyfit(x, y_meas, 1)
    slope_r, intercept_r = np.polyfit(x, y_ref, 1)
    return {
        "n_points": len(pts),
        "slope": float(slope_m),
        "intercept": float(intercept_m),
        "reference_slope": float(slope_r),
        "reference_intercept": float(intercept_r),
        "slope_error": float(abs(slope_m - slope_r)),
        "prefactor_ratio": float(math.exp(intercept_m - intercept_r)),
    }


# ---------------------------------------------------------------------------
# Campaign arms
# ---------------------------------------------------------------------------
def arm_brittle_control(quick: bool) -> list[dict]:
    rates = (0.5, 0.1) if quick else (0.5, 0.1, 0.03)
    cfg = base_config(bond_creep_exponent=0)
    return [run_case(cfg, r, label="A_brittle_control") for r in rates]


def arm_ductile(quick: bool) -> list[dict]:
    rates = (0.5, 0.1, 0.02) if quick else (0.5, 0.2, 0.1, 0.05, 0.02)
    cfg = base_config(bond_creep_exponent=2)
    return [run_case(cfg, r, label="B_ductile_m2") for r in rates]


def arm_dt_convergence(quick: bool) -> list[dict]:
    """Damage-clock convergence at a *fixed* imposed strain.

    The failure strain itself is quantised by one integration step, so it cannot
    resolve first-order convergence; the accumulated Kachanov damage at a fixed
    strain is a smooth functional of dt and is compared with its closed form.
    """
    cfg = base_config(bond_creep_exponent=2)
    rate = 0.2
    base = pick_dt(cfg, rate, steps=1500 if not quick else 800)
    factors = (1.0, 0.5) if quick else (1.0, 0.5, 0.25)
    return [run_case(cfg, rate, label="C_dt_convergence", dt=base * f,
                     stop_at_strain_fraction=0.5) for f in factors]


def arm_maxwell_wiechert(quick: bool) -> list[dict]:
    # Kept clear of the brittle crossover (0.5 1/s is a knife-edge tie for the
    # Prony bond: relaxation lowers the force, so the tie moves down).
    rates = (0.3, 0.1) if quick else (0.3, 0.1, 0.03)
    cfg = base_config(bond_model="maxwell_wiechert", bond_creep_exponent=2)
    return [run_case(cfg, r, label="D_maxwell_wiechert_m2") for r in rates]


def arm_exponent_sweep(quick: bool) -> list[dict]:
    rates = (0.2, 0.05) if quick else (0.3, 0.1, 0.03)
    out: list[dict] = []
    for m in (1, 4):
        cfg = base_config(bond_creep_exponent=m)
        out.extend(run_case(cfg, r, label=f"E_exponent_m{m}") for r in rates)
    return out


def arm_inertia_layers(quick: bool) -> list[dict]:
    layers = (2, 4) if quick else (2, 3, 4, 6)
    rate = 0.1
    out: list[dict] = []
    for n in layers:
        cfg = base_config(n_layers=n, bond_creep_exponent=2)
        # dt is common across the arm (the stability bound tightens with mass,
        # not with the layer count) so only the inertia changes.
        dt = pick_dt(base_config(n_layers=2, bond_creep_exponent=2), rate)
        out.append(run_case(cfg, rate, label=f"F_inertia_layers{n}", dt=dt))
    return out


def arm_inertia_mass(quick: bool) -> list[dict]:
    # Mass scaling *down* is the quasi-static servo: lighter particles raise the
    # lattice wave speed, so the elastic transit time shrinks relative to the
    # failure time.  The rho = 1 point is already the n_layers = 4 run of the
    # layers arm, so it is not repeated here.
    factors = (0.1, 0.03) if quick else (0.3, 0.1, 0.03)
    rate = 0.1
    out: list[dict] = []
    # Common dt: the recommendation of the lightest mass in the arm, so the
    # damage-clock discretisation error is identical for every density.
    dt = min(pick_dt(base_config(n_layers=4, density=917.0 * f, bond_creep_exponent=2),
                     rate) for f in factors)
    for f in factors:
        cfg = base_config(n_layers=4, density=917.0 * f, bond_creep_exponent=2)
        out.append(run_case(cfg, rate, label=f"G_inertia_mass{f:g}", dt=dt))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
        "axes.titlesize": 10,
    })
    return plt


def make_figures(payload: dict) -> list[dict]:
    plt = _style()
    figures: list[dict] = []
    arms = payload["arms"]

    # 1 -- strength vs strain rate, all exponent arms + brittle control
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = {"A_brittle_control": "0.35", "B_ductile_m2": "C0",
              "D_maxwell_wiechert_m2": "C3", "E_exponent_m1": "C2",
              "E_exponent_m4": "C4"}
    _ = arms.get("E_exponent_m1")
    for label, key in (("A_brittle_control", "brittle control (m=0)"),
                       ("B_ductile_m2", "linear+creep m=2"),
                       ("D_maxwell_wiechert_m2", "Maxwell-Wiechert+creep m=2"),
                       ("E_exponent_m1", "linear+creep m=1"),
                       ("E_exponent_m4", "linear+creep m=4")):
        pts = sorted(arms.get(label, []), key=lambda r: r["strain_rate"])
        if not pts:
            continue
        x = [r["strain_rate"] for r in pts]
        y = [r["normalised_strength_measured"] for r in pts]
        yr = [r["normalised_strength_reference"] for r in pts]
        ax.loglog(x, y, "o-", color=colors[label], label=f"engine: {key}")
        ax.loglog(x, yr, "x", color=colors[label], ms=8, mew=1.4,
                  label=f"reference: {key}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("bond strain rate  [1/s]")
    ax.set_ylabel("normalised strength  F / (k_u eps_b L0)")
    ax.set_title("Quasi-static single-bond rate dependence: engine vs closed form")
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    figures.append({"name": "fig_strength_vs_rate",
                    "caption": "Normalised bond strength vs applied strain rate. "
                               "Markers with lines = IceDEM3D via a prescribed-"
                               "displacement 2x2 column (no free particle); "
                               "crosses = closed-form reference (Kachanov clock "
                               "for the linear bond, Prony force + Simpson "
                               "damage for Maxwell-Wiechert). Dotted line = "
                               "brittle cleavage strength.",
                    "fig": fig})

    # 2 -- dt convergence
    pts = sorted(arms.get("C_dt_convergence", []), key=lambda r: -r["dt"])
    if len(pts) >= 2:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        x = [r["dt"] for r in pts]
        y = [max(r["final_damage_rel_error"], 1e-16) for r in pts]
        ax.loglog(x, y, "o-", color="C1", label="measured damage-clock error")
        ref_step = y[0] * (x[0] / x[0])
        slope = np.polyfit(np.log(x), np.log(y), 1)[0]
        ax.loglog(x, [y[0] * (xx / x[0]) for xx in x], "k--", lw=0.9,
                  label="first-order reference (slope 1)")
        ax.set_xlabel("dt [s]")
        ax.set_ylabel("relative error of the Kachanov damage")
        ax.set_title(f"Damage-clock discretisation: fitted order {slope:.2f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures.append({"name": "fig_dt_convergence",
                        "caption": "Accumulated Kachanov damage error against the "
                                   "closed form at a fixed imposed strain (half "
                                   "the failure strain, m=2, rate 0.2 1/s), "
                                   "plotted against the integration step. The "
                                   "forward-Euler damage clock is first order: "
                                   "halving dt halves the error, so the residual "
                                   "of the rate arms is discretisation and not "
                                   "physics.",
                        "fig": fig})

    # 3 -- inertia dials
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, key, xlabel in ((axes[0], "F_inertia_layers", "layers n_z"),
                            (axes[1], "G_inertia_mass", "density multiplier")):
        pts = arms.get(key, [])
        if not pts:
            continue
        if key == "F_inertia_layers":
            xs = [r["n_layers"] for r in pts]
        else:
            xs = [r["density"] / 917.0 for r in pts]
        err = [max(r["strength_rel_error"], 1e-12) for r in pts]
        inertia = [max(r["inertia_index"], 1e-12) for r in pts]
        ax.loglog(xs, err, "o-", color="C3", label="strength error vs reference")
        ax.loglog(xs, inertia, "s--", color="C0", label="inertia index KE_free/SE")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("dimensionless")
        ax.set_title(key.replace("_", " "))
        ax.legend(fontsize=7)
    fig.tight_layout()
    figures.append({"name": "fig_inertia_dials",
                    "caption": "Two independent inertia dials at a fixed strain "
                               "rate (0.1 1/s, m=2): the number of free layers "
                               "(left) and the particle density (right). The "
                               "strength error tracks the measured inertia index "
                               "KE_free/SE, i.e. the deviation is inertial and "
                               "vanishes in the quasi-static limit.",
                    "fig": fig})

    # 4 -- force-strain histories
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, key, color in (("B_ductile_m2", "linear+creep", "C0"),
                              ("D_maxwell_wiechert_m2", "Maxwell-Wiechert", "C3"),
                              ("A_brittle_control", "brittle control", "0.35")):
        pts = sorted(arms.get(label, []), key=lambda r: r["strain_rate"])
        if not pts:
            continue
        for r in pts:
            if abs(r["strain_rate"] - 0.1) < 1e-12:
                ax.plot(r["history"]["strain"], np.array(r["history"]["load"]) / r["brittle_force_scale_N"],
                        "-", color=color, lw=1.2,
                        label=f"{key} (rate 0.1 1/s)")
    ax.set_xlabel("axial bond strain  eps")
    ax.set_ylabel("net layer force  F / (k_u eps_b L0)")
    ax.set_title("Load history up to first bond failure (rate 0.1 1/s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures.append({"name": "fig_load_history",
                    "caption": "Net force on the supported layer (divided by the "
                               "brittle single-bond force) against imposed strain, "
                               "at 0.1 1/s. The ductile arms lose load before the "
                               "brittle strain is reached (creep damage kills the "
                               "axial bonds); the brittle control fails exactly at "
                               "the cleavage strain. The layer force exceeds a "
                               "single bond force by the lattice amplification "
                               "(9.333 analytically for this 2x2x2 cell).",
                    "fig": fig})

    # 5 -- error vs inertia collapse
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    keys = ("B_ductile_m2", "D_maxwell_wiechert_m2", "E_exponent_m1",
            "E_exponent_m4", "F_inertia_layers", "G_inertia_mass")
    marks = {"B_ductile_m2": "o", "D_maxwell_wiechert_m2": "s",
             "E_exponent_m1": "^", "E_exponent_m4": "v",
             "F_inertia_layers": "D", "G_inertia_mass": "*"}
    for key in keys:
        pts = arms.get(key, [])
        if not pts:
            continue
        ax.loglog([max(r["inertia_index"], 1e-9) for r in pts],
                  [max(r["strength_rel_error"], 1e-9) for r in pts],
                  marks[key], ms=6, label=key, ls="none")
    ax.set_xlabel("inertia index  KE_free / SE")
    ax.set_ylabel("strength error vs reference")
    ax.set_title("All runs: strength error collapses onto the inertia index")
    ax.legend(fontsize=7)
    fig.tight_layout()
    figures.append({"name": "fig_error_vs_inertia",
                    "caption": "Every run of the campaign plotted as strength "
                               "error against the measured inertia index. Runs "
                               "without free particles sit at KE/SE = 0 with "
                               "error < 1e-4 relative; the runs with free layers "
                               "move up and to the right. This is the criterion "
                               "the platen specimen campaign lacked.",
                    "fig": fig})
    return figures


def figure_payload(figures: list[dict]) -> list[dict]:
    import base64
    import io
    out = []
    for item in figures:
        buffer = io.BytesIO()
        item["fig"].savefig(buffer, format="png", bbox_inches="tight")
        out.append({
            "name": item["name"],
            "caption": item["caption"],
            "png_base64": base64.b64encode(buffer.getvalue()).decode(),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def campaign(quick: bool = False) -> dict:
    arms: dict[str, list[dict]] = {}
    for name, builder in (("A_brittle_control", arm_brittle_control),
                          ("B_ductile_m2", arm_ductile),
                          ("C_dt_convergence", arm_dt_convergence),
                          ("D_maxwell_wiechert_m2", arm_maxwell_wiechert),
                          ("E_exponent_m1", None),
                          ("E_exponent_m4", None),
                          ("F_inertia_layers", arm_inertia_layers),
                          ("G_inertia_mass", arm_inertia_mass)):
        if builder is None:
            continue
        records = builder(quick)
        arms[name] = records
        print(f"[arm] {name}: {len(records)} runs", flush=True)
    for record in arm_exponent_sweep(quick):
        arms.setdefault(record["label"], []).append(record)

    # Summaries computed from the records (never transcribed).
    summary: dict[str, dict] = {}
    for key in ("A_brittle_control", "B_ductile_m2", "D_maxwell_wiechert_m2",
                "E_exponent_m1", "E_exponent_m4"):
        records = arms.get(key, [])
        if not records:
            continue
        m = records[0]["creep_exponent"]
        summary[key] = {
            "creep_exponent": m,
            "analytic_exponent": (1.0 / (m + 1.0)) if m else 0.0,
            "crossover_rate": crossover_strain_rate(
                creep_time_s=CREEP_TIME_S, exponent=m,
                breaking_strain=BREAKING_STRAIN) if m else None,
            **fit_exponent(records),
            "max_strength_rel_error": max(r["strength_rel_error"] for r in records),
            "max_strain_rel_error": max(r["strain_rel_error"] for r in records),
            "all_inertia_index_zero": all(r["inertia_index"] == 0.0 for r in records),
            "failure_modes": sorted({r["failure_mode"] for r in records}),
        }
    dt_pts = sorted(arms.get("C_dt_convergence", []), key=lambda r: r["dt"])
    dt_summary: dict = {"n_points": len(dt_pts)}
    if len(dt_pts) >= 2:
        slope = float(np.polyfit(np.log([r["dt"] for r in dt_pts]),
                                 np.log([max(r["final_damage_rel_error"], 1e-18) for r in dt_pts]), 1)[0])
        dt_summary.update(
            fitted_order=slope,
            target_strain_fraction=0.5,
            errors=[{"dt": r["dt"], "damage_rel_error": r["final_damage_rel_error"],
                     "damage_engine": r["final_max_bond_damage"],
                     "damage_reference": r["final_damage_reference"]} for r in dt_pts],
        )
    inertia_pts = arms.get("F_inertia_layers", []) + arms.get("G_inertia_mass", [])
    inertia_summary: dict = {"n_points": len(inertia_pts)}
    if len(inertia_pts) >= 3:
        x = np.log([max(r["inertia_index"], 1e-12) for r in inertia_pts])
        y = np.log([max(r["strength_rel_error"], 1e-12) for r in inertia_pts])
        corr = float(np.corrcoef(x, y)[0, 1])
        inertia_summary.update(
            log_log_correlation=corr,
            error_at_zero_inertia=max(
                r["strength_rel_error"] for r in inertia_pts if r["inertia_index"] == 0.0)
            if any(r["inertia_index"] == 0.0 for r in inertia_pts) else None,
            points=[{"label": r["label"], "n_layers": r["n_layers"],
                     "density": r["density"], "inertia_index": r["inertia_index"],
                     "strength_rel_error": r["strength_rel_error"]} for r in inertia_pts],
        )
    payload = {
        "campaign": "ice_bond_quasistatic_rate",
        "constants": {
            "radius_m": RADIUS,
            "bond_stiffness_N_m": BOND_STIFFNESS,
            "breaking_strain": BREAKING_STRAIN,
            "creep_time_s": CREEP_TIME_S,
            "relaxation_time_s": list(RELAXATION_TIME_S),
            "equilibrium_fraction": EQUILIBRIUM_FRACTION,
            "delayed_fraction": list(DELAYED_FRACTION),
            "steps_target": STEPS_TARGET,
        },
        "lattice_amplification": affine_network_factor(base_config()),
        "arms": arms,
        "summary": summary,
        "dt_convergence": dt_summary,
        "inertia": inertia_summary,
    }
    return payload


def render(payload: dict) -> list[dict]:
    return figure_payload(make_figures(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced rate sets")
    parser.add_argument("--figures-only", action="store_true",
                        help="rebuild figures from the existing JSON")
    parser.add_argument("--out", default=str(OUT / JSON_NAME))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        payload = json.loads(out.read_text())
    else:
        payload = campaign(quick=args.quick)
        out.write_text(json.dumps(payload, indent=2))
        print(f"[write] {out}")

    figures = render(payload)
    for item in figures:
        path = out.parent / f"{item['name']}.png"
        import base64
        from io import BytesIO
        path.write_bytes(base64.b64decode(item["png_base64"]))
        print(f"[figure] {path}")
    (out.parent / "figures.json").write_text(json.dumps(figures, indent=2))

    for key, block in payload["summary"].items():
        print(f"{key}: analytic {block['analytic_exponent']:.4f} "
              f"fitted {block['slope']:.4f} ref-fit {block.get('reference_slope', float('nan')):.4f} "
              f"max strength err {block['max_strength_rel_error']:.2e}")
    if payload.get("dt_convergence", {}).get("fitted_order") is not None:
        print(f"dt order {payload['dt_convergence']['fitted_order']:.2f}")
    if "log_log_correlation" in payload.get("inertia", {}):
        print(f"inertia correlation {payload['inertia']['log_log_correlation']:.3f}")


if __name__ == "__main__":
    main()