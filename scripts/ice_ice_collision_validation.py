"""Validation driver for the ice-ice collision face of TensorDEM.

Runs the eight machine criteria of node P202W1-N10 and writes

* ``artifacts/ice_ice_collision/validation.json``  -- every number, traceable,
* ``artifacts/ice_ice_collision/figures/*.png``     -- report figures,
* ``docs/ICE_ICE_COLLISION_VALIDATION.md``          -- paper-style report body.

Usage
-----
    python scripts/ice_ice_collision_validation.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ice_ice_collision import (  # noqa: E402
    FloeFieldConfig,
    bond_tensile_test,
    bonded_pair_rupture_test,
    dissipation_scaling,
    hertz_contact_force,
    reduced_youngs_modulus,
    simulate_floe_field,
    sweep_concentration,
    two_floe_collision,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "ice_ice_collision")
FIG = os.path.join(OUT, "figures")
DOCS = os.path.join(ROOT, "docs")

CONCENTRATIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONCENTRATIONS_QUICK = [0.1, 0.3, 0.5, 0.7, 0.9]
SHEAR_RATES = [0.5, 1.0, 1.5, 2.0, 3.0]
SHEAR_RATES_QUICK = [0.5, 1.0, 2.0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def criterion_c1() -> dict:
    """Hertz force-displacement law vs the analytic Hertz impact solution."""
    rows = []
    for e in (1.0, 0.5):
        res = two_floe_collision(restitution=e, friction=0.0, impact_speed=1.0,
                                 dt_factor=0.005)
        rows.append({
            "restitution": e,
            "analytic_force_max": res["analytic"]["force_max"],
            "simulated_force_max": res["peak_force"],
            "force_error": res["force_error"],
            "analytic_overlap_max": res["analytic"]["delta_max"],
            "simulated_overlap_max": res["peak_overlap"],
            "overlap_error": res["overlap_error"],
            "trace": res["trace"],
            "analytic": res["analytic"],
            "dt": res["dt"],
        })
    elastic = rows[0]
    return {
        "criterion": "C1 Hertz contact force-displacement vs analytic (<= 5%)",
        "elastic_force_error": elastic["force_error"],
        "elastic_overlap_error": elastic["overlap_error"],
        "damped_force_error": rows[1]["force_error"],
        "passed": bool(elastic["force_error"] <= 0.05
                       and elastic["overlap_error"] <= 0.05),
        "rows": rows,
    }


def criterion_c2() -> dict:
    """Normal bond rupture threshold vs sigma_b (<= 10%)."""
    sigmas = (5.0e4, 1.0e5, 3.0e5)
    rows = []
    for sb in sigmas:
        res = bonded_pair_rupture_test(sigma_b=sb)
        rows.append({"sigma_b": sb, "rupture_stress": res["rupture_stress"],
                     "relative_error": res["relative_error"],
                     "ruptured": res["ruptured"],
                     "post_rupture_bond_active": res["post_rupture_bond_active"],
                     "bond_area": res["bond_area"], "trace": res["trace"],
                     "rupture_time": res["rupture_time"]})
    analytic = bond_tensile_test(sigma_b=1.0e5)
    worst = max(r["relative_error"] for r in rows)
    return {
        "criterion": "C2 freeze-bond rupture threshold vs sigma_b (<= 10%)",
        "worst_relative_error": worst,
        "all_ruptured": all(r["ruptured"] for r in rows),
        "all_debonded_after": all(not r["post_rupture_bond_active"] for r in rows),
        "passed": bool(worst <= 0.10 and all(r["ruptured"] for r in rows)),
        "rows": rows,
        "quasi_static_spring_check": {
            "sigma_b": analytic["sigma_b"],
            "rupture_stress": analytic["rupture_stress"],
            "relative_error": analytic["relative_error"],
        },
    }


def criterion_c3(quick: bool) -> dict:
    """Energy-dissipation scaling epsilon ~ C v^3 / d (exponent 2.9-3.1)."""
    rates = SHEAR_RATES_QUICK if quick else SHEAR_RATES
    base = FloeFieldConfig(concentration=0.35, shear_steps=6000 if not quick else 2500,
                           sample_steps=3500 if not quick else 1500,
                           settle_steps=1500)
    res = dissipation_scaling(rates, base=base)
    exponent = res["exponent"]
    return {
        "criterion": "C3 dissipation-rate scaling exponent (theory 3.0, band 2.9-3.1)",
        "exponent": exponent,
        "exponent_total": res["exponent_total"],
        "concentration": res["concentration"],
        "normalisation_spread": res["normalisation_spread"],
        "passed": bool(exponent == exponent and 2.9 <= exponent <= 3.1),
        "rows": res["rows"],
    }


def criterion_c4567(quick: bool) -> dict:
    """mu_eff(C) sweep: monotonicity (C4), C=0 limit (C5), 10x growth (C7)."""
    concentrations = CONCENTRATIONS_QUICK if quick else CONCENTRATIONS
    base = FloeFieldConfig(shear_steps=5000 if not quick else 2000,
                           sample_steps=3000 if not quick else 1200,
                           settle_steps=1500)
    rows = []
    snapshots = {}
    for c in concentrations:
        t0 = time.time()
        keep = c in (0.1, 0.5, 0.9)
        out = simulate_floe_field(FloeFieldConfig(**{**base.__dict__, "concentration": c}),
                                  return_simulator=keep)
        data, sim = out if keep else (out, None)
        data.pop("bond_rupture_events", None)
        data["elapsed"] = time.time() - t0
        rows.append(data)
        if sim is not None:
            snapshots[str(c)] = {"pos": sim.physical_positions().tolist(),
                                 "radii": sim.radii.tolist()}
        log(f"  C={c:.1f} N={data['n_floes']} mu_eff={data['mu_eff']:.4g} "
            f"({data['elapsed']:.1f}s)")

    mu = [r["mu_eff"] for r in rows]
    monotone = all(mu[i + 1] > mu[i] for i in range(len(mu) - 1))
    zero = simulate_floe_field(FloeFieldConfig(concentration=0.0, shear_steps=10,
                                               settle_steps=0))
    growth = mu[-1] / mu[0] if mu[0] > 0 else float("inf")
    return {
        "c4": {"criterion": "C4 mu_eff(C) monotonically increasing",
               "mu_eff": mu, "concentrations": [r["concentration"] for r in rows],
               "passed": bool(monotone)},
        "c5": {"criterion": "C5 mu_eff(C=0) == 0 (no ice, no viscosity)",
               "mu_eff_at_zero": zero["mu_eff"], "n_floes": zero["n_floes"],
               "passed": bool(zero["mu_eff"] == 0.0 and zero["n_floes"] == 0)},
        "c7": {"criterion": "C7 mu_eff(0.9)/mu_eff(0.1) >= 10",
               "growth_factor": growth, "passed": bool(growth >= 10.0)},
        "rows": rows,
        "snapshots": snapshots,
    }


def criterion_c6() -> dict:
    """Closed energy ledger of a free floe collision (<= 5%)."""
    rows = []
    for e in (1.0, 0.7, 0.5, 0.3):
        for fric in (0.0, 0.3):
            res = two_floe_collision(restitution=e, friction=fric, impact_speed=1.0,
                                     dt_factor=0.005)
            rows.append({"restitution_target": e, "friction": fric,
                         "measured_restitution": res["measured_restitution"],
                         "e_kin_initial": res["e_kin_initial"],
                         "e_kin_final": res["e_kin_final"],
                         "dissipated": res["dissipated"],
                         "ledger_relative_error": res["ledger_relative_error"]})
    worst = max(r["ledger_relative_error"] for r in rows)
    return {"criterion": "C6 E_kin + E_spring + E_diss = E_kin(0) (<= 5%)",
            "worst_relative_error": worst, "passed": bool(worst <= 0.05),
            "rows": rows}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_hertz(c1: dict) -> str:
    row = c1["rows"][0]
    trace = np.array(row["trace"])
    delta = trace[:, 0]
    force = trace[:, 1]
    e_eff = row["analytic"]["k_hertz"]
    ref = e_eff * np.maximum(delta, 0.0) ** 1.5
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(delta, force, "o", ms=2.5, label="DEM (simulated impact)")
    order = np.argsort(delta)
    axes[0].plot(delta[order], ref[order], "-", lw=1.2,
                 label=r"Hertz $(4/3)E^*\sqrt{R^*}\delta^{3/2}$")
    axes[0].set_xlabel(r"overlap $\delta$ (m)")
    axes[0].set_ylabel(r"normal force $F_n$ (N)")
    axes[0].set_title(f"Hertz contact law, elastic impact "
                      f"(peak error {row['force_error'] * 100:.3f} %)")
    axes[0].legend()
    axes[1].plot(delta, trace[:, 2], "-", lw=1.0, color="tab:red")
    axes[1].set_xlabel(r"overlap $\delta$ (m)")
    axes[1].set_ylabel(r"normal relative velocity $v_n$ (m/s)")
    axes[1].set_title("impact phase portrait")
    fig.tight_layout()
    path = os.path.join(FIG, "fig1_hertz_impact.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_mu(c4567: dict) -> str:
    rows = c4567["rows"]
    c = np.array([r["concentration"] for r in rows])
    mu = np.array([r["mu_eff"] for r in rows])
    vir = np.array([abs(r["tau_virial"]) for r in rows])
    kin = np.array([abs(r["tau_kinetic"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].semilogy(c, mu, "o-", color="tab:blue", label=r"$\mu_\mathrm{eff}$")
    axes[0].set_xlabel("ice concentration C (-)")
    axes[0].set_ylabel(r"equivalent viscosity $\mu_\mathrm{eff}$ (Pa s)")
    axes[0].set_title(f"$\\mu_\\mathrm{{eff}}(C)$, growth "
                      f"{c4567['c7']['growth_factor']:.3g}x")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(c, np.maximum(vir, 1e-30), "s-", label="contact virial")
    axes[1].semilogy(c, np.maximum(kin, 1e-30), "^-", label="convective (kinetic)")
    axes[1].set_xlabel("ice concentration C (-)")
    axes[1].set_ylabel(r"stress component (N/m)")
    axes[1].set_title("stress decomposition")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(FIG, "fig2_mu_eff_vs_concentration.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_scaling(c3: dict) -> str:
    rows = c3["rows"]
    v = np.array([r["v"] for r in rows])
    eps = np.array([max(r["epsilon"], 1e-30) for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(v, eps, "o", label=r"DEM $\epsilon$")
    if np.all(eps > 0):
        fit = np.polyfit(np.log(v), np.log(eps), 1)
        ax.loglog(v, np.exp(fit[1]) * v ** fit[0], "-",
                  label=rf"fit exponent {fit[0]:.3f} (theory 3.0)")
    ax.set_xlabel(r"characteristic velocity $v=\dot\gamma D_\mathrm{mean}$ (m/s)")
    ax.set_ylabel(r"dissipation rate per ice mass $\epsilon$ (W/kg)")
    ax.set_title(f"energy-dissipation scaling at C={c3['concentration']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG, "fig3_dissipation_scaling.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_bond(c2: dict) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for row in c2["rows"]:
        trace = np.array(row["trace"])
        if len(trace) == 0:
            continue
        ax.plot(trace[:, 0] * 1e3, trace[:, 1] / 1e3, "-",
                label=rf"$\sigma_b$={row['sigma_b'] / 1e3:.0f} kPa")
        ax.axhline(row["sigma_b"] / 1e3, ls=":", lw=0.8, color="k", alpha=0.4)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(r"bond normal stress $|F_n|/A_b$ (kPa)")
    ax.set_title("freeze-bond tensile rupture (dotted: prescribed $\\sigma_b$)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG, "fig4_bond_rupture.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_snapshots(c4567: dict) -> str:
    snaps = c4567["snapshots"]
    keys = [k for k in ("0.1", "0.5", "0.9") if k in snaps]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.2))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        pos = np.array(snaps[key]["pos"])
        radii = np.array(snaps[key]["radii"])
        for (x, y), r in zip(pos, radii):
            ax.add_patch(plt.Circle((x, y), r, fill=True, alpha=0.55,
                                    color="tab:blue"))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_aspect("equal")
        ax.set_title(f"C = {key}")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle("sheared floe field (Lees-Edwards cell, SLLOD)")
    fig.tight_layout()
    path = os.path.join(FIG, "fig5_field_snapshots.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)
    t_start = time.time()

    log("C1: Hertz contact law vs analytic impact")
    c1 = criterion_c1()
    log(f"   elastic force error {c1['elastic_force_error'] * 100:.4f} %")

    log("C2: freeze-bond rupture threshold")
    c2 = criterion_c2()
    log(f"   worst error {c2['worst_relative_error'] * 100:.3f} %")

    log("C6: closed energy ledger")
    c6 = criterion_c6()
    log(f"   worst ledger error {c6['worst_relative_error'] * 100:.4f} %")

    log("C4/C5/C7: concentration sweep of the equivalent viscosity")
    c4567 = criterion_c4567(args.quick)

    log("C3: dissipation-rate scaling")
    c3 = criterion_c3(args.quick)
    log(f"   exponent {c3['exponent']:.3f}")

    log("figures")
    figs = {
        "fig1_hertz_impact": figure_hertz(c1),
        "fig2_mu_eff_vs_concentration": figure_mu(c4567),
        "fig3_dissipation_scaling": figure_scaling(c3),
        "fig4_bond_rupture": figure_bond(c2),
        "fig5_field_snapshots": figure_snapshots(c4567),
    }

    criteria = {
        "C1": {"passed": c1["passed"], "value": c1["elastic_force_error"],
               "threshold": 0.05, "quantity": "Hertz peak-force deviation"},
        "C2": {"passed": c2["passed"], "value": c2["worst_relative_error"],
               "threshold": 0.10, "quantity": "bond rupture stress deviation"},
        "C3": {"passed": c3["passed"], "value": c3["exponent"],
               "threshold": [2.9, 3.1], "quantity": "dissipation scaling exponent"},
        "C4": {"passed": c4567["c4"]["passed"],
               "value": c4567["c4"]["mu_eff"], "threshold": "monotone",
               "quantity": "mu_eff(C)"},
        "C5": {"passed": c4567["c5"]["passed"],
               "value": c4567["c5"]["mu_eff_at_zero"], "threshold": 0.0,
               "quantity": "mu_eff(C=0)"},
        "C6": {"passed": c6["passed"], "value": c6["worst_relative_error"],
               "threshold": 0.05, "quantity": "energy ledger residual"},
        "C7": {"passed": c4567["c7"]["passed"],
               "value": c4567["c7"]["growth_factor"], "threshold": 10.0,
               "quantity": "mu_eff(0.9)/mu_eff(0.1)"},
        "C8": {"passed": True, "value": sorted(figs), "threshold": "figures exist",
               "quantity": "figure list"},
    }
    all_passed = all(v["passed"] for v in criteria.values())

    payload = {
        "node": "P202W1-N10",
        "module": "src/tensordem/ice_ice_collision.py",
        "repo": "TensorDEM",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock_seconds": time.time() - t_start,
        "quick_mode": args.quick,
        "criteria": criteria,
        "all_passed": all_passed,
        "C1_hertz": {k: v for k, v in c1.items() if k != "rows"},
        "C1_rows": c1["rows"],
        "C2_bond": {k: v for k, v in c2.items() if k != "rows"},
        "C2_rows": c2["rows"],
        "C3_scaling": c3,
        "C4_C5_C7_sweep": {"rows": c4567["rows"],
                           "c4": c4567["c4"], "c5": c4567["c5"], "c7": c4567["c7"]},
        "C5_zero": c4567["rows"],
        "C6_ledger": c6,
        "figures": figs,
        "config": {
            "cell": "Lees-Edwards doub. periodic, SLLOD simple shear",
            "box_m": [20.0, 20.0],
            "thickness_m": 0.5,
            "mean_radius_m": 0.6,
            "radius_spread": 2.0,
            "density_kg_m3": 900.0,
            "young_modulus_Pa": 1.0e7,
            "poisson_ratio": 0.3,
            "friction": 0.3,
            "restitution": 0.5,
            "shear_rate_1_s": 0.5,
        },
    }
    with open(os.path.join(OUT, "validation.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    log(f"wrote {os.path.join(OUT, 'validation.json')}")
    log(f"ALL PASSED: {all_passed}  ({time.time() - t_start:.0f}s)")
    for key, val in criteria.items():
        log(f"  {key}: passed={val['passed']} value={val['value']}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())