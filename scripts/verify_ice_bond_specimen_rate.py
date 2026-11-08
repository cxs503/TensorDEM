"""Specimen-level strain-rate campaign for the rate-dependent DEM ice bond.

Purpose
-------
``tensordem.ice_bond_viscoelastic`` verified the Maxwell-Wiechert bond law and
the Kachanov damage clock at the *constitutive* (single-bond) level.  The gap
ledger entry left open was the *engine* call site and a bonded-specimen
benchmark: does the emergent uniaxial compressive strength of a bonded
particle specimen follow the same rate law once the law is wired into the
solver, with fracture, stress redistribution and inertia all present?

Four arms separate the two physical mechanisms:

A ``linear``                 legacy elastic bond, brittle strain criterion only
                             -> negative control: strength must be rate
                                independent.
B ``linear+creep``           elastic bond + Kachanov clock (m = 2)
                             -> closed-form anchor sigma_f ~ edot**(1/(m+1)) = 1/3.
C ``maxwell_wiechert``       rate-dependent bond, no creep damage
                             -> rate *stiffening* only (relaxation spectrum).
D ``maxwell_wiechert+creep`` both mechanisms
                             -> the physical model; compared, not fitted.

Every number in the JSON is produced by the run; the analytic reference values
are computed in the report step from the same JSON, so nothing is transcribed.

Usage
-----
    PYTHONPATH=src python scripts/verify_ice_bond_specimen_rate.py
    PYTHONPATH=src python scripts/verify_ice_bond_specimen_rate.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordem.dem3d_triaxial import TriaxialConfig, TriaxialRunner

# ---------------------------------------------------------------------------
# Specimen / material constants (single source of truth for the campaign)
# ---------------------------------------------------------------------------
NX, NY, NZ = 4, 4, 3
RADIUS = 0.025
BOND_STIFFNESS = 2_000.0          # N/m, unrelaxed (instantaneous) stiffness
BREAKING_STRAIN = 0.015
CREEP_EXPONENT = 2                # m -> ductile envelope exponent 1/(m+1) = 1/3
CREEP_TIME_S = 5.0e-3             # t_ref
RELAXATION_TIME_S = (5.0e-3, 5.0e-2)
EQUILIBRIUM_FRACTION = 0.35
DELAYED_FRACTION = (0.45, 0.20)
STRAIN_RATES = (4.0, 1.0, 0.25)   # s^-1, axial engineering strain rate

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_bond_specimen"


def analytic_exponent() -> float:
    """Kachanov ductile envelope exponent 1/(m+1)."""
    return 1.0 / (CREEP_EXPONENT + 1.0)


def analytic_transition_rate() -> float:
    """Rate at which creep damage and brittle cleavage fail at the same strain.

    With F ~ t (linear ramp) the damage reaches 1 at ``t_f = (m+1) t_ref`` when
    ``F_ref`` equals the brittle-cleavage force, so the crossover strain rate is
    ``eps_b / ((m+1) t_ref)``.  This is a prediction of the *time* comparison,
    not a tuned stress.
    """
    return BREAKING_STRAIN / ((CREEP_EXPONENT + 1.0) * CREEP_TIME_S)


def analytic_brittle_strength() -> float:
    """Rigid-plate upper bound: sigma = k_u * eps_b * L0 * (bonds per area).

    A simple-cubic lattice cut by the mid-plane has ``nx * ny`` bonds, each
    carrying ``k_u * eps_b * L0`` at the cleavage strain, spread over the gross
    area ``(nx * 2r)(ny * 2r)``.
    """
    l0 = 2.0 * RADIUS
    force = BOND_STIFFNESS * BREAKING_STRAIN * l0 * (NX * NY)
    return force / ((NX * 2.0 * RADIUS) * (NY * 2.0 * RADIUS))


def arm_configs(arm: str, rate: float, *, dt_hint: float) -> TriaxialConfig:
    viscoelastic = arm.startswith("maxwell")
    creep = arm.endswith("creep")
    height = NZ * 2.0 * RADIUS
    speed = 0.5 * rate * height          # both platens oppose -> closure 2*v
    # Cap the run at twice the cleavage strain: if the specimen has not failed by
    # then, the peak is a plateau and the extra strain only costs wall clock.
    max_steps = int(math.ceil(2.0 * BREAKING_STRAIN / rate / dt_hint)) + 20
    return TriaxialConfig(
        nx=NX, ny=NY, nz=NZ, radius=RADIUS,
        bond_stiffness=BOND_STIFFNESS,
        breaking_strain=BREAKING_STRAIN,
        protocol="ucs_opposed",
        loading_speed=speed,
        max_axial_steps=max_steps,
        post_peak_fraction=0.6,
        drag=0.1, contact_damping=0.5, platen_damping=0.5, friction_coefficient=0.0,
        bond_model="maxwell_wiechert" if viscoelastic else "linear",
        bond_equilibrium_fraction=EQUILIBRIUM_FRACTION,
        bond_delayed_elastic_fraction=DELAYED_FRACTION,
        bond_relaxation_time_s=RELAXATION_TIME_S,
        bond_creep_exponent=CREEP_EXPONENT if creep else 0,
        bond_creep_time_s=CREEP_TIME_S,
    )


def run_one(arm: str, rate: float, *, dt_hint: float) -> dict:
    cfg = arm_configs(arm, rate, dt_hint=dt_hint)
    runner = TriaxialRunner(cfg)
    record = runner.run()
    sim = runner.sim
    axial = [r for r in record["rows"] if r["phase"] == "axial"] or record["rows"]
    peak = max(axial, key=lambda r: r["axial_stress_Pa"])
    kinetic = [r for r in axial]
    return {
        "arm": arm,
        "strain_rate_s-1": float(rate),
        "platen_speed_m_s": float(cfg.loading_speed),
        "sigma_peak_Pa": float(record["sigma1_Pa"]),
        "strain_at_peak": float(record["strain_at_peak"]),
        "n_steps": int(record["n_steps"]),
        "dt_s": float(sim.dt),
        "broken_bonds": int(record["final_broken_bonds"]),
        "final_kinetic_energy_J": float(record["final_kinetic_energy_J"]),
        "final_mechanical_energy_J": float(record["final_mechanical_energy_J"]),
        "bond_law": sim.bond_law_report(),
        "energy_ledger": {k: record["energy_ledger"][k] for k in
                          ("e_mech_t_J", "input_work_J", "dissipation_J",
                           "fracture_release_J", "closure_residual_relative")},
        "history": [
            {"strain": float(r["axial_engineering_strain"]),
             "stress_Pa": float(r["axial_stress_Pa"]),
             "broken_bonds": int(r["broken_bonds"])}
            for r in axial[:: max(1, len(axial) // 60)]
        ],
        "peak_row": {"strain": float(peak["axial_engineering_strain"]),
                     "stress_Pa": float(peak["axial_stress_Pa"])},
    }


def power_law_slope(rates: list[float], strengths: list[float]) -> float:
    """Least-squares slope of log10(sigma) against log10(rate)."""
    xs = [math.log10(r) for r in rates]
    ys = [math.log10(s) for s in strengths]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                       help="single rate, used as a smoke test in CI")
    args = parser.parse_args()

    rates = STRAIN_RATES[:1] if args.quick else STRAIN_RATES
    arms = ("linear", "linear+creep", "maxwell_wiechert", "maxwell_wiechert+creep")
    torch.set_num_threads(1)
    dt_hint = 3.4e-5

    runs: list[dict] = []
    for arm in arms:
        for rate in rates:
            row = run_one(arm, rate, dt_hint=dt_hint)
            runs.append(row)
            print(f"{arm:26s} edot={rate:6.3f}  sigma_f={row['sigma_peak_Pa']:9.2f} Pa  "
                  f"eps*={row['strain_at_peak']:.5f}  broken={row['broken_bonds']:3d}  "
                  f"steps={row['n_steps']:6d}")

    summary: dict[str, dict] = {}
    for arm in arms:
        rows = [r for r in runs if r["arm"] == arm]
        rows.sort(key=lambda r: r["strain_rate_s-1"])
        rr = [r["strain_rate_s-1"] for r in rows]
        ss = [r["sigma_peak_Pa"] for r in rows]
        summary[arm] = {
            "strain_rates_s-1": rr,
            "sigma_peak_Pa": ss,
            "fitted_exponent": power_law_slope(rr, ss),
            "strength_spread_relative": (max(ss) - min(ss)) / min(ss),
            "creep_failed_bonds_per_run": [r["bond_law"]["creep_failed_bonds"] for r in rows],
            "bond_dissipation_J_per_run": [r["bond_law"]["cumulative_bond_dissipation_J"] for r in rows],
            "max_bond_damage_per_run": [r["bond_law"]["max_bond_damage"] for r in rows],
            "max_ledger_residual_relative": max(
                abs(r["energy_ledger"]["closure_residual_relative"]) for r in rows),
        }

    payload = {
        "campaign": "ice_bond_specimen_rate_dependence",
        "specimen": {"nx": NX, "ny": NY, "nz": NZ, "radius_m": RADIUS,
                     "unrelaxed_bond_stiffness_N_m": BOND_STIFFNESS,
                     "breaking_strain": BREAKING_STRAIN},
        "bond_law": {"equilibrium_fraction": EQUILIBRIUM_FRACTION,
                     "delayed_elastic_fraction": list(DELAYED_FRACTION),
                     "relaxation_time_s": list(RELAXATION_TIME_S),
                     "creep_exponent": CREEP_EXPONENT,
                     "creep_time_s": CREEP_TIME_S},
        "analytic_references": {
            "ductile_exponent_1_over_m_plus_1": analytic_exponent(),
            "creep_brittle_transition_rate_s-1": analytic_transition_rate(),
            "rigid_plate_brittle_strength_Pa": analytic_brittle_strength(),
        },
        "runs": runs,
        "summary": summary,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("quick.json" if args.quick else "verification.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("\narm                          fitted slope   spread(rel)   max|ledger|")
    for arm, row in summary.items():
        print(f"{arm:26s} {row['fitted_exponent']:+.4f}      "
              f"{row['strength_spread_relative']:.3e}    "
              f"{row['max_ledger_residual_relative']:.2e}")
    print(f"\nanalytic: exponent={analytic_exponent():.6f}  "
          f"transition rate={analytic_transition_rate():.4f} 1/s  "
          f"brittle bound={analytic_brittle_strength():.1f} Pa")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())