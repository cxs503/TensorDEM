"""Round 17 (m21066) evidence generator: does a capped recovery phase bound the strain?

Runs the shipped servo-shear device twice on the *identical* configuration --

  * ``max_recover_strain=None``  -> round-17 default, historical behaviour
  * ``max_recover_strain=0.05``  -> the new strain ceiling

-- and saves the strain trajectory and the budget bookkeeping.  Everything the
report quotes is written here, programmatically, so no number is hand-copied.

Usage:  PYTHONPATH=src python benchmarks/plane_shear_mu_i/run_strain_ceiling_r17.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tensordem.mu_i_rheology import (
    ServoShearCase,
    simulate_servo_shear,
    strain_budget_status,
)

OUT = Path(__file__).resolve().parent / "runs" / "round17"
RADIUS, KN, PRESSURE, FILL, SEED = 0.005, 6.25e4, 62.5e3, 0.04, 5279

BASE = dict(radius=RADIUS, kn=KN, normal_stress=PRESSURE, fill_height=FILL,
            wall_mode="smooth", seed=SEED, inertial_number=0.3,
            shear_strain=0.5, sample_stride=10, settle_steps=300)

# (label, max_recover_strain, min_shear_steps)
ARMS = [
    ("historical", None, 200),                       # pre-round-17 behaviour
    ("ceiling_only", 0.05, 200),                     # cap alone: floor still bites
    ("ceiling_and_strain_only_budget", 0.025 * 0.5, 0),  # the round-17 driver
]


def one(label, cap, min_shear_steps):
    case = ServoShearCase(max_recover_strain=cap,
                          min_shear_steps=min_shear_steps, **BASE)
    res = simulate_servo_shear(case, return_series=True)
    series = res["series"]
    dt = res["dt"]
    trace = [{"step": i, "strain": float(s.get("strain", float("nan"))),
              "phi_bed": float(s.get("phi_bed_now", float("nan"))),
              "mu_full": float(s.get("full_mu", float("nan")))}
             for i, s in enumerate(series)]
    return {
        "label": label,
        "max_recover_strain": cap,
        "min_shear_steps": min_shear_steps,
        "shear_strain_requested": res["shear_strain_requested"],
        "shear_strain_achieved": res["shear_strain_achieved"],
        "strain_overrun_ratio": res["strain_overrun_ratio"],
        "strain_budget_respected": res["strain_budget_respected"],
        "strain_budget_status": res["strain_budget_status"],
        "recover_termination": res["recover_termination"],
        "shear_steps": res["shear_steps"],
        "shear_steps_executed": res["shear_steps_executed"],
        "recover_steps": res["recover_steps"],
        "servo_valid": res["servo_valid"],
        "pressure_control_error_final": res["pressure_control_error_final"],
        "dt": dt,
        "n_particles": res["n_particles"],
        "n_samples": res["n_samples"],
        "trace": trace[::max(1, len(trace) // 400)],
    }


def worst_case_table():
    """Re-derive the step-budget arithmetic for the six nominal I (round 16)."""
    rows = []
    strain = 2.0
    for I in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        probe = ServoShearCase(radius=RADIUS, kn=KN, normal_stress=PRESSURE,
                               fill_height=FILL, inertial_number=I,
                               wall_mode="smooth", seed=SEED)
        dt, d = probe.timestep, probe.particle_diameter
        gamma_dot = I * math.sqrt(PRESSURE / probe.density) / d
        driver_floor = int(math.ceil(strain / (gamma_dot * dt))) + 100
        shear_steps = int(max(driver_floor, math.ceil(strain / (gamma_dot * dt))))
        worst = (shear_steps + probe.max_recover_steps) * gamma_dot * dt
        rows.append({
            "I_target": I, "gamma_dot": gamma_dot,
            "min_shear_steps_floor": driver_floor,
            "shear_steps": shear_steps,
            "worst_case_strain": worst,
            "worst_case_overrun": worst / strain,
            "budget": strain_budget_status(worst, strain),
        })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"config": {k: v for k, v in BASE.items()},
               "worst_case_table": worst_case_table(),
               "runs": [one(*a) for a in ARMS]}
    (OUT / "strain_ceiling_r17.json").write_text(json.dumps(payload, indent=2))
    for r in payload["runs"]:
        print(f"{r['label']:<32} cap={r['max_recover_strain']!s:<6} "
              f"requested={r['shear_strain_requested']:.3f} "
              f"achieved={r['shear_strain_achieved']:.4f} "
              f"overrun={r['strain_overrun_ratio']:6.2f}x "
              f"status={r['strain_budget_status']:<9} "
              f"term={r['recover_termination']:<20} "
              f"servo_valid={r['servo_valid']}")
    print(f"wrote {OUT / 'strain_ceiling_r17.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())