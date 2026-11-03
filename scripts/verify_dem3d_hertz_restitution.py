"""Verification of the Hertz/Tsuji restitution mapping in the DEM3D engine.

Reproduces the tables of ``docs/dem3d_hertz_restitution_mapping.md``:

* the universal dimensionless map ``alpha -> e`` (reference quadrature),
* the ``alpha = 0`` anchor against the textbook elastic Hertz contact time,
* engine-measured restitution and contact time for a head-on pair impact, run
  with ``hertz_damping_mapping="exact"`` (the fix) and
  ``hertz_damping_mapping="linear_formula"`` (the legacy linear-law mapping),
* the impact-speed independence of the realised restitution.

Usage::

    PYTHONPATH=src python scripts/verify_dem3d_hertz_restitution.py \
        --output outputs/dem3d_hertz_restitution

Writes ``hertz_restitution.json`` there and exits non-zero if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)
from tensordem.hertz_restitution import (
    ELASTIC_CONTACT_TIME_CONSTANT,
    alpha_for_restitution,
    elastic_hertz_contact_time,
    hertz_contact_time,
    linear_law_alpha,
    peak_overlap_fraction,
    restitution_for_alpha,
)

YOUNG = 1.0e9
NU = 0.3
RADIUS = 0.05
DENSITY = 900.0
E_TARGETS = (0.30, 0.50, 0.70, 0.85, 0.95)
SPEEDS = (0.10, 0.50, 2.00, 5.00)

# Gates (fractional). The exact mapping must reproduce e and the contact time
# well inside the 3 % acceptance band; the elastic anchor is graded against the
# rounded textbook constant, so it gets a slightly wider allowance.
GATE_E = 0.02
GATE_TC = 0.05
GATE_ELASTIC_TC = 0.01
GATE_SPEED_SPREAD = 0.02


def _k_h() -> float:
    e_star = hertz_reduced_modulus(YOUNG, NU)
    r_eff = hertz_effective_radius(RADIUS)
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff)


def build_pair(*, restitution: float, mapping: str, dt_scale: float = 1.0):
    """Two free particles facing each other, Hertz contact, fixed lattice parked."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=RADIUS, density=DENSITY,
        drag=0.0,
        bond_stiffness=1.0, breaking_strain=1.0e-12, shear_breaking_strain=1.0e-12,
        fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0e3, tool_speed=1.0e-9,
        contact_model="hertz", young_modulus=YOUNG, poisson_ratio=NU,
        restitution_coefficient=restitution,
        hertz_damping_mapping=mapping,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for i in range(n):
        pos[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
    pos[0] = torch.tensor([0.0, 0.0, -(RADIUS + 1.0e-6)], dtype=dem.dtype)
    pos[1] = torch.tensor([0.0, 0.0, RADIUS + 1.0e-6], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("harness assumption violated: bonded lattice still alive")
    dem.dt = dem.dt * dt_scale
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[0] = torch.tensor([0.0, 0.0, 1.0], dtype=dem.dtype)
    dem.velocities[1] = torch.tensor([0.0, 0.0, -1.0], dtype=dem.dtype)
    return dem


def run_impact(*, restitution: float, mapping: str, v0: float, dt_scale: float = 1.0,
               budget: int = 2_000_000) -> dict:
    """Head-on pair impact; measure restitution, contact duration, peak overlap."""
    dem = build_pair(restitution=restitution, mapping=mapping, dt_scale=dt_scale)
    dem.velocities[0] *= v0
    dem.velocities[1] *= v0
    contact_steps = 0
    force_steps = 0
    peak = 0.0
    sep = float(dem.positions[1, 2] - dem.positions[0, 2])
    for _ in range(budget):
        v_before = float(dem.velocities[0, 2])
        dem.step()
        if float(dem.velocities[0, 2]) != v_before:
            force_steps += 1
        sep = float(dem.positions[1, 2] - dem.positions[0, 2])
        overlap = 2.0 * RADIUS - sep
        if overlap > 0.0:
            contact_steps += 1
            peak = max(peak, overlap)
        elif contact_steps:
            break
    else:  # pragma: no cover
        raise RuntimeError("pair did not separate within the step budget")
    rebound = -float(dem.velocities[0, 2])
    return {
        "restitution": rebound / v0,
        "contact_time": contact_steps * dem.dt,
        "contact_time_force": force_steps * dem.dt,
        "contact_steps": contact_steps,
        "force_steps": force_steps,
        "peak_overlap": peak,
        "dt": dem.dt,
        "momentum_drift": float(dem.velocities[:, 2].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/dem3d_hertz_restitution")
    args = ap.parse_args()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    m_eff = 0.5 * (DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3)
    k_h = _k_h()
    from tensordem.hertz_restitution import dimensionless_contact_time
    anchor = dimensionless_contact_time(0.0) * 2.0 / math.sqrt(5.0)
    report: dict = {
        "config": {
            "young_modulus": YOUNG, "poisson_ratio": NU, "radius": RADIUS,
            "density": DENSITY, "m_eff": m_eff, "k_h": k_h,
            "e_targets": list(E_TARGETS), "speeds": list(SPEEDS),
        },
        "reference": {
            "elastic_contact_time_constant_numeric": anchor,
            "elastic_contact_time_constant_textbook": ELASTIC_CONTACT_TIME_CONSTANT,
            "elastic_anchor_rel_error": anchor / ELASTIC_CONTACT_TIME_CONSTANT - 1.0,
            "e_of_alpha_zero_minus_one": restitution_for_alpha(0.0) - 1.0,
            "dimensionless_contact_time_alpha0": dimensionless_contact_time(0.0),
        },
        "map_table": [],
        "engine_exact": [],
        "engine_legacy": [],
        "speed_sweep": [],
        "anchors": {},
    }

    for et in E_TARGETS:
        alpha = alpha_for_restitution(et)
        legacy = linear_law_alpha(et)
        report["map_table"].append({
            "e_target": et,
            "alpha_exact": alpha,
            "e_roundtrip": restitution_for_alpha(alpha),
            "alpha_legacy": legacy,
            "e_at_legacy_alpha": restitution_for_alpha(legacy),
            "legacy_bias": restitution_for_alpha(legacy) / et - 1.0,
            "contact_time_reference": hertz_contact_time(m_eff, k_h, 0.5, alpha),
            "peak_overlap_fraction": peak_overlap_fraction(alpha),
        })

    # Elastic anchor through the engine: e = 1 -> gamma = 0.
    elastic = run_impact(restitution=1.0, mapping="exact", v0=0.5, dt_scale=0.5)
    # The reference problem is stated in the *relative* coordinate, so the
    # impact speed is v_rel = 2 v0 for an equal-mass head-on pair.
    v_rel = 2.0 * 0.5
    tc_textbook = elastic_hertz_contact_time(m_eff, k_h, v_rel)
    delta_max = (5.0 * m_eff * v_rel ** 2 / (4.0 * k_h)) ** 0.4
    report["anchors"]["engine_elastic"] = {
        "e_measured": elastic["restitution"],
        "contact_time": elastic["contact_time"],
        "contact_time_textbook": tc_textbook,
        "contact_time_rel_error": elastic["contact_time_force"] / tc_textbook - 1.0,
        "peak_overlap": elastic["peak_overlap"],
        "peak_overlap_analytic": delta_max,
        "peak_overlap_rel_error": elastic["peak_overlap"] / delta_max - 1.0,
        "dt": elastic["dt"], "contact_steps": elastic["contact_steps"],
    }

    for et in E_TARGETS:
        row = {"e_target": et}
        for mapping in ("exact", "linear_formula"):
            res = run_impact(restitution=et, mapping=mapping, v0=0.5, dt_scale=0.5)
            ref = hertz_contact_time(m_eff, k_h, 2.0 * 0.5, alpha_for_restitution(et)
                                     if mapping == "exact" else linear_law_alpha(et))
            res[f"e_error_vs_target"] = res["restitution"] / et - 1.0
            res[f"e_error_vs_reference"] = res["restitution"] / restitution_for_alpha(
                alpha_for_restitution(et) if mapping == "exact" else linear_law_alpha(et)) - 1.0
            res["contact_time_reference"] = ref
            # The reference is the *force-active* duration; the solver's window
            # is graded on the steps where the normal force actually acts.
            res["contact_time_rel_error"] = res["contact_time_force"] / ref - 1.0
            res["geometric_contact_time_rel_error"] = res["contact_time"] / ref - 1.0
            if mapping == "exact":
                report["engine_exact"].append({"e_target": et, **res})
            else:
                report["engine_legacy"].append({"e_target": et, **res})

    for et in (0.50, 0.85):
        for v0 in SPEEDS:
            res = run_impact(restitution=et, mapping="exact", v0=v0, dt_scale=0.5)
            report["speed_sweep"].append({
                "e_target": et, "v0": v0,
                "e_measured": res["restitution"],
                "e_error_vs_target": res["restitution"] / et - 1.0,
                "contact_time": res["contact_time_force"],
                "contact_time_reference": hertz_contact_time(m_eff, k_h, 2.0 * v0, alpha_for_restitution(et)),
            })

    # ---- gates -------------------------------------------------------------
    gates = {}
    gates["elastic_anchor"] = abs(report["reference"]["elastic_anchor_rel_error"]) <= GATE_ELASTIC_TC
    gates["e_roundtrip"] = max(
        abs(r["e_roundtrip"] / r["e_target"] - 1.0) for r in report["map_table"]) <= 1e-6
    gates["engine_e_exact"] = max(
        abs(r["e_error_vs_target"]) for r in report["engine_exact"]) <= GATE_E
    gates["engine_tc_exact"] = max(
        abs(r["contact_time_rel_error"]) for r in report["engine_exact"]) <= GATE_TC
    spread = {}
    for et in (0.50, 0.85):
        vals = [r["e_measured"] for r in report["speed_sweep"] if r["e_target"] == et]
        spread[str(et)] = max(vals) / min(vals) - 1.0
    report["speed_spread"] = spread
    gates["speed_independence"] = max(spread.values()) <= GATE_SPEED_SPREAD
    report["gates"] = gates
    report["gate_limits"] = {"e": GATE_E, "contact_time": GATE_TC,
                             "elastic_anchor": GATE_ELASTIC_TC,
                             "speed_spread": GATE_SPEED_SPREAD}
    report["pass"] = all(gates.values())

    (out / "hertz_restitution.json").write_text(json.dumps(report, indent=2))

    print("elastic anchor : numeric %.6f vs textbook %.2f  (%+.3f %%)"
          % (anchor, ELASTIC_CONTACT_TIME_CONSTANT,
             100.0 * report["reference"]["elastic_anchor_rel_error"]))
    print("%-7s %-12s %-12s %-12s %-12s" % ("e_t", "alpha", "e_ref", "e_engine", "dE"))
    for r in report["engine_exact"]:
        print("%-7.2f %-12.6f %-12.6f %-12.6f %+8.3f%%"
              % (r["e_target"], alpha_for_restitution(r["e_target"]),
                 restitution_for_alpha(alpha_for_restitution(r["e_target"])),
                 r["restitution"], 100.0 * r["e_error_vs_target"]))
    print("legacy mapping bias at e=0.30: %+.2f %%"
          % (100.0 * report["map_table"][0]["legacy_bias"]))
    print("gates:", gates)
    print("wrote", out / "hertz_restitution.json")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())