"""Generate the exact axisymmetric Young-Laplace reference table used by the
wet-ice capillary-bridge capability (P35-N2).

The reference is the *existing, independently validated* solver
``tensordem.liquid_bridge.solve_axisymmetric_bridge`` (its own invariants are
checked in ``tests/test_liquid_bridge.py``: the axial force must be a first
integral, i.e. the neck and contact-line evaluations agree to ~1 %).

Two disjoint sets are written:

``train``
    points used to calibrate the DEM-grade closed form
    (``tensordem.wet_ice_cohesion.capillary``).
``validate``
    held-out points that the calibration never sees; the reported errors of the
    closed form are measured on this set only.

Every row stores the raw solver output, so the table is auditable without
re-running scipy: ``force_over_f0``, ``force_contact_line_over_f0`` (first-integral
check), ``neck_radius_over_R``, ``pressure_difference``, ``residual`` and the
solver's own ``converged`` flag (reported honestly, not filtered).

Usage::

    PYTHONPATH=src python scripts/gen_wet_ice_capillary_reference.py \
        --out artifacts/wet_ice_cohesion/capillary_reference.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

from tensordem.liquid_bridge import solve_axisymmetric_bridge

R = 5.0e-4   # m, particle radius of the reference case
GAMMA = 0.0759  # N/m, water at 0 C (IAPWS)
F0 = 2.0 * math.pi * GAMMA * R


def _row(volume_over_r3: float, gap_over_r: float, theta_deg: float, n_steps: int) -> dict:
    volume = volume_over_r3 * R ** 3
    gap = gap_over_r * R
    try:
        sol = solve_axisymmetric_bridge(
            R, GAMMA, volume, gap, theta_deg, n_steps=n_steps
        )
    except Exception as exc:  # pragma: no cover - solver brittleness is data
        return {
            "volume_over_R3": volume_over_r3,
            "gap_over_R": gap_over_r,
            "theta_deg": theta_deg,
            "converged": False,
            "error": f"{type(exc).__name__}: {exc}"[:120],
        }
    fcl = sol.extra.get("force_at_contact_line")
    return {
        "volume_over_R3": volume_over_r3,
        "gap_over_R": gap_over_r,
        "theta_deg": theta_deg,
        "converged": bool(sol.converged),
        "residual": float(sol.residual),
        "force": float(sol.force),
        "force_over_f0": float(sol.force / F0),
        "force_contact_line_over_f0": None if fcl is None else float(fcl / F0),
        "neck_radius_over_R": float(sol.neck_radius / R),
        "pressure_difference": float(sol.pressure_difference),
        "half_filling_angle_deg": float(sol.half_filling_angle_deg),
    }


def build(n_steps: int = 200) -> dict:
    t0 = time.time()
    train = []
    # (a) peak branch: s = 0, theta = 0, log-spaced volumes
    for v in (0.008, 0.015, 0.03, 0.07, 0.12):
        train.append(_row(v, 0.0, 0.0, n_steps))
    # (b) contact-angle branch
    for v, th in ((0.03, 30.0), (0.03, 45.0)):
        train.append(_row(v, 0.0, th, n_steps))
    # (c) gap branch
    for v in (0.02, 0.05, 0.1):
        for s in (0.02, 0.05, 0.1):
            train.append(_row(v, s, 0.0, n_steps))

    validate = []
    for v in (0.0025, 0.006, 0.012, 0.025, 0.04, 0.06, 0.09, 0.14, 0.18):
        validate.append(_row(v, 0.0, 0.0, n_steps))
    for v, s in ((0.012, 0.03), (0.025, 0.01), (0.04, 0.08), (0.06, 0.04),
                 (0.09, 0.15), (0.14, 0.05), (0.025, 0.12)):
        validate.append(_row(v, s, 0.0, n_steps))
    for v, th in ((0.05, 20.0), (0.09, 50.0), (0.03, 60.0)):
        validate.append(_row(v, 0.0, th, n_steps))

    meta = {
        "generator": "scripts/gen_wet_ice_capillary_reference.py",
        "solver": "tensordem.liquid_bridge.solve_axisymmetric_bridge",
        "particle_radius_m": R,
        "surface_tension_N_per_m": GAMMA,
        "f0_definition": "2*pi*gamma*R",
        "f0_N": F0,
        "n_steps": n_steps,
        "wall_clock_s": round(time.time() - t0, 2),
        "note": (
            "converged=False rows are kept: the shooting solver is brittle for "
            "very small bridges and near theta=90 deg. Rows with residual > 1e-2 "
            "are excluded by the analysis, rows with residual < 1e-3 are used."
        ),
    }
    return {"meta": meta, "train": train, "validate": validate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/wet_ice_cohesion/capillary_reference.json")
    ap.add_argument("--n-steps", type=int, default=200)
    args = ap.parse_args()
    data = build(args.n_steps)
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {args.out}  train={len(data['train'])} validate={len(data['validate'])} "
          f"({data['meta']['wall_clock_s']} s)")


if __name__ == "__main__":
    main()