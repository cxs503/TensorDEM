"""Diagnostics for the H1 C2 (rubble angle of repose) case.

Answers three questions with numbers instead of prose:

1. **Cost per step** -- how long does one integration step cost for ``n`` bodies,
   so the settling budget (``n_steps``) can be sized against real wall time.
2. **Where the time goes** -- kinetic / potential / spring energy of the ledger
   sampled through the run, so "not settled" can be attributed either to a slow
   decay (physics) or to a numerical floor (pe_spring spikes).
3. **What the free-surface fit sees** -- per-annulus heights and radii used by
   :func:`tensordem.polyhedron_floe3d.study._pile_metrics`, which is what turns a
   pile into an angle.

    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/diagnose_repose.py \
        --bodies 27 --steps 600 --every 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.polyhedron_floe3d.contact import ContactLaw  # noqa: E402
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody  # noqa: E402
from tensordem.polyhedron_floe3d.geometry import voronoi_tessellate  # noqa: E402
from tensordem.polyhedron_floe3d.study import (  # noqa: E402
    _pile_metrics,
    _random_rotation,
)


def build(n_bodies: int, cell_half: float, mu: float, kn: float, zeta: float,
          dt: float, seed: int, n_xy: int):
    rng = np.random.default_rng(seed)
    sites = rng.uniform(-cell_half, cell_half, (n_bodies + 40, 3))
    cells = [c for c in voronoi_tessellate(sites, cell_half) if c.volume > 1e-7][:n_bodies]
    spacing = 2.05 * max(cell.bounding_radius for cell in cells)
    bodies = []
    for k, cell in enumerate(cells):
        ix, iy = k % n_xy, (k // n_xy) % n_xy
        iz = k // (n_xy * n_xy)
        bodies.append(FloeBody(
            cell, 917.0,
            position=[(ix - 0.5 * (n_xy - 1)) * spacing,
                      (iy - 0.5 * (n_xy - 1)) * spacing,
                      0.02 + spacing * (iz + 0.5)],
            rotation=_random_rotation(rng), label="r%d" % k))
    system = DEMSystem3D(bodies,
                         law=ContactLaw(kn=kn, zeta=zeta, mu=mu,
                                        mu_rolling=0.02, kt_ratio=0.25),
                         gravity=9.81, dt=dt)
    system.add_box([0.0, 0.0, 0.75], [0.5, 0.5, 0.75], bottom=True, sides=True, top=False)
    system.record_initial_energy()
    return system, cells, spacing


def annulus_profile(system):
    """The exact view ``_pile_metrics`` gets: (r_mid, top_height) per annulus."""
    positions = np.array([b.position for b in system.bodies])
    centre = positions[:, :2].mean(axis=0)
    radius = np.linalg.norm(positions[:, :2] - centre, axis=1)
    top = positions[:, 2] + np.array([b.shape.bounding_radius
                                      for b in system.bodies]) * 0.5
    bins = np.linspace(0.0, max(float(radius.max()), 1e-6) * 0.95, 7)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (radius >= lo) & (radius < hi)
        if np.count_nonzero(mask) >= 3:
            rows.append({"r_mid": 0.5 * (lo + hi),
                         "top_p90": float(np.percentile(top[mask], 90)),
                         "n_in_bin": int(np.count_nonzero(mask))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=27)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--mu", type=float, default=0.4)
    ap.add_argument("--kn", type=float, default=2.0e5)
    ap.add_argument("--zeta", type=float, default=0.2)
    ap.add_argument("--dt", type=float, default=1.0e-4)
    ap.add_argument("--cell-half", type=float, default=0.13)
    ap.add_argument("--nxy", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system, cells, spacing = build(args.bodies, args.cell_half, args.mu, args.kn,
                                  args.zeta, args.dt, args.seed, args.nxy)
    radius_b = max(c.bounding_radius for c in cells)
    # stability numbers: contact period vs the step, and the drop height
    m_min = min(b.mass for b in system.bodies)
    omega_contact = math.sqrt(args.kn / m_min)
    out = {
        "setup": {"n_bodies": len(system.bodies), "spacing_m": spacing,
                  "max_bounding_radius_m": radius_b, "min_mass_kg": m_min,
                  "contact_omega_rad_s": omega_contact,
                  "steps_per_contact_period": 2.0 * math.pi / (omega_contact * args.dt),
                  "drop_height_m": max(b.position[2] for b in system.bodies),
                  "mu": args.mu, "mu_rolling": 0.02, "kt_ratio": 0.25},
        "samples": [],
    }
    t0 = time.time()
    for step in range(1, args.steps + 1):
        system.step()
        if step % args.every and step != args.steps:
            continue
        system.refresh_energy()
        motion = max(float(np.linalg.norm(b.velocity))
                     + float(np.linalg.norm(b.omega)) * b.shape.bounding_radius
                     for b in system.bodies)
        metrics = _pile_metrics(system)
        positions = np.array([b.position for b in system.bodies])
        out["samples"].append({
            "step": step, "t_s": float(system.time),
            "ke_trans": float(system.ke_trans), "ke_rot": float(system.ke_rot),
            "pe_gravity": float(system.pe_gravity),
            "pe_spring": float(system.pe_spring),
            "dissipated": float(system.dissipated),
            "energy_residual": float(system.energy_residual()),
            "max_motion": motion,
            "n_contacts": metrics["n_contacts"],
            "repose_angle_deg": metrics["repose_angle_deg"],
            "spread_radius_m": float(np.linalg.norm(
                positions[:, :2] - positions[:, :2].mean(axis=0), axis=1).max()),
            "profile": annulus_profile(system),
            "wall_clock_s": time.time() - t0,
        })
        print(json.dumps({k: v for k, v in out["samples"][-1].items()
                          if k != "profile"}, indent=None), flush=True)
    out["wall_clock_s"] = time.time() - t0
    out["s_per_step"] = out["wall_clock_s"] / max(args.steps, 1)
    out["profile_final"] = out["samples"][-1]["profile"] if out["samples"] else []
    print(json.dumps({"s_per_step": out["s_per_step"],
                      "projected_30000_step_minutes": out["s_per_step"] * 30000 / 60.0}))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())