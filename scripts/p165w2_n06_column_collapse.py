"""P165-W2-N06 (f): column-collapse repose-angle convergence in the DEM engine.

A loose cubic packing of monodisperse spheres is held by four vertical gate walls
on a frictional floor.  The gates are switched off and the column collapses; the
free-surface angle of the resulting heap is measured from the particle cloud
(Delaunay-free radial-bin envelope, :func:`tensordem.superquadric_dem.repose_angle_from_cloud`).

Three axes are scanned (one run per invocation so the work can be sharded):

* ``--n``      particle number  (particle-number convergence)
* ``--dt-scale`` time-step scale (time-step convergence)
* ``--mu``     inter-particle friction (material scan)

The run stops when the measured angle plateaus (``--plateau-deg`` over
``--plateau-blocks`` blocks) or at ``--max-steps``.

Usage (single shard)::

    PYTHONPATH=src python scripts/p165w2_n06_column_collapse.py \
        --n 2000 --mu 0.45 --out runs/p165w2_n06/column_n2000_mu0.45.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.flat_punch import reduced_modulus                        # noqa: E402
from tensordem.superquadric_dem import (                                # noqa: E402
    SQSystem,
    SQSystemConfig,
    Wall,
    energy_drift_per_1000,
    make_superquadric_body,
    repose_angle_from_cloud,
)
from tensordem.superquadric_kernel import ContactMaterial               # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_case(n_target: int, radius: float, aspect: float = 1.0):
    """Loose cubic packing of ~``n_target`` spheres in a square footprint.

    The initial column keeps a fixed aspect ratio ``layers / per_side = aspect``
    so that growing N means *growing the system at fixed particle size* (the
    same material, more grains), not refining the resolution of one fixed
    geometry.  Returns ``(positions, half_width, height, n_actual)``.
    """
    spacing = 2.2 * radius
    per_side = max(int(round((n_target / max(aspect, 1e-6)) ** (1.0 / 3.0))), 2)
    layers = max(int(round(aspect * per_side)), 2)
    n = per_side ** 2 * layers
    half_width = 0.5 * per_side * spacing
    height = layers * spacing
    offset = -0.5 * (per_side - 1) * spacing
    pos = []
    for kz in range(layers):
        for ky in range(per_side):
            for kx in range(per_side):
                pos.append([offset + kx * spacing, offset + ky * spacing,
                            radius * 1.02 + kz * spacing])
    return np.array(pos[:n], dtype=float), half_width, height, n


def run_case(n_target: int, radius: float = 0.003, mu: float = 0.45,
             mu_rolling: float = 0.10, e_star: float = 1.0e7,
             gravity: float = 9.81, aspect: float = 0.5,
             dt_scale: float = 1.0, hold_steps: int = 400,
             max_steps: int = 20000, block: int = 500,
             plateau_deg: float = 0.15, plateau_blocks: int = 3,
             seed: int = 0, label: str = "") -> dict:
    rho = 2500.0
    pos, half_width, height, n = build_case(n_target, radius, aspect=aspect)
    rng = np.random.default_rng(seed)
    pos[:, :2] += 1e-4 * radius * rng.standard_normal((len(pos), 2))

    bodies = [make_superquadric_body(radius, radius, radius, 2.0, 2.0, p, rho)
              for p in pos]
    walls = [Wall(np.array([0.0, 0.0, 1.0]), 0.0, mu=mu, label="floor")]
    # Wall convention: the half-space kept free of particles is ``n . x >= offset``
    # (see ``Wall.signed_distance``), i.e. the normal points *into* the material.
    # A box of half-width hw therefore uses offset = -hw for all four side walls.
    for nrm in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])):
        walls.append(Wall(nrm, -half_width, mu=mu, label="gate"))
    mat = ContactMaterial(e_star=reduced_modulus(e_star, 0.3, e_star, 0.3),
                          g_star=e_star / 5.2, mu=mu, mu_rolling=mu_rolling,
                          restitution=0.4)
    cfg = SQSystemConfig(material=mat,
                         gravity=np.array([0.0, 0.0, -float(gravity)]))
    sysm = SQSystem(bodies, walls, cfg)
    sysm.dt = sysm.dt * float(dt_scale)
    sysm.collect_contacts = False
    sysm.config.energy_every = block

    t0 = time.perf_counter()
    sysm.run(hold_steps)                     # settle inside the gates
    hold_pen = sysm.max_penetration_ratio
    for w in walls[1:]:
        w.active = False                     # release
    e0 = sysm.energy()
    angles, steps = [], 0
    plateau = False
    t_release = sysm.time
    # a collapse needs at least the free-fall time of the column before the heap
    # angle means anything; without this floor the plateau test fires on the still
    # untouched cubic packing (measured: theta = 0 at t = 0.2 t_fall).
    t_fall = math.sqrt(2.0 * height / float(gravity))
    # 2.2 free-fall times is where the measured angle stops moving in the
    # N = 512 probe (see docs/P165W2_N06_dem_superquadric_maturity.md)
    min_steps = int(2.2 * t_fall / sysm.dt)
    while steps < max_steps:
        sysm.run(block)
        steps += block
        ang, profile = repose_angle_from_cloud(sysm.positions())
        angles.append(ang)
        if steps >= min_steps and len(angles) > plateau_blocks:
            recent = angles[-(plateau_blocks + 1):]
            if max(recent) - min(recent) < plateau_deg:
                plateau = True
                break
    pos_final = sysm.positions()
    angle, profile = repose_angle_from_cloud(pos_final)
    drift = energy_drift_per_1000([e0, sysm.energy()], steps)
    out = {
        "label": label or f"n{n_target}_mu{mu}_dt{dt_scale}",
        "n_target": n_target, "n_actual": n, "radius": radius,
        "mu": mu, "mu_rolling": mu_rolling, "e_star": e_star,
        "gravity": float(gravity), "aspect_initial": aspect,
        "density": rho, "dt": sysm.dt, "dt_scale": dt_scale,
        "half_width": half_width, "fill_height": height,
        "aspect": height / (2.0 * half_width),
        "hold_steps": hold_steps, "settle_steps": steps,
        "min_settle_steps": min_steps, "t_fall_s": t_fall,
        "total_steps": hold_steps + steps,
        "repose_angle_deg": angle,
        "repose_angle_history": angles,
        "plateau_reached": plateau,
        "surface_profile": profile,
        "hold_penetration_ratio": hold_pen,
        "max_penetration_ratio": sysm.max_penetration_ratio,
        "energy_drift_per_1000": drift,
        "physical_time_release": steps * sysm.dt,
        "wall_clock_s": time.perf_counter() - t0,
        "final_positions_shape": list(pos_final.shape),
        "_final_positions": pos_final,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--radius", type=float, default=0.003)
    ap.add_argument("--mu", type=float, default=0.45)
    ap.add_argument("--mu-rolling", type=float, default=0.10)
    ap.add_argument("--e-star", type=float, default=1.0e7)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--aspect", type=float, default=0.5)
    ap.add_argument("--dt-scale", type=float, default=1.0)
    ap.add_argument("--hold-steps", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--plateau-deg", type=float, default=0.15)
    ap.add_argument("--plateau-blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res = run_case(args.n, radius=args.radius, mu=args.mu,
                   mu_rolling=args.mu_rolling, e_star=args.e_star,
                   gravity=args.gravity, aspect=args.aspect,
                   dt_scale=args.dt_scale, hold_steps=args.hold_steps,
                   max_steps=args.max_steps, block=args.block,
                   plateau_deg=args.plateau_deg,
                   plateau_blocks=args.plateau_blocks, seed=args.seed)
    cloud = res.pop("_final_positions")
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                     f"cloud_{res['label']}.npz"),
                        pos=cloud, radius=res["radius"])
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("surface_profile", "repose_angle_history")},
                     indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()