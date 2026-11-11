"""P165-W2-N06 (f): column-collapse repose-angle convergence on the array engine.

Same physical case as ``scripts/p165w2_n06_column_collapse.py`` (loose cubic
packing of monodisperse spheres, four gate walls, frictional floor, gates
released, free-surface angle measured from the cloud), but on
:class:`tensordem.sphere_pack_dem.SpherePack`, whose state is numpy arrays and
whose neighbour list is a Verlet list.  That is what makes the particle-number
series up to N = 32 768 affordable: the parent object-oriented engine costs
0.3 s/step at N = 2 048 and 0.9 s/step at N = 4 000 (measured), i.e. ~7 h for
one settled N = 32 k run, whereas the array engine runs the same case in
minutes (``wall_clock_s`` in every JSON says which).

The engine's force law is the parent's (shared kernel functions, same
arguments); equivalence on identical initial states is a test
(``tests/test_sphere_pack_dem.py``).

Usage (one run per invocation so the sweep can be sharded)::

    PYTHONPATH=src python scripts/p165w2_n06_column_fast.py \
        --n 2048 --radius 0.02 --mu 0.45 --out runs/p165w2_n06/f/n2048.json
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
from tensordem.sphere_pack_dem import (                                  # noqa: E402
    SpherePack,
    SpherePackConfig,
    SphereWall,
)
from tensordem.superquadric_dem import repose_angle_from_cloud           # noqa: E402
from tensordem.superquadric_kernel import ContactMaterial                # noqa: E402


def build_bed(radius: float, half_width: float, layers: int = 2,
              spacing_factor: float = 2.02):
    """Fixed rough bed (immobile particle layers) under the column.

    Why a rough bed: a heap of *spheres* on a smooth frictional wall has no
    angle of repose at all -- the parent engine applies no rolling resistance at
    wall contacts, so the grains roll outwards and flatten into a monolayer
    (measured: 0.00 deg on the smooth floor, see the report).  A bed of frozen
    particles gives the base a no-slip condition, which is the standard DEM way
    of measuring a repose angle; the angle then reflects the *particle-particle*
    friction/rolling law, i.e. the physics this node is about.
    """
    sp = spacing_factor * radius
    pad = 2.0 * radius
    n_side = int(math.ceil(2.0 * (half_width + pad) / sp))
    xs = (np.arange(n_side) - 0.5 * (n_side - 1)) * sp
    pts = []
    for k in range(layers):
        off = 0.0 if k % 2 == 0 else 0.5 * sp
        X, Y = np.meshgrid(xs + off, xs + off, indexing="ij")
        layer = np.column_stack([X.ravel(), Y.ravel(),
                                 np.full(X.size, radius * 1.02 + k * sp)])
        pts.append(layer[(np.abs(layer[:, 0]) <= half_width + pad)
                         & (np.abs(layer[:, 1]) <= half_width + pad)])
    return np.vstack(pts)


def build_case(n_target: int, radius: float, aspect: float = 0.5):
    """Loose cubic packing: fixed aspect ``layers / per_side`` (system grows with N)."""
    spacing = 2.2 * radius
    per_side = max(int(round((n_target / max(aspect, 1e-6)) ** (1.0 / 3.0))), 2)
    layers = max(int(round(aspect * per_side)), 2)
    n = per_side ** 2 * layers
    half_width = 0.5 * per_side * spacing
    height = layers * spacing
    offset = -0.5 * (per_side - 1) * spacing
    pos = np.empty((layers, per_side, per_side, 3), dtype=float)
    pos[..., 0] = offset + np.arange(per_side)[None, :, None] * spacing
    pos[..., 1] = offset + np.arange(per_side)[None, None, :] * spacing
    pos[..., 2] = radius * 1.02 + np.arange(layers)[:, None, None] * spacing
    return pos.reshape(-1, 3), half_width, height, n


def run_case(n_target: int, radius: float = 0.02, mu: float = 0.45,
             mu_rolling: float = 0.10, e_star: float = 1.0e7,
             gravity: float = 9.81, aspect: float = 2.0,
             dt_scale: float = 1.0, hold_steps: int = 400,
             max_steps: int = 40000, block: int = 500,
             plateau_deg: float = 0.5, plateau_blocks: int = 3,
             seed: int = 0, density: float = 2500.0, label: str = "",
             bed_layers: int = 2, jitter: float = 0.3,
             damping_global: float = 2.0, damping_rotational: float = 200.0,
             quantile: float = 0.9, n_bins: int = 12) -> dict:
    pos, half_width, height, n = build_case(n_target, radius, aspect=aspect)
    rng = np.random.default_rng(seed)
    sp = 2.2 * radius
    if jitter > 0.0:
        # randomise the initial lattice: an *ordered* cubic packing jams and keeps a
        # vertical face standing (measured: no collapse at all), whereas a loose
        # random packing is what a collapsed column experiment actually starts from
        pos[:, 0] += jitter * sp * rng.standard_normal(len(pos))
        pos[:, 1] += jitter * sp * rng.standard_normal(len(pos))
        pos[:, 2] += 0.5 * jitter * sp * rng.standard_normal(len(pos))
    else:
        pos[:, :2] += 1e-4 * radius * rng.standard_normal((len(pos), 2))
    if bed_layers:
        bed = build_bed(radius, half_width, layers=bed_layers)
        z0 = radius * 1.02 + bed_layers * 2.02 * radius + 1e-3 * radius
        pos[:, 2] += z0
        all_pos = np.vstack([bed, pos])
        fixed = np.zeros(len(all_pos), dtype=bool)
        fixed[:len(bed)] = True
        n_moving = len(pos)
    else:
        all_pos, fixed, n_moving = pos, None, len(pos)
    height = float(np.max(pos[:, 2]) + radius)
    walls = [SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, mu=mu, label="floor")]
    for nrm in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])):
        walls.append(SphereWall(nrm, -half_width, mu=mu, label="gate"))
    mat = ContactMaterial(e_star=reduced_modulus(e_star, 0.3, e_star, 0.3),
                          g_star=e_star / 5.2, mu=mu, mu_rolling=mu_rolling,
                          restitution=0.4)
    cfg = SpherePackConfig(material=mat, gravity=np.array([0.0, 0.0, -float(gravity)]),
                           damping_global=damping_global,
                           damping_rotational=damping_rotational)
    pack = SpherePack(all_pos, np.full(len(all_pos), float(radius)), density, walls, cfg,
                      fixed=fixed)
    pack.dt = pack.dt * float(dt_scale)

    moving_mask = pack.moving
    cloud_moving = lambda: pack.pos[moving_mask]
    t0 = time.perf_counter()
    pack.run(hold_steps)                       # settle inside the gates
    hold_pen = pack.max_penetration_ratio
    for w in walls[1:]:
        w.active = False                       # release
    t_release = pack.time
    t_fall = math.sqrt(2.0 * height / float(gravity))
    min_steps = int(2.2 * t_fall / pack.dt)
    ke0 = pack.kinetic_energy()
    angles, steps, step_times = [], 0, []
    plateau = False
    t_loop = time.perf_counter()
    while steps < max_steps:
        pack.run(block)
        steps += block
        ang, profile = repose_angle_from_cloud(cloud_moving(), n_bins=n_bins,
                                               use_quantile=quantile)
        angles.append(ang)
        step_times.append(time.perf_counter())
        if steps >= min_steps and len(angles) > plateau_blocks:
            recent = angles[-(plateau_blocks + 1):]
            if max(recent) - min(recent) < plateau_deg:
                plateau = True
                break
    wall_loop = time.perf_counter() - t_loop
    pos_final = pack.positions()
    angle, profile = repose_angle_from_cloud(pos_final[moving_mask], n_bins=n_bins,
                                             use_quantile=quantile)
    ke1 = pack.kinetic_energy()
    out = {
        "label": label or f"n{n_target}_mu{mu}_dt{dt_scale}",
        "engine": "tensordem.sphere_pack_dem.SpherePack",
        "n_target": n_target, "n_actual": n, "radius": radius,
        "mu": mu, "mu_rolling": mu_rolling, "e_star": e_star,
        "gravity": float(gravity), "aspect_initial": aspect,
        "density": density, "dt": pack.dt, "dt_scale": dt_scale,
        "half_width": half_width, "fill_height": height,
        "hold_steps": hold_steps, "settle_steps": steps,
        "bed_layers": bed_layers, "n_bed": int(len(all_pos) - n_moving),
        "jitter": jitter, "damping_global": damping_global,
        "damping_rotational": damping_rotational,
        "angle_quantile": quantile, "angle_n_bins": n_bins,
        "n_particles_total": int(len(all_pos)),
        "min_settle_steps": min_steps, "t_fall_s": t_fall,
        "total_steps": hold_steps + steps,
        "repose_angle_deg": angle,
        "repose_angle_history": angles,
        "plateau_reached": plateau,
        "surface_profile": profile,
        "hold_penetration_ratio": hold_pen,
        "max_penetration_ratio": pack.max_penetration_ratio,
        "kinetic_energy_start": ke0,
        "kinetic_energy_end": ke1,
        "kinetic_energy_rel_change": abs(ke1 - ke0) / max(abs(ke0), 1e-30),
        "ms_per_step_settle": 1e3 * wall_loop / max(steps, 1),
        "physical_time_release": steps * pack.dt,
        "time_release_s": t_release,
        "wall_clock_s": time.perf_counter() - t0,
        "final_positions_shape": list(pos_final.shape),
    }
    return out, pos_final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=0.45)
    ap.add_argument("--mu-rolling", type=float, default=0.10)
    ap.add_argument("--e-star", type=float, default=1.0e7)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--aspect", type=float, default=2.0)
    ap.add_argument("--bed-layers", type=int, default=2)
    ap.add_argument("--dt-scale", type=float, default=1.0)
    ap.add_argument("--hold-steps", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--plateau-deg", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--damping-global", type=float, default=2.0)
    ap.add_argument("--damping-rotational", type=float, default=200.0)
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--plateau-blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res, cloud = run_case(args.n, radius=args.radius, mu=args.mu,
                          mu_rolling=args.mu_rolling, e_star=args.e_star,
                          gravity=args.gravity, aspect=args.aspect,
                          dt_scale=args.dt_scale, hold_steps=args.hold_steps,
                          max_steps=args.max_steps, block=args.block,
                          plateau_deg=args.plateau_deg,
                          plateau_blocks=args.plateau_blocks, seed=args.seed,
                          bed_layers=args.bed_layers, jitter=args.jitter,
                          damping_global=args.damping_global,
                          damping_rotational=args.damping_rotational,
                          quantile=args.quantile, n_bins=args.n_bins)
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                     f"cloud_{res['label']}.npz"),
                        pos=cloud, radius=res["radius"])
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("surface_profile", "repose_angle_history")},
                     indent=1))


if __name__ == "__main__":
    main()