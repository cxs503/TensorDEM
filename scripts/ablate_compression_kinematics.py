#!/usr/bin/env python3
"""Ablation of the plane-strain compression driver's kinematics (P82W2-N02).

The pre-registered calibration of ``mu(I)`` pools a simple-shear element test
with a plane-strain compression element test.  The two disagreed by 0.02-0.03 in
``mu`` at the same ``I`` (P82W4-N02, honest FAIL of criterion c).  This script
isolates the three kinematic defects of the earlier compression driver and shows
what each one does to the measured stress ratio:

``legacy``      static piston wall, positions remapped without the affine
                velocity increment ("teleport" remap), rate taken as the
                nominal ``2 |eps_y|`` -- the P82W4-N02 implementation;
``piston``      legacy + the piston carries its real velocity;
``remap``       piston + the affine velocity increment on the x remap;
``measured``    remap + the rate measured from the bulk velocity gradient
                (this last one is an analysis choice, not a driver change, and is
                applied to every variant here so that the table is comparable).

Every variant is integrated here with the same packing, seed and time step; the
only differences are the three flags.  Output: JSON + a markdown table.

    python scripts/ablate_compression_kinematics.py --rates 1.5 4.0 --jobs 6
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.rubble_postprocess import (   # noqa: E402
    contact_view,
    rheology_point,
)
from tensordem.rubble_scenarios import (     # noqa: E402
    bulk_deformation_rate,
    bulk_mask,
    make_compression_cell,
)

DENSITY = 917.0
DEFAULT_OUT = REPO / "artifacts" / "rubble_rheology_ablation"
CELL = dict(n_particles=380, mean_radius=0.014, width=0.40, height=0.45,
            phi=0.80, thickness=0.03)


def run_variant(*, strain_rate: float, seed: int = 0, total_strain: float = 0.30,
                snapshots: int = 12, steady_fraction: float = 0.45,
                piston_velocity: bool, affine_velocity: bool) -> dict:
    """Plane-strain compression with the two kinematic switches.

    ``piston_velocity=False, affine_velocity=False`` reproduces the P82W4-N02
    driver bit for bit (static piston, position-only remap).
    """
    system = make_compression_cell(seed=seed, **CELL)
    width, height = CELL["width"], CELL["height"]
    thickness = CELL["thickness"]
    dt = system.dt
    n_steps = max(int(round(total_strain / (strain_rate * dt))), 50)
    start = int(round(steady_fraction * n_steps))
    every = max(1, (n_steps - start) // snapshots)
    v_piston = strain_rate * height
    margin = 3.0 * system.mean_radius
    h, w = height, width
    if piston_velocity:
        system.set_wall_velocity(np.array([[0.0, 0.0], [0.0, -v_piston]]))
    points = []
    for step in range(1, n_steps + 1):
        dh = v_piston * dt
        system.walls[1] -= np.array([0.0, dh])
        h -= dh
        factor = width * height / (w * h)
        old_x = system.pos[:, 0].copy()
        system.pos[:, 0] *= factor
        if affine_velocity:
            system.vel[:, 0] += (factor - 1.0) * old_x / dt
        system.walls[:, :, 0] *= factor
        w *= factor
        system.periodic_x = w
        system.step(1)
        if step > start and (step - start) % every == 0:
            system.contact_forces()
            view = contact_view(system.contacts)
            if view.count == 0:
                continue
            mask = bulk_mask(_sample(system, w, h, thickness), margin=margin)
            if int(np.sum(mask)) < 20:
                continue
            grad = bulk_deformation_rate(system.pos[mask], system.vel[mask],
                                         length_x=w, min_particles=20)
            point = rheology_point(_sample(system, w, h, thickness), view,
                                   shear_rate=grad["gamma_dot"],
                                   density=DENSITY, particle_mask=mask)
            point["nominal_rate"] = 2.0 * v_piston / h
            point["eps_dot_y"] = grad["eps_dot_y"]
            points.append(point)
    mu = np.array([p["mu"] for p in points])
    rate = np.array([p["shear_rate"] for p in points])
    nominal = np.array([p["nominal_rate"] for p in points])
    return {"rate_nominal": 2.0 * strain_rate, "n": len(points),
            "I_measured": float(np.mean([p["I"] for p in points])),
            "mu_measured_rate": float(mu.mean()),
            "mu_se": float(mu.std(ddof=1) / math.sqrt(len(points))),
            "pressure": float(np.mean([p["pressure"] for p in points])),
            "measured_over_nominal_rate": float(rate.mean() / nominal.mean()),
            "per_snapshot_mu": mu.round(4).tolist()}


def _sample(system, width, height, thickness):
    from tensordem.rubble_postprocess import ParticleSample
    return ParticleSample(system.pos.copy(), system.radii.copy(),
                          (float(width), float(height)), thickness,
                          (True, False), "ablation", 0.0)


VARIANTS = {"legacy": (False, False), "piston": (True, False),
            "remap": (True, True)}


def _job(args):
    name, rate, seed = args
    piston, remap = VARIANTS[name]
    t = time.time()
    out = run_variant(strain_rate=rate, seed=seed, piston_velocity=piston,
                      affine_velocity=remap)
    out.update({"variant": name, "seed": seed, "wall_time_s": round(time.time() - t, 1)})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rates", type=float, nargs="+", default=[1.5, 4.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    jobs = [(v, r, args.seed) for v in VARIANTS for r in args.rates]
    with Pool(min(args.jobs, len(jobs))) as pool:
        rows = pool.map(_job, jobs)
    rows.sort(key=lambda r: (r["rate_nominal"], r["variant"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(rows, indent=1))
    lines = ["| variant | nominal 2|eps_y| (1/s) | I (measured) | mu | mu SE | P (kPa) "
             "| measured/nominal rate |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['rate_nominal']:.2f} | {r['I_measured']:.5f} "
                     f"| {r['mu_measured_rate']:.4f} | {r['mu_se']:.4f} "
                     f"| {r['pressure'] / 1e3:.1f} | {r['measured_over_nominal_rate']:.3f} |")
    table = "\n".join(lines)
    (args.out / "ablation.md").write_text(table + "\n")
    print(table, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())