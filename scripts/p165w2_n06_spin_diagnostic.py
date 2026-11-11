"""P165-W2-N06 model diagnostic: why the parent column test has no repose angle.

Three measured facts, each with a negative control, written to
``artifacts/dem/p165w2_n06/model_diagnostic.json``:

1. **Spin pump.** The parent sphere-sphere path takes the *centre* velocities as
   the tangential slip.  A rolling grain therefore never "sees" that its spin has
   already removed the slip, and the tangential spring keeps spinning it up
   (measured |omega| of order 10^2-10^3 rad/s, rotational kinetic energy of the
   same order as the drop height's potential energy).  ``spin_slip=True``
   evaluates the Cundall-Strack slip at the *contact point* (``v + omega x r``),
   the same convention the parent already uses for wall contacts.
2. **Ordered lattice jams.** The parent case builder starts from a perfect cubic
   lattice; compacted under gravity it locks and keeps a vertical face standing
   (no collapse), so there is nothing to measure.
3. **Smooth floor control.** Without a rough base the grains roll away and the
   measured angle is 0 deg (see the ``smooth`` case of the series artefact).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p165w2_n06_column_fast import build_bed, build_case            # noqa: E402
from tensordem.flat_punch import reduced_modulus                    # noqa: E402
from tensordem.sphere_pack_dem import (                              # noqa: E402
    SpherePack, SpherePackConfig, SphereWall)
from tensordem.superquadric_dem import repose_angle_from_cloud       # noqa: E402
from tensordem.superquadric_kernel import ContactMaterial            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = 0.02


def material(mu_rolling: float = 0.10) -> ContactMaterial:
    return ContactMaterial(e_star=reduced_modulus(1e7, 0.3, 1e7, 0.3),
                           g_star=1e7 / 5.2, mu=0.45, mu_rolling=mu_rolling,
                           restitution=0.4)


def build(n: int = 1024, jitter: float = 0.3, bed: int = 2):
    pos, hw, _h, _n = build_case(n, R, aspect=2.0)
    pos = pos.copy()
    sp = 2.2 * R
    rng = np.random.default_rng(0)
    if jitter > 0.0:
        pos[:, 0] += jitter * sp * rng.standard_normal(len(pos))
        pos[:, 1] += jitter * sp * rng.standard_normal(len(pos))
        pos[:, 2] += 0.5 * jitter * sp * rng.standard_normal(len(pos))
    if bed:
        base = build_bed(R, hw, layers=bed)
        pos[:, 2] += R * 1.02 + bed * 2.02 * R
        allp = np.vstack([base, pos])
        fixed = np.zeros(len(allp), dtype=bool)
        fixed[:len(base)] = True
    else:
        allp, fixed = pos, None
    walls = [SphereWall(np.array([0.0, 0.0, 1.0]), 0.0, mu=0.45, label="floor")]
    for nrm in (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
                np.array([0, 1.0, 0]), np.array([0, -1.0, 0])):
        walls.append(SphereWall(nrm, -hw, mu=0.45, label="gate"))
    return allp, fixed, walls, hw


def spin_test(spin_slip: bool, damping_rotational: float, steps: int = 1500) -> dict:
    allp, fixed, walls, hw = build()
    cfg = SpherePackConfig(material=material(), gravity=np.array([0, 0, -9.81]),
                           spin_slip=spin_slip, damping_rotational=damping_rotational)
    pack = SpherePack(allp, np.full(len(allp), R), 2500.0, walls, cfg, fixed=fixed)
    pack.run(400)
    for w in walls[1:]:
        w.active = False
    mv = pack.moving
    ke_r = []
    for _ in range(steps // 100):
        pack.run(100)
        ke_r.append(0.5 * float(np.sum(pack.inertia[mv, None] * pack.omega[mv] ** 2)))
    return {"spin_slip": spin_slip, "damping_rotational": damping_rotational,
            "steps": steps,
            "omega_max_rad_s": float(np.abs(pack.omega[mv]).max()),
            "ke_rot_end_J": ke_r[-1],
            "ke_rot_series_J": ke_r,
            "ke_trans_end_J": 0.5 * float(np.sum(
                pack.mass[mv] * np.einsum("ij,ij->i", pack.vel[mv], pack.vel[mv]))),
            "surface_speed_over_centre_speed":
                float(np.abs(pack.omega[mv]).max() * R /
                      max(np.abs(pack.vel[mv]).max(), 1e-12))}


def jam_test(jitter: float, steps: int = 4000) -> dict:
    allp, fixed, walls, hw = build(jitter=jitter)
    cfg = SpherePackConfig(material=material(), gravity=np.array([0, 0, -9.81]),
                           damping_global=2.0, damping_rotational=200.0)
    pack = SpherePack(allp, np.full(len(allp), R), 2500.0, walls, cfg, fixed=fixed)
    pack.run(400)
    for w in walls[1:]:
        w.active = False
    p0 = pack.pos[pack.moving].copy()
    pack.run(steps)
    q = pack.pos[pack.moving]
    r0 = float(np.linalg.norm(p0[:, :2], axis=1).max())
    r1 = float(np.linalg.norm(q[:, :2], axis=1).max())
    ang, _ = repose_angle_from_cloud(q, n_bins=12, use_quantile=0.9)
    return {"jitter": jitter, "steps_after_release": steps,
            "n_moving": int(pack.moving.sum()),
            "z_max_initial": float(p0[:, 2].max()), "z_max_final": float(q[:, 2].max()),
            "z_mean_initial": float(p0[:, 2].mean()), "z_mean_final": float(q[:, 2].mean()),
            "r_max_initial": r0, "r_max_final": r1,
            "repose_angle_deg": ang,
            "height_retained_fraction": float(q[:, 2].mean() / max(p0[:, 2].mean(), 1e-30))}


def main() -> None:
    out = {"campaign_node": "P165-W2-N06", "repo": "TensorDEM",
           "case": "column collapse, N=1024, r=20 mm, rough bed (see fast_column_collapse.json)",
           "spin_pump": [spin_test(False, 0.0), spin_test(True, 0.0),
                         spin_test(False, 200.0), spin_test(True, 200.0)],
           "jam": [jam_test(0.0), jam_test(0.3)]}
    art = os.path.join(REPO, "artifacts", "dem", "p165w2_n06")
    os.makedirs(art, exist_ok=True)
    path = os.path.join(art, "model_diagnostic.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
