"""Reproduce the flat-facet contact non-uniqueness of the batched kernel (m6292).

Reads nothing but ``tensordem.superquadric``; prints one row per
``contact_convergence_study`` case, the sweep-count spread of the reported
overlap, and the analytic (exact-plane) overlap for reference.
"""

from __future__ import annotations

import json

import numpy as np

from tensordem.superquadric import (
    SuperquadricBatch,
    _quat_to_matrix,
    contact_convergence_study,
    superquadric_contact_batch,
)


def _plane_overlap(body_a: SuperquadricBatch, body_b: SuperquadricBatch, axis: int) -> float:
    """Exact overlap for two bodies whose nearest facets share one local axis.

    For a plane at distance ``h`` from the centre, the perpendicular support
    distance along axis ``k`` of the world frame is the axis-projection of the
    rotated half-extent, so the exact face-to-face overlap is
    ``(h_a + h_b) - |dc| `` with ``h = |(R^T e_k)·ext|``.
    """
    e = np.zeros(3)
    e[axis] = 1.0
    ha = abs(np.dot(body_a.rot[0].T @ e, np.array([body_a.a[0], body_a.b[0], body_a.c[0]])))
    hb = abs(np.dot(body_b.rot[0].T @ e, np.array([body_b.a[0], body_b.b[0], body_b.c[0]])))
    dc = float(np.linalg.norm(body_b.center[0] - body_a.center[0]))
    return ha + hb - dc


def main() -> None:
    study = contact_convergence_study(n_cases=40)
    print(json.dumps({k: v for k, v in study.items() if k != "rows"}, indent=2))
    rows = sorted(study["rows"], key=lambda r: -(max(r["overlap_by_iteration"]) - min(r["overlap_by_iteration"])))
    print("\nworst 8 cases by sweep spread:")
    for r in rows[:8]:
        vals = np.array(r["overlap_by_iteration"])
        print(f"  n={r['shape_exponent']:5.1f} ext={np.round(r['half_extents'], 5)} "
              f"spread={vals.max() - vals.min():.3e}  vals={np.round(vals, 6)}")

    # deterministic flat-facet case: two aligned rounded cubes, face-to-face,
    # the same overlap must come out for every sweep count.
    radius = 0.01
    for n in (8.0, 16.0, 40.0):
        ca = np.array([[0.0, 0.0, 0.0]])
        cb = np.array([[0.0, 0.0, 2.0 * radius * 0.97]])
        ba = SuperquadricBatch(np.array([radius]), np.array([radius]), np.array([radius]),
                               np.array([n]), np.array([n]), ca, np.eye(3))
        bb = SuperquadricBatch(np.array([radius]), np.array([radius]), np.array([radius]),
                               np.array([n]), np.array([n]), cb, np.eye(3))
        vals = [float(superquadric_contact_batch(ba, bb, n_iter=it)["overlap"][0])
                for it in (1, 2, 3, 4, 6, 8, 12, 20)]
        print(f"aligned cube n={n}: exact={_plane_overlap(ba, bb, 2):.9e} "
              f"kernel={np.round(vals, 9)} spread={max(vals) - min(vals):.2e}")

    # tilted flat facet: same overlap from both sides (symmetry) but the
    # reported normal flips between near-tied facet axes.
    rng = np.random.default_rng(11)
    for k in range(4):
        rot_a = _quat_to_matrix(rng.normal(size=(1, 4)))
        rot_b = _quat_to_matrix(rng.normal(size=(1, 4)))
        ca = np.zeros((1, 3))
        cb = np.array([[0.3, -0.2, 1.94]]) * radius
        ba = SuperquadricBatch(np.array([radius]), np.array([radius]), np.array([radius]),
                               np.array([16.0]), np.array([16.0]), ca, rot_a)
        bb = SuperquadricBatch(np.array([radius]), np.array([radius]), np.array([radius]),
                               np.array([16.0]), np.array([16.0]), cb, rot_b)
        out = [superquadric_contact_batch(ba, bb, n_iter=it) for it in (3, 4, 6, 20)]
        ov = [float(o["overlap"][0]) for o in out]
        nn = [np.round(o["normal"][0], 4).tolist() for o in out]
        print(f"tilted case {k}: overlap={np.round(ov, 6)} spread={max(ov) - min(ov):.2e}")
        print(f"                 normals={nn}")


if __name__ == "__main__":
    main()