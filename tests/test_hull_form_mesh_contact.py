"""Ship-hull-form (Wigley) contact verification for ``hull_mesh_contact``.

Pins the invariants behind ``scripts/m16794_wigley_hull_contact_audit.py`` and
``docs/AUDIT_GAP_DEM_G3_WIGLEY_REAL_HULL_CONTACT.md`` (GAP-DEM-G3 open parts:
*real ship-hull form* geometry and *penalty stiffness/damping calibration*):

* the analytic reference is itself checked against a brute-force dense grid, so
  a wrong reference cannot silently bless the kernel (the trap recorded in
  ``docs/dem3d_hull_mesh_contact_curved_audit.md``);
* on a fine hull mesh the kernel reproduces the analytic penalty force;
* the directional sagitta rule ``(k_u h_u^2 + k_v h_v^2)/8`` is a *sufficient*
  bound on the relative force error (``predicted >= measured``);
* restitution of the implemented spring-dashpot law follows
  ``e = exp(-beta*pi/sqrt(1-beta^2))``;
* nothing is emitted where there is no contact, and action/reaction balances.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
_SPEC = importlib.util.spec_from_file_location(
    "m16794_wigley_audit", Path(__file__).resolve().parents[1] / "scripts"
    / "m16794_wigley_hull_contact_audit.py")
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

from tensordem.hull_form_mesh import (  # noqa: E402
    WigleyHull,
    wigley_curvature,
    wigley_directional_curvature,
    wigley_distance,
    wigley_inside,
    wigley_mesh,
)


@pytest.fixture(scope="module")
def fine():
    return wigley_mesh(160, 64, audit.HULL)


# --- the reference ---------------------------------------------------------

def test_reference_never_worse_than_dense_grid():
    res = audit.reference_self_validation(n_points=25, seed=3)
    assert res["n_solver_worse_than_grid"] == 0
    # an independent 401x401 local grid around the solver foot must not improve it
    assert res["max_local_gain_m"] < 1e-9


def test_reference_known_geometry():
    hull = WigleyHull()
    # midship waterline point: the surface touches y = B/2 = 0.2 at x = 0, z = 0
    d, foot, _uv = wigley_distance([0.0, 0.30, 0.0], hull)
    assert d == pytest.approx(0.10, rel=1e-3)
    assert foot[1] == pytest.approx(0.20, rel=1e-3)
    # a point on the centreplane below the keel is nearest to the keel line
    d2, foot2, _uv2 = wigley_distance([0.0, 0.0, -0.30], hull)
    assert d2 == pytest.approx(0.05, rel=1e-3)
    assert abs(foot2[2] + hull.draft) < 1e-2


def test_inside_test():
    assert wigley_inside([0.0, 0.0, -0.10])
    assert not wigley_inside([0.0, 0.30, -0.10])
    assert not wigley_inside([2.5, 0.0, -0.10])


# --- geometry accuracy -----------------------------------------------------

def test_fine_mesh_matches_analytic_force(fine):
    verts, faces = fine
    worst = 0.0
    for name, u, v in audit.ZONES:
        for side in (1.0, -1.0):
            centre, _f, _n = audit.sphere_at(u, v, 0.010, side=side)
            f_mesh, _r = audit.kernel_force(centre, verts, faces)
            f_ref, _d = audit.wigley_contact_reference(
                centre, audit.R_P, stiffness=audit.K, hull=audit.HULL)
            rel = abs(np.linalg.norm(f_mesh[0]) - np.linalg.norm(f_ref[0])) \
                / np.linalg.norm(f_ref[0])
            worst = max(worst, rel)
    assert worst < 0.01, f"fine-mesh force error {worst:.3%} exceeds 1 %"


def test_no_force_without_contact(fine):
    verts, faces = fine
    centre = np.array([[0.0, 0.60, -0.12]])          # 0.4 m off the side
    f, r = audit.kernel_force(centre, verts, faces)
    assert np.linalg.norm(f[0]) == 0.0
    assert np.linalg.norm(r) == 0.0


def test_action_reaction(fine):
    res = audit.action_reaction_check()
    assert res["residual_n"] < 1e-9
    assert res["force_n"] > 0.0


def test_broad_phase_superset():
    res = audit.broad_phase_completeness(n_queries=120, seed=5)
    assert res["n_missing_superset_violations"] == 0


# --- the digitisation rule -------------------------------------------------

@pytest.mark.parametrize("n_u,n_v", [(24, 10), (48, 20)])
def test_directional_sagitta_rule_is_sufficient(n_u, n_v):
    verts, faces = wigley_mesh(n_u, n_v, audit.HULL)
    pen = 0.010
    worst_rel, worst_pred = 0.0, 0.0
    for name, u, v in audit.ZONES:
        ku, kv = wigley_directional_curvature(u, v, audit.HULL)
        cu = float(np.linalg.norm(audit.HULL.surface(min(u + 1.0 / n_u, 1.0), v)
                                  - audit.HULL.surface(u, v)))
        cv = float(np.linalg.norm(audit.HULL.surface(u, min(v + 1.0 / n_v, 1.0))
                                  - audit.HULL.surface(u, v)))
        pred = (ku * cu * cu + kv * cv * cv) / 8.0 / pen
        worst_pred = max(worst_pred, pred)
        centre, _f, _n = audit.sphere_at(u, v, pen)
        f_mesh, _r = audit.kernel_force(centre, verts, faces)
        f_ref, _d = audit.wigley_contact_reference(centre, audit.R_P,
                                                   stiffness=audit.K, hull=audit.HULL)
        rel = abs(np.linalg.norm(f_mesh[0]) - np.linalg.norm(f_ref[0])) \
            / np.linalg.norm(f_ref[0])
        worst_rel = max(worst_rel, rel)
        assert rel < 0.10, f"{name}: {rel:.3%} at {n_u}x{n_v}"
    # the rule is a sufficient bound on the worst error over the hull form
    assert worst_rel <= worst_pred + 1e-9, \
        f"measured {worst_rel:.4f} > predicted {worst_pred:.4f}"


def test_curvature_directional_vs_principal():
    ku, kv = wigley_directional_curvature(0.5, 0.7, audit.HULL)
    k1, k2 = wigley_curvature(0.5, 0.7, audit.HULL)
    # section curvature dominates at midship and is the principal one
    assert kv == pytest.approx(k1, rel=1e-2)
    assert ku < kv
    assert k2 >= 0.0


# --- damping / restitution calibration -------------------------------------

def test_restitution_matches_closed_form():
    res = audit.restitution_study(mass=1.0, v0=0.02, betas=(0.2,),
                                  steps_per_tc=(100,), n_u=16, n_v=8)
    row = res["rows"][0]
    assert row["e_measured"] == pytest.approx(row["e_reference"], rel=0.03)
    assert 0.0 < row["e_measured"] < 1.0


def test_restitution_dt_convergence():
    res = audit.restitution_study(mass=1.0, v0=0.02, betas=(0.3,),
                                  steps_per_tc=(50, 100), n_u=16, n_v=8)
    coarse, fine = res["rows"]
    assert fine["rel_err"] <= coarse["rel_err"] + 1e-6
    assert fine["rel_err"] < 0.05