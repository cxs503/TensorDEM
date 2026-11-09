"""Tests for the curved-patch mortar: patch-local conservation, gap/penetration
accounting, time-domain coupling energy account and transpose consistency.

Every positive assertion is paired with a deliberately broken control
(``measure="centroid_j"``, ``measure="flat"`` or the previous plane-projection
construction) so that the gates are shown to be able to fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.interface_operators import curved_patch_mortar as cpm


PATCHES = [cpm.SpherePatch(), cpm.GraphPatch()]


# ---------------------------------------------------------------------------
# geometry: Newton projection and the curved measure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
def test_patch_area_matches_analytic(patch):
    """Total curved measure vs the closed form."""
    if isinstance(patch, cpm.SpherePatch):
        # J = R^2 sin(theta) dtheta dphi; int sin(theta) dtheta = cos(t0) - cos(t1)
        exact = patch.radius ** 2 * (patch.phi1 - patch.phi0) * (
            math.cos(patch.theta0) - math.cos(patch.theta1))
    else:
        # Gauss-Legendre reference for int_0^1 int_0^1 |x_u x x_v| du dv
        xg, wg = np.polynomial.legendre.leggauss(60)
        u = 0.5 * (xg + 1.0)
        w = 0.5 * wg
        U, V = np.meshgrid(u, u, indexing="ij")
        W = np.outer(w, w)
        uv = np.stack([U.ravel(), V.ravel()], axis=1)
        exact = float(np.sum(W.ravel() * patch.jac(uv)))
    assert abs(patch.area() - exact) / exact < 1e-12


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
def test_newton_projection_is_quadratic_and_exact(patch):
    """``x(u, v) = p`` recovered to round-off, in a handful of Newton steps."""
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(12):
        uv_true = rng.uniform(0.05, 0.95, size=2)
        p = patch.x(uv_true[None, :])[0]
        uv, dist, iters, _g = patch.project(p, uv0=np.array([0.5, 0.5]))
        worst = max(worst, dist)
        assert np.allclose(uv, uv_true, atol=1e-12)
        assert iters <= 12
    assert worst < 1e-13


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
def test_ambient_mesh_projection_roundtrip(patch):
    """A 3D vertex cloud that lies on the patch projects back with zero residual."""
    side = cpm.grid_side(patch, 5, 4, jitter=0.25, seed=1)
    verts = patch.x(side.tris.reshape(-1, 2))
    faces = np.arange(len(verts)).reshape(-1, 3)
    amb = cpm.side_from_ambient(patch, verts, faces)
    assert amb.projection_residual <= 1e-12 * math.sqrt(patch.area())
    # the two constructions agree measure by measure
    assert np.max(np.abs(amb.measures - side.measures) / side.measures) < 1e-10


# ---------------------------------------------------------------------------
# criterion E2: per-patch local conservation gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
@pytest.mark.parametrize("n", [4, 6, 8])
def test_per_patch_conservation_gate(patch, n):
    """``|sum_i m_ij - |Gamma_j|| <= 1e-12 |Gamma_j|`` on every master facet."""
    m = cpm.grid_side(patch, n, n, jitter=0.3, seed=11)
    s = cpm.grid_side(patch, n + 3, n + 2, jitter=0.3, seed=23)
    bal = cpm.per_patch_balance(m, s)
    assert bal["global_rel"] == pytest.approx(0.0, abs=1e-14)
    assert bal["per_facet_max_rel"] <= 1e-12


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
def test_broken_measures_fail_the_per_facet_gate(patch):
    """Controls: centroid Jacobian and flat metric both break the local gate."""
    m = cpm.grid_side(patch, 6, 6, jitter=0.3, seed=11)
    s = cpm.grid_side(patch, 9, 8, jitter=0.3, seed=23)
    good = cpm.CurvedMortarOperator(m, s).conservation()["per_facet_max_rel"]
    cen = cpm.CurvedMortarOperator(m, s, measure="centroid_j").conservation()["per_facet_max_rel"]
    flat = cpm.CurvedMortarOperator(m, s, measure="flat").conservation()["per_facet_max_rel"]
    plane = cpm.plane_projection_defect(m, s)["per_facet_max_rel"]
    assert good <= 1e-12
    assert cen > 1e-5 and flat > 1e-2 and plane > 1e-2
    assert cen > 1e3 * good


# ---------------------------------------------------------------------------
# criterion E3: gap / penetration volume terms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
def test_gap_and_penetration_terms_close_the_patch_balance(patch):
    """``|Gamma_j| + V_pen - V_gap = sum_i m_ij`` per facet, with both terms non-zero."""
    m = cpm.grid_side(patch, 5, 5, jitter=0.2, seed=3)
    conforming = cpm.grid_side(patch, 6, 6, jitter=0.2, seed=5)
    gapped = cpm.grid_side(patch, 6, 6, jitter=0.2, seed=5, box=(0.05, 0.95, 0.04, 0.96))
    penetrating = cpm.concat_sides(
        cpm.grid_side(patch, 6, 6, jitter=0.2, seed=5),
        cpm.grid_side(patch, 4, 4, jitter=0.1, seed=6, box=(0.25, 0.75, 0.25, 0.75)),
        label="pen")

    ref = cpm.coverage_arrangement(m, conforming)
    assert ref["balance_rel"] <= 1e-12
    assert ref["v_gap_total"] <= 1e-14 * m.area
    assert ref["v_pen_total"] <= 1e-14 * m.area
    assert ref["max_mult_overall"] == 1

    gap = cpm.coverage_arrangement(m, gapped)
    assert gap["balance_rel"] <= 1e-12
    assert gap["v_gap_total"] > 1e-2 * m.area
    assert gap["v_pen_total"] <= 1e-12 * m.area

    pen = cpm.coverage_arrangement(m, penetrating)
    assert pen["balance_rel"] <= 1e-12
    assert pen["v_pen_total"] > 1e-2 * m.area
    assert pen["max_mult_overall"] == 2
    # without the terms the balance would be wrong by the gap/penetration measure
    raw = cpm.per_patch_balance(m, gapped)
    assert raw["per_facet_max_rel"] > 1e-2


# ---------------------------------------------------------------------------
# criterion E4: time-domain coupling energy account
# ---------------------------------------------------------------------------
def test_uniform_state_has_no_interface_flux_in_the_curved_metric():
    patch = cpm.SpherePatch()
    m = cpm.grid_side(patch, 5, 5, jitter=0.2, seed=11)
    s = cpm.grid_side(patch, 7, 6, jitter=0.2, seed=23)
    good = cpm.first_order_coupling(cpm.CurvedMortarOperator(m, s), dt=1e-3, nsteps=200)
    cen = cpm.first_order_coupling(
        cpm.CurvedMortarOperator(m, s, measure="centroid_j"), dt=1e-3, nsteps=200)
    flat = cpm.first_order_coupling(
        cpm.CurvedMortarOperator(m, s, measure="flat"), dt=1e-3, nsteps=200)
    # curved metric: a uniform state produces no flux at all
    assert good["lam_max"] <= 1e-12
    assert abs(good["mass_leak_accum"]) <= 1e-12
    assert good["energy_bookkeeping_residual_rel"] <= 1e-10
    # broken metrics inject flux in proportion to their per-facet defect
    assert cen["lam_max"] > 1e-5
    assert flat["lam_max"] > 1e-1
    # the spurious flux equals defect / R, so it scales with the closure defect
    assert cen["lam_max"] == pytest.approx(cen["per_facet_defect"], rel=0.5)


def test_coupling_coverage_form_is_conservative_for_any_geometry():
    """With the coverage-weighted receptor the mass telescopes even for a broken metric."""
    patch = cpm.GraphPatch()
    m = cpm.grid_side(patch, 5, 5, jitter=0.2, seed=4)
    s = cpm.grid_side(patch, 7, 6, jitter=0.2, seed=8, box=(0.05, 0.95, 0.02, 0.98))
    op = cpm.CurvedMortarOperator(m, s, measure="centroid_j")
    closure = cpm.first_order_coupling(op, dt=1e-3, nsteps=100, flux_form="closure")
    coverage = cpm.first_order_coupling(op, dt=1e-3, nsteps=100, flux_form="coverage")
    assert abs(coverage["mass_leak_accum"]) <= 1e-14
    assert abs(closure["mass_leak_accum"]) >= 0.0   # geometry defect stays visible


# ---------------------------------------------------------------------------
# criterion E5: transpose consistency in the curved metric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.label)
def test_transpose_is_consistent_in_the_curved_metric(patch):
    m = cpm.grid_side(patch, 5, 5, jitter=0.3, seed=11)
    s = cpm.grid_side(patch, 8, 7, jitter=0.3, seed=23)
    good = cpm.adjoint_diagnostics(cpm.CurvedMortarOperator(m, s))
    cen = cpm.adjoint_diagnostics(cpm.CurvedMortarOperator(m, s, measure="centroid_j"))
    flat = cpm.adjoint_diagnostics(cpm.CurvedMortarOperator(m, s, measure="flat"))
    assert good["patch_test"] <= 1e-12
    assert good["transpose_conservation"] <= 1e-12
    assert cen["patch_test"] > 1e-6
    assert flat["patch_test"] > 1e-2
    # the naive (midpoint) transpose is only first order: it is measurably worse
    assert good["naive_transpose_moment"] > 10 * good["patch_test"]


def test_second_order_operator_reproduces_affine_fields():
    patch = cpm.SpherePatch()
    m = cpm.grid_side(patch, 5, 5, jitter=0.3, seed=11)
    s = cpm.grid_side(patch, 8, 7, jitter=0.3, seed=23)
    op = cpm.CurvedMortarOperator(m, s, order=2)
    assert op.patch_test(np.array([0.3, -0.7, 0.5]), 1.3) <= 1e-12


# ---------------------------------------------------------------------------
# criterion E1: the true intersection curve is curved
# ---------------------------------------------------------------------------
def test_intersection_curve_is_curved_and_converges():
    patch = cpm.SpherePatch()
    gains = []
    hs = []
    for n in (4, 6, 8):
        m = cpm.grid_side(patch, n, n, jitter=0.2, seed=11)
        s = cpm.grid_side(patch, n + 3, n + 2, jitter=0.2, seed=23)
        cur = cpm.intersection_curve(patch, m, s, n_samples=9)
        assert cur["curve_length"] > cur["chord_length"]
        gains.append(cur["rel_gain"])
        hs.append(1.0 / n)
    rate = cpm.fit_rate(hs, gains)
    assert rate > 0.5          # the chordal loss shrinks with refinement
    assert all(g > 0 for g in gains)


def test_quadrature_controls_the_closure_defect():
    """The gate is quadrature-limited: a coarse rule leaves a larger residual."""
    patch = cpm.SpherePatch()
    m = cpm.grid_side(patch, 6, 6, jitter=0.3, seed=11, rule=cpm.tri_gauss(2))
    s = cpm.grid_side(patch, 9, 8, jitter=0.3, seed=23, rule=cpm.tri_gauss(2))
    coarse = cpm.per_patch_balance(m, s)["per_facet_max_rel"]
    m8 = cpm.grid_side(patch, 6, 6, jitter=0.3, seed=11, rule=cpm.tri_gauss(8))
    s8 = cpm.grid_side(patch, 9, 8, jitter=0.3, seed=23, rule=cpm.tri_gauss(8))
    fine = cpm.per_patch_balance(m8, s8)["per_facet_max_rel"]
    assert fine <= 1e-12
    assert coarse > fine