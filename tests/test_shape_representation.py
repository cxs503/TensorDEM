"""Acceptance tests for the P40-N10 shape-representation study.

They pin the four quantitative claims made by the node:

* the elliptical Hertz closed form reproduces the independent Boussinesq
  integral-equation solution (criterion a),
* the isotropic equivalent-radius law does **not** (the representation error the
  node quantifies),
* the inscribed N-sphere clump error decreases monotonically with N and the
  clump never leaves the body,
* the 2-D pile engine is energy-stable and the repose estimator recovers a known
  slope.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.shape_representation import (
    legendre_relation,
    PileConfig2D,
    Superquadric,
    _Pile,
    clump_surface_deviation,
    complete_elliptic_KE,
    elliptic_hertz,
    elliptic_hertz_quadrature,
    equivalent_radius_hertz,
    fit_inscribed_clump,
    repose_angle_from_pile,
)

E_STAR = 1.0e7
FORCE = 1.0


def test_complete_elliptic_integrals():
    K0, E0 = complete_elliptic_KE(0.0)
    assert K0 == pytest.approx(math.pi / 2, rel=1e-12)
    assert E0 == pytest.approx(math.pi / 2, rel=1e-12)
    K, E = complete_elliptic_KE(0.9)
    # reference values (Abramowitz & Stegun table 17.1, modulus k = 0.9)
    assert K == pytest.approx(2.2805491384227703, rel=1e-10)
    # corrected 2026-11: the pre-gate module pinned E = 2.0953194538471797, which is
    # K * (1 - sum_{n>=1} 2^{n-1} c_n^2) -- the AGM series with its leading n = 0 term
    # c_0^2/2 = k^2/2 dropped.  The tabulated value is 1.1716970527816143; two
    # independent invariants pin it: E(k) <= pi/2 and the Legendre relation below.
    assert E == pytest.approx(1.1716970527816143, rel=1e-10)
    assert E <= math.pi / 2
    for k in (0.3, 0.7, 0.9, 0.99):
        assert legendre_relation(k) == pytest.approx(math.pi / 2, rel=1e-13)


def test_circular_limit_matches_analytic_hertz():
    R = 0.05
    A = 1.0 / R
    delta_ana = (9.0 * FORCE ** 2 / (16.0 * E_STAR ** 2 * R)) ** (1.0 / 3.0)
    stiff_ana = 2.0 * E_STAR * math.sqrt(R * delta_ana)
    cf = elliptic_hertz(A, A, E_STAR, FORCE)
    q = elliptic_hertz_quadrature(A, A, E_STAR, FORCE)
    assert cf.delta == pytest.approx(delta_ana, rel=1e-6)
    assert q.delta == pytest.approx(delta_ana, rel=1e-6)
    assert cf.stiffness == pytest.approx(stiff_ana, rel=1e-6)
    assert q.stiffness == pytest.approx(stiff_ana, rel=1e-6)


@pytest.mark.parametrize("ratio", [1.5, 3.0, 8.0, 20.0])
def test_elliptic_closed_form_matches_reference_within_2pct(ratio):
    """Acceptance criterion (a): closed form vs. reference <= 2 %."""
    A = 1.0 / 0.05
    B = A / ratio
    cf = elliptic_hertz(A, B, E_STAR, FORCE)
    q = elliptic_hertz_quadrature(A, B, E_STAR, FORCE)
    assert abs(cf.stiffness / q.stiffness - 1.0) < 0.02
    assert abs(cf.delta / q.delta - 1.0) < 0.02
    assert abs(cf.a_tight / q.a_tight - 1.0) < 0.02
    # the contact ellipse is elongated in the flat-curvature direction
    assert cf.a_flat > cf.a_tight


def test_equivalent_radius_law_is_the_error_source():
    """The isotropic shortcut must deviate measurably -- that is the finding."""
    A = 1.0 / 0.05
    q8 = elliptic_hertz_quadrature(A, A / 8.0, E_STAR, FORCE)
    eq8 = equivalent_radius_hertz(A, A / 8.0, E_STAR, FORCE)
    assert abs(eq8.stiffness / q8.stiffness - 1.0) > 0.01
    q1 = elliptic_hertz_quadrature(A, A, E_STAR, FORCE)
    eq1 = equivalent_radius_hertz(A, A, E_STAR, FORCE)
    assert abs(eq1.stiffness / q1.stiffness - 1.0) < 1e-10


def test_clump_geometry_converges_with_N():
    sq = Superquadric(a=0.0194, b=0.0194, c=0.0252)
    errs = []
    for n in (8, 32, 128):
        centers, radii = fit_inscribed_clump(sq, n)
        dev = clump_surface_deviation(sq, centers, radii, n_samples=2000)
        errs.append(dev.rel_max)
        # every sphere centre lies inside the body and the union stays inside
        # within the direction-sampling tolerance of the inscribed radius
        rng = np.random.default_rng(3)
        d = rng.normal(size=(500, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        for c, r in zip(centers, radii):
            assert np.all(np.abs(c) <= np.array([sq.a, sq.b, sq.c]) * 1.001)
            pts = c + r * d
            F = ((np.abs(pts[:, 0] / sq.a) ** sq.n2
                  + np.abs(pts[:, 1] / sq.b) ** sq.n2) ** (sq.n1 / sq.n2)
                 + np.abs(pts[:, 2] / sq.c) ** sq.n1) - 1.0
            # the sampled ball surface stays inside the body up to the
            # direction-sampling tolerance of the inscribed radius (~1e-2)
            assert F.max() < 5.0e-2
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.5 * errs[0]


def test_pile_engine_is_energy_stable():
    cfg = PileConfig2D(channel="sphere", n_particles=60, radius=0.02, dt=1.2e-4,
                       n_steps=1200, n_settle=900, mu_slide=0.55, mu_roll=0.06)
    pile = _Pile(cfg)
    e0 = sum(pile.energy())
    peak = 0.0
    for _ in range(1200):
        pile.step()
        ke, pe = pile.energy()
        peak = max(peak, ke)
        assert np.all(np.isfinite(pile.pos))
    ke, pe = pile.energy()
    # no blow-up: kinetic energy must stay far below the release potential
    assert ke < 0.05 * e0
    assert peak < 0.2 * e0
    # the pile ends up resting on the floor
    assert pile.pos[:, 1].min() > 0.0


def test_repose_estimator_recovers_known_slope():
    # synthetic triangular pile with a 32 deg flank
    angle = math.radians(32.0)
    rng = np.random.default_rng(5)
    pts = []
    for _ in range(400):
        x = rng.uniform(-0.15, 0.15)
        h = 0.18 - abs(x) * math.tan(angle)
        y = rng.uniform(0.0, max(h, 0.01))
        pts.append((x, y))
    pts = np.array(pts)
    phi, spread = repose_angle_from_pile(pts, 0.02)
    assert phi == pytest.approx(32.0, abs=3.0)


def test_static_pile_pseudo_dissipation_is_small():
    cfg = PileConfig2D(channel="sphere", n_particles=60, radius=0.02, dt=1.2e-4,
                       n_steps=1600, n_settle=1200, viscous_damp=0.97)
    pile = _Pile(cfg)
    for _ in range(1200):
        pile.step()
    prev = sum(pile.energy())
    ratios = []
    for _ in range(400):
        pile.step()
        e = sum(pile.energy())
        ratios.append(abs(e - prev) / max(abs(e), 1e-30))
        prev = e
    assert float(np.mean(ratios)) < 1.0e-2  # settled quiescent window