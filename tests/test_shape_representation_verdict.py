"""Tests for the non-spherical representation verdict (campaign P27W2-N07).

Covers: the ellipsoid/plane Hertz closed form and its independent verification
(C1), the superellipsoid pole power law, the clump surface-error convergence
(C2 geometry half) and the verdict rules (C5).  No long DEM run is part of the
test suite -- the macroscopic channel is documented separately in
``docs/dem_nonspherical_representation_verdict.md``.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensordem.shape_representation.hertz_ellipsoid import (
    axisymmetric_indenter_law,
    ellipsoid_plane_contact,
    elliptic_terms,
    curvature_ratio_function,
    displacement_quadrature,
    pole_contact_exponent,
    superellipsoid_pole_gap,
    verify_closed_form,
    verify_elliptic_relation,
)
from tensordem.shape_representation.clump_fit import clump_convergence
from tensordem.shape_representation.verdict import assemble_verdict


# ---------------------------------------------------------------------------
# C1: ellipsoid / plane Hertz closed form
# ---------------------------------------------------------------------------

def test_sphere_limit_reproduces_hertz():
    """A = B collapses to the textbook Hertz solution (a, delta, k = 2 E* a)."""
    R, e_star, P = 10e-3, 1.0e9, 100.0
    c = ellipsoid_plane_contact(R, R, e_star, load=P)
    a_h = (3.0 * P * R / (4.0 * e_star)) ** (1.0 / 3.0)
    assert c.a_major == pytest.approx(a_h, rel=1e-10)
    assert c.a_minor == pytest.approx(a_h, rel=1e-10)
    assert c.approach == pytest.approx(a_h * a_h / R, rel=1e-10)   # delta = a^2/R
    assert c.stiffness == pytest.approx(2.0 * e_star * a_h, rel=1e-10)


def test_elliptic_terms_series_matches_scipy():
    """The small-q series used to avoid catastrophic cancellation is accurate."""
    from scipy.special import ellipk, ellipe
    for q in (1e-6, 1e-5, 9e-5):
        K, E = elliptic_terms(q)
        assert K == pytest.approx(float(ellipk(q)), rel=1e-9)
        assert E == pytest.approx(float(ellipe(q)), rel=1e-9)


def test_curvature_ratio_limits():
    assert curvature_ratio_function(0.0) == pytest.approx(1.0, abs=1e-12)
    assert curvature_ratio_function(1.0 - 1e-12) < 0.1


@pytest.mark.parametrize("ratio", [1.0, 0.6, 0.25])
def test_closed_form_satisfies_integral_equation(ratio):
    """C1: stiffness from the closed form vs the half-space integral equation."""
    c = ellipsoid_plane_contact(1.0 / ratio, 1.0, 1.0e9, load=1.0)
    v = verify_closed_form(c, n_theta=1024, n_s=20, n_ring=2, n_theta_pts=6)
    assert v.stiffness_rel_error < 1e-8          # criterion is 2 %
    assert v.flatness_spread < 1e-8              # w + g is constant inside
    assert v.approach_rel_error < 1e-8


def test_free_boundary_scan_recovers_contact_shape():
    """The contact-ellipse axis ratio is recovered from the integral equation alone."""
    out = verify_elliptic_relation(0.5, n_theta=768, n_s=14, ratios=9)
    assert out["axis_ratio_rel_error"] < 0.02
    assert out["min_y_residual_rel"] < 0.05


def test_displacement_quadrature_is_quadratic_inside():
    """The Hertz pressure field gives an exactly quadratic interior displacement."""
    w = displacement_quadrature([0.0, 0.3, 0.0], [0.0, 0.0, 0.2], 1.0, 0.5, 1.0, 1.0,
                                n_theta=1024, n_s=20)
    w2 = displacement_quadrature([0.0, 0.6, 0.0], [0.0, 0.0, 0.4], 1.0, 0.5, 1.0, 1.0,
                                 n_theta=1024, n_s=20)
    # doubling (x, y) must quadruple the quadratic part
    q1 = w[0] - w[1]
    q2 = w[0] - w[2]
    q1d = w2[0] - w2[1]
    q2d = w2[0] - w2[2]
    assert q1d / q1 == pytest.approx(4.0, rel=1e-6)
    assert q2d / q2 == pytest.approx(4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# superellipsoid pole / power-law indenter
# ---------------------------------------------------------------------------

def test_pole_gap_ellipsoid_radii():
    """m = 2 gives the exact ellipsoid pole radii R_x = a^2/c, R_y = b^2/c."""
    rec = superellipsoid_pole_gap(3e-3, 2e-3, 4e-3, 2.0)
    assert rec["pole_radius_x_m"] == pytest.approx(3e-3 ** 2 / 4e-3, rel=1e-12)
    assert rec["pole_radius_y_m"] == pytest.approx(2e-3 ** 2 / 4e-3, rel=1e-12)
    assert rec["hertzian"] is True
    assert rec["force_exponent_1_plus_1_over_m"] == pytest.approx(1.5)


def test_pole_gap_not_hertzian_for_m_above_two():
    rec = superellipsoid_pole_gap(3e-3, 3e-3, 4e-3, 8.0)
    assert rec["hertzian"] is False
    assert pole_contact_exponent(8.0) == pytest.approx(1.125)


@pytest.mark.parametrize("m", [1.0, 2.0, 4.0])
def test_axisymmetric_indenter_exponent(m):
    """Love's equation reproduces P ~ delta^(1+1/m)."""
    out = axisymmetric_indenter_law(m, n_cells=80, n_gauss=24, kappa=1.0, e_star=1.0)
    assert out["exponent_rel_error"] < 5e-3
    if m == 2.0:
        assert out["hertz_max_rel_error"] < 5e-3


# ---------------------------------------------------------------------------
# C2 (geometry): clump convergence
# ---------------------------------------------------------------------------

def test_clump_error_decreases_with_n_and_follows_half_power():
    out = clump_convergence(4e-3, 4e-3, 4e-3, 8.0, 8.0, (8, 64), n_samples=200)
    e8 = out["relative_error_covering"]["8"]
    e64 = out["relative_error_covering"]["64"]
    assert e64 < e8
    assert -0.9 < out["convergence_exponent_covering"] < -0.2


# ---------------------------------------------------------------------------
# C5: verdict artefact
# ---------------------------------------------------------------------------

def _write(dirpath, name, payload):
    with open(os.path.join(dirpath, name), "w") as fh:
        json.dump(payload, fh)


def test_verdict_is_honest_about_missing_channels(tmp_path):
    """Without measured fragments the verdict must say NOT_CERTIFIED, not PASS."""
    v = assemble_verdict(str(tmp_path))
    assert v["criteria"]["C1_ellipsoid_plane_stiffness"]["status"] == "NOT_CERTIFIED"
    assert v["criteria"]["C3_static_dissipation"]["status"] == "NOT_CERTIFIED"
    assert v["criteria"]["C4_shape_effect_repose"]["status"] == "NOT_CERTIFIED"
    assert isinstance(v["limitations"], list) and v["limitations"]


def test_verdict_marks_pass_from_measured_fragments(tmp_path):
    hertz = {
        "worst_stiffness_rel_error": 1e-12,
        "worst_flatness_spread": 1e-12,
        "sphere_limit": {"stiffness_rel_error": 1e-16},
        "power_law_indenter": [
            {"power_m": 2.0, "exponent_rel_error": 1e-12, "hertz_max_rel_error": 1e-5},
            {"power_m": 4.0, "exponent_rel_error": 1e-12},
        ],
        "superellipsoid_pole": [
            {"superellipsoid_exponent_m": 2.0, "force_exponent_1_plus_1_over_m": 1.5,
             "pole_radius_x_m": 1.0, "pole_radius_y_m": 1.0, "hertzian": True},
            {"superellipsoid_exponent_m": 8.0, "force_exponent_1_plus_1_over_m": 1.125,
             "pole_radius_x_m": 1.0, "pole_radius_y_m": 1.0, "hertzian": False},
        ],
        "rows": [],
    }
    geom = {"shapes": {"m8": {
        "rows": [{"n_spheres": 8, "covering": {"max_distance_ratio": 0.8},
                  "kmeans": {"max_distance_ratio": 0.9}},
                 {"n_spheres": 256, "covering": {"max_distance_ratio": 0.1},
                  "kmeans": {"max_distance_ratio": 0.9}}],
        "relative_error_covering": {"8": 0.8, "256": 0.1},
        "relative_error_kmeans": {"8": 0.9, "256": 0.9},
        "convergence_exponent_covering": -0.5,
        "convergence_exponent_kmeans": 0.0}}}
    diss = {"cases": {"sphere": {"passes": True, "drift_per_step": 1e-9}}}
    _write(tmp_path, "hertz_anchors.json", hertz)
    _write(tmp_path, "geom_convergence.json", geom)
    _write(tmp_path, "dissipation.json", diss)
    v = assemble_verdict(str(tmp_path))
    assert v["criteria"]["C1_ellipsoid_plane_stiffness"]["status"] == "PASS"
    assert v["criteria"]["C1b_superellipsoid_pole_law"]["status"] == "PASS"
    assert v["criteria"]["C3_static_dissipation"]["status"] == "PASS"
    # no repose case -> C2/C4 cannot be certified
    assert v["criteria"]["C2_clump_convergence"]["status"] == "NOT_CERTIFIED"
    assert v["decision_table"]["m=8"]["bulk_use_admissible_N"] == []  # 0.10 > 5%
    assert v["contact_law"]["by_exponent"]["m=8"]["pole_force_exponent"] == pytest.approx(1.125)