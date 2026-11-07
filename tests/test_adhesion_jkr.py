"""Machine-checked verification of the JKR adhesive normal contact kernel.

Every assertion is against an independent anchor:

* closed-form load–radius relation vs. bracketed Brent inversion (C2),
* pull-off force vs. ``1.5 pi gamma R*`` (C1),
* zero-load radius vs. the analytic ``(6 pi gamma R*^2/K)^(1/3)`` (C2),
* strain energy release rate ``G = gamma`` (physics anchor, not a tautology),
* Hertz degeneration for ``gamma <= 1e-12`` (C4),
* separation work vs. the universal ``(3/5) pi gamma a_c^2`` and its
  ``(a_c/R*)^2`` scaling to the free-sphere surface energy (C6).
"""
from __future__ import annotations

import math

import pytest

from tensordem.adhesion_jkr import (
    GAMMA_FLOOR,
    jkr_approach,
    jkr_energy_audit,
    jkr_energy_release_rate,
    jkr_force_from_radius,
    jkr_hertz_prefactor,
    jkr_normal_force,
    jkr_pull_off_force,
    jkr_pull_off_radius,
    jkr_radius_from_force_brent,
    jkr_radius_from_force_closed,
    jkr_reduced_modulus,
    jkr_reduced_radius,
    jkr_zero_load_radius,
)
from tensordem.dem3d import hertz_normal_force

# Reference pair: two identical glass-like spheres.
R_PARTICLE = 1.0e-3      # m
E_PARTICLE = 1.1e6       # Pa (soft model material, keeps JKR valid)
NU = 0.3
GAMMA = 0.01             # J/m^2 work of adhesion


@pytest.fixture(scope="module")
def params():
    r_star = jkr_reduced_radius(R_PARTICLE, R_PARTICLE)
    e_star = jkr_reduced_modulus(E_PARTICLE, NU, E_PARTICLE, NU)
    return r_star, e_star, jkr_hertz_prefactor(e_star)


def test_reduced_radius_and_modulus_match_hertz_conventions():
    r_star = jkr_reduced_radius(R_PARTICLE, R_PARTICLE)
    assert r_star == pytest.approx(R_PARTICLE / 2.0, rel=1e-15)
    e_star = jkr_reduced_modulus(E_PARTICLE, NU, E_PARTICLE, NU)
    assert e_star == pytest.approx(E_PARTICLE / (2.0 * (1.0 - NU ** 2)), rel=1e-15)
    # a rigid wall (R2 = inf) gives R* = R1
    assert jkr_reduced_radius(R_PARTICLE, float("inf")) == R_PARTICLE


def test_pull_off_force_closed_form(params):
    """C1: F_pull-off = 1.5 pi gamma R* (exact, 0 % error)."""
    r_star, _e_star, k = params
    a_c = jkr_zero_load_radius(r_star, k, GAMMA)
    a_p = jkr_pull_off_radius(r_star, k, GAMMA)
    f_min = min(jkr_force_from_radius(a, r_star, k, GAMMA)
                for a in (a_p * 0.999999, a_p, a_p * 1.000001))
    expected = -1.5 * math.pi * GAMMA * r_star
    assert jkr_pull_off_force(r_star, GAMMA) == pytest.approx(expected, rel=1e-15)
    assert f_min == pytest.approx(expected, rel=1e-3)
    # pull-off radius = a_c / 4**(1/3)
    assert a_p == pytest.approx(a_c / 4.0 ** (1.0 / 3.0), rel=1e-12)


def test_zero_load_radius_closed_form(params):
    """C2: a(F=0) = (6 pi gamma R*^2 / K)^(1/3)."""
    r_star, _e_star, k = params
    a_c = jkr_zero_load_radius(r_star, k, GAMMA)
    a_from_force = jkr_radius_from_force_closed(0.0, r_star, k, GAMMA)
    a_from_brent = jkr_radius_from_force_brent(0.0, r_star, k, GAMMA)
    assert a_from_force == pytest.approx(a_c, rel=1e-14)
    assert a_from_brent == pytest.approx(a_c, rel=1e-10)
    # the force at a_c must vanish
    assert abs(jkr_force_from_radius(a_c, r_star, k, GAMMA)) < 1e-12 * abs(
        jkr_pull_off_force(r_star, GAMMA)
    )


@pytest.mark.parametrize("fraction", [-0.95, -0.5, 0.0, 0.5, 2.0, 10.0])
def test_brent_inversion_round_trip(params, fraction):
    """Closed form and bracketed Brent inversion are exact inverses."""
    r_star, _e_star, k = params
    f_c = jkr_pull_off_force(r_star, GAMMA)
    force = fraction * abs(f_c)
    a_brent = jkr_radius_from_force_brent(force, r_star, k, GAMMA)
    a_closed = jkr_radius_from_force_closed(force, r_star, k, GAMMA)
    assert a_brent == pytest.approx(a_closed, rel=2e-8)
    f_back = jkr_force_from_radius(a_brent, r_star, k, GAMMA)
    if force > f_c:
        assert f_back == pytest.approx(force, rel=1e-6, abs=1e-3 * abs(f_c))


def test_energy_release_rate_equals_surface_energy(params):
    """Independent physics anchor: equilibrium <=> G = gamma."""
    r_star, e_star, k = params
    for fraction in (-0.9, -0.5, 0.0, 1.0, 5.0):
        force = fraction * abs(jkr_pull_off_force(r_star, GAMMA))
        a = jkr_radius_from_force_brent(force, r_star, k, GAMMA) if force <= 0 else \
            jkr_radius_from_force_closed(force, r_star, k, GAMMA)
        g = jkr_energy_release_rate(force, a, e_star, r_star)
        assert g == pytest.approx(GAMMA, rel=1e-9)


def test_gamma_floor_degenerates_to_hertz(params):
    """C4: gamma <= 1e-12 must reproduce the legacy Hertz kernel (< 1e-9 rel)."""
    r_star, e_star, k = params
    for overlap in (1e-7, 1e-6, 1e-5, 1e-4):
        f_jkr = jkr_normal_force(overlap, r_star, e_star, 1e-12)
        f_hertz = hertz_normal_force(overlap, e_star, r_star)
        assert f_jkr == pytest.approx(float(f_hertz), rel=1e-12)
        assert f_jkr == pytest.approx(k * math.sqrt(r_star) * overlap ** 1.5, rel=1e-15)
        # zero adhesion is strictly below the floor handling
        assert jkr_normal_force(overlap, r_star, e_star, 0.0) == f_jkr


def test_adhesion_continuity_above_floor(params):
    """Crossing the gamma floor must not jump the force by more than O(1e-3)."""
    r_star, e_star, _k = params
    overlap = 1e-5
    f_floor = jkr_normal_force(overlap, r_star, e_star, GAMMA_FLOOR)
    f_just_above = jkr_normal_force(overlap, r_star, e_star, 10.0 * GAMMA_FLOOR)
    assert abs(f_just_above - f_floor) / f_floor < 1e-3


def test_tension_branch_and_pull_off(params):
    """Tensile branch is monotone and snaps off past the pull-off separation."""
    r_star, e_star, k = params
    a_p = jkr_pull_off_radius(r_star, k, GAMMA)
    delta_p = jkr_approach(a_p, r_star, k, GAMMA)
    a_c = jkr_zero_load_radius(r_star, k, GAMMA)
    delta_c = jkr_approach(a_c, r_star, k, GAMMA)
    assert delta_p < 0.0 < delta_c
    # just inside the pull-off separation the force equals the pull-off value
    f_at_pull_off = jkr_normal_force(delta_p + 0.02 * abs(delta_p), r_star, e_star, GAMMA)
    assert f_at_pull_off == pytest.approx(jkr_pull_off_force(r_star, GAMMA), rel=5e-2)
    # beyond pull-off: contact lost
    assert jkr_normal_force(delta_p * 1.05, r_star, e_star, GAMMA) == 0.0
    # compressive branch (beyond the adhesive zero-load approach) is positive
    assert jkr_normal_force(2.0 * delta_c, r_star, e_star, GAMMA) > 0.0
    # the JKR status quo at zero load is *tension*, not zero overlap
    assert delta_c < jkr_approach(jkr_zero_load_radius(r_star, k, GAMMA), r_star, k, GAMMA) + 1e-18


def test_separation_work_universal_ratio(params):
    """C6: W_sep = (3/5) pi gamma a_c^2, parameter-free."""
    r_star, _e_star, k = params
    audit = jkr_energy_audit(r_star, k, GAMMA, n_samples=12001)
    assert audit.ratio_to_contact_area == pytest.approx(0.6, rel=2e-4)
    assert audit.work_separation == pytest.approx(
        0.6 * math.pi * GAMMA * audit.a_c ** 2, rel=2e-4
    )


@pytest.mark.parametrize(
    "gamma,r_star,k",
    [
        (0.01, 1e-3, 1.4652e6),
        (0.05, 5e-4, 4.4e6),
        (0.002, 2e-3, 2.9e6),
    ],
)
def test_separation_work_ratio_is_parameter_independent(gamma, r_star, k):
    """C6: the ratio is a pure number (dimensional analysis + JKR theory)."""
    audit = jkr_energy_audit(r_star, k, gamma, n_samples=8001)
    assert audit.ratio_to_contact_area == pytest.approx(0.6, rel=2e-3)


def test_energy_audit_free_sphere_scaling(params):
    """C6 (literal form): W_sep / (4 pi gamma R*^2) = 0.15 (a_c/R*)^2 exactly."""
    r_star, _e_star, k = params
    audit = jkr_energy_audit(r_star, k, GAMMA, n_samples=12001)
    prediction = 0.15 * (audit.a_c / r_star) ** 2
    assert audit.ratio_to_free_sphere == pytest.approx(prediction, rel=1e-3)
    # the literal 4 pi gamma R^2 comparison is O(1e-4) here, NOT 1
    assert audit.ratio_to_free_sphere < 1e-3
    # implementation integrity of the quadrature
    assert audit.closure_error < 1e-5


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        jkr_reduced_radius(-1.0, 1.0)
    with pytest.raises(ValueError):
        jkr_reduced_modulus(1e6, 0.6, 1e6, 0.3)
    with pytest.raises(ValueError):
        jkr_force_from_radius(-1.0, 1e-3, 1e6, GAMMA)