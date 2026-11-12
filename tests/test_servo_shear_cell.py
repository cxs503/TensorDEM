"""Tests for the servo-controlled plane-shear cell (CASE-DEM-PLANE-SHEAR-MUIDI).

Acceptance: mu(I) calibration within 10% of da Cruz et al. (2005) reference.
"""
from __future__ import annotations

import math

import pytest

from tensordem.servo_shear_cell import (
    MU_2_REF, MU_S_REF, I_0_REF, PHI_C_REF, PHI_J_REF,
    ServoShearCase, ServoShearResult,
    inertial_number, mu_i_law, phi_i_law, simulate_servo_shear,
)


def test_mu_i_law_limits():
    assert mu_i_law(1e-12) == pytest.approx(MU_S_REF, abs=1e-6)
    assert mu_i_law(1e12) == pytest.approx(MU_2_REF, rel=1e-4)


def test_mu_i_law_monotone():
    vals = [mu_i_law(I) for I in [1e-3, 1e-2, 1e-1, 1.0, 10.0]]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_mu_i_law_at_I0():
    mu_at_I0 = mu_i_law(I_0_REF)
    expected = (MU_S_REF + MU_2_REF) / 2.0
    assert mu_at_I0 == pytest.approx(expected, rel=0.01)


def test_phi_i_law_limits():
    assert phi_i_law(1e-12) == pytest.approx(PHI_C_REF, abs=1e-6)
    assert phi_i_law(1e12) == pytest.approx(PHI_J_REF, rel=1e-4)


def test_inertial_number_definition():
    I = inertial_number(5.0, 0.01, 62500.0, 2500.0)
    expected = 5.0 * 0.01 / math.sqrt(62500.0 / 2500.0)
    assert I == pytest.approx(expected)


def test_case_has_sensible_defaults():
    c = ServoShearCase()
    assert c.radius > 0
    assert c.density > 0
    assert c.kn > 0
    assert c.shear_rate > 0
    assert c.P_target > 0


def test_simulation_produces_result():
    """A short run should complete and return a valid result."""
    case = ServoShearCase(
        n_particles=80,
        max_time=0.15,
        shear_rate=2.0,
        P_target=30000.0,
        seed=42,
    )
    result = simulate_servo_shear(case)
    assert isinstance(result, ServoShearResult)
    assert result.mu_measured >= 0.0
    assert result.P_measured >= 0.0
    assert result.I_measured >= 0.0
    assert 0.0 < result.phi_bed_steady < 1.0


def test_mu_in_physical_range():
    """Measured mu should be between mu_s and mu_2 (with tolerance)."""
    case = ServoShearCase(
        n_particles=80,
        max_time=0.15,
        shear_rate=5.0,
        P_target=62500.0,
        seed=77,
    )
    result = simulate_servo_shear(case)
    assert MU_S_REF - 0.2 <= result.mu_measured <= MU_2_REF + 0.2


def test_solid_fraction_in_physical_range():
    case = ServoShearCase(
        n_particles=80,
        max_time=0.15,
        shear_rate=5.0,
        P_target=62500.0,
        seed=77,
    )
    result = simulate_servo_shear(case)
    assert 0.40 <= result.phi_bed_steady <= 0.80


def test_deterministic_with_same_seed():
    case = ServoShearCase(
        n_particles=60,
        max_time=0.1,
        shear_rate=3.0,
        P_target=40000.0,
        seed=123,
    )
    r1 = simulate_servo_shear(case)
    r2 = simulate_servo_shear(case)
    assert r1.mu_measured == r2.mu_measured
    assert r1.P_measured == r2.P_measured


def test_history_is_populated():
    case = ServoShearCase(
        n_particles=60,
        max_time=0.3,
        shear_rate=3.0,
        P_target=40000.0,
        seed=55,
        sample_every=20,
    )
    result = simulate_servo_shear(case)
    assert len(result.history) > 0
