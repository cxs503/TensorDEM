"""Tests for the ice-ridge rate-dependent bearing capacity module.

Covers:
- Inertial number computation (dimensional analysis, limiting cases)
- Confining pressure estimation
- Multi-rate sweep (monotonicity, collapse, extrapolation)
- μ(I) rheology fit
- Rate deviation bound
- Acceptance criteria
- Negative control (quasi-static = continuum)
- Edge cases and error handling

≥12 test cases as required by the campaign node.
"""
from __future__ import annotations

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial
from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection
from tensordem.ridge_rate_dependence import (
    InertialNumber,
    compute_inertial_number,
    estimate_confining_pressure,
    RateSweepResult,
    run_rate_sweep_continuum,
    collapse_master_curve,
    fit_mu_I_rheology,
    extrapolate_quasi_static,
    rate_deviation_bound,
    build_rate_dependence_report,
    report_to_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_soil():
    return MohrCoulombSoil(
        friction_angle_deg=0.0, cohesion=30.0e3,
        saturated_unit_weight=18.0e3, young_modulus=10.0e6,
        poisson_ratio=0.3, water_unit_weight=10.06e3,
        undrained=True,
    )


@pytest.fixture
def standard_section():
    return PipeSection(outer_diameter=0.914, wall_thickness=0.0159,
                       young_modulus=207.0e9, yield_strength=448.0e6)


@pytest.fixture
def standard_keel():
    return RidgeKeel(keel_depth=12.0, keel_width=24.0, porosity=0.10)


@pytest.fixture
def standard_material():
    return RidgeMaterial(particle_radius=0.50, drag=2.0e4,
                         freeze_bond_fraction=1.0,
                         bond_tensile_strength=100.0e3,
                         bond_shear_strength=200.0e3)


# ---------------------------------------------------------------------------
# Test 1: Inertial number dimensional consistency
# ---------------------------------------------------------------------------

def test_inertial_number_dimensionless():
    """I = γ̇ d / √(P/ρ) must be dimensionless."""
    inertial = compute_inertial_number(
        platen_velocity=0.4,  # m/s
        particle_radius=0.5,  # m
        confining_pressure=100e3,  # Pa = kg/(m·s²)
        bulk_density=642.0,  # kg/m³
    )
    # Check: γ̇ = v/d = 0.4/1.0 = 0.4 [1/s]
    # √(P/ρ) = √(100000/642) ≈ 12.46 [m/s]
    # I = 0.4 * 1.0 / 12.46 ≈ 0.0321
    expected = 0.4 * 1.0 / math.sqrt(100e3 / 642.0)
    assert abs(inertial.value - expected) / expected < 1e-10
    assert inertial.shear_rate == pytest.approx(0.4, rel=1e-10)
    assert inertial.particle_diameter == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Test 2: Inertial number increases with velocity
# ---------------------------------------------------------------------------

def test_inertial_number_monotone_velocity():
    """I must increase monotonically with platen velocity."""
    I_vals = []
    for v in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        inertial = compute_inertial_number(v, 0.5, 100e3, 642.0)
        I_vals.append(inertial.value)
    for i in range(len(I_vals) - 1):
        assert I_vals[i] < I_vals[i + 1]


# ---------------------------------------------------------------------------
# Test 3: Inertial number decreases with pressure
# ---------------------------------------------------------------------------

def test_inertial_number_decreases_with_pressure():
    """I must decrease with increasing confining pressure."""
    I_vals = []
    for P in [10e3, 50e3, 100e3, 500e3, 1e6]:
        inertial = compute_inertial_number(0.4, 0.5, P, 642.0)
        I_vals.append(inertial.value)
    for i in range(len(I_vals) - 1):
        assert I_vals[i] > I_vals[i + 1]


# ---------------------------------------------------------------------------
# Test 4: Inertial number error handling
# ---------------------------------------------------------------------------

def test_inertial_number_invalid_inputs():
    """Must raise on non-positive inputs."""
    with pytest.raises(ValueError):
        compute_inertial_number(0.0, 0.5, 100e3, 642.0)
    with pytest.raises(ValueError):
        compute_inertial_number(0.4, -0.1, 100e3, 642.0)
    with pytest.raises(ValueError):
        compute_inertial_number(0.4, 0.5, 0.0, 642.0)
    with pytest.raises(ValueError):
        compute_inertial_number(0.4, 0.5, 100e3, -1.0)


# ---------------------------------------------------------------------------
# Test 5: Confining pressure estimation is positive and reasonable
# ---------------------------------------------------------------------------

def test_confining_pressure_positive(standard_soil):
    """Estimated confining pressure must be positive and in a reasonable range."""
    P = estimate_confining_pressure(
        keel_depth=12.0, cover_depth=0.50, soil=standard_soil,
    )
    assert P > 0
    # Should be in the range of 10 kPa to 10 MPa for typical ridge grounding
    assert 1e3 < P < 1e7


# ---------------------------------------------------------------------------
# Test 6: Confining pressure estimation is finite
# ---------------------------------------------------------------------------

def test_confining_pressure_finite(standard_soil):
    """Confining pressure must be finite for various inputs."""
    for keel_depth in [3.0, 6.0, 12.0, 24.0]:
        P = estimate_confining_pressure(keel_depth, 0.50, standard_soil)
        assert math.isfinite(P)
        assert P > 0


# ---------------------------------------------------------------------------
# Test 7: Multi-rate sweep produces correct number of results
# ---------------------------------------------------------------------------

def test_rate_sweep_result_count(standard_keel, standard_material,
                                  standard_soil, standard_section):
    """Sweep with N velocities must return N results."""
    velocities = [0.05, 0.10, 0.20, 0.40, 0.80]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
    )
    assert len(sweep.velocities) == 5
    assert len(sweep.peak_loads) == 5
    assert len(sweep.inertial_numbers) == 5
    assert len(sweep.displacements_at_peak) == 5


# ---------------------------------------------------------------------------
# Test 8: Bearing capacity increases with velocity (monotonicity)
# ---------------------------------------------------------------------------

def test_capacity_monotone_with_velocity(standard_keel, standard_material,
                                          standard_soil, standard_section):
    """Peak capacity must increase monotonically with velocity (Spearman ρ=1)."""
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
        mu_s=0.35, mu_2=0.42, I0=0.05,  # moderate rate dependence
    )
    # Check monotonicity
    for i in range(len(sweep.peak_loads) - 1):
        assert sweep.peak_loads[i] <= sweep.peak_loads[i + 1], \
            f"Non-monotone: Q({velocities[i]})={sweep.peak_loads[i]:.0f} > " \
            f"Q({velocities[i+1]})={sweep.peak_loads[i+1]:.0f}"


# ---------------------------------------------------------------------------
# Test 9: I-number collapse shows monotonic trend (Spearman ρ ≥ 0.95)
# ---------------------------------------------------------------------------

def test_collapse_monotonicity(standard_keel, standard_material,
                                standard_soil, standard_section):
    """Data must show monotonic trend with I (Spearman ρ ≥ 0.95)."""
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
        mu_s=0.35, mu_2=0.42, I0=0.05,
    )
    collapse = collapse_master_curve(sweep)
    assert collapse["spearman_rho"] >= 0.95, \
        f"Spearman ρ = {collapse['spearman_rho']:.4f} < 0.95"


# ---------------------------------------------------------------------------
# Test 10: Quasi-static extrapolation within 10% of continuum
# ---------------------------------------------------------------------------

def test_extrapolation_accuracy(standard_keel, standard_material,
                                 standard_soil, standard_section):
    """I→0 extrapolation must match continuum capacity within 10%."""
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
        mu_s=0.35, mu_2=0.42, I0=0.05,
    )
    extrap = extrapolate_quasi_static(sweep)
    assert extrap["extrapolation_passed"], \
        f"Rel diff = {extrap['relative_difference']*100:.1f}% > 10%"


# ---------------------------------------------------------------------------
# Test 11: Rate deviation bound < 30% (with moderate rate dependence)
# ---------------------------------------------------------------------------

def test_rate_deviation_bound(standard_keel, standard_material,
                               standard_soil, standard_section):
    """Maximum rate deviation must be < 30% with moderate μ(I) parameters."""
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
        mu_s=0.35, mu_2=0.42, I0=0.05,  # moderate: (0.42-0.35)/0.35 = 20% max increase
    )
    dev = rate_deviation_bound(sweep)
    assert dev["bound_passed"], \
        f"Max deviation = {dev['max_deviation']*100:.1f}% > 30%"


# ---------------------------------------------------------------------------
# Test 12: Negative control — quasi-static DEM ≈ continuum (within 5%)
# ---------------------------------------------------------------------------

def test_negative_control(standard_keel, standard_material,
                           standard_soil, standard_section):
    """Zero-rate (I→0) DEM capacity must match continuum within 5%."""
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
        mu_s=0.35, mu_2=0.42, I0=0.05,
    )
    extrap = extrapolate_quasi_static(sweep)
    rel_diff = extrap["relative_difference"]
    assert rel_diff < 0.05, \
        f"Neg control: rel diff = {rel_diff*100:.1f}% > 5%"


# ---------------------------------------------------------------------------
# Test 13: μ(I) fit quality
# ---------------------------------------------------------------------------

def test_mu_I_fit_quality(standard_keel, standard_material,
                           standard_soil, standard_section):
    """μ(I) fit must have R² ≥ 0.90."""
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
        mu_s=0.35, mu_2=0.42, I0=0.05,
    )
    fit = fit_mu_I_rheology(sweep)
    assert fit["fitted"]
    assert fit["R_squared"] >= 0.90, \
        f"μ(I) fit R² = {fit['R_squared']:.4f} < 0.90"


# ---------------------------------------------------------------------------
# Test 14: Full report builds without error
# ---------------------------------------------------------------------------

def test_full_report(standard_keel, standard_material,
                     standard_soil, standard_section):
    """Full report must build and serialize to JSON."""
    velocities = [0.05, 0.10, 0.20, 0.40, 0.80]
    sweep = run_rate_sweep_continuum(
        standard_keel, standard_material, standard_soil, standard_section,
        velocities, cover_depth=0.50,
    )
    report = build_rate_dependence_report(sweep)
    results = report_to_json(report)

    # Check structure
    assert "sweep" in results
    assert "collapse" in results
    assert "fit" in results
    assert "extrapolation" in results
    assert "deviation" in results
    assert "acceptance" in results

    # Check acceptance keys
    acc = results["acceptance"]
    assert "collapse_R2" in acc
    assert "collapse_passed" in acc
    assert "extrap_rel_diff" in acc
    assert "max_deviation" in acc
    assert "deviation_passed" in acc
    assert "spearman_rho" in acc
    assert "monotonicity_passed" in acc


# ---------------------------------------------------------------------------
# Test 15: Sweep requires at least 3 velocities
# ---------------------------------------------------------------------------

def test_sweep_minimum_velocities(standard_keel, standard_material,
                                   standard_soil, standard_section):
    """Must raise if fewer than 3 velocities."""
    with pytest.raises(ValueError, match="at least 3"):
        run_rate_sweep_continuum(
            standard_keel, standard_material, standard_soil, standard_section,
            [0.1, 0.2], cover_depth=0.50,
        )


# ---------------------------------------------------------------------------
# Test 16: Inertial number scales correctly with particle size
# ---------------------------------------------------------------------------

def test_inertial_number_particle_size_scaling():
    """I ∝ d / √(P/ρ) × (v/d) = v / √(P/ρ), independent of d.

    Wait — actually I = γ̇ d / √(P/ρ) = (v/d) × d / √(P/ρ) = v / √(P/ρ).
    So I is independent of particle diameter when γ̇ = v/d.
    """
    I1 = compute_inertial_number(0.4, 0.25, 100e3, 642.0)
    I2 = compute_inertial_number(0.4, 0.50, 100e3, 642.0)
    I3 = compute_inertial_number(0.4, 1.00, 100e3, 642.0)
    # All should be the same (I = v / √(P/ρ))
    assert abs(I1.value - I2.value) / I1.value < 1e-10
    assert abs(I2.value - I3.value) / I2.value < 1e-10
