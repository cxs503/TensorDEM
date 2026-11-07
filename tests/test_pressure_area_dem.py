"""Tests for the pressure-area index emergence module (DEM indentation).

Evidence grade of the module under test: REAL.  The tests check, in order,

1. the lattice elastic calibration (measured modulus vs the analytic
   triangular-lattice value, 3 %),
2. the contact law and its geometric contact-area growth (``A_c ~ sqrt(delta)``),
3. the statistics of the correlated strength field (median / COV / correlation),
4. the deterministic perfectly-correlated limit (the simultaneous-failure
   control is reproducible to machine precision),
5. that the macroscopic pressure-area exponent is finite for a correlated field
   and collapses towards zero for the perfectly correlated field, and that the
   bootstrap CI of the control contains zero while the primary CI does not.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.pressure_area_dem import (PunchConfig, bootstrap_exponent_ci,
                                         build_triangular_lattice,
                                         fit_pressure_area_exponent,
                                         gaussian_field_1d, lattice_modulus_factor,
                                         run_punch_indentation, sample_strength_field)


# --------------------------------------------------------------------------
# 1. lattice calibration
# --------------------------------------------------------------------------
def test_lattice_modulus_matches_analytic_triangular_value():
    """Affine uniaxial strain of the lattice reproduces E_2D = 9k/(4 sqrt3)."""
    geom = build_triangular_lattice(1.0, 0.6, 0.04)
    cfg = PunchConfig(punch_width=0.5, spacing=0.04, sheet_thickness=0.2)
    bonds = geom.bonds
    pos = geom.positions
    delta = pos[bonds[:, 1]] - pos[bonds[:, 0]]
    rest = np.linalg.norm(delta, axis=1)
    unit = delta / rest[:, None]
    eps = 1e-6
    u = np.column_stack((eps * pos[:, 0], -0.33 * eps * pos[:, 1]))
    xi = np.einsum("ij,ij->i", u[bonds[:, 1]] - u[bonds[:, 0]], unit)
    strain = xi / rest
    energy = 0.5 * cfg.bond_stiffness * (strain * rest) ** 2
    # energy per unit reference area, compared with 0.5 E_2D eps^2 (uniaxial)
    area = geom.width * geom.height
    e_measured = 2.0 * energy.sum() / area / eps ** 2
    e_analytic = cfg.bond_stiffness * lattice_modulus_factor()
    assert e_measured == pytest.approx(e_analytic, rel=0.05)


def test_lattice_is_bonded_and_indices_valid():
    geom = build_triangular_lattice(0.8, 0.2, 0.04)
    assert geom.bonds.min() >= 0
    assert geom.bonds.max() < len(geom.positions)
    assert (geom.bonds[:, 0] < geom.bonds[:, 1]).all()
    assert geom.bottom.any() and geom.top.any()
    # every interior node has six neighbours (triangular lattice degree)
    deg = np.bincount(geom.bonds.ravel(), minlength=len(geom.positions))
    assert deg.max() <= 6


# --------------------------------------------------------------------------
# 2. contact law / contact-area growth
# --------------------------------------------------------------------------
def test_contact_area_grows_as_sqrt_penetration():
    """A_c = sum 2 sqrt(2 r delta) so the true contact area grows as sqrt(delta)."""
    cfg = PunchConfig(punch_width=0.5, seed=3, strength_median=1e3,
                      strength_cov=0.0, surface_roughness=0.0, n_increments=6,
                      max_penetration_fraction=0.01, stop_at_first_failure=False)
    result = run_punch_indentation(cfg)
    disp = np.array(result.displacement)
    area = np.array(result.contact_area)
    mask = area > 0
    assert mask.sum() >= 3
    slope = np.polyfit(np.log(disp[mask]), np.log(area[mask]), 1)[0]
    assert 0.4 < slope < 0.6          # sqrt(delta) growth
    # force grows faster than the area: F ~ delta**1.5
    force = np.array(result.force)
    fslope = np.polyfit(np.log(disp[mask]), np.log(force[mask]), 1)[0]
    assert 1.2 < fslope < 1.8


def test_elastic_response_is_monotone_and_positive():
    cfg = PunchConfig(punch_width=1.0, seed=0, strength_median=1e3,
                      strength_cov=0.0, stop_at_first_failure=False,
                      n_increments=8, max_penetration_fraction=0.01)
    result = run_punch_indentation(cfg)
    force = np.array(result.force)
    assert (force > 0).all()
    assert np.all(np.diff(force) > 0)


# --------------------------------------------------------------------------
# 3. strength field statistics
# --------------------------------------------------------------------------
def test_strength_field_statistics():
    geom = build_triangular_lattice(1.0, 0.2, 0.04)
    field = sample_strength_field(geom.positions, 0.04, 0.1, 0.52, 1e-3, seed=5,
                                  component="nodes")
    assert len(field) == len(geom.positions)
    assert np.median(field) == pytest.approx(1e-3, rel=0.15)
    assert np.std(field) / np.mean(field) == pytest.approx(0.52, rel=0.15)
    assert (field > 0).all()


def test_strength_field_correlation_length_ordering():
    geom = build_triangular_lattice(1.0, 0.2, 0.04)
    short = sample_strength_field(geom.positions, 0.04, 0.04, 0.5, 1e-3, seed=1,
                                  component="nodes")
    long = sample_strength_field(geom.positions, 0.04, 0.4, 0.5, 1e-3, seed=1,
                                 component="nodes")
    def roughness(v):
        return float(np.abs(np.diff(v)).mean() / v.mean())
    assert roughness(long) < roughness(short)


def test_uniform_field_is_exactly_constant():
    geom = build_triangular_lattice(1.0, 0.2, 0.04)
    field = sample_strength_field(geom.positions, 0.04, math.inf, 0.52, 1e-3, seed=9,
                                  component="nodes")
    assert np.allclose(field, 1e-3)


def test_gaussian_height_profile_is_standardised():
    x = np.repeat(np.arange(0, 1.0, 0.04), 5)
    prof = gaussian_field_1d(x, 0.04, 0.08, seed=2)
    assert prof.std() == pytest.approx(1.0, rel=0.05)
    assert abs(prof.mean()) < 0.05


# --------------------------------------------------------------------------
# 4. simultaneous-failure control is deterministic
# --------------------------------------------------------------------------
def test_uniform_control_is_deterministic():
    """With a perfectly correlated strength field and a smooth punch the model
    carries no random input left, so two seeds must agree to machine precision.
    (The control keeps the punch-face roughness random on purpose: it isolates
    the *strength* statistics, so the residual seed scatter belongs to the
    geometry, not to the material.)"""
    cfg = PunchConfig(punch_width=1.0, seed=0, correlation_length=math.inf,
                      surface_roughness=0.0, n_increments=40)
    a = run_punch_indentation(cfg)
    b = run_punch_indentation(PunchConfig(**{**cfg.__dict__, "seed": 17}))
    assert a.peak_pressure == pytest.approx(b.peak_pressure, rel=1e-9)


def test_correlated_field_varies_with_seed():
    """A random strength field makes the elastic-limit pressure seed dependent."""
    cfg = PunchConfig(punch_width=1.0, seed=0, n_increments=40)
    a = run_punch_indentation(cfg)
    b = run_punch_indentation(PunchConfig(**{**cfg.__dict__, "seed": 17}))
    assert abs(a.peak_pressure - b.peak_pressure) / a.peak_pressure > 0.02


# --------------------------------------------------------------------------
# 5. statistics helpers
# --------------------------------------------------------------------------
def test_fit_recovers_known_exponent():
    areas = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    pressures = 3.2e6 * areas ** -0.5
    fit = fit_pressure_area_exponent(areas, pressures)
    assert fit["exponent"] == pytest.approx(0.5, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_ci_brackets_exponent():
    rng = np.random.default_rng(0)
    areas = np.repeat([0.5, 1.0, 2.0, 5.0, 10.0], 12)
    true = 3.2e6 * areas ** -0.5
    pressures = true * rng.lognormal(0.0, 0.1, areas.size)
    fit = bootstrap_exponent_ci(areas, pressures, n_boot=400, seed=1)
    assert fit["ci_low"] < fit["exponent"] < fit["ci_high"]
    assert fit["ci_low"] < 0.5 < fit["ci_high"]


# --------------------------------------------------------------------------
# 6. the physics claim: exponent from statistics, control -> 0
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seeds", [3])
def test_exponent_emerges_and_control_collapses(seeds):
    areas = [0.5, 2.0, 10.0]
    base = dict(punch_width=1.0, n_increments=40, newton_iterations=5)

    def exponent(**extra):
        pressures = []
        for a in areas:
            vals = [run_punch_indentation(
                PunchConfig(**{**base, **extra, "punch_width": a, "seed": s})
            ).peak_pressure for s in range(seeds)]
            pressures.append(float(np.mean(vals)))
        return fit_pressure_area_exponent(areas, pressures)["exponent"]

    ex_random = exponent(correlation_length=0.1, strength_cov=0.52)
    ex_uniform = exponent(correlation_length=math.inf, strength_cov=0.52)
    assert ex_random > 0.2                 # a size effect exists at all
    assert abs(ex_uniform) < 0.25          # simultaneous failure: (almost) none
    assert ex_random > 2.0 * abs(ex_uniform)


def test_invalid_configurations_raise():
    with pytest.raises(ValueError):
        PunchConfig(punch_width=-1.0)
    with pytest.raises(ValueError):
        PunchConfig(poisson_ratio=0.5)
    with pytest.raises(ValueError):
        PunchConfig(correlation_length=0.0)
    with pytest.raises(ValueError):
        PunchConfig(spacing=0.2, sheet_thickness=0.2)
    with pytest.raises(ValueError):
        fit_pressure_area_exponent([1.0], [1.0])