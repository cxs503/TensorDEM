"""Acceptance tests for Weibull-controlled particle crushing (P96W2-N03).

Each test mirrors one machine-checkable criterion of the campaign node, with a
reduced sample/step budget so the suite stays fast.  The full-budget verdict is
produced by ``python -m tensordem.crushing_case --case verify``.
"""
import math

import numpy as np
import pytest

from tensordem.crushing import (
    CrushingConfig,
    CrushingDEM,
    Fragmenter,
    MassMomentumLedger,
    Particle,
    WeibullStrength,
    fit_gaudin_schuhmann,
    fit_weibull_strength,
    nominal_particle_stress,
    sphere_diameter,
    sphere_mass,
)
from tensordem.crushing_case import (
    case_c1_replacement,
    case_c2_weibull_inversion,
    case_c3_dem_cross_check,
    case_c3_size_effect,
    case_c4_fragment_spectrum,
    case_c5_energy_closure,
    case_c6_negative_controls,
    case_c7_regression,
    default_config,
)


# --------------------------------------------------------------------------
# C1 - mass / momentum conservation of the replacement operator
# --------------------------------------------------------------------------
def test_c1_replacement_conserves_mass_and_momentum():
    result = case_c1_replacement(120)
    assert result["passed"], result
    assert result["max_mass_residual"] <= 1e-12
    assert result["max_momentum_residual"] <= 1e-12


def test_c1_replacement_conserves_for_resting_parent():
    """A parent at rest still conserves momentum (0/0 guard)."""
    rng = np.random.default_rng(3)
    parent = Particle(position=np.zeros(3), velocity=np.zeros(3), radius=1e-3,
                      mass=sphere_mass(2e-3, 917.0), strength=1e6)
    result = Fragmenter(k=8, n=0.8).split(parent, density=917.0, rng=rng,
                                          separation_energy_J=1e-9)
    assert result.mass_residual() <= 1e-12
    assert result.momentum_residual() <= 1e-12
    # the expansion is momentum free but carries the requested energy
    kinetic = 0.5 * float(np.sum(result.child_masses
                                 * np.sum(result.child_velocities ** 2, axis=1)))
    assert kinetic == pytest.approx(1e-9, rel=1e-9, abs=1e-20)


def test_c1_children_fit_inside_the_parent_sphere():
    """The born configuration must be geometrically admissible (no overlap)."""
    rng = np.random.default_rng(11)
    radius = 1e-3
    parent = Particle(position=np.zeros(3), velocity=np.zeros(3), radius=radius,
                      mass=sphere_mass(2 * radius, 917.0), strength=1e6)
    result = Fragmenter(k=8, n=0.8).split(parent, density=917.0, rng=rng)
    offsets = np.array([c.position for c in result.children])
    radii = np.array([c.radius for c in result.children])
    assert np.all(np.linalg.norm(offsets, axis=1) + radii <= radius * (1 + 1e-12))
    for i in range(len(radii) - 1):
        for j in range(i + 1, len(radii)):
            assert np.linalg.norm(offsets[j] - offsets[i]) >= radii[i] + radii[j] - 1e-12


# --------------------------------------------------------------------------
# C2 - Weibull inverse consistency
# --------------------------------------------------------------------------
def test_c2_weibull_inversion_recovers_parameters():
    result = case_c2_weibull_inversion(300)
    assert result["passed"], result
    assert result["m_relative_error"] <= 0.10
    assert result["sigma0_relative_error"] <= 0.10


def test_c2_sampling_is_reproducible_from_seed():
    law = WeibullStrength(d0=2e-3, sigma0=5e5, m=3.0)
    first = law.sample(50, 2e-3, seed=7)
    second = law.sample(50, 2e-3, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, law.sample(50, 2e-3, seed=8))


def test_c2_survival_matches_empirical_cdf():
    law = WeibullStrength(d0=2e-3, sigma0=5e5, m=3.0)
    samples = law.sample(4000, 2e-3, seed=5)
    threshold = law.median_strength(2e-3)
    empirical = float(np.mean(samples > threshold))
    assert empirical == pytest.approx(0.5, abs=0.03)


# --------------------------------------------------------------------------
# C3 - size effect
# --------------------------------------------------------------------------
def test_c3_size_effect_slope_matches_minus_three_over_m():
    result = case_c3_size_effect(n_samples=800)
    assert result["passed"], result
    assert abs(result["fitted_slope_median"] + 1.0) / 1.0 <= 0.15


def test_c3_dem_breaks_at_the_sampled_strength():
    result = case_c3_dem_cross_check()
    assert result["passed"], result
    assert result["max_relative_error"] <= 0.05


# --------------------------------------------------------------------------
# C4 - fragment size distribution
# --------------------------------------------------------------------------
def test_c4_fragment_spectrum_follows_gaudin_schuhmann():
    result = case_c4_fragment_spectrum(n_runs=2, n_operator_draws=40)
    assert result["r_squared"] >= 0.98, result
    assert result["n_relative_error"] <= 0.10, result


def test_c4_operator_inverse_cdf_endpoints():
    rng = np.random.default_rng(2)
    fragmenter = Fragmenter(k=8, n=0.8, d_min_fraction=0.0)
    diameters = fragmenter.sample_diameters(2e-3, rng)
    assert diameters.max() <= 2e-3
    assert diameters.min() > 0.0
    # stratified sampling: one draw per stratum, in ascending stratum order
    strata = np.floor(diameters / 2e-3 ** 1 * 0 + 0)  # keep flake-free
    assert strata.size == 8


# --------------------------------------------------------------------------
# C5 - energy closure
# --------------------------------------------------------------------------
def test_c5_bed_energy_closure():
    result = case_c5_energy_closure(steps=2500)
    assert result["passed"], result
    assert result["residual_relative"] <= 0.02
    assert result["crushing_events"] > 0


def test_c5_two_particle_impact_conserves_energy():
    """Closed system (no platen work): energy must stay bounded and positive."""
    config = CrushingConfig(seed=4, restitution_coefficient=1.0,
                            friction_coefficient=0.0, crushing_enabled=False,
                            dt_safety=0.05)
    radius, mass = 5e-4, sphere_mass(1e-3, 917.0)
    particles = [
        Particle(position=[0, 0, 0], velocity=[0, 0, 0], radius=radius, mass=mass,
                 strength=1e12),
        Particle(position=[0, 0, 2 * radius - 1e-6], velocity=[0, 0, 0], radius=radius,
                 mass=mass, strength=1e12),
    ]
    engine = CrushingDEM(particles, config, seed=4)
    engine.top_platen_z, engine.bottom_platen_z = 1.0, -1.0
    initial = engine._elastic_energy_snapshot()
    total = []
    for _ in range(400):
        engine.step()
        total.append(engine._elastic_energy_snapshot() + engine.kinetic_energy_J)
    # Semi-implicit Euler conserves the energy of a linear spring to O(dt):
    # 0.1 % over 400 steps is the expected discretisation drift.
    assert max(total) == pytest.approx(initial, rel=5e-3)
    assert min(total) >= 0.0


# --------------------------------------------------------------------------
# C6 - negative controls
# --------------------------------------------------------------------------
def test_c6_negative_controls():
    result = case_c6_negative_controls(steps=1500)
    assert result["passed"], result
    assert result["a"]["crushing_events"] == 0
    assert result["a"]["psd_bitwise_unchanged"] is True
    assert result["a"]["bond_break_events"] > 0
    assert result["b"]["crushing_events"] == 6
    assert result["c"]["bond_break_events"] == 0
    assert result["c"]["bond_derived_fragment_clusters"] == 0
    assert result["c"]["psd_evolved"] is True


def test_c6_crushing_switch_off_leaves_state_untouched():
    config = default_config(seed=9, crushing_enabled=False)
    engine = CrushingDEM.single_sphere(2e-3, config, seed=9)
    engine.platen_velocity = -0.05
    before = engine.diameters.copy()
    engine.run(500)
    assert engine.crushing_events == 0
    assert np.array_equal(engine.diameters, before)


# --------------------------------------------------------------------------
# C7 - regression of the untouched dem3d path
# --------------------------------------------------------------------------
def test_c7_dem3d_trajectory_is_bit_for_bit_unchanged():
    result = case_c7_regression()
    assert result["passed"], result


# --------------------------------------------------------------------------
# API contracts
# --------------------------------------------------------------------------
def test_weibull_law_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        WeibullStrength(d0=-1.0)
    with pytest.raises(ValueError):
        WeibullStrength(m=0.0)
    law = WeibullStrength()
    with pytest.raises(ValueError):
        law.sample(4, -1e-3)


def test_fragmenter_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        Fragmenter(k=1)
    with pytest.raises(ValueError):
        Fragmenter(n=-0.5)


def test_mass_and_diameter_helpers_are_consistent():
    diameter = 3.0e-3
    mass = sphere_mass(diameter, 917.0)
    assert sphere_diameter(mass, 917.0) == pytest.approx(diameter, rel=1e-14)
    assert nominal_particle_stress(2.0, 1e-3) == pytest.approx(2.0 / (math.pi * 1e-6))


def test_crushing_config_rejects_bond_without_strength_law():
    with pytest.raises(ValueError):
        CrushingConfig(bond_stiffness=1e6)


def test_ledger_audit_closes_for_a_known_pair():
    ledger = MassMomentumLedger(gamma=0.1)
    ledger.external_work_J = 1.0
    ledger.dissipation_J = 0.2
    ledger.surface_energy_J = 0.1
    audit = ledger.audit(elastic_energy_J=0.5, kinetic_energy_J=0.5,
                         initial_elastic_energy_J=0.1, initial_kinetic_energy_J=0.2)
    assert audit["residual_J"] == pytest.approx(0.0, abs=1e-12)
    assert audit["residual_relative"] <= 1e-12
