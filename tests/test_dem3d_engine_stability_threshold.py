"""Engine-level verification of the explicit time-step stability threshold.

Closes the ``【未验证】`` item in ``docs/dem3d_timestep_stability.md`` ("engine-level
threshold crossing verified dynamically") without relaxing the production guard:
the constructor still refuses ``dt > recommended_dt``, and the threshold is probed
by assigning ``model.dt`` after construction inside the verification harness.

Reference values are independent of the engine's force routine: the stiffness
matrix is assembled from the rest geometry, its largest eigenvalue comes from
``torch.linalg.eigvalsh``, and the amplification factor from the closed-form
two-by-two matrix of semi-implicit Euler.
"""
import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.engine_stability_threshold import (
    assemble_stiffness_matrix,
    exact_omega_max,
    guard_still_refuses,
    make_elastic_lattice,
    measure_growth_exponent,
    measure_onset_ratio,
    predict_critical_dt,
    refined_damping_dt,
    run_energy_history,
)
from tensordem.stability import (
    gershgorin_omega_bound,
    particle_stiffness_sums,
    symplectic_euler_growth_factor,
)


@pytest.fixture(scope="module")
def lattice():
    config = make_elastic_lattice()
    model = IceDEM3D(config)
    omega = exact_omega_max(model)
    dt_crit, elastic_limit, _ = predict_critical_dt(
        omega_max=omega, mass=config.mass, damping_coefficient=config.drag
    )
    return config, model, omega, dt_crit, elastic_limit


def test_assembled_stiffness_matrix_is_symmetric_positive_semidefinite(lattice):
    _, model, _, _, _ = lattice
    stiffness = assemble_stiffness_matrix(model)
    assert stiffness.shape == (3 * len(model.positions),) * 2
    assert torch.allclose(stiffness, stiffness.T)
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    assert float(eigenvalues.min()) > -1e-6 * float(eigenvalues.max())
    # A free-free bonded lattice carries six rigid-body modes at (numerically) zero.
    assert int((eigenvalues.abs() < 1e-6 * float(eigenvalues.max())).sum()) >= 6


def test_row_sum_bound_is_conservative_for_the_diagonal_bond_lattice(lattice):
    config, model, omega, _, _ = lattice
    sums = particle_stiffness_sums(model)
    bound = gershgorin_omega_bound(sums, config.mass, free_mask=~model.fixed)
    assert bound >= omega
    # The bound is a bound, not an estimate: it must not be tight to machine
    # precision on a 26-neighbour lattice.
    assert bound / omega > 1.0


def test_bounded_run_below_the_threshold(lattice):
    config, _, _, _, elastic_limit = lattice
    record = run_energy_history(config, dt=0.9 * elastic_limit, n_steps=600)
    assert record["diverged"] is False
    assert record["broken_bonds"] == 0
    assert record["peak_over_initial"] < 5.0


def test_divergent_run_above_the_threshold(lattice):
    config, _, _, _, elastic_limit = lattice
    record = run_energy_history(
        config, dt=1.05 * elastic_limit, n_steps=400, blowup_ratio=1.0e6
    )
    assert record["diverged"] is True
    assert record["diverged_at_step"] is not None
    # blast stays in the linear regime: no bond is ever deleted
    assert record["broken_bonds"] == 0


def test_measured_onset_matches_the_closed_form(lattice):
    config, _, _, _, elastic_limit = lattice
    onset = measure_onset_ratio(
        config,
        dt_crit=elastic_limit,
        n_steps=400,
        low=0.90,
        high=1.10,
        bisections=14,
        resolution=1.0e-3,
    )
    assert onset["probing_width"] <= 1.0e-3
    assert abs(onset["onset_ratio"] - 1.0) <= 1.0e-3


def test_growth_factor_matches_the_closed_form_spectral_radius(lattice):
    config, _, omega, _, elastic_limit = lattice
    ratio = 1.0002
    record = run_energy_history(
        config, dt=ratio * elastic_limit, n_steps=600, blowup_ratio=1.0e3
    )
    measured = measure_growth_exponent(record["history"])
    predicted = symplectic_euler_growth_factor(ratio * elastic_limit, omega, config.drag)
    assert record["diverged"] is True
    assert predicted > 1.0
    assert abs(measured - predicted) / predicted < 0.01


def test_marginal_step_grows_algebraically(lattice):
    config, _, _, _, elastic_limit = lattice
    record = run_energy_history(
        config, dt=elastic_limit, n_steps=600, blowup_ratio=math.inf
    )
    history = record["history"]
    half = len(history) // 2
    ratio = history[-1] / history[half]
    assert record["diverged"] is False
    # repeated eigenvalue at -1 -> amplitude ~ n -> energy ~ n^2 (x4 over 2x steps)
    assert abs(ratio - 4.0) < 0.15


def test_damping_bound_regime_onset_matches_refined_limit(lattice):
    config, _, omega, _, _ = lattice
    drag = 400.0
    damped = make_elastic_lattice(drag=drag)
    _, elastic_limit, damping_limit = predict_critical_dt(
        omega_max=omega, mass=config.mass, damping_coefficient=drag
    )
    assert damping_limit < elastic_limit
    refined = refined_damping_dt(
        omega_max=omega, mass=config.mass, damping_coefficient=drag
    )
    # g = 2 - d^2 / 2 makes the refined limit strictly sharper than 2 m / c
    assert refined < damping_limit
    onset = measure_onset_ratio(
        damped,
        dt_crit=damping_limit,
        n_steps=600,
        low=0.90,
        high=1.10,
        bisections=14,
        resolution=1.0e-3,
    )
    refined_ratio = refined / damping_limit
    # the engine stops diverging inside 1% of the refined closed-form limit
    assert abs(onset["onset_ratio"] - refined_ratio) < 0.01


def test_constructor_guard_still_refuses_unstable_steps(lattice):
    config, _, _, _, _ = lattice
    guard = guard_still_refuses(config)
    assert guard["refused"] is True
    assert "exceeds conservative limit" in guard["message"]
    oversized = DEM3DConfig(**{**config.__dict__, "dt": config.recommended_dt * 1.05})
    with pytest.raises(ValueError):
        IceDEM3D(oversized)


def test_engine_recommended_step_stays_far_inside_the_limit(lattice):
    config, model, _, _, elastic_limit = lattice
    assert config.recommended_dt < elastic_limit
    # the documented 0.022 cap of the merged module
    assert config.recommended_dt / elastic_limit < 0.05
    assert float(model.dt) == pytest.approx(config.recommended_dt)