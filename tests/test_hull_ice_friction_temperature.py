"""Tests for the hull-ice friction temperature calibration (law, fit, DEM)."""
import math

import pytest
import torch

from tensordem.hull_ice_friction import (
    MU_LITERATURE_HIGH,
    MU_LITERATURE_LOW,
    T_MELT_C,
    DEMFrictionConfig,
    dem_temperature_sweep,
    fit_mu_temperature,
    friction_temperature_model,
    micro_contact_friction,
    run_dem_friction_experiment,
)

SHORT = dict(steps=400, n_x=6, n_y=3, transient_fraction=0.5)


def test_micro_contact_friction_increases_as_ice_cools():
    values = [micro_contact_friction(t) for t in (-5.0, -10.0, -15.0, -20.0, -25.0)]
    assert values == sorted(values), values
    assert values[0] < values[-1]
    # reference temperature anchors the literature value exactly
    assert micro_contact_friction(-10.0) == pytest.approx(MU_LITERATURE_LOW)


def test_micro_contact_friction_validation():
    with pytest.raises(ValueError):
        micro_contact_friction(5.0)  # above the melting point
    with pytest.raises(ValueError):
        micro_contact_friction(-10.0, mu_ref=0.0)
    with pytest.raises(ValueError):
        micro_contact_friction(-10.0, exponent=0.0)
    with pytest.raises(ValueError):
        micro_contact_friction(-10.0, reference_temperature_C=1.0)
    with pytest.raises(ValueError):
        micro_contact_friction(-10.0, mu_ref=float("nan"))


def test_friction_temperature_model_shape_and_derivative_sign():
    parameters = {"mu0": 0.02, "alpha": 0.06, "beta": 0.5}
    temperatures = torch.linspace(-25.0, -1.0, 25, dtype=torch.float64)
    values = friction_temperature_model(temperatures, **parameters)
    assert values.shape == temperatures.shape
    # temperatures ascend from -25 to -1 degC, so mu must decrease along the array
    assert bool((values[:-1] > values[1:]).all()), "d mu / dT must be negative"
    scalar = friction_temperature_model(-10.0, **parameters)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(0.02 + 0.06 * math.sqrt(10.0))
    assert friction_temperature_model(T_MELT_C, **parameters) == pytest.approx(0.02)


def test_friction_temperature_model_validation():
    with pytest.raises(ValueError):
        friction_temperature_model(-10.0, mu0=0.1, alpha=-0.1, beta=0.5)
    with pytest.raises(ValueError):
        friction_temperature_model(-10.0, mu0=0.1, alpha=0.1, beta=0.0)
    with pytest.raises(ValueError):
        friction_temperature_model(-10.0, mu0=float("inf"), alpha=0.1, beta=0.5)


def test_fit_recovers_known_coefficients():
    truth = {"mu0": 0.02, "alpha": 0.062, "beta": 0.45}
    temperatures = [-5.0, -10.0, -15.0, -20.0, -25.0]
    values = [friction_temperature_model(t, **truth) for t in temperatures]
    fit = fit_mu_temperature(temperatures, values)
    assert fit["mu0"] == pytest.approx(truth["mu0"], abs=2e-3)
    assert fit["alpha"] == pytest.approx(truth["alpha"], rel=5e-2)
    assert fit["beta"] == pytest.approx(truth["beta"], abs=5e-2)
    assert fit["r_squared"] > 0.999
    assert fit["max_abs_residual"] < 1e-3
    assert fit["monotone_decreasing_in_T"] is True
    assert len(fit["mu_fitted"]) == len(temperatures)


def test_fit_validation():
    with pytest.raises(ValueError):
        fit_mu_temperature([-5.0, -10.0], [0.1, 0.15])
    with pytest.raises(ValueError):
        fit_mu_temperature([-5.0, 5.0, -20.0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        fit_mu_temperature([-5.0, -10.0, float("nan")], [0.1, 0.2, 0.3])


def test_dem_friction_experiment_returns_consistent_decomposition():
    result = run_dem_friction_experiment(DEMFrictionConfig(temperature_C=-10.0, seed=3, **SHORT))
    assert result.mu > 0.0
    assert result.mu == pytest.approx(result.mu_adhesion + result.mu_ploughing)
    assert result.mu_adhesion > result.mu_ploughing  # adhesion dominated at this pressure
    assert result.interface_mu == pytest.approx(micro_contact_friction(-10.0))
    assert result.normal_load_N > 0.0
    assert result.bonds_broken == 0
    assert result.sliding_distance_m > 0.0


def test_dem_friction_reproducible_with_seed():
    first = run_dem_friction_experiment(DEMFrictionConfig(temperature_C=-10.0, seed=7, **SHORT))
    second = run_dem_friction_experiment(DEMFrictionConfig(temperature_C=-10.0, seed=7, **SHORT))
    assert first.mu == pytest.approx(second.mu, rel=1e-12)
    third = run_dem_friction_experiment(DEMFrictionConfig(temperature_C=-10.0, seed=8, **SHORT))
    assert third.mu != first.mu


def test_dem_friction_follows_the_interface_law():
    warm = run_dem_friction_experiment(DEMFrictionConfig(temperature_C=-5.0, seed=11, **SHORT))
    cold = run_dem_friction_experiment(DEMFrictionConfig(temperature_C=-20.0, seed=11, **SHORT))
    assert cold.mu > warm.mu
    # the measured ratio tracks the imposed contact law to within a few percent
    assert warm.mu == pytest.approx(warm.interface_mu, rel=0.10)
    assert cold.mu == pytest.approx(cold.interface_mu, rel=0.10)


def test_dem_config_validation():
    with pytest.raises(ValueError):
        DEMFrictionConfig(temperature_C=5.0)
    with pytest.raises(ValueError):
        DEMFrictionConfig(n_x=2)
    with pytest.raises(ValueError):
        DEMFrictionConfig(timestep=-1.0)
    with pytest.raises(ValueError):
        DEMFrictionConfig(transient_fraction=1.5)


def test_dem_temperature_sweep_statistics():
    records = dem_temperature_sweep(
        [-5.0, -20.0], repeats=2, template=DEMFrictionConfig(**SHORT), base_seed=5
    )
    assert len(records) == 2
    for record in records:
        assert record["repeats"] == 2
        assert record["mu_mean"] > 0.0
        assert record["mu_std"] >= 0.0
        assert record["mu_min"] <= record["mu_mean"] <= record["mu_max"]
        assert len(record["runs"]) == 2
    assert records[1]["mu_mean"] > records[0]["mu_mean"]


def test_dem_temperature_sweep_validation():
    with pytest.raises(ValueError):
        dem_temperature_sweep([-5.0], repeats=0)
    with pytest.raises(ValueError):
        dem_temperature_sweep(
            [-5.0], repeats=1, template=DEMFrictionConfig(**SHORT), temperature_C=-5.1
        )