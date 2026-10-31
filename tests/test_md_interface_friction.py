"""Tests for the reduced MD ice/iron interface shear module."""
import math

import pytest

from tensordem.md_interface_friction import (
    MDFrictionConfig,
    film_mean_square_displacement,
    md_temperature_sweep,
    run_md_interface_shear,
)

SHORT = dict(steps=600, width=3, height=3)


def test_md_shear_returns_finite_positive_friction():
    result = run_md_interface_shear(MDFrictionConfig(temperature_C=-10.0, seed=2, **SHORT))
    assert math.isfinite(result.mu) and result.mu > 0.0
    assert result.mu < 10.0
    assert result.normal_force_nN != 0.0
    assert result.molecules > 0
    assert result.substrate_atoms > 0
    assert result.steps_averaged > 0


def test_md_thermostat_tracks_the_target_temperature():
    result = run_md_interface_shear(MDFrictionConfig(temperature_C=-10.0, seed=2, **SHORT))
    assert abs(result.kinetic_temperature_K - result.temperature_K) < 60.0


def test_md_config_validation():
    with pytest.raises(ValueError):
        MDFrictionConfig(temperature_C=3.0)
    with pytest.raises(ValueError):
        MDFrictionConfig(width=1)
    with pytest.raises(ValueError):
        MDFrictionConfig(steps=10)
    with pytest.raises(ValueError):
        MDFrictionConfig(transient_fraction=0.0)
    with pytest.raises(ValueError):
        MDFrictionConfig(lj_epsilon_iron_water_kJ=-1.0)


def test_md_corresponding_states_temperature_map():
    config = MDFrictionConfig(temperature_C=-20.0, surrogate_melting_K=546.3)
    assert config.thermostat_temperature_K == pytest.approx(546.3 * (253.15 / 273.15))
    assert MDFrictionConfig(temperature_C=-20.0).thermostat_temperature_K == pytest.approx(253.15)


def test_md_temperature_sweep_structure():
    records = md_temperature_sweep(
        [-5.0, -20.0], repeats=2, template=MDFrictionConfig(**SHORT)
    )
    assert [record["temperature_C"] for record in records] == [-5.0, -20.0]
    for record in records:
        assert record["mu_mean"] > 0.0
        assert record["mu_std"] >= 0.0
        assert len(record["runs"]) == 2
        assert record["normal_force_nN_mean"] != 0.0


def test_free_film_msd_increases_with_temperature():
    cold = film_mean_square_displacement(200.0, steps=300)
    warm = film_mean_square_displacement(400.0, steps=300)
    assert warm > cold > 0.0