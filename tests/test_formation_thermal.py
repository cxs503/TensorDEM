"""Tests for the channel refreezing (Stefan) layer."""
from __future__ import annotations

import math

import pytest

from tensordem.formation_thermal import (ChannelEncounterConfig, ChannelRefreezing,
                                        ChannelThermalConfig, channel_thickness_profile,
                                        neumann_thickness, similarity_lambda,
                                        stefan_thickness)


def test_similarity_lambda_solves_the_transcendental_equation():
    for stefan in (0.02, 0.125, 0.5, 1.5):
        lam = similarity_lambda(stefan)
        residual = lam * math.exp(lam * lam) * math.erf(lam) - stefan / math.sqrt(math.pi)
        assert abs(residual) < 1e-10


def test_similarity_solution_limits_to_the_stefan_formula():
    # For a small Stefan number the exact solution must approach sqrt(2 k dT t / rho L).
    config = ChannelThermalConfig(specific_heat=1.0e2)  # Ste ~ 0.006
    t = 3.6e3
    stefan = stefan_thickness(t, config)
    exact = neumann_thickness(t, config)
    assert abs(stefan - exact) < 0.01 * stefan


def test_numeric_solver_matches_stefan_within_five_percent_at_three_times():
    config = ChannelThermalConfig()
    model = ChannelRefreezing(config)
    times = [600.0, 21_600.0, 86_400.0]
    numeric = model.thickness_history(times)
    for t, value in zip(times, numeric):
        reference = stefan_thickness(t, config)
        assert value > 0.0
        assert abs(value - reference) / reference < 0.05


def test_numeric_solver_tracks_the_exact_neumann_solution():
    config = ChannelThermalConfig()
    model = ChannelRefreezing(config)
    times = [3_600.0, 172_800.0, 604_800.0]
    numeric = model.thickness_history(times)
    for t, value in zip(times, numeric):
        reference = neumann_thickness(t, config)
        assert abs(value - reference) / reference < 0.03


def test_thickness_is_monotone_in_time_and_ordered_history_is_irrelevant():
    model = ChannelRefreezing(ChannelThermalConfig())
    times = [86_400.0, 600.0, 3_600.0]
    ascending = model.thickness_history(sorted(times))
    shuffled = model.thickness_history(times)
    assert shuffled == pytest.approx([ascending[2], ascending[0], ascending[1]])
    assert ascending[0] < ascending[1] < ascending[2]


def test_grid_refinement_reduces_discretisation_error():
    t = 86_400.0
    coarse = ChannelRefreezing(ChannelThermalConfig(nodes=20)).thickness(t)
    fine = ChannelRefreezing(ChannelThermalConfig(nodes=160)).thickness(t)
    reference = neumann_thickness(t, ChannelThermalConfig())
    assert abs(fine - reference) <= abs(coarse - reference) + 1e-6
    assert abs(fine - coarse) / fine < 0.05


def test_colder_air_grows_thicker_ice():
    warm = ChannelRefreezing(ChannelThermalConfig(air_temperature=-5.0)).thickness(3_600.0)
    cold = ChannelRefreezing(ChannelThermalConfig(air_temperature=-30.0)).thickness(3_600.0)
    assert cold > warm > 0.0


def test_configuration_validation():
    with pytest.raises(ValueError):
        ChannelThermalConfig(air_temperature=5.0)
    with pytest.raises(ValueError):
        ChannelThermalConfig(nodes=4)
    with pytest.raises(ValueError):
        ChannelThermalConfig(latent_heat=-1.0)
    with pytest.raises(ValueError):
        ChannelRefreezing(ChannelThermalConfig()).thickness(-1.0)
    with pytest.raises(ValueError):
        ChannelRefreezing(ChannelThermalConfig(), initial_thickness=-0.1)


def test_channel_width_shrinks_and_closure_time_is_consistent():
    model = ChannelRefreezing(ChannelThermalConfig())
    width_now = model.channel_width(0.0, 23.0)
    width_later = model.channel_width(86_400.0, 23.0)
    assert width_now == pytest.approx(23.0, abs=1e-9)
    assert 0.0 < width_later < width_now
    assert width_now - width_later == pytest.approx(2.0 * model.lateral_growth(86_400.0), rel=1e-9)


def test_initial_brash_layer_is_carried_forward():
    plain = ChannelRefreezing(ChannelThermalConfig())
    seeded = ChannelRefreezing(ChannelThermalConfig(), initial_thickness=0.2)
    assert seeded.thickness(0.0) == pytest.approx(0.2, abs=1e-3)
    assert seeded.thickness(3_600.0) > plain.thickness(3_600.0)


def test_channel_thickness_profile_shape():
    config = ChannelEncounterConfig(brash_equivalent_fraction=0.5, edge_zone_fraction=0.1,
                                    edge_thickness_amplification=1.6)
    positions = [0.0, 11.0, 13.0, 15.0, 20.0]
    profile = channel_thickness_profile(positions, parent_thickness=1.0, channel_width=23.0,
                                        refrozen_thickness=0.05, beam=20.0, config=config)
    # inside the cleared width: refrozen layer + brash equivalent
    assert profile[0] == pytest.approx(0.05 + 0.5 * 1.0)
    assert profile[1] == pytest.approx(0.05 + 0.5 * 1.0)
    # ridge ramp between |y| = 11.5 m and 13.5 m
    assert 0.55 < profile[2] < 1.6
    # undisturbed parent sheet beyond the ridge zone
    assert profile[3] == pytest.approx(1.0)
    assert profile[4] == pytest.approx(1.0)
    # The ridge bank is thicker than the parent sheet it borders (pile-up).
    assert profile[2] > profile[3]


def test_channel_thickness_profile_validation():
    with pytest.raises(ValueError):
        channel_thickness_profile([0.0], parent_thickness=-1.0, channel_width=20.0,
                                  refrozen_thickness=0.0, beam=20.0)
    with pytest.raises(ValueError):
        ChannelEncounterConfig(edge_thickness_amplification=0.5)