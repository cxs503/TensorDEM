"""Geometry statistics of the multi-year ridge keel (campaign criterion (a))."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.myi_ridge import geometry


STATS = geometry.MyiRidgeStatistics()


def test_lognormal_closed_form_matches_moments():
    mu, sigma = geometry.lognormal_from_moments(16.0, 0.35)
    mean = math.exp(mu + 0.5 * sigma ** 2)
    cv = math.sqrt(math.exp(sigma ** 2) - 1.0)
    assert mean == pytest.approx(16.0, rel=1e-12)
    assert cv == pytest.approx(0.35, rel=1e-12)


def test_keel_depth_sample_mean_within_five_percent():
    depths = geometry.sample_keel_depths(STATS, 4000, seed=101)
    assert depths.mean() == pytest.approx(STATS.keel_depth_mean, rel=0.05)
    assert depths.min() >= STATS.keel_depth_bounds[0]
    assert depths.max() <= STATS.keel_depth_bounds[1]


def test_porosity_sample_inside_band():
    porosity = geometry.sample_porosity(STATS, 4000, seed=102)
    low, high = STATS.porosity_bounds
    assert porosity.min() >= low - 1e-12
    assert porosity.max() <= high + 1e-12
    assert abs(porosity.mean() - STATS.porosity_mean) < 0.01


def test_block_size_distribution_bounds_and_mean():
    blocks = geometry.sample_block_sizes(STATS, 4000, seed=103)
    low, high = STATS.block_size_bounds
    assert blocks.min() >= low - 1e-12
    assert blocks.max() <= high + 1e-12
    ensemble = geometry.ensemble_statistics(STATS, 4000, seed=103)
    assert abs(ensemble["block_diameter_mean_error"]) < 0.05


def test_sampling_is_reproducible_and_seed_dependent():
    first = geometry.sample_keel_depths(STATS, 200, seed=7)
    second = geometry.sample_keel_depths(STATS, 200, seed=7)
    other = geometry.sample_keel_depths(STATS, 200, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_keel_section_porosity_and_depth_errors():
    for depth in (10.0, 20.0, 30.0):
        for porosity in (0.20, 0.30, 0.40):
            section = geometry.build_keel_section(depth, porosity, STATS, seed=11)
            stats = geometry.section_statistics(section, STATS)
            assert abs(stats["porosity_error"]) < 0.02
            assert abs(stats["keel_depth_error"]) < 0.05
            assert stats["blocks"] >= 1


def test_keel_section_block_fraction_and_area_conservation():
    section = geometry.build_keel_section(20.0, 0.30, STATS, seed=12)
    stats = geometry.section_statistics(section, STATS)
    assert 0.5 < stats["block_area_fraction"] <= 1.0
    assert stats["solid_area"] <= section.envelope_area * (1.0 - 0.30) * 1.02


def test_envelope_geometry_relations():
    assert geometry.keel_width(20.0, STATS) == pytest.approx(
        40.0 * math.tan(math.radians(STATS.flank_angle_deg)))
    assert geometry.keel_envelope_half_width(0.0, 20.0, STATS) == pytest.approx(0.0)
    assert geometry.keel_envelope_half_width(20.0, 20.0, STATS) == pytest.approx(
        0.5 * geometry.keel_width(20.0, STATS))
    assert geometry.sail_height(20.0, STATS) == pytest.approx(
        STATS.sail_to_keel_ratio * 20.0)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        geometry.MyiRidgeStatistics(keel_depth_bounds=(30.0, 10.0))
    with pytest.raises(ValueError):
        geometry.MyiRidgeStatistics(porosity_mean=0.5)
    with pytest.raises(ValueError):
        geometry.sample_keel_depths(STATS, 0)
    with pytest.raises(ValueError):
        geometry.build_keel_section(20.0, 0.7, STATS)
    with pytest.raises(TypeError):
        geometry.keel_width("20", STATS)