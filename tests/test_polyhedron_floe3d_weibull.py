"""Weibull weakest-link statistics: calibration, size effect and fragmentation.

The tests separate three things that are easy to conflate:

* the *estimator* (does maximum likelihood / the probability plot recover a known
  modulus and scale?);
* the *law* (does the size effect follow ``sigma_0 ~ V^{-1/m}``? does the survival
  probability of a body follow the weakest-link form?);
* the *model* (is the fragment-size distribution of a given fragmentation model
  Weibullian at all? -- measured, not assumed).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.polyhedron_floe3d.weibull import (
    fit_mle,
    fit_probability_plot,
    fragment_flaw_model,
    fragmentation_monte_carlo,
    kolmogorov_smirnov,
    probability_plot_points,
    sample_strengths,
    size_effect_study,
    weakest_link_scale,
    weakest_link_survival,
    weibull_cdf,
)


def test_weibull_cdf_and_survival_are_consistent():
    x = np.linspace(1e-3, 5.0, 100)
    cdf = weibull_cdf(x, 2.5, 1.7)
    assert np.all(np.diff(cdf) >= 0.0)
    assert abs(weibull_cdf(1.7, 1.0, 1.7) - (1.0 - math.exp(-1.0))) < 1e-12


def test_sampling_reproduces_the_cdf():
    rng = np.random.default_rng(0)
    samples = sample_strengths(4000, 3.0, 2.0e6, rng)
    for fraction in (0.25, 0.5, 0.75):
        quantile = np.quantile(samples, fraction)
        assert abs(weibull_cdf(quantile, 3.0, 2.0e6) - fraction) < 0.03


def test_maximum_likelihood_recovers_the_modulus_and_scale():
    rng = np.random.default_rng(1)
    samples = sample_strengths(2000, 3.5, 1.2e6, rng)
    fit = fit_mle(samples)
    assert abs(fit.modulus - 3.5) / 3.5 < 0.05
    assert abs(fit.scale - 1.2e6) / 1.2e6 < 0.03
    assert kolmogorov_smirnov(samples, 3.5, 1.2e6) < 0.05


def test_probability_plot_fit_recovers_the_parameters():
    rng = np.random.default_rng(2)
    samples = sample_strengths(1500, 2.0, 5.0e5, rng)
    fit = fit_probability_plot(samples)
    assert abs(fit.modulus - 2.0) / 2.0 < 0.1
    assert fit.r_squared > 0.98
    x, y = probability_plot_points(samples)
    assert x.size == y.size == 1500


def test_size_effect_slope_is_minus_one_over_modulus():
    rng = np.random.default_rng(3)
    modulus, scale = 3.5, 1.0e6
    scales = {}
    for size in (0.5, 1.0, 2.0, 4.0):
        drawn = sample_strengths(2000, modulus,
                                 weakest_link_scale(scale, size, 1.0, modulus), rng)
        scales[size] = fit_mle(drawn).scale
    effect = size_effect_study(scales)
    assert abs(effect["modulus_implied"] - modulus) / modulus < 0.1
    assert effect["r_squared"] > 0.99


def test_weakest_link_survival_matches_the_closed_form():
    # a body of volume 8 V0 has an effective strength scale sigma0 (V0/8)^{1/m}
    modulus, scale = 3.0, 1.0
    stress = 0.6 * weakest_link_scale(scale, 8.0, 1.0, modulus)
    expected = math.exp(-8.0 * (stress / scale) ** modulus)
    assert abs(weakest_link_survival(stress, modulus, scale, 8.0, 1.0) - expected) < 1e-12


def test_flaw_fragment_model_conserves_mass_exactly():
    rng = np.random.default_rng(4)
    n = 10
    spacing = 0.02
    grid = np.linspace(-0.09, 0.09, n)
    centres = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    active = np.ones(centres.shape[0], dtype=bool)
    cell_volume = spacing ** 3
    body_volume = cell_volume * active.sum()
    for level in (3.0, 6.0, 12.0):
        stress = np.full(centres.shape[0], level * 1.0e6)
        outcome = fragment_flaw_model(centres, active, stress, cell_volume,
                                      modulus=3.5, scale=1.2e6, size_ref=1.0, rng=rng)
        assert abs(float(outcome.volumes.sum()) - body_volume) / body_volume < 1e-14
        assert outcome.volumes.size >= 1


def test_flaw_fragment_sizes_are_weibull_distributed():
    """The classic result: the fragment-size probability plot is a straight line."""
    rng = np.random.default_rng(5)
    n = 12
    spacing = 0.02
    grid = np.linspace(-0.11, 0.11, n)
    centres = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    active = np.ones(centres.shape[0], dtype=bool)
    cell_volume = spacing ** 3
    stress = np.full(centres.shape[0], 7.0 * 1.2e6)
    masses = []
    for _ in range(25):
        outcome = fragment_flaw_model(centres, active, stress, cell_volume,
                                      modulus=3.5, scale=1.2e6, size_ref=1.0, rng=rng)
        masses.extend(outcome.volumes.tolist())
    fit = fit_probability_plot(np.asarray(masses))
    assert fit.r_squared > 0.95
    assert fit.modulus > 1.0


def test_mean_fragment_volume_scales_with_stress_to_the_minus_modulus():
    rng = np.random.default_rng(6)
    n = 12
    spacing = 0.02
    grid = np.linspace(-0.11, 0.11, n)
    centres = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    active = np.ones(centres.shape[0], dtype=bool)
    cell_volume = spacing ** 3
    modulus = 3.5
    levels = (3.0, 6.0, 12.0)
    means = []
    for level in levels:
        stress = np.full(centres.shape[0], level * 1.2e6)
        collected = []
        for _ in range(12):
            outcome = fragment_flaw_model(centres, active, stress, cell_volume,
                                          modulus=modulus, scale=1.2e6, size_ref=1.0,
                                          rng=rng)
            collected.extend(outcome.volumes.tolist())
        means.append(float(np.mean(collected)))
    slope = float(np.polyfit(np.log(levels), np.log(means), 1)[0])
    assert abs(slope + modulus) / modulus < 0.15


def test_percolation_model_is_not_weibullian():
    """The rejected model, kept as a measured counter-example.

    Independent cell strengths + connected components produce percolation cluster
    statistics: the probability plot of that model is clearly bent, which is why
    the flaw-influence model is the one used for the reported fragment sizes.
    """
    rng = np.random.default_rng(7)
    n = 12
    spacing = 0.02
    centres = np.stack(np.meshgrid(*(3 * [np.linspace(-0.11, 0.11, n)]),
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    active = np.ones(centres.shape[0], dtype=bool)
    cell_volume = spacing ** 3
    cell_scale = 1.1 * 1.2e6
    scale = cell_scale * (cell_volume / 1.0) ** (1.0 / 3.5)
    stress = np.where(active, 1.2e6, np.inf).reshape(n, n, n)
    masses = []
    for _ in range(20):
        outcome = fragmentation_monte_carlo((n, n, n), cell_size=spacing,
                                           stress_field=stress, modulus=3.5,
                                           scale=scale, size_ref=1.0, rng=rng,
                                           active=active)
        masses.extend(outcome.masses.tolist())
    assert masses, "the percolation model produced no fragments"
    fit = fit_probability_plot(np.asarray(masses))
    assert fit.r_squared < 0.95