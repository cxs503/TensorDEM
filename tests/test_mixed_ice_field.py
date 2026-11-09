"""Bimodal FY/MY floe-thickness distribution: sampler and quadrature.

The two properties that matter are checked against independent references: the
normal quantile against SciPy's, the lognormal parameters against sample
statistics, and the area-tilted label probability against the definition of the
MY *area* fraction.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from tensordem.mixed_ice import aging
from tensordem.mixed_ice.field import (FloeDistribution, MixedFieldSpec,
                                       inv_norm_cdf, lognormal_params,
                                       lognormal_quantile, my_distribution,
                                       my_floe_probability, quadrature_field,
                                       sample_field)


def test_inverse_normal_matches_scipy():
    q = np.array([1e-6, 1e-3, 0.02425, 0.3, 0.5, 0.7, 0.97575, 0.999, 1 - 1e-6])
    # Acklam's approximation is accurate to a relative 1.2e-9 (absolute error grows
    # with the value in the far tail)
    assert np.allclose(inv_norm_cdf(q), stats.norm.ppf(q), rtol=1e-8, atol=1e-8)
    # the tails are symmetric by construction
    assert inv_norm_cdf(0.1) == pytest.approx(-inv_norm_cdf(0.9), rel=1e-12)
    with pytest.raises(ValueError):
        inv_norm_cdf(0.0)


def test_lognormal_parameters_reproduce_mean_and_cv():
    mean, cv = 2.5, 0.4
    mu, sigma = lognormal_params(mean, cv)
    rng = np.random.default_rng(7)
    draws = rng.lognormal(mu, sigma, 400_000)
    assert draws.mean() == pytest.approx(mean, rel=5e-3)
    assert draws.std() / draws.mean() == pytest.approx(cv, rel=1e-2)
    # quantile at the median equals exp(mu)
    assert lognormal_quantile(0.5, mean, cv) == pytest.approx(math.exp(mu), rel=1e-12)


def test_floe_distribution_mean_area():
    dist = FloeDistribution(thickness_mean_m=1.0, thickness_cv=0.3,
                            diameter_mean_m=100.0, diameter_cv=0.5)
    assert dist.mean_area_m2 == pytest.approx(
        math.pi / 4.0 * 100.0 ** 2 * 1.25, rel=1e-12)


def test_my_distribution_mean_is_the_ageing_law_averaged_over_the_window():
    dist = my_distribution((2.0, 5.0))
    reference = float(np.mean(aging.class_thickness_m(np.linspace(2.0, 5.0, 256))))
    assert dist.thickness_mean_m == pytest.approx(reference, rel=1e-6)
    assert dist.label == "MY"


def test_area_tilted_label_probability_is_below_the_area_fraction():
    """MY floes are larger, so the per-floe MY probability must be < p."""
    spec = MixedFieldSpec(my_fraction=0.8)
    q = my_floe_probability(spec)
    assert 0.0 < q < 0.8
    # and the tilt is exactly the mean-area ratio
    w_fy, w_my = spec.fy.mean_area_m2, spec.my_dist.mean_area_m2
    assert q == pytest.approx(0.8 * w_fy / (0.8 * w_fy + 0.2 * w_my), rel=1e-12)
    assert my_floe_probability(MixedFieldSpec(my_fraction=0.0)) == 0.0
    assert my_floe_probability(MixedFieldSpec(my_fraction=1.0)) == 1.0


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_quadrature_realises_the_target_area_fraction_exactly(p):
    sample = quadrature_field(MixedFieldSpec(my_fraction=p))
    assert sample.realised_my_area_fraction == pytest.approx(p, abs=1e-12)
    if p < 1.0:
        assert sample.class_mean(sample.thickness_m, False) == pytest.approx(
            MixedFieldSpec(my_fraction=p).fy.thickness_mean_m, rel=2e-3)
    if p > 0.0:
        assert sample.class_mean(sample.thickness_m, True) == pytest.approx(
            MixedFieldSpec(my_fraction=p).my_dist.thickness_mean_m, rel=2e-3)


@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_monte_carlo_sampler_hits_the_target_area_fraction(p):
    sample = sample_field(MixedFieldSpec(my_fraction=p), n=200_000, seed=11)
    assert sample.realised_my_area_fraction == pytest.approx(p, abs=0.01)


def test_sampler_and_quadrature_agree_on_the_thickness_marginals():
    spec = MixedFieldSpec(my_fraction=0.4, seed=3)
    quad = quadrature_field(spec)
    mc = sample_field(spec, n=200_000, seed=3)
    for is_my in (False, True):
        q = quad.class_mean(quad.thickness_m, is_my)
        m = mc.class_mean(mc.thickness_m, is_my)
        assert m == pytest.approx(q, rel=0.03), (is_my, m, q)


def test_sampler_material_state_follows_the_ageing_law():
    spec = MixedFieldSpec(my_fraction=0.5, my_age_range=(2.0, 5.0), seed=5)
    sample = sample_field(spec, n=50_000)
    my = sample.is_my
    # MY ice is colder, less saline and stronger than first-year ice
    assert sample.class_mean(sample.temperature_c, True) < sample.class_mean(sample.temperature_c, False)
    assert sample.class_mean(sample.salinity_permille, True) < sample.class_mean(sample.salinity_permille, False)
    assert sample.class_mean(sample.compressive_strength_pa, True) > sample.class_mean(sample.compressive_strength_pa, False)
    assert np.all(sample.age_years[my] >= 2.0) and np.all(sample.age_years[my] <= 5.0)
    assert np.allclose(sample.age_years[~my], 1.0)


def test_histogram_and_summary_shapes():
    spec = MixedFieldSpec(my_fraction=0.4, seed=1)
    sample = sample_field(spec, n=5_000)
    hist = sample.as_histogram(bins=20)
    assert len(hist["bin_edges_m"]) == 21
    assert sum(hist["FY"]) + sum(hist["MY"]) == sample.n
    summary = sample.summary()
    assert summary["n"] == sample.n
    assert 0.0 <= summary["realised_my_area_fraction"] <= 1.0


def test_spec_validation():
    with pytest.raises(ValueError):
        MixedFieldSpec(my_fraction=1.5)
    with pytest.raises(ValueError):
        MixedFieldSpec(my_age_range=(0.5, 3.0))
    with pytest.raises(TypeError):
        MixedFieldSpec(seed=1.5)  # type: ignore[arg-type]


def test_spec_is_serialisable_and_hashed():
    spec = MixedFieldSpec(my_fraction=0.4)
    payload = spec.as_dict()
    assert payload["my_fraction"] == 0.4
    assert len(payload["aging_ledger_sha256"]) == 64
    assert payload["my"]["label"] == "MY"