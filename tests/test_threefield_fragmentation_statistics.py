"""Independent verification of the ``threefield`` fragmentation / Weibull layer.

Protocol id: ``tensordem-fragmentation-statistics-verification-v1``

The three-field report quotes fragment-size statistics obtained from
``fragmentation.fit_weibull`` / ``fit_weibull_cdf`` and grades the size
distribution against the experimental reference law ``Weibull(1.8, 0.12)``.
Nothing in the repository ever compared those estimators with an independent
reference, so the evidence grade was ``WITHHELD``. Every assertion below uses a
reference that is *not* the module under test:

* beam closed form ``sigma_b = 3 rho_w C_d u'^2 L^2 / (8 h)`` and a bisection
  root of ``sigma_b(L) = sigma_f``;
* ``scipy.stats.weibull_min`` (CDF and maximum-likelihood fit);
* ``scipy.stats.kstest`` (two-sided KS statistic);
* an externally computed cascade mass ``sum rho_i L^2 h``.

Two defects found while writing this module are pinned as regressions:
``fragment()`` used to hard-code ``rho_i = 917`` (ignoring
``BreakupProperties.ice_density_kg_m3``, which leaked 1.9 % of the inventory
through the coupled mass ledger at other densities), and
``kolmogorov_smirnov()`` used to return the plotting-position CDF deviation
rather than the KS statistic.

Run: ``python -m pytest tests/test_threefield_fragmentation_statistics.py -q``
"""
from __future__ import annotations

import math

import pytest
import scipy.stats as st
import torch

from tensordem.threefield.coupling import fragmentation_stage
from tensordem.threefield.fragmentation import (
    BreakupProperties,
    bending_stress_pa,
    cdf_agreement_r2,
    critical_floe_length,
    fit_weibull,
    fit_weibull_cdf,
    floe_volume_m3,
    fragment,
    kolmogorov_smirnov,
    kolmogorov_smirnov_plotting_position,
    weibull_cdf,
)

REFERENCE_WEIBULL = (1.8, 0.12)


def _sample(size: int, seed: int = 20261031):
    return st.weibull_min.rvs(
        REFERENCE_WEIBULL[0], scale=REFERENCE_WEIBULL[1], size=size, random_state=seed
    )


def _cascade(u_rms: float = 2.0, props: BreakupProperties | None = None, seed: int = 7):
    props = props or BreakupProperties()
    return fragment(
        [20.0, 12.0, 5.0, 1.0, 0.4], u_rms, props, generator=torch.Generator().manual_seed(seed)
    )


class TestBendingCriterion:
    """The breakup criterion against the textbook free-ended beam closed form."""

    @pytest.mark.parametrize("length,u_rms", [(0.5, 3.0), (2.0, 1.0), (5.0, 0.2), (12.0, 0.05)])
    def test_bending_stress_matches_beam_closed_form(self, length, u_rms) -> None:
        props = BreakupProperties()
        closed = (
            3.0 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2 * length**2
            / (8.0 * props.thickness_m)
        )
        measured = bending_stress_pa(length, u_rms, props)
        assert abs(measured - closed) / closed < 1e-14

    @pytest.mark.parametrize("u_rms", [0.05, 0.25, 1.0, 2.5])
    def test_critical_length_is_the_root_of_the_stress_criterion(self, u_rms) -> None:
        """Bisection root of ``sigma_b(L) - sigma_f``, coded independently."""
        props = BreakupProperties()
        low, high = 1e-6, 1e3
        for _ in range(200):
            mid = 0.5 * (low + high)
            stress = (
                3.0 * props.water_density_kg_m3 * props.drag_coefficient * u_rms**2 * mid**2
                / (8.0 * props.thickness_m)
            )
            if stress > props.flexural_strength_pa:
                high = mid
            else:
                low = mid
        root = 0.5 * (low + high)
        assert abs(critical_floe_length(u_rms, props) - root) / root < 1e-12

    def test_calm_race_has_no_critical_length(self) -> None:
        assert critical_floe_length(0.0, BreakupProperties()) == math.inf


class TestWeibullReference:
    """The Weibull CDF and the two estimators against SciPy."""

    @pytest.mark.parametrize("shape,scale", [(1.0, 0.2), (1.8, 0.12), (3.5, 2.0)])
    @pytest.mark.parametrize("x", [0.01, 0.05, 0.12, 0.5, 2.0])
    def test_cdf_matches_scipy(self, shape, scale, x) -> None:
        assert abs(weibull_cdf(x, shape, scale) - st.weibull_min.cdf(x, shape, scale=scale)) < 1e-14

    @pytest.mark.parametrize("size", [50, 200, 800])
    def test_ks_statistic_matches_scipy(self, size) -> None:
        samples = _sample(size, seed=size)
        reference = float(
            st.kstest(
                samples,
                lambda x: st.weibull_min.cdf(x, REFERENCE_WEIBULL[0], scale=REFERENCE_WEIBULL[1]),
            ).statistic
        )
        measured = kolmogorov_smirnov(list(samples), *REFERENCE_WEIBULL)
        assert abs(measured - reference) < 1e-12

    @pytest.mark.parametrize("size", [20, 100, 400])
    def test_plotting_position_deviation_is_below_the_exact_statistic(self, size) -> None:
        """The figure curve is a smoothed CDF, so it cannot exceed the KS distance."""
        samples = list(_sample(size, seed=size + 1))
        exact = kolmogorov_smirnov(samples, *REFERENCE_WEIBULL)
        plotting = kolmogorov_smirnov_plotting_position(samples, *REFERENCE_WEIBULL)
        assert 0.0 < plotting <= exact + 1e-15

    def test_ks_rejects_an_empty_sample(self) -> None:
        with pytest.raises(ValueError):
            kolmogorov_smirnov([], *REFERENCE_WEIBULL)

    def test_cdf_least_squares_fit_recovers_the_reference_law(self) -> None:
        """n = 400 census: shape within 5 %, scale within 3 % of the true law."""
        samples = list(_sample(400, seed=5))
        fit = fit_weibull_cdf(samples)
        assert abs(fit["shape"] / REFERENCE_WEIBULL[0] - 1.0) < 0.05
        assert abs(fit["scale"] / REFERENCE_WEIBULL[1] - 1.0) < 0.03
        assert fit["r_squared"] > 0.99
        # and it is the better CDF fit of the two estimators on this sample
        log_fit = fit_weibull(samples)
        assert cdf_agreement_r2(samples, fit["shape"], fit["scale"])["r_squared"] >= (
            cdf_agreement_r2(samples, log_fit["shape"], log_fit["scale"])["r_squared"]
        )

    def test_double_log_fit_is_biased_low_on_the_shape(self) -> None:
        """Known small-sample bias of the transformed-CDF least squares fit.

        The bias is bounded (< 15 % at n = 400) and one-sided (shape low), so the
        report must not quote it as an unbiased shape estimate.
        """
        samples = list(_sample(400, seed=5))
        fit = fit_weibull(samples)
        bias = fit["shape"] / REFERENCE_WEIBULL[0] - 1.0
        assert -0.15 < bias < 0.0
        assert abs(fit["scale"] / REFERENCE_WEIBULL[1] - 1.0) < 0.05


class TestCascadeConservation:
    """Mass conservation of the breakup cascade, against an external mass sum."""

    @pytest.mark.parametrize("density", [917.0, 900.0, 950.0])
    @pytest.mark.parametrize("u_rms", [1.0, 2.0, 3.0])
    def test_cascade_conserves_mass(self, density, u_rms) -> None:
        props = BreakupProperties(ice_density_kg_m3=density, thickness_m=0.10)
        result = _cascade(u_rms=u_rms, props=props)
        external = sum(density * floe_volume_m3(L, props) for L in result["lengths_m"])
        assert abs(result["final_mass_kg"] - external) / external < 1e-12

    @pytest.mark.parametrize("density", [900.0, 950.0])
    def test_cascade_uses_the_property_density(self, density) -> None:
        """Regression: the cascade used to hard-code ``rho_i = 917``.

        At ``rho_i = 900`` the returned masses were 1.85 % heavy, which the
        coupled ledger (built with the property density) then reported as a
        silent 1.9 % mass creation.
        """
        props = BreakupProperties(ice_density_kg_m3=density, thickness_m=0.10)
        result = _cascade(props=props)
        measured = result["final_mass_kg"]
        wrong = 917.0 / density * measured
        assert abs(measured / wrong - 1.0) > 1e-3  # the two conventions differ
        external = sum(density * floe_volume_m3(L, props) for L in result["lengths_m"])
        assert abs(measured - external) / external < 1e-12

    @pytest.mark.parametrize("density", [917.0, 900.0, 950.0])
    def test_coupled_ledger_closes_at_the_property_density(self, density) -> None:
        """Regression for the ledger leak measured through ``fragmentation_stage``."""
        props = BreakupProperties(ice_density_kg_m3=density, thickness_m=0.10)
        inventory = [
            {"mass_kg": density * 4.0 * props.thickness_m, "diameter_m": 2.0},
            {"mass_kg": density * 1.0 * props.thickness_m, "diameter_m": 1.0},
        ]
        stage = fragmentation_stage(inventory, 2.0, props)
        assert stage["initial_mass_kg"] > 0.0
        assert abs(stage["mass_error_kg"]) / stage["initial_mass_kg"] < 1e-12

    def test_cascade_terminates_below_the_critical_length(self) -> None:
        result = _cascade()
        assert result["split_events"] > 0
        assert max(result["lengths_m"]) <= result["critical_length_m"]
        assert min(result["lengths_m"]) > 0.0

    def test_calm_race_does_not_split(self) -> None:
        result = _cascade(u_rms=0.0)
        assert result["split_events"] == 0
        assert sorted(result["lengths_m"], reverse=True) == [20.0, 12.0, 5.0, 1.0, 0.4]


class TestSizeDistributionGate:
    """Negative control for the report's size-distribution gate."""

    @pytest.mark.parametrize("seed", [7, 11])
    def test_cascade_census_does_not_follow_the_experimental_reference(self, seed) -> None:
        """Pins the honest-FAIL: the gate fails on raw *and* normalised lengths.

        The failure is therefore a model-reference mismatch, not an artefact of
        comparing metre-scale lengths against a 0.12 m scale parameter.
        """
        lengths = _cascade(seed=seed)["lengths_m"]
        assert len(lengths) >= 3
        raw = cdf_agreement_r2(lengths, *REFERENCE_WEIBULL)["r_squared"]
        normalised = cdf_agreement_r2(
            [L / max(lengths) for L in lengths], *REFERENCE_WEIBULL
        )["r_squared"]
        assert raw < 0.0
        assert normalised < 0.0

    def test_estimator_spread_at_census_size_is_reported(self) -> None:
        """At n <= 60 the two estimators disagree by tens of percent on the shape.

        Consequence for the report: a shape quoted from a ~50-floe census carries
        an estimator spread, not a converged statistic.
        """
        lengths = _cascade(seed=7)["lengths_m"]
        log_fit = fit_weibull(lengths)
        cdf_fit = fit_weibull_cdf(lengths)
        spread = abs(cdf_fit["shape"] - log_fit["shape"]) / log_fit["shape"]
        assert spread > 0.2
        assert cdf_fit["r_squared"] >= log_fit["r_squared"]