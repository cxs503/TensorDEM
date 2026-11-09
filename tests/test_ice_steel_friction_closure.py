"""Acceptance tests for the ice-steel friction coverage operator + closure.

Campaign node ``p48w3-n01-ice-steel-friction``:

* **A1** coverage operator returns an area ratio in ``[0, 1]`` with an auditable
  nearest-row distance;
* **A2** the delivered ``mu(p, v, T)`` reproduces the calibration rows to < 5 %
  (interpolation branch);
* **A3** the extrapolation error drops below 15 % (it was 37.6 %);
* **A4** negative control: ``pv = 0`` degenerates to the Coulomb constant;
* **A5** this suite is green.

Every physical intermediate (Stefan melt thickness, Reynolds squeeze time,
melting onset, film area fraction) is also cross-checked against the analytic
relationship it is derived from, so a silent regression cannot hide behind the
end-to-end numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_steel_friction import (
    CALIBRATION_ANCHORS,
    EXTRAPOLATION_PROBES,
    VALIDATION_ANCHORS,
    fit_ice_steel_friction,
)
from tensordem.ice_steel_friction_closure import (
    BLEND_LENGTH,
    CONTACT_RADIUS_M,
    EXTRAPOLATION_GATE,
    ICE_ALPHA,
    ICE_CP,
    ICE_K,
    ICE_RHO,
    INTERIOR_GATE,
    LATENT_HEAT,
    REGIME_BOUNDARY,
    REGIME_MELT,
    REGIME_MIXED,
    REGIMES,
    SLIDE_DURATION_S,
    SURFACE_ROUGHNESS_M,
    V_PLATEAU,
    WATER_VISCOSITY,
    ClosureResult,
    blok_partition_ice,
    calibration_law,
    coverage,
    coverage_grid,
    film_area_fraction,
    film_shear_mu,
    friction_grid,
    friction_heat_flux,
    hull_gap,
    ice_steel_friction,
    melting_onset_time,
    melt_film_thickness,
    mu_coulomb_plateau,
    parameter_sensitivity,
    squeeze_out_time,
    validity_gap,
)
from tensordem.interface_thermal.conduction import semi_infinite_surface_temperature
from tensordem.interface_thermal.lubrication import stefan_melt_rate


def _mu(anchor, **kwargs) -> float:
    result = ice_steel_friction(anchor.v_m_per_s, anchor.temperature_c,
                                anchor.pressure_mpa, **kwargs)
    assert isinstance(result, ClosureResult)
    return result.mu


def _rel_error(model: float, reference: float) -> float:
    return abs(model - reference) / reference


# ---------------------------------------------------------------------------
# A1 -- coverage operator
# ---------------------------------------------------------------------------
def test_coverage_area_ratio_bounds_and_labels():
    rng = np.random.default_rng(20281109)
    for _ in range(200):
        v = float(10.0 ** rng.uniform(-4.0, 0.5))
        t = float(rng.uniform(-30.0, -1.0))
        p = float(10.0 ** rng.uniform(-1.3, 0.4))
        result = coverage(v, t, p)
        assert 0.0 <= result.area_ratio <= 1.0
        assert result.mode in {"interior", "boundary", "extrapolation"}
        assert result.nearest_distance >= 0.0
        assert 0.0 <= result.hull_gap
        assert result.power_density_w_per_m2 == pytest.approx(p * 1.0e6 * v)
        assert result.temperature_support >= 3


def test_coverage_nearest_row_is_auditable():
    """The nearest distance is reproducible by brute force over the anchors."""
    query = coverage(0.2, -25.0, 2.0)
    spans = query.normalisation
    log_v = math.log10(0.2)
    log_p = math.log10(2.0)
    candidates = []
    for anchor in CALIBRATION_ANCHORS:
        distance = math.sqrt(
            ((math.log10(anchor.v_m_per_s) - log_v)
             / (spans["log10_v"][1] - spans["log10_v"][0])) ** 2
            + ((math.log10(anchor.pressure_mpa) - log_p)
               / (spans["log10_p"][1] - spans["log10_p"][0])) ** 2
            + ((anchor.temperature_c + 25.0)
               / (spans["temperature_c"][1] - spans["temperature_c"][0])) ** 2
        )
        candidates.append((distance, anchor))
    distance, anchor = min(candidates, key=lambda item: item[0])
    assert query.nearest_distance == pytest.approx(distance, rel=1e-12, abs=1e-12)
    assert query.nearest_anchor == anchor.label
    assert query.nearest_source == anchor.source
    assert query.nearest_deltas["dv_m_per_s"] == pytest.approx(0.2 - anchor.v_m_per_s)
    assert query.nearest_deltas["dT_c"] == pytest.approx(-25.0 - anchor.temperature_c)
    assert query.nearest_deltas["dp_mpa"] == pytest.approx(2.0 - anchor.pressure_mpa)


def test_coverage_ranks_data_rich_and_data_poor_points():
    """Interior points are covered, extrapolation probes are not."""
    interior = coverage(0.01, -10.0, 0.2)
    corner_of_range = coverage(2.0, -2.0, 0.05)
    assert interior.area_ratio > 0.99
    assert interior.mode == "interior"
    for probe in EXTRAPOLATION_PROBES:
        result = coverage(probe.v_m_per_s, probe.temperature_c, probe.pressure_mpa)
        assert result.area_ratio <= interior.area_ratio
        assert result.mode != "interior"
    assert corner_of_range.mode == "extrapolation"


def test_coverage_grid_shapes_and_range():
    grid = coverage_grid([1e-4, 1e-2, 0.5], -10.0, [0.05, 0.5, 2.0])
    assert grid["area_ratio"].shape == (3, 3)
    assert np.all(grid["area_ratio"] >= 0.0) and np.all(grid["area_ratio"] <= 1.0)
    assert grid["mode"].shape == (3, 3)


# ---------------------------------------------------------------------------
# A2 -- interpolation branch stays inside the calibration data
# ---------------------------------------------------------------------------
def test_a2_interpolation_error_below_five_percent():
    errors = [_rel_error(_mu(anchor), anchor.mu) for anchor in CALIBRATION_ANCHORS]
    assert max(errors) < INTERIOR_GATE, (
        f"interior max error {max(errors):.2%} exceeds the 5 % gate")


def test_a2_inside_the_data_the_delivery_is_the_interpolant():
    """Inside the validity envelope the blend weight is 1 (no closure leakage)."""
    law = calibration_law()
    for anchor in CALIBRATION_ANCHORS:
        result = ice_steel_friction(anchor.v_m_per_s, anchor.temperature_c,
                                    anchor.pressure_mpa)
        assert result.blend_weight_interior > 0.99
        assert result.mu == pytest.approx(
            float(law.mu(anchor.v_m_per_s, anchor.temperature_c,
                         anchor.pressure_mpa)), rel=1e-9)


def test_a2_holdout_rows_are_reported_not_hidden():
    """Independent rows: 4 of 5 stay within the campaign gate (documented)."""
    errors = [_rel_error(_mu(anchor), anchor.mu) for anchor in VALIDATION_ANCHORS]
    assert sum(error < 0.10 for error in errors) >= 4
    assert max(errors) < 0.40, (
        "the high-power-density holdout row is the disclosed weak corner")


# ---------------------------------------------------------------------------
# A3 -- extrapolation error below 15 % (was 37.6 %)
# ---------------------------------------------------------------------------
def test_a3_extrapolation_error_below_gate():
    errors = {}
    for probe in EXTRAPOLATION_PROBES:
        errors[probe.label] = _rel_error(_mu(probe), probe.mu)
    assert max(errors.values()) < EXTRAPOLATION_GATE, errors


def test_a3_the_interpolant_alone_fails_the_same_gate():
    """The improvement is real: the pure interpolant stays at ~37.6 %."""
    law = calibration_law()
    probe = EXTRAPOLATION_PROBES[0]
    interpolant = float(law.mu(probe.v_m_per_s, probe.temperature_c,
                              probe.pressure_mpa))
    baseline = _rel_error(interpolant, probe.mu)
    assert baseline > 0.30
    assert baseline > _rel_error(_mu(probe), probe.mu)


def test_a3_probes_leave_the_validity_envelope():
    for probe in EXTRAPOLATION_PROBES:
        assert validity_gap(probe.v_m_per_s, probe.temperature_c,
                            probe.pressure_mpa)["gap"] > BLEND_LENGTH
    assert validity_gap(0.01, -10.0, 0.2)["gap"] == 0.0


# ---------------------------------------------------------------------------
# A4 -- negative control: pv = 0 gives the Coulomb constant
# ---------------------------------------------------------------------------
def test_a4_zero_power_density_is_the_coulomb_constant():
    values = [ice_steel_friction(v, -10.0, 0.2).mu
              for v in (0.0, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4)]
    assert max(values) - min(values) < 1e-12
    reference = float(mu_coulomb_plateau(0.0, -10.0, 0.2))
    assert values[0] == pytest.approx(reference, rel=1e-12)
    # the plateau is the slowest measured sweep value, i.e. a Coulomb number
    assert 0.13 < values[0] < 0.16


def test_a4_regime_is_boundary_and_film_vanishes_at_zero_power():
    result = ice_steel_friction(0.0, -10.0, 0.2)
    assert result.regime == REGIME_BOUNDARY
    assert result.film_area_fraction == pytest.approx(0.0)
    assert result.heat_flux_w_per_m2 == pytest.approx(0.0)
    assert math.isinf(result.melting_onset_s)


def test_a4_low_power_density_keeps_the_boundary_regime():
    """Low power density: no film, no closure correction, Coulomb branch only.

    The *boundary* branch is the Coulomb plateau: the delivered coefficient is
    then the calibrated Coulomb value with zero film contribution (the remaining
    speed dependence belongs to the fitted dry-friction law, not to melting).
    """
    law = calibration_law()
    for v in (1e-4, 1e-3, 1e-2):
        result = ice_steel_friction(v, -20.0, 0.1)
        assert result.regime == REGIME_BOUNDARY
        assert result.film_area_fraction == pytest.approx(0.0)
        assert result.mu == pytest.approx(
            float(law.mu(max(v, V_PLATEAU), -20.0, 0.1)), rel=1e-9)


# ---------------------------------------------------------------------------
# Physics chain: every intermediate is checked against its own analytic law
# ---------------------------------------------------------------------------
def test_blok_partition_sums_to_one_and_favours_steel():
    partition = blok_partition_ice()
    assert partition["partition_1"] + partition["partition_2"] == pytest.approx(1.0)
    assert partition["partition_1"] == pytest.approx(0.1306, abs=1e-3)


def test_friction_heat_flux_definition():
    flux = friction_heat_flux(0.08, 0.2, 2.0)
    assert float(flux) == pytest.approx(0.08 * 2.0e6 * 0.2)


def test_melting_onset_inverts_the_flash_temperature_solution():
    flux = 4000.0
    temperature = -25.0
    onset = float(melting_onset_time(flux, temperature))
    surface = float(semi_infinite_surface_temperature(
        onset, ICE_ALPHA, ICE_K, flux, temperature))
    assert surface == pytest.approx(0.0, abs=1e-9)
    # analytic closed form of the same statement
    assert onset == pytest.approx(math.pi * ICE_K ** 2 * temperature ** 2
                                  / (4.0 * ICE_ALPHA * flux ** 2), rel=1e-12)


def test_melt_thickness_follows_the_stefan_rate():
    flux = 6000.0
    temperature = -10.0
    engagement = 120.0
    film = melt_film_thickness(flux, temperature, 0.2, engagement_s=engagement,
                               pressure_mpa=1e-3, radius_m=CONTACT_RADIUS_M)
    onset = float(np.atleast_1d(film["onset_time_s"])[0])
    growth = float(np.atleast_1d(film["growth_m"])[0])
    expected_rate = stefan_melt_rate(flux, ICE_RHO, LATENT_HEAT + ICE_CP * 10.0)
    assert growth == pytest.approx(expected_rate * (engagement - onset), rel=1e-12)


def test_melt_thickness_vanishes_without_flux():
    film = melt_film_thickness(0.0, -10.0, 0.2, pressure_mpa=1.0)
    assert float(np.atleast_1d(film["thickness_m"])[0]) == pytest.approx(0.0)


def test_reynolds_steady_state_balances_melt_supply():
    """At h_ss the squeeze drain rate equals the melt supply rate."""
    flux_ice = 5000.0
    temperature = -25.0
    pressure = 2.0
    film = melt_film_thickness(flux_ice, temperature, 0.2,
                               pressure_mpa=pressure, radius_m=CONTACT_RADIUS_M)
    thickness = float(np.atleast_1d(film["thickness_m"])[0])
    drain = (2.0 * pressure * 1e6 * thickness ** 3
             / (3.0 * WATER_VISCOSITY * CONTACT_RADIUS_M ** 2))
    assert drain == pytest.approx(float(np.atleast_1d(film["melt_rate_m_per_s"])[0]),
                                  rel=1e-6)


def test_squeeze_time_scalings():
    base = float(squeeze_out_time(1e-7, 1.0))
    assert float(squeeze_out_time(1e-7, 2.0)) == pytest.approx(base / 2.0, rel=1e-12)
    thinner = float(squeeze_out_time(5e-8, 1.0))
    assert thinner == pytest.approx(base * 4.0, rel=1e-12)
    assert float(squeeze_out_time(0.0, 1.0)) == pytest.approx(0.0)


def test_film_area_fraction_is_bounded_and_monotone():
    heights = np.array([0.0, 1e-8, 1e-7, 5e-7, 1e-6, 1e-5])
    area = film_area_fraction(heights)
    assert np.all(area >= 0.0) and np.all(area <= 1.0)
    assert np.all(np.diff(area) >= 0.0)
    assert area[-1] == pytest.approx(1.0)
    assert area[1] == pytest.approx(1e-8 / SURFACE_ROUGHNESS_M)


def test_film_shear_mu_is_the_viscous_limit():
    mu = float(film_shear_mu(1.0, 2.0, 1e-7))
    assert mu == pytest.approx(WATER_VISCOSITY * 1.0 / (1e-7 * 2.0e6))


# ---------------------------------------------------------------------------
# Regime structure
# ---------------------------------------------------------------------------
def test_three_regimes_are_reachable_and_ordered():
    boundary = ice_steel_friction(1e-4, -10.0, 0.2)
    melt = ice_steel_friction(2.0, -10.0, 1.0)
    assert boundary.regime == REGIME_BOUNDARY
    assert melt.regime == REGIME_MELT
    assert melt.film_area_fraction > boundary.film_area_fraction
    assert melt.mu < boundary.mu
    mixed_found = any(
        result.regime == REGIME_MIXED
        for result in (
            ice_steel_friction(v, t, p)
            for v in (0.1, 0.2, 0.5, 1.0) for t in (-20.0, -10.0, -5.0)
            for p in (0.2, 0.5, 1.0, 2.0)))
    assert mixed_found
    assert set(REGIMES) == {REGIME_BOUNDARY, REGIME_MIXED, REGIME_MELT}


def test_mu_never_exceeds_the_dry_coulomb_value():
    for v in (0.05, 0.5, 2.0, 10.0):
        for t in (-30.0, -10.0, -2.0):
            for p in (0.05, 0.5, 2.0):
                result = ice_steel_friction(v, t, p)
                assert result.mu_physical <= result.mu_coulomb + 1e-12
                assert result.mu > 0.0


def test_mu_is_non_increasing_in_speed_above_the_plateau():
    speeds = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0])
    values = np.array([ice_steel_friction(float(v), -10.0, 2.0).mu for v in speeds])
    assert np.all(np.diff(values) <= 1e-9), values


def test_grid_evaluation_matches_scalar_evaluation():
    grid = friction_grid([1e-3, 0.1, 1.0], -15.0, [0.2, 1.0])
    for i, p in enumerate(grid["pressure_mpa"]):
        for j, v in enumerate(grid["v_m_per_s"]):
            scalar = ice_steel_friction(float(v), -15.0, float(p))
            assert grid["mu"][i, j] == pytest.approx(scalar.mu, rel=1e-12)
            assert grid["regime"][i, j] == scalar.regime


def test_parameter_sensitivity_covers_the_declared_band():
    rows = parameter_sensitivity(EXTRAPOLATION_PROBES)
    assert len(rows) == 9
    for row in rows:
        assert row["max_rel_error"] >= row["rms_rel_error"]
        assert len(row["points"]) == len(EXTRAPOLATION_PROBES)
        for point in row["points"]:
            assert point["regime"] in REGIMES


# ---------------------------------------------------------------------------
# Domain guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(v=-1.0, temperature_c=-10.0, pressure_mpa=0.2),
    dict(v=0.1, temperature_c=1.0, pressure_mpa=0.2),
    dict(v=0.1, temperature_c=-70.0, pressure_mpa=0.2),
    dict(v=0.1, temperature_c=-10.0, pressure_mpa=0.0),
    dict(v=0.1, temperature_c=-10.0, pressure_mpa=1e3),
])
def test_domain_guards(kwargs):
    with pytest.raises(ValueError):
        ice_steel_friction(**kwargs)


def test_coverage_domain_guards():
    with pytest.raises(ValueError):
        coverage(-0.1, -10.0, 0.2)
    with pytest.raises(ValueError):
        coverage(0.1, 5.0, 0.2)


def test_hull_gap_zero_for_data_and_positive_outside():
    for anchor in CALIBRATION_ANCHORS:
        assert hull_gap(anchor.v_m_per_s, anchor.temperature_c,
                        anchor.pressure_mpa) == pytest.approx(0.0)
    assert hull_gap(5.0, -45.0, 1.5) > 0.0