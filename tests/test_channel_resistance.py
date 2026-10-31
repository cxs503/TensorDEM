"""Tests for the Riska channel-resistance calibration, convoy optimiser and API."""
from __future__ import annotations

import math

import pytest

from tensordem.channel_resistance import (
    DEFAULT_REFERENCE,
    RiskaReference,
    calibrate_riska_coefficients,
    convoy_energy_metre,
    handle_channel_resistance,
    optimal_escort_spacing,
    riska_channel_resistance,
    spearman_rank_correlation,
)

REF = DEFAULT_REFERENCE


def synthetic_measurements(
    coefficient: float = 0.05,
    p: float = 1.0,
    a: float = 1.5,
    beam: float = 4.0,
    sigma_f: float = 0.5e6,
) -> list[dict]:
    rows = []
    for concentration in (0.3, 0.6):
        for thickness in (0.5, 1.0, 1.5):
            resistance = riska_channel_resistance(
                beam_m=beam,
                thickness_m=thickness,
                concentration=concentration,
                velocity_ms=1.0,
                channel_coefficient=coefficient,
                concentration_exponent=p,
                thickness_exponent=a,
                reference=RiskaReference(sigma_f_pa=sigma_f),
            )
            rows.append(
                {
                    "concentration": concentration,
                    "thickness_m": thickness,
                    "beam_m": beam,
                    "velocity_ms": 1.0,
                    "resistance_n": resistance,
                }
            )
    return rows


# ------------------------------------------------------------------ Riska formula
def test_riska_formula_scales_linearly_with_beam_and_thickness() -> None:
    base = dict(
        thickness_m=1.0, concentration=0.6, velocity_ms=1.0,
        channel_coefficient=0.05, thickness_exponent=1.0,
    )
    r1 = riska_channel_resistance(beam_m=4.0, **base)
    r2 = riska_channel_resistance(beam_m=8.0, **base)
    assert r2 == pytest.approx(2.0 * r1, rel=1e-12)
    # Explicit ``B h sigma_f`` factor: linear in h when the shape exponent is 0.
    t_base = dict(
        beam_m=4.0, concentration=0.6, velocity_ms=1.0,
        channel_coefficient=0.05, thickness_exponent=0.0,
    )
    t1 = riska_channel_resistance(thickness_m=1.0, **t_base)
    t2 = riska_channel_resistance(thickness_m=2.0, **t_base)
    assert t2 == pytest.approx(2.0 * t1, rel=1e-12)
    # Shape exponent acts on (h/h_ref)^a on top of the explicit factor.
    s_base = dict(t_base, thickness_exponent=1.5)
    s1 = riska_channel_resistance(thickness_m=1.0, **s_base)
    s2 = riska_channel_resistance(thickness_m=2.0, **s_base)
    assert s2 / s1 == pytest.approx(2.0 * 2.0**1.5, rel=1e-12)


def test_riska_formula_velocity_term_and_validation() -> None:
    slow = riska_channel_resistance(
        beam_m=4.0, thickness_m=1.0, concentration=0.6, velocity_ms=0.0,
        channel_coefficient=0.05, velocity_coefficient=0.1,
    )
    fast = riska_channel_resistance(
        beam_m=4.0, thickness_m=1.0, concentration=0.6, velocity_ms=2.0,
        channel_coefficient=0.05, velocity_coefficient=0.1,
    )
    assert fast == pytest.approx(1.2 * slow, rel=1e-12)
    with pytest.raises(ValueError):
        riska_channel_resistance(
            beam_m=0.0, thickness_m=1.0, concentration=0.6, velocity_ms=1.0,
            channel_coefficient=0.05,
        )
    with pytest.raises(ValueError):
        riska_channel_resistance(
            beam_m=4.0, thickness_m=1.0, concentration=0.6, velocity_ms=1.0,
            channel_coefficient=-1.0,
        )


# --------------------------------------------------------------------- calibration
def test_calibration_recovers_synthetic_coefficients() -> None:
    rows = synthetic_measurements(coefficient=0.042, p=0.9, a=1.6)
    calibration = calibrate_riska_coefficients(rows)
    assert calibration.channel_coefficient == pytest.approx(0.042, rel=2e-3)
    assert calibration.concentration_exponent == pytest.approx(0.9, rel=2e-3)
    assert calibration.thickness_exponent == pytest.approx(1.6, rel=2e-3)
    assert calibration.max_relative_error < 1e-6


def test_calibration_leave_one_out_is_stable_on_synthetic_data() -> None:
    rows = synthetic_measurements(coefficient=0.05, p=1.0, a=1.25)
    calibration = calibrate_riska_coefficients(rows)
    assert calibration.leave_one_out_max_relative_error < 1e-6


def test_calibration_rejects_zero_or_short_input() -> None:
    rows = synthetic_measurements()
    with pytest.raises(ValueError):
        calibrate_riska_coefficients(rows[:2])
    bad = [dict(rows[0], resistance_n=0.0)] + rows[1:]
    with pytest.raises(ValueError):
        calibrate_riska_coefficients(bad)


def test_calibration_reports_per_case_relative_errors() -> None:
    rows = synthetic_measurements(coefficient=0.03, p=0.8, a=1.4)
    calibration = calibrate_riska_coefficients(rows)
    assert len(calibration.cases) == len(rows)
    for case in calibration.cases:
        assert case["predicted_n"] > 0.0
        assert case["relative_error"] == pytest.approx(
            abs(case["predicted_n"] - case["resistance_n"]) / case["resistance_n"]
        )


def test_spearman_correlation_extremes_and_midranks() -> None:
    assert spearman_rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    rho = spearman_rank_correlation([1, 2, 2, 3], [1, 2, 3, 4])
    assert 0.8 < rho < 1.0
    with pytest.raises(ValueError):
        spearman_rank_correlation([1, 2], [1, 2, 3])


# ------------------------------------------------------------------------- convoy
def convoy_kwargs(**overrides) -> dict:
    params = dict(
        ship_length_m=2.355,
        lead_resistance_n=85_000.0,
        escort_channel_resistance_n=85_000.0,
        n_escort=2,
        consolidation_gain=0.35,
        consolidation_length_m=6.0,
        interference_penalty=0.6,
        interference_length_m=2.0,
    )
    params.update(overrides)
    return params


def test_convoy_energy_has_an_interior_minimum() -> None:
    optimum = optimal_escort_spacing(**convoy_kwargs())
    assert optimum["interior"] is True
    lo, hi = optimum["spacing_bracket_m"]
    assert lo < optimum["spacing_opt_m"] < hi
    assert optimum["curvature_j_per_m3"] > 0.0
    assert optimum["energy_opt_j_per_m"] < optimum["energy_at_lower_bound_j_per_m"]
    assert optimum["energy_opt_j_per_m"] < optimum["energy_at_upper_bound_j_per_m"]


def test_convoy_energy_rises_at_both_brackets() -> None:
    kwargs = convoy_kwargs()
    e_lo = convoy_energy_metre(spacing_m=0.5 * kwargs["ship_length_m"], **{
        k: v for k, v in kwargs.items() if k != "ship_length_m"
    })
    e_mid = convoy_energy_metre(spacing_m=1.6 * kwargs["ship_length_m"], **{
        k: v for k, v in kwargs.items() if k != "ship_length_m"
    })
    e_hi = convoy_energy_metre(spacing_m=3.0 * kwargs["ship_length_m"], **{
        k: v for k, v in kwargs.items() if k != "ship_length_m"
    })
    assert e_mid < e_lo and e_mid < e_hi


def test_optimum_shifts_to_smaller_spacing_when_consolidation_dominates() -> None:
    weak = optimal_escort_spacing(**convoy_kwargs())["spacing_opt_m"]
    strong = optimal_escort_spacing(**convoy_kwargs(consolidation_gain=1.2))["spacing_opt_m"]
    assert strong < weak


def test_boundary_optimum_is_flagged_not_reported_as_interior() -> None:
    optimum = optimal_escort_spacing(
        **convoy_kwargs(interference_penalty=0.0, consolidation_gain=0.5)
    )
    assert optimum["interior"] is False
    assert math.isnan(optimum["curvature_j_per_m3"])


def test_convoy_validation() -> None:
    with pytest.raises(ValueError):
        convoy_energy_metre(
            spacing_m=-1.0, lead_resistance_n=1e5, escort_channel_resistance_n=1e5
        )
    with pytest.raises(ValueError):
        convoy_energy_metre(
            spacing_m=1.0, lead_resistance_n=1e5, escort_channel_resistance_n=1e5,
            escort_efficiency=0.0,
        )


# --------------------------------------------------------------------------- API
def test_api_handler_returns_reproducible_contract() -> None:
    response = handle_channel_resistance(
        {
            "beam_m": 4.0,
            "thickness_m": 1.0,
            "concentration": 0.6,
            "velocity_ms": 0.5,
            "sigma_f_pa": 5.0e5,
            "channel_coefficient": 0.042,
            "concentration_exponent": 0.9,
            "thickness_exponent": 1.6,
        }
    )
    assert response["status"] == "ok"
    assert response["resistance_n"] > 0.0
    expected = riska_channel_resistance(
        beam_m=4.0,
        thickness_m=1.0,
        concentration=0.6,
        velocity_ms=0.5,
        channel_coefficient=0.042,
        concentration_exponent=0.9,
        thickness_exponent=1.6,
        reference=RiskaReference(sigma_f_pa=5.0e5),
    )
    assert response["resistance_n"] == pytest.approx(expected, rel=1e-12)
    assert "Riska" in response["model"]
    assert response["limitations"]


def test_api_handler_convoy_block_and_errors() -> None:
    payload = {
        "beam_m": 4.0,
        "thickness_m": 1.0,
        "concentration": 0.6,
        "channel_coefficient": 0.042,
        "convoy": {
            "ship_length_m": 2.355,
            "n_escort": 2,
            "spacing_m": 2.0,
            "consolidation_gain": 0.35,
            "consolidation_length_m": 6.0,
            "interference_penalty": 0.6,
            "interference_length_m": 2.0,
        },
    }
    response = handle_channel_resistance(payload)
    assert "convoy" in response
    assert response["convoy"]["interior"] is True
    assert "energy_at_requested_spacing_j_per_m" in response["convoy"]
    with pytest.raises(KeyError):
        handle_channel_resistance({"thickness_m": 1.0})
    with pytest.raises(ValueError):
        handle_channel_resistance("not-a-mapping")