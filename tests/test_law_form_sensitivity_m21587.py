"""Tests for :mod:`tensordem.granular_column.law_form` (m21587).

Every assertion is an invariant of the *arithmetic*, not a copy of a stored
number: the reference-law defect is derivable in closed form, and the fitting
helpers are checked against synthetic data with known coefficients.
"""
from __future__ import annotations

import math

import pytest

from tensordem.granular_column import law_form as L


# ---------------------------------------------------------------------------
# closed-form facts about the 2004 law
# ---------------------------------------------------------------------------
def test_crossover_is_analytic():
    assert L.crossover_aspect() == pytest.approx((1.6 / 1.24) ** 2, rel=1e-15)
    assert L.crossover_aspect() < L.CARD_BREAK


def test_branches_meet_at_the_crossover():
    a_star = L.crossover_aspect()
    assert L.law_linear(a_star) == pytest.approx(L.law_sqrt(a_star), rel=1e-15)


def test_card_break_is_discontinuous_and_quantified():
    d = L.discontinuity()
    # the linear branch is the upper one just above the crossover
    assert d["absolute_jump"] > 0.0
    assert d["relative_jump"] == pytest.approx(
        (L.law_linear(1.7) - L.law_sqrt(1.7)) / L.law_sqrt(1.7), rel=1e-15)
    # the jump is far below the 15 % acceptance band of the anchor
    assert 0.0 < d["relative_jump"] < 0.01
    assert d["break_minus_crossover"] == pytest.approx(1.7 - d["crossover_aspect"])


def test_slope_ratio_across_break_matches_derivatives():
    d = L.discontinuity()
    expect = L.LINEAR_PREFACTOR / (0.5 * L.SQRT_PREFACTOR / math.sqrt(1.7))
    assert d["slope_ratio_across_break"] == pytest.approx(expect, rel=1e-14)
    assert 1.5 < d["slope_ratio_across_break"] < 3.0


def test_piecewise_switch_semantics():
    assert L.law_piecewise(1.0) == pytest.approx(L.law_linear(1.0))
    assert L.law_piecewise(2.0) == pytest.approx(L.law_sqrt(2.0))
    # at exactly the break the square-root branch is used (>= semantics)
    assert L.law_piecewise(1.7) == pytest.approx(L.law_sqrt(1.7))
    # just below it, the linear branch is used -> jump of the law itself
    assert L.law_piecewise(1.7 - 1e-12) == pytest.approx(L.law_linear(1.7 - 1e-12))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        L.law_sqrt(-1.0)
    with pytest.raises(ValueError):
        L.law_piecewise(1.0, break_at=0.0)
    with pytest.raises(ValueError):
        L.discontinuity(break_at=-2.0)
    with pytest.raises(ValueError):
        L.law_linear(float("nan"))
    with pytest.raises(ValueError):
        L.crossover_aspect(linear_prefactor=0.0)


# ---------------------------------------------------------------------------
# fitting helpers against synthetic data with known coefficients
# ---------------------------------------------------------------------------
def _points(fn, aspects, gpr=12.0):
    return [{"case_id": f"syn_{a}", "aspect_measured": float(a),
             "runout_ratio": float(fn(a)), "grains_per_radius": gpr}
            for a in aspects]


def test_power_fit_recovers_exact_coefficients():
    pts = _points(lambda a: L.law_sqrt(a), (1.8, 2.0, 2.5, 3.0, 3.5))
    fit = L.fit_single_power(pts)
    assert fit["prefactor"] == pytest.approx(1.6, rel=1e-9)
    assert fit["exponent"] == pytest.approx(0.5, rel=1e-9)
    assert fit["rms_relative"] == pytest.approx(0.0, abs=1e-12)


def test_power_fit_recovers_an_arbitrary_exponent():
    fn = lambda a: 1.0 + 0.9 * a ** 0.75
    fit = L.fit_single_power(_points(fn, (0.5, 1.0, 1.5, 2.0, 3.0)))
    assert fit["prefactor"] == pytest.approx(0.9, rel=1e-9)
    assert fit["exponent"] == pytest.approx(0.75, rel=1e-9)


def test_power_fit_reports_residual_for_misfit_data():
    # a piecewise-generated set cannot be fitted exactly by one smooth law
    pts = _points(lambda a: L.law_piecewise(a), (0.5, 1.0, 1.5, 2.0, 3.0))
    fit = L.fit_single_power(pts)
    assert fit["rms_relative"] > 1e-4
    assert fit["max_abs_relative"] >= fit["rms_relative"]


def test_power_fit_needs_two_usable_points():
    with pytest.raises(ValueError):
        L.fit_single_power([{"aspect_measured": 1.5, "runout_ratio": 0.9}])
    with pytest.raises(ValueError):
        L.fit_single_power([{"aspect_measured": 1.5, "runout_ratio": 2.0},
                            {"aspect_measured": 1.5, "runout_ratio": 2.1}])


def test_error_table_is_signed_and_reference_consistent():
    pts = _points(lambda a: L.law_piecewise(a), (1.0, 2.0))
    rows = L.error_table(pts)
    assert all(abs(r["err_piecewise"]) < 1e-12 for r in rows)
    # sqrt-only extrapolation over-predicts the deep point
    assert rows[1]["err_sqrt_only"] == pytest.approx(0.0, abs=1e-12)
    # linear-only extrapolation under-predicts a=2 badly
    assert rows[1]["err_linear_only"] < -0.05


def test_rms_by_form_picks_the_generating_form():
    pts = _points(lambda a: L.law_sqrt(a), (1.8, 2.0, 3.0))
    agg = L.rms_by_form(L.error_table(pts))
    assert "sqrt_only" in agg["best_forms"]
    assert agg["sqrt_only"]["n_within_0p03"] == 3
    # the linear branch still lands within 3 % near the crossover (a=1.8); it is
    # the deep point that exposes it, so the invariant is on the worst case.
    assert agg["linear_only"]["n_within_0p03"] < 3
    assert agg["linear_only"]["max_abs_relative"] > 0.05
    assert agg["linear_only"]["rms_relative"] > agg["sqrt_only"]["rms_relative"]


def test_rms_by_form_rejects_empty_input():
    with pytest.raises(ValueError):
        L.rms_by_form([])


def test_local_slope_and_degenerate_base():
    assert L.local_slope(1.0, 2.0, 2.0, 3.0) == pytest.approx(1.0)
    assert math.isnan(L.local_slope(1.0, 2.0, 1.0, 3.0))


def test_slope_ratio_uses_extreme_points_of_each_side():
    lo = _points(lambda a: L.law_linear(a), (1.0, 1.4))
    hi = _points(lambda a: L.law_sqrt(a), (1.8, 2.4))
    out = L.slope_ratio_across_break(lo, hi)
    assert out["slope_below"] == pytest.approx(L.LINEAR_PREFACTOR, rel=1e-12)
    assert out["slope_above"] == pytest.approx(
        L.local_slope(1.8, L.law_sqrt(1.8), 2.4, L.law_sqrt(2.4)), rel=1e-12)
    assert out["slope_ratio_measured"] > 0.0
    with pytest.raises(ValueError):
        L.slope_ratio_across_break(lo[:1], hi)


def test_error_trend_recovers_a_synthetic_drift():
    pts = [{"aspect_measured": a, "err_piecewise": 0.02 * a - 0.01}
           for a in (1.0, 1.5, 2.0, 3.0)]
    trend = L.error_trend(pts)
    assert trend["slope_per_aspect"] == pytest.approx(0.02, rel=1e-9)
    assert trend["intercept"] == pytest.approx(-0.01, abs=1e-12)
    assert trend["r2"] == pytest.approx(1.0, abs=1e-12)


def test_error_trend_needs_two_points():
    with pytest.raises(ValueError):
        L.error_trend([{"aspect_measured": 1.0, "err_piecewise": 0.0}])