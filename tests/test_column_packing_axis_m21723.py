"""Tests for the ``phi_0`` axis bookkeeping (m21723).

Two layers:

* **closed-form synthetic** -- every claim of
  :mod:`tensordem.granular_column.packing_axis` is exercised on data whose
  answer is known analytically (a perfect line must come back to 1e-12, a
  replicate-only abscissa must be *refused*, a slope with a large stderr must
  not be allowed to explain a spread);
* **pinned geometry** -- ``phi_0`` of the frozen ``a = 3`` / 8 grains per
  ``r_i`` packing, the value the m21656 audit table reports, so that any change
  to the lattice builder or to the ``phi_0`` definition breaks the test instead
  of silently re-basing the whole ladder.

The DEM sweep results themselves are checked in
``tests/test_column_packing_axis_sweep_m21723.py`` (skipped while the run
directory is absent).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular_column.packing_axis import (
    CONFOUND_SHARE_COMPARABLE,
    CONFOUND_SHARE_DOMINATED,
    confound_decomposition,
    difference_quotient,
    ols_slope,
    slope_is_resolved,
    solid_fraction,
)


# --------------------------------------------------------------------------
# solid_fraction
# --------------------------------------------------------------------------
def test_solid_fraction_matches_hand_value():
    # 1000 grains of r_p = 1 mm filling a column of r_i = 2 cm, h_i = 5 cm.
    n, r_p, r_i, h_i = 1000, 1e-3, 2e-2, 5e-2
    expect = n * (4.0 / 3.0 * math.pi * r_p ** 3) / (math.pi * r_i ** 2 * h_i)
    assert solid_fraction(n, r_p, r_i, h_i) == pytest.approx(expect, rel=1e-15)
    # Sphere volume fraction cannot exceed the Kepler bound for any packing.
    assert expect < 0.7405


def test_solid_fraction_scales_as_documented():
    base = solid_fraction(100, 1.0, 10.0, 20.0)
    # Doubling the grain count at constant geometry doubles phi_0.
    assert solid_fraction(200, 1.0, 10.0, 20.0) == pytest.approx(2 * base, rel=1e-15)
    # Doubling the column height halves it.
    assert solid_fraction(100, 1.0, 10.0, 40.0) == pytest.approx(base / 2, rel=1e-15)


@pytest.mark.parametrize("bad", [
    (0, 1.0, 10.0, 20.0),      # no grains
    (10, 0.0, 10.0, 20.0),     # zero radius
    (10, 1.0, -1.0, 20.0),     # negative column radius
    (10, 1.0, 10.0, float("nan")),
    (10, float("inf"), 10.0, 20.0),
])
def test_solid_fraction_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        solid_fraction(*bad)


def test_pinned_phi0_of_the_frozen_a3_g8_packing():
    """The m21656 audit reports phi_0 = 0.6263 for a = 3, 8 grains / r_i."""
    from tensordem.granular_column.axisym import (build_axisymmetric_column,
                                                  radial_extent)
    import case_column_collapse_time_m18261 as base

    case = base.make_case({**base.BASE, "case_id": "PIN", "aspect_ratio": 3.0,
                           "grains_per_radius": 8.0})
    pos, mat = build_axisymmetric_column(case)
    ext = radial_extent(pos, mat.radius)
    h_i = float(pos[:, 2].max() + mat.radius)
    assert ext["p995"] == pytest.approx(0.47273, abs=5e-4)
    assert solid_fraction(len(pos), mat.radius, ext["p995"], h_i) == pytest.approx(
        0.62631, abs=5e-4)


# --------------------------------------------------------------------------
# ols_slope
# --------------------------------------------------------------------------
def test_ols_slope_recovers_an_exact_line_to_1e12():
    x = np.array([0.475, 0.506, 0.570, 0.626])
    y = 7.25 - 5.0 * x
    fit = ols_slope(x, y)
    assert fit["status"] == "fit"
    assert fit["slope"] == pytest.approx(-5.0, abs=1e-12)
    assert fit["intercept"] == pytest.approx(7.25, abs=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-15)
    assert fit["stderr"] == pytest.approx(0.0, abs=1e-15)
    assert fit["n"] == 4 and fit["n_distinct"] == 4


def test_ols_slope_refuses_two_points():
    fit = ols_slope([0.5, 0.6], [4.0, 3.5])
    assert fit["status"] == "indeterminate"
    assert "fewer than" in fit["reason"]
    assert math.isnan(fit["slope"])
    assert slope_is_resolved(fit)["resolved"] is False


def test_ols_slope_refuses_replicates_only():
    """Three points, two distinct abscissae: the response is not identifiable."""
    fit = ols_slope([0.5, 0.5, 0.6], [4.0, 4.2, 3.5])
    assert fit["status"] == "indeterminate"
    assert "distinct" in fit["reason"]


def test_ols_slope_refuses_degenerate_abscissa():
    fit = ols_slope([0.5, 0.5, 0.5], [4.0, 4.2, 3.5])
    assert fit["status"] == "indeterminate"


def test_ols_slope_drops_non_finite_pairs_then_refuses():
    fit = ols_slope([0.5, 0.6, float("nan"), 0.7],
                    [4.0, float("inf"), 3.0, 3.4])
    assert fit["status"] == "indeterminate"
    assert fit["n"] == 2


def test_ols_slope_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ols_slope([1.0, 2.0, 3.0], [1.0, 2.0])


def test_ols_slope_of_pure_noise_is_not_resolved():
    rng = np.random.default_rng(20251123)
    x = np.array([0.475, 0.506, 0.570, 0.626, 0.65])
    y = 4.0 + rng.normal(scale=0.05, size=x.size)
    fit = ols_slope(x, y)
    assert fit["status"] == "fit"
    verdict = slope_is_resolved(fit)
    assert verdict["resolved"] is False
    assert "stderr" in verdict["reason"]


def test_ols_slope_of_a_clear_trend_is_resolved():
    rng = np.random.default_rng(7)
    x = np.array([0.475, 0.506, 0.570, 0.626, 0.65])
    y = 7.2 - 5.0 * x + rng.normal(scale=0.005, size=x.size)
    fit = ols_slope(x, y)
    assert slope_is_resolved(fit)["resolved"] is True


def test_slope_is_resolved_needs_two_sigma():
    # A fit written by hand: |slope| = 1.9 * stderr must not count as resolved.
    fit = {"status": "fit", "slope": 1.9, "stderr": 1.0}
    assert slope_is_resolved(fit)["resolved"] is False
    fit["slope"] = 2.1
    assert slope_is_resolved(fit)["resolved"] is True


def test_slope_is_resolved_needs_defined_stderr():
    fit = {"status": "fit", "slope": 5.0, "stderr": float("nan")}
    out = slope_is_resolved(fit)
    assert out["resolved"] is False
    assert "degrees of freedom" in out["reason"]


def test_slope_is_resolved_rejects_nan_slope():
    assert slope_is_resolved({"status": "fit", "slope": float("nan"),
                              "stderr": 0.1})["resolved"] is False


# --------------------------------------------------------------------------
# difference_quotient
# --------------------------------------------------------------------------
def test_difference_quotient_is_labelled_as_significance_free():
    out = difference_quotient(0.506, 4.30, 0.626, 4.12)
    assert out["status"] == "difference_quotient"
    assert out["slope"] == pytest.approx((4.12 - 4.30) / (0.626 - 0.506))
    assert math.isnan(out["stderr"])
    assert "no significance" in out["reason"]
    # It may never be promoted to a resolved fit through slope_is_resolved.
    assert slope_is_resolved(out)["resolved"] is False


def test_difference_quotient_rejects_duplicate_abscissa():
    with pytest.raises(ValueError):
        difference_quotient(0.5, 1.0, 0.5, 2.0)


def test_difference_quotient_rejects_non_finite():
    with pytest.raises(ValueError):
        difference_quotient(0.5, float("nan"), 0.6, 2.0)


# --------------------------------------------------------------------------
# confound_decomposition
# --------------------------------------------------------------------------
def _decomp(**over):
    kw = dict(slope=-5.0, dphi=0.116, observed_spread=0.58, reference=3.78,
              resolved=True)
    kw.update(over)
    return confound_decomposition(**kw)


def test_confound_arithmetic():
    out = _decomp()
    assert out["implied_shift"] == pytest.approx(5.0 * 0.116)
    assert out["implied_relative"] == pytest.approx(5.0 * 0.116 / 3.78)
    assert out["share"] == pytest.approx(0.58 / 0.58)


def test_confound_verdict_thresholds_are_exact():
    # share == 1.0 -> dominated
    assert _decomp().get("verdict") == "packing_dominated"
    # share == exactly CONFOUND_SHARE_DOMINATED -> dominated (inclusive)
    out = confound_decomposition(slope=-CONFOUND_SHARE_DOMINATED, dphi=1.0,
                                 observed_spread=1.0, reference=1.0, resolved=True)
    assert out["share"] == pytest.approx(CONFOUND_SHARE_DOMINATED)
    assert out["verdict"] == "packing_dominated"
    # share == exactly CONFOUND_SHARE_COMPARABLE -> comparable (inclusive)
    out = confound_decomposition(slope=-CONFOUND_SHARE_COMPARABLE, dphi=1.0,
                                 observed_spread=1.0, reference=1.0, resolved=True)
    assert out["verdict"] == "comparable"
    # just below -> resolution_dominated
    out = confound_decomposition(slope=-0.199, dphi=1.0, observed_spread=1.0,
                                 reference=1.0, resolved=True)
    assert out["verdict"] == "resolution_dominated"


def test_confound_unresolved_slope_may_not_explain_anything():
    out = _decomp(resolved=False)
    assert out["verdict"] == "indeterminate"
    assert "not resolved" in out["reason"]
    # The magnitude is still reported, but not as an explanation.
    assert out["share"] == pytest.approx(1.0)


def test_confound_zero_spread_is_indeterminate():
    out = _decomp(observed_spread=0.0)
    assert out["verdict"] == "indeterminate"
    assert "zero" in out["reason"]


@pytest.mark.parametrize("bad", ["slope", "dphi", "observed_spread", "reference"])
def test_confound_non_finite_inputs_are_indeterminate(bad):
    out = _decomp(**{bad: float("nan")})
    assert out["verdict"] == "indeterminate"
    assert bad in out["reason"]


def test_confound_negative_inputs_are_taken_in_absolute_value():
    a = confound_decomposition(slope=-5.0, dphi=0.116, observed_spread=0.58,
                               reference=3.78, resolved=True)
    b = confound_decomposition(slope=5.0, dphi=-0.116, observed_spread=-0.58,
                               reference=-3.78, resolved=True)
    assert a["implied_shift"] == pytest.approx(b["implied_shift"])
    assert a["verdict"] == b["verdict"]


def test_confound_flags_a_shift_that_alone_exceeds_tolerance():
    out = confound_decomposition(slope=-8.0, dphi=0.116, observed_spread=3.0,
                                 reference=3.78, resolved=True, tolerance=0.15)
    assert out["implied_relative"] > 0.15
    assert out["exceeds_tolerance_alone"] is True
    out = confound_decomposition(slope=-1.0, dphi=0.116, observed_spread=3.0,
                                 reference=3.78, resolved=True, tolerance=0.15)
    assert out["exceeds_tolerance_alone"] is False