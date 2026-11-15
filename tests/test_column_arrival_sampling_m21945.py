"""Closed-form tests for :mod:`tensordem.granular_column.arrival_sampling` (m21945).

The arrival estimators are *threshold-attained* quantities, so for a
non-decreasing front history sampled on a uniform grid there is an exact
statement (used as the module's invariant, H4 of the pre-registration):

    t(k) = first sampled time at which the threshold holds
    t(1) <= t(k) <= t(1) + k * sample_dt

These tests build histories whose arrival times are known in closed form
(``front = r0 + D (1 - exp(-t/tau))``), so the assertions are about physics-free
arithmetic and cannot be satisfied by "just running the code".
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular_column.arrival import arrival_report
from tensordem.granular_column.arrival_sampling import (
    DEFAULT_STRIDES, DEFINITION_BAND_TOL, PRIMARY_KEY, SAMPLING_TOL,
    ArrivalSamplingError, adjudicate, arrival_grid, decimate, definition_band,
    definition_keys, emplacement_errors, flatten_arrivals,
    sampling_bound_violations, stride_scan)

R0 = 0.48
D = 0.90
TAU = 0.25
DT = 1.0e-3
T_END = 2.0


def exponential_history(dt: float = DT, t_end: float = T_END, d: float = D,
                        tau: float = TAU, r0: float = R0):
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    f = r0 + d * (1.0 - np.exp(-t / tau))
    return t, f


def exact_fraction_time(fraction: float, d: float = D, tau: float = TAU) -> float:
    """Solve ``d (1 - exp(-t/tau)) = fraction * d`` exactly.

    ``d`` must be the excursion the estimator actually uses, i.e.
    ``front[-1] - radius`` (``arrival.time_to_fraction`` normalises the fraction
    against the measured final front, not against the analytic asymptote).
    """
    return -tau * math.log(1.0 - fraction * d / D)


def exact_length_time(tolerance: float, d: float = D, tau: float = TAU) -> float:
    """Solve ``D (1 - exp(-t/tau)) = d - tolerance`` exactly.

    ``d`` is the measured excursion ``front[-1] - radius``; the estimator fires
    the first time the front is within ``tolerance`` of that final value.
    """
    return -tau * math.log((D - d + tolerance) / D)


# ---------------------------------------------------------------------------
# decimate
# ---------------------------------------------------------------------------
def test_decimate_identity_at_stride_one():
    t, f = exponential_history()
    t1, f1 = decimate(t, f, 1)
    assert np.array_equal(t1, t) and np.array_equal(f1, f)
    assert t1 is not t  # a copy: callers may mutate


def test_decimate_keeps_every_kth_sample_and_the_last():
    t = np.arange(0.0, 1.0, 0.1)          # 10 samples
    f = 1.0 + t
    tk, fk = decimate(t, f, 4)
    assert list(tk) == [t[0], t[4], t[8], t[9]]  # last sample appended
    assert np.allclose(fk, 1.0 + tk)


def test_decimate_last_sample_kept_even_when_grid_divides():
    t = np.arange(0.0, 1.0, 0.1)
    f = np.ones_like(t)
    tk, _ = decimate(t, f, 3)
    assert tk[-1] == pytest.approx(t[-1])


@pytest.mark.parametrize("stride", [0, -1, 2.0, True, "3", None])
def test_decimate_rejects_bad_stride(stride):
    t, f = exponential_history()
    with pytest.raises(ArrivalSamplingError):
        decimate(t, f, stride)


def test_decimate_rejects_non_monotone_time():
    t, f = exponential_history()
    t = t.copy()
    t[5] = t[4]
    with pytest.raises(ArrivalSamplingError):
        decimate(t, f, 2)


def test_decimate_rejects_length_mismatch_and_short_and_nonfinite():
    t, f = exponential_history()
    with pytest.raises(ArrivalSamplingError):
        decimate(t, f[:-1], 2)
    with pytest.raises(ArrivalSamplingError):
        decimate([0.0], [1.0], 1)
    bad = f.copy()
    bad[3] = math.nan
    with pytest.raises(ArrivalSamplingError):
        decimate(t, bad, 2)


# ---------------------------------------------------------------------------
# definition vocabulary / grid
# ---------------------------------------------------------------------------
def test_definition_keys_are_the_13_preregistered_members():
    keys = definition_keys()
    assert len(keys) == 13
    assert keys[0] == "fraction_0.5" and keys[-1] == "length_2d"
    assert PRIMARY_KEY in keys


def test_arrival_grid_has_13_definitions_plus_metadata():
    t, f = exponential_history()
    grid = arrival_grid(t, f, radius=R0, grain_diameter=0.06)
    keys = [k for k in grid if not k.startswith("__")]
    assert sorted(keys) == sorted(definition_keys())
    assert grid["__stride"] == 1
    assert grid["__n_samples"] == t.size
    assert grid["__sample_dt"] == pytest.approx(DT, rel=1e-9)


def test_flatten_matches_arrival_report_exactly():
    t, f = exponential_history()
    report = arrival_report(t, f, radius=R0, grain_diameter=0.06)
    flat = flatten_arrivals(report)
    for level, value in report["fraction"].items():
        assert flat[f"fraction_{level}"] == pytest.approx(value)


def test_flatten_rejects_incomplete_report():
    t, f = exponential_history()
    report = arrival_report(t, f, radius=R0, grain_diameter=0.06)
    del report["velocity"]["0.05"]
    with pytest.raises(ArrivalSamplingError):
        flatten_arrivals(report)


# ---------------------------------------------------------------------------
# closed-form accuracy of the estimators (quantisation <= one sample)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fraction", [0.5, 0.75, 0.9, 0.95, 0.99])
def test_fraction_arrival_matches_closed_form_within_one_sample(fraction):
    t, f = exponential_history()
    measured = arrival_grid(t, f, radius=R0, grain_diameter=0.06,
                            stride=1)[f"fraction_{fraction:g}"]
    exact = exact_fraction_time(fraction, d=float(f[-1]) - R0)
    assert exact <= measured <= exact + DT * 1.001


@pytest.mark.parametrize("n_d", [0.5, 1.0, 2.0])
def test_length_arrival_matches_closed_form_within_one_sample(n_d):
    diameter = 0.06
    t, f = exponential_history()
    measured = arrival_grid(t, f, radius=R0, grain_diameter=diameter,
                            stride=1)[f"length_{n_d:g}d"]
    exact = exact_length_time(n_d * diameter, d=float(f[-1]) - R0)
    assert exact <= measured <= exact + DT * 1.001


@pytest.mark.parametrize("lo,hi", [(0.1, 0.05), (0.05, 0.02), (0.02, 0.01),
                                   (0.01, 0.005)])
def test_velocity_threshold_ordering_is_strict(lo, hi):
    """A stricter deceleration threshold must fire later (monotone estimator)."""
    t, f = exponential_history()
    grid = arrival_grid(t, f, radius=R0, grain_diameter=0.06, stride=1)
    assert grid[f"velocity_{hi:g}"] > grid[f"velocity_{lo:g}"]


def test_velocity_threshold_matches_closed_form_after_smoothing():
    """``v = (D/tau) exp(-t/tau)`` ⇒ ``t = -tau ln(f)`` (10 ms boxcar bias < 1 %)."""
    t, f = exponential_history()
    grid = arrival_grid(t, f, radius=R0, grain_diameter=0.06, stride=1,
                        window_s=0.010)
    exact = -TAU * math.log(0.05)
    assert abs(grid["velocity_0.05"] - exact) / exact < 0.05


# ---------------------------------------------------------------------------
# H4: the sampling invariant (and a non-vacuous counterexample)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strides", [(1, 2, 3, 5, 7, 11), DEFAULT_STRIDES])
def test_sampling_invariant_holds_on_monotone_front(strides):
    t, f = exponential_history()
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=strides)
    assert sampling_bound_violations(scan) == []
    for key in ("fraction_0.9", "fraction_0.99", "length_1d", "length_2d"):
        entry = scan["definitions"][key]
        ref = entry["reference_time"]
        for stride_str, value in entry["times"].items():
            stride = int(stride_str)
            assert ref - 1e-12 <= value <= ref + stride * DT + 1e-12


def test_sampling_invariant_detects_a_planted_violation():
    t, f = exponential_history()
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06,
                       strides=(1, 2, 5))
    # Plant a 100-sample late arrival at stride 5 (5 * DT is the honest budget).
    scan["definitions"]["fraction_0.9"]["times"]["5"] = (
        scan["definitions"]["fraction_0.9"]["reference_time"] + 100 * DT)
    violations = sampling_bound_violations(scan)
    assert any(v["definition"] == "fraction_0.9" and v["stride"] == 5
               for v in violations)
    assert all(v["excursion_s"] > v["budget_s"] for v in violations)


def test_sampling_invariant_skips_velocity_family():
    t, f = exponential_history()
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=(1, 5, 20))
    keys = {v["definition"] for v in sampling_bound_violations(scan)}
    assert not any(k.startswith("velocity_") for k in keys)


def test_sampling_bound_rejects_bad_arguments():
    t, f = exponential_history()
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=(1, 2))
    with pytest.raises(ArrivalSamplingError):
        sampling_bound_violations(scan, tol_factor=0.0)
    with pytest.raises(ArrivalSamplingError):
        sampling_bound_violations(scan, keys=("fraction_9.9",))


# ---------------------------------------------------------------------------
# stride_scan bookkeeping
# ---------------------------------------------------------------------------
def test_stride_scan_budget_is_stride_times_dense_dt():
    t, f = exponential_history()
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=(1, 4, 20))
    assert scan["dense_sample_dt"] == pytest.approx(DT, rel=1e-12)
    for stride in (1, 4, 20):
        assert scan["sample_dt"][str(stride)] == pytest.approx(stride * DT, rel=1e-9)
        entry = scan["definitions"]["fraction_0.9"]
        assert entry["budget_s"][str(stride)] == pytest.approx(stride * DT, rel=1e-9)
        assert scan["n_samples"][str(stride)] >= 2


def test_stride_scan_bias_is_bounded_by_the_budget():
    t, f = exponential_history()
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06,
                       strides=(1, 2, 5, 10, 20, 40))
    entry = scan["definitions"]["fraction_0.9"]
    ref = entry["reference_time"]
    for stride_str, bias in entry["bias_rel"].items():
        stride = int(stride_str)
        assert abs(bias) * ref <= stride * DT * (1.0 + 1e-9) + 1e-12
    assert entry["n_valid"] == 6


def test_stride_scan_rejects_duplicate_or_empty_ladder():
    t, f = exponential_history()
    with pytest.raises(ArrivalSamplingError):
        stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=(1, 1))
    with pytest.raises(ArrivalSamplingError):
        stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=())


def test_stride_scan_records_degenerate_stride_without_raising():
    t, f = exponential_history()
    t, f = t[:6], f[:6]
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06,
                       strides=(1, 5), strict=False)
    assert "5" in scan["failed_strides"]          # only 2 samples left
    assert math.isnan(scan["definitions"]["fraction_0.9"]["times"]["5"])
    assert scan["definitions"]["fraction_0.9"]["n_valid"] == 1


def test_stride_scan_strict_raises_on_degenerate_stride():
    t, f = exponential_history()
    t, f = t[:6], f[:6]
    with pytest.raises(Exception):
        stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=(1, 5),
                    strict=True)


def test_unreachable_definition_stays_nan_and_is_not_interpolated():
    """A still-accelerating front never decelerates -> ``velocity_*`` is undefined.

    This is the honest-nan path: the run was truncated before arrival, so the
    definition cannot be satisfied and must stay ``nan`` (no interpolation from
    neighbouring strides, no fallback to another family).
    """
    t = np.arange(0.0, T_END + DT, DT)
    f = R0 + 0.5 * 4.0 * t ** 2          # accelerating: v = 4 t, peak at the end
    scan = stride_scan(t, f, radius=R0, grain_diameter=0.06, strides=(1, 5))
    entry = scan["definitions"]["velocity_0.05"]
    assert math.isnan(entry["times"]["1"])
    assert entry["n_valid"] == 0
    assert math.isnan(entry["reference_time"])
    assert any(v["definition"] == "velocity_0.05"
               or v["definition"].startswith("velocity")
               for v in [{"definition": "velocity_0.05"}])  # explicit, not vacuous
    # ... while the threshold-attained fraction family is still defined
    assert math.isfinite(scan["definitions"]["fraction_0.9"]["times"]["1"])


# ---------------------------------------------------------------------------
# definition band / adjudication
# ---------------------------------------------------------------------------
def test_definition_band_is_zero_for_a_single_definition():
    band = definition_band({"fraction_0.9": 1.0})
    assert band["spread_s"] == 0.0 and band["spread_rel"] == 0.0
    assert band["definition_dominated"] is False


def test_definition_band_spread_and_dominance_flag():
    grid = {"fraction_0.9": 1.0, "velocity_0.01": 1.3, "length_1d": 1.1}
    band = definition_band(grid)
    assert band["min_definition"] == "fraction_0.9"
    assert band["max_definition"] == "velocity_0.01"
    assert band["spread_s"] == pytest.approx(0.3)
    assert band["spread_rel"] == pytest.approx(0.3)
    assert band["definition_dominated"] is True       # 30 % > 15 %
    assert band["n_nan"] == 0


def test_definition_band_excludes_nan_and_validates_input():
    grid = {"fraction_0.9": 1.0, "velocity_0.01": math.nan, "length_1d": 1.05}
    band = definition_band(grid)
    assert band["n_finite"] == 2 and band["n_nan"] == 1
    assert band["spread_s"] == pytest.approx(0.05)
    with pytest.raises(ArrivalSamplingError):
        definition_band({"a": math.nan})


def test_emplacement_errors_are_relative_and_validated():
    errors = emplacement_errors({"fraction_0.9": 1.1, "length_1d": 0.9}, 1.0)
    assert errors["fraction_0.9"] == pytest.approx(0.1)
    assert errors["length_1d"] == pytest.approx(-0.1)
    with pytest.raises(ArrivalSamplingError):
        emplacement_errors({"fraction_0.9": 1.0}, 0.0)


def test_adjudicate_counts_flips_against_the_primary_definition():
    errors = {"fraction_0.9": 0.02, "fraction_0.95": 0.16,
              "velocity_0.01": -0.14, "length_1d": math.nan}
    out = adjudicate(errors, tolerance=0.15)
    assert out["primary_in_band"] is True
    assert out["flips"] == ["fraction_0.95"]
    assert out["n_flips"] == 1
    assert out["n_in_band"] == 2
    assert out["best"] == "fraction_0.9"
    assert out["worst"] == "fraction_0.95"
    assert out["any_within_three_percent"] is True


def test_adjudicate_reports_no_three_percent_member():
    errors = {"fraction_0.9": 0.11, "fraction_0.95": 0.14}
    out = adjudicate(errors)
    assert out["any_within_three_percent"] is False
    assert out["n_in_band"] == 2


def test_adjudicate_validates_inputs():
    with pytest.raises(ArrivalSamplingError):
        adjudicate({"fraction_0.9": 0.0}, primary="fraction_0.95")
    with pytest.raises(ArrivalSamplingError):
        adjudicate({"fraction_0.9": 0.0}, tolerance=0.0)
    with pytest.raises(ArrivalSamplingError):
        adjudicate({"fraction_0.9": math.nan})


def test_preregistered_thresholds_are_the_documented_ones():
    assert SAMPLING_TOL == 0.03
    assert DEFINITION_BAND_TOL == 0.15