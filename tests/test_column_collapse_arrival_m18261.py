"""Arrival-definition tests (mission m18261).

The definitions are the *acceptance criterion* of the second Lube et al. (2004)
prediction, so they are tested against histories whose answer is known in closed
form, and every rejection path is exercised: a definition that can be silently
satisfied by a truncated or non-monotone history is worthless as a criterion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular_column.arrival import (
    DEFAULT_WINDOW_S, PRIMARY_FRACTION, PRIMARY_LENGTH_TOLERANCE_D,
    PRIMARY_VELOCITY_FRACTION, ArrivalError, arrival_report, front_speed,
    smooth_front, time_to_fraction, time_to_length_tolerance,
    time_to_velocity_threshold)

R0, RUNOUT, TAU = 0.48, 0.30, 0.15
GRAIN_D = 0.06


def exponential_history(n: int = 4001, t_max: float = 2.0):
    """``r(t) = r0 + R (1 - exp(-t/tau))`` — the classic relaxation form."""
    t = np.linspace(0.0, t_max, n)
    return t, R0 + RUNOUT * (1.0 - np.exp(-t / TAU))


def test_fraction_definition_matches_the_closed_form():
    t, r = exponential_history()
    for fraction in (0.5, 0.75, 0.9, 0.99):
        got = time_to_fraction(t, r, R0, r[-1], fraction)
        want = -TAU * math.log(1.0 - fraction)
        assert got == pytest.approx(want, rel=0.02), (fraction, got, want)


def test_length_tolerance_matches_the_closed_form():
    t, r = exponential_history()
    for nd in (0.5, 1.0, 2.0):
        got = time_to_length_tolerance(t, r, nd * GRAIN_D, r[-1])
        want = -TAU * math.log(nd * GRAIN_D / RUNOUT)
        assert got == pytest.approx(want, rel=0.02), (nd, got, want)


def test_velocity_threshold_matches_the_closed_form():
    """``v = (R/tau) exp(-t/tau)`` -> ``t(v < f v_peak) = -tau ln f``."""
    t, r = exponential_history()
    for fraction in (0.10, 0.02, 0.01):
        got = time_to_velocity_threshold(t, r, fraction)
        want = -TAU * math.log(fraction)
        assert got == pytest.approx(want, rel=0.03), (fraction, got, want)


def test_velocity_threshold_is_nan_when_the_front_never_slows():
    t = np.linspace(0.0, 1.0, 201)
    r = R0 + 0.3 * t                      # constant speed: v == v_peak throughout
    assert math.isnan(time_to_velocity_threshold(t, r, 0.01))
    # speed still rising at the end of the run -> no arrival may be declared
    r2 = R0 + 0.3 * t ** 2
    assert math.isnan(time_to_velocity_threshold(t, r2, 0.5))


def test_truncated_history_cannot_fake_arrival_at_large_fraction():
    """A run cut short must report ``nan``, never an arrival time."""
    t, r = exponential_history()
    cut = t < 0.05                        # front has barely moved yet
    assert math.isnan(time_to_fraction(t[cut], r[cut], R0, r[-1], 0.9))
    assert math.isnan(time_to_velocity_threshold(t[cut], r[cut], 0.01))


def test_smoothing_window_is_stride_invariant():
    """The same ``window_s`` must mean the same physics at any sample stride."""
    t, r = exponential_history(n=8001)
    fine = arrival_report(t, r, radius=R0, grain_diameter=GRAIN_D)
    stride = 5
    coarse = arrival_report(t[::stride], r[::stride], radius=R0,
                            grain_diameter=GRAIN_D)
    for family in ("velocity", "fraction", "length"):
        for key, value in fine[family].items():
            other = coarse[family][key]
            if math.isnan(value) or math.isnan(other):
                continue
            assert other == pytest.approx(value, rel=0.03), (family, key, value, other)


def test_smoothing_does_not_manufacture_a_start_up_spike():
    t, r = exponential_history()
    speed = front_speed(t, r)
    analytic_peak = RUNOUT / TAU
    assert speed.max() <= 1.02 * analytic_peak
    assert np.all(np.isfinite(speed))


@pytest.mark.parametrize("kwargs, message", [
    ({"time": [0, 1, 2], "front": [0.4, 0.5, 0.6]}, "at least 4"),
    ({"time": [0, 1, 2, 3, 4], "front": [0.4, 0.5, 0.6]}, "equal length"),
    ({"time": [0, 2, 1, 3, 4], "front": [0.4, 0.5, 0.6, 0.7, 0.8]}, "strictly increasing"),
    ({"time": [0, 1, 2, 3, 4], "front": [0.4, 0.5, 0.45, 0.6, 0.7]}, "non-decreasing"),
    ({"time": [0, 1, 2, 3, np.nan], "front": [0.4, 0.5, 0.6, 0.7, 0.8]}, "finite"),
])
def test_history_validation_rejects_unusable_input(kwargs, message):
    with pytest.raises(ArrivalError) as excinfo:
        smooth_front(np.asarray(kwargs["time"], float),
                     np.asarray(kwargs["front"], float))
    assert message in str(excinfo.value)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_radius_must_be_positive_and_finite(bad):
    t, r = exponential_history()
    with pytest.raises(ArrivalError):
        time_to_fraction(t, r, bad, r[-1], 0.9)


def test_primary_members_are_the_preregistered_ones():
    t, r = exponential_history()
    report = arrival_report(t, r, radius=R0, grain_diameter=GRAIN_D)
    assert set(report["primary"]) == {"fraction_0.9", "velocity_0.01", "length_1d"}
    assert report["primary"]["fraction_0.9"] == pytest.approx(
        report["fraction"][f"{PRIMARY_FRACTION:g}"])
    assert report["primary"]["velocity_0.01"] == pytest.approx(
        report["velocity"][f"{PRIMARY_VELOCITY_FRACTION:g}"])
    assert report["primary"]["length_1d"] == pytest.approx(
        report["length"][f"{PRIMARY_LENGTH_TOLERANCE_D:g}"])
    assert report["window_s"] == DEFAULT_WINDOW_S
    # every definition must be later at a stricter (smaller) velocity fraction
    # keys sorted by their numeric value ascending -> arrival times descending
    fractions = sorted(report["velocity"], key=float)
    times = [report["velocity"][f] for f in fractions]
    assert all(b <= a + 1e-12 for a, b in zip(times, times[1:]))


def test_final_overshoot_is_reported_not_hidden():
    """A front that recedes is rejected; an overshoot is recorded."""
    t, r = exponential_history()
    final = float(r[-1])
    r = r.copy()
    r[-1] += 0.001                        # trailing extra grain at the very edge
    report = arrival_report(t, r, radius=R0, grain_diameter=GRAIN_D, final=final)
    assert report["front_final_overshoot"] == pytest.approx(0.001)
    assert report["front_final"] == pytest.approx(final)