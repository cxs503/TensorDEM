"""Tests for the IN52-N02 resolution ladder and Richardson extrapolation.

Every check is against a closed form (an exact exponential, a synthetic
``h^p`` error law) or measures the shipped shear-wave kernel; none of them
re-tunes anything.
"""
from __future__ import annotations

import math

import pytest

from tensordem.threefield.resolution_study import (
    DecaySample,
    applicability_envelope,
    decay_ladder,
    fit_decay_rate,
    richardson_extrapolate,
)


# -- fitting -----------------------------------------------------------------
def test_fit_recovers_a_synthetic_exponential():
    rate = 0.0137
    times = list(range(0, 120))
    amplitudes = [math.exp(-rate * t) for t in times]
    assert fit_decay_rate(times, amplitudes, start=0, stop=120) == pytest.approx(rate, rel=1e-12)


def test_window_length_does_not_bias_an_exact_exponential():
    """A pure exponential has no window dependence: this is the null model the
    real measurement is graded against."""
    rate = 0.0042
    times = list(range(0, 200))
    amplitudes = [1e-3 * math.exp(-rate * t) for t in times]
    short = fit_decay_rate(times, amplitudes, start=20, stop=140)
    long = fit_decay_rate(times, amplitudes, start=20, stop=200)
    assert short == pytest.approx(long, rel=1e-12)


def test_fit_skips_nonpositive_amplitudes_and_rejects_a_thin_window():
    times = list(range(10))
    amplitudes = [1.0, 0.9, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # only two positive samples survive the filter
    with pytest.raises(ValueError):
        fit_decay_rate(times, amplitudes, start=0, stop=10)
    with pytest.raises(ValueError):
        fit_decay_rate(times, amplitudes, start=2, stop=10)
    # an inverted window is a caller error, not a silent empty fit
    with pytest.raises(ValueError):
        fit_decay_rate(times, amplitudes, start=8, stop=4)


def test_fit_window_is_indexed_by_step_not_by_list_position():
    """The shipped history starts at ``fit_from + 1``; a window given in steps
    must not be shifted by that offset."""
    rate = 0.01
    times = list(range(21, 221))
    amplitudes = [math.exp(-rate * t) for t in times]
    # the window [20, 120) covers steps 21..119 whatever the list offset is
    assert fit_decay_rate(times, amplitudes, start=20, stop=120) == pytest.approx(rate, rel=1e-12)


# -- Richardson --------------------------------------------------------------
def test_richardson_recovers_a_second_order_limit():
    hs = (2.0 * math.pi / 16, 2.0 * math.pi / 32, 2.0 * math.pi / 64)
    errors = [0.5 * h * h for h in hs]
    fit = richardson_extrapolate(hs, errors)
    assert fit.order == pytest.approx(2.0, rel=1e-9)
    assert fit.error_limit == pytest.approx(0.0, abs=1e-12)
    assert fit.converged


def test_richardson_order_fit_sees_a_non_integer_order():
    hs = (0.4, 0.2, 0.1)
    errors = [0.25 * h**1.5 for h in hs]
    assert richardson_extrapolate(hs, errors).order == pytest.approx(1.5, rel=1e-9)


def test_wrong_assumed_order_moves_the_limit():
    """Negative control: the order is not a cosmetic label."""
    hs = (0.4, 0.2, 0.1)
    errors = [0.7 * h * h for h in hs]
    fitted = richardson_extrapolate(hs, errors).error_limit
    wrong = richardson_extrapolate(hs, errors, assumed_order=1.0).error_limit
    assert abs(fitted) < 1e-12
    assert abs(wrong) > 1e-3


def test_richardson_rejects_a_non_halving_ladder():
    with pytest.raises(ValueError):
        richardson_extrapolate((0.3, 0.2, 0.1), (0.1, 0.05, 0.02))
    with pytest.raises(ValueError):
        richardson_extrapolate((0.4, 0.2), (0.1, 0.05))


# -- envelope ----------------------------------------------------------------
def _sample(tau0: float, size: int, error: float, mode: int = 1) -> DecaySample:
    nu = (tau0 - 0.5) / 3.0
    k = 2.0 * math.pi * mode / size
    rate = nu * k * k * (1.0 + error)
    return DecaySample(
        tau0=tau0,
        size=size,
        mode=mode,
        start=20,
        stop=220,
        rate=rate,
        nu_nominal=nu,
        rate_nominal=nu * k * k,
        rate_discrete=nu * 2.0 * (1.0 - math.cos(k)),
        amplitude_final=1e-4,
    )


def test_envelope_reads_the_limits_off_measured_points():
    samples = [
        _sample(0.55, 16, 0.002),
        _sample(0.8, 16, 0.004),
        _sample(1.0, 16, -0.009),
        # same fluid, same cell wavenumber, finer lattice: the collapse says the
        # error must be identical
        _sample(1.0, 32, -0.009, mode=2),
        _sample(1.4, 16, -0.031),
        _sample(1.4, 32, 0.008),
        _sample(1.4, 32, -0.05, mode=3),
    ]
    envelope = applicability_envelope(samples, tolerance=0.01)
    assert envelope.per_lattice[16]["tau0_max"] == pytest.approx(1.0)
    assert envelope.per_lattice[16]["tau0_min"] == pytest.approx(0.55)
    assert envelope.per_lattice[16]["failing_tau0"] == [pytest.approx(1.4)]
    # the limit is quoted on the finest lattice booked, per the docstring
    assert envelope.tau0_limit == pytest.approx(1.4)
    assert envelope.per_lattice[32]["tau0_max"] == pytest.approx(1.4)
    assert envelope.kh_limit == pytest.approx(2.0 * math.pi / 32)
    assert envelope.kh_limit_tau0 == pytest.approx(1.4)
    assert envelope.collapse_max_error_difference == pytest.approx(0.0, abs=1e-12)
    assert envelope.collapse_pairs == 1
    assert any(abs(entry["error"]) > 0.01 for entry in envelope.failing)


def test_envelope_collapse_flags_a_lattice_dependent_error():
    """Negative control: a genuine lattice effect (not a k h effect) shows up as a
    spread inside one (tau0, k h) group."""
    samples = [_sample(1.4, 16, -0.031), _sample(1.4, 32, -0.010, mode=2)]
    envelope = applicability_envelope(samples)
    assert envelope.collapse_max_error_difference == pytest.approx(0.021, rel=1e-9)
    assert envelope.collapse_pairs == 1


# -- the shipped measurement --------------------------------------------------
def test_ladder_reproduces_and_shrinks_the_tau0_1p4_fail():
    """The published honest FAIL is resolution-driven: halving ``k h`` must cut it.

    Assertions are one-sided on purpose -- a *smaller* error on the finer lattice
    is the claim; the exact 1 %-band statement is booked in
    ``artifacts/threefield/resolution_ladder.json`` from the three-level ladder.
    """
    ladder = decay_ladder([1.4], [16, 32], modes=(1,), steps=140, windows=(120,), start=20)
    coarse = ladder[(1.4, 16, 1)].window(120)
    fine = ladder[(1.4, 32, 1)].window(120)
    assert coarse.error_nominal < -0.03           # the m2949 honest FAIL, reproduced
    assert abs(fine.error_nominal) < abs(coarse.error_nominal) / 2.0
    assert abs(fine.error_nominal) < 0.015


def test_ladder_windows_agree_away_from_the_transient():
    """Two windows on the same trajectory must give the same rate to well inside
    the criterion band -- otherwise the published number depends on the window."""
    ladder = decay_ladder([0.8], [16], modes=(1,), steps=220, windows=(120, 200), start=20)
    point = ladder[(0.8, 16, 1)]
    assert point.window_sensitivity < 0.01