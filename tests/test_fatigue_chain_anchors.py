"""Analytical anchors of the F(t) -> rainflow -> S-N chain.

Judge (a): constant-amplitude cycles must reproduce ``D = n / N(dS)`` exactly.
Judge (b): a single pulse must reproduce the Duhamel closed-form response.
Judge (c) (the ASTM E1049-85 counting regression) lives in
``test_fatigue_chain_astm.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.fatigue_chain.damage import (
    PowerLawSNCurve,
    damage_from_stress,
    equal_amplitude_damage,
    narrowband_rayleigh_damage,
)
from tensordem.fatigue_chain.response import (
    ModalChannel,
    duhamel_convolution,
    half_sine_closed_form,
    modal_stress_history,
    sdof_response,
)
from tensordem.whipping.fatigue import DNV_CURVES


def _sine_turning_points(amplitude: float, periods: int, freq: float = 2.0):
    """A sine sampled exactly at its turning points (range is 2A in float)."""
    period = 1.0 / freq
    t = np.arange(0.0, periods * period + period / 4.0, period / 4.0)
    return t, amplitude * np.cos(2.0 * math.pi * t / period)


@pytest.mark.parametrize("amplitude_MPa,periods", [(60.0, 40), (25.0, 40),
                                                   (150.0, 25), (90.0, 12)])
@pytest.mark.parametrize("curve_name", ["D_air"])
def test_constant_amplitude_miner_is_exact(amplitude_MPa, periods, curve_name):
    curve = DNV_CURVES[curve_name]
    _, sigma = _sine_turning_points(amplitude_MPa * 1e6, periods)
    result = damage_from_stress(sigma, curve)
    analytic = periods / curve.cycles_to_failure(np.array([2.0 * amplitude_MPa * 1e6]))[0]
    assert result.n_cycles == periods
    assert result.max_range_Pa == pytest.approx(2.0 * amplitude_MPa * 1e6, rel=1e-15)
    assert abs(result.damage - analytic) / analytic < 1e-12, (
        result.damage, analytic)


def test_equal_amplitude_damage_matches_closed_form():
    curve = DNV_CURVES["D_air"]
    d = equal_amplitude_damage(curve, 80e6, 12)
    expect = 12.0 / curve.cycles_to_failure(np.array([80e6]))[0]
    assert d == pytest.approx(expect, rel=1e-15)


def test_single_pulse_matches_duhamel_closed_form():
    """Judge (b): peak response of a half-sine pulse vs the closed form."""
    F0, td, fn = 1.0e6, 0.05, 2.0
    t = np.arange(0.0, 0.30, 1.0 / 4000.0)
    force = np.where(t <= td, F0 * np.sin(math.pi * t / td), 0.0)
    q_num = sdof_response(t, force, fn, 0.0)
    q_cf = half_sine_closed_form(t, F0, td, fn)
    peak_rel = abs(q_num.max() - q_cf.max()) / q_cf.max()
    assert peak_rel < 0.03, peak_rel
    assert peak_rel < 1e-3, "the exact recurrence should beat the 3 % judge"
    # the same in stress units through the modal channel API
    ch = [ModalChannel(fn, 0.0, 6.2)]
    sigma = modal_stress_history(t, force, ch)
    assert sigma.max() == pytest.approx(6.2 * q_num.max(), rel=1e-12)


def test_sdof_matches_duhamel_convolution_when_damped():
    F0, td, fn, zeta = 1.0e6, 0.05, 2.0, 0.05
    t = np.arange(0.0, 0.12, 1.0 / 5000.0)
    force = np.where(t <= td, F0 * np.sin(math.pi * t / td), 0.0)
    q_rec = sdof_response(t, force, fn, zeta)
    q_conv = duhamel_convolution(t, force, fn, zeta)
    assert np.max(np.abs(q_rec - q_conv)) / np.max(np.abs(q_conv)) < 1e-3


def test_sdof_static_limit_is_unity_gain():
    t = np.arange(0.0, 2.0, 1e-3)
    force = np.ones_like(t) * 3.0
    q = sdof_response(t, force, 1.0, 0.05)
    assert q[-1] == pytest.approx(3.0, rel=1e-6)


def test_narrowband_rayleigh_limit_for_a_sine():
    """A pure sine is the narrow-band limit: Rayleigh with the sine's RMS."""
    curve = PowerLawSNCurve(3.0)
    amplitude = 40e6
    periods = 200
    rms = amplitude / math.sqrt(2.0)
    d_rain = periods / curve.cycles_to_failure(np.array([2.0 * amplitude]))[0]
    d_ray = narrowband_rayleigh_damage(curve, rms, periods)
    # the Rayleigh integral of a sine's single range is wider than the single
    # cycle, so the narrow-band formula is only an order-of-magnitude estimate
    assert 0.5 < d_ray / d_rain < 5.0


def test_response_rejects_bad_input():
    with pytest.raises(ValueError):
        ModalChannel(0.0, 0.01, 1.0)
    with pytest.raises(ValueError):
        ModalChannel(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        sdof_response([0.0, 1.0], [1.0], 1.0)
    with pytest.raises(ValueError):
        half_sine_closed_form([0.0, 1.0], 1.0, 0.05, 2.0, damping=0.01)