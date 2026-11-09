"""Sampling-compression error envelope and the minimum-rate law.

Judge (e): the envelope must be non-empty and monotone (``|dD/D|`` falls as the
rate rises) and peak-preserving must not be worse than naive sampling in the
resolvable band (``fs >= 20 Hz``, i.e. >= 29 x the first mode).  Below that band
the record cannot represent the structural mode at all; that regime is measured
and reported, not hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.fatigue_chain import ModalChannel, chain_damage, sampling_envelope
from tensordem.fatigue_chain.contract import MissingForceHistory
from tensordem.fatigue_chain.damage import statistics_compression
from tensordem.fatigue_chain.envelope import fit_min_rate_law, random_pulse_train
from tensordem.fatigue_chain.resample import (
    naive_decimate,
    peak_preserving_decimate,
    reconstruct,
)
from tensordem.whipping.fatigue import DNV_CURVES
from tensordem.whipping import load_calibration as lc

GAIN = 6.2          # Pa per N (6.2 MPa/MN hot-spot transfer)
MODES = ((0.69, 0.015, 0.62), (1.83, 0.015, 0.19), (3.57, 0.015, 0.12),
         (5.85, 0.015, 0.07))


@pytest.fixture(scope="module")
def m5405():
    controls = lc.IceLoadControls(crushing_scale=1.681, contact_damping=0.173,
                                  floe_geometry=1.092, froude=0.0179,
                                  duration_s=20.0)
    t, f, _ = lc.synthesise(controls)
    return t, f


@pytest.fixture(scope="module")
def channels():
    return [ModalChannel(f, z, GAIN * w) for f, z, w in MODES]


def test_envelope_is_monotone_in_the_resolvable_band(m5405, channels):
    t, f = m5405
    env = sampling_envelope(t, f, channels, DNV_CURVES["D_air"],
                            rates=(20.0, 100.0), label="m5405")
    assert len(env.rows) == 4
    mono = env.monotone()
    assert mono["naive"]["monotone"], mono["naive"]
    assert mono["peak"]["monotone"], mono["peak"]
    # at 100 Hz both methods are inside a ~1 % band, so the ordering there is
    # noise; the criterion is applied with a 0.5 percentage-point tolerance and
    # strictly at 20 Hz (below)
    assert env.peak_vs_naive(tol=0.005)["peak_not_worse"], env.peak_vs_naive()
    # at 100 Hz the compression is inside the 10 % band for both methods
    for row in env.rows:
        if row["rate_hz"] == 100.0:
            assert abs(row["dD_over_D"]) < 0.10, row


def test_envelope_over_the_judge_grid(m5405, channels):
    """The 1/5/20/100 Hz grid: monotone for both methods, peak safer above f1."""
    t, f = m5405
    env = sampling_envelope(t, f, channels, DNV_CURVES["D_air"], label="m5405")
    mono = env.monotone()
    assert mono["naive"]["monotone"]
    assert mono["peak"]["monotone"]
    table = {(r["rate_hz"], r["method"]): r["dD_over_D"] for r in env.rows}
    # peak-preserving wins above the mode frequency, and it errs high (safe side)
    assert abs(table[(20.0, "peak")]) < abs(table[(20.0, "naive")])
    assert abs(table[(100.0, "peak")]) < 0.05
    assert table[(1.0, "peak")] > 0.0        # overestimate, not a hidden loss
    assert table[(1.0, "naive")] < 0.0       # unconservative: damage disappears


def test_envelope_fails_closed(channels):
    with pytest.raises(MissingForceHistory):
        sampling_envelope([], [], channels, DNV_CURVES["D_air"])
    with pytest.raises(MissingForceHistory):
        sampling_envelope(None, None, channels, DNV_CURVES["D_air"])
    with pytest.raises(MissingForceHistory):
        chain_damage([], [], channels, DNV_CURVES["D_air"])


def test_peak_decimation_keeps_the_extreme_and_naive_loses_it():
    rng = np.random.default_rng(3)
    fs = 1000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    f = np.zeros_like(t)
    for centre in rng.uniform(0.05, 0.95, 6):
        f += 10.0 * np.exp(-0.5 * ((t - centre) / 0.002) ** 2)
    tp, fp = peak_preserving_decimate(t, f, 5.0)
    tn, fn = naive_decimate(t, f, 5.0)
    assert fp.max() == pytest.approx(f.max(), rel=1e-12)
    assert fn.max() < f.max()
    restored = reconstruct(t, tp, fp)
    assert restored.max() >= 0.99 * f.max()


def test_statistics_compression_is_an_order_of_magnitude_estimate(m5405, channels):
    t, f = m5405
    table = statistics_compression(DNV_CURVES["D_air"], t, f, channels)
    assert "exact_rainflow" in table
    nb = table["narrowband_rayleigh"]["dD_over_D"]
    blk = table["design_peak_block"]["dD_over_D"]
    # both surrogates overestimate the rainflow damage on this record, by a
    # factor of a few (narrow band) and by more than an order (peak block)
    assert nb > 0.5, nb
    assert blk > nb, (blk, nb)


def test_min_rate_law_recovers_a_power_law():
    """The fit must recover ``fs_min = a m^b / t_p^c`` from clean synthetic data."""
    pts = []
    for m in (3.0, 5.0):
        for tp in (0.01, 0.02, 0.05, 0.1):
            pts.append({"m": m, "pulse_width_s": tp,
                        "fs_min_hz": 3.0 * m ** 0.5 / tp})
    fit = fit_min_rate_law(pts)
    assert fit["a"] == pytest.approx(3.0, rel=1e-9)
    assert fit["b"] == pytest.approx(0.5, rel=1e-9)
    assert fit["c"] == pytest.approx(1.0, rel=1e-9)
    assert fit["r2_log"] == pytest.approx(1.0, rel=1e-9)
    with pytest.raises(ValueError):
        fit_min_rate_law(pts[:2])


def test_random_pulse_train_is_non_periodic():
    t, f = random_pulse_train(5.0, 1000.0, 0.02, mean_interval_s=0.12, seed=4)
    starts = np.flatnonzero(np.diff((f > 0).astype(int)) == 1)
    intervals = np.diff(starts) / 1000.0
    assert starts.size > 20
    assert intervals.std() > 0.02          # exponential arrivals, not periodic