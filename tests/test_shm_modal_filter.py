"""Tests of the modal filter, including the 1 % analytic anchor at 20 dB SNR."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.shm.model import HullGirderROM, SHMConfig
from tensordem.shm.modal_filter import ModalFilter, harmonic_modal_estimate
from tensordem.whipping.response import modal_response

DT = 0.01


@pytest.fixture(scope="module")
def rom() -> HullGirderROM:
    return HullGirderROM(SHMConfig(n_modes=2))


def _modal_sine(rom, mode, amplitude=2.0e5, duration=240.0):
    """Steady-state single-mode excitation of mode ``mode`` (truth and strain)."""
    t = np.arange(0.0, duration, DT)
    p = np.zeros((rom.n_modes, t.size))
    p[mode] = amplitude * np.sin(rom.omega[mode] * t)
    solution = modal_response(t, p, rom.omega, rom.zeta)
    return t, solution.q, rom.strain_history(solution.q)


def _relative(error, reference):
    return float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(reference**2)))


def test_noiseless_single_mode_recovery(rom):
    """Least squares on an exact strain vector returns the exact modal coordinates."""
    _, truth, clean = _modal_sine(rom, 0, duration=20.0)
    q = ModalFilter(rom.H).apply(clean).q
    assert _relative(q - truth, truth) < 1e-10


def test_analytic_anchor_harmonic_one_percent_at_20db(rom):
    """Node criterion (a): single-mode sine, SNR 20 dB -> q_n error < 1 %."""
    t, truth, clean = _modal_sine(rom, 0)
    rms = float(np.sqrt(np.mean(clean**2)))
    snr_db = 20.0
    rng = np.random.default_rng(2024)
    strain = clean + rng.normal(0.0, rms / 10 ** (snr_db / 20.0), clean.shape)
    filt = ModalFilter(rom.H)
    q = harmonic_modal_estimate(strain, DT, rom.omega, window_s=5.0, estimator=filt)
    trim = int(100.0 / DT)
    sl = slice(trim, t.size - trim)
    err = _relative(q[0, sl] - truth[0, sl], truth[0, sl])
    assert err < 0.01, f"harmonic modal error {err:.4%}"


def test_instantaneous_estimator_is_noise_limited(rom):
    """Negative control: the per-sample filter alone cannot beat the 10 % noise floor.

    Documented, not a defect: at a 20 dB sensor SNR the unbiased least-squares
    estimate carries a ~5-15 % random error, which is why the anchor uses the
    harmonic estimator and why the load inversion band-limits its comparison.
    """
    _, truth, clean = _modal_sine(rom, 0)
    rms = float(np.sqrt(np.mean(clean**2)))
    rng = np.random.default_rng(7)
    strain = clean + rng.normal(0.0, rms / 10.0, clean.shape)
    q = ModalFilter(rom.H).apply(strain).q
    sl = slice(int(100.0 / DT), -int(100.0 / DT))
    err = _relative(q[0, sl] - truth[0, sl], truth[0, sl])
    assert 0.02 < err < 0.20, err


def test_snr_sweep_monotone_error(rom):
    """Node criterion (g): error falls with SNR over the 10-30 dB sweep."""
    _, truth, clean = _modal_sine(rom, 0, duration=240.0)
    rms = float(np.sqrt(np.mean(clean**2)))
    filt = ModalFilter(rom.H)
    errors = {}
    for snr_db in (10.0, 20.0, 30.0):
        rng = np.random.default_rng(11)
        strain = clean + rng.normal(0.0, rms / 10 ** (snr_db / 20.0), clean.shape)
        q = harmonic_modal_estimate(strain, DT, rom.omega, window_s=5.0, estimator=filt)
        sl = slice(int(100.0 / DT), -int(100.0 / DT))
        errors[snr_db] = _relative(q[0, sl] - truth[0, sl], truth[0, sl])
    assert errors[30.0] < errors[20.0] < errors[10.0]
    assert errors[10.0] < 0.05


def test_negative_control_zero_strain(rom):
    """Zero measurement -> zero modal coordinates and zero residual."""
    strain = np.zeros((rom.H.shape[0], 200))
    result = ModalFilter(rom.H).apply(strain)
    assert np.allclose(result.q, 0.0)
    assert result.residual_rms() == 0.0


def test_tikhonov_reduces_noise_gain(rom):
    """Regularisation trades bias for noise amplification (as documented)."""
    plain = ModalFilter(rom.H)
    regularised = ModalFilter(rom.H, regularization=1e-2)
    assert regularised.noise_gain < plain.noise_gain
    # with a small lambda the response to a clean single-mode strain stays close
    _, _, clean = _modal_sine(rom, 0, duration=10.0)
    q_plain = plain.apply(clean).q
    q_reg = regularised.apply(clean).q
    assert np.max(np.abs(q_reg - q_plain)) < 0.05 * np.max(np.abs(q_plain))


def test_negative_regularization_rejected(rom):
    with pytest.raises(ValueError):
        ModalFilter(rom.H, regularization=-1.0)