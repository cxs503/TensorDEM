"""Tests of the load inversion (modal force -> physical ice load)."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.shm.load_inversion import (
    bandlimit,
    differentiate,
    invert_modal_force,
    load_error_metrics,
    savgol_cutoff_hz,
)
from tensordem.shm.model import HullGirderROM, SHMConfig
from tensordem.shm.twin import AcquisitionConfig, IceLoadTwin

DT = 0.01


def triangle_load(t, rate, amp, rise_fraction=0.8):
    ph = np.mod(t * rate, 1.0)
    return amp * np.where(ph < rise_fraction, ph / rise_fraction,
                          (1.0 - ph) / (1.0 - rise_fraction))


@pytest.fixture(scope="module")
def twin():
    return IceLoadTwin(SHMConfig(n_modes=3), AcquisitionConfig())


def test_crushing_cycle_inversion_acceptance(twin):
    """Node criterion (b): peak error < 15 %, energy error < 10 %."""
    dt = twin.dt
    t = np.arange(0.0, 100.0, dt)
    load = triangle_load(t, 0.5, 8.0e6)
    sim = twin.simulate(load, snr_db=20.0, seed=11)
    res = twin.process(sim["strain"], reference_load=load)
    m = res.metrics
    assert m["peak_error"] < 0.15, m["peak_error"]
    assert m["energy_error"] < 0.10, m["energy_error"]
    assert m["correlation"] > 0.95, m["correlation"]


def test_raw_peak_error_exposes_bandwidth_limit(twin):
    """Honest limitation: without band-limiting the peak error is much larger."""
    dt = twin.dt
    t = np.arange(0.0, 100.0, dt)
    load = 8.0e6 * np.mod(t * 0.5, 1.0)          # discontinuous brittle release
    sim = twin.simulate(load, snr_db=20.0, seed=5)
    res = twin.process(sim["strain"], reference_load=load)
    band = res.metrics
    raw = load_error_metrics(res.point_force, load, dt, cut_edges=0.05)
    assert band["peak_error"] < 0.25
    assert raw["peak_error"] > band["peak_error"]


def test_differentiation_of_analytic_signal():
    """Savitzky-Golay derivatives are exact for a low-order polynomial."""
    t = np.arange(0.0, 2.0, DT)
    signal = 3.0 + 2.0 * t + 0.5 * t**2
    d1 = differentiate(signal, DT, deriv=1, window=31, polyorder=3)
    d2 = differentiate(signal, DT, deriv=2, window=31, polyorder=3)
    assert np.allclose(d1[20:-20], 2.0 + t[20:-20], atol=1e-6)
    assert np.allclose(d2[20:-20], 1.0, atol=1e-6)


def test_inversion_is_linear(twin):
    """Doubling the ice load doubles the inverted force (linear operator)."""
    dt = twin.dt
    t = np.arange(0.0, 60.0, dt)
    load = triangle_load(t, 0.5, 5.0e6)
    sim1 = twin.simulate(load, snr_db=None)
    sim2 = twin.simulate(2.0 * load, snr_db=None)
    f1 = twin.process(sim1["strain"]).point_force
    f2 = twin.process(sim2["strain"]).point_force
    assert np.allclose(f2, 2.0 * f1, rtol=1e-8, atol=1e-3 * np.max(np.abs(f1)))


def test_static_load_recovered_quasi_statically(twin):
    """A constant load must be recovered as a constant (except start-up edge)."""
    dt = twin.dt
    t = np.arange(0.0, 150.0, dt)
    load = np.full(t.size, 5.0e6)
    sim = twin.simulate(load, snr_db=None)
    res = twin.process(sim["strain"])
    # the modal response starts from rest, so the first ~1/zeta*omega seconds carry
    # the start-up transient; the quasi-static regime is asserted afterwards
    interior = res.point_force[int(100.0 / dt):]
    assert np.max(np.abs(interior - 5.0e6)) / 5.0e6 < 0.01


def test_negative_control_zero_load(twin):
    """Zero load -> zero modal force and zero inverted load."""
    strain = np.zeros((twin.rom.H.shape[0], 400))
    res = twin.process(strain)
    assert np.allclose(res.point_force, 0.0, atol=1e-9)
    assert np.allclose(res.modal_force, 0.0, atol=1e-9)
    assert res.fatigue["damage"] == 0.0


def test_error_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        load_error_metrics(np.zeros(32), np.zeros(33), DT)


def test_bandlimit_and_window_validation():
    with pytest.raises(ValueError):
        bandlimit(np.zeros(100), DT, 0.0)
    with pytest.raises(ValueError):
        differentiate(np.zeros(100), DT, deriv=1, window=10)
    with pytest.raises(ValueError):
        differentiate(np.zeros(100), DT, deriv=3)
    assert 0.0 < savgol_cutoff_hz(51, DT) < 0.5 / DT