"""Tests of the streaming rainflow counter and the Miner damage accumulator."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.shm.fatigue_update import FatigueAccumulator, StreamingRainflow
from tensordem.shm.model import HullGirderROM, SHMConfig
from tensordem.shm.twin import AcquisitionConfig, IceLoadTwin
from tensordem.whipping.rainflow import rainflow as batch_rainflow

DT = 0.01


def triangle_load(t, rate, amp, rise_fraction=0.8):
    ph = np.mod(t * rate, 1.0)
    return amp * np.where(ph < rise_fraction, ph / rise_fraction,
                          (1.0 - ph) / (1.0 - rise_fraction))


def test_streaming_matches_batch_rainflow():
    """With gate = 0 the streaming counter reproduces ASTM E1049-85 exactly."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0.0, 1.0, 2000)
    counter = StreamingRainflow(gate=0.0)
    counter.update(signal)
    streamed = counter.all_cycles()
    batch = batch_rainflow(signal)
    assert len(streamed) == len(batch)
    for a, b in zip(streamed, batch):
        assert a.count == pytest.approx(b.count, abs=1e-12)
        assert a.range == pytest.approx(b.range, rel=1e-12, abs=1e-12)


def test_streaming_is_chunk_invariant():
    """Feeding the record in chunks gives the same cycles as one shot."""
    rng = np.random.default_rng(5)
    signal = np.cumsum(rng.normal(0.0, 1.0, 3000))
    whole = StreamingRainflow(gate=0.0)
    whole.update(signal)
    chunks = StreamingRainflow(gate=0.0)
    for start in range(0, signal.size, 137):
        chunks.update(signal[start:start + 137])
    assert len(chunks.all_cycles()) == len(whole.all_cycles())
    assert (sum(c.count for c in chunks.all_cycles()) ==
            pytest.approx(sum(c.count for c in whole.all_cycles())))


def test_closed_cycle_invariant():
    """sum(counts) = (n_turning - 1) / 2 for a closed turning-point sequence."""
    rng = np.random.default_rng(9)
    signal = rng.normal(0.0, 1.0, 501)
    counter = StreamingRainflow(gate=0.0)
    counter.update(signal)
    closed = counter.closed
    if closed:
        total = sum(c.count for c in closed)
        assert total <= 0.5 * (counter.n_turning - 1) + 1e-9


def test_gate_suppresses_noise_cycles():
    """A hysteresis gate removes noise cycles; without it the count explodes."""
    t = np.arange(0.0, 60.0, DT)
    signal = 40e6 * np.sin(2 * np.pi * 0.5 * t) + np.random.default_rng(2).normal(0, 1.5e6, t.size)
    free = StreamingRainflow(gate=0.0)
    free.update(signal)
    gated = StreamingRainflow(gate=2.0e6)
    gated.update(signal)
    assert len(gated.closed) < 0.5 * len(free.closed)
    assert len(gated.closed) > 100         # the physical cycles survive


def test_damage_is_monotone_and_zero_without_load():
    acc = FatigueAccumulator(gate=1.0e6)
    history = []
    rng = np.random.default_rng(4)
    for _ in range(20):
        acc.update(rng.normal(0.0, 5.0e6, 100))
        history.append(acc.damage_closed)
    assert all(b >= a - 1e-18 for a, b in zip(history, history[1:]))
    empty = FatigueAccumulator(gate=1.0e6)
    empty.update(np.zeros(500))
    assert empty.total_damage() == 0.0


def test_miner_damage_scales_with_stress_cube():
    """DNV D-curve: damage of a fixed cycle count scales with dSigma^3."""
    acc_low = FatigueAccumulator(gate=0.0)
    acc_high = FatigueAccumulator(gate=0.0)
    # amplitudes chosen so that both stress ranges sit above the DNV D kink
    # (52.6 MPa), i.e. inside the m = 3 branch of the S-N curve
    for acc, amp in ((acc_low, 40e6), (acc_high, 80e6)):
        t = np.arange(0.0, 20.0, DT)
        acc.update(amp * np.sin(2 * np.pi * 0.5 * t))
    ratio = acc_high.total_damage() / acc_low.total_damage()
    assert ratio == pytest.approx(8.0, rel=0.02)


def test_thousand_cycle_fatigue_acceptance():
    """Node criterion (d): 1000 crushing cycles, online D vs direct rainflow < 10 %."""
    twin = IceLoadTwin(SHMConfig(n_modes=3), AcquisitionConfig())
    dt = twin.dt
    n_cycles, rate = 1000, 0.5
    t = np.arange(0.0, n_cycles / rate, dt)
    load = triangle_load(t, rate, 8.0e6)
    sim = twin.simulate(load, snr_db=20.0, seed=3)
    res = twin.process(sim["strain"], reference_load=load)

    def batch(stress):
        acc = FatigueAccumulator(gate=1.0e6)
        acc.update(stress)
        return acc.total_damage()

    d_same = batch(res.sigma_deck)
    d_truth = batch(sim["sigma_deck"])
    assert d_same > 0.0 and d_truth > 0.0
    assert abs(res.fatigue["damage"] - d_same) / d_same < 1e-4   # streaming == batch
    assert abs(res.fatigue["damage"] - d_truth) / d_truth < 0.10  # physics accuracy
    assert 900 < res.fatigue["cycles"] < 1200