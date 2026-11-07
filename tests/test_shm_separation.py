"""Tests of the 0.5 Hz rigid-body / elastic separation."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.shm.separation import separation_criteria, split_rigid_elastic

DT = 0.01


def test_quasi_static_record_is_rigid_dominated():
    """Node criterion (c), first half: < 0.5 Hz energy fraction > 80 %."""
    t = np.arange(0.0, 200.0, DT)
    signal = 6.0e6 * (0.5 - 0.5 * np.cos(2 * np.pi * 0.15 * t)) ** 2
    split = split_rigid_elastic(signal, DT, cutoff_hz=0.5)
    assert split.rigid_fraction > 0.80, split.as_dict()
    verdict = separation_criteria(split.rigid_fraction, split.elastic_fraction)
    assert verdict["rigid_dominated"]


def test_vibratory_record_is_elastic_dominated():
    """Node criterion (c), second half: > 0.5 Hz energy fraction > 70 %."""
    t = np.arange(0.0, 200.0, DT)
    signal = np.sin(2 * np.pi * 0.9 * t) + 0.4 * np.sin(2 * np.pi * 2.1 * t)
    split = split_rigid_elastic(signal, DT, cutoff_hz=0.5)
    assert split.elastic_fraction > 0.70, split.as_dict()
    verdict = separation_criteria(split.rigid_fraction, split.elastic_fraction)
    assert verdict["elastic_dominated"]


def test_bands_reconstruct_the_signal():
    """rigid + elastic == signal, and the energy closure is within the cross term."""
    rng = np.random.default_rng(3)
    signal = np.cumsum(rng.normal(0.0, 1.0, 3000)) * 1e4
    split = split_rigid_elastic(signal, DT, cutoff_hz=0.5)
    assert np.allclose(split.rigid + split.elastic, signal, atol=1e-9)
    assert split.as_dict()["energy_closure"] == pytest.approx(1.0, abs=0.15)


def test_degenerate_records_and_validation():
    with pytest.raises(ValueError):
        split_rigid_elastic(np.zeros(800), DT, cutoff_hz=0.0)
    with pytest.raises(ValueError):
        split_rigid_elastic(np.zeros(8), DT, cutoff_hz=0.5)
    zero = split_rigid_elastic(np.zeros(800), DT, cutoff_hz=0.5)
    assert zero.rigid_fraction == 0.0 and zero.elastic_fraction == 0.0