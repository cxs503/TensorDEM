"""Tests of the DEM online comparison and the model-correction trigger."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.shm.dem_compare import compare_events, correction_trigger, event_metrics

DT = 0.01


def _pulse_train(t, rate=0.4, amp=1.0e7, width=0.15):
    phase = np.mod(t * rate, 1.0)
    return amp * np.where(phase < width, np.sin(np.pi * phase / width) ** 2, 0.0)


def test_identical_records_give_zero_error():
    t = np.arange(0.0, 60.0, DT)
    ref = _pulse_train(t)
    m = event_metrics(ref, ref, DT, bandlimit_hz=2.0)
    assert m["peak_error"] < 1e-12
    assert m["energy_error"] < 1e-12
    assert m["correlation"] == pytest.approx(1.0, abs=1e-9)
    assert m["lag_samples"] == 0


def test_scaled_record_reports_peak_error():
    t = np.arange(0.0, 60.0, DT)
    ref = _pulse_train(t)
    est = 1.2 * ref
    m = event_metrics(est, ref, DT, bandlimit_hz=2.0)
    assert m["peak_error"] == pytest.approx(0.2, abs=0.01)
    assert m["energy_error"] == pytest.approx(0.44, abs=0.03)   # variance ratio 1.44


def test_lag_detection():
    t = np.arange(0.0, 60.0, DT)
    ref = _pulse_train(t)
    est = np.roll(ref, 10)
    m = event_metrics(est, ref, DT, bandlimit_hz=2.0, cut_edges=0.15)
    assert m["lag_samples"] == pytest.approx(10, abs=2)


def test_correction_trigger_thresholds():
    good = {"peak_error": 0.10, "energy_error": 0.08, "correlation": 0.95,
            "peak_reference_N": 1.0e7, "peak_estimate_N": 1.05e7}
    assert correction_trigger(good)["triggered"] is False
    bad = {"peak_error": 0.45, "energy_error": 0.12, "correlation": 0.92,
           "peak_reference_N": 1.0e7, "peak_estimate_N": 0.7e7}
    verdict = correction_trigger(bad)
    assert verdict["triggered"] is True and verdict["peak_exceeded"] is True
    assert verdict["suggested_amplitude_scale"] == pytest.approx(1.0e7 / 0.7e7, rel=1e-3)
    shape = {"peak_error": 0.05, "energy_error": 0.05, "correlation": 0.40,
             "peak_reference_N": 1.0e7, "peak_estimate_N": 1.0e7}
    assert correction_trigger(shape)["shape_mismatch"] is True


def test_compare_events_aggregates_five_events():
    rng = np.random.default_rng(12)
    t = np.arange(0.0, 40.0, DT)
    records = []
    for k in range(5):
        ref = _pulse_train(t, rate=0.3 + 0.05 * k, amp=1.0e7)
        scale = 1.0 + 0.05 * (k - 2)
        records.append({"estimate": scale * ref + rng.normal(0, 2e5, t.size),
                        "reference": ref, "dt": DT, "label": f"E{k + 1}",
                        "bandlimit_hz": 2.0})
    summary = compare_events(records, min_events=5)
    assert summary["n_events"] == 5
    assert summary["n_events_meets_minimum"] is True
    assert summary["correlation_min"] > 0.85
    assert summary["peak_error_max"] < 0.20
    assert summary["n_corrections_triggered"] == 0


def test_compare_events_requires_records():
    with pytest.raises(ValueError):
        compare_events([])