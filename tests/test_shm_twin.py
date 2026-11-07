"""End-to-end tests of the streaming ice-load twin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tensordem.shm.model import SHMConfig
from tensordem.shm.twin import AcquisitionConfig, IceLoadTwin, simulate_strain

ROOT = Path(__file__).resolve().parents[1]


def triangle_load(t, rate, amp, rise_fraction=0.8):
    ph = np.mod(t * rate, 1.0)
    return amp * np.where(ph < rise_fraction, ph / rise_fraction,
                          (1.0 - ph) / (1.0 - rise_fraction))


@pytest.fixture(scope="module")
def twin():
    return IceLoadTwin(SHMConfig(n_modes=3), AcquisitionConfig())


def test_end_to_end_metrics_and_latency(twin):
    """Node criteria (b) + (e): inversion accuracy and < 50 ms per window."""
    dt = twin.dt
    t = np.arange(0.0, 80.0, dt)
    load = triangle_load(t, 0.5, 8.0e6)
    sim = twin.simulate(load, snr_db=20.0, seed=21)
    res = twin.process(sim["strain"], reference_load=load, q_truth=sim["q"])
    assert res.metrics["correlation"] > 0.95
    assert res.metrics["peak_error"] < 0.15
    latency = res.latency_summary()
    assert latency["median_ms"] < 50.0
    assert latency["p99_ms"] < 50.0
    assert latency["within_limit"], latency
    payload = res.as_dict()
    for key in ("split_load", "split_strain", "fatigue", "latency", "metrics", "rom"):
        assert key in payload
    assert payload["modal_q_error"]["mode_1"]["relative_error"] < 0.15


def test_chunk_size_invariance(twin):
    """Streaming with 1 s and 2 s windows must agree (no double counting)."""
    dt = twin.dt
    t = np.arange(0.0, 60.0, dt)
    load = triangle_load(t, 0.4, 6.0e6)
    sim = twin.simulate(load, snr_db=20.0, seed=8)
    coarse = IceLoadTwin(SHMConfig(n_modes=3), AcquisitionConfig(chunk_s=2.0))
    fine = twin.process(sim["strain"])
    block = coarse.process(sim["strain"])
    n = fine.point_force.size
    trim = slice(int(5.0 / dt), n - int(5.0 / dt))
    scale = np.max(np.abs(fine.point_force[trim]))
    err = np.max(np.abs(fine.point_force[trim] - block.point_force[trim])) / scale
    assert err < 0.05, err
    assert block.fatigue["cycles"] == pytest.approx(fine.fatigue["cycles"], rel=0.10)


def test_twin_rejects_short_or_non_finite_records(twin):
    with pytest.raises(ValueError):
        twin.process(np.zeros((twin.rom.H.shape[0], 10)))
    with pytest.raises(ValueError):
        twin.process(np.full((twin.rom.H.shape[0], 500), np.nan))


def test_simulate_strain_scales_noise_with_snr():
    rom_twin = IceLoadTwin(SHMConfig(n_modes=2), AcquisitionConfig())
    t = np.arange(0.0, 40.0, 0.01)
    load = triangle_load(t, 0.5, 5.0e6)
    low = rom_twin.simulate(load, snr_db=10.0, seed=1)
    high = rom_twin.simulate(load, snr_db=30.0, seed=1)
    assert low["noise_sigma"] > high["noise_sigma"]
    measured = 20 * np.log10(np.sqrt(np.mean(low["strain_clean"] ** 2)) / low["noise_sigma"])
    assert measured == pytest.approx(10.0, abs=0.2)


def test_dem_event_through_twin_if_cached(twin):
    """Replay a cached DEM ice event end-to-end (see scripts/run_shm_twin_case.py)."""
    cache = ROOT / "artifacts" / "shm_twin" / "dem_events.npz"
    if not cache.exists():
        pytest.skip("DEM event cache missing; run scripts/run_shm_twin_case.py")
    data = np.load(cache)
    t_raw, f_raw = data["t1"], data["f1"]
    grid = np.arange(0.0, t_raw[-1], twin.dt)
    load = np.interp(grid, t_raw, f_raw)
    load = load / load.max() * 1.0e7
    sim = twin.simulate(load, snr_db=20.0, seed=3)
    res = twin.process(sim["strain"], reference_load=load)
    assert res.metrics["correlation"] > 0.90
    assert res.metrics["peak_error"] < 0.25
    assert np.isfinite(res.point_force).all()