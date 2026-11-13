"""Regression guard for the R3 biaxial-ensemble post-peak reference selection.

Board audit m17722 (CASE-BIAXIAL-SHEAR-BAND, mission m17693) found that the
committed R3 driver could not reproduce its own ensemble: at HEAD the
``post_peak`` cumulative window was selected with an exact-equality test
(``abs(strain - peak_strain) < 1e-9``), which returns an empty list whenever
``peak["axial_strain"]`` does not land bit-for-bit on the snapshot grid.  The
empty list then reached ``start[0]`` and raised ``IndexError`` **after** the DEM
specimen had already run, aborting the specimen without a measurement
(reproduced 2026-11-13 on ``anchor_s11``, 110.6 s wall, full traceback
``run_r3_ensemble.py:150``).

These tests pin the repaired behaviour of the two helpers the driver now uses,
without running any DEM (seconds, no torch):
``_nearest_snapshot``  -- closest snapshot, ``None`` for an empty record;
``_cumulative_starts`` -- ``(label, reference or None)`` for both windows.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MOD_PATH = (pathlib.Path(__file__).resolve().parents[1]
             / "benchmarks" / "biaxial_shear_band" / "run_r3_ensemble.py")


@pytest.fixture(scope="module")
def r3():
    spec = importlib.util.spec_from_file_location("run_r3_ensemble_audit", _MOD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshots(n: int = 13) -> list[dict]:
    return [{"strain": round(0.01 * i, 12), "positions": None, "window": None}
            for i in range(n)]


def test_nearest_snapshot_exact_match(r3):
    snaps = _snapshots()
    assert r3._nearest_snapshot(snaps, 0.05)["strain"] == pytest.approx(0.05)


def test_nearest_snapshot_off_grid_peak_is_a_snapshot_not_a_list(r3):
    """The historical defect: an off-grid peak produced ``[]`` then IndexError."""
    snaps = _snapshots()
    peak = 0.107 + 1e-9                      # off the 0.01 grid, as torch gives
    start = r3._nearest_snapshot(snaps, peak)
    assert isinstance(start, dict)           # not the old empty list
    assert start["strain"] == pytest.approx(0.11)
    assert abs(start["strain"] - peak) == pytest.approx(0.003, abs=1e-3)


def test_nearest_snapshot_empty_record_is_none(r3):
    assert r3._nearest_snapshot([], 0.1) is None


def test_cumulative_starts_uses_first_late_pair_and_nearest_peak(r3):
    snaps = _snapshots()
    pairs = list(zip(snaps[5:-1], snaps[6:]))
    starts = dict(r3._cumulative_starts(snaps, pairs, 0.107))
    assert set(starts) == {"late_window", "post_peak"}
    assert starts["late_window"] is snaps[5]
    assert starts["post_peak"] is snaps[11]
    # the driving loop skips ``None`` starts instead of indexing them
    for label, start in r3._cumulative_starts(snaps, pairs, 0.107):
        assert start is not None


def test_cumulative_starts_without_pairs_returns_none_references(r3):
    snaps = _snapshots()
    starts = dict(r3._cumulative_starts(snaps, [], 0.107))
    assert starts["late_window"] is None
    assert starts["post_peak"] is not None      # peak is known independently
    assert dict(r3._cumulative_starts([], [], 0.107))["post_peak"] is None