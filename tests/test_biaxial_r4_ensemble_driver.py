"""Regression tests for the R4/R5 ensemble driver.

The defect these tests pin down: ``run_r4_cumulative_ensemble.main`` called
``pool.imap_unordered(_run_cached, configs, [cache]*n, [resume]*n)``.
``imap_unordered`` accepts one item argument, so **any** ``--workers > 1`` run
died with ``TypeError: Pool.imap_unordered() takes from 3 to 4 positional
arguments but 5 were given`` before executing a single specimen.  The round had
only ever been driven through the single-specimen shard wrapper, which calls
``_run_cached`` directly, so the broken branch was never reached in production.

The tests use a stub multiprocessing context so no child process (and no DEM
run) is needed: the pool contract exercised is exactly the one the real pool
offers (``imap_unordered(func, iterable)``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "benchmarks", "biaxial_shear_band",
                      "run_r4_cumulative_ensemble.py")


def _driver():
    spec = importlib.util.spec_from_file_location("r4_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["r4_driver"] = module
    spec.loader.exec_module(module)
    return module


class _StubPool:
    """Minimal ``imap_unordered(func, iterable)`` pool, in-process."""

    used: list = []

    def __init__(self, size):
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, iterable, chunksize=None):
        # a real Pool accepts a third *chunksize* argument; anything beyond
        # that is the historical defect
        _StubPool.used.append((self.size, func))
        for item in iterable:
            yield func(item)


class _StubContext:
    def Pool(self, size):                          # noqa: N802 - mirrors mp API
        return _StubPool(size)


def _canned_record(config, cache_dir=None, resume=False):
    """Cheap stand-in for ``_run_cached`` — no DEM, no I/O."""
    theta = 56.0 if config["seed"] % 2 else 54.0
    return {
        "name": config["name"], "axis": config["axis"], "seed": config["seed"],
        "confining_stress": config["confining_stress"],
        "one_step": {"theta_deg": theta - 1.0, "theta_std_deg": 2.0,
                     "resolved": True},
        "cumulative": {"late_window": {
            "strain_span": 0.04, "slip_over_spacing": 1.0,
            "radon": theta, "radon_z": 12.0, "radon_eta2": 0.5,
            "resolved": True, "bootstrap_radon": {"std_deg": 0.9}}},
        "friction_angle_deg": 35.0, "dilation_angle_deg": 50.0,
        "theory_angles_deg": {}, "wall_time_s": 0.0,
    }


def test_parallel_branch_dispatches(monkeypatch, tmp_path):
    """``--workers > 1`` must run the whole ensemble (this failed with TypeError)."""
    mod = _driver()
    _StubPool.used = []
    monkeypatch.setattr(mod.mp, "get_context", lambda method=None: _StubContext())
    monkeypatch.setattr(mod, "_run_cached", _canned_record)

    out = tmp_path / "ensemble.json"
    mod.main(["--workers", "2", "--quick", "--out", str(out),
              "--figure", str(tmp_path / "f.png"), "--cache", str(tmp_path / "c")])

    payload = json.loads(out.read_text())
    assert len(payload["specimens"]) == 8          # quick = 8 anchors, no p200
    assert all("error" not in spec for spec in payload["specimens"])
    assert _StubPool.used and _StubPool.used[0][0] == 2, "pool was not used"
    # the verdict must be assembled from the dispatched records
    stats = payload["verdict"]["primary_cumulative_late_window_radon"]
    assert stats["n"] == 8 and math.isfinite(stats["mean_deg"])


def test_serial_branch_matches_parallel(monkeypatch, tmp_path):
    """Serial and parallel dispatch must produce the same verdict."""
    mod = _driver()
    monkeypatch.setattr(mod.mp, "get_context", lambda method=None: _StubContext())
    monkeypatch.setattr(mod, "_run_cached", _canned_record)

    serial_out = tmp_path / "serial.json"
    mod.main(["--workers", "1", "--quick", "--out", str(serial_out),
              "--figure", str(tmp_path / "f1.png"), "--cache", str(tmp_path / "c1")])
    parallel_out = tmp_path / "parallel.json"
    mod.main(["--workers", "3", "--quick", "--out", str(parallel_out),
              "--figure", str(tmp_path / "f2.png"), "--cache", str(tmp_path / "c2")])

    a = json.loads(serial_out.read_text())["verdict"]
    b = json.loads(parallel_out.read_text())["verdict"]
    for key in ("primary_cumulative_late_window_radon", "specimens_resolved",
                "failure_modes", "passed"):
        assert a[key] == b[key], key


def test_per_specimen_failure_does_not_kill_the_pool(tmp_path):
    """``_run_cached_safe`` converts an exception into an error record."""
    mod = _driver()

    def boom(config, cache_dir, resume):
        raise RuntimeError("specimen exploded")

    mod._run_cached = boom
    record = mod._run_cached_safe({"name": "x", "axis": "anchor", "seed": 1,
                                   "confining_stress": 1e5}, str(tmp_path), False)
    assert "error" in record and "specimen exploded" in record["error"]
    assert record["name"] == "x"


def _record(name, late=None, axis="anchor", confining_stress=1.0e5):
    return {"name": name, "axis": axis, "seed": 1,
            "confining_stress": confining_stress,
            "one_step": {"theta_deg": math.nan, "theta_std_deg": math.nan,
                         "resolved": False},
            "cumulative": {} if late is None else {"late_window": late},
            "friction_angle_deg": 35.0, "dilation_angle_deg": 50.0}


def test_failure_modes_are_not_pooled():
    """Three physically distinct outcomes must be classified separately."""
    mod = _driver()
    resolved = {"strain_span": 0.04, "slip_over_spacing": 1.0, "radon": 56.0,
                "radon_z": 12.0, "radon_eta2": 0.5, "resolved": True}
    degenerate = {"strain_span": 0.0, "slip_over_spacing": 0.0,
                  "error": "BandMeasurementError('local strain field is degenerate')",
                  "resolved": False}
    weak = {"strain_span": 0.01, "slip_over_spacing": 0.2, "radon": 70.0,
            "radon_z": 0.5, "radon_eta2": 0.05, "resolved": False}
    records = [_record("a", resolved), _record("b", degenerate),
               _record("c", weak), _record("d", None)]
    verdict = mod._analyse(records)
    modes = verdict["failure_modes"]
    assert modes["resolved"]["specimens"] == ["a"]
    assert modes["degenerate_window"]["specimens"] == ["b"]
    assert modes["unresolved_measurement"]["specimens"] == ["c"]
    assert modes["no_post_peak_window"]["specimens"] == ["d"]
    assert verdict["specimens_resolved"] == 1


def test_literature_band_is_reported_separately_from_tolerance():
    """55-60 deg (measured range) and 60+/-15 % (tolerance) are different claims."""
    mod = _driver()

    def late(theta):
        return {"strain_span": 0.04, "slip_over_spacing": 1.0, "radon": theta,
                "radon_z": 12.0, "radon_eta2": 0.5, "resolved": True}

    inside = mod._analyse([_record(f"s{i}", late(56.0)) for i in range(8)]) 
    assert inside["within_literature_band"] is True
    assert inside["primary_cumulative_late_window_radon"]["within_tolerance"] is True
    assert inside["passed"] is True                      # 8/8 resolved

    below = mod._analyse([_record(f"s{i}", late(50.0)) for i in range(8)])
    # 16.7 % off the 60 deg anchor: outside tolerance but the flag must not lie
    assert below["within_literature_band"] is False
    assert below["primary_cumulative_late_window_radon"]["within_tolerance"] is False
    assert below["passed"] is False


def test_resolution_bar_is_enforced():
    """Accuracy passing does not rescue a 5/8 ensemble: the conjunction decides."""
    mod = _driver()

    def late(theta, resolved=True):
        return {"strain_span": 0.04, "slip_over_spacing": 1.0, "radon": theta,
                "radon_z": 12.0 if resolved else 0.5,
                "radon_eta2": 0.5 if resolved else 0.05,
                "resolved": resolved}

    records = [_record(f"s{i}", late(55.0)) for i in range(5)]
    records += [_record(f"n{i}", late(80.0, resolved=False)) for i in range(3)]
    verdict = mod._analyse(records)
    assert verdict["primary_cumulative_late_window_radon"]["within_tolerance"] is True
    assert verdict["specimens_resolved"] == 5
    assert verdict["passed"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))