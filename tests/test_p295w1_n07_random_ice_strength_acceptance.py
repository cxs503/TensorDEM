"""Maturity-evidence tests for P295W1-N07 (L2 entry, L3 caller, L4 benchmark).

The physics/statistics of the module are covered by
``tests/test_p295w1_n07_random_ice_strength.py``.  This file covers what the
maturity gate asks for and the numerical tests cannot see:

* the **entry point** (``python -m tensordem.random_ice_strength``) resolves,
  prints the criteria and returns the verdict as an exit code;
* the **integration caller** -- :class:`tensordem.ice_channel3d.slab.SlabChannelSim`
  really consumes the random-strength ensemble, and the deterministic default
  is byte-for-byte unchanged when the switch is off;
* the **benchmark artifact** is internally consistent (every non-advisory row
  within tolerance) and the predeclared thresholds still match the mission
  brief (S1-S7 numbers), so a future edit cannot silently relax a criterion.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensordem.ice_channel3d.slab import (
    IceSheetProps,
    SlabChannelSim,
    SlabConfig,
    stochastic_bending_capacity_per_width_n_per_m,
)
from tensordem.random_ice_strength import CRITERIA

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/artifacts/p295w1_n07_random_ice_strength_benchmark.json"
ACCEPT = ROOT / "runs/P295W1_N07/acceptance.json"
STUDY = ROOT / "runs/P295W1_N07/study.json"


def _ice() -> IceSheetProps:
    return IceSheetProps(thickness_m=1.0, flexural_strength_pa=0.54e6,
                         compressive_strength_pa=2.0e6, elastic_modulus_pa=5.0e9)


# ------------------------------------------------------------------ L2 entry
def test_module_entry_point_resolves():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    proc = subprocess.run([sys.executable, "-m", "tensordem.random_ice_strength", "--help"],
                          cwd=ROOT, capture_output=True, text=True, timeout=180,
                          env=env)
    assert proc.returncode == 0, proc.stderr
    assert "--out-dir" in proc.stdout and "--artifact-dir" in proc.stdout


def test_entry_point_exit_code_follows_the_verdict(monkeypatch, capsys):
    import tensordem.random_ice_strength.study as study_mod
    from tensordem.random_ice_strength.__main__ import main

    fake = {"checks": {"criterion_a": {"pass": True, "value": 0.99},
                       "criterion_b": {"pass": False, "value": 0.5}},
            "all_pass": False, "artifacts": {"written": []}}
    monkeypatch.setattr(study_mod, "run_study", lambda **kw: fake)
    assert main(["--no-write", "--quiet"]) == 1
    fake["all_pass"] = True
    fake["checks"]["criterion_b"]["pass"] = True
    assert main(["--no-write", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "criterion_a" in out or "criteria pass" in out


# ------------------------------------------------------------ L3 integration
def test_channel_solver_consumes_the_random_strength_ensemble():
    ice = _ice()
    cfg = SlabConfig(stochastic_strength=True, strength_field_realisations=20,
                     frames=5, seed=3)
    sim = SlabChannelSim(cfg, ice)
    assert sim.strength_ensemble is not None
    ens = sim.strength_ensemble
    # the detach capacity is the ensemble quantile, not the deterministic value
    assert sim.bend_per_width == pytest.approx(ens["per_width_quantile"])
    assert sim.bend_per_width < ice.bending_per_width_n_per_m
    assert ens["quantile"] == pytest.approx(cfg.strength_field_quantile)
    assert 0.0 < ens["per_width_cov"] < 1.0
    summary = sim.run().as_dict()
    series = np.asarray(summary["resistance_series_kN"], dtype=float)
    assert series.size > 0 and np.all(np.isfinite(series))


def test_deterministic_default_is_unchanged():
    ice = _ice()
    sim = SlabChannelSim(SlabConfig(frames=5, seed=3), ice)
    assert sim.strength_ensemble is None
    assert sim.bend_per_width == pytest.approx(ice.bending_per_width_n_per_m)


def test_integration_wrapper_matches_the_module_ensemble():
    kw = dict(width_m=50.0, correlation_length_m=10.0, weibull_modulus=6.0,
              n_real=20, db_m=1.0, quantile=0.05, seed=7)
    direct = stochastic_bending_capacity_per_width_n_per_m(0.54e6, 1.0, 5.0e9, **kw)
    again = stochastic_bending_capacity_per_width_n_per_m(0.54e6, 1.0, 5.0e9, **kw)
    assert direct == again                                   # deterministic seeding
    assert direct["per_width_mean"] > direct["per_width_quantile"]
    assert direct["width_m"] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        stochastic_bending_capacity_per_width_n_per_m(0.54e6, 1.0, 5.0e9,
                                                      width_m=50.0, quantile=1.5)


# ------------------------------------------------------------ L4 benchmark
def test_benchmark_artifact_is_internally_consistent():
    if not BENCH.exists():
        pytest.skip("benchmark artifact not built in this checkout")
    payload = json.loads(BENCH.read_text())
    rows = payload["rows"]
    assert payload["n_checked"] == sum(1 for r in rows if not r["advisory"])
    assert payload["n_pass"] == sum(1 for r in rows if r["pass"])
    assert payload["all_pass"] is True
    for row in rows:
        assert row["reference_source"], row["key"]
        if row["advisory"]:
            continue
        assert row["pass"] is True, row
        assert row["rel_error"] <= row["tolerance"] + 1e-12, row
        assert row["tolerance"] <= 0.35, row          # no silently loose rows
    exponent = next(r for r in rows if r["key"] == "c_size_effect_exponent")
    assert len(exponent["seed_exponents"]) == 3
    assert exponent["seed_spread"] < 0.2              # spread reported, not hidden


def test_predeclared_thresholds_still_match_the_brief():
    assert CRITERIA["qq_r2_min"] == pytest.approx(0.95)          # S1
    assert CRITERIA["correlation_l2_max"] == pytest.approx(0.10)   # S2
    assert CRITERIA["mu_convergence_rel_max"] == pytest.approx(0.05)  # S3
    assert CRITERIA["cov_min"] == pytest.approx(0.10) and CRITERIA["cov_max"] == 0.30
    assert CRITERIA["n_ensemble"] == 50
    assert CRITERIA["widths_m"][0] == pytest.approx(10.0)
    assert CRITERIA["widths_m"][-1] == pytest.approx(100.0)
    assert CRITERIA["deterministic_rel_max"] == pytest.approx(0.20)  # S7


def test_acceptance_artifact_reports_a_verdict():
    if not ACCEPT.exists() or not STUDY.exists():
        pytest.skip("acceptance artifact not built in this checkout")
    acc = json.loads(ACCEPT.read_text())
    study = json.loads(STUDY.read_text())
    assert acc["n_criteria"] == len(study["checks"])
    assert acc["all_criteria_pass"] is study["all_pass"]
    for key, chk in study["checks"].items():
        assert acc["criteria"][key]["pass"] is bool(chk["pass"])
        value = acc["criteria"][key]["value"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert math.isfinite(float(value))
    for path in acc["level_evidence"]["L2_implementation"]["impl_files"]:
        assert (ROOT / path).exists(), path
