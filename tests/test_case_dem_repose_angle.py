"""Tests for CASE-DEM-REPOSE-ANGLE (glass-bead repose benchmark, ref 22 deg).

Fast unit-level guards only (no full 1.84 s lifting-cylinder run: that leg is
submitted to the compute queue and its artefacts live in
``artifacts/CASE-DEM-REPOSE-ANGLE``).  What is guarded here:

* the reference numbers are *derived* consistently from the cited source
  (bulk density / packing fraction -> particle density; 22 deg -> mu = 0.40),
* the pre-registered leg list is the one the report claims (counts, held-out
  seeds never appear in the calibration scan),
* the analysis layer behaves: bracket interpolation, extrapolation refusal,
  monotonicity test, and the frozen criterion arithmetic (including the ±3 deg
  tolerance and the WITHHELD path when a leg has no valid run),
* one worker call never raises into the pool: an estimator refusal is returned
  as a record with ``error`` (this is what a jammed/stub heap produces).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "case_dem_repose_angle"))

import run_case as case  # noqa: E402


# ---------------------------------------------------------------------------
# Reference bookkeeping
# ---------------------------------------------------------------------------

def test_reference_derivation_is_self_consistent():
    d = case.ref_derived()
    assert d["rho_particle_kg_m3"] == pytest.approx(1480.0 / 0.59, rel=1e-12)
    assert 2400.0 < d["rho_particle_kg_m3"] < 2600.0      # glass, physical
    assert d["mu_from_angle"] == pytest.approx(math.tan(math.radians(22.0)))
    # the paper's own two numbers (22 deg and mu = 0.40) must agree to ~2 %
    assert d["mu_quoted_vs_tan_rel"] < 0.02


def test_reference_record_carries_its_location():
    assert "1307.4638" in case.REF["paper"]
    assert "p. 1" in case.REF["location"]
    assert case.REF["theta_exp_deg"] == 22.0
    assert case.REF["packing_fraction"] == 0.59


# ---------------------------------------------------------------------------
# Pre-registered design
# ---------------------------------------------------------------------------

def test_leg_list_matches_prereg():
    p1 = case.build_phase1()
    assert sum(1 for r in p1 if r["leg"] == "P") == 3
    assert sum(1 for r in p1 if r["leg"] == "N") == 1
    scan = [r for r in p1 if r["leg"] == "S"]
    assert len(scan) == len(case.MU_R_NODES) * len(case.SEEDS_SCAN) == 18
    p2 = case.build_phase2(0.404)
    assert sum(1 for r in p2 if r["leg"] == "C") == 3
    for leg in ("M", "E", "W", "L", "R"):
        assert any(r["leg"] == leg for r in p2), leg


def test_holdout_seeds_are_never_used_for_calibration():
    p1 = case.build_phase1()
    scan_seeds = {r["seed"] for r in p1 if r["leg"] == "S"}
    pred_seeds = {r["seed"] for r in p1 if r["leg"] == "P"}
    holdout = set(case.SEEDS_HOLDOUT)
    assert scan_seeds.isdisjoint(holdout)
    assert pred_seeds == holdout


def test_prediction_leg_uses_reference_friction_and_small_rolling():
    p1 = [r for r in case.build_phase1() if r["leg"] == "P"]
    mu = case.ref_derived()["mu_from_angle"]
    assert all(r["mu"] == pytest.approx(mu) for r in p1)
    assert all(r["mu_rolling"] == case.MU_R_GLASS_PRIOR for r in p1)
    assert case.MU_R_GLASS_PRIOR < math.tan(math.radians(case.TARGET_DEFAULT))


# ---------------------------------------------------------------------------
# Analysis layer
# ---------------------------------------------------------------------------

def _agg(leg, nodes, theta, sigma=1.0, n=2):
    return {f"{leg}|{label}": {"leg": leg, "label": label, "n_seeds": n,
                               "n_valid": n, "n_error": 0,
                               "all": {"n": n, "mean": t, "sigma": sigma,
                                       "min": t, "max": t,
                                       "dtheta": t - case.TARGET_DEFAULT,
                                       "abs_dtheta": abs(t - case.TARGET_DEFAULT)},
                               "unflagged": None, "n_flagged": 0, "seeds": [],
                               "flagged_seeds": []}
            for label, t in zip(nodes, theta)}


def test_inversion_brackets_and_interpolates():
    agg = _agg("S", ["mu_r=0.2", "mu_r=0.3", "mu_r=0.4"], [11.0, 17.4, 23.0])
    inv = case.invert_scan(agg)
    assert inv["status"] == "bracketed"
    # target 22 lies between 17.4 (mu_r 0.3) and 23.0 (mu_r 0.4)
    frac = (22.0 - 17.4) / (23.0 - 17.4)
    assert inv["mu_r_star"] == pytest.approx(0.3 + frac * 0.1)
    assert 0.30 <= inv["mu_r_star"] <= 0.40


def test_inversion_refuses_extrapolation():
    agg = _agg("S", ["mu_r=0.1", "mu_r=0.2"], [3.0, 8.0])
    inv = case.invert_scan(agg)
    assert inv["status"] == "out_of_bracket"
    assert "mu_r_star" not in inv


def test_monotonicity_detection():
    good = _agg("S", ["mu_r=0.1", "mu_r=0.2", "mu_r=0.3"], [5.7, 11.3, 16.7])
    assert case.monotone_ok(good)["monotone_within_2sigma"] is True
    bad = _agg("S", ["mu_r=0.1", "mu_r=0.2", "mu_r=0.3"], [5.7, 25.0, 16.7],
               sigma=0.1)
    assert case.monotone_ok(bad)["monotone_within_2sigma"] is False


def test_criteria_arithmetic_within_and_outside_tolerance():
    d = case.ref_derived()
    agg = {}
    agg.update(_agg("S", ["mu_r=0.3", "mu_r=0.4"], [17.5, 23.5]))
    agg.update(_agg("P", ["prediction mu_r=0.05"], [3.0]))
    agg.update(_agg("N", ["frictionless control"], [2.0]))
    agg.update(_agg("C", ["holdout at mu_r*"], [22.9], sigma=0.5, n=3))
    crit = case.evaluate(agg, {"mu_r_star": 0.38}, case.best_stat(agg["C|holdout at mu_r*"]),
                         case.best_stat(agg["N|frictionless control"]), {})
    assert crit["AC-1"]["verdict"] == "FAIL"      # 3.0 deg, 19 deg below ref
    assert crit["AC-2"]["verdict"] == "PASS"      # 22.9 deg -> 0.9 deg
    assert crit["AC-3"]["verdict"] == "PASS"      # flat control
    assert crit["CASE"]["verdict"] == "PASS"
    assert crit["CASE"]["route"] == "calibrated"

    agg2 = dict(agg)
    agg2["C|holdout at mu_r*"] = _agg("C", ["holdout at mu_r*"], [26.5], n=3)[
        "C|holdout at mu_r*"]
    crit2 = case.evaluate(agg2, {}, case.best_stat(agg2["C|holdout at mu_r*"]),
                          case.best_stat(agg2["N|frictionless control"]), {})
    assert crit2["AC-2"]["verdict"] == "FAIL"     # 4.5 deg > 3 deg
    assert crit2["CASE"]["verdict"] == "FAIL"


def test_criteria_withheld_when_no_valid_run():
    agg = _agg("N", ["frictionless control"], [20.0])
    agg["N|frictionless control"]["all"] = None
    agg["N|frictionless control"]["unflagged"] = None
    crit = case.evaluate(agg, {}, None, case.best_stat(agg["N|frictionless control"]), {})
    assert crit["AC-2"]["verdict"] == "WITHHELD"
    assert crit["AC-3"]["verdict"] == "WITHHELD"


# ---------------------------------------------------------------------------
# Worker robustness
# ---------------------------------------------------------------------------

def test_flags_tolerate_estimator_without_n_fit_bins():
    """simulate() forwards n_bins but not n_fit_bins -> flags must not KeyError."""
    run = {"max_speed_m_s": 0.01, "r2": 0.95, "n_bins": 12,
           "discharge_ratio": 2.0, "theta_r_deg": 22.0}
    assert case.is_valid(run)
    fl = case.flags(run)
    assert fl["short_flank"] is False and case.any_flag(run) is False
    run2 = dict(run, n_bins=3)
    assert case.flags(run2)["short_flank"] is True


def test_worker_returns_error_record_instead_of_raising():
    """A protocol that cannot form a flank must come back as a record."""
    from tensordem.particle import clump  # noqa: F401  (ensure package import works)
    task = case._base(leg="S", label="stub", mu=math.tan(math.radians(22.0)),
                      mu_rolling=0.3, seed=19975, max_steps=5)
    rec = case.run_one(task)
    assert rec["leg"] == "S" and rec["seed"] == 19975
    if rec.get("error") is None:            # a flank happened to be found: fine
        assert rec["theta_r_deg"] is not None
    else:
        assert "error" in rec and "theta_r_deg" not in rec
        assert case.is_valid(rec) is False
        assert case.flags(rec)["run_error"] is True