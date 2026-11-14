"""Regression tests for the round-2 deposition rigid-limit verification.

These lock the *evidence* invariants, not the physics verdict: if someone later
restores a "p = 1.007 measured at 4 levels" claim, or a "-0.20% accuracy" claim,
or a green seed-coverage gate over a single-seed controlling tier, the suite
goes red.

Run::

    python -m pytest tests/test_xval_dem_round2_deposition_rigid_limit_audit.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

_AUDIT_PATH = os.path.join(
    ROOT, "benchmarks", "xval_dem_round2_deposition_rigid_limit_audit.py")


def _load_audit():
    spec = importlib.util.spec_from_file_location("xval_rigid_limit_audit",
                                                  _AUDIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


@pytest.fixture(scope="module")
def runs(audit):
    return audit.load_runs()


# ---------------------------------------------------------------------------
# the extrapolation primitive
# ---------------------------------------------------------------------------
def test_exact_power_law_is_recovered(audit):
    """A clean p = 1 family must be extrapolated back to its own limit."""
    limit, amp = 0.6366, -0.9
    pts = [(kn, limit + amp / kn) for kn in (4000.0, 16000.0, 64000.0, 256000.0)]
    r = audit.richardson(pts)
    assert r["order"] == pytest.approx(1.0, abs=1e-9)
    assert r["fallback"] is False
    assert r["limit"] == pytest.approx(limit, abs=1e-12)


def test_non_monotone_ladder_falls_back_and_says_so(audit):
    """Out-of-bounds order must be reported as a fallback, never as measured."""
    pts = [(4000.0, 0.6424), (16000.0, 0.6379),
           (64000.0, 0.6368), (256000.0, 0.6357)]
    r = audit.richardson(pts)
    assert r["fallback"] is True
    assert r["order"] is None
    assert r["p_used"] == pytest.approx(1.0)


def test_audit_reimplementation_matches_shipped_library(audit, runs):
    """The audit must not be an opinion: same inputs, same number."""
    points = [(p[0], p[1]) for p in audit.ladder_sensitivity(
        runs, 0.0)["variants"][-1]["points"]]
    x = audit.crosscheck_library(points)
    assert x["agree"] is True
    assert x["abs_diff"] < 1e-12


# ---------------------------------------------------------------------------
# F2 -- the quoted order does not belong to the quoted limit
# ---------------------------------------------------------------------------
def test_four_level_order_is_not_measured(audit, runs):
    """The 4-level mu=0 fit must report p as assumed, not as measured."""
    budget = audit.error_budget(runs)
    assert budget["fit4_order"] is None, (
        "if the 4-level order became measurable, re-run the audit doc: the "
        "'p = 1.007 at 4 levels' critique (F2) would need revisiting")
    assert budget["fit4_p_used"] == pytest.approx(1.0)


def test_three_level_order_is_measured_and_near_one(audit, runs):
    """The number the report quotes (1.007) lives on the 3-level sub-ladder."""
    variants = audit.ladder_sensitivity(runs, 0.0)["variants"]
    three = [v for v in variants if v["ladder"] == "3-level"]
    assert three, "3-level ladders must be part of the sensitivity set"
    for v in three:
        assert v["fallback"] is False
        assert 0.7 < v["order"] < 1.1


# ---------------------------------------------------------------------------
# F3 -- the advertised precision is below the noise floor
# ---------------------------------------------------------------------------
def test_ladder_family_spread_exceeds_claimed_deviation(audit, runs):
    ls = audit.ladder_sensitivity(runs, 0.0)
    budget = audit.error_budget(runs)
    assert ls["limit_spread"] > budget["claimed_deviation"], (
        "the extrapolation family is tighter than the claimed deviation; the "
        "-0.20% statement would then be resolvable and F3 must be withdrawn")
    assert ls["limit_spread"] / audit.PHI_RCP > 1e-3


def test_claim_is_below_the_dominant_systematic(audit, runs):
    budget = audit.error_budget(runs)
    assert budget["claim_below_systematic"] is True


# ---------------------------------------------------------------------------
# F4 -- correction smaller than its own held-out error
# ---------------------------------------------------------------------------
def test_correction_is_below_own_prediction_error(audit, runs):
    budget = audit.error_budget(runs)
    assert budget["correction_below_prediction_error"] is True
    assert budget["correction"] < budget["limit_shift"]


# ---------------------------------------------------------------------------
# F5 -- coverage of the controlling tier
# ---------------------------------------------------------------------------
def test_reported_seed_count_at_kn64000_is_wrong(audit, runs):
    """The round-2 report says 1 seed at kn=64000; there are 3."""
    cov = audit.coverage(runs)
    assert cov["actual_kn64000_seeds"] == 3
    assert cov["report_claims_kn64000_seeds"] == 1
    assert cov["actual_kn64000_seeds"] != cov["report_claims_kn64000_seeds"]


def test_base_tier_coverage_is_not_tier_coverage(audit, runs):
    """The gate must not be able to pass on the base tier alone."""
    cov = audit.coverage(runs)
    worst = min(t["n_seeds"] for t in cov["tiers"])
    assert worst < 3, (
        "all tiers now have >= 3 seeds: the G16 gate blindness finding (F5) is "
        "fixed by data rather than by code, update the audit")
    assert cov["base_tier_seeds"] >= 3


# ---------------------------------------------------------------------------
# F6 -- temporal convergence is not tested where it matters
# ---------------------------------------------------------------------------
def test_temporal_check_reports_untested_levels(audit, runs):
    temp = audit.temporal_check(runs)
    assert temp["rows"], "the dt probe at kn=4000 must be found"
    assert 256000.0 in [float(x) for x in temp["untested_levels"]] or \
        temp["stiffest_tested"]
    for r in temp["rows"]:
        if r["seed_scatter"]:
            # a check whose difference is below the scatter is not a check
            assert r["diff"] < 1e-2


# ---------------------------------------------------------------------------
# F8 -- the mu = 0.1 tier must not be judged against a frictionless band
# ---------------------------------------------------------------------------
def test_friction_lowers_packing_fraction_monotonically(audit, runs):
    fric = audit.friction_trend(runs)
    assert fric["monotone_decreasing"] is True
    assert fric["friction_shift_at_kn4000_pct"] < 0.0
    assert fric["mu01_limit"] < 0.6366 + 1e-9


# ---------------------------------------------------------------------------
# anchor acceptance
# ---------------------------------------------------------------------------
def test_anchor_acceptance_holds_for_every_aggregation(audit, runs):
    """< 15% must pass on all six ladders, not only the quoted one."""
    ls = audit.ladder_sensitivity(runs, 0.0)
    assert all(abs(v["vs_rcp_pct"]) < 15.0 for v in ls["variants"])