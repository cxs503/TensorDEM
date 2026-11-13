"""Round-5 (m16991) machinery tests for CASE-COLUMN-COLLAPSE.

These tests cover the **decision machinery** of
``benchmarks/case_column_collapse_round5_m16991.py`` — the bisection that
locates the required contact friction, the censoring branches, and the
pre-registered H5 verdict rule — with a synthetic monotone evaluator, so they
run in well under a second and do not touch the DEM kernel.  The physics runs
themselves are stored as per-case JSON provenance under
``runs/case_column_collapse_r5_m16991/`` and summarised in
``docs/case_column_collapse_round5_m16991.md``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmarks", "case_column_collapse_round5_m16991.py")


def _load():
    spec = importlib.util.spec_from_file_location("w5", BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["w5"] = mod
    spec.loader.exec_module(mod)
    return mod


W5 = _load()


def _monotone(root: float, slope: float = -1.0):
    """Synthetic signed error: ``e(mu) = slope * (mu - root)``."""
    def f(mu: float) -> dict:
        return {"signed_error": slope * (mu - root), "stop_reason": "synthetic",
                "runout_ratio_p995": float("nan"), "runout_ratio_max": float("nan"),
                "reference": float("nan"), "n_particles": 0,
                "wall_clock_s": 0.0, "cached": False, "case_id": "synthetic"}
    return f


def test_signed_error_uses_reference_law():
    # a = 1 is in the linear regime: r_inf/r_i = 1 + 1.24 -> 2.24.
    ref = W5.lube_2004_runout_ratio(1.0)
    assert ref == pytest.approx(2.24, abs=1e-12)
    assert W5.signed_error(ref, 1.0) == pytest.approx(0.0)
    assert W5.signed_error(1.15 * ref, 1.0) == pytest.approx(0.15, abs=1e-12)


def test_bisection_finds_root_and_never_invents_one():
    trace = W5.bisect_mu(1.0, 0.1, lo=0.9, hi=2.4, tol=0.05, max_eval=8,
                         log=lambda *_: None, evaluate_fn=_monotone(1.37))
    assert trace["censoring"] is None
    assert trace["mu_req"] == pytest.approx(1.37, abs=0.06)
    assert trace["bracket"][0] < 1.37 < trace["bracket"][1]
    # A bracketed root requires both signs observed, not just an interpolation.
    errs = [e["signed_error"] for e in trace["evals"]]
    assert min(errs) <= 0.0 <= max(errs)


def test_bisection_reports_low_censoring_when_root_above_bracket():
    trace = W5.bisect_mu(1.0, 0.1, lo=0.9, hi=2.4, tol=0.05,
                         log=lambda *_: None, evaluate_fn=_monotone(3.0))
    assert trace["mu_req"] is None
    assert trace["censoring"] == "low"
    assert trace["lower_bound"] == 2.4


def test_bisection_reports_high_censoring_when_root_below_bracket():
    trace = W5.bisect_mu(1.0, 0.1, lo=0.9, hi=2.4, tol=0.05,
                         log=lambda *_: None, evaluate_fn=_monotone(0.5))
    assert trace["mu_req"] is None
    assert trace["censoring"] == "high"


def test_bisection_respects_evaluation_budget():
    trace = W5.bisect_mu(1.0, 0.1, lo=0.9, hi=2.4, tol=1e-6, max_eval=5,
                         log=lambda *_: None, evaluate_fn=_monotone(1.37))
    assert len(trace["evals"]) <= 5


def _chain(aspect: float, mur: float, mu_req: float | None, censoring=None) -> dict:
    return {"chain": W5.chain_id(aspect, mur), "aspect": aspect,
            "mu_rolling": mur, "mu_req": mu_req, "censoring": censoring,
            "evals": [], "note": ""}


def test_gap_table_pairs_branches_by_mu_rolling():
    chains = {"a1_mur0.2": _chain(1.0, 0.2, 1.10),
              "a3_mur0.2": _chain(3.0, 0.2, 1.40)}
    rows = W5.gap_table(chains)
    assert len(rows) == 1
    row = rows[0]
    assert row["mu_req_linear"] == pytest.approx(1.10)
    assert row["mu_req_sqrt"] == pytest.approx(1.40)
    assert row["gap"] == pytest.approx(0.30)


def test_gap_table_merges_round4_level_without_faking_a_gap():
    chains = {"a1_mur0.2": _chain(1.0, 0.2, 1.10),
              "a3_mur0.2": _chain(3.0, 0.2, None, censoring="low")}
    rows = W5.gap_table(chains, extra_mur={0.5: {"mu_req_linear": 0.514,
                                                 "mu_req_sqrt": 0.804}})
    by_mur = {r["mu_rolling"]: r for r in rows}
    assert by_mur[0.2]["gap"] is None            # sqrt censored -> no gap
    assert by_mur[0.2]["sqrt_lower_bound_only"] is True
    assert by_mur[0.5]["gap"] == pytest.approx(0.29)
    assert by_mur[0.5]["source"] == "round4"


def test_verdict_refuted_needs_three_bracketed_levels_same_sign():
    rows = [{"mu_rolling": m, "gap": g} for m, g in
            ((0.1, 0.45), (0.2, 0.33), (0.3, 0.30), (0.5, 0.29))]
    v = W5.verdict(rows)
    assert v["h5"] == "REFUTED"
    assert v["n_bracketed_mur"] == 4


def test_verdict_supported_on_sign_change():
    rows = [{"mu_rolling": 0.1, "gap": -0.05}, {"mu_rolling": 0.2, "gap": 0.10},
            {"mu_rolling": 0.3, "gap": 0.25}]
    assert W5.verdict(rows)["h5"] == "SUPPORTED"


def test_verdict_incomplete_with_one_bracketed_level():
    rows = [{"mu_rolling": 0.2, "gap": None}, {"mu_rolling": 0.5, "gap": 0.29}]
    assert W5.verdict(rows)["h5"] == "INCOMPLETE"


def test_preregistered_plan_is_frozen():
    """The chains, bracket and tolerance are pre-registered; assert them so a
    later edit that silently re-opens the search space fails loudly."""
    assert W5.BRANCH_ASPECTS == (1.0, 3.0)
    assert W5.W5_MURS == (0.1, 0.2, 0.3)
    assert (W5.MU_LO, W5.MU_HI) == (0.9, 2.4)
    assert W5.TOLERANCE == 0.15
    assert len(W5.CHAINS) == 6
    # The bracket must abut (not overlap) round 4's grid ceiling.
    assert W5.MU_LO == 0.9


def test_protocol_matches_round1_benchmark():
    """Only (mu, mu_r, a) may differ from the round-1/2/3/4 protocol."""
    spec = importlib.util.spec_from_file_location(
        "w1", os.path.join(ROOT, "benchmarks", "case_column_collapse_lube2004.py"))
    w1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w1)
    varied = {"mu", "mu_rolling", "aspect_ratio"}
    for k, v in W5.BASE.items():
        if k in varied:
            continue
        assert w1.BASE[k] == v, f"{k} differs from the round-1 protocol"