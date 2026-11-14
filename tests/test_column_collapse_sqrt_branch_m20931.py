"""Tests for the a=3 square-root-branch refinement rung (m20931).

The anchor is CASE-COLUMN-COLLAPSE: tolerance < 15 % against the Lube et al.
(2004) closed-form law.  These tests pin (a) the frozen case registry, (b) the
reduction/verdict logic (so the reported verdict is reproducible from the
stored provenance and cannot be hand-edited), and (c) the committed artifact,
when present, against a fresh recomputation.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmarks",
                     "case_column_collapse_sqrt_branch_m20931.py")


def _load():
    spec = importlib.util.spec_from_file_location("_cc_sqrt_m20931", BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MOD = _load()


# ---------------------------------------------------------------------------
# (a) frozen pre-registered case / ladder
# ---------------------------------------------------------------------------
def test_registered_case_matches_prereg():
    assert MOD.CASE["case_id"] == "SB_a3_g16"
    assert MOD.CASE["aspect_ratio"] == 3.0          # square-root branch
    assert MOD.CASE["grains_per_radius"] == 16.0    # third rung (8, 12, 16)
    assert MOD.TOLERANCE == 0.15
    assert [cid for _, cid in MOD.LADDER] == ["R_a3_g8", "R_a3_g12", "SB_a3_g16"]


def test_case_keeps_column_radius_and_sag_ratio_fixed():
    """gpr=16 must refine the discretisation, not change r_i or the sag ratio."""
    case = MOD.BASE_MOD.make_case(MOD.CASE)
    assert case.column_radius == pytest.approx(0.48, rel=1e-12)
    assert case.radius == pytest.approx(0.015, rel=1e-12)
    assert case.kn == pytest.approx(1.5e4, rel=1e-12)
    sag = math.pi * case.density * case.gravity * case.radius ** 2 / (6.0 * case.kn)
    coarse = MOD.BASE_MOD.make_case({"case_id": "R_a3_g8", "axis": "R",
                                     "aspect_ratio": 3.0,
                                     "grains_per_radius": 8.0})
    sag_ref = (math.pi * coarse.density * coarse.gravity * coarse.radius ** 2
               / (6.0 * coarse.kn))
    assert sag == pytest.approx(sag_ref, rel=1e-12)


def test_reference_law_is_the_paper_formula():
    """a = 3 lies on the square-root branch; reference never transcribed."""
    ref = MOD.BASE_MOD.lube_2004_runout_ratio(3.0)
    assert ref == pytest.approx(1.0 + 1.6 * math.sqrt(3.0), rel=1e-12)


# ---------------------------------------------------------------------------
# (b) reduction + verdict logic (pure functions)
# ---------------------------------------------------------------------------
def _row(gpr, runout, ref, err=None):
    return {"case_id": f"c{gpr:g}", "grains_per_radius": float(gpr),
            "runout_ratio": runout, "ref_runout_ratio": ref,
            "err_runout": (runout - ref) / ref if err is None else err,
            "energy_residual": 0.001, "stop_reason": "at_rest"}


def test_judge_h1_converging_and_in_band():
    rows = [_row(8, 4.30, 3.81), _row(12, 4.10, 3.81), _row(16, 3.95, 3.81)]
    out = MOD.judge(rows)
    assert out["hypothesis"] == "H1" and out["verdict"] == "pass"
    assert out["contracting"] is True and out["in_band_at_finest"] is True


def test_judge_h2_non_contracting_and_out_of_band():
    rows = [_row(8, 4.10, 3.81), _row(12, 4.35, 3.81), _row(16, 4.70, 3.81)]
    out = MOD.judge(rows)
    assert out["hypothesis"] == "H2" and out["verdict"] == "fail"
    assert out["contracting"] is False
    assert out["err_gpr16"] > 0.15


def test_judge_h3_contracting_but_out_of_band():
    rows = [_row(8, 4.10, 3.81), _row(12, 4.60, 3.81), _row(16, 4.40, 3.81)]
    out = MOD.judge(rows)
    assert out["hypothesis"] == "H3" and out["verdict"] == "incomplete"


def test_judge_refuses_short_ladder():
    out = MOD.judge([_row(8, 4.10, 3.81), _row(12, 4.25, 3.81)])
    assert out["hypothesis"] == "incomplete"
    assert "incomplete" in out["reason"]


def test_difference_ratio_detects_contraction():
    conv = [_row(8, 4.0, 3.81), _row(12, 4.4, 3.81), _row(16, 4.45, 3.81)]
    assert MOD.difference_ratio(conv) < 1.0
    div = [_row(8, 4.0, 3.81), _row(12, 4.2, 3.81), _row(16, 4.6, 3.81)]
    assert MOD.difference_ratio(div) > 1.0


def test_richardson_bracket_bounds_a_converging_ladder():
    """For a monotone ladder with p=2 behaviour the extrapolation is closer."""
    ref = 3.81
    rows = [_row(8, 4.30, ref), _row(12, 4.10, ref), _row(16, 3.99, ref)]
    rich = MOD.richardson_bracket(rows)
    assert rich["available"] is True
    assert abs(rich["err_inf_p2"]) < abs(rows[-1]["err_runout"])


# ---------------------------------------------------------------------------
# (c) committed artifact, when present, is reproducible from provenance
# ---------------------------------------------------------------------------
VERDICT = os.path.join(MOD.OUT_DIR, "verdict.json")


@pytest.mark.skipif(not os.path.exists(VERDICT),
                    reason="compute artifact not fetched into the worktree")
def test_artifact_recomputes_to_the_same_verdict():
    with open(VERDICT) as fh:
        payload = json.load(fh)
    rows = payload["rows"]
    if len(rows) < 3:
        assert payload["verdict"]["hypothesis"] == "incomplete"
        return
    for r in rows:
        assert r["err_runout"] == pytest.approx(
            (r["runout_ratio"] - r["ref_runout_ratio"]) / r["ref_runout_ratio"],
            rel=1e-12)
    fresh = MOD.judge(rows)
    for key in ("hypothesis", "verdict", "contracting", "in_band_at_finest"):
        pass
    for key in ("hypothesis", "verdict", "contracting", "in_band_at_finest"):
        assert fresh[key] == payload["verdict"][key]
    assert fresh["err_gpr16"] == pytest.approx(payload["verdict"]["err_gpr16"],
                                               rel=1e-12)


@pytest.mark.skipif(not os.path.exists(VERDICT),
                    reason="compute artifact not fetched into the worktree")
def test_artifact_case_is_finite_and_physical():
    with open(VERDICT) as fh:
        rows = json.load(fh)["rows"]
    if len(rows) < 3:
        pytest.skip("third rung not yet integrated/fetched")
    finest = rows[-1]
    assert np.isfinite(finest["runout_ratio"])
    assert finest["runout_ratio"] >= 1.0
    assert finest["stop_reason"] in ("at_rest", "not_converged")
    assert np.isfinite(finest["energy_residual"])
    assert finest["aspect_measured"] == pytest.approx(3.0, abs=0.15)
    assert finest["grains_per_radius"] == 16.0