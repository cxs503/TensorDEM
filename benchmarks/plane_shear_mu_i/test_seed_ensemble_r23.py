"""Round-23 gate: the seed ensemble must be recomputable, pre-registered, honest.

Every assertion here either (a) recomputes a published number from the raw
per-run JSON, or (b) pins a claim that the round must **not** make.  Nothing
in this file runs the DEM device -- the artefact is the committed run set
(``runs/seed_ensemble/``), so the gate is cheap and can run in CI.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

import seed_ensemble as SE  # noqa: E402

RUNS = SE.RUNS
ANALYSIS = RUNS / "analysis_r23.json"
TABLES = RUNS / "tables_r23.md"


# ---------------------------------------------------------------------------
# artefact presence
# ---------------------------------------------------------------------------
#: arms that carry a verdict (inside the archived law's validity).  The
#: ``I = 0.3`` arm is outside ``I_sat = 0.2``, is never judged, and its seeds are
#: optional: this round did not run them (local compute envelope), so it is not
#: a completeness requirement.
JUDGED_I = tuple(I for I in SE.I_TARGETS if SE.law_mu(I)[1])


def test_ensemble_is_complete_on_the_judged_arms():
    missing = [SE._point_path(I, s, SE.MU_CONTACT).name
               for I in JUDGED_I for s in SE.SEEDS
               if not SE._point_path(I, s, SE.MU_CONTACT).exists()]
    assert missing == [], f"ensemble incomplete on judged arms: {missing}"


def test_ladder_is_complete():
    missing = [m for m in SE.LADDER_MU
               if not SE._point_path(SE.LADDER_I, SE.SEED, m).exists()]
    assert missing == [], f"ladder incomplete: {missing}"


def test_prereg_was_written_before_the_runs():
    """The pre-registration file must exist and match the module's contract."""
    pre = json.loads((RUNS / "prereg.json").read_text())
    assert pre["tolerance"] == SE.PREREG["tolerance"]
    assert set(pre["predictions"]) == {"P1_spread", "P2_dip", "P3_bias",
                                      "P4_friction"}
    # every decision rule must belong to a registered prediction (P1..P3 have
    # an explicit numeric rule; P4's rule is inside its prediction text)
    assert set(pre["decision_rules"]) <= {n.split("_")[0]
                                          for n in pre["predictions"]}
    # the axes are part of the registration: a later axis change is visible
    assert pre["axes"]["seeds"] == list(SE.SEEDS)
    assert pre["axes"]["I_targets"] == list(SE.I_TARGETS)
    # and the reference is the *archived* law, not the (0.36, 0.84, 0.30) quote
    assert "0.22" in pre["reference"]
    assert "0.36" not in pre["reference"]


# ---------------------------------------------------------------------------
# the reference law
# ---------------------------------------------------------------------------
def test_reference_is_the_archived_dacruz_law():
    assert SE.law_mu(0.1)[0] == pytest.approx(0.32, abs=1e-12)
    assert SE.law_mu(0.1)[1] is True
    # outside the law's validity the value is returned but flagged
    val, ok = SE.law_mu(0.3)
    assert ok is False
    assert val == pytest.approx(0.52, abs=1e-12)


def test_the_misattributed_triple_is_not_used():
    a = json.loads(ANALYSIS.read_text())
    assert a["prereg"]["reference"].startswith("archived da Cruz 2005")
    for entry in a["points"].values():
        assert entry["mu_law_archived"] == pytest.approx(0.22 + 1.0 * entry["I_target"],
                                                         abs=1e-12)


# ---------------------------------------------------------------------------
# statistics are recomputable
# ---------------------------------------------------------------------------
def test_mean_sd_ci_against_known_values():
    st = SE.mean_sd([1.0, 2.0, 3.0, 4.0, 5.0])
    assert st["mean"] == pytest.approx(3.0)
    assert st["sd"] == pytest.approx(math.sqrt(2.5))
    assert st["cv"] == pytest.approx(math.sqrt(2.5) / 3.0)
    lo, hi = SE.ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5.0)
    assert hi - 3.0 == pytest.approx(half)
    assert 3.0 - lo == pytest.approx(half)


def test_analysis_matches_the_raw_points():
    a = json.loads(ANALYSIS.read_text())
    for key, entry in a["points"].items():
        if entry["n_seeds"] == 0:
            continue
        I = entry["I_target"]
        raw = [json.loads(SE._point_path(I, s, SE.MU_CONTACT).read_text())
               for s in entry["seeds"]]
        for name in SE.MU_ORDINATES:
            vals = [r[name] for r in raw]
            st = SE.mean_sd(vals)
            assert [float(x) for x in entry["mu"][name]["values"]] == vals
            assert entry["mu"][name]["mean"] == pytest.approx(st["mean"], rel=1e-12)
            assert entry["mu"][name]["sd"] == pytest.approx(st["sd"], rel=1e-12)
            lo, hi = SE.ci95(vals)
            assert entry["mu"][name]["ci95"][0] == pytest.approx(lo, rel=1e-12)
            assert entry["mu"][name]["ci95"][1] == pytest.approx(hi, rel=1e-12)
            law = entry["mu_law_archived"]
            assert entry["mu"][name]["n_within_tol"] == sum(
                1 for v in vals if abs(v / law - 1.0) <= SE.TOL)
        for name in SE.PHI_ORDINATES:
            vals = [r[name] for r in raw]
            assert entry["phi"][name]["mean"] == pytest.approx(
                sum(vals) / len(vals), rel=1e-12)


def test_verdict_is_a_recomputation_not_a_statement():
    a = json.loads(ANALYSIS.read_text())
    v = a["verdict"]
    judged = [k for k, e in a["points"].items() if e["law_in_validity"]]
    assert v["judged_points"] == judged
    inside = [k for k in judged
              if a["points"][k]["mu"]["mu_full_zz_final"]["n_within_tol"]
              == a["points"][k]["n_seeds"]]
    assert v["points_all_seeds_within_tol"] == inside
    assert v["passes_acceptance"] == (len(inside) == len(judged) and judged)


def test_dip_flag_is_recomputed_from_the_deltas():
    a = json.loads(ANALYSIS.read_text())
    d = a["dip"]
    st = SE.mean_sd(d["delta"])
    expect = bool(st["mean"] < 0 and abs(st["mean"]) > 2.0 * st["sem"]
                  and d["n_negative"] >= len(d["delta"]) - 1)
    assert d["systematic"] is expect
    assert d["n_negative"] == sum(1 for x in d["delta"] if x < 0)
    assert len(d["delta"]) == len(d["seeds"])


def test_the_dip_is_judged_only_inside_the_laws_validity():
    a = json.loads(ANALYSIS.read_text())
    d = a["dip"]
    assert SE.law_mu(d["I_low"])[1] is True
    assert SE.law_mu(d["I_high"])[1] is True


# ---------------------------------------------------------------------------
# parity: the device reproduces the archived single-seed points
# ---------------------------------------------------------------------------
def test_archive_seed_reproduces_round14_bit_for_bit():
    a = json.loads(ANALYSIS.read_text())
    par = a["parity_vs_archived"]
    assert par, "no parity rows: the archive seed 5279 is missing"
    assert all(p["rel_diff"] == 0.0 for p in par), \
        [(p["I"], p["key"], p["rel_diff"]) for p in par if p["rel_diff"] != 0.0]


# ---------------------------------------------------------------------------
# the tables and the doc may not contradict the analysis
# ---------------------------------------------------------------------------
def test_tables_carry_the_analysis_numbers():
    a = json.loads(ANALYSIS.read_text())
    text = TABLES.read_text()
    assert "programmatic tables" in text
    for k, e in a["points"].items():
        m = e["mu"]["mu_full_zz_final"]
        assert f"{m['mean']:.4f}" in text, (k, "mean", m["mean"])
        assert f"{100 * m['cv']:+.1f}" in text, (k, "cv", m["cv"])
        assert f"{m['n_within_tol']}/{m['n']}" in text
    verdict = "PASS" if a["verdict"]["passes_acceptance"] else "FAIL"
    assert f"**{verdict}**" in text


def test_round23_doc_exists_and_does_not_overstate():
    doc = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                       "docs", "CASE_DEM_PLANE_SHEAR_MUIDI_R23.md")
    assert os.path.exists(doc), "round-23 doc missing"
    text = open(doc, encoding="utf-8").read()
    a = json.loads(ANALYSIS.read_text())
    # every headline number in the doc must be one the analysis produced
    for k, e in a["points"].items():
        if e["n_seeds"] == 0:
            continue
        assert f"{e['mu']['mu_full_zz_final']['mean']:.4f}" in text, k
    assert "R23" in text or "round 23" in text
    # an honest round reports its own limits
    assert "局限" in text or "Limit" in text or "limit" in text