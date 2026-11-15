"""Anchor-policy tests for the friction-dependent RCP reference table.

Mission m21145 (CASE-DEM-RCP-PACKING-FRACTION).  The case had two analysis
paths judging the same low-friction intercept against two different anchors
(-1.22 % vs the mu-indexed table entry, +3.07 % vs the frictionless literature
value).  These tests pin the policy that resolves it: an accuracy claim is
admissible **only** against a ``literature`` anchor; ``derived`` /
``unverified`` anchors may rank measurements or feed a necessary-condition
bracket, never certify them.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from tensordem.rcp_references import (
    PROVENANCE_TIERS,
    RCPReference,
    anchored_frictions,
    bracket_check,
    bracket_limits,
    frictionless_reference,
    get_reference,
    judge,
    provenance_table,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "runs")
AUDIT_JSON = os.path.join(RUNS, "anchor_provenance_audit.json")


# ---------------------------------------------------------------------------
# provenance coverage
# ---------------------------------------------------------------------------

def test_every_reference_declares_a_provenance_tier():
    for row in provenance_table():
        assert row["provenance"] in PROVENANCE_TIERS, row


def test_unknown_provenance_tier_is_rejected():
    with pytest.raises(ValueError):
        RCPReference(mu=0.2, phi_rcp=0.61, phi_uncertainty=0.01,
                     source="x", provenance="folklore")


def test_only_frictionless_entries_are_literature_anchored():
    assert anchored_frictions() == [0.0]
    assert frictionless_reference().anchored


def test_low_friction_entry_is_derived_not_literature():
    ref = get_reference(0.1)
    assert ref.provenance == "derived"
    assert not ref.anchored


def test_interpolated_reference_is_derived():
    ref = get_reference(0.2, method="interpolate")
    assert ref.provenance == "derived"
    assert not ref.anchored


# ---------------------------------------------------------------------------
# claim policy
# ---------------------------------------------------------------------------

def test_frictionless_claim_passes_on_literature_anchor():
    out = judge(0.6366, 0.0)
    assert out["verdict"] == "PASS"
    assert out["anchor_provenance"] == "literature"
    assert out["in_band"] and out["within_tolerance"]


def test_frictionless_claim_fails_outside_tolerance():
    assert judge(0.70, 0.0)["verdict"] == "FAIL"
    assert judge(0.60, 0.0)["verdict"] == "FAIL"


def test_claim_on_derived_anchor_is_withheld_even_when_numerically_close():
    # the mu=0.1 table entry is 0.6250; a measurement sitting exactly on it
    # must NOT be certified by that entry (self-referential anchor)
    out = judge(0.6250, 0.1)
    assert out["verdict"] == "WITHHELD"
    assert out["rel_error_vs_anchor"] if "rel_error_vs_anchor" in out else True
    assert out["abs_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["policy"] == "bracketed"
    assert out["strongest_statement"] == "PASS_BRACKET"


@pytest.mark.parametrize("mu", [0.1, 0.3, 0.5, 1.0])
def test_no_accuracy_claim_on_non_literature_anchor(mu):
    assert judge(0.62, mu)["verdict"] == "WITHHELD"


def test_withheld_verdict_still_carries_the_bracket():
    out = judge(0.6170, 0.1)
    assert out["verdict"] == "WITHHELD"
    assert out["bracket"]["verdict"] in ("PASS_BRACKET", "FAIL_BRACKET")
    assert out["bracket"]["inside"] is True


# ---------------------------------------------------------------------------
# bracket is a necessary condition only
# ---------------------------------------------------------------------------

def test_bracket_rejects_absurd_packing_fractions():
    assert bracket_check(0.50)["verdict"] == "FAIL_BRACKET"      # denser than RLP? no: looser
    assert bracket_check(0.70)["verdict"] == "FAIL_BRACKET"      # above any disordered limit
    assert bracket_check(0.62)["verdict"] == "PASS_BRACKET"


def test_bracket_upper_edge_is_tightened_by_friction():
    lo, hi_frictionless = bracket_limits()
    _, hi_frictional = bracket_check(0.62, mu=0.1)["bracket"]
    assert lo <= hi_frictional <= hi_frictionless


# ---------------------------------------------------------------------------
# artifact consistency (skipped when the audit has not been generated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(AUDIT_JSON),
                    reason="anchor provenance audit artifacts not generated")
def test_audit_gate_no_claim_on_derived_anchor_passes():
    res = json.load(open(AUDIT_JSON))
    gates = {g["id"]: g for g in res["gates"]}
    assert gates["A5_no_claim_on_derived_anchor"]["pass"]
    assert gates["A6_policy_coverage"]["pass"]


@pytest.mark.skipif(not os.path.exists(AUDIT_JSON),
                    reason="anchor provenance audit artifacts not generated")
def test_audit_reported_verdicts_match_recomputation():
    """Every reported verdict is reproducible from the run artifacts."""
    res = json.load(open(AUDIT_JSON))
    for row in res["verdicts"]:
        out = judge(row["phi"], row["tier_mu"])
        assert out["verdict"] == row["verdict"], row
        assert out["anchor_provenance"] == row["anchor_provenance"], row


@pytest.mark.skipif(not os.path.exists(AUDIT_JSON),
                    reason="anchor provenance audit artifacts not generated")
def test_audit_frictionless_production_matches_run_files():
    res = json.load(open(AUDIT_JSON))
    prod = res["tiers"]["0"]["production"]
    phis = []
    for tag in prod["paths"]:
        d = json.load(open(os.path.join(ROOT, tag)))
        spec = d["spec"]
        if (spec["n_particles"] == 10000 and spec["mu"] == 0.0
                and spec["kn"] == 4000.0):
            phis.append(d["metrics"]["phi_bulk"])
    assert phis, "no frictionless production run artifacts found"
    assert np.mean(phis) == pytest.approx(prod["phi"], abs=1e-9)
    assert judge(prod["phi"], 0.0)["verdict"] == "PASS"


@pytest.mark.skipif(not os.path.exists(AUDIT_JSON),
                    reason="anchor provenance audit artifacts not generated")
def test_audit_low_friction_tier_is_not_certified():
    res = json.load(open(AUDIT_JSON))
    rows = [r for r in res["verdicts"] if r["tier_mu"] == 0.1]
    assert rows
    assert all(r["verdict"] == "WITHHELD" for r in rows), rows