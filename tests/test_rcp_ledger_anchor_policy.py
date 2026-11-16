"""Ledger-path anchor policy tests (m23840, see-through of m21145 / PR#1699).

The m21145 anchor policy -- an accuracy claim (<3 % or any tolerance) is
admissible **only** against a ``literature`` anchor; ``derived`` /
``unverified`` anchors may rank measurements or feed the necessary-condition
bracket, never certify them -- was enforced on the audit path
(``anchor_provenance_audit.py``, gate A5) but the *merged ledger path*
(``aggregate.py`` -> ``runs/summary.json``) still certified the mu=0.1 tier
against the self-declared interpolated 0.6250 entry (G13/G18b "PASS").

These tests pin the policy on the ledger path:

* G12/G18 (mu=0, literature anchor) stay certified PASS/FAIL;
* G13/G18b (mu=0.1, derived anchor) carry ``claim_status = WITHHELD`` and a
  bracket verdict, never a certified accuracy claim;
* no gate in the committed ``summary.json`` may pass as a certified accuracy
  claim on a non-literature anchor (and a reference gate without a provenance
  marker is itself a violation);
* the audit's ``ledger_claims()`` cross-check detects a stale (old-format)
  ledger, so the A5 gate cannot be silently bypassed by reverting
  ``aggregate.py``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "benchmarks", "case_rcp_deposition")
SUMMARY = os.path.join(BENCH, "runs", "summary.json")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(BENCH, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gates():
    with open(SUMMARY) as f:
        return json.load(f)["gates"]


def _gate(gates, gid):
    hit = [g for g in gates if g["id"] == gid]
    assert len(hit) == 1, f"gate {gid} not found exactly once"
    return hit[0]


# ---------------------------------------------------------------------------
# unit level: the policy itself (no artefacts)
# ---------------------------------------------------------------------------

class TestAnchorPolicyUnit:
    def test_derived_anchor_never_certifies(self):
        from tensordem.rcp_references import RCPReference, judge
        ref = RCPReference(mu=0.1, phi_rcp=0.625, phi_uncertainty=0.008,
                          source="interpolated", provenance="derived")
        st = ref.claim_status(0.62628)
        assert st["verdict"] == "WITHHELD"
        assert st["within_tolerance"]  # the number is close -- the claim is
        # still withheld: closeness to a constructed number is not validation
        j = judge(0.62628, 0.1)
        assert j["verdict"] == "WITHHELD"
        assert j["strongest_statement"] in ("PASS_BRACKET", "FAIL_BRACKET")

    def test_literature_anchor_certifies(self):
        from tensordem.rcp_references import RCPReference
        ref = RCPReference(mu=0.0, phi_rcp=0.6366, phi_uncertainty=0.005,
                           source="Scott & Kilgour (1969)",
                           provenance="literature")
        assert ref.claim_status(0.6424)["verdict"] == "PASS"
        assert ref.claim_status(0.6000)["verdict"] == "FAIL"

    def test_unverified_anchor_never_certifies(self):
        from tensordem.rcp_references import RCPReference
        ref = RCPReference(mu=0.3, phi_rcp=0.60, phi_uncertainty=0.010,
                           source="composite", provenance="unverified")
        assert ref.claim_status(0.6000)["verdict"] == "WITHHELD"


# ---------------------------------------------------------------------------
# artefact level: the committed ledger
# ---------------------------------------------------------------------------

class TestLedgerArtefact:
    def test_g13_is_withheld_bracket(self):
        g = _gate(_gates(), "G13_stiffness_limit_vs_ref_frictional")
        assert g["anchor_provenance"] == "derived"
        assert g["claim_status"] == "WITHHELD"
        assert g["verdict"] == "PASS_BRACKET"
        assert g["strongest_statement"] == "PASS_BRACKET"
        assert g["pass"] is True  # inside the bracket: necessary condition met
        assert g["phi_ref"] == pytest.approx(0.625)
        lo, hi = g["bracket"]
        assert lo <= g["got"] <= hi

    def test_g18b_is_withheld_bracket(self):
        g = _gate(_gates(), "G18b_production_vs_friction_ref")
        assert g["anchor_provenance"] == "derived"
        assert g["claim_status"] == "WITHHELD"
        assert g["verdict"] == "PASS_BRACKET"
        assert g["pass"] is True
        assert g["phi_ref"] == pytest.approx(0.625)

    def test_g12_g18_certified_against_literature(self):
        for gid in ("G12_stiffness_limit_vs_ref",
                    "G18_production_vs_friction_ref"):
            g = _gate(_gates(), gid)
            assert g["anchor_provenance"] == "literature"
            assert g["claim_status"] == "CERTIFIED"
            assert g["verdict"] == "PASS"
            assert g["pass"] is True

    def test_no_certified_claim_on_non_literature_anchor(self):
        """A5 invariant on the ledger: a certified claim needs a literature
        anchor; a reference gate without a provenance marker is a violation."""
        bad = []
        for g in _gates():
            if "phi_ref_source" not in g:
                continue
            prov = g.get("anchor_provenance")
            if prov is None:
                bad.append(g["id"])
            elif prov != "literature" and g.get("claim_status") != "WITHHELD":
                bad.append(g["id"])
        assert bad == [], f"certified claims on non-literature anchors: {bad}"

    def test_withheld_gate_keeps_ranking_number(self):
        """The WITHHELD downgrade must not discard the measurement: the
        deviation against the table entry stays machine-readable for ranking."""
        g = _gate(_gates(), "G13_stiffness_limit_vs_ref_frictional")
        assert g["value"] == pytest.approx(abs(g["got"] / g["phi_ref"] - 1.0),
                                           rel=1e-12)


# ---------------------------------------------------------------------------
# cross-path enforcement: the audit's ledger scan
# ---------------------------------------------------------------------------

class TestAuditLedgerScan:
    def test_committed_ledger_is_clean(self):
        mod = _load("anchor_provenance_audit")
        assert mod.ledger_claims() == []

    def test_stale_ledger_is_detected(self, tmp_path):
        """A reference gate without a provenance marker (old aggregate.py
        format) must be flagged, so reverting aggregate.py fails A5."""
        mod = _load("anchor_provenance_audit")
        stale = {"gates": [
            {"id": "G13_stiffness_limit_vs_ref_frictional",
             "pass": True, "phi_ref": 0.625,
             "phi_ref_source": "Pournin et al. (2001) PRE; interpolated"},
            {"id": "G18b_production_vs_friction_ref",
             "pass": True, "phi_ref": 0.625,
             "phi_ref_source": "Pournin et al. (2001) PRE; interpolated",
             "anchor_provenance": "derived", "claim_status": "CERTIFIED"},
            {"id": "G12_stiffness_limit_vs_ref", "pass": True,
             "phi_ref": 0.6366,
             "phi_ref_source": "Scott & Kilgour (1969)",
             "anchor_provenance": "literature", "claim_status": "CERTIFIED"},
            {"id": "G8_settled", "pass": True},  # not a reference gate
        ]}
        with open(os.path.join(tmp_path, "summary.json"), "w") as f:
            json.dump(stale, f)
        mod.RUNS = str(tmp_path)
        assert mod.ledger_claims() == [
            "G13_stiffness_limit_vs_ref_frictional",
            "G18b_production_vs_friction_ref",
        ]

    def test_missing_ledger_is_not_a_violation(self, tmp_path):
        mod = _load("anchor_provenance_audit")
        mod.RUNS = str(tmp_path)
        assert mod.ledger_claims() == []
