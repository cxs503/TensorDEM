"""Tests for the doc numeric-claim audit (`scripts/audit_doc_numeric_claims.py`).

The audit's value rests entirely on its ability to *fail* on a bad claim, so the
core of this module is a set of negative controls: synthetic documents with a
seeded inconsistency that the corresponding check must reject.  Without them a
"0 failures" audit result would be indistinguishable from a check that does
nothing (the documentation analogue of replacing a solver with a stub and still
reporting a pass).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_doc_numeric_claims", ROOT / "scripts" / "audit_doc_numeric_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load()


def run_on(tmp_path, markdown, name="doc.md"):
    """Run the audit over a synthetic tree holding one document."""
    d = tmp_path / "docs"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(markdown, encoding="utf-8")
    return audit.audit(str(tmp_path))


HEAD_5 = "| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? |\n|---|---|---|---|---|\n"
HEAD_3 = "| 关键量 | 相对误差 | < 3 % ? |\n|---|---|---|\n"


class TestVerdictLogic:
    """Check B: a verdict must be explainable by the row's own error values."""

    def test_false_yes_is_rejected(self, tmp_path):
        rows = run_on(tmp_path, HEAD_3 + "| 剪切模量 | 8.0 % | 是 |\n")
        assert len(rows) == 1
        assert rows[0]["check_b"] is False
        assert rows[0]["status"] == "FAIL"

    def test_false_no_is_rejected(self, tmp_path):
        rows = run_on(tmp_path, HEAD_3 + "| 剪切模量 | 0.4 % | 否 |\n")
        assert rows[0]["check_b"] is False
        assert rows[0]["status"] == "FAIL"

    @pytest.mark.parametrize("cell,verdict,ok", [
        ("0.0822 %", True, True),     # percent reading of a sub-percent error
        ("0.00082", True, True),      # the same error as a fraction
        ("6.93 %", False, True),      # percent reading, genuinely out of tolerance
        ("0.0693", False, True),      # the same error as a fraction
        ("0.0822 %", False, False),   # 0.08 % cannot justify a "no"
        ("< 1e-15", True, True),      # inequality form still readable
    ])
    def test_unit_reading_matrix(self, tmp_path, cell, verdict, ok):
        word = "是" if verdict else "否"
        rows = run_on(tmp_path, HEAD_3 + f"| 量 | {cell} | {word} |\n")
        assert rows[0]["check_b"] is ok


class TestRecompute:
    """Check A: the stated error must be reproducible from basis and measured."""

    def test_consistent_row_passes(self, tmp_path):
        md = HEAD_5 + "| 质量半径 | 0.1 | 0.099918 | 0.0822 % | 是 |\n"
        rows = run_on(tmp_path, md)
        assert rows[0]["check_a"] is True
        assert rows[0]["status"] == "PASS"

    def test_wrong_error_is_rejected(self, tmp_path):
        # measured is 0.099918 vs basis 0.1 -> 0.082 %, not 8.2 %
        md = HEAD_5 + "| 质量半径 | 0.1 | 0.099918 | 8.2 % | 是 |\n"
        rows = run_on(tmp_path, md)
        assert rows[0]["check_a"] is False
        assert rows[0]["status"] == "FAIL"

    def test_composite_reference_and_measured_column_is_not_judged(self, tmp_path):
        # "measured vs reference" is one column; treating it as both sides would
        # manufacture a 0 % recomputation and a spurious failure
        md = ("| check | quantity | measured vs reference | sample | relative error | < 3%? |\n"
              "|---|---|---|---|---:|---|\n"
              "| C1 | closed form vs Simpson | max abs 3.531e-14 | 61 heights | 3.946e-12% | yes |\n")
        rows = run_on(tmp_path, md)
        assert rows[0]["check_a"] is None
        assert rows[0]["status"] != "FAIL"
        assert "composite" in rows[0]["why_not_recomputed"]

    def test_coarse_measurement_is_not_judged(self, tmp_path):
        # 0.09 vs 0.1 is only 2 significant digits: the interval is too wide to
        # discriminate, so the row must be reported, not silently passed.
        md = HEAD_5 + "| 量 | 0.1 | 0.09 | 0.5 % | 是 |\n"
        rows = run_on(tmp_path, md)
        assert rows[0]["check_a"] is None


class TestUncheckable:
    """Rows that cannot be machine-checked must be labelled, never passed."""

    def test_prose_error_cell_is_uncheckable_not_pass(self, tmp_path):
        md = HEAD_3 + "| 摩擦起始角 | 0.00 deg (bracket 0.1 deg = 0.3 % of 11.3 deg) | 是 |\n"
        rows = run_on(tmp_path, md)
        assert rows[0]["status"] == "UNCHECKABLE"
        assert rows[0]["check_b"] is None
        assert "prose" in rows[0]["reason"]

    def test_verdict_without_error_column_is_uncheckable(self, tmp_path):
        md = "| 关键量 | < 3 % ? |\n|---|---|\n| 能量守恒 | 是 |\n"
        rows = run_on(tmp_path, md)
        assert rows[0]["status"] == "UNCHECKABLE"
        assert "without an error column" in rows[0]["reason"]

    def test_table_without_verdict_column_is_ignored(self, tmp_path):
        md = "| 量 | 值 |\n|---|---|\n| 密度 | 917 |\n"
        assert run_on(tmp_path, md) == []


class TestRatchet:
    """--check must fail on a new bad claim and on a claim that degrades."""

    def _ledger(self, tmp_path, md, write=True):
        d = tmp_path / "docs"
        d.mkdir(parents=True, exist_ok=True)
        (d / "doc.md").write_text(md, encoding="utf-8")
        code = audit.main([str(tmp_path), "--ledger", "docs/ledger.json"]
                          + (["--write"] if write else ["--check"]))
        return code

    def test_new_failing_claim_breaks_the_ratchet(self, tmp_path):
        self._ledger(tmp_path, HEAD_3 + "| 量 | 0.4 % | 是 |\n")
        assert self._ledger(tmp_path, HEAD_3 + "| 量 | 9.4 % | 是 |\n", write=False) == 1

    def test_unrelated_clean_edit_is_allowed(self, tmp_path):
        self._ledger(tmp_path, HEAD_3 + "| 量 | 0.4 % | 是 |\n")
        assert self._ledger(tmp_path, HEAD_3 + "| 量 | 0.4 % | 是 |\n"
                            "| 另一量 | 1.2 % | 是 |\n", write=False) == 0

    def test_claim_degrading_to_uncheckable_breaks_the_ratchet(self, tmp_path):
        self._ledger(tmp_path, HEAD_3 + "| 量 | 0.4 % | 是 |\n")
        degraded = HEAD_3 + "| 量 | ~0.4 % 见附录 (0.1 deg) | 是 |\n"
        assert self._ledger(tmp_path, degraded, write=False) == 1

    def test_fixed_claim_is_not_a_regression(self, tmp_path):
        self._ledger(tmp_path, HEAD_3 + "| 量 | 9.4 % | 是 |\n")
        assert self._ledger(tmp_path, HEAD_3 + "| 量 | 0.4 % | 是 |\n", write=False) == 0


class TestRealTree:
    """The checked-in ledger must describe the checked-in documentation."""

    def test_committed_ledger_matches_tree(self):
        ledger = json.loads((ROOT / "docs" / "doc_numeric_claims_ledger.json").read_text())
        assert ledger["schema"] == audit.SCHEMA
        fresh = audit.summarise(audit.audit(str(ROOT)))
        assert fresh["by_status"] == ledger["by_status"]
        assert fresh["rows_with_verdict"] == ledger["rows_with_verdict"]

    def test_no_failing_claims_in_tree(self):
        fresh = audit.summarise(audit.audit(str(ROOT)))
        assert [r for r in fresh["records"] if r["status"] == "FAIL"] == []

    def test_ratchet_check_passes(self):
        assert audit.main([str(ROOT), "--check"]) == 0

    def test_coverage_is_reported_not_assumed(self):
        """The audit must state how much of the corpus it actually decides."""
        fresh = audit.summarise(audit.audit(str(ROOT)))
        assert fresh["rows_with_verdict"] > 100
        assert fresh["check_b_evaluated"] > 0
        # a decision is only a decision if some rows were actually judged
        assert fresh["by_status"].get("PASS", 0) > 0
        assert sum(fresh["by_status"].values()) == fresh["rows_with_verdict"]