"""Known-answer and negative-control tests for the verification-evidence ledger.

The ledger (``scripts/verification_evidence_ledger.py``) is a *detector*: it
claims that a doc's verification note is only worth what its executable backing
is worth.  A detector is worthless unless it is checked against cases whose
answer is known independently, so this module pins three kinds of case:

* **known answers** taken from the repository's own debt ledger
  (``docs/dem_analytic_verification_debt_restoration.md`` §4): four restored
  analytic contracts must grade ``REAL_BACKED`` and the one deliberately
  deferred contract must grade ``WITHHELD``;
* **negative controls**: synthetic docs whose referenced symbol or test module
  does not exist must be flagged, so a detector that always answers "fine"
  fails here;
* **precision controls**: prose and schema tags that merely *look* like
  references (``tensordem.restart/1``, "report what passed or failed") must not
  be graded as claims, so a detector that always answers "broken" fails too.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "verification_evidence_ledger", ROOT / "scripts" / "verification_evidence_ledger.py"
)
ledger_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ledger_mod)

extract_refs = ledger_mod.extract_refs
grade_doc = ledger_mod.grade_doc
quantified_claim = ledger_mod.quantified_claim

# The four contracts restored by m2536, and the one left as honest debt.
RESTORED_DOCS = (
    "dem3d_hertz_contact_verification.md",
    "dem3d_restitution_verification.md",
    "dem3d_coulomb_friction_cone_verification.md",
    "dem3d_tangential_friction_verification.md",
)
DEFERRED_DOC = "fracture_gen_mass_footprint_audit.md"
DEFERRED_TEST = "tests/test_fracture_gen_mass_footprint.py"


class ExtractionTests(unittest.TestCase):
    def test_schema_tag_is_not_a_symbol_reference(self):
        """``tensordem.restart/1`` is a file-schema tag, not a library symbol."""
        refs = extract_refs("old `tensordem.restart/1` files still load")
        self.assertEqual(refs["symbols"], [])

    def test_hyphenated_schema_tag_is_not_a_symbol_reference(self):
        refs = extract_refs("schema `tensordem.ice-floe-physics/2`")
        self.assertEqual(refs["symbols"], [])

    def test_real_references_are_found(self):
        refs = extract_refs(
            "Run `PYTHONPATH=src python scripts/verify_dem3d_restitution.py` then "
            "`pytest tests/test_dem3d_restitution.py`; see `tensordem.contact_tangential`."
        )
        self.assertEqual(refs["tests"], ["tests/test_dem3d_restitution.py"])
        self.assertEqual(refs["scripts"], ["scripts/verify_dem3d_restitution.py"])
        self.assertEqual(refs["symbols"], ["contact_tangential"])

    def test_policy_prose_is_not_a_quantified_claim(self):
        """"report what passed or failed" carries no result and is not a claim."""
        self.assertFalse(
            quantified_claim("Report exactly what was run, what passed or failed, and what was not tested.")
        )

    def test_percentage_figure_is_a_quantified_claim(self):
        self.assertTrue(quantified_claim("Verification: rel. error 0.42 % against the reference."))


class GradingTests(unittest.TestCase):
    def _root(self, tmp_path: Path, text: str, name: str = "note.md") -> tuple[Path, Path]:
        docs = tmp_path / "docs"
        docs.mkdir(exist_ok=True)
        doc = docs / name
        doc.write_text(text)
        return tmp_path, doc

    def _grade(self, tmp_path: Path, text: str, *, symbols=(), collect=None) -> str:
        root, doc = self._root(tmp_path, text)
        refs = extract_refs(text)
        status = {name: name in symbols for name in refs["symbols"]}
        return grade_doc(
            doc, refs, status, collect or {}, root=root, claims=quantified_claim(text)
        )["grade"]

    # --- negative controls: the detector must fire -------------------------
    def test_ghost_test_module_is_flagged(self):
        grade = self._grade(
            self._fresh(), "Verified to 0.1 % by `tests/test_ghost_module.py`."
        )
        self.assertEqual(grade, "PARTIAL")

    def test_ghost_symbol_makes_the_doc_withheld(self):
        text = "Verified to 0.1 % by `tests/test_dem3d_restitution.py` using `tensordem.ghost_symbol`."
        grade = self._grade(self._fresh(), text, symbols=())
        self.assertEqual(grade, "WITHHELD")

    def test_non_collecting_test_module_makes_the_doc_withheld(self):
        text = f"Verified to 0.1 % by `{DEFERRED_TEST}`."
        grade = self._grade(self._fresh(), text, collect={DEFERRED_TEST: "error"})
        self.assertEqual(grade, "WITHHELD")

    def test_quantified_claim_without_any_executable_is_unbacked(self):
        grade = self._grade(self._fresh(), "Verification: rel. error 0.42 % on the reference case.")
        self.assertEqual(grade, "UNBACKED")

    # --- precision controls: the detector must not fire on prose ----------
    def test_prose_without_numbers_is_no_claim(self):
        grade = self._grade(self._fresh(), "This note describes the verification workflow.")
        self.assertEqual(grade, "NO_CLAIM")

    def test_backed_doc_grades_real(self):
        text = "Verified to 0.1 % by `tests/test_dem3d_restitution.py`."
        grade = self._grade(self._fresh(), text)
        self.assertEqual(grade, "REAL_BACKED")

    # -- helpers ----------------------------------------------------------
    _counter = 0

    def _fresh(self) -> Path:
        """A unique throwaway directory (unittest has no tmp_path fixture)."""
        import tempfile

        GradingTests._counter += 1
        path = Path(tempfile.mkdtemp(prefix=f"ledger_case_{GradingTests._counter}_"))
        self.addCleanup(lambda p=path: __import__("shutil").rmtree(p, ignore_errors=True))
        # The cited, known-good test module exists; only the *ghost* cases must fail.
        tests = path / "tests"
        tests.mkdir()
        (tests / "test_dem3d_restitution.py").write_text("# stub for ledger grading\n")
        return path


class RepositoryKnownAnswerTests(unittest.TestCase):
    """Integration: the ledger must reproduce the repository's own debt record."""

    @classmethod
    def setUpClass(cls):
        cls.ledger = ledger_mod.build_ledger()
        cls.by_doc = {entry["doc"]: entry for entry in cls.ledger["docs"]}

    def test_every_doc_is_graded(self):
        self.assertEqual(self.ledger["n_docs"], len(self.ledger["docs"]))
        self.assertGreaterEqual(self.ledger["n_docs"], 70)

    def test_restored_analytic_contracts_are_backed(self):
        """m2536 §3: Hertz / restitution / friction cone / tangential were restored."""
        for name in RESTORED_DOCS:
            with self.subTest(doc=name):
                entry = self.by_doc[f"docs/{name}"]
                self.assertEqual(entry["grade"], "REAL_BACKED", entry["missing_symbols"])

    def test_deferred_contract_is_still_withheld(self):
        """m2536 §4: it still cannot be collected, so its claim is not reproducible."""
        entry = self.by_doc[f"docs/{DEFERRED_DOC}"]
        self.assertEqual(entry["grade"], "WITHHELD")
        self.assertIn(DEFERRED_TEST, entry["broken_tests"])

    def test_no_doc_cites_a_missing_executable(self):
        """Zero PARTIAL: every script and test module a doc names exists on disk."""
        partial = [e["doc"] for e in self.ledger["docs"] if e["grade"] == "PARTIAL"]
        self.assertEqual(partial, [])

    def test_ledger_is_self_consistent(self):
        counts: dict[str, int] = {}
        for entry in self.ledger["docs"]:
            self.assertIn(entry["grade"],
                          {"REAL_BACKED", "PARTIAL", "WITHHELD", "UNBACKED", "NO_CLAIM"})
            counts[entry["grade"]] = counts.get(entry["grade"], 0) + 1
        self.assertEqual(counts, self.ledger["grades"])


if __name__ == "__main__":
    unittest.main()