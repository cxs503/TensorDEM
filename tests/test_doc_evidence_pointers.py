"""Evidence-pointer gate: every path a document cites must be resolvable.

The rule enforced here is the one an external reader relies on:

* R1 — a cited source file (`code`), document (`doc`) or test (`test`) must be
  tracked by git, so it exists in a fresh clone;
* R2 — no cited path may be an ORPHAN (no file in `scripts/`, `benchmarks/`,
  `tests/`, `src/` mentions it), because then no command in the tree can ever
  rebuild it — the claim behind it is unverifiable;
* R3 — a cited run artifact (`data`/`figure`) may be absent from the clone (they
  are not committed), but it must have a producer (REGENERABLE) and must appear
  in the committed ledger.

The ledger `docs/doc_evidence_ledger.json` is the ratchet: it freezes the exact
pointer set, so adding a document that cites a non-existent file fails the gate
until the ledger is regenerated and the new pointer is fixed or justified.

`ClassificationControlTests` are the anti-vacuity controls: they run the audit
on a synthetic tree in which the four outcomes are known, so the gate cannot
silently degrade into "everything resolves".
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import audit_doc_evidence as audit  # noqa: E402


class DocumentedEvidenceGateTests(unittest.TestCase):
    """The committed ledger and the invariants derived from it."""

    @classmethod
    def setUpClass(cls):
        cls.report = audit.scan()
        cls.entries = cls.report["entries"]
        cls.unresolved = [e for e in cls.entries if e["status"] not in ("TRACKED", "EXTERNAL")]

    def test_committed_ledger_is_current(self):
        self.assertEqual(
            audit.main(["--check"]),
            0,
            "docs/doc_evidence_ledger.json is stale: re-run scripts/audit_doc_evidence.py",
        )

    def test_cited_source_docs_and_tests_are_tracked(self):
        broken = [
            f"{e['doc']}:{e['line']} {e['pointer']}"
            for e in self.entries
            if e["kind"] in ("code", "doc", "test") and e["status"] != "TRACKED"
        ]
        self.assertEqual(broken, [], "documents cite source/doc/test paths git does not have")

    def test_no_orphan_pointers(self):
        orphans = [
            f"{e['doc']}:{e['line']} {e['pointer']}"
            for e in self.unresolved
            if e.get("remedy") == "ORPHAN"
        ]
        self.assertEqual(orphans, [], "cited artifacts that no file in the tree can rebuild")

    def test_uncommitted_artifacts_have_a_producer(self):
        bad = [
            f"{e['doc']}:{e['line']} {e['pointer']}"
            for e in self.unresolved
            if e["kind"] not in ("data", "figure")
            or e.get("remedy") != "REGENERABLE"
            or not e.get("producers")
        ]
        self.assertEqual(bad, [], "uncommitted artifacts must name the script that rebuilds them")

    def test_historical_quotes_are_confined_to_the_audit(self):
        # An audit has to quote the broken pointers it repaired. That exemption is
        # a marked block, and it must stay confined to the audit document itself.
        users = set(self.report.get("historical_quotes", {}))
        self.assertEqual(
            users,
            {"docs/dem3d_doc_evidence_pointer_audit.md"},
            "audit:historical blocks are a repair ledger, not a general waiver",
        )

    def test_audit_covers_the_documentation(self):
        self.assertGreater(self.report["documents_scanned"], 50)
        self.assertGreater(self.report["pointers"], 300)
        # A ponytail: the audit must see plenty of resolvable pointers, otherwise
        # the regex silently stopped matching and every gate above passes vacuously.
        tracked = [e for e in self.entries if e["status"] == "TRACKED"]
        self.assertGreater(len(tracked), 0.6 * len(self.entries))


class ClassificationControlTests(unittest.TestCase):
    """Known-outcome controls on a synthetic tree (no git needed)."""

    DOC = """# control document

inline: `artifacts/run/report.json` and `scripts/make.py`
link: [report](../docs/sibling.md) | [tracked](https://example.org/x.json)
missing: `outputs/none/gone.json` and `outputs/other/gone2.json`
untracked: `artifacts/run/local_only.json`
bad test: `tests/test_alpha.py::TestCaseA::test_present`
bad node: `tests/test_alpha.py::TestCaseA::test_absent`
"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "docs").mkdir()
        (root / "docs" / "sub").mkdir()
        (root / "docs" / "sub" / "control.md").write_text(self.DOC, encoding="utf-8")
        (root / "docs" / "sibling.md").write_text("sibling\n", encoding="utf-8")
        (root / "artifacts" / "run").mkdir(parents=True)
        (root / "artifacts" / "run" / "local_only.json").write_text("{}\n", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "test_alpha.py").write_text(
            "class TestCaseA:\n    def test_present(self):\n        pass\n", encoding="utf-8"
        )
        tracked = {
            "docs/sub/control.md",
            "docs/sibling.md",
            "scripts/make.py",
            "artifacts/run/report.json",
            "tests/test_alpha.py",
            # producer of the two `outputs/...` artifacts
            "scripts/build_outputs.py",
        }
        # only `gone.json` has a producer; `gone2.json` must grade ORPHAN
        producers = {"gone.json": ["scripts/build_outputs.py"]}
        self.by_pointer = {
            e["pointer"]: e
            for e in audit.scan(root=root, tracked=tracked, producers=producers)["entries"]
        }

    def tearDown(self):
        self.tmp.cleanup()

    def status(self, pointer):
        return self.by_pointer[pointer]["status"]

    def test_tracked_pointer_resolves(self):
        self.assertEqual(self.status("scripts/make.py"), "TRACKED")
        self.assertEqual(self.status("artifacts/run/report.json"), "TRACKED")

    def test_document_relative_link_resolves(self):
        entry = self.by_pointer["../docs/sibling.md"]
        self.assertEqual(entry["status"], "TRACKED")
        self.assertEqual(entry["resolved"], "docs/sibling.md")

    def test_urls_and_placeholders_are_not_pointers(self):
        self.assertFalse(any("example.org" in p for p in self.by_pointer))

    def test_present_but_untracked_is_not_called_tracked(self):
        self.assertEqual(self.status("artifacts/run/local_only.json"), "UNTRACKED")

    def test_absent_with_producer_is_regenerable(self):
        entry = self.by_pointer["outputs/none/gone.json"]
        self.assertEqual(entry["status"], "MISSING")
        self.assertEqual(entry["remedy"], "REGENERABLE")
        self.assertEqual(entry["producers"], ["scripts/build_outputs.py"])

    def test_absent_without_producer_is_orphan(self):
        entry = self.by_pointer["outputs/other/gone2.json"]
        self.assertEqual(entry["status"], "MISSING")
        self.assertEqual(entry["remedy"], "ORPHAN")
        self.assertEqual(entry["producers"], [])

    def test_referenced_test_node_must_exist(self):
        self.assertEqual(self.status("tests/test_alpha.py::TestCaseA::test_present"), "TRACKED")
        self.assertEqual(self.status("tests/test_alpha.py::TestCaseA::test_absent"), "MISSING")


class OrphanControlTests(unittest.TestCase):
    """A pointer with no producer and no file must be graded ORPHAN."""

    def test_orphan_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "d.md").write_text("`outputs/x/vanish.json`\n", encoding="utf-8")
            (root / "scripts").mkdir()
            (root / "scripts" / "unrelated.py").write_text("print(1)\n", encoding="utf-8")
            entries = audit.scan(root=root, tracked={"docs/d.md"}, producers={})["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["status"], "MISSING")
            self.assertEqual(entries[0]["remedy"], "ORPHAN")


class LostEvidenceControlTests(unittest.TestCase):
    """A blob the ledger recorded as committed but the tree lost must be named.

    This is the shape of the m4857 regression: the ledger said ``TRACKED`` for
    three files under ``artifacts/``, the merge that landed the document did not
    carry the blobs, and ``--check`` reported only a bare ``+/-`` diff. Those two
    outcomes are not the same maintenance action -- a TRACKED -> absent flip has
    to be repaired by restoring the blob, whereas ordinary staleness is repaired
    by regenerating the ledger -- so the gate names them separately.
    """

    def test_tracked_then_gone_is_reported(self):
        committed = [
            {"doc": "docs/d.md", "line": 7, "pointer": "artifacts/run/e.json", "status": "TRACKED"},
            {"doc": "docs/d.md", "line": 9, "pointer": "scripts/make.py", "status": "TRACKED"},
        ]
        current = [
            {"doc": "docs/d.md", "line": 7, "pointer": "artifacts/run/e.json", "status": "MISSING"},
            {"doc": "docs/d.md", "line": 9, "pointer": "scripts/make.py", "status": "TRACKED"},
        ]
        self.assertEqual(
            audit.lost_from_history(committed, current),
            ["docs/d.md:7 artifacts/run/e.json -> MISSING"],
        )

    def test_still_present_is_not_a_loss(self):
        entry = {"doc": "docs/d.md", "line": 7, "pointer": "a/b.json", "status": "TRACKED"}
        self.assertEqual(audit.lost_from_history([entry], [dict(entry)]), [])

    def test_pointer_the_document_no_longer_cites_is_not_a_loss(self):
        entry = {"doc": "docs/d.md", "line": 7, "pointer": "a/b.json", "status": "TRACKED"}
        self.assertEqual(audit.lost_from_history([entry], []), [])

    def test_an_already_uncommitted_pointer_is_not_a_loss(self):
        committed = [{"doc": "docs/d.md", "line": 7, "pointer": "a/b.json", "status": "MISSING"}]
        current = [{"doc": "docs/d.md", "line": 7, "pointer": "a/b.json", "status": "MISSING"}]
        self.assertEqual(audit.lost_from_history(committed, current), [])


class StaleLedgerDiagnosisTests(unittest.TestCase):
    """End-to-end: ``--check`` on a doctored ledger names the lost blob."""

    def test_check_names_a_blob_the_ledger_called_committed(self):
        report = audit.scan()
        target = next(
            e
            for e in report["entries"]
            if e["status"] == "MISSING" and e["kind"] in ("data", "figure")
        )
        doctored = json.loads(json.dumps(report))
        for entry in doctored["entries"]:
            if (entry["doc"], entry["pointer"]) == (target["doc"], target["pointer"]):
                entry["status"] = "TRACKED"
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            ledger.write_text(json.dumps(doctored, indent=1) + "\n", encoding="utf-8")
            err = io.StringIO()
            with unittest.mock.patch.object(audit, "LEDGER", ledger), contextlib.redirect_stderr(err):
                rc = audit.main(["--check"])
        self.assertEqual(rc, 1)
        message = err.getvalue()
        self.assertIn("recorded as committed but gone from the checkout", message)
        self.assertIn(f"{target['doc']}:{target['line']} {target['pointer']}", message)


if __name__ == "__main__":
    unittest.main()