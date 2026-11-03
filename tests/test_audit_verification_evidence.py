"""Tests for the static verification-evidence audit.

Two layers:

* **Fixture tests** build a synthetic mini-repository in ``tmp_path`` so each
  evidence class and each grade rule is exercised with a known answer. These
  tests do not depend on the real repository state, so they keep working when
  tests are added or renamed.
* **Repository gates** run the audit on this repository and assert the ledger is
  a faithful, idempotent index: it must cover every source module and every test
  file exactly once, agree with its own CSV/Markdown outputs, and stay stable
  across runs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_verification_evidence as audit_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repo(root: Path) -> Path:
    """Build a synthetic repository exercising every evidence class."""
    _write(
        root / "src" / "tensordem" / "quant_mod.py",
        "def value():\n    return 1.0\n",
    )
    _write(
        root / "src" / "tensordem" / "exact_mod.py",
        "def checksum():\n    return 'abc'\n",
    )
    _write(
        root / "src" / "tensordem" / "order_mod.py",
        "def grows(x):\n    return x\n",
    )
    _write(
        root / "src" / "tensordem" / "error_mod.py",
        "def parse(x):\n    if x < 0:\n        raise ValueError('x')\n    return x\n",
    )
    _write(
        root / "src" / "tensordem" / "smoke_mod.py",
        "def state():\n    return {'a': 1}\n",
    )
    _write(
        root / "src" / "tensordem" / "lonely_mod.py",
        "def never_tested():\n    return 0\n",
    )
    _write(root / "src" / "tensordem" / "__init__.py", "")

    _write(
        root / "tests" / "test_quant.py",
        "import pytest\n"
        "from tensordem import quant_mod\n\n"
        "def test_reference_value():\n"
        "    assert quant_mod.value() == pytest.approx(1.0)\n\n"
        "def test_tolerance_bound():\n"
        "    assert abs(quant_mod.value() - 1.0) < 1e-12\n",
    )
    _write(
        root / "tests" / "test_exact.py",
        "from tensordem.exact_mod import checksum\n\n"
        "def test_digest_matches():\n"
        "    assert checksum() == 'abc'\n",
    )
    _write(
        root / "tests" / "test_order.py",
        "from tensordem.order_mod import grows\n\n"
        "def test_monotone():\n"
        "    assert grows(2) > grows(1)\n",
    )
    _write(
        root / "tests" / "test_order_tolerance.py",
        "from tensordem.order_mod import grows\n\n"
        "def test_relative_error_bounded():\n"
        "    assert abs(grows(2) - 2) / 2 < 0.01\n",
    )
    _write(
        root / "tests" / "test_error.py",
        "import pytest\n"
        "from tensordem.error_mod import parse\n\n"
        "def test_negative_rejected():\n"
        "    with pytest.raises(ValueError):\n"
        "        parse(-1)\n",
    )
    _write(
        root / "tests" / "test_smoke.py",
        "from tensordem.smoke_mod import state\n\n"
        "def test_state_exists():\n"
        "    assert state()\n",
    )
    _write(root / "docs" / "quant_doc.md", "Uses `tensordem.quant_mod` for the reference.\n")
    _write(root / "docs" / "notes.md", "Unrelated prose about ice.\n")
    return root


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_classify_tolerance_assertions_are_quantitative():
    assert audit_mod.classify_test_body("assert x == pytest.approx(2.0)") == "QUANTITATIVE"
    assert audit_mod.classify_test_body("self.assertAlmostEqual(a, b, places=9)") == "QUANTITATIVE"
    assert audit_mod.classify_test_body("assert abs(err) < 1e-9") == "QUANTITATIVE"
    assert audit_mod.classify_test_body("np.testing.assert_allclose(a, b)") == "QUANTITATIVE"


def test_classify_exact_and_ordering_are_distinguished():
    assert audit_mod.classify_test_body("assert checksum() == 'abc'") == "EXACT_VALUE"
    assert audit_mod.classify_test_body("assert grows(2) > grows(1)") == "ORDERING"


def test_ordering_assertion_with_declared_tolerance_is_quantitative():
    # A fractional second argument is a tolerance, not a monotonicity claim.
    assert (
        audit_mod.classify_test_body("self.assertLess(abs(error) / work, .01)")
        == "QUANTITATIVE"
    )
    assert (
        audit_mod.classify_test_body("self.assertLess(abs(delta), 1e-12)")
        == "QUANTITATIVE"
    )


def test_classify_error_path_smoke_and_none():
    assert audit_mod.classify_test_body("with pytest.raises(ValueError):\n    f()") == "ERROR_PATH"
    assert audit_mod.classify_test_body("assert result\n") == "SMOKE"
    assert audit_mod.classify_test_body("result = f()\n") == "NONE"


def test_split_test_functions_ignores_non_test_defs():
    source = "def helper():\n    pass\n\ndef test_a():\n    pass\nclass K:\n    def test_b(self):\n        pass\n"
    names = [name for name, _ in audit_mod.split_test_functions(source)]
    assert names == ["test_a", "test_b"]


# ---------------------------------------------------------------------------
# grading on the synthetic repository
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_ledger(tmp_path_factory):
    root = _make_repo(tmp_path_factory.mktemp("repo"))
    output = root / "docs" / "verification-evidence"
    return audit_mod.audit(root, output)


def _row(ledger, module):
    for row in ledger["module_grades"]:
        if row["module"] == module:
            return row
    raise AssertionError(f"module {module} missing from ledger")


def test_quantitative_module_with_docs_is_documented(fixture_ledger):
    row = _row(fixture_ledger, "quant_mod")
    assert row["grade"] == "QUANTITATIVE_DOCUMENTED"
    assert row["documented"] is True
    assert row["direct_test_count"] == 2


def test_quantitative_module_without_docs_is_flagged(fixture_ledger):
    row = _row(fixture_ledger, "order_mod")
    # Only the tolerance-carrying ordering test should reach QUANTITATIVE.
    assert row["grade"] == "QUANTITATIVE_UNDOCUMENTED"
    assert row["documented"] is False


@pytest.mark.parametrize(
    ("module", "grade"),
    [
        ("exact_mod", "EXACT_ONLY"),
        ("error_mod", "ERROR_PATH_ONLY"),
        ("smoke_mod", "SMOKE_ONLY"),
        ("lonely_mod", "NO_TEST"),
    ],
)
def test_weak_modules_get_the_weakest_correct_grade(fixture_ledger, module, grade):
    assert _row(fixture_ledger, module)["grade"] == grade


def test_never_tested_module_has_no_attribution(fixture_ledger):
    row = _row(fixture_ledger, "lonely_mod")
    assert row["attribution"] == "none"
    assert row["test_count"] == 0


def test_file_level_mentions_are_credited_but_marked_as_such(fixture_ledger):
    """A file-scope import must credit the module without pretending to be direct."""
    root_docs = fixture_ledger["summary"]["doc_file_count"]
    assert root_docs == 2
    row = _row(fixture_ledger, "order_mod")
    assert row["file_only_test_count"] == row["test_count"] - row["direct_test_count"]


def test_summary_counts_match_module_rows(fixture_ledger):
    summary = fixture_ledger["summary"]
    rows = fixture_ledger["module_grades"]
    assert summary["module_count"] == len(rows)
    assert sum(summary["grade_counts"].values()) == len(rows)
    counts = {}
    for row in rows:
        counts[row["grade"]] = counts.get(row["grade"], 0) + 1
    assert summary["grade_counts"] == dict(sorted(counts.items()))


def test_generated_output_excluded_from_documentation_scan(tmp_path):
    """Re-running must not let the generated ledger document its own modules."""
    root = _make_repo(tmp_path)
    output = root / "docs" / "verification-evidence"
    first = audit_mod.audit(root, output)
    audit_mod.write_outputs(first, output)
    second = audit_mod.audit(root, output)
    assert first == second
    assert _row(second, "lonely_mod")["documented"] is False


def test_debt_register_does_not_document_modules(tmp_path):
    """The register names weak modules by construction; it must not credit them."""
    root = _make_repo(tmp_path)
    _write(
        root / "docs" / "verification_debt_register.md",
        "Gap: `lonely_mod` has no tests.\n",
    )
    ledger = audit_mod.audit(root, root / "docs" / "verification-evidence")
    assert _row(ledger, "lonely_mod")["documented"] is False
    assert _row(ledger, "lonely_mod")["grade"] == "NO_TEST"


def test_outputs_are_idempotent_and_consistent(tmp_path):
    root = _make_repo(tmp_path)
    output = root / "docs" / "verification-evidence"
    ledger = audit_mod.audit(root, output)
    paths = audit_mod.write_outputs(ledger, output)
    snapshot = {name: path.read_bytes() for name, path in paths.items()}
    audit_mod.write_outputs(audit_mod.audit(root, output), output)
    for name, path in paths.items():
        assert path.read_bytes() == snapshot[name], f"{name} is not idempotent"

    saved = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert saved == ledger
    with paths["modules_csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == ledger["summary"]["module_count"]
    assert {row["module"] for row in rows} == {r["module"] for r in ledger["module_grades"]}

    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Interpretation contract" in markdown
    assert f"| source modules audited | {ledger['summary']['module_count']} |" in markdown
    weak = [
        row["module"]
        for row in ledger["module_grades"]
        if row["grade"] not in ("QUANTITATIVE_DOCUMENTED", "QUANTITATIVE_UNDOCUMENTED")
    ]
    for module in weak:
        assert f"`{module}`" in markdown, f"{module} missing from the weak-module table"


# ---------------------------------------------------------------------------
# repository gates against the real tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repo_ledger():
    output = REPO_ROOT / "docs" / "verification-evidence"
    return audit_mod.audit(REPO_ROOT, output)


def test_ledger_covers_every_module_and_test_file(repo_ledger):
    modules = set(audit_mod.collect_modules(REPO_ROOT))
    test_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in audit_mod.collect_test_files(REPO_ROOT)
    }
    assert {row["module"] for row in repo_ledger["module_grades"]} == modules
    assert {row["test_file"] for row in repo_ledger["test_files"]} == test_files
    assert len(repo_ledger["module_grades"]) == len(modules)


def test_no_module_is_completely_untested(repo_ledger):
    untested = [row["module"] for row in repo_ledger["module_grades"] if row["grade"] == "NO_TEST"]
    assert untested == [], f"modules with no referencing test: {untested}"


def test_repository_ledger_is_reproducible(repo_ledger):
    again = audit_mod.audit(REPO_ROOT, REPO_ROOT / "docs" / "verification-evidence")
    assert again == repo_ledger


def test_committed_ledger_matches_a_fresh_run(repo_ledger):
    committed = REPO_ROOT / "docs" / "verification-evidence" / "verification_ledger.json"
    assert committed.is_file(), "committed ledger is missing; run the audit script"
    assert json.loads(committed.read_text(encoding="utf-8")) == repo_ledger