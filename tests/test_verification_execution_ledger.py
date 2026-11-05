"""Tests for ``scripts/verification_execution_ledger.py``.

The instrument is the thing being tested, so its classifier is pinned against
synthetic fixture suites with *known* outcomes rather than against the live
repository (which changes under it):

* a passing file,
* an assertion failure,
* a module that cannot be imported (the failure mode that aborts a whole
  ``pytest tests/`` session, and therefore the reason this tool exists),
* a file with no test functions,
* a file that exceeds the per-file timeout.

The determinism requirement is explicit: with ``--no-timings`` two runs over the
same tree must produce byte-identical JSON, otherwise the committed artefact is
not reproducible evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import verification_execution_ledger as vel  # noqa: E402

PASSING = '''
"""Fixture: mentions tensordem.hull_geometry in its docstring only."""


def test_one():
    assert 1 + 1 == 2


def test_two():
    assert "a" < "b"
'''

FAILING = '''
def test_fails():
    assert 1 + 1 == 3
'''

BROKEN_IMPORT = '''
import a_module_that_does_not_exist_anywhere

def test_never_runs():
    assert True
'''

NO_TESTS = '''
VALUE = 1
'''

SLOW = '''
import time


def test_slow():
    time.sleep(30)
'''


def _fixture(tmp_path: Path, **files: str) -> Path:
    for name, text in files.items():
        (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")
    return tmp_path


def test_scan_static_reads_modules_and_counts_test_functions():
    modules, declared = vel.scan_static(_write_tmp("scan", PASSING))
    assert modules == ["hull_geometry"]
    assert declared == 2


def _write_tmp(name: str, text: str) -> Path:
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix=f"vel_{name}_"))
    path = directory / "test_fixture.py"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "text,exit_code,expected",
    [
        ("", 0, "green"),
        ("", 1, "red"),
        ("", 5, "no_tests"),
        ("ERROR collecting tests/x.py\nImportError while importing test module", 2, "collection_error"),
        ("", None, "timeout"),
    ],
)
def test_classify_maps_pytest_outcomes(text, exit_code, expected):
    assert vel.classify(exit_code, text) == expected


def test_isolated_run_reports_one_outcome_per_file(tmp_path):
    test_dir = _fixture(
        tmp_path,
        test_green=PASSING,
        test_red=FAILING,
        test_broken=BROKEN_IMPORT,
        test_empty=NO_TESTS,
    )
    ledger = vel.build_ledger(test_dir, "test_*.py", jobs=2, timeout=120, keep_timings=False)
    outcomes = {record["test_file"]: record["outcome"] for record in ledger["test_files"]}

    assert outcomes["test_green.py"] == "green"
    assert outcomes["test_red.py"] == "red"
    # the file that kills a whole-session run is contained, not fatal
    assert outcomes["test_broken.py"] == "collection_error"
    assert outcomes["test_empty.py"] == "no_tests"

    summary = ledger["summary"]
    assert summary["n_test_files"] == 4
    assert summary["outcome_counts"] == {
        "collection_error": 1,
        "green": 1,
        "no_tests": 1,
        "red": 1,
    }
    assert summary["totals"]["passed"] >= 2
    red = next(r for r in ledger["test_files"] if r["test_file"] == "test_red.py")
    assert red["failed"] == 1
    assert any("test_fails" in failing for failing in red["failing_ids"])
    broken = next(r for r in ledger["test_files"] if r["test_file"] == "test_broken.py")
    assert broken["passed"] == 0  # no evidence at all, unlike a red file


def test_timeout_is_contained_and_labelled(tmp_path):
    test_dir = _fixture(tmp_path, test_slow=SLOW, test_green=PASSING)
    ledger = vel.build_ledger(test_dir, "test_*.py", jobs=1, timeout=1.0, keep_timings=False)
    outcomes = {record["test_file"]: record["outcome"] for record in ledger["test_files"]}
    assert outcomes["test_slow.py"] == "timeout"
    assert outcomes["test_green.py"] == "green"


def test_module_index_takes_the_worst_outcome(tmp_path):
    test_dir = _fixture(
        tmp_path,
        test_a=PASSING,
        test_b=FAILING.replace("def test_fails", "import os  # tensordem.fracture_gen\ndef test_fails"),
    )
    ledger = vel.build_ledger(test_dir, "test_*.py", jobs=2, timeout=120, keep_timings=False)
    modules = {entry["module"]: entry for entry in ledger["source_modules"]}
    assert modules["fracture_gen"]["outcome"] == "red"
    assert modules["fracture_gen"]["test_files"] == ["test_b.py"]
    assert modules["hull_geometry"]["outcome"] == "green"


def test_json_is_byte_identical_without_timings(tmp_path):
    test_dir = _fixture(tmp_path, test_green=PASSING, test_red=FAILING)
    kwargs = dict(pattern="test_*.py", jobs=2, timeout=120, keep_timings=False)
    first = json.dumps(vel.build_ledger(test_dir, **kwargs), indent=2, sort_keys=True)
    second = json.dumps(vel.build_ledger(test_dir, **kwargs), indent=2, sort_keys=True)
    assert first == second
    assert '"seconds"' not in first


def test_written_artefacts_cover_every_file(tmp_path):
    test_dir = _fixture(tmp_path, test_green=PASSING, test_red=FAILING)
    ledger = vel.build_ledger(test_dir, "test_*.py", jobs=2, timeout=120, keep_timings=False)
    out_dir = tmp_path / "out"
    paths = vel.write_ledger(ledger, out_dir)
    assert paths["json"].exists() and paths["csv"].exists() and paths["md"].exists()

    rows = paths["csv"].read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("test_file,outcome")
    assert len(rows) == 1 + len(ledger["test_files"])
    markdown = paths["md"].read_text(encoding="utf-8")
    assert "test_red.py" in markdown and "None: every test file ran green" not in markdown