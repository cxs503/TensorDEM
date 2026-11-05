"""Execution-evidence ledger for the TensorDEM test suite.

Why this exists
---------------
The repository carries a *static* evidence ledger
(``scripts/audit_verification_evidence.py``) that grades each source module by
the *style* of the assertions its tests contain.  Style says nothing about
whether those tests currently pass: a module can be graded
``QUANTITATIVE_DOCUMENTED`` while its tests are red, skipped, or cannot even be
imported.  This repository has been in exactly that state (published
verification tables whose backing code was missing from the tree), so the
execution outcome is the missing half of the evidence.

The second reason is structural.  A single module that fails to *import* aborts
the whole `pytest tests/` session during collection, so one broken file destroys
the execution evidence for every other file.  This tool therefore runs one
subprocess per test file, isolated: a collection error is recorded as a property
of that file and the rest of the suite is still measured.

What it records
---------------
Per test file: outcome (``green``/``red``/``collection_error``/``no_tests``/
``timeout``), the pytest counters, the failing node ids, the wall time, and the
``tensordem`` source modules the file imports.
Per source module: the union of the outcomes of every test file that imports it,
with green/red/untested classification -- i.e. which merged capabilities have
currently executable evidence.

Determinism
-----------
Wall times are excluded by ``--no-timings`` so the JSON is byte-identical across
runs on the same tree; the committed artefact is generated that way.  Nothing in
the ledger is transcribed by hand.

Usage
-----
    python scripts/verification_execution_ledger.py                 # whole tests/
    python scripts/verification_execution_ledger.py --jobs 4 --no-timings
    python scripts/verification_execution_ledger.py --test-dir /tmp/fixture
    python scripts/verification_execution_ledger.py --pattern 'test_dem3d_*.py'
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DIR = REPO_ROOT / "tests"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "verification-execution"

#: ``from tensordem.x import ...`` / ``import tensordem.x`` / ``tensordem.x.``
_IMPORT_RE = re.compile(r"\btensordem\.([A-Za-z_][A-Za-z0-9_]*)")
_TEST_DEF_RE = re.compile(r"^\s*def (test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
_COUNTER_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|warning|warnings|deselected)")
_SUMMARY_FAILED_RE = re.compile(r"^(FAILED|ERROR) (\S+)", re.MULTILINE)

PASSING_EXIT_CODES = {0}
#: pytest exit code 5 == "no tests were collected"
NO_TESTS_EXIT_CODE = 5


@dataclass
class TestFileRecord:
    test_file: str
    source_modules: list[str] = field(default_factory=list)
    declared_tests: int = 0
    outcome: str = "green"
    exit_code: int | None = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failing_ids: list[str] = field(default_factory=list)
    seconds: float | None = None
    log_tail: str = ""


def discover(test_dir: Path, pattern: str) -> list[Path]:
    """Test files matching ``pattern``, sorted for reproducibility."""
    return sorted(p for p in test_dir.glob(pattern) if p.is_file())


def scan_static(path: Path) -> tuple[list[str], int]:
    """Source modules named by a test file, and how many test functions it declares.

    The scan is deliberately textual (it must work on files that cannot be
    imported): it catches the ``from tensordem.x import`` form used throughout
    this repository plus dotted attribute use.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 0
    modules = sorted(set(_IMPORT_RE.findall(text)))
    return modules, len(_TEST_DEF_RE.findall(text))


def _counters(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for value, word in _COUNTER_RE.findall(output):
        key = {"error": "errors", "errors": "errors"}.get(word, word)
        if key in counts:
            counts[key] = max(counts[key], int(value))
    return counts


def classify(exit_code: int | None, output: str) -> str:
    """Map a pytest run to one of the four outcomes this ledger distinguishes.

    ``collection_error`` is checked before ``red`` because pytest reports
    collection failures with exit code 2 while ordinary assertion failures use
    exit code 1; the difference matters -- a collection error means the tests in
    that file produced *no* evidence at all, not that they disagreed with a
    reference.
    """
    if exit_code is None:
        return "timeout"
    if exit_code == NO_TESTS_EXIT_CODE:
        return "no_tests"
    if exit_code == 0:
        return "green"
    if re.search(r"ERROR collecting", output) or (
        "ImportError while importing test module" in output
    ):
        return "collection_error"
    if exit_code in (2, 3, 4):  # interrupted / internal error / usage error
        return "collection_error" if "collect" in output.lower() else "red"
    return "red"


def run_one(
    path: Path, root: Path, timeout: float, keep_timings: bool, base: Path | None = None
) -> TestFileRecord:
    modules, declared = scan_static(path)
    base = base or root
    try:
        label = str(path.relative_to(base))
    except ValueError:  # fixture trees outside the repository
        label = str(path)
    record = TestFileRecord(
        test_file=label,
        source_modules=modules,
        declared_tests=declared,
    )
    env = dict(os.environ)
    # Keep the sweep polite: these are verification runs, not benchmarks.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(path),
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=no",
        "-rf",
    ]
    started = _now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + "\nTIMEOUT"
        exit_code = None
    elapsed = _now() - started

    record.exit_code = exit_code
    record.outcome = classify(exit_code, output)
    counts = _counters(output)
    record.passed, record.failed = counts["passed"], counts["failed"]
    record.errors, record.skipped = counts["errors"], counts["skipped"]
    record.failing_ids = sorted(
        {f"{kind}:{node}" for kind, node in _SUMMARY_FAILED_RE.findall(output)}
    )
    # pytest prints the wall time in its summary line; normalise it away so the
    # ledger stays byte-identical across runs (see --no-timings).
    normalised = re.sub(r"\bin [0-9.]+s\b", "in <t>s", output)
    record.log_tail = "\n".join(normalised.strip().splitlines()[-4:])
    if keep_timings:
        record.seconds = round(elapsed, 2)
    return record


def _now() -> float:
    import time

    return time.perf_counter()


def module_index(records: list[TestFileRecord]) -> dict[str, dict[str, object]]:
    """Per source module: which test files back it and their worst outcome."""
    severity = {"collection_error": 4, "timeout": 3, "red": 2, "no_tests": 1, "green": 0}
    index: dict[str, dict[str, object]] = {}
    for record in records:
        for module in record.source_modules:
            entry = index.setdefault(
                module,
                {"module": module, "test_files": [], "outcome": "untested"},
            )
            entry["test_files"].append(record.test_file)
            if severity[record.outcome] > severity.get(str(entry["outcome"]), -1):
                entry["outcome"] = record.outcome
    for entry in index.values():
        entry["test_files"] = sorted(set(entry["test_files"]))
    return dict(sorted(index.items()))


def summarise(records: list[TestFileRecord]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.outcome] = counts.get(record.outcome, 0) + 1
    totals = {
        key: sum(getattr(record, key) for record in records)
        for key in ("declared_tests", "passed", "failed", "errors", "skipped")
    }
    red = sorted(
        f"{record.test_file} [{record.outcome}]" for record in records if record.outcome != "green"
    )
    return {
        "n_test_files": len(records),
        "outcome_counts": dict(sorted(counts.items())),
        "totals": totals,
        "non_green": red,
    }


def build_ledger(
    test_dir: Path, pattern: str, jobs: int, timeout: float, keep_timings: bool
) -> dict[str, object]:
    files = discover(test_dir, pattern)
    # Labels are relative to the repository when the tree is inside it (so the
    # committed ledger reads ``tests/test_x.py``), otherwise to the test dir
    # itself (fixture trees).
    base = REPO_ROOT if test_dir.resolve().is_relative_to(REPO_ROOT) else test_dir
    records: list[TestFileRecord] = []
    if jobs <= 1:
        records = [run_one(path, REPO_ROOT, timeout, keep_timings, base) for path in files]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(run_one, path, REPO_ROOT, timeout, keep_timings, base)
                for path in files
            ]
            records = [future.result() for future in futures]
    records.sort(key=lambda record: record.test_file)
    test_files = []
    for record in records:
        entry = asdict(record)
        if not keep_timings:
            entry.pop("seconds")  # keeps the JSON byte-identical across runs
        test_files.append(entry)
    modules = module_index(records)
    return {
        "schema": "tensordem.verification-execution-ledger/1",
        "test_dir": (
            str(test_dir.resolve().relative_to(REPO_ROOT))
            if test_dir.resolve().is_relative_to(REPO_ROOT)
            else str(test_dir)
        ),
        "pattern": pattern,
        "timings_included": keep_timings,
        "summary": summarise(records),
        "test_files": test_files,
        "source_modules": list(modules.values()),
        "module_outcome_counts": _module_counts(modules),
    }


def _module_counts(modules: dict[str, dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in modules.values():
        outcome = str(entry["outcome"])
        counts[outcome] = counts.get(outcome, 0) + 1
    return dict(sorted(counts.items()))


def write_ledger(ledger: dict[str, object], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "execution_ledger.json"
    csv_path = out_dir / "execution_ledger.csv"
    md_path = out_dir / "execution_ledger.md"

    json_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = "test_file,outcome,declared_tests,passed,failed,errors,skipped,failing_ids,source_modules"
    lines = [header]
    for record in ledger["test_files"]:  # type: ignore[index]
        lines.append(
            ",".join(
                [
                    str(record["test_file"]),
                    str(record["outcome"]),
                    str(record["declared_tests"]),
                    str(record["passed"]),
                    str(record["failed"]),
                    str(record["errors"]),
                    str(record["skipped"]),
                    ";".join(record["failing_ids"]),
                    ";".join(record["source_modules"]),
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(ledger), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def render_markdown(ledger: dict[str, object]) -> str:
    summary = ledger["summary"]  # type: ignore[index]
    totals = summary["totals"]  # type: ignore[index]
    out = [
        "# TensorDEM execution-evidence ledger (generated)",
        "",
        "Generated by `scripts/verification_execution_ledger.py`; do not edit by hand.",
        "",
        "## Totals",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| test files run | {summary['n_test_files']} |",
    ]
    for key, value in sorted(summary["outcome_counts"].items()):
        out.append(f"| files `{key}` | {value} |")
    for key, value in totals.items():
        out.append(f"| {key} | {value} |")
    out += ["", "## Non-green test files", ""]
    if summary["non_green"]:
        out += ["| test file | outcome |", "| --- | --- |"]
        out += [f"| `{line}` | |" for line in summary["non_green"]]
    else:
        out.append("None: every test file ran green in isolation.")
    out += ["", "## Source-module outcomes", "", "| outcome | modules |", "| --- | --- |"]
    for key, value in sorted(ledger["module_outcome_counts"].items()):  # type: ignore[index]
        out.append(f"| {key} | {value} |")
    out += ["", "## Per-file detail", "", "| test file | outcome | declared | passed | failed | skipped |", "| --- | --- | --- | --- | --- | --- |"]
    for record in ledger["test_files"]:  # type: ignore[index]
        out.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                record["test_file"],
                record["outcome"],
                record["declared_tests"],
                record["passed"],
                record["failed"],
                record["skipped"],
            )
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-timings",
        action="store_true",
        help="omit wall times so the JSON is byte-identical across runs",
    )
    parser.add_argument("--no-write", action="store_true", help="print the summary only")
    args = parser.parse_args(argv)

    ledger = build_ledger(
        args.test_dir.resolve(),
        args.pattern,
        args.jobs,
        args.timeout,
        keep_timings=not args.no_timings,
    )
    if not args.no_write:
        paths = write_ledger(ledger, args.out_dir.resolve())
        for kind, path in paths.items():
            print(f"wrote {kind}: {path}")
    summary = ledger["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())