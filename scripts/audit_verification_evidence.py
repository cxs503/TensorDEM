"""Build a machine-checked ledger of TensorDEM verification evidence.

The audit is **static and heuristic**: it reads source, test and documentation
files and records *what kind of evidence* exists for every source module in
``src/tensordem``. Each test function is classified by its strongest assertion
style:

``QUANTITATIVE``
    A number is compared against a tolerance or reference:
    ``assertAlmostEqual``, ``pytest.approx``, ``numpy.allclose``,
    ``math.isclose``, a relative-error helper, a ``< k e-n`` bound, or an
    ordering assertion that carries an explicit fractional tolerance.
``EXACT_VALUE``
    An exact value/identity is compared: ``assertEqual``, ``assert x == y``
    (for example a checksum against ``hashlib``).
``ORDERING``
    Only inequality / monotonicity statements, without a declared tolerance.
``ERROR_PATH``
    Only invalid-input assertions (``assertRaises`` / ``pytest.raises``).
``SMOKE``
    Any other bare assertion (truth, finiteness, type, non-empty).
``NONE``
    No assertion at all.

The ledger answers a narrow question -- *which modules are backed by
quantitative tests, and are those modules documented?* -- and nothing more. It
never runs a simulation, never judges physical validity, and a
``QUANTITATIVE`` grade is not a claim that the reference behind the tolerance is
correct or independent. See ``docs/verification-evidence/README.md`` for the
interpretation contract.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCHEMA = "tensordem-verification-evidence-ledger-v1"
DEFAULT_OUTPUT = ROOT / "docs" / "verification-evidence"

# Strongest first; a test's class is the first one that matches.
EVIDENCE_CLASSES: tuple[str, ...] = (
    "QUANTITATIVE",
    "EXACT_VALUE",
    "ORDERING",
    "ERROR_PATH",
    "SMOKE",
    "NONE",
)

GRADES: tuple[str, ...] = (
    "QUANTITATIVE_DOCUMENTED",
    "QUANTITATIVE_UNDOCUMENTED",
    "EXACT_ONLY",
    "ORDERING_ONLY",
    "ERROR_PATH_ONLY",
    "SMOKE_ONLY",
    "NO_TEST",
)

_SKIP_MODULES = frozenset({"__init__", "__main__"})

_TEST_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(test_\w+)", re.MULTILINE)

_QUANTITATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"assertAlmostEqual"),
    re.compile(r"\bapprox\("),
    re.compile(r"assert_allclose"),
    re.compile(r"\bisclose\("),
    re.compile(r"rel_err|relative_error|percent_error"),
    re.compile(r"<\s*\d+(?:\.\d+)?e-\d+"),
)
_ORDERING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"assertGreater|assertLess|assertGreaterEqual|assertLessEqual"),
    # Bare comparison operators inside an assertion, e.g. ``assert a > b``. A
    # comparison against a named (non-literal) tolerance is deliberately counted
    # as ordering, not as quantitative evidence.
    re.compile(r"\bassert\b[^\n]*(?:>|<)"),
)
# A fractional second argument (``, .01)``) next to an ordering assertion is a
# declared tolerance, e.g. ``assertLess(abs(err)/scale, .01)``.
_TOLERANCE_ARG = re.compile(r",\s*0?\.\d+\s*\)")
_SCIENTIFIC_LITERAL = re.compile(r"\d+(?:\.\d+)?e-\d+")
_EXACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"assertEqual|assertIs\("),
    re.compile(r"\bassert\b[^\n]*=="),
)
_ERROR_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"assertRaises|pytest\.raises"),
)
_SMOKE_PATTERN = re.compile(r"\bassert\b")

_REFERENCE_HINT = re.compile(
    r"\b(analytic(?:al)?|closed[- ]form|literature|reference|hand[ -]?calc\w*|benchmark)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TestFunctionRecord:
    file: str
    name: str
    evidence: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def classify_test_body(body: str) -> str:
    """Return the strongest evidence class present in a single test body."""
    quantitative = any(p.search(body) for p in _QUANTITATIVE_PATTERNS)
    if not quantitative and any(p.search(body) for p in _ORDERING_PATTERNS):
        # An ordering assertion with an explicit fractional tolerance is a
        # numerical comparison, not a bare monotonicity statement.
        quantitative = bool(_SCIENTIFIC_LITERAL.search(body) or _TOLERANCE_ARG.search(body))
    if quantitative:
        return "QUANTITATIVE"
    if any(p.search(body) for p in _EXACT_PATTERNS):
        return "EXACT_VALUE"
    if any(p.search(body) for p in _ORDERING_PATTERNS):
        return "ORDERING"
    if any(p.search(body) for p in _ERROR_PATH_PATTERNS):
        return "ERROR_PATH"
    if _SMOKE_PATTERN.search(body):
        return "SMOKE"
    return "NONE"


def split_test_functions(source: str) -> list[tuple[str, str]]:
    """Split a test module into ``(name, body)`` pairs.

    String literals are not parsed, so a ``def test_`` occurring inside a
    multi-line string would create a spurious record. That is a documented
    limitation of this static audit, not a correctness requirement.
    """
    matches = list(_TEST_DEF.finditer(source))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        blocks.append((match.group(1), source[start:end]))
    return blocks


def _word_reference(name: str) -> re.Pattern[str]:
    # A dotted reference such as ``tensordem.buoyancy_wet`` is the normal way a
    # module is used, so a preceding dot must NOT disqualify a match. Only word
    # characters disqualify it, which keeps ``dem`` from matching inside
    # ``tensordem`` or ``dem3d``.
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")


def _mentions(source: str, names: Sequence[str]) -> list[str]:
    return [name for name in names if _word_reference(name).search(source)]


def _strongest(evidences: Sequence[str]) -> str:
    for candidate in EVIDENCE_CLASSES:
        if candidate in evidences:
            return candidate
    return "NONE"


def collect_modules(root: Path) -> list[str]:
    src = root / "src" / "tensordem"
    if not src.is_dir():
        raise FileNotFoundError(f"source package not found: {src}")
    return sorted(
        path.stem for path in src.glob("*.py") if path.stem not in _SKIP_MODULES
    )


def collect_test_files(root: Path) -> list[Path]:
    tests = root / "tests"
    if not tests.is_dir():
        raise FileNotFoundError(f"test directory not found: {tests}")
    return sorted(tests.glob("test_*.py"))


# Documentation that enumerates module names *by construction* is excluded from the
# documentation scan. Counting it would be self-fulfilling: this ledger and its debt
# register name every weak module, which would otherwise mark those modules as
# "documented" and erase the very finding they record.
_DOC_SCAN_EXCLUSIONS: tuple[str, ...] = ("docs/verification_debt_register.md",)


def collect_doc_files(
    root: Path, output_dir: Path | None = None, exclusions: Sequence[str] | None = None
) -> list[Path]:
    """Documentation markdown files eligible for module-name matching.

    Excludes the generated output directory (keeps the ledger idempotent) and the
    audit documents listed in ``_DOC_SCAN_EXCLUSIONS`` (prevents the metric from
    being satisfied by the report that describes it).
    """
    docs = root / "docs"
    if not docs.is_dir():
        return []
    excluded_dir = Path(output_dir).resolve() if output_dir is not None else None
    excluded_files = {
        (root / rel).resolve()
        for rel in (exclusions if exclusions is not None else _DOC_SCAN_EXCLUSIONS)
    }
    files = []
    for path in sorted(docs.rglob("*.md")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if excluded_dir is not None and excluded_dir in resolved.parents:
            continue
        if resolved in excluded_files:
            continue
        files.append(path)
    return files


def collect_script_files(root: Path) -> list[Path]:
    scripts = root / "scripts"
    if not scripts.is_dir():
        return []
    return sorted(path for path in scripts.glob("*.py") if path.is_file())


def audit(root: Path = ROOT, output_dir: Path | None = None) -> dict[str, Any]:
    """Return the ledger dictionary for ``root`` (deterministic, no timestamps)."""
    root = Path(root)
    modules = collect_modules(root)
    test_files = collect_test_files(root)
    docs = collect_doc_files(root, output_dir or DEFAULT_OUTPUT)
    scripts = collect_script_files(root)

    direct_tests: dict[str, list[str]] = {name: [] for name in modules}
    direct_evidence: dict[str, list[str]] = {name: [] for name in modules}
    file_tests: dict[str, list[str]] = {name: [] for name in modules}
    file_evidence: dict[str, list[str]] = {name: [] for name in modules}
    module_scripts: dict[str, list[str]] = {name: [] for name in modules}
    module_docs: dict[str, list[str]] = {name: [] for name in modules}
    reference_hints: dict[str, int] = {name: 0 for name in modules}

    test_records: list[TestFunctionRecord] = []
    test_file_rows: list[dict[str, Any]] = []

    for test_path in test_files:
        source = _read(test_path)
        rel = test_path.relative_to(root).as_posix()
        functions = split_test_functions(source)
        counts = {name: 0 for name in EVIDENCE_CLASSES}
        if not functions:
            counts["NONE"] = 1
        file_modules = _mentions(source, modules)
        function_evidence: dict[str, str] = {}
        for name, body in functions:
            evidence = classify_test_body(body)
            counts[evidence] += 1
            test_id = f"{rel}::{name}"
            test_records.append(TestFunctionRecord(rel, name, evidence))
            function_evidence[test_id] = evidence
            for module in _mentions(body, modules):
                direct_tests[module].append(test_id)
                direct_evidence[module].append(evidence)
                if _REFERENCE_HINT.search(body):
                    reference_hints[module] += 1
            body_modules = _mentions(body, modules)
            has_hint = bool(_REFERENCE_HINT.search(body))
            for module in file_modules:
                if module not in body_modules:
                    file_tests[module].append(test_id)
                    file_evidence[module].append(evidence)
                    if has_hint:
                        reference_hints[module] += 1
        test_file_rows.append(
            {
                "test_file": rel,
                "test_functions": len(functions),
                "modules_referenced": len(file_modules),
                **{f"{key.lower()}_count": counts[key] for key in EVIDENCE_CLASSES},
            }
        )

    for script_path in scripts:
        source = _read(script_path)
        rel = script_path.relative_to(root).as_posix()
        for module in _mentions(source, modules):
            module_scripts[module].append(rel)

    for doc_path in docs:
        source = _read(doc_path)
        rel = doc_path.relative_to(root).as_posix()
        for module in _mentions(source, modules):
            module_docs[module].append(rel)

    module_rows: list[dict[str, Any]] = []
    for module in modules:
        tests = sorted(set(direct_tests[module]) | set(file_tests[module]))
        evidences = list(direct_evidence[module]) + list(file_evidence[module])
        best = _strongest(evidences)
        # Confidence of the grade: "direct" when the strongest class was reached
        # by a test that names the module, "file" when only same-file tests did,
        # "none" when no test references it.
        if not tests:
            attribution = "none"
        elif best in direct_evidence[module]:
            attribution = "direct"
        else:
            attribution = "file"
        direct_count = len(set(direct_tests[module]))
        file_only_count = len(tests) - direct_count
        if not tests:
            grade = "NO_TEST"
        elif best == "QUANTITATIVE":
            grade = (
                "QUANTITATIVE_DOCUMENTED"
                if module_docs[module]
                else "QUANTITATIVE_UNDOCUMENTED"
            )
        elif best == "EXACT_VALUE":
            grade = "EXACT_ONLY"
        elif best == "ORDERING":
            grade = "ORDERING_ONLY"
        elif best == "ERROR_PATH":
            grade = "ERROR_PATH_ONLY"
        else:
            grade = "SMOKE_ONLY"
        module_rows.append(
            {
                "module": module,
                "grade": grade,
                "best_evidence": best,
                "attribution": attribution,
                "documented": bool(module_docs[module]),
                "test_count": len(tests),
                "direct_test_count": direct_count,
                "file_only_test_count": file_only_count,
                "tests": tests,
                "scripts": module_scripts[module],
                "docs": module_docs[module],
                "reference_hint_tests": reference_hints[module],
            }
        )

    grade_counts: dict[str, int] = {}
    for row in module_rows:
        grade_counts[row["grade"]] = grade_counts.get(row["grade"], 0) + 1

    attribution_counts: dict[str, int] = {}
    for row in module_rows:
        key = row["attribution"]
        attribution_counts[key] = attribution_counts.get(key, 0) + 1

    return {
        "schema": LEDGER_SCHEMA,
        "summary": {
            "module_count": len(modules),
            "test_file_count": len(test_files),
            "test_function_count": len(test_records),
            "doc_file_count": len(docs),
            "script_file_count": len(scripts),
            "grade_counts": {key: grade_counts[key] for key in sorted(grade_counts)},
            "attribution_counts": {
                key: attribution_counts[key] for key in sorted(attribution_counts)
            },
        },
        "module_grades": module_rows,
        "test_files": test_file_rows,
        "test_functions": [
            {"file": record.file, "name": record.name, "evidence": record.evidence}
            for record in test_records
        ],
    }


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _csv_text(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    return stream.getvalue()


GRADE_MEANINGS = {
    "QUANTITATIVE_DOCUMENTED": "tolerance/reference assertion present; module named in docs/",
    "QUANTITATIVE_UNDOCUMENTED": "tolerance/reference assertion present; module not named in docs/",
    "EXACT_ONLY": "only exact-value comparisons (e.g. checksums) without a tolerance/reference",
    "ORDERING_ONLY": "only inequality/monotonicity assertions",
    "ERROR_PATH_ONLY": "only invalid-input assertions",
    "SMOKE_ONLY": "only bare truth/finiteness/type assertions",
    "NO_TEST": "no test function references the module",
}


def render_markdown(ledger: dict[str, Any]) -> str:
    summary = ledger["summary"]
    lines = [
        "# TensorDEM verification-evidence ledger",
        "",
        "Generated by `scripts/audit_verification_evidence.py` "
        f"(schema `{ledger['schema']}`). Re-generate rather than hand-edit:",
        "",
        "```bash",
        "python scripts/audit_verification_evidence.py",
        "```",
        "",
        "## Scope",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| source modules audited | {summary['module_count']} |",
        f"| test files | {summary['test_file_count']} |",
        f"| test functions | {summary['test_function_count']} |",
        f"| documentation files | {summary['doc_file_count']} |",
        f"| scripts | {summary['script_file_count']} |",
        "",
        "## Grades",
        "",
        "| grade | meaning | modules |",
        "| --- | --- | --- |",
    ]
    for grade in GRADES:
        lines.append(
            f"| `{grade}` | {GRADE_MEANINGS[grade]} | "
            f"{summary['grade_counts'].get(grade, 0)} |"
        )
    lines += [
        "",
        "## Modules below the quantitative grade",
        "",
        "| module | grade | tests (direct) | grade confidence | documented |",
        "| --- | --- | --- | --- | --- |",
    ]
    weak = [
        row
        for row in ledger["module_grades"]
        if row["grade"] not in ("QUANTITATIVE_DOCUMENTED", "QUANTITATIVE_UNDOCUMENTED")
    ]
    if not weak:
        lines.append("| (none) | - | - | - | - |")
    for row in weak:
        lines.append(
            f"| `{row['module']}` | `{row['grade']}` | "
            f"{row['test_count']} ({row['direct_test_count']}) | "
            f"{row['attribution']} | {'yes' if row['documented'] else 'no'} |"
        )
    undocumented = [
        row["module"] for row in ledger["module_grades"] if not row["documented"]
    ]
    lines += [
        "",
        "## Modules not named in any documentation file",
        "",
        (", ".join(f"`{name}`" for name in undocumented) if undocumented else "(none)"),
        "",
        "## Interpretation contract",
        "",
        "1. The grade is a **static, heuristic** classification of assertion style. "
        "It is not a physical-validation statement and it never runs the solver.",
        "2. `QUANTITATIVE` means *a number is compared against a tolerance or a "
        "reference*. It does not mean the reference behind it is independent or "
        "correct: a module checked against a hand-typed copy of its own output is "
        "still `QUANTITATIVE`.",
        "3. `attribution` is `direct` when a test function body names the module, and "
        "`file` when the module only appears elsewhere in a test file (for example "
        "in a file-scope import). `file` attribution is coarse: it credits every "
        "test in that file, so treat it as weaker evidence.",
        "4. `reference_hint_tests` counts test functions that both credit the module "
        "and mention `analytic`, `closed-form`, `literature`, `reference`, "
        "`hand calculation` or `benchmark`. It is a reviewer hint, not a guarantee.",
        "5. Generated output in `docs/verification-evidence/` is excluded from the "
        "documentation scan, which keeps the ledger idempotent.",
        "",
        "The companion register `docs/verification_debt_register.md` records the debts "
        "this ledger exposes, with proposed single-agent scopes.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(ledger: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "json": output_dir / "verification_ledger.json",
        "modules_csv": output_dir / "verification_ledger_modules.csv",
        "tests_csv": output_dir / "verification_ledger_tests.csv",
        "markdown": output_dir / "verification_ledger.md",
    }
    _write_atomic(
        paths["json"],
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    module_rows = [
        {
            "module": row["module"],
            "grade": row["grade"],
            "best_evidence": row["best_evidence"],
            "attribution": row["attribution"],
            "documented": row["documented"],
            "test_count": row["test_count"],
            "direct_test_count": row["direct_test_count"],
            "file_only_test_count": row["file_only_test_count"],
            "reference_hint_tests": row["reference_hint_tests"],
            "tests": ";".join(row["tests"]),
            "docs": ";".join(row["docs"]),
        }
        for row in ledger["module_grades"]
    ]
    _write_atomic(
        paths["modules_csv"],
        _csv_text(
            module_rows,
            [
                "module",
                "grade",
                "best_evidence",
                "attribution",
                "documented",
                "test_count",
                "direct_test_count",
                "file_only_test_count",
                "reference_hint_tests",
                "tests",
                "docs",
            ],
        ),
    )
    _write_atomic(
        paths["tests_csv"],
        _csv_text(
            ledger["test_files"],
            [
                "test_file",
                "test_functions",
                "modules_referenced",
                "quantitative_count",
                "exact_value_count",
                "ordering_count",
                "error_path_count",
                "smoke_count",
                "none_count",
            ],
        ),
    )
    _write_atomic(paths["markdown"], render_markdown(ledger))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    ledger = audit(args.root, args.output)
    paths = write_outputs(ledger, args.output)
    print(json.dumps(ledger["summary"], indent=2, sort_keys=True))
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())