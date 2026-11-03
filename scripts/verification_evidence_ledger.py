#!/usr/bin/env python3
"""Verification-evidence ledger for TensorDEM documentation.

Motivation
----------
The repository publishes ~70 verification notes under ``docs/``.  Each of them
claims that some capability was verified against an analytic law, a literature
value or a reference integrator.  Those claims are only worth what their
*executable backing* is worth: a merged doc plus a merged test module is still
``WITHHELD``-grade evidence if the referenced ``src`` symbols were dropped by a
delivery gate (exactly the failure mode documented in
``docs/dem_analytic_verification_debt_restoration.md``).

This script builds a machine-readable ledger that, for every doc, records

* which test modules / scripts / ``tensordem`` symbols / artefacts it cites,
* whether each reference actually exists, imports, and is collected by pytest,
* and a resulting evidence grade.

Grades (deliberately coarse, one per doc)
-----------------------------------------
``REAL_BACKED``  verification claim + all cited executables exist and collect.
``PARTIAL``      claim is executable, but at least one citation is missing.
``WITHHELD``     claim exists but nothing cited can be executed, or a cited test
                 module fails to collect (the m2536 signature).
``NO_CLAIM``     the doc makes no verification claim (design/charter notes).

The ledger is a *detector*, so it is itself checked against known answers in
``tests/test_verification_evidence_ledger.py``.

Usage
-----
    PYTHONPATH=src python scripts/verification_evidence_ledger.py \
        --output outputs/verification_ledger
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# reference extraction
# ---------------------------------------------------------------------------
TEST_RE = re.compile(r"\btests/test_[A-Za-z0-9_]+\.py\b")
SCRIPT_RE = re.compile(r"\b(?:scripts|benchmarks|examples)/[A-Za-z0-9_]+\.py\b")
SYMBOL_RE = re.compile(r"\btensordem\.([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_/-])")
ARTIFACT_RE = re.compile(r"\b(?:outputs|artifacts|results)/(?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+")

CLAIM_WORDS = (
    "verif", "validat", "audit", "benchmark", "reproduc",
    "reference value", "analytic", "closed form",
)
# Quantified evidence markers: a percentage figure or a named error metric.
# Prose such as "report what passed or failed" is a policy statement, not a
# result, and must not be graded as a claim.
RESULT_RE = re.compile(r"[0-9]+\.[0-9]+\s*%|(?:relative|rel\.)\s*error|±")


def verification_claim(text: str) -> bool:
    """A doc makes a verification claim if it discusses verification at all."""
    low = text.lower()
    return any(word in low for word in CLAIM_WORDS)


def quantified_claim(text: str) -> bool:
    """A claim that carries a number, i.e. an assertion that could be wrong."""
    return verification_claim(text) and bool(RESULT_RE.search(text))


def extract_refs(text: str) -> dict[str, list[str]]:
    def uniq(items):
        seen, out = set(), []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return {
        "tests": uniq(TEST_RE.findall(text)),
        "scripts": uniq(SCRIPT_RE.findall(text)),
        "symbols": uniq(SYMBOL_RE.findall(text)),
        "artifacts": uniq(ARTIFACT_RE.findall(text)),
    }


# ---------------------------------------------------------------------------
# resolvers
# ---------------------------------------------------------------------------
def _library_symbols(src: Path) -> set[str]:
    """Every top-level name defined by any ``src/tensordem`` submodule.

    Static (AST) so that the ledger does not import the whole library to answer
    a question about names, and so that a module that fails to import is still
    audited for the names it defines.
    """
    import ast

    names: set[str] = set()
    for path in sorted((src / "tensordem").rglob("*.py")):
        names.add(path.stem)
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:  # a broken module defines nothing addressable
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                for element in getattr(node.value, "elts", []):
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        names.add(element.value)
    return names


def _import_targets(names: list[str], root: Path = ROOT) -> dict[str, bool]:
    """Check whether ``tensordem.<name>`` resolves inside *this* checkout.

    A reference counts as resolved when the name is a submodule (file or
    package directory) or a public top-level name of any submodule.  This is the
    check that catches the m2536 signature: doc and test merged, ``src`` hunk
    dropped, so the cited symbol exists nowhere in the library.
    """
    if not names:
        return {}
    src = root / "src"
    known = _library_symbols(src)
    status: dict[str, bool] = {}
    for name in names:
        status[name] = (
            (src / "tensordem" / f"{name}.py").exists()
            or (src / "tensordem" / name).is_dir()
            or name in known
        )
    return status


def collection_status() -> dict[str, str]:
    """Map test module -> "ok" | "error" using a pytest collect-only run."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    status: dict[str, str] = {}
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        line = line.strip()
        if line.startswith("ERROR "):
            status[line.split()[1].strip()] = "error"
    # everything else in the test tree that pytest managed to collect is ok
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        rel = f"tests/{path.name}"
        status.setdefault(rel, "ok")
    return status


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def grade_doc(doc: Path, refs: dict[str, list[str]], symbol_status: dict[str, bool],
              collect: dict[str, str], root: Path = ROOT,
              claims: bool = True) -> dict:
    tests_present = {ref: (root / ref).exists() for ref in refs["tests"]}
    scripts_present = {ref: (root / ref).exists() for ref in refs["scripts"]}
    artifacts_present = {ref: (root / ref).exists() for ref in refs["artifacts"]}
    tests_collect = {ref: collect.get(ref, "unknown") for ref in refs["tests"]}

    executables = list(refs["tests"]) + list(refs["scripts"])
    broken_tests = [r for r in refs["tests"] if tests_collect.get(r) == "error"]
    missing_exec = [r for r in refs["tests"] if not tests_present[r]]
    missing_exec += [r for r in refs["scripts"] if not scripts_present[r]]
    missing_symbols = [s for s, ok in symbol_status.items() if not ok]

    if broken_tests or missing_symbols:
        # a cited executable cannot run, or cited code does not exist:
        # the doc's verification claim cannot be reproduced by anyone.
        grade = "WITHHELD"
    elif missing_exec:
        grade = "PARTIAL"
    elif executables:
        grade = "REAL_BACKED"
    elif claims:
        # A quantified verification claim with nothing runnable cited at all.
        grade = "UNBACKED"
    else:
        grade = "NO_CLAIM"

    return {
        "doc": str(doc.relative_to(root)),
        "grade": grade,
        "tests": refs["tests"],
        "scripts": refs["scripts"],
        "symbols": refs["symbols"],
        "n_artifacts": len(refs["artifacts"]),
        "artifacts_present": sum(artifacts_present.values()),
        "tests_present": tests_present,
        "scripts_present": scripts_present,
        "tests_collect": tests_collect,
        "broken_tests": broken_tests,
        "missing_executables": missing_exec,
        "missing_symbols": missing_symbols,
    }


def build_ledger(root: Path = ROOT, collect: dict[str, str] | None = None) -> dict:
    docs = sorted((root / "docs").glob("*.md"))
    refs_by_doc = {doc: extract_refs(doc.read_text(errors="replace")) for doc in docs}
    collect = collection_status() if collect is None else collect

    all_symbols = sorted({s for refs in refs_by_doc.values() for s in refs["symbols"]})
    symbol_status = _import_targets(all_symbols)

    entries = []
    for doc, refs in refs_by_doc.items():
        text = doc.read_text(errors="replace")
        claims = quantified_claim(text)
        doc_symbols = {name: symbol_status.get(name, False) for name in refs["symbols"]}
        entry = grade_doc(doc, refs, doc_symbols, collect, root=root, claims=claims)
        entry["claims_verification"] = claims
        entries.append(entry)

    by_grade: dict[str, int] = {}
    for entry in entries:
        by_grade[entry["grade"]] = by_grade.get(entry["grade"], 0) + 1

    return {
        "root": str(root),
        "n_docs": len(docs),
        "grades": by_grade,
        "collection": {"error_modules": sorted(m for m, s in collect.items() if s == "error"),
                       "n_test_modules": len(collect)},
        "symbol_status": symbol_status,
        "docs": entries,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def write_markdown(ledger: dict, path: Path) -> None:
    lines = [
        "# TensorDEM verification-evidence ledger",
        "",
        "Generated by `scripts/verification_evidence_ledger.py`. One row per doc in `docs/`.",
        "",
        "| grade | docs |",
        "|---|---|",
    ]
    for grade in ("REAL_BACKED", "PARTIAL", "WITHHELD", "UNBACKED", "NO_CLAIM"):
        lines.append(f"| {grade} | {ledger['grades'].get(grade, 0)} |")
    lines += [
        "",
        f"Test modules that fail to collect: {len(ledger['collection']['error_modules'])} "
        f"of {ledger['collection']['n_test_modules']}",
        "",
    ]
    for module in ledger["collection"]["error_modules"]:
        lines.append(f"- `{module}`")
    lines += ["", "## Docs whose verification claim cannot be reproduced", ""]
    for entry in ledger["docs"]:
        if entry["grade"] in ("WITHHELD", "PARTIAL", "UNBACKED"):
            reason = entry["missing_symbols"] or entry["missing_executables"] or entry["broken_tests"]
            lines.append(f"- `{entry['doc']}` **{entry['grade']}** — {reason}")
    lines += ["", "## Backed docs", "", "| doc | tests | scripts |", "|---|---|---|"]
    for entry in ledger["docs"]:
        if entry["grade"] == "REAL_BACKED":
            lines.append(
                f"| `{entry['doc']}` | {len(entry['tests'])} | {len(entry['scripts'])} |"
            )
    path.write_text("\n".join(lines) + "\n")


def write_figure(ledger: dict, png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is a test extra
        return False

    order = ["REAL_BACKED", "PARTIAL", "WITHHELD", "UNBACKED", "NO_CLAIM"]
    counts = [ledger["grades"].get(g, 0) for g in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colours = ["#2a7f62", "#c9a227", "#b3402f", "#8a6d3b", "#9a9a9a"]
    axes[0].bar(order, counts, color=colours)
    axes[0].set_ylabel("docs")
    axes[0].set_title("Evidence grade per verification note")
    axes[0].tick_params(axis="x", rotation=20)
    for i, value in enumerate(counts):
        axes[0].text(i, value, str(value), ha="center", va="bottom", fontsize=9)

    withheld = [e for e in ledger["docs"] if e["grade"] == "WITHHELD"]
    labels = [e["doc"].replace("docs/", "").replace(".md", "")[:34] for e in withheld] or ["none"]
    missing = [len(e["missing_symbols"]) + len(e["broken_tests"]) for e in withheld] or [0]
    axes[1].barh(labels, missing, color="#b3402f")
    axes[1].set_xlabel("unresolvable references")
    axes[1].set_title("WITHHELD notes: why")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/verification_ledger")
    parser.add_argument("--no-collect", action="store_true",
                        help="skip the pytest collect-only probe (fast, less complete)")
    args = parser.parse_args()

    collect = {} if args.no_collect else collection_status()
    ledger = build_ledger(collect=collect)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    (out / "ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True))
    write_markdown(ledger, out / "ledger.md")
    has_png = write_figure(ledger, out / "evidence_grades.png")

    print(json.dumps({k: ledger[k] for k in ("n_docs", "grades", "collection")}, indent=2))
    print(f"wrote {out} (figure: {has_png})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())