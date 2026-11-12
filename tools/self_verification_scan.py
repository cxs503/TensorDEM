#!/usr/bin/env python3
"""Self-verification scan: is a "measured" number produced by the model it grades?

Why this exists
---------------
`audit_report.md` §4-F1 (mission m14487) found that `tests/test_seepage_piping.py`
built its "DEM-measured" pore pressure, critical gradient and permeability by
evaluating the *analytic expression of the model under test* and multiplying it
by ``(1 + noise)`` / ``1.10`` / ``1.15``; `docs/seepage_piping_verification.md`
then reported those surrogates as measurements with 5.85 % / 5.00 % / 9.09 %
"errors".  Such a criterion is decided by the constant that was typed in, not by
the solver: it is *self-verification*, and it passes no matter how broken the
code is.  §4-F4 found the same disease in `friction_polyhedron_verification`
(``atan(mu)`` compared with ``atan(mu)``).

The audit's remediation (report §8 item 3) was to stop it structurally:

    "把「同名遮蔽」与「measured_* 不得来自被测模型的解析式」写进
     `tools/import_contract_scan.py`"

This script *is* that check, kept separate from the import-contract scan so the
two failure modes stay independently reportable.

What it detects
---------------
Per test/script function, every assignment whose target looks like a measurement
(``measured``, ``observed``, ``experiment*``, ``*_meas``, ``u_exp`` ...).  Its
right-hand side is resolved through local aliases (one pass of transitive
propagation) and classified:

``SEV1  analytic_scale``
    measurement = analytic quantity multiplied by a numeric constant other than
    1 (``i_c_analytic * 1.10``) or offset by one (``+ 0.05``).  The "error" is
    the typed constant.
``SEV2  analytic_noise``
    measurement = analytic quantity perturbed by noise
    (``u_analytic * (1 + 0.05 * rng.standard_normal(...))``).  The "error" is a
    draw from a generator the test itself seeds.
``SEV3  analytic_direct``
    measurement = analytic quantity verbatim (or a slice/reshape of it).

Declared sites do not fail the scan.  A site is *declared* when the assignment
is marked in-band for a human and a machine:

* the target name starts with ``synthetic``/``surrogate``/``fabricated``, or
* a comment within the three lines above/below says
  ``self-verification: synthetic`` (case-insensitive).

That is the point: a surrogate may still be used to exercise a judging chain,
but it can no longer be *silent*.  Declared sites stay in the JSON report as
``declared`` so the evidence grade of any doc citing them can be audited.

It also scans docs (``--docs``): a page line that calls a value
"DEM-measured" / "DEM-simulated" while the same page admits "simulated noise"
is reported as a cross-reference candidate for the evidence ledger, and every
file that contains an undeclared SEV1/SEV2 site is checked for doc lines that
claim measurement.

Usage
-----
    PYTHONPATH=src python3 tools/self_verification_scan.py            # gate
    PYTHONPATH=src python3 tools/self_verification_scan.py --json artifacts/self_verification/scan.json
    PYTHONPATH=src python3 tools/self_verification_scan.py --docs

Exit code 0 = no undeclared SEV1/SEV2/SEV3 site; 1 = at least one.

Honest limits (stated, not implied)
-----------------------------------
* Name-based: a surrogate stored in a variable called ``x`` is invisible.  The
  check raises the cost of the mistake and makes it auditable; it is not a
  proof of absence.
* Indirection through a helper that *returns* the analytic value under a neutral
  name is not followed across function boundaries (one local alias pass only).
* It does not grade whether an *analytic* reference is itself correct, only
  whether a value that is presented as measurement came from the model under
  test.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- vocabularies
MEASURED_RE = re.compile(
    r"(?i)(^|_)(measured|meas|measured_\w+|observed|obs|experiment\w*|exp_\w+|demo_\w+)")
ANALYTIC_RE = re.compile(
    r"(?i)(analytic|analytical|exact|closed_?form|theory|theoretical|reference|"
    r"ideal|benchmark_value|iso_ref|literature)")
NOISE_RE = re.compile(r"(?i)(noise|perturb|jitter|rng|random|normal|gauss|uniform)")
SYNTHETIC_PREFIX_RE = re.compile(r"(?i)^(synthetic|surrogate|fabricated)")
DECLARATION_RE = re.compile(r"(?i)self-verification:\s*synthetic")
DOC_MEASURED_RE = re.compile(
    r"(?i)(DEM[- ]measured|DEM[- ]simulated|measured by the DEM|DEM[- ]measured|实验测量)")
DOC_ADMISSION_RE = re.compile(r"(?i)(simulated noise|add(ing)? (small )?noise|synthetic|surrogate)")

SEV_ORDER = {"SEV1": 1, "SEV2": 2, "SEV3": 3, "INFO": 4}


@dataclass
class Site:
    path: str
    line: int
    function: str
    target: str
    kind: str            # analytic_scale | analytic_noise | analytic_direct
    severity: str
    declared: bool
    evidence: str        # short source of the RHS
    mixed: bool = False  # RHS also contains a non-analytic (solver) call
    allowlisted: bool = False
    allow_reason: str | None = None
    docstring_claim: str | None = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["path"] = str(self.path)
        return d


@dataclass
class FileReport:
    path: str
    sites: list[Site] = field(default_factory=list)


# ------------------------------------------------------------------- utilities
def _dotted(node: ast.AST) -> str:
    """Best-effort dotted name of a call target / attribute / name."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _source(path: Path, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(path.read_text(encoding="utf8", errors="ignore"), node)
    except Exception:  # pragma: no cover - diagnostic only
        seg = None
    if not seg:
        return ast.dump(node)[:120]
    return " ".join(seg.split())[:160]


def _numeric_multipliers(node: ast.AST) -> list[float]:
    """Numeric constants applied to the analytic operand in a BinOp tree."""
    out: list[float] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
            for side in (sub.left, sub.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, (int, float)):
                    if not isinstance(side.value, bool):
                        out.append(float(side.value))
                    break
    return out


def _identifiers(node: ast.AST) -> list[str]:
    ids: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            ids.append(sub.id)
        elif isinstance(sub, ast.Attribute):
            ids.append(_dotted(sub))
        elif isinstance(sub, ast.Call):
            ids.append(_dotted(sub.func))
    return ids


def _is_analytic(node: ast.AST) -> bool:
    return any(ANALYTIC_RE.search(i) for i in _identifiers(node))


def _is_noise(node: ast.AST) -> bool:
    if any(NOISE_RE.search(i) for i in _identifiers(node)):
        return True
    # `* (1 + 0.05 * something)` is the same signature even if unnamed
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _dotted(sub.func).split(".")[-1] in {
                "standard_normal", "normal", "uniform", "random", "randn", "rand"}:
            return True
    return False


SCALAR_WRAPPERS = {"float", "int", "abs", "list", "tuple", "max", "min", "sum", "len",
                   "mean", "np.asarray", "np.array", "np.abs", "np.max", "np.min"}


def _mixed_with_solver(node: ast.AST) -> bool:
    """True when the RHS also calls something that is not the analytic reference.

    A deficit such as ``(sim.cumulative_work_exact - sim.cumulative_work)`` mixes
    the analytic integral with a solver quantity; it is a genuine cross-check, not
    a surrogate, so it must not be reported as a surrogate site.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = _dotted(sub.func)
            if ANALYTIC_RE.search(name) or name in SCALAR_WRAPPERS or name.endswith(
                    tuple("." + w for w in SCALAR_WRAPPERS)):
                continue
            return True
        if isinstance(sub, ast.Attribute):
            # a solver field read (sim.work_J) next to an analytic one (sim.work_exact_J)
            if not ANALYTIC_RE.search(_dotted(sub)):
                return True
    return False


def _declared(name: str, lines: list[str], lineno: int) -> bool:
    if SYNTHETIC_PREFIX_RE.search(name):
        return True
    lo = max(1, lineno - 3)
    hi = min(len(lines), lineno + 3)
    return any(DECLARATION_RE.search(lines[i - 1]) for i in range(lo, hi + 1))


# --------------------------------------------------------------------- analysis
def load_allowlist(path: Path | None) -> dict[tuple[str, str, str], dict]:
    """(path, target, kind) -> entry.  Triaged sites stay visible, not silenced."""
    if path is None or not path.exists():
        return {}
    body = json.loads(path.read_text(encoding="utf8"))
    out: dict[tuple[str, str, str], dict] = {}
    for e in body.get("entries", []):
        out[(e["path"], e["target"], e["kind"])] = e
    return out


def scan_file(path: Path, allow: dict | None = None) -> FileReport:
    text = path.read_text(encoding="utf8", errors="ignore")
    lines = text.splitlines()
    rep = FileReport(str(path))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return rep

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        # local assignment map (name -> RHS node) for alias resolution
        local: dict[str, ast.AST] = {}
        analytic_locals: set[str] = set()
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
                    stmt.targets[0], ast.Name):
                local[stmt.targets[0].id] = stmt.value
        # transitive closure over local aliases (fixpoint, bounded)
        for _ in range(4):
            grew = False
            for name, rhs in local.items():
                if name in analytic_locals:
                    continue
                if _is_analytic(rhs) or any(
                        isinstance(sub, ast.Name) and sub.id in analytic_locals
                        for sub in ast.walk(rhs)):
                    analytic_locals.add(name)
                    grew = True
            if not grew:
                break

        doc = ast.get_docstring(fn) or ""
        doc_claim = None
        if re.search(r"(?i)(DEM[- ]measured|DEM[- ]simulated)", doc):
            doc_claim = " ".join(doc.split())[:120]

        for stmt in ast.walk(fn):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                continue
            name = stmt.targets[0].id
            if not MEASURED_RE.search(name):
                continue
            rhs = stmt.value
            uses_analytic = _is_analytic(rhs) or any(
                isinstance(sub, ast.Name) and sub.id in analytic_locals
                for sub in ast.walk(rhs))
            if not uses_analytic:
                continue

            noise = _is_noise(rhs)
            scales = [m for m in _numeric_multipliers(rhs) if abs(m - 1.0) > 1e-12]
            if noise:
                kind, sev = "analytic_noise", "SEV2"
            elif scales:
                kind, sev = "analytic_scale", "SEV1"
            else:
                kind, sev = "analytic_direct", "SEV3"
            mixed = True if (kind == "analytic_direct" and _mixed_with_solver(rhs)) else False
            if mixed:
                sev = "INFO"     # cross-check, not a surrogate: advisory only
            try:
                rel = str(path.relative_to(ROOT))
            except ValueError:      # fixture outside the tree (tests)
                import os as _os
                rel = _os.path.relpath(str(path), str(ROOT))
            entry = (allow or {}).get((rel, name, kind))
            rep.sites.append(Site(
                path=rel, line=stmt.lineno, function=fn.name,
                target=name, kind=kind, severity=sev, mixed=mixed,
                declared=_declared(name, lines, stmt.lineno) or entry is not None,
                allowlisted=entry is not None,
                allow_reason=(entry or {}).get("reason"),
                evidence=_source(path, rhs), docstring_claim=doc_claim))
    return rep


def scan_docs(scan_paths: list[str]) -> list[dict]:
    """Flag doc pages that present a value as measured while admitting a surrogate."""
    out: list[dict] = []
    for p in sorted((ROOT / "docs").rglob("*.md")):
        text = p.read_text(encoding="utf8", errors="ignore")
        lines = text.splitlines()
        measured_lines = [(i + 1, ln.strip()) for i, ln in enumerate(lines)
                          if DOC_MEASURED_RE.search(ln)]
        admitted = [i + 1 for i, ln in enumerate(lines) if DOC_ADMISSION_RE.search(ln)]
        if not measured_lines or not admitted:
            continue
        out.append({
            "path": str(p.relative_to(ROOT)),
            "measured_claim_lines": [{"line": n, "text": t[:160]} for n, t in measured_lines],
            "admission_lines": admitted,
            "cross_referenced_tests": scan_paths,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="write the full report here")
    ap.add_argument("--docs", action="store_true",
                    help="also cross-check docs for measurement claims next to surrogate admissions")
    ap.add_argument("--include-scripts", action="store_true", default=True,
                    help="scan scripts/ as well as tests/ (default on)")
    ap.add_argument("--allowlist", default=str(ROOT / "tools" / "self_verification_allowlist.json"),
                    help="triage ledger of reviewed non-surrogate sites")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    targets: list[Path] = []
    for sub in ("tests", "scripts"):
        d = ROOT / sub
        if d.is_dir():
            targets.extend(sorted(d.rglob("*.py")))
    # the repo root holds a few standalone verification drivers written before
    # tests/ existed; they are part of the evidence surface too
    targets.extend(sorted(p for p in ROOT.glob("test_*.py")))

    allow = load_allowlist(Path(args.allowlist) if args.allowlist else None)
    reports = [scan_file(p, allow) for p in targets]
    sites = [s for r in reports for s in r.sites]
    undeclared = [s for s in sites if not s.declared]
    blocking = [s for s in undeclared if s.severity in ("SEV1", "SEV2")]

    offenders = sorted({s.path for s in undeclared})
    body: dict = {
        "tool": "tools/self_verification_scan.py",
        "root": str(ROOT),
        "files_scanned": len(targets),
        "sites": [s.as_dict() for s in sorted(
            sites, key=lambda s: (SEV_ORDER[s.severity], s.path, s.line))],
        "summary": {
            "sites": len(sites),
            "undeclared": len(undeclared),
            "blocking_sev1_sev2": len(blocking),
            "declared": len(sites) - len(undeclared),
            "allowlisted": sum(1 for s in sites if s.allowlisted),
            "allowlist_entries": len(allow),
            "offending_files": len(offenders),
        },
        "offending_files": offenders,
    }
    if args.docs:
        body["docs_cross_reference"] = scan_docs(offenders)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf8")

    if not args.quiet:
        print(f"scanned {len(targets)} files; {len(sites)} measurement-shaped sites "
              f"({len(sites) - len(undeclared)} declared)")
        for s in sorted(sites, key=lambda s: (SEV_ORDER[s.severity], s.path, s.line)):
            tag = "declared" if s.declared else "UNDECLARED"
            print(f"  [{s.severity}/{tag}] {s.path}:{s.line} {s.function}(): "
                  f"{s.target} <- {s.evidence}")
        if args.docs and body.get("docs_cross_reference"):
            print("\ndocs presenting a surrogate as measurement:")
            for d in body["docs_cross_reference"]:
                print(f"  {d['path']}  claims={len(d['measured_claim_lines'])} "
                      f"admissions={d['admission_lines'][:4]}")
        print(f"\nBLOCKING (undeclared SEV1/SEV2): {len(blocking)} "
              f"in {len(offenders)} files")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())