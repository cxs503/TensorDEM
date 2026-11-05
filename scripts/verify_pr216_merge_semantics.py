#!/usr/bin/env python3
"""Machine-decision of the PR #216 cure: was the merge of ``agent/m3558`` into
``main`` *lossless*?

The cure has one conflicting file, ``docs/doc_evidence_ledger.json``, which is a
summary of the whole ``docs/`` tree produced by
``scripts/audit_doc_evidence.py``.  Picking either side of that conflict silently
drops the other side's documents from the audit (a "take mine" resolution has
already bitten this repository once), so the cure regenerates it from the merged
worktree instead.  Regeneration is only acceptable if it is provably lossless,
hence this script: every claim about the cure is decided here, not asserted in
prose.

Checks
------
C1 file-union          every path added/modified on either parent is present in
                       the merged tree; when only one parent touched it the blob
                       is byte-identical to that parent's version; for the
                       both-sided path the merged file must be exactly what the
                       generator produces from the merged tree.
C2 pointer-union       no evidence pointer of either parent ledger disappears;
                       producer lists may grow but never shrink.
C3 count-monotonicity  every ``counts``/``remedy`` bucket and the document /
                       pointer scalars are >= both parents.
C4 conflict-markers    no git conflict marker survives in any tracked text file.
C5 source-anchors      for every ``src/`` file touched by a parent, the merged
                       file still defines every symbol that parent defined.

Usage::

    python scripts/verify_pr216_merge_semantics.py [--json OUT] [--figure OUT]

Exit code 0 iff every check passes.  The pinned SHAs are a *record* of this
specific cure, not a generic linter; reusing the script for another merge means
updating the constants below.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- the cure this script certifies -----------------------------------------
MERGE_BASE = "e12a2ab"    # main before PR #203..#216 wave: common ancestor
MAIN = "6c6f98b"          # main tip at cure time (Merge PR #166)
BRANCH = "d7c946c"        # agent/m3558 tip before cure (PR #216 head)
MERGE = "65294fe"         # cure merge commit (parents: MAIN, BRANCH)
BOTH_SIDED = {"docs/doc_evidence_ledger.json"}  # the regenerated artefact

TEXT_EXT = {".md", ".py", ".json", ".toml", ".cfg", ".txt", ".yml", ".yaml", ".csv"}
CONFLICT_RE = re.compile(r"^(<{7} |={7}$|>{7} )", re.MULTILINE)
SYMBOL_RE = re.compile(
    r"^(?:\s*)(?:def|class)\s+([A-Za-z_]\w*)|^([A-Z][A-Z0-9_]{2,})\s*=", re.MULTILINE
)


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def have(ref: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{ref}^{{commit}}"], cwd=ROOT, capture_output=True
    ).returncode == 0


def blob(ref: str, path: str) -> str | None:
    """Object id of ``path`` in ``ref`` (None when the path does not exist there)."""
    out = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or None


def changed(base: str, ref: str) -> dict[str, str]:
    rows = git("diff", "--name-status", base, ref).splitlines()
    return {line.split("\t", 1)[1]: line.split("\t", 1)[0] for line in rows}


def ledger_of(ref: str) -> dict:
    return json.loads(git("show", f"{ref}:docs/doc_evidence_ledger.json"))


def ledger_at_head() -> dict:
    return json.loads((ROOT / "docs" / "doc_evidence_ledger.json").read_text(encoding="utf-8"))


def key(entry: dict) -> tuple:
    return (entry["doc"], entry["pointer"], entry["status"])


def symbols(text: str) -> set[str]:
    return {a or b for a, b in SYMBOL_RE.findall(text)}


def fresh_scan() -> dict:
    """What the generator produces from *this* worktree."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_doc_evidence.py"), "--json", "-"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


# --- checks -----------------------------------------------------------------
def c1_file_union() -> dict:
    ours, theirs = changed(MERGE_BASE, MAIN), changed(MERGE_BASE, BRANCH)
    union = sorted(set(ours) | set(theirs))
    both = sorted(set(ours) & set(theirs))
    detail, failures = [], []
    for path in union:
        merged_blob = blob(MERGE, path)
        if merged_blob is None:
            failures.append(f"{path}: absent from the merge")
            continue
        if path in both:
            detail.append({"path": path, "resolution": "both-sided"})
            continue
        ref, side = (MAIN, "main") if path in ours else (BRANCH, "branch")
        same = merged_blob == blob(ref, path)
        detail.append({"path": path, "resolution": side, "identical": same})
        if not same:
            failures.append(f"{path}: differs from its only parent ({side})")
    # the both-sided artefact must be a pure function of the merged tree
    committed, fresh = ledger_at_head(), fresh_scan()
    regenerable = (
        committed["entries"] == fresh["entries"]
        and committed["counts"] == fresh["counts"]
        and committed["remedy"] == fresh["remedy"]
        and committed["documents_scanned"] == fresh["documents_scanned"]
    )
    if not regenerable:
        failures.append("docs/doc_evidence_ledger.json is not reproducible from the merged tree")
    return {
        "ok": not failures,
        "paths_main": len(ours),
        "paths_branch": len(theirs),
        "paths_union": len(union),
        "both_sided": both,
        "single_sided_identical": sum(1 for d in detail if d.get("identical")),
        "single_sided_total": sum(1 for d in detail if "identical" in d),
        "regenerated_artefact_reproducible": regenerable,
        "files": detail,
        "failures": failures,
    }


def c2_pointer_union() -> dict:
    merged = {key(e): e for e in ledger_at_head()["entries"]}
    report, failures, enriched = {}, [], 0
    for name, ref in (("main", MAIN), ("branch", BRANCH)):
        entries = ledger_of(ref)["entries"]
        lost = [key(e) for e in entries if key(e) not in merged]
        if lost:
            failures.extend(f"main-lost {k}" if name == "main" else f"branch-lost {k}" for k in lost[:10])
        for e in entries:
            m = merged.get(key(e))
            if m is None:
                continue
            if not set(e.get("producers") or []) <= set(m.get("producers") or []):
                failures.append(f"producers shrank for {key(e)}")
            elif len(m.get("producers") or []) > len(e.get("producers") or []):
                enriched += 1
        report[name] = {"pointers": len(entries), "lost": len(lost)}
    report.update({"ok": not failures, "merged_pointers": len(merged),
                   "enriched": enriched, "failures": failures})
    return report


def c3_count_monotonicity() -> dict:
    """Every bucket of the regenerated ledger must dominate both parents."""
    merged = ledger_at_head()
    parents = {name: ledger_of(ref) for name, ref in (("main", MAIN), ("branch", BRANCH))}
    rows, failures = [], []

    def compare(label: str, value: int, getter) -> None:
        side = {name: getter(led) for name, led in parents.items()}
        ok = value >= max(side.values())
        rows.append({"bucket": label, "main": side["main"], "branch": side["branch"],
                     "merged": value, "ok": ok})
        if not ok:
            failures.append(f"{label}: merged {value} < max({side['main']}, {side['branch']})")

    compare("documents_scanned", merged["documents_scanned"], lambda d: d["documents_scanned"])
    compare("pointers", merged["pointers"], lambda d: d["pointers"])
    for field in ("counts", "remedy"):
        for bucket, value in merged[field].items():
            compare(f"{field}.{bucket}", value, lambda d, f=field, b=bucket: d[f].get(b, 0))
    return {"ok": not failures, "rows": rows, "failures": failures}


def c4_conflict_markers() -> dict:
    hits = []
    for path in git("ls-files").splitlines():
        if Path(path).suffix not in TEXT_EXT:
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if CONFLICT_RE.search(text):
            hits.append(path)
    return {"ok": not hits, "hits": hits, "scanned": "tracked text files"}


def c5_source_anchors() -> dict:
    ours, theirs = changed(MERGE_BASE, MAIN), changed(MERGE_BASE, BRANCH)
    paths = sorted(p for p in set(ours) | set(theirs) if p.startswith("src/"))
    rows, failures = [], []
    for path in paths:
        local = ROOT / path
        if not local.exists():
            failures.append(f"{path}: missing from the merge")
            continue
        merged_syms = symbols(local.read_text(encoding="utf-8"))
        for ref, name in ((MAIN, "main"), (BRANCH, "branch")):
            if path not in (ours if name == "main" else theirs):
                continue
            text = git("show", f"{ref}:{path}")
            lost = symbols(text) - merged_syms
            rows.append({"path": path, "side": name, "lost_symbols": sorted(lost)})
            if lost:
                failures.append(f"{path}: lost {sorted(lost)} from {name}")
    return {"ok": not failures, "files": rows, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="write the decision record here")
    args = ap.parse_args(argv)

    missing = [r for r in (MERGE_BASE, MAIN, BRANCH, MERGE) if not have(r)]
    if missing:
        print(f"history unavailable (shallow clone?), cannot verify: {missing}", file=sys.stderr)
        return 2

    checks = {
        "C1 file-union": c1_file_union(),
        "C2 pointer-union": c2_pointer_union(),
        "C3 count-monotonicity": c3_count_monotonicity(),
        "C4 conflict-markers": c4_conflict_markers(),
        "C5 source-anchors": c5_source_anchors(),
    }
    overall = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    record = {
        "schema": "tensordem.pr216-cure-verification/1",
        "shas": {"merge_base": MERGE_BASE, "main": MAIN, "branch": BRANCH, "merge": MERGE},
        "overall": overall,
        "checks": checks,
    }
    for name, c in checks.items():
        print(f"{name:24s} {'ok' if c['ok'] else 'FAIL'}")
        for f in c.get("failures", []):
            print(f"   - {f}")
    print(f"overall                  {overall}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())