#!/usr/bin/env python3
"""Audit the evidence pointers that TensorDEM documentation makes.

A "pointer" is any repo-relative path with a file extension that a Markdown
document under ``docs/`` refers to, either as a Markdown link
(``[text](artifacts/threefield/wake_lbm.json)``) or inline as a bare token
(``scripts/run_ice_ridge_grounding.py``, ``tests/test_ridge_grounding.py::...``).
Every published number in this repository is supposed to be traceable to such a
file, so a pointer that does not resolve is a broken evidence chain.

Resolution order for a token (Markdown semantics first, repo semantics second):

1. the token relative to the repository root,
2. the token relative to the directory of the document that cites it,
3. the absolute path if it starts with ``/`` or ``~`` (recorded, never resolved).

Classification of a pointer:

===============  ==========================================================
TRACKED          git knows the file (a fresh clone has it)
UNTRACKED        on disk but git-ignored/untracked (invisible in a clone)
EXTERNAL         absolute path or URL, outside the repository
MISSING          absent from the worktree under every candidate path
===============  ==========================================================

For non-TRACKED pointers the audit also records whether some file in
``scripts/``, ``benchmarks/``, ``tests/`` or ``src/`` mentions the same
path/basename: REGENERABLE (a producer exists, the claim can be rebuilt) or
ORPHAN (nothing in the tree can recreate it, so the claim behind it is
unverifiable as shipped).

Usage::

    python scripts/audit_doc_evidence.py             # print + write ledger
    python scripts/audit_doc_evidence.py --check     # exit 1 if ledger is stale
    python scripts/audit_doc_evidence.py --json -    # print ledger to stdout

The ledger is ``docs/doc_evidence_ledger.json``; the human-readable discussion
is ``docs/doc_evidence_pointer_audit.md``.
"""
from __future__ import annotations

import argparse
import json
import difflib
import posixpath
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
LEDGER = DOCS / "doc_evidence_ledger.json"
SCHEMA = "tensordem.doc-evidence-ledger/1"

EXT = r"(?:json|png|csv|md|py|npz|npy|txt|pdf|svg|ya?ml|toml|log)"
# Inline token: at least one '/', a known extension, bounded by non-path chars.
POINTER_RE = re.compile(
    r"(?<![\w/.~-])((?:\.[\w.+-]*/)?[\w.+-]+(?:/[\w.+-]+)+\." + EXT + r")(?![\w])"
)
# Markdown link target: [text](target) and [text](target#anchor).
LINK_RE = re.compile(r"\]\(\s*<?([^)>\s]+)>?\s*\)")
TESTNODE_RE = re.compile(r"(tests/[\w./+-]+\.py(?:::[A-Za-z_][\w:]*)*)")

PRODUCER_DIRS = ("scripts", "benchmarks", "tests", "src")
# An audit that quotes a broken pointer to explain its repair must be able to do so
# without re-introducing the pointer. Quotes go between these two markers, and only
# the audit documents may use them (enforced in tests/test_doc_evidence_pointers.py).
HIST_START = "<!-- audit:historical -->"
HIST_END = "<!-- /audit:historical -->"


def git_tracked(root: Path = ROOT) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def skip(token: str) -> bool:
    if token.startswith(("http://", "https://", "ftp://", "mailto:")):
        return True
    if any(ch in token for ch in "<>{}*?$ "):
        return True
    return "..." in token


def kind_of(token: str) -> str:
    suffix = token.rsplit(".", 1)[-1].lower()
    if suffix == "py":
        return "code"
    if suffix == "md":
        return "doc"
    if suffix in {"png", "svg", "pdf"}:
        return "figure"
    return "data"


def producer_index(root: Path = ROOT) -> dict[str, list[str]]:
    """Map basename -> sorted list of tree files that mention the basename."""
    index: dict[str, set[str]] = defaultdict(set)
    for sub in PRODUCER_DIRS:
        for path in sorted((root / sub).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:  # pragma: no cover - unreadable file
                continue
            for name in set(re.findall(r"[\w.+-]+\.[A-Za-z0-9]+", text)):
                index[name].add(rel)
    return {k: sorted(v) for k, v in index.items()}


def candidates(token: str, doc: str) -> list[str]:
    """Repo-relative candidate paths for a pointer, Markdown order last."""
    out = []
    token = token.split("#", 1)[0].split("::", 1)[0]
    if not token:
        return out
    if token.startswith(("/", "~")):
        return [token]
    root_rel = posixpath.normpath(token)
    out.append(root_rel)
    # a Markdown link relative to docs/ writes "../artifacts/x"; the same file is
    # cited as "artifacts/x" elsewhere, so try the repo-rooted reading too.
    stripped = root_rel
    while stripped.startswith("../"):
        stripped = stripped[3:]
    if stripped not in out:
        out.append(stripped)
    doc_rel = posixpath.normpath(posixpath.join(posixpath.dirname(doc), token))
    if doc_rel not in out:
        out.append(doc_rel)
    return out


def suggest(pointer: str, tracked: set[str]) -> list[str]:
    """Propose existing repository paths for a dangling pointer."""
    base = posixpath.basename(pointer)
    hits = sorted(p for p in tracked if posixpath.basename(p) == base)
    if hits:
        return hits[:3]
    parts = pointer.split("/")
    if len(parts) >= 2:
        tail = "/".join(parts[-2:])
        hits = sorted(p for p in tracked if p.endswith("/" + tail))
        if hits:
            return hits[:3]
    pool = [p for p in tracked if p.startswith(parts[0] + "/")] or sorted(tracked)
    names = {posixpath.basename(p): p for p in pool}
    return [names[m] for m in difflib.get_close_matches(base, list(names), n=3, cutoff=0.8)]


def classify(
    token: str, doc: str, tracked: set[str], root: Path = ROOT
) -> tuple[str, str | None]:
    cands = candidates(token, doc)
    if not cands:
        return "MISSING", None
    if cands[0].startswith(("/", "~")):
        return "EXTERNAL", cands[0]
    for cand in cands:
        if cand in tracked:
            return "TRACKED", cand
    for cand in cands:
        if (root / cand).exists():
            return "UNTRACKED", cand
    fallback = next((c for c in cands if not c.startswith("..")), cands[0])
    return "MISSING", fallback


def scan(
    root: Path = ROOT,
    docs_dir: Path | None = None,
    tracked: set[str] | None = None,
    producers: dict[str, list[str]] | None = None,
) -> dict:
    root = Path(root)
    docs_dir = (root / "docs") if docs_dir is None else Path(docs_dir)
    tracked = git_tracked(root) if tracked is None else tracked
    producers = producer_index(root) if producers is None else producers
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    docs = sorted(docs_dir.rglob("*.md"))
    historical: dict[str, int] = defaultdict(int)
    for doc in docs:
        rel_doc = doc.relative_to(root).as_posix()
        in_historical = False
        for lineno, line in enumerate(doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped == HIST_START:
                in_historical = True
                continue
            if stripped == HIST_END:
                in_historical = False
                continue
            if in_historical:
                historical[rel_doc] += 1
                continue
            tokens = [tok for tok in POINTER_RE.findall(line)]
            tokens += [tok for tok in LINK_RE.findall(line) if POINTER_RE.fullmatch(tok.split("#", 1)[0])]
            tokens += TESTNODE_RE.findall(line)
            for token in tokens:
                if skip(token):
                    continue
                key = (rel_doc, token)
                if key in seen:
                    continue
                seen.add(key)
                is_test_node = bool(TESTNODE_RE.fullmatch(token))
                status, resolved = classify(token, rel_doc, tracked, root)
                kind = "test" if is_test_node else kind_of(token)
                if is_test_node and status == "TRACKED":
                    path, _, node = token.partition("::")
                    leaf = node.split("::")[-1] if node else ""
                    if leaf and leaf not in (root / path).read_text(encoding="utf-8", errors="replace"):
                        status = "MISSING"  # name referenced but absent from the file
                entry = {
                    "doc": rel_doc,
                    "line": lineno,
                    "pointer": token,
                    "kind": kind,
                    "status": status,
                }
                if resolved and resolved != posixpath.normpath(token.split("#", 1)[0]):
                    entry["resolved"] = resolved
                if status not in {"TRACKED", "EXTERNAL"}:
                    base = posixpath.basename(resolved or token)
                    cand = [p for p in producers.get(base, []) if p != rel_doc]
                    entry["producers"] = cand
                    entry["remedy"] = "REGENERABLE" if cand else "ORPHAN"
                    if not cand:
                        entry["suggested"] = suggest(resolved or token, tracked)
                entries.append(entry)
    entries.sort(key=lambda e: (e["doc"], e["line"], e["pointer"]))
    counts = Counter(f"{e['kind']}:{e['status']}" for e in entries)
    remedy = Counter(e.get("remedy", "TRACKED") for e in entries)
    per_doc = Counter(e["doc"] for e in entries if e["status"] != "TRACKED")
    return {
        "schema": SCHEMA,
        "generated_by": "scripts/audit_doc_evidence.py",
        "documents_scanned": len(docs),
        "pointers": len(entries),
        "counts": dict(sorted(counts.items())),
        "remedy": dict(sorted(remedy.items())),
        "unresolved_per_doc": dict(sorted(per_doc.items())),
        "historical_quotes": dict(sorted(historical.items())),
        "entries": entries,
    }


def figures(report: dict, out_dir: Path) -> list[Path]:
    """Render the audit summary figures (matplotlib, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    kinds = ("code", "doc", "test", "data", "figure")
    tracked = [sum(1 for e in report["entries"] if e["kind"] == k and e["status"] == "TRACKED") for k in kinds]
    absent = [
        sum(1 for e in report["entries"] if e["kind"] == k and e["status"] not in ("TRACKED", "EXTERNAL"))
        for k in kinds
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(kinds, tracked, label="tracked by git (in a clone)", color="#2a7f62")
    ax.bar(kinds, absent, bottom=tracked, label="absent from clone (rebuildable)", color="#c46a3a")
    for i, (t, a) in enumerate(zip(tracked, absent)):
        ax.text(i, t + a + 2, f"{t}+{a}", ha="center", fontsize=8)
    ax.set_ylabel("pointers")
    ax.set_title("Cited evidence pointers by kind and resolution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "doc_evidence_by_kind.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    roots: dict[str, int] = {}
    for e in report["entries"]:
        if e["status"] in ("TRACKED", "EXTERNAL"):
            continue
        root = e["pointer"].split("/")[0]
        roots[root] = roots.get(root, 0) + 1
    items = sorted(roots.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4a6fa5")
    ax.set_xlabel("cited artifacts not in the clone")
    ax.set_title("Where the uncommitted evidence lives")
    fig.tight_layout()
    path = out_dir / "doc_evidence_unresolved_roots.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def write_ledger(report: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def summary(report: dict) -> str:
    lines = [
        f"documents scanned : {report['documents_scanned']}",
        f"pointers          : {report['pointers']}",
        f"status counts     : {report['counts']}",
        f"remedy            : {report['remedy']}",
    ]
    unresolved = [e for e in report["entries"] if e["status"] != "TRACKED"]
    orphans = [e for e in unresolved if e.get("remedy") == "ORPHAN"]
    lines.append(f"unresolved        : {len(unresolved)} ({len(orphans)} orphan)")
    for e in orphans:
        lines.append(f"  ORPHAN {e['doc']}:{e['line']} {e['pointer']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="fail if the committed ledger is stale")
    ap.add_argument("--json", default=None, help="write the ledger here ('-' = stdout)")
    ap.add_argument("--figures", default=None, help="directory for the summary figures")
    args = ap.parse_args(argv)

    report = scan()
    if args.json == "-":
        print(json.dumps(report, indent=1))
        return 0
    if args.json:
        write_ledger(report, Path(args.json))
    if args.check:
        if not LEDGER.exists():
            print(f"ledger missing: {LEDGER}", file=sys.stderr)
            return 1
        committed = json.loads(LEDGER.read_text(encoding="utf-8"))
        if committed.get("schema") != SCHEMA:
            print(f"ledger schema {committed.get('schema')!r} != {SCHEMA!r}", file=sys.stderr)
            return 1
        key = lambda e: (e["doc"], e["pointer"], e["status"])  # noqa: E731
        old = sorted(map(key, committed.get("entries", [])))
        new = sorted(map(key, report["entries"]))
        if old != new:
            old_set, new_set = set(old), set(new)
            print("ledger is stale; re-run scripts/audit_doc_evidence.py", file=sys.stderr)
            for k in [k for k in new if k not in old_set][:10]:
                print(f"  + {k}", file=sys.stderr)
            for k in [k for k in old if k not in new_set][:10]:
                print(f"  - {k}", file=sys.stderr)
            return 1
        print("ledger up to date: " + report["counts"].__repr__())
        return 0
    write_ledger(report)
    if args.figures:
        for path in figures(report, Path(args.figures)):
            print(f"figure: {path}")
    print(summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())