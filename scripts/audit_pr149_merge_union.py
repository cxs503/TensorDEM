#!/usr/bin/env python3
"""Audit a three-way merge for silent content loss (PR #149 cure, mission m3863).

A merge is only trustworthy if *both* sides' intents survive it.  Textual
``git merge`` can drop content silently when a resolution is done by mechanical
overlay (a documented failure mode of this repository), so this script proves
the property explicitly instead of asserting it:

for every path touched by either side since the fork point it checks

1. **addition preservation** -- every line a side added relative to the fork
   point is present in the merged tree (at the same path, or, when the
   resolution deliberately relocated an artifact namespace, at another path in
   the merged tree, reported separately);
2. **deletion preservation** -- every line a side deleted relative to the fork
   point does not reappear in the merged tree (a revival is reported for human
   review instead of being treated as acceptable);
3. **blob preservation** -- the exact side blob (including binaries) of a
   conflicted path exists somewhere in the merged tree, i.e. neither side's
   artifact was replaced by the other side's version;
4. **structural sanity** -- every merged Python file parses and defines no
   duplicated top-level symbol (``def``/``class``) name.

Usage::

    python scripts/audit_pr149_merge_union.py \
        --base 8cc88f7 --main e12a2ab --branch d85d8ac --merged 985d982 \
        --out runs/pr149_union_audit.json

Exit code 0 iff no silent loss was detected (relocations and review items are
reported but do not fail the gate; ``LOST`` lines do).
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import pathlib
import subprocess
import sys

BINARY_SNIFF = 8000


def git(*args: str) -> bytes:
    """Run git in the current repository and return raw stdout bytes."""
    return subprocess.run(("git",) + args, check=True, capture_output=True).stdout


def tree_paths(rev: str) -> dict[str, str]:
    """path -> blob sha for every blob in ``rev``."""
    out = {}
    for line in git("ls-tree", "-r", "--full-tree", rev).decode().splitlines():
        meta, path = line.split("\t", 1)
        _mode, kind, sha = meta.split()
        if kind == "blob":
            out[path] = sha
    return out


def blob_text(sha: str) -> str | None:
    raw = git("cat-file", "blob", sha)
    if b"\0" in raw[:BINARY_SNIFF]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers(text: str) -> collections.Counter:
    """Numeric literals of a text block -- a proxy for 'the measured values'."""
    return collections.Counter(NUMBER.findall(text))


def top_symbols(text: str) -> set[str]:
    """Top-level ``def``/``class`` names of a Python source block."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


def side_count(path: str, side: dict[str, str], line: str) -> int:
    sha = side.get(path)
    if sha is None:
        return 0
    text = blob_text(sha)
    if text is None:
        return 0
    return collections.Counter(text.splitlines())[line]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", default="runs/pr149_union_audit.json")
    ap.add_argument(
        "--accept",
        default=None,
        help="JSON file of human-reviewed line-level exceptions "
        "(path -> {reason, lines}). Without it every reflowed line counts as loss.",
    )
    args = ap.parse_args()

    base = tree_paths(args.base)
    main = tree_paths(args.main)
    branch = tree_paths(args.branch)
    merged = tree_paths(args.merged)

    merged_line_index: set[str] = set()
    merged_text: dict[str, str] = {}
    global_symbols: set[str] = set()
    global_numbers: collections.Counter = collections.Counter()
    for path, sha in merged.items():
        text = blob_text(sha)
        if text is None:
            continue
        merged_text[path] = text
        merged_line_index.update(text.splitlines())
        if path.endswith(".py"):
            global_symbols |= top_symbols(text)
        global_numbers |= numbers(text)

    def side_changes(side: dict[str, str]) -> dict[str, list[str]]:
        return {p for p in set(base) | set(side) if base.get(p) != side.get(p)}

    touched = side_changes(main) | side_changes(branch)
    conflicted = side_changes(main) & side_changes(branch)

    report: dict[str, object] = {
        "base": args.base,
        "main": args.main,
        "branch": args.branch,
        "merged_tree": args.merged,
        "counts": {"touched": len(touched), "conflict_surface": len(conflicted)},
        "files": {},
        "verdict": {},
    }
    accepted: dict[str, dict] = {}
    if args.accept:
        acc = json.loads(pathlib.Path(args.accept).read_text())
        accepted = {p: {"reason": v["reason"], "lines": set(v["lines"])} for p, v in acc.items()}

    files: dict[str, dict] = {}
    lost_total = 0
    unreviewed_lost = 0
    reviewed_total = 0
    revived_total = 0
    semantic_missing = [0]
    semantic_api_total = [0]
    semantic_numbers_total = [0]

    for path in sorted(touched):
        entry: dict[str, object] = {"conflict_surface": path in conflicted}
        b_sha = base.get(path)
        b_text = blob_text(b_sha) if b_sha else None
        base_counter = collections.Counter(b_text.splitlines()) if b_text is not None else None

        side_blob_ok: dict[str, object] = {}
        for label, side in (("main", main), ("branch", branch)):
            s_sha = side.get(path)
            if s_sha is None:
                side_blob_ok[label] = "absent_on_this_side"
                continue
            # the side's own blob must survive somewhere in the merged tree
            side_blob_ok[label] = "preserved" if s_sha in merged.values() else "REPLACED"
        entry["blob_preservation"] = side_blob_ok

        additions: dict[str, object] = {}
        deletions: dict[str, object] = {}
        for label, side in (("main", main), ("branch", branch)):
            if side.get(path) == base.get(path):
                continue  # untouched by this side
            s_sha = side.get(path)
            s_text = blob_text(s_sha) if s_sha else None
            if base_counter is None or s_text is None:
                additions[label] = {"mode": "binary_or_added", "checked_by": "blob_preservation"}
                continue
            added = collections.Counter(s_text.splitlines()) - base_counter
            removed = base_counter - collections.Counter(s_text.splitlines())

            merged_text_at = blob_text(merged[path]) if path in merged else None
            at_path = collections.Counter(merged_text_at.splitlines()) if merged_text_at else collections.Counter()

            missing_at_path = {
                line: n - at_path[line]
                for line, n in added.items()
                if at_path[line] < n
            }
            lost, relocated, reviewed = {}, {}, {}
            acc = accepted.get(path, {"lines": set(), "reason": ""})
            for line, n in missing_at_path.items():
                if line in merged_line_index:
                    relocated[line] = n
                elif line in acc["lines"]:
                    reviewed[line] = n
                else:
                    lost[line] = n
            revived = {}
            for line, n in removed.items():
                # a line counts as revived only if the merged tree holds more
                # copies than *either* side did (otherwise it simply lives on in
                # another function/hunk of the same file)
                ceiling = max(side_count(path, main, line), side_count(path, branch, line))
                if at_path[line] > ceiling:
                    revived[line] = at_path[line] - ceiling

            additions[label] = {
                "added_lines": sum(added.values()),
                "present_at_path": sum(added.values()) - sum(missing_at_path.values()),
                "relocated_elsewhere": relocated,
                "reviewed_exception": reviewed,
                "review_reason": acc["reason"] if reviewed else "",
                "LOST": lost,
            }
            deletions[label] = {"deleted_lines": sum(removed.values()), "revived_in_merge": revived}
            lost_total += sum(lost.values())
            unreviewed_lost += sum(lost.values())
            reviewed_total += sum(reviewed.values())
            revived_total += sum(revived.values())
        # semantic proxies: measured numbers and public Python API must survive
        # even when a resolution reflows prose or relocates a whole artifact
        semantic: dict[str, dict] = {}
        scoped_text = merged_text.get(path)
        scoped_symbols = top_symbols(scoped_text) if scoped_text and path.endswith(".py") else set()
        scoped_numbers = numbers(scoped_text) if scoped_text else collections.Counter()
        for label, side in (("main", main), ("branch", branch)):
            if side.get(path) == base.get(path):
                continue
            s_sha = side.get(path)
            s_text = blob_text(s_sha) if s_sha else None
            if s_text is None or base.get(path) is None:
                continue  # pure binary/blob-level case, covered above
            base_numbers = numbers(base_counter and blob_text(base[path]) or "")
            side_numbers = numbers(s_text) - base_numbers
            missing_numbers = sorted(n for n in side_numbers if scoped_numbers[n] < side_numbers[n])
            missing_numbers = [n for n in missing_numbers if global_numbers[n] < side_numbers[n]]
            info = {"numbers_added": sum(side_numbers.values()), "numbers_missing": missing_numbers}
            if path.endswith(".py"):
                api = top_symbols(s_text) - top_symbols(blob_text(base[path]) or "")
                info["api_added"] = sorted(api)
                info["api_missing"] = sorted(api - scoped_symbols - global_symbols)
                semantic_api_total[0] += len(api)
                semantic_missing[0] += len(info["api_missing"])
            semantic_numbers_total[0] += sum(side_numbers.values())
            semantic_missing[0] += len(missing_numbers)
            semantic[label] = info
        if semantic:
            entry["semantic"] = semantic

        if additions:
            entry["additions"] = additions
        if deletions:
            entry["deletions"] = deletions
        files[path] = entry

    # structural sanity of the merged Python files
    py = {"files": 0, "parse_errors": [], "duplicate_defs": {}}
    for path, text in merged_text.items():
        if not path.endswith(".py"):
            continue
        py["files"] += 1
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError as exc:  # pragma: no cover - reported, not raised
            py["parse_errors"].append(f"{path}: {exc}")
            continue
        seen: dict[str, int] = collections.Counter()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen[node.name] += 1
        dups = {name: n for name, n in seen.items() if n > 1}
        if dups:
            py["duplicate_defs"][path] = dups

    report["files"] = files
    report["python_sanity"] = py
    report["verdict"] = {
        "lost_lines": lost_total,
        "revived_lines_review": revived_total,
        "parse_errors": len(py["parse_errors"]),
        "duplicate_defs": sum(len(v) for v in py["duplicate_defs"].values()),
        "semantic_missing_tokens": semantic_missing[0],
        "semantic_tokens_checked": semantic_api_total[0] + semantic_numbers_total[0],
        "line_level_reviewed_exceptions": reviewed_total,
        "line_level_unreviewed_loss": unreviewed_lost,
        "silent_loss_detected": unreviewed_lost > 0 or semantic_missing[0] > 0,
        "ok": unreviewed_lost == 0 and semantic_missing[0] == 0
        and not py["parse_errors"] and not py["duplicate_defs"],
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for path, entry in files.items():
        if not entry["conflict_surface"]:
            continue
        print(f"[conflict] {path}")
        for label, info in entry.get("additions", {}).items():
            if "LOST" in info:
                print(
                    f"    {label}: added={info['added_lines']} at_path={info['present_at_path']}"
                    f" relocated={len(info['relocated_elsewhere'])} LOST={len(info['LOST'])}"
                )
        print(f"    blobs: {entry['blob_preservation']}")
        for label, info in entry.get("semantic", {}).items():
            print(
                f"    {label}: numbers={info['numbers_added']} "
                f"missing={info['numbers_missing']} api_missing={info.get('api_missing', [])}"
            )
    print(json.dumps(report["verdict"], indent=2))
    return 0 if report["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())