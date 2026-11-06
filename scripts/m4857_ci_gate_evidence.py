"""Programmatic evidence for the m4857 repair: main's evidence-ledger red.

The incident, stated as a claim that a script can check:

``docs/ci_clearance_m4829.md`` cites three files under ``artifacts/m4829_ci_gate/``
and ``docs/doc_evidence_ledger.json`` records all three as ``TRACKED`` (i.e. "git
carries this blob, so the citation resolves in a fresh clone").  The merge commit
that landed that document on ``main`` is single-parent, and the three blobs never
arrived, so on ``main`` the same gate grades them ``MISSING`` and CI goes red.

This script re-derives, from git alone, (a) what the main-branch ledger claims,
(b) what ``git ls-tree`` / ``git ls-files`` say about the blobs at the merge
commit and at the repair commit, and (c) the exit code of the gate on both trees.
Nothing is typed by hand into the numbers below.

Usage::

    python scripts/m4857_ci_gate_evidence.py      # writes artifacts/ + figures
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "m4857_ci_gate"

MERGE_COMMIT = "cb6c562"  # Merge #346 (m4829) via merge-master -- the red main
REPAIR_COMMIT = "HEAD"

POINTERS = (
    "artifacts/m4829_ci_gate/gate_evidence.json",
    "artifacts/m4829_ci_gate/provenance_gate_matrix.png",
    "artifacts/m4829_ci_gate/ledger_regeneration.png",
)
DOC = "docs/ci_clearance_m4829.md"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def _tree_has(path: str, rev: str) -> bool:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", rev, "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return bool(out)


def _ledger_at(rev: str) -> dict:
    text = _git("show", f"{rev}:docs/doc_evidence_ledger.json")
    return json.loads(text)


def _ledger_verdicts(ledger: dict) -> dict[str, str]:
    verdicts = {}
    for entry in ledger.get("entries", []):
        if entry["doc"] == DOC and entry["pointer"] in POINTERS:
            verdicts[entry["pointer"]] = entry["status"]
    return verdicts


def _tracked_at(rev: str) -> set[str]:
    return {line for line in _git("ls-tree", "-r", "--name-only", rev).splitlines() if line}


def _gate_exit(rev: str) -> int:
    """Exit code of the ledger gate as a fresh clone of ``rev`` would see it.

    Two things are pinned to the revision: the ledger ``--check`` compares
    against, and the set of tracked paths (``git ls-tree``). Without pinning the
    index too, the worktree's restored blobs would mask the regression -- the
    stale ledger would look current, which is exactly the false green this
    script exists to rule out.
    """
    import contextlib
    import io
    import tempfile
    import unittest.mock

    sys.path.insert(0, str(ROOT / "scripts"))
    import audit_doc_evidence as audit

    tracked = _tracked_at(rev)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "ledger.json"
        ledger.write_text(json.dumps(_ledger_at(rev), indent=1) + "\n", encoding="utf-8")
        with unittest.mock.patch.object(audit, "LEDGER", ledger), unittest.mock.patch.object(
            audit, "git_tracked", lambda *a, **k: tracked
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                return audit.main(["--check"]), err.getvalue()


def _gate_row(rev: str) -> dict:
    code, message = _gate_exit(rev)
    return {
        "exit": code,
        "names_the_lost_blobs": "recorded as committed but gone from the checkout" in message,
    }


def collect() -> dict:
    ledger_main = _ledger_at(MERGE_COMMIT)
    verdicts = _ledger_verdicts(ledger_main)
    rows = []
    for pointer in POINTERS:
        rows.append(
            {
                "pointer": pointer,
                "ledger_verdict_on_main": verdicts.get(pointer, "ABSENT_FROM_LEDGER"),
                "blob_at_merge_commit": _tree_has(pointer, MERGE_COMMIT),
                "blob_at_repair_commit": _tree_has(pointer, REPAIR_COMMIT),
                "file_present_in_worktree": (ROOT / pointer).is_file(),
            }
        )
    merge_is_single_parent = len(_git("rev-list", "--parents", "-n", "1", MERGE_COMMIT).split()) == 2
    return {
        "schema": "tensordem.m4857-ci-gate-evidence/1",
        "generated_by": "scripts/m4857_ci_gate_evidence.py",
        "merge_commit": _git("rev-parse", "--short", MERGE_COMMIT).strip(),
        "merge_commit_is_single_parent": merge_is_single_parent,
        "repair_commit": _git("rev-parse", "--short", REPAIR_COMMIT).strip(),
        "document": DOC,
        "documents_identical_between_commits": _git(
            "diff", "--name-only", MERGE_COMMIT, REPAIR_COMMIT, "--", "docs"
        ).strip() == "",
        "rows": rows,
        "gate": {
            "merge_commit": _gate_row(MERGE_COMMIT),
            "repair_commit": _gate_row(REPAIR_COMMIT),
        },
        "blobs_missing_on_main": sum(1 for r in rows if not r["blob_at_merge_commit"]),
        "blobs_recorded_as_tracked_on_main": sum(
            1 for r in rows if r["ledger_verdict_on_main"] == "TRACKED"
        ),
        "blobs_restored": sum(
            1 for r in rows if r["blob_at_repair_commit"] and not r["blob_at_merge_commit"]
        ),
    }


def figure(evidence: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = evidence["rows"]
    labels = [r["pointer"].rsplit("/", 1)[-1] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    for i, row in enumerate(rows):
        claim = 1.0 if row["ledger_verdict_on_main"] == "TRACKED" else 0.0
        before = 1.0 if row["blob_at_merge_commit"] else 0.0
        after = 1.0 if row["blob_at_repair_commit"] else 0.0
        ax.scatter([0], [i], s=260, c="#2a7f62" if claim else "#c46a3a", marker="s")
        ax.scatter([1], [i], s=260, c="#2a7f62" if before else "#c46a3a", marker="s")
        ax.scatter([2], [i], s=260, c="#2a7f62" if after else "#c46a3a", marker="s")
        if claim and not before:
            ax.annotate(
                "claim false\non main",
                (1, i),
                textcoords="offset points",
                xytext=(14, 9),
                fontsize=7,
                color="#8a2f1d",
            )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["ledger says\nTRACKED", f"in git at\n{evidence['merge_commit']}",
                        f"in git at\n{evidence['repair_commit']}"], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("green = present, red = absent", fontsize=9)

    gate = evidence["gate"]
    commits = [evidence["merge_commit"], evidence["repair_commit"]]
    exits = [gate["merge_commit"]["exit"], gate["repair_commit"]["exit"]]
    ax2.bar(commits, exits, color=["#c46a3a" if e else "#2a7f62" for e in exits], width=0.5)
    for i, e in enumerate(exits):
        ax2.text(i, e + 0.03, f"exit {e}", ha="center", fontsize=9)
    ax2.set_ylabel("audit_doc_evidence.py --check")
    ax2.set_title("ledger gate exit code", fontsize=9)
    ax2.set_ylim(0, 1.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = collect()
    (OUT / "gate_evidence.json").write_text(
        json.dumps(evidence, indent=1) + "\n", encoding="utf-8"
    )
    print(figure(evidence, OUT / "gate_matrix.png"))
    print(json.dumps({k: v for k, v in evidence.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())