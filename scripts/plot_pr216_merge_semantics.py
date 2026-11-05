#!/usr/bin/env python3
"""Figure for the PR #216 cure: the merged doc-evidence ledger is a strict
union of both parents' evidence chains.

Left panel  – evidence pointers per side and merged, against the naive sum
              (main + branch, which double counts the 512 shared pointers).
Right panel – per-status buckets of the three ledgers: no bucket may regress,
              which is the visual form of check C3 in
              ``scripts/verify_pr216_merge_semantics.py``.

Usage: python scripts/plot_pr216_merge_semantics.py [--out PNG]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAIN, BRANCH = "6c6f98b", "d7c946c"


def ledger(ref: str) -> dict:
    for candidate in (ref, f"origin/{ref}"):
        out = subprocess.run(["git", "show", f"{candidate}:docs/doc_evidence_ledger.json"],
                             cwd=ROOT, capture_output=True, text=True)
        if out.returncode == 0:
            return json.loads(out.stdout)
    raise SystemExit(f"cannot read ledger at {ref}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/pr216/pr216_merge_semantics.png")
    args = ap.parse_args()

    sides = {"main\n6c6f98b": ledger(MAIN), "branch\nagent/m3558": ledger(BRANCH)}
    merged = json.loads((ROOT / "docs" / "doc_evidence_ledger.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    labels = list(sides) + ["merged\n(cure)", "naive sum\n(upper bound)"]
    values = [d["pointers"] for d in sides.values()] + [
        merged["pointers"], sum(d["pointers"] for d in sides.values())
    ]
    colors = ["#8aa8c8", "#8aa8c8", "#2a7f62", "#d9d9d9"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + 8, str(v), ha="center", fontsize=9)
    ax.set_ylabel("evidence pointers")
    ax.set_title("Merged pointer count is a strict union, not a pick")

    ax = axes[1]
    buckets = sorted({b for d in sides.values() for b in d["counts"]} | set(merged["counts"]))
    width = 0.26
    for i, (name, d) in enumerate(list(sides.items()) + [("merged", merged)]):
        ys = [d["counts"].get(b, 0) for b in buckets]
        ax.bar([x + (i - 1) * width for x in range(len(buckets))], ys, width=width,
               label=name.replace("\n", " "),
               color=["#8aa8c8", "#b8c9dc", "#2a7f62"][i])
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([b.replace(":", "\n") for b in buckets], fontsize=7)
    ax.set_ylabel("pointers")
    ax.set_title("No status bucket regresses in the merge (check C3)")
    ax.legend(fontsize=8)

    fig.suptitle("PR #216 cure: doc-evidence ledger regenerated on the merged tree", fontsize=11)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())