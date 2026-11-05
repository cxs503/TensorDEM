#!/usr/bin/env python3
"""Figure + numbers for the PR #149 merge-gate re-verification (mission m2772).

Everything plotted is read back from committed records -- the union audit, the
verification record and the figure file on disk -- so no number is typed by
hand.  Writes ``docs/figures/pr149_regate_audit.png`` and
``runs/pr149_regate_report.json``.

Usage::

    python scripts/make_pr149_regate_report.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
AUDIT = REPO / "runs" / "pr149_union_audit.json"
VERIFICATION = REPO / "runs" / "pr149_verification.json"
FIGURE = REPO / "docs" / "figures" / "pr149_regate_audit.png"
OUT = REPO / "runs" / "pr149_regate_report.json"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit = json.loads(AUDIT.read_text())
    verification = json.loads(VERIFICATION.read_text())
    verdict = audit["verdict"]
    rerun = verification["rerun_2c6327f"]
    cap = rerun["artifact_cap"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # --- panel A: union-audit verdict, one bar per non-negotiable invariant ---
    ax = axes[0]
    names = ["unreviewed\nlost lines", "revived\nlines", "missing\nsemantic tokens",
             "parse\nerrors", "duplicate\ndefs", "reviewed\nreflows"]
    vals = [verdict["line_level_unreviewed_loss"], verdict["revived_lines_review"],
            verdict["semantic_missing_tokens"], verdict["parse_errors"],
            verdict["duplicate_defs"], verdict["line_level_reviewed_exceptions"]]
    colors = ["#c53030" if v else "#2b6cb0" for v in vals[:5]] + ["#d69e2e"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v),
                ha="center", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(f"A  Union audit (tree {audit['merged_tree'][:8]})\n"
                 f"{verdict['semantic_tokens_checked']} semantic tokens checked, "
                 f"gate {'PASS' if verdict['ok'] else 'FAIL'}", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25 + 0.5)
    ax.grid(alpha=0.3, axis="y")

    # --- panel B: artifact cap ---
    ax = axes[1]
    sizes = [cap["bytes_before"], cap["bytes_after"], cap["cap_bytes"]]
    labels = [f"before\n(dpi 140)", f"after\n(dpi 112)", "gate cap"]
    bars = ax.bar(labels, [s / 1024 for s in sizes],
                  color=["#c53030", "#2f855a", "#4a5568"])
    for b, s in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, s / 1024 + 2, f"{s:,} B",
                ha="center", fontsize=9)
    ax.set_ylabel("KiB")
    ax.set_title("B  Artifact cap on the audit figure\n"
                 f"{-100 * (1 - cap['bytes_after'] / cap['bytes_before']):.1f} % smaller, "
                 "same audit JSON", fontsize=10)
    ax.set_ylim(0, max(sizes) / 1024 * 1.2)
    ax.grid(alpha=0.3, axis="y")

    # --- panel C: verification mass, merged vs main baseline ---
    ax = axes[2]
    base = verification["baseline"]
    merged = verification["merged"]
    rerun_tests = rerun["tests"]
    groups = ["full suite\n(main e12a2ab)", "full suite\n(merged tree)",
              "targeted re-run\n(this commit)"]
    counts = [base["passed"], merged["passed"], rerun_tests["passed"]]
    fails = [base["failed"] + base["errors"], merged["failed"] + merged["errors"],
             rerun_tests["failed"]]
    x = range(len(groups))
    bars = ax.bar(x, counts, width=0.55, color="#2b6cb0", label="passed")
    ax.bar(x, fails, width=0.55, bottom=counts, color="#c53030", label="failed/error")
    for xi, c, f in zip(x, counts, fails):
        ax.text(xi, c + f + 15, f"{c} passed / {f} fail", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("tests")
    ax.set_title("C  Test mass\nmerged = main + the branch's 17 new tests", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("PR #149 re-verification: main-based union, artifact cap, ledger currency",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=112)

    record = {
        "figure": str(FIGURE.relative_to(REPO)),
        "figure_bytes": FIGURE.stat().st_size,
        "audit_verdict": verdict,
        "artifact_cap": cap,
        "tests": {
            "baseline_passed": base["passed"],
            "merged_passed": merged["passed"],
            "rerun_passed": rerun_tests["passed"],
            "rerun_failed": rerun_tests["failed"],
            "delta_vs_main": merged["passed"] - base["passed"],
        },
        "pr_head_reconciliation": verification["pr_head_reconciliation"],
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {FIGURE.relative_to(REPO)} ({record['figure_bytes']} B) and "
          f"{OUT.relative_to(REPO)}")
    print(json.dumps(record["tests"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())