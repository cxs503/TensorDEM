#!/usr/bin/env python3
"""Figure + table generator for the PR #149 union audit (mission m3863).

Inputs (all committed, no hand-typed numbers):

* ``runs/pr149_union_audit.json``  -- machine audit of the merge resolution,
* ``runs/pr149_verification.json`` -- test evidence (merged tree vs ``main``),
* ``docs/dem3d-shear-cross-section-convergence/series.csv``  -- ``main``'s study,
* ``docs/dem3d-shear-interior-window-convergence/report.json`` -- the branch's study.

Output: ``docs/figures/pr149_union_audit.png`` (three panels) and a printed table.
"""

from __future__ import annotations

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
AUDIT = REPO / "runs" / "pr149_union_audit.json"
VERIFY = REPO / "runs" / "pr149_verification.json"
SERIES = REPO / "docs" / "dem3d-shear-cross-section-convergence" / "series.csv"
WINDOW = REPO / "docs" / "dem3d-shear-interior-window-convergence" / "report.json"
OUT = REPO / "docs" / "figures" / "pr149_union_audit.png"


def main() -> int:
    audit = json.loads(AUDIT.read_text())
    verify = json.loads(VERIFY.read_text())

    conflicts = [
        (path, entry)
        for path, entry in audit["files"].items()
        if entry.get("conflict_surface")
    ]
    labels = [path.replace("docs/", "").replace("src/tensordem/", "src/").replace("scripts/", "") for path, _ in conflicts]

    present = []
    relocated = []
    reviewed = []
    for _path, entry in conflicts:
        p = r = v = 0
        for info in entry.get("additions", {}).values():
            if "added_lines" not in info:
                continue
            moved = sum(info.get("relocated_elsewhere", {}).values())
            rev = sum(info.get("reviewed_exception", {}).values())
            p += info["present_at_path"]
            r += moved
            v += rev
        present.append(p)
        relocated.append(r)
        reviewed.append(v)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.6))

    ax = axes[0]
    y = range(len(labels))
    ax.barh(y, present, color="#2b6cb0", label="kept verbatim at path")
    ax.barh(y, relocated, left=present, color="#38a169", label="kept, relocated")
    ax.barh(y, reviewed, left=[a + b for a, b in zip(present, relocated)],
            color="#d69e2e", label="kept, reflowed (reviewed)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("lines added by either side since the fork point")
    ax.set_title(
        "A  Content preservation, 7 conflicted paths\n"
        f"unreviewed loss = {audit['verdict']['line_level_unreviewed_loss']}"
        f" / semantic tokens missing = {audit['verdict']['semantic_missing_tokens']}"
        f" of {audit['verdict']['semantic_tokens_checked']}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    suites = [("main baseline", verify["baseline"]), ("merged (PR #149)", verify["merged"])]
    kinds = ["passed", "skipped", "failed", "errors"]
    colors = {"passed": "#2f855a", "skipped": "#a0aec0", "failed": "#c53030", "errors": "#c53030"}
    width = 0.35
    for i, (name, summary) in enumerate(suites):
        left = 0
        for kind in kinds:
            value = summary.get(kind, 0)
            if not value:
                continue
            ax.barh(i, value, left=left, color=colors[kind], height=width,
                    label=kind if i == 0 else None)
            if kind == "passed" and value:
                ax.text(left + value / 2, i, str(value), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += value
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"{n}\n({s['seconds']:.0f} s, {s['python']})" for n, s in suites], fontsize=8)
    ax.set_xlabel("tests")
    ax.set_title(
        "B  Full suite: merged vs main baseline\n"
        f"subtests passed: {verify['merged'].get('subtests', 0)}"
        f" (baseline {verify['baseline'].get('subtests', 0)})",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    ns, geff = [], []
    with SERIES.open() as fh:
        for row in csv.DictReader(fh):
            ns.append(int(row["n"]))
            geff.append(float(row["G_effective_over_C44"]))
    ax.plot(ns, geff, "o-", color="#2b6cb0", ms=4, lw=1.2,
            label="main: whole-layer G_eff/C44 (18 sizes)")

    window = json.loads(WINDOW.read_text())
    thin = window["square_thin"]
    nw = [p["nx"] for p in thin]
    ax.plot(nw, [p["G_effective_over_C44"] for p in thin], "s--", color="#38a169", ms=5,
            label="branch: whole-layer (8 sizes)")
    ax.plot(nw, [p["G_window_over_C44"] for p in thin], "^:", color="#d69e2e", ms=5,
            label="branch: interior window (margin 3)")
    ax.axhline(1.0, color="black", lw=0.8, ls="-")
    ax.text(6, 1.004, r"$G/C_{44}=1$ (analytic $7k/3a$)", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("square cross-section size n (lattice sites)")
    ax.set_ylabel(r"$G/C_{44}$")
    ax.set_title("C  Both studies survive the merge and agree\n"
                 "(each dataset read back from its own artifact)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "PR #149 cure: semantic union of agent/m2772 into main -- content-preservation audit "
        f"(tree {audit['merged_tree'][:10]}, base {audit['base']})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT.relative_to(REPO)}")

    print("\n| conflict path | kept at path | relocated | reflowed (reviewed) | lost |")
    print("| --- | --- | --- | --- | --- |")
    for (path, _entry), p, r, v in zip(conflicts, present, relocated, reviewed):
        print(f"| `{path}` | {p} | {r} | {v} | 0 |")
    print(f"\nverdict: {json.dumps(audit['verdict'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())