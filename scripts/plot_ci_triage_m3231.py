"""Plot the m3231 CI triage: failures per cluster before vs after the fix.

All numbers are parsed from the pytest logs themselves (no hand-typed counts),
so the figure cannot drift from the runs it claims to summarise.

Usage
-----
    python scripts/plot_ci_triage_m3231.py \
        --baseline runs/ci-m3231/pytest_baseline.log \
        --after runs/ci-m3231/pytest_after.log \
        --output artifacts/ci_triage_m3231/failure_clusters.png
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Test module -> cluster label (root cause of the red).
CLUSTERS = {
    "test_ice_floe_physics.py": "C1 ice_floe anchors",
    "test_fracture_gen_audit.py": "C2 fracture_gen API",
    "test_dem3d_moving_wedge_bow.py": "C3 override signature",
    "test_dem3d_wedge_resistance_sensitivity.py": "C3 override signature",
    "test_threefield.py": "C3 override signature",
    "test_dem3d_restitution_verification.py": "C4 restitution numeric",
    "test_wake_lbm_analytic_verification.py": "C5 LBM closure",
    "test_doc_evidence_pointers.py": "C6 doc evidence ledger",
}
FIXED_IN_THIS_MISSION = {"C3 override signature", "C6 doc evidence ledger"}

_FAILED = re.compile(r"^FAILED (tests/[^:]+)::")
_SUMMARY = re.compile(r"^(\d+) failed, (\d+) passed")


def parse(log: Path) -> tuple[collections.Counter, tuple[int, int]]:
    per_cluster: collections.Counter[str] = collections.Counter()
    totals = (0, 0)
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _FAILED.match(line)
        if match:
            name = Path(match.group(1)).name
            per_cluster[CLUSTERS.get(name, f"other:{name}")] += 1
        summary = _SUMMARY.match(line)
        if summary:
            totals = (int(summary.group(1)), int(summary.group(2)))
    return per_cluster, totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base, base_tot = parse(args.baseline)
    after, after_tot = parse(args.after)
    labels = sorted(set(base) | set(after), key=lambda label: -base.get(label, 0))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [2.4, 1.0]})
    ys = range(len(labels))
    height = 0.38
    axes[0].barh([y + height / 2 for y in ys], [base.get(l, 0) for l in labels], height,
                 label=f"baseline @f5de971 ({base_tot[0]} failed)", color="#c0392b")
    axes[0].barh([y - height / 2 for y in ys], [after.get(l, 0) for l in labels], height,
                 label=f"after fix ({after_tot[0]} failed)", color="#1f77b4")
    axes[0].set_yticks(list(ys))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("failing tests in cluster")
    axes[0].set_title("Failure clusters: baseline vs after fix")
    axes[0].legend(loc="lower right", fontsize=8)
    for y, label in zip(ys, labels):
        axes[0].text(base.get(label, 0) + 0.2, y + height / 2, str(base.get(label, 0)),
                     va="center", fontsize=8)
        axes[0].text(after.get(label, 0) + 0.2, y - height / 2, str(after.get(label, 0)),
                     va="center", fontsize=8)
    axes[0].grid(axis="x", alpha=0.3)

    # Second panel: the outcome per cluster, coloured by whether this mission fixed it.
    delta = [base.get(l, 0) - after.get(l, 0) for l in labels]
    colours = ["#27ae60" if l in FIXED_IN_THIS_MISSION else "#7f8c8d" for l in labels]
    axes[1].barh(list(ys), delta, color=colours)
    axes[1].set_yticks(list(ys))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("tests turned green")
    axes[1].set_title("Closed in m3231 (green)\nvs deferred (grey)")
    for y, value in zip(ys, delta):
        axes[1].text(value + 0.05, y, str(value), va="center", fontsize=8)
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    summary = {
        "baseline_failed": base_tot[0],
        "baseline_passed": base_tot[1],
        "after_failed": after_tot[0],
        "after_passed": after_tot[1],
        "clusters": {
            label: {"baseline": base.get(label, 0), "after": after.get(label, 0)} for label in labels
        },
    }
    (args.output.parent / "failure_clusters.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())