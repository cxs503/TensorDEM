"""Plot the verification-evidence ledger (requires the optional `viz` extra).

Reads ``docs/verification-evidence/verification_ledger.json`` and writes a
two-panel figure: the distribution of evidence classes over test functions, and the
per-module grade / attribution split. The script is intentionally separate from
``audit_verification_evidence.py`` so that the audit itself stays dependency-free.

    python scripts/plot_verification_evidence.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "verification-evidence" / "verification_ledger.json"
DEFAULT_FIGURE = ROOT / "docs" / "verification-evidence" / "verification_evidence.png"

CLASS_ORDER = ("QUANTITATIVE", "EXACT_VALUE", "ORDERING", "ERROR_PATH", "SMOKE", "NONE")
GRADE_ORDER = (
    "QUANTITATIVE_DOCUMENTED",
    "QUANTITATIVE_UNDOCUMENTED",
    "EXACT_ONLY",
    "ORDERING_ONLY",
    "ERROR_PATH_ONLY",
    "SMOKE_ONLY",
    "NO_TEST",
)


def build_figure(ledger: dict, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_counts = Counter(record["evidence"] for record in ledger["test_functions"])
    grade_counts = Counter(row["grade"] for row in ledger["module_grades"])
    attribution_counts = Counter(row["attribution"] for row in ledger["module_grades"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    labels = list(CLASS_ORDER)
    values = [class_counts.get(label, 0) for label in labels]
    bars = axes[0].bar(labels, values, color="#3b6ea5")
    axes[0].set_title(
        f"Test-function evidence class\n(n = {sum(values)} test functions)"
    )
    axes[0].set_ylabel("test functions")
    axes[0].tick_params(axis="x", rotation=40, labelsize=8)
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value, str(value),
            ha="center", va="bottom", fontsize=8,
        )

    labels = list(GRADE_ORDER)
    values = [grade_counts.get(label, 0) for label in labels]
    bars = axes[1].bar(labels, values, color="#7a4fa3")
    axes[1].set_title(f"Module evidence grade\n(n = {sum(values)} modules)")
    axes[1].set_ylabel("modules")
    axes[1].tick_params(axis="x", rotation=40, labelsize=8)
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, value, str(value),
            ha="center", va="bottom", fontsize=8,
        )

    labels = ["direct", "file", "none"]
    values = [attribution_counts.get(label, 0) for label in labels]
    bars = axes[2].bar(labels, values, color="#c1743a")
    axes[2].set_title("Test-to-module attribution confidence")
    axes[2].set_ylabel("modules")
    for bar, value in zip(bars, values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, value, str(value),
            ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle(
        "TensorDEM verification-evidence ledger "
        f"(schema {ledger['schema']})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110)
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    path = build_figure(ledger, args.output)
    print(f"figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())