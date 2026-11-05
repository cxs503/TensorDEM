#!/usr/bin/env python3
"""Figures for `docs/dem3d_doc_numeric_consistency_audit.md`.

Reads only the checked-in ledger (`docs/doc_numeric_claims_ledger.json`), so the
numbers drawn here are the numbers the audit actually decided -- nothing is
hand-transcribed.  Written for the headless CI path (Agg backend).
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "doc_numeric_claims_ledger.json"
OUT = ROOT / "docs" / "figures"

COLOURS = {"PASS": "#2b7a3d", "FAIL": "#b3291f", "UNCHECKABLE": "#8a8a8a"}


def fig_status_by_doc(ledger, path):
    """Rows per document, split by audit outcome."""
    per = {}
    for r in ledger["records"]:
        per.setdefault(r["doc"], {}).setdefault(r["status"], 0)
        per[r["doc"]][r["status"]] += 1
    order = sorted(per, key=lambda d: (-sum(per[d].values()), d))
    fig, ax = plt.subplots(figsize=(9.0, 0.30 * len(order) + 1.6))
    left = [0] * len(order)
    for status in ("PASS", "FAIL", "UNCHECKABLE"):
        vals = [per[d].get(status, 0) for d in order]
        ax.barh(range(len(order)), vals, left=left, color=COLOURS[status],
                label=status, height=0.72)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([d.replace("docs/", "") for d in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("error-verdict table rows")
    ax.set_title(f"Numeric-claim audit: {ledger['rows_with_verdict']} verdict rows "
                 f"in {len(order)} documents", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    for i, d in enumerate(order):
        if per[d].get("FAIL"):
            ax.annotate("FAIL", (sum(per[d].values()), i), fontsize=7,
                        color=COLOURS["FAIL"], va="center",
                        xytext=(3, 0), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_recomputability(ledger, path):
    """Why a verdict row can or cannot be recomputed from the document itself."""
    reasons = dict(ledger.get("not_recomputed") or {})
    labels, values, colours = [], [], []
    recomputed = ledger["recomputed_rows"]
    labels.append("recomputed from reference + measured")
    values.append(recomputed)
    colours.append(COLOURS["PASS"])
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        labels.append(reason)
        values.append(n)
        colours.append(COLOURS["UNCHECKABLE"])
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.barh(range(len(labels)), values, color=colours, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    total = sum(values)
    for i, v in enumerate(values):
        ax.annotate(f"{v}  ({100.0 * v / total:.0f}%)", (v, i), fontsize=8,
                    va="center", xytext=(3, 0), textcoords="offset points")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("verdict rows")
    ax.set_title("Content layer coverage: only rows carrying both numbers can be "
                 "recomputed", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    fig_status_by_doc(ledger, OUT / "doc_numeric_claims_status.png")
    fig_recomputability(ledger, OUT / "doc_numeric_claims_recomputability.png")
    print("wrote", OUT / "doc_numeric_claims_status.png")
    print("wrote", OUT / "doc_numeric_claims_recomputability.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())