"""Build the Lindqvist/Riska identifiability audit for the MT Uikku case set.

Writes ``artifacts/threefield/literature_identifiability.json`` and two figures::

    artifacts/threefield/figures/literature_as_shipped_errors.png
    artifacts/threefield/figures/literature_speed_signature.png

Every digit quoted in ``docs/threefield/LITERATURE_IDENTIFIABILITY.md`` is copied
from the JSON this script writes, so the document cannot drift from the code.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensordem.threefield.literature_check import (  # noqa: E402
    TOLERANCE_PCT,
    format_markdown_table,
    load_cases,
    run_audit,
    shipped_speed_signature,
)
from tensordem.threefield.resistance import HullGeometry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "threefield"
FIG_DIR = OUT_DIR / "figures"


def _case_lookup(report):
    """Typed cases (from the audit report ids) so the shipped chain can be re-run."""
    return {case.test_id: case for case in load_cases()}


def figure_as_shipped_errors(report, path: Path) -> None:
    cases = report["cases"]
    labels = [c["test_id"] for c in cases]
    lind = report["as_shipped"]["lindqvist"]["per_case_pct"]
    risk = report["as_shipped"]["riska"]["per_case_pct"]
    best_l = report["best_feasible_fit"]["lindqvist"]["per_case_pct"]
    best_r = report["best_feasible_fit"]["riska"]["per_case_pct"]
    index = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.bar(index - 1.5 * width, lind, width, label="shipped Lindqvist (default hull)")
    ax.bar(index - 0.5 * width, risk, width, label="shipped Riska (default hull)")
    ax.bar(index + 0.5 * width, best_l, width, hatch="//",
           label="Lindqvist, best hull in envelope")
    ax.bar(index + 1.5 * width, best_r, width, hatch="//",
           label="Riska, best hull in envelope")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for sign, style in ((1.0, "--"), (-1.0, "--")):
        ax.axhline(sign * TOLERANCE_PCT, color="tab:red", linestyle=style, linewidth=0.9)
    ax.text(0.02, TOLERANCE_PCT + 1.5, "+/-15 % tolerance gate", color="tab:red",
            fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    ax.set_xlabel("MT Uikku level-ice test")
    ax.set_ylabel("signed error vs published prediction [%]")
    ax.set_title("Shipped level-ice baselines vs published predictions\n"
                 "(positive = shipped chain over-predicts)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_speed_signature(report, path: Path) -> None:
    lookup = _case_lookup(report)
    published_kN = {case["test_id"]: case["published_riska_n"] / 1e3
                    for case in report["cases"]}
    groups = report["published_riska_linearity"]["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    for group in groups:
        members = [lookup[tid] for tid in group["test_ids"]]
        v = np.array([m.speed_m_s for m in members])
        r = np.array([published_kN[m.test_id] for m in members])
        ax.plot(v, r, "o", label="published Riska, h = %.2f m" % group["thickness_m"])
        vv = np.linspace(v.min(), v.max(), 50)
        ax.plot(vv, (group["c1_n"] + group["c2_n_per_m_s"] * vv) / 1e3,
                linewidth=1.0, color="tab:gray")
        shipped = [shipped_speed_signature(lookup[tid], HullGeometry())
                   for tid in group["test_ids"]]
        ax.plot(v, [(s["c1_n"] + s["c2_n_per_m_s"] * vi) / 1e3
                    for s, vi in zip(shipped, v)],
                "s--", label="shipped Riska, h = %.2f m (default hull)" % group["thickness_m"])
    ax.set_xlabel("ice drift speed V [m/s]")
    ax.set_ylabel("resistance [kN]")
    ax.set_title("Published Riska column is exactly affine in V;\nthe shipped chain is nearly flat")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    published = report["published_speed_signature_s_per_m"]
    feasible = report["feasible_speed_signature"]["c2_over_c1_s_per_m"]
    default = np.mean([abs(shipped_speed_signature(c, HullGeometry())["c2_over_c1_s_per_m"])
                       for c in lookup.values()])
    bars = ax.bar(["published\n(from data)", "shipped\n(default hull)",
                   "shipped, max over\nhull envelope"],
                  [published, default, feasible],
                  color=["tab:blue", "tab:orange", "tab:green"])
    for bar, value in zip(bars, (published, default, feasible)):
        ax.text(bar.get_x() + bar.get_width() / 2, value, "%.3f" % value,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("relative speed dependence C2 / C1 [s/m]")
    ax.set_title("Hull-insensitive signature: gap factor %.2f x"
                 % report["speed_signature_gap_factor"])
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    report = run_audit(samples=3000, signature_samples=20000, seed=20261104)
    (OUT_DIR / "literature_identifiability.json").write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
    )
    figure_as_shipped_errors(report, FIG_DIR / "literature_as_shipped_errors.png")
    figure_speed_signature(report, FIG_DIR / "literature_speed_signature.png")
    print(format_markdown_table(report))
    print()
    print("verdict:", json.dumps(report["verdict"], indent=1))
    print("as shipped:", {k: round(v["max_abs_pct"], 2)
                          for k, v in report["as_shipped"].items()})
    print("best feasible:", {k: round(v["max_abs_pct"], 2)
                             for k, v in report["best_feasible_fit"].items()})
    print("speed signature gap:", round(report["speed_signature_gap_factor"], 3))
    print("wrote", OUT_DIR / "literature_identifiability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())