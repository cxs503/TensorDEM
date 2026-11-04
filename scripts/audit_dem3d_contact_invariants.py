"""Audit the DEM3D contact/bond force kernel with metamorphic invariants.

Runs :mod:`tensordem.contact_invariants` over fissured and intact lattices and
writes a machine-readable report (JSON + CSV) plus an optional figure.

    python scripts/audit_dem3d_contact_invariants.py --output-dir results-dem3d-contact-invariants
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensordem.contact_invariants import (  # noqa: E402
    DEFAULT_CASES,
    INVARIANTS,
    TOLERANCES,
    monitor_purity,
    run_audit,
    summarise,
    violations,
)


def run(seeds: tuple[int, ...], output_dir: Path, figure: bool) -> dict:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_audit(seeds=seeds)
    rows = [result.as_row() for result in results]
    summary = summarise(results)
    purity = [
        {"case": case.name, **monitor_purity(case, seed=0, steps=60)}
        for case in DEFAULT_CASES
        if case.friction_coefficient > 0.0
        or case.rolling_resistance_coefficient > 0.0
    ]
    bad = violations(results)

    with (output_dir / "invariant_results.csv").open("w", newline="") as handle:
        fieldnames = ["invariant", "case", "seed", "residual"]
        extras = sorted({key for row in rows for key in row} - set(fieldnames))
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extras, restval="")
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "invariant_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["invariant", "case", "samples", "max_residual", "tolerance", "passed"])
        for row in summary:
            writer.writerow(
                [
                    row["invariant"],
                    row["case"],
                    row["samples"],
                    f"{row['max_residual']:.3e}",
                    f"{row['tolerance']:.1e}",
                    row["passed"],
                ]
            )

    report = {
        "seeds": list(seeds),
        "cases": [case.name for case in DEFAULT_CASES],
        "invariants": list(INVARIANTS),
        "tolerances": TOLERANCES,
        "samples": len(results),
        "violations": [
            {"invariant": v.invariant, "case": v.case, "seed": v.seed, "residual": v.residual}
            for v in bad
        ],
        "summary": summary,
        "monitor_purity": purity,
        "worst_residual": max((row["max_residual"] for row in summary), default=0.0),
        "passed": not bad,
    }
    if figure:
        report["figure"] = str(plot(report, output_dir / "contact_invariants.png"))

    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, default=str))
    (output_dir / "audit_report.md").write_text(markdown(report))
    return report


def plot(report: dict, path: Path) -> Path:
    """One figure: invariant residuals against tolerance, plus purity evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = report["summary"]
    labels = [f"{row['invariant']}\n{row['case']}" for row in summary]
    values = [max(row["max_residual"], 1.0e-18) for row in summary]
    tolerances = [row["tolerance"] for row in summary]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 7.0), constrained_layout=True)
    colors = ["#2b6cb0" if row["passed"] else "#c53030" for row in summary]
    positions = range(len(summary))
    left.barh(list(positions), values, color=colors, height=0.7)
    left.set_yticks(list(positions))
    left.set_yticklabels(labels, fontsize=6.0)
    left.set_xscale("log")
    left.set_xlim(1.0e-18, 1.0e-4)
    left.vlines(tolerances, [p - 0.4 for p in positions], [p + 0.4 for p in positions],
                colors="#1a202c", linestyles=":", linewidth=1.0, label="tolerance")
    left.set_xlabel("worst relative residual over seeds (log scale)")
    left.set_title("Metamorphic invariants of the DEM3D contact kernel\n"
                   f"{report['samples']} samples, {len(report['violations'])} violations")
    left.legend(loc="lower right", fontsize=7)
    left.invert_yaxis()
    left.grid(axis="x", which="both", alpha=0.3)

    purity = report.get("monitor_purity") or []
    if purity:
        names = [row["case"] for row in purity]
        metrics = {
            "position divergence (m)": [max(row["position_divergence_m"], 1e-18) for row in purity],
            "velocity divergence (m/s)": [max(row["velocity_divergence_m_s"], 1e-18) for row in purity],
            "probe repeat defect (N)": [max(row["probe_repeat_defect_N"], 1e-18) for row in purity],
        }
        width = 0.25
        for index, (label, series) in enumerate(metrics.items()):
            offsets = [position + (index - 1) * width for position in range(len(names))]
            right.bar(offsets, series, width=width, label=label)
        right.set_yscale("log")
        right.set_xticks(range(len(names)))
        right.set_xticklabels(names, fontsize=7, rotation=12)
        right.axhline(1.2e-16, color="#c53030", linestyle="--", linewidth=1.0,
                      label="float64 round-off floor")
        right.set_title("Monitor purity: monitoring a run must not change it\n"
                        "(pre-fix position divergence at this state: 1.5e-4 m)")
        right.legend(fontsize=7)
        right.grid(axis="y", alpha=0.3)

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown(report: dict) -> str:
    """Render the audit report as a markdown table."""
    lines = [
        "# DEM3D contact-kernel invariance audit",
        "",
        f"Samples: {report['samples']} "
        f"(cases: {', '.join(report['cases'])}; seeds: {report['seeds']})",
        "",
        "| invariant | case | samples | worst residual | tolerance | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['invariant']} | {row['case']} | {row['samples']} | "
            f"{row['max_residual']:.2e} | {row['tolerance']:.1e} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines += ["", "## Monitor purity", "", "| case | steps | Δx (m) | Δv (m/s) | probe defect (N) |",
              "| --- | --- | --- | --- | --- |"]
    for row in report.get("monitor_purity") or []:
        lines.append(
            f"| {row['case']} | {int(row['steps'])} | {row['position_divergence_m']:.2e} | "
            f"{row['velocity_divergence_m_s']:.2e} | {row['probe_repeat_defect_N']:.2e} |"
        )
    lines += ["", f"Overall: {'PASS' if report['passed'] else 'FAIL'}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results-dem3d-contact-invariants")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)
    report = run(tuple(args.seeds), args.output_dir, args.figure)
    print(json.dumps({"passed": report["passed"], "samples": report["samples"],
                      "worst_residual": report["worst_residual"],
                      "violations": report["violations"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())