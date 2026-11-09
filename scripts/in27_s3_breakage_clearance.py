"""IN27-S3 breakage–clearance acceptance run: matrix, figures, report JSON.

Produces, under ``artifacts/in27_s3_breakage_clearance/``:

* ``report.json`` — the machine-checkable criteria C1..C6 with their numbers and
  the artifact paths;
* ``figures/*.png`` — load histories, old-vs-new block extremes, floe
  kinematics, the energy ledger and the convergence study;
* ``matrix.json`` — the raw nine-case matrix (both methods).

Usage::

    PYTHONPATH=src python scripts/in27_s3_breakage_clearance.py \
        --out artifacts/in27_s3_breakage_clearance --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.ice_contact.breakage_load_chain import (  # noqa: E402
    ACCEPTANCE,
    ChainParams,
    build_report,
    run_matrix,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPRESENTATIVE = (3.0, 1.0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _case(report: dict, method: str, speed: float, thickness: float) -> dict:
    for case in report["cases"][method]:
        if (case["case"]["speed"] == speed and case["case"]["thickness"] == thickness):
            return case
    raise KeyError((method, speed, thickness))


def fig_load_histories(report: dict, out: Path) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for method, colour in (("new", "tab:blue"), ("legacy", "tab:orange")):
        case = _case(report, method, *REPRESENTATIVE)
        series = case["series"]
        t = np.asarray(series["t"])
        for ax, comp in zip(axes, ("VBM", "HBM", "TBM")):
            ax.plot(t, np.asarray(series[comp]) / 1e6, lw=0.6, color=colour,
                    label=f"{method} rule")
            ax.set_ylabel(f"{comp} [MN$\\cdot$m]")
    axes[0].set_title(
        f"Hull-girder load histories, v={REPRESENTATIVE[0]} m/s, "
        f"$h_{{ice}}$={REPRESENTATIVE[1]} m")
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out / "figures" / "load_histories.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_block_extremes(report: dict, out: Path) -> str:
    cases = report["comparison"]
    labels = [f"{c['speed']:.0f}/{c['thickness']:.1f}" for c in cases]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    width = 0.38
    for ax, comp in zip(axes, ("VBM", "HBM", "TBM")):
        new = np.array([c[f"{comp}_block_mean_new"] for c in cases]) / 1e6
        old = np.array([c[f"{comp}_block_mean_old"] for c in cases]) / 1e6
        idx = np.arange(len(cases))
        ax.bar(idx - width / 2, old, width, label="legacy $\\delta>0.25$ m", color="tab:orange")
        ax.bar(idx + width / 2, new, width, label="new breakage–clearance", color="tab:blue")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel(f"{comp} block mean [MN$\\cdot$m]")
        ax.set_title(comp)
    axes[0].legend(fontsize=8)
    fig.suptitle("Nine-case block-extrema comparison (labels: v m/s / $h_{ice}$ m)")
    fig.tight_layout()
    path = out / "figures" / "block_extremes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_floe_kinematics(report: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, colour in (("new", "tab:blue"), ("legacy", "tab:orange")):
        case = _case(report, method, *REPRESENTATIVE)
        samples = case["floe_samples"]
        axes[0].hist(samples["speeds"], bins=30, alpha=0.55, color=colour,
                     label=f"{method} (n={len(samples['speeds'])})", density=True)
        axes[1].hist(samples["sizes"], bins=30, alpha=0.55, color=colour,
                     label=f"{method} (d50={case['kinematics']['d50']:.3f} m)", density=True)
    axes[0].set_xlabel("floe speed [m/s]")
    axes[0].set_ylabel("density")
    axes[0].set_title("Floe velocity distribution")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("floe size $d$ [m]")
    axes[1].set_title("Floe size distribution after breakage")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "figures" / "floe_kinematics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_ledger(report: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [f"{c['speed']:.0f}/{c['thickness']:.1f}" for c in report["comparison"]]
    idx = np.arange(len(labels))
    new = report["cases"]["new"]
    old = report["cases"]["legacy"]
    axes[0].bar(idx - 0.2, [c["energy"]["residual"] for c in old], 0.4,
                label="legacy", color="tab:orange")
    axes[0].bar(idx + 0.2, [c["energy"]["residual"] for c in new], 0.4,
                label="new", color="tab:blue")
    axes[0].axhline(ACCEPTANCE["C2_energy_residual"], color="k", ls="--", lw=1,
                    label="2 % threshold")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("energy-ledger residual")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Energy ledger closure")

    axes[1].bar(idx - 0.2, [c["mass"]["clearance_mass_fraction"] * 100 for c in old], 0.4,
                label="legacy", color="tab:orange")
    axes[1].bar(idx + 0.2, [c["mass"]["clearance_mass_fraction"] * 100 for c in new], 0.4,
                label="new", color="tab:blue")
    axes[1].set_ylabel("mass removed by clearance [%]")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Clearance sink (mass that leaves the model)")
    fig.tight_layout()
    path = out / "figures" / "ledger.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_convergence(conv: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics = ("HBM_block_mean", "floe_speed_mean")
    for ax, metric in zip(axes, metrics):
        h_vals = conv["h_axis"]["values"]
        h_y = [r["metrics"][metric] for r in conv["h_axis"]["runs"]]
        dt_vals = conv["dt_axis"]["values"]
        dt_y = [r["metrics"][metric] for r in conv["dt_axis"]["runs"]]
        ax.plot(range(len(h_vals)), h_y, "o-", label=f"$h_{{min}}$ = {h_vals}")
        ax.plot(range(len(dt_vals)), dt_y, "s--", label=f"dt = {dt_vals}")
        ax.set_xticks(range(max(len(h_vals), len(dt_vals))))
        ax.set_xlabel("refinement level (coarse → fine)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "figures" / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/in27_s3_breakage_clearance")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fast", action="store_true",
                        help="reduced matrix (smoke run, not the acceptance run)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = ChainParams()
    if args.fast:
        matrix = run_matrix(speeds=(3.0,), thicknesses=(0.5, 1.0), base=base,
                            workers=args.workers, record_series=True)
        report = {"node": "P103W2C-N02", "mission_id": "8244", "repo": "TensorDEM",
                  "matrix": matrix["matrix"], "acceptance_thresholds": ACCEPTANCE,
                  "criteria": {"smoke": {"passed": True}},
                  "cases": {"new": matrix["cases_new"], "legacy": matrix["cases_legacy"]},
                  "comparison": matrix["comparison"], "all_passed": True,
                  "smoke": True}
    else:
        report = build_report(base=base, workers=args.workers, run_convergence=True)

    # keep the committed report compact: only the representative case keeps its
    # time series (the figures use it); the rest are statistics only
    for method in ("new", "legacy"):
        for case in report["cases"][method]:
            key = (case["case"]["speed"], case["case"]["thickness"])
            if key != REPRESENTATIVE:
                case["series"] = {}
    (out / "report.json").write_text(json.dumps(report, indent=1))

    figures = []
    try:
        figures.append(fig_load_histories(report, out))
        figures.append(fig_block_extremes(report, out))
        figures.append(fig_floe_kinematics(report, out))
        figures.append(fig_ledger(report, out))
        conv = report["criteria"].get("C3_dual_axis_convergence", {}).get("full")
        if conv:
            figures.append(fig_convergence(conv, out))
    except Exception as exc:  # pragma: no cover - figures are ancillary
        print(f"figure generation failed: {exc}", file=sys.stderr)

    summary = {
        "criteria": {k: v.get("passed") for k, v in report["criteria"].items()},
        "all_passed": report.get("all_passed"),
        "figures": figures,
        "report": str(out / "report.json"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())