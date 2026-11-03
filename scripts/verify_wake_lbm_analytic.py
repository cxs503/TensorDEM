"""Analytic verification of the D3Q19 + Smagorinsky propeller-race solver.

Regenerates ``artifacts/wake_lbm_verification/`` (JSON ladder, CSV table,
figure) and prints a machine-checkable summary.  Exit status is 1 when any rung
of the ladder fails, so the script can gate CI.

    PYTHONPATH=src python scripts/verify_wake_lbm_analytic.py
    PYTHONPATH=src python scripts/verify_wake_lbm_analytic.py --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import torch

from tensordem.threefield.wake_lbm_verification import (
    HARNESS_SCHEMA,
    PeriodicShearWave,
    log_linear_decay_rate,
    run_verification_suite,
)

DEFAULT_OUT = os.path.join("artifacts", "wake_lbm_verification")


def _decay_panels():
    """Measured viscous decay of the shear mode for three relaxation times."""
    panels = []
    for tau0 in (0.55, 0.7, 1.0):
        fixture = PeriodicShearWave(nx=32, tau0=tau0, amplitude=1.0e-4)
        trace = fixture.history(1, sample_every=1)
        while trace[-1][1] > 0.30 * fixture.amplitude0 and fixture.steps < 20000:
            trace.extend(fixture.history(10, sample_every=10))
        rate = log_linear_decay_rate(trace, window=0.30)
        panels.append(
            {
                "tau0": tau0,
                "viscosity": fixture.viscosity,
                "wavenumber": fixture.wavenumber,
                "time": [t for t, _ in trace],
                "amplitude": [a / fixture.amplitude0 for _, a in trace],
                "measured_rate": rate,
                "predicted_rate": fixture.viscosity * fixture.wavenumber**2,
                "relative_error": abs(rate - fixture.viscosity * fixture.wavenumber**2)
                / (fixture.viscosity * fixture.wavenumber**2),
            }
        )
    return panels


def _figure(report, decay_panels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    axis = axes[0]
    for panel in decay_panels:
        tau0 = panel["tau0"]
        times = panel["time"]
        predicted = [
            math.exp(-panel["predicted_rate"] * t) for t in times
        ]
        axis.semilogy(times, panel["amplitude"], lw=1.2, label=f"sim tau0={tau0}")
        axis.semilogy(
            times,
            predicted,
            "k--",
            lw=0.8,
            label=f"exp(-nu k^2 t), nu={panel['viscosity']:.3g}",
        )
    axis.set_xlabel("lattice steps")
    axis.set_ylabel("A(t) / A(0)")
    axis.set_title("V5  shear-wave decay vs closed form")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)

    axis = axes[1]
    convergence = report["data"]["V5"]["convergence"]
    ks = [entry["wavenumber"] for entry in convergence]
    err = [entry["relative_error"] for entry in convergence]
    axis.loglog(ks, err, "o-", label="measured decay-rate error")
    reference = [err[0] * (k / ks[0]) ** 2 for k in ks]
    axis.loglog(ks, reference, "k--", label=r"$k^2$ reference (2nd order)")
    axis.set_xlabel("wavenumber k")
    axis.set_ylabel("|gamma_meas - nu k^2| / (nu k^2)")
    axis.set_title("V5  O(k^2) convergence of the viscosity")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")

    axis = axes[2]
    from tensordem.threefield.wake_lbm_verification import sgs_damped_decay_check

    sgs = sgs_damped_decay_check(steps=120)
    measured = sgs["amplitudes"]
    predicted = sgs["predicted"]
    steps = list(range(len(measured)))
    axis.plot(steps, measured, "o", ms=3, label="D3Q19 + Smagorinsky")
    axis.plot(steps, predicted, "k--", lw=1.0, label="closed-form ODE")
    axis.set_xlabel("sample (thinned)")
    axis.set_ylabel("mode amplitude A")
    axis.set_title("V8  SGS-damped decay vs analytic ODE")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    figure.suptitle(
        "TensorDEM wake_lbm analytic verification "
        f"({HARNESS_SCHEMA}); worst rung error "
        f"{max(row['worst_relative_error'] for row in report['rows']):.2e}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    os.makedirs(args.outdir, exist_ok=True)

    report = run_verification_suite(quick=args.quick)
    payload = dict(report)
    payload["decay_panels"] = _decay_panels()
    with open(os.path.join(args.outdir, "analytic_verification.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    with open(os.path.join(args.outdir, "analytic_verification.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "quantity",
                "anchor",
                "samples",
                "worst_relative_error",
                "tolerance",
                "passed",
                "note",
            ]
        )
        for row in report["rows"]:
            writer.writerow(
                [
                    row["level"],
                    row["quantity"],
                    row["anchor"],
                    row["samples"],
                    f"{row['worst_relative_error']:.6e}",
                    f"{row['tolerance']:.1e}",
                    row["passed"],
                    row["note"],
                ]
            )

    markdown = [
        "| level | quantity | anchor | samples | worst rel. error | tolerance | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        markdown.append(
            "| {level} | {quantity} | {anchor} | {samples} | {err:.3e} | {tol:.0e} | "
            "{passed} |".format(
                level=row["level"],
                quantity=row["quantity"],
                anchor=row["anchor"],
                samples=row["samples"],
                err=row["worst_relative_error"],
                tol=row["tolerance"],
                passed="yes" if row["passed"] else "**NO**",
            )
        )
    with open(os.path.join(args.outdir, "summary.md"), "w") as handle:
        handle.write("\n".join(markdown) + "\n")

    if not args.no_figure:
        _figure(report, payload["decay_panels"], os.path.join(args.outdir, "verification.png"))

    print("\n".join(markdown))
    print()
    for row in report["rows"]:
        if row["note"]:
            print(f"{row['level']}: {row['note']}")
    print(f"\nladder ok = {report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())