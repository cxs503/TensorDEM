"""Run the 2-D buoyancy / wet-drag free-float verification campaign.

Writes a deterministic JSON report and (optionally) a figure showing the
measured heave history against the analytic damped-oscillator reference.

Usage::

    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/verify_buoyancy_free_float.py \
        --out artifacts/buoyancy_free_float/verification.json \
        --figure docs/figures/buoyancy_free_float_verification.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tensordem.buoyancy_verification import (
    FreeFloatParameters,
    run_campaign,
)

ROOT = Path(__file__).resolve().parent.parent


def _write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _plot(report: dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixture = report["fixture"]
    heave = report["checks"]["damped_heave"]
    times = heave["history"]["time"]
    dev = [y - fixture["h_eq_analytic_m"] for y in heave["history"]["height"]]
    analytic = heave["analytic"]
    amp = heave["amplitude"]
    zeta = fixture["zeta"]
    omega_n = fixture["omega_n_rad_s"]
    envelope = [amp * math.exp(-zeta * omega_n * t) for t in times]
    resid = [d - a for d, a in zip(dev, analytic)]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    ax.plot([t * 1e3 for t in times], [d * 1e3 for d in dev], "o", ms=2.0,
            color="#1f4e79", label="DEM force law (RK4)")
    ax.plot([t * 1e3 for t in times], [a * 1e3 for a in analytic], "-", lw=1.6,
            color="#c00000", label="analytic damped heave")
    ax.plot([t * 1e3 for t in times], [e * 1e3 for e in envelope], "--", lw=1.0,
            color="0.4", label=r"$\pm A e^{-\zeta\omega_n t}$")
    ax.plot([t * 1e3 for t in times], [-e * 1e3 for e in envelope], "--", lw=1.0,
            color="0.4")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$y - h_{eq}$ [mm]")
    ax.set_title(
        f"(a) free float, $\\omega_d$ err "
        f"{100 * heave['omega_rel_error']:.3f}%"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    conv = report["checks"]["timestep_convergence"]
    dts = conv["dts"]
    diffs = conv["abs_differences"] + [float("nan")]
    finite = [(dt, d) for dt, d in zip(dts, diffs) if d == d and d > 0.0]
    ax.loglog([d[0] for d in finite], [d[1] for d in finite], "s-", color="#1f4e79",
              label="RK4 refinement")
    order = conv["observed_order"]
    ref_x = [dts[0], dts[-1]]
    if finite:
        anchor_x, anchor_y = finite[0]
        ref_y = [anchor_y, anchor_y * (ref_x[1] / anchor_x) ** order]
        ax.loglog(ref_x, ref_y, "--", color="#c00000",
                  label=f"slope {order:.2f}")
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"$|y(t_{end}) - y_{ref}|$ [m]")
    ax.set_title("(b) timestep convergence of the trajectory")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    lin = report["checks"]["linearisation_limit"]
    amps = [r["amplitude"] for r in lin["rows"]]
    errs = [r["waveform_max_rel_error"] for r in lin["rows"]]
    ax.loglog(amps, errs, "o-", color="#1f4e79", label="max |sim - linearised| / A")
    ax.axhline(0.03, color="#c00000", ls="--", lw=1.0, label="3% gate")
    ax.set_xlabel("excitation amplitude A [m]")
    ax.set_ylabel("normalised waveform error")
    ax.set_title("(c) residual is O(amplitude) nonlinearity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "TensorDEM 2-D buoyancy force-law verification: "
        f"$f_{{sub}}(h_{{eq}})=\\rho_i/\\rho_w={fixture['f_eq_expected']:.6f}$, "
        f"Archimedes draft error {100 * report['checks']['archimedes_draft']['draft_rel_error']:.2e}%",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "buoyancy_free_float" / "verification.json")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "docs" / "figures" / "buoyancy_free_float_verification.png")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="shorten the settling window (CI budget)")
    args = parser.parse_args(argv)

    report = run_campaign(FreeFloatParameters(), include_histories=True, quick=args.quick)
    written = _write_report(report, args.out)
    print(f"report -> {written}")
    if not args.no_figure:
        print(f"figure -> {_plot(report, args.figure)}")
    for name, check in report["checks"].items():
        verdict = "PASS" if check["passed"] else "FAIL"
        print(f"{verdict:4s} {name}")
    print("ALL PASSED" if report["all_passed"] else "CAMPAIGN FAILED")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())