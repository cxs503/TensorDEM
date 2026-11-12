"""R2 estimator audit for CASE-BIAXIAL-SHEAR-BAND (no DEM stepping).

Background: the delivered case measured a band inclination of ``44.25 +- 0.8`` /
``44.67 +- 0.5`` deg for its two resolved runs and reported an honest FAIL
against the 60 deg dense-sand anchor.  This script shows that the *estimator*
responsible for that number was biased, and quantifies the bias:

* synthetic bands of known inclination are injected into the exact lattice
  geometry of the case (24 x 48 and 16 x 32 spacings, jitter 3 %), displaced
  with a homogeneous biaxial background plus a finite-width tangential slip
  band, and measured through the unmodified production pipeline;
* the same strain field is also measured with the pre-R2 estimator (baseline);
* the analytic prediction of the pre-R2 bias,
  ``theta_reported = atan((W/H) tan theta_true)``, is tabulated alongside.

Outputs ``artifacts/biaxial_shear_band/r2_estimator_audit.json`` and
``artifacts/biaxial_shear_band/r2_estimator_audit.png``.
Run: ``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r2_estimator_audit.py``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

from tensordem.biaxial_band_diagnostics import (  # noqa: E402
    aspect_ratio_distortion_deg,
    estimator_bias_sweep,
)

SPACING = 1.5
JITTER = 0.03
WALL_MARGIN_DIAMETERS = 2.0
SLIP_FRACTION = 0.04          # band slip as a fraction of the specimen height
BAND_WIDTH_DIAMETERS = 3.0


def lattice(columns: int, rows: int, *, seed: int = 7) -> np.ndarray:
    """Jittered triangular disc lattice, identical to the case specimen."""
    row_spacing = SPACING * math.sqrt(3.0) / 2.0
    width = (columns - 1) * SPACING
    generator = torch.Generator().manual_seed(seed)
    points = []
    for row in range(rows):
        offset = 0.5 * SPACING if row % 2 else 0.0
        for column in range(columns):
            x = -0.5 * width + column * SPACING + offset
            if abs(x) > 0.5 * width + 1e-9:
                continue
            points.append((x, -0.5 * (rows - 1) * row_spacing + row * row_spacing))
    pos = torch.tensor(points, dtype=torch.float64)
    return (pos + (JITTER * SPACING) * torch.randn(pos.shape, dtype=torch.float64,
                                                   generator=generator)).numpy()


def window_of(pos: np.ndarray) -> tuple[float, float, float, float]:
    margin = WALL_MARGIN_DIAMETERS * 2.0 * 0.75
    return (float(pos[:, 0].min()) + margin, float(pos[:, 0].max()) - margin,
            float(pos[:, 1].min()) + margin, float(pos[:, 1].max()) - margin)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/r2_estimator_audit.json")
    parser.add_argument("--figure", default="artifacts/biaxial_shear_band/r2_estimator_audit.png")
    args = parser.parse_args(argv)

    angles = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    payload: dict = {
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "stage": "R2 estimator audit",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "angles_deg": angles,
        "settings": {"spacing": SPACING, "jitter": JITTER,
                     "wall_margin_diameters": WALL_MARGIN_DIAMETERS,
                     "slip_fraction_of_height": SLIP_FRACTION,
                     "band_width_diameters": BAND_WIDTH_DIAMETERS},
        "specimens": {},
    }

    for columns, rows in ((24, 48), (16, 32)):
        pos = lattice(columns, rows)
        window = window_of(pos)
        inside = ((pos[:, 0] >= window[0]) & (pos[:, 0] <= window[1])
                  & (pos[:, 1] >= window[2]) & (pos[:, 1] <= window[3]))
        window_points = pos[inside]
        width = float(np.ptp(window_points[:, 0]))
        height = float(np.ptp(window_points[:, 1]))
        sweep = estimator_bias_sweep(
            pos, angles_deg=angles, measurement_window=window,
            slip=SLIP_FRACTION * float(np.ptp(pos[:, 1])),
            band_width=BAND_WIDTH_DIAMETERS * 2.0 * 0.75)
        sweep["prediction_check_deg"] = {
            "predicted": [float(v) for v in
                          aspect_ratio_distortion_deg(angles, width=width, height=height)],
            "legacy_measured": [row["legacy_recovered_deg"] for row in sweep["rows"]],
            "max_abs_deviation": float(np.max(np.abs(
                np.asarray([row["legacy_recovered_deg"] for row in sweep["rows"]])
                - aspect_ratio_distortion_deg(angles, width=width, height=height)))),
        }
        sweep["bounds"] = window
        payload["specimens"][f"{columns}x{rows}"] = sweep
        print(f"[{columns}x{rows}] aspect {sweep['aspect_ratio']:.3f} "
              f"fixed max|err| {sweep['max_abs_error_deg']:.2f} deg, "
              f"legacy max|err| {sweep['legacy_max_abs_error_deg']:.2f} deg, "
              f"prediction residual {sweep['prediction_check_deg']['max_abs_deviation']:.2f} deg",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    print(f"[written] {args.out}")

    _figure(payload, args.figure)
    return 0


def _figure(payload: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    colours = {"24x48": "tab:blue", "16x32": "tab:green"}
    for name, sweep in payload["specimens"].items():
        angles = np.asarray(sweep["angles_deg"])
        axes[0].plot(angles, [row["recovered_deg"] for row in sweep["rows"]], "o-",
                     color=colours[name], label=f"R2 estimator, {name}")
        axes[0].plot(angles, [row["legacy_recovered_deg"] for row in sweep["rows"]], "s--",
                     color=colours[name], alpha=0.55,
                     label=f"pre-R2 estimator, {name}")
        pred = np.asarray(sweep["prediction_check_deg"]["predicted"])
        axes[0].plot(angles, pred, ":", color="k", lw=1.2)
    axes[0].plot([15, 80], [15, 80], "-", color="0.6", lw=1.0, label="1:1")
    axes[0].plot([], [], ":", color="k", lw=1.2, label=r"$\arctan((W/H)\tan\theta)$")
    axes[0].set_xlabel("injected band inclination [deg from $\\sigma_3$]")
    axes[0].set_ylabel("recovered inclination [deg]")
    axes[0].set_title("(a) estimator recovery on the case lattice")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for name, sweep in payload["specimens"].items():
        angles = np.asarray(sweep["angles_deg"])
        axes[1].plot(angles, [row["legacy_error_deg"] for row in sweep["rows"]], "s--",
                     color=colours[name], alpha=0.6, label=f"pre-R2 bias, {name}")
        axes[1].plot(angles, [row["error_deg"] for row in sweep["rows"]], "o-",
                     color=colours[name], label=f"R2 bias, {name}")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel("injected band inclination [deg]")
    axes[1].set_ylabel("recovered $-$ injected [deg]")
    axes[1].set_title("(b) bias before / after the fix")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[figure] {path}")


if __name__ == "__main__":
    raise SystemExit(main())