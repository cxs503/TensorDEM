"""R4 SNR check: does a cumulative window beat a one-step increment at fixed noise?

This is the *synthetic* half of R4.  The R3 noise audit already showed that the
Radon focus degrades once the displacement noise exceeds ~0.5 % of the particle
spacing; R3 predicted -- but never measured -- that a cumulative late-window
measurement (one displacement baseline over the whole post-peak window, 6x the
tangential slip of a single 0.01-strain increment at the same positional noise)
moves the case back into the safe branch.  R4 measures that prediction here on
the exact 24 x 48 case lattice, and separately runs the DEM ensemble
(``run_r4_cumulative_ensemble.py``).

Design (predeclared): angles {55, 60, 65} deg, slips {0.0033 m = one 0.01-strain
increment, 0.02 m = cumulative late window}, noise {0, 0.25, 0.5} % of the
spacing, 6 independent noise realisations each; metrics = RMS and signed bias of
the measured inclination.  Decision rule: the cumulative slip must have a lower
RMS error than the one-step slip at every non-zero noise level, at all three
angles.

Outputs ``artifacts/biaxial_shear_band/r4_snr_check.json`` and
``r4_snr_check.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

ANGLES = [55.0, 60.0, 65.0]
SLIPS = {"one_step": 0.0033, "cumulative": 0.02}
NOISE = [0.0, 0.0025, 0.005]
REPEATS = 6
SPACING = 1.5
BAND_WIDTH = 1.5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/r4_snr_check.json")
    parser.add_argument("--figure", default="artifacts/biaxial_shear_band/r4_snr_check.png")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import torch
    torch.set_num_threads(1)
    from tensordem.biaxial_shear_band import _make_specimen, measure_band
    from tensordem.biaxial_band_diagnostics import synthetic_band_configuration
    from tensordem.ridge_dem import RidgeMaterial

    material = RidgeMaterial(freeze_bond_fraction=0.35, particle_radius=0.75)
    assembly, geometry = _make_specimen(
        material, columns=24, rows=48, confining_stress=100.0e3, jitter=0.03,
        servo_gain=3.0e-4, servo_velocity_limit=2.0, wall_friction=0.55)
    pos = assembly.pos.detach().numpy().copy()
    window = (-0.5 * geometry["width"], 0.5 * geometry["width"],
              -0.5 * geometry["height"], 0.5 * geometry["height"])
    rng = np.random.default_rng(20261113)

    rows = []
    started = time.time()
    for label, slip in SLIPS.items():
        for angle in ANGLES:
            clean = synthetic_band_configuration(pos, angle_deg=angle, slip=slip,
                                                 band_width=BAND_WIDTH)
            increment = clean - pos
            for noise in NOISE:
                for repeat in range(REPEATS):
                    jitter = (rng.standard_normal(pos.shape) * noise * SPACING
                              if noise else np.zeros_like(pos))
                    now = pos + increment + jitter
                    theta = float(measure_band(pos, now, bounds=window)
                                  ["folded_angle_deg"])
                    rows.append({"window": label, "slip_m": slip,
                                 "angle_true_deg": angle, "noise_fraction": noise,
                                 "repeat": repeat, "theta_deg": theta,
                                 "error_deg": theta - angle})
            print(f"[done] {label} angle {angle:.0f}", flush=True)

    summary = {}
    for label in SLIPS:
        summary[label] = {}
        for noise in NOISE:
            errs = np.array([r["error_deg"] for r in rows if r["window"] == label
                             and r["noise_fraction"] == noise])
            per_angle = {}
            for angle in ANGLES:
                e = np.array([r["error_deg"] for r in rows if r["window"] == label
                              and r["noise_fraction"] == noise
                              and r["angle_true_deg"] == angle])
                per_angle[f"{angle:g}"] = {"rms_deg": float(np.sqrt(np.mean(e ** 2))),
                                           "bias_deg": float(e.mean())}
            summary[label][f"{noise:g}"] = {
                "n": int(errs.size),
                "rms_deg": float(np.sqrt(np.mean(errs ** 2))),
                "bias_deg": float(errs.mean()),
                "max_abs_deg": float(np.max(np.abs(errs))),
                "per_angle": per_angle,
            }

    # decision rule: cumulative RMS < one_step RMS at every non-zero noise level,
    # at every angle
    checks = []
    for noise in NOISE:
        if noise == 0.0:
            continue
        for angle in ANGLES:
            a = summary["one_step"][f"{noise:g}"]["per_angle"][f"{angle:g}"]["rms_deg"]
            b = summary["cumulative"][f"{noise:g}"]["per_angle"][f"{angle:g}"]["rms_deg"]
            checks.append({"noise_fraction": noise, "angle_true_deg": angle,
                           "rms_one_step_deg": a, "rms_cumulative_deg": b,
                           "holds": bool(b < a)})
    payload = {
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "round": "R4",
        "artefact": "synthetic SNR check of the cumulative-window measurement",
        "lattice": {"columns": 24, "rows": 48, "particles": int(pos.shape[0]),
                    "spacing_m": SPACING,
                    "width_m": geometry["width"], "height_m": geometry["height"]},
        "design": {"angles_deg": ANGLES, "slips_m": SLIPS, "noise_fractions": NOISE,
                   "repeats": REPEATS, "band_width_m": BAND_WIDTH,
                   "window": window, "seed": 20261113},
        "summary": summary,
        "decision_rule": ("cumulative RMS < one-step RMS at every non-zero noise "
                          "level and every angle"),
        "checks": checks,
        "decision_holds": bool(all(c["holds"] for c in checks)),
        "wall_time_s": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for label, style in (("one_step", "s--"), ("cumulative", "o-")):
        ax.plot([100 * n for n in NOISE],
                [summary[label][f"{n:g}"]["rms_deg"] for n in NOISE], style,
                label=f"{label} (slip {SLIPS[label]} m)")
    ax.set_xlabel("displacement noise (% of particle spacing)")
    ax.set_ylabel("RMS inclination error (deg)")
    ax.set_title("R4 synthetic SNR check on the 24×48 case lattice\n"
                 "(known-angle bands, 3 angles × 6 noise realisations)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=140)
    plt.close(fig)
    print(json.dumps({"decision_holds": payload["decision_holds"],
                      "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())