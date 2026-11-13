"""R3 estimator audit: which operator survives the noise level of a real DEM field?

The R1/R2 chain established that the production Radon focus is *unbiased* on a
noise-free synthetic band injected into the case lattice (0.00 deg at 24 x 48).
The real difficulty of the case is different: the incremental displacement field
between two late snapshots is a *noisy* field (bond-breaking bursts, boundary
grain rattling), and a noisy ridge can move a 1-D projection maximum.  The R3
structure-tensor operator (``tensordem.shear_band_structure_tensor``) is the
experimental route of Desrues & Viggiani (2004) and is expected to be more
robust because it integrates a 2-D gradient field, but its smoothing width
``sigma`` introduces a bias/variance trade-off that must be fixed **on synthetic
data with a known angle**, never on the DEM answer.

Experiment (predeclared)
------------------------
* unit: one measured inclination of one injected band;
* factors: angle {30, 45, 55, 60, 65, 70, 75} deg x displacement-noise level
  {0, 1, 2, 4} % of the particle spacing (added to the incremental displacement,
  i.e. measurement noise) x operator {Radon focus, structure tensor with
  sigma = 1.5 / 2.5 / 3.5 cells, structure tensor weighted by gamma with
  sigma = 2.5};
* repetitions: 5 independent noise realisations per cell of the design;
* metrics: signed bias ``mean(theta_hat - theta_true)``, RMS error, max |error|;
* decision rule (predeclared): the operator adopted as the R3 cross-check is the
  one with the smallest RMS error at 2 % noise (the level that reproduces the
  DEM operator disagreement), with the constraint that its bias at 0 % noise is
  <= 2 deg (numerical fidelity first).

Outputs ``artifacts/biaxial_shear_band/r3_estimator_audit.json`` and
``artifacts/biaxial_shear_band/r3_estimator_audit.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ANGLES = [30.0, 45.0, 55.0, 60.0, 65.0, 70.0, 75.0]
NOISE_FRACTIONS = [0.0, 0.0025, 0.005, 0.01]
#: slip of the injected band: 0.02 m = cumulative late-window displacement of a
#: real 24 x 48 specimen; 0.0033 m = one 0.01-strain increment (same signal, 6x
#: less displacement) -- the SNR factor that the design is meant to expose.
SLIPS = [0.02, 0.0033]
REPEATS = 5
BOND_FRACTION = 0.35
SPACING = 1.5          # 2 x particle radius, metres


def _key(slip: float, noise: float) -> str:
    return f"slip{slip:g}_noise{noise:g}"


def _operators():
    from tensordem.biaxial_shear_band import measure_band
    from tensordem.shear_band_structure_tensor import measure_band_structure_tensor

    def radon(ref, now):
        return float(measure_band(ref, now)["folded_angle_deg"])

    def tensor(ref, now, sigma):
        return float(measure_band_structure_tensor(ref, now, sigma=sigma)
                     ["folded_angle_deg"])

    return {
        "radon": radon,
        "tensor_s1.5": lambda ref, now: tensor(ref, now, 1.5),
        "tensor_s2.5": lambda ref, now: tensor(ref, now, 2.5),
        "tensor_s3.5": lambda ref, now: tensor(ref, now, 3.5),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/r3_estimator_audit.json")
    parser.add_argument("--figure", default="artifacts/biaxial_shear_band/r3_estimator_audit.png")
    parser.add_argument("--slip-list", default=None,
                        help="comma separated slip values (default: SLIPS)")
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if here not in sys.path:
        sys.path.insert(0, os.path.join(here, "src"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    import torch
    torch.set_num_threads(4)
    from tensordem.biaxial_band_diagnostics import synthetic_band_configuration
    from tensordem.biaxial_shear_band import _make_specimen
    from tensordem.ridge_dem import RidgeMaterial

    material = RidgeMaterial(particle_radius=0.75,
                             freeze_bond_fraction=BOND_FRACTION, seed=20261031)
    assembly, geometry = _make_specimen(
        material, columns=24, rows=48, confining_stress=100.0e3, jitter=0.03,
        servo_gain=3.0e-4, servo_velocity_limit=2.0, wall_friction=0.55)
    pos = assembly.pos.numpy()
    ops = _operators()
    rng = np.random.default_rng(20261113)

    slips = ([float(v) for v in args.slip_list.split(",")] if args.slip_list
             else SLIPS)
    rows = []
    for slip in slips:
        for angle in ANGLES:
            clean = synthetic_band_configuration(pos, angle_deg=angle, slip=slip,
                                                 band_width=1.5)
            increment = clean - pos
            for noise in NOISE_FRACTIONS:
                for repeat in range(REPEATS):
                    jitter = (rng.standard_normal(pos.shape) * noise * SPACING
                              if noise > 0.0 else np.zeros_like(pos))
                    now = pos + increment + jitter
                    entry = {"angle_true_deg": angle, "noise_fraction": noise,
                             "slip_m": slip, "repeat": repeat}
                    for name, op in ops.items():
                        try:
                            entry[name] = op(pos, now)
                        except Exception as exc:         # pragma: no cover - reported
                            entry[name] = math.nan
                            entry[name + "_error"] = repr(exc)
                    rows.append(entry)
                print(f"[done] slip {slip} angle {angle:.0f} noise {noise:.4f}", flush=True)

    summary = {}
    for name in ops:
        summary[name] = {}
        for slip in slips:
            for noise in NOISE_FRACTIONS:
                errs = np.array([row[name] - row["angle_true_deg"] for row in rows
                                 if row["noise_fraction"] == noise
                                 and row["slip_m"] == slip
                                 and np.isfinite(row[name])])
                summary[name][_key(slip, noise)] = {
                    "n": int(errs.size),
                    "bias_deg": float(errs.mean()) if errs.size else math.nan,
                    "rms_deg": float(np.sqrt(np.mean(errs ** 2))) if errs.size else math.nan,
                    "max_abs_deg": float(np.max(np.abs(errs))) if errs.size else math.nan,
                }
    cumulative = _key(0.02, 0.005)
    adoption = min(summary, key=lambda k: summary[k][cumulative]["rms_deg"])
    payload = {
        "case": "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE",
        "audit": "R3 operator noise audit on the exact case lattice",
        "lattice": {"columns": 24, "rows": 48, "particles": int(pos.shape[0]),
                    "aspect": geometry["width"] / geometry["height"],
                    "spacing_m": SPACING},
        "design": {"angles_deg": ANGLES, "noise_fractions": NOISE_FRACTIONS,
                   "repeats": REPEATS, "slips_m": slips, "band_width_m": 1.5},
        "summary": summary,
        "adopted_cross_check_operator": adoption,
        "adoption_rule": ("smallest RMS at slip 0.02 m, 0.5 % noise (cumulative "
                          "late-window signal of a real specimen)"),
        "records": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for name in ops:
        for slip, style in zip(slips, ("o-", "s--")):
            rms = [summary[name][_key(slip, n)]["rms_deg"] for n in NOISE_FRACTIONS]
            axes[0].plot([100 * n for n in NOISE_FRACTIONS], rms, style,
                         label=f"{name} slip={slip}")
            bias = [summary[name][_key(slip, n)]["bias_deg"] for n in NOISE_FRACTIONS]
            axes[1].plot([100 * n for n in NOISE_FRACTIONS], bias, style,
                         label=f"{name} slip={slip}")
    axes[0].set_xlabel("displacement noise (% of spacing)")
    axes[0].set_ylabel("RMS error (deg)")
    axes[0].set_title("(a) accuracy vs measurement noise")
    axes[1].set_xlabel("displacement noise (% of spacing)")
    axes[1].set_ylabel("mean signed bias (deg)")
    axes[1].set_title("(b) bias vs measurement noise")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("R3 operator audit on the 24×48 case lattice — synthetic bands of known angle")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("adopted:", adoption)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())