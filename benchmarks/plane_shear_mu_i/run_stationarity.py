"""CASE-DEM-PLANE-SHEAR-MUIDI round 5: strain-stationarity audit of the sweep.

Motivation
----------
Rounds 1-3 (docs/CASE_DEM_PLANE_SHEAR_MUIDI{,_R2,_R3}.md) judged ``mu(I)`` on
the tail average of a shear phase truncated at ``shear_strain = 0.2`` (with a
40 % warm-up, i.e. the reading window spans only ``0.08 < strain < 0.2``).
Round 3 (m16497) then found the *thick* bed arm reproduces ``phi(I)`` within
~1 % but reports ``mu = 0.20-0.28`` at ``I = 1e-2..3e-2`` -- below the
quasi-static limit ``mu_s = 0.36`` of the very rheology being verified.  A
friction coefficient below its own I -> 0 limit is not a physics result; it is
a symptom of a *non-stationary* reading (the bed is still in its start-up
transient and the platen has not yet transmitted a steady shear stress).

This script isolates that question with a dedicated experiment-design axis:

    strain ladder   -- the same shear point re-run to total strains
                       {0.2 (round-3 protocol), 1, 2, 4, 8} with the warm-up
                       fraction of the code held fixed, so every rung of the
                       ladder is a self-consistent measurement whose ONLY
                       difference is how long the bed has been sheared.

Three quantities are recorded per run:

* ``mu_tail``    -- the round-3 style reading (mean of per-sample ``tau/P``,
                    tail 60 % of the sampled window);
* ``mu_blocks``  -- block averages over fixed strain bins, used to measure
                    whether the reading has levelled off (``drift_pct``);
* ``mu_ratio``   -- ``<tau>/<P>`` on the same window, i.e. how much of the
                    reading is the mean-of-ratios vs ratio-of-means choice.

No physics, contact law, reference constants or device geometry are changed:
the strain ladder is a pure measurement-protocol control on the same
``tensordem.mu_i_rheology.simulate_servo_shear`` device as round 3.

Usage (one strain-ladder point; the sweep driver fans these out)::

    python run_stationarity.py --I 3e-2 --strain 4.0 --arm thick

Output: ``runs/stationarity/<arm>__I<I>__s<strain>.json``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "stationarity"
OUT.mkdir(parents=True, exist_ok=True)

# --- round-3 device configuration (unchanged) ------------------------------
P = 6.25e4          # Pa, servo-held normal stress
KN = 6.25e4         # N/m, P/kn = 1
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
R = 0.005
RHO = 2500.0
W = 0.12
SEED = 5279
SETTLE_STEPS_BASE = 2500

# reference mu(I) of the ledger acceptance criterion
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_C, PHI_J = 0.58, 0.64

ARMS = {
    "baseline": {"kn": KN, "box_depth": 0.05, "fill_height": 0.06},
    "thick": {"kn": KN, "box_depth": 0.08, "fill_height": 0.09},
    "stiff": {"kn": 6.25e5, "box_depth": 0.05, "fill_height": 0.06},
}


def ref_mu(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def ref_phi(I: float) -> float:
    return PHI_C + (PHI_J - PHI_C) / (I_0 / I + 1.0)


def block_stats(series: list[dict], bin_width: float) -> list[dict]:
    """Averages of the shear-phase samples inside fixed strain bins."""
    if not series:
        return []
    strain = np.array([s["strain"] for s in series], dtype=float)
    mu = np.array([s["mu"] for s in series], dtype=float)
    tau = np.array([s["shear_stress_pa"] for s in series], dtype=float)
    pn = np.array([s["normal_stress_pa"] for s in series], dtype=float)
    pz = np.array([s["plate_z"] for s in series], dtype=float)
    s0 = float(strain.min())
    bins = np.floor((strain - s0) / bin_width).astype(int)
    rows = []
    for b in sorted(set(bins.tolist())):
        sel = bins == b
        rows.append({
            "strain_lo": float(s0 + b * bin_width),
            "strain_hi": float(s0 + (b + 1) * bin_width),
            "n": int(sel.sum()),
            "mu": float(np.nanmean(mu[sel])),
            "mu_ratio": float(np.nanmean(tau[sel]) / np.nanmean(pn[sel])),
            "tau_pa": float(np.nanmean(tau[sel])),
            "p_pa": float(np.nanmean(pn[sel])),
            "plate_z": float(np.nanmean(pz[sel])),
        })
    return rows


def run_ladder_point(I: float, strain: float, arm: str = "thick",
                     bin_width: float = 0.5, sample_stride: int = 10,
                     keep_series: int = 1500,
                     settle_steps: int | None = None) -> dict:
    cfg = ARMS[arm]
    kn = float(cfg["kn"])
    scale = math.sqrt(kn / KN)
    if settle_steps is None:
        settle_steps = int(round(SETTLE_STEPS_BASE * scale))
    case = ServoShearCase(inertial_number=I, mu=MU_CONTACT,
                          mu_rolling=MU_ROLLING, restitution=E, kn=kn,
                          normal_stress=P, shear_strain=strain,
                          settle_steps=settle_steps,
                          min_shear_steps=1,          # strain target is exact
                          sample_stride=sample_stride,
                          packing_gap=0.02, jitter=0.01, seed=SEED,
                          radius=R, density=RHO, box_width=W,
                          box_depth=float(cfg["box_depth"]),
                          fill_height=float(cfg["fill_height"]))
    t0 = time.perf_counter()
    res = simulate_servo_shear(case, return_series=True)
    wall = time.perf_counter() - t0
    series = res["series"]
    blocks = block_stats(series, bin_width)

    strain_arr = np.array([s["strain"] for s in series], dtype=float)
    n_tail = max(1, int(0.6 * len(series)))
    tail = series[-n_tail:]
    mu_tail = float(np.nanmean([s["mu"] for s in tail]))
    mu_ratio = float(np.nanmean([s["shear_stress_pa"] for s in tail])
                     / np.nanmean([s["normal_stress_pa"] for s in tail]))
    # drift: relative difference of the first and last block averages
    drift = (float("nan") if len(blocks) < 2
             else 100.0 * (blocks[-1]["mu"] / blocks[0]["mu"] - 1.0))
    half = len(series) // 2
    mu_first_half = float(np.nanmean([s["mu"] for s in series[:half]]))
    mu_second_half = float(np.nanmean([s["mu"] for s in series[half:]]))
    half_drift = 100.0 * (mu_second_half / mu_first_half - 1.0)

    # compact series for figures (uniform downsample, keep first/last)
    if len(series) > keep_series:
        idx = np.unique(np.linspace(0, len(series) - 1, keep_series).astype(int))
    else:
        idx = np.arange(len(series))
    compact = [{"strain": series[i]["strain"], "mu": series[i]["mu"],
                "p_pa": series[i]["normal_stress_pa"],
                "tau_pa": series[i]["shear_stress_pa"],
                "plate_z": series[i]["plate_z"]} for i in idx]

    i_meas = res["inertial_number"]
    v_part = 4.0 / 3.0 * math.pi * R ** 3
    phi_steady = res["phi_bed_steady"]
    return {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 5,
        "experiment": "strain_stationarity_ladder",
        "arm": arm,
        "I_target": I,
        "strain_target": strain,
        "strain_sampled_lo": float(strain_arr.min()),
        "strain_sampled_hi": float(strain_arr.max()),
        "I_measured": i_meas,
        "mu_tail": mu_tail,
        "mu_ratio_of_means": mu_ratio,
        "mu_ratio_bias_pct": 100.0 * (mu_tail / mu_ratio - 1.0),
        "mu_blocks": blocks,
        "block_bin_width": bin_width,
        "mu_block_first": blocks[0]["mu"] if blocks else float("nan"),
        "mu_block_last": blocks[-1]["mu"] if blocks else float("nan"),
        "mu_block_drift_pct": drift,
        "mu_half_drift_pct": half_drift,
        "phi_bed_steady": phi_steady,
        "phi_bed_settled": res["phi_bed_settled"],
        "plate_height_settled": res["plate_height_settled"],
        "plate_height_steady": res["plate_height_steady"],
        "dilatancy": res["dilatancy"],
        "coordination": res["coordination_number"],
        "normal_stress_measured_pa": res["normal_stress_measured_pa"],
        "normal_stress_applied_pa": res["normal_stress_applied_pa"],
        "pressure_control_error": (res["normal_stress_measured_pa"]
                                   / res["normal_stress_applied_pa"] - 1.0),
        "shear_stress_pa": res["shear_stress_pa"],
        "shear_rate_imposed": res["shear_rate_imposed"],
        "n_particles": res["n_particles"],
        "settle_steps": res["settle_steps"],
        "shear_steps": res["shear_steps"],
        "dt": res["dt"],
        "plate_travel_in_d": (res["shear_rate_imposed"] * res["dt"]
                              * res["shear_steps"] * res["plate_height_settled"]
                              / (2.0 * R)),
        "n_samples": res["n_samples"],
        "wall_clock_s": wall,
        "ref_mu": ref_mu(i_meas),
        "ref_phi": ref_phi(i_meas),
        "mu_err_pct": 100.0 * (mu_tail / ref_mu(i_meas) - 1.0),
        "phi_err_pct": 100.0 * (phi_steady / ref_phi(i_meas) - 1.0),
        "series": compact,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=float, required=True)
    ap.add_argument("--strain", type=float, required=True)
    ap.add_argument("--arm", choices=tuple(ARMS), default="thick")
    ap.add_argument("--bin-width", type=float, default=0.5)
    ap.add_argument("--sample-stride", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rec = run_ladder_point(args.I, args.strain, arm=args.arm,
                           bin_width=args.bin_width,
                           sample_stride=args.sample_stride)
    tag = f"{args.arm}__I{args.I:g}__s{args.strain:g}"
    path = Path(args.out) if args.out else OUT / f"{tag}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(f"[{tag}] I={rec['I_measured']:.4g} mu={rec['mu_tail']:.4f} "
          f"(ref {rec['ref_mu']:.4f}, {rec['mu_err_pct']:+.1f}%) "
          f"phi={rec['phi_bed_steady']:.4f} (ref {rec['ref_phi']:.4f}, "
          f"{rec['phi_err_pct']:+.1f}%) drift={rec['mu_block_drift_pct']:+.2f}% "
          f"ratio_bias={rec['mu_ratio_bias_pct']:+.2f}% "
          f"travel={rec['plate_travel_in_d']:.1f}d wc={rec['wall_clock_s']:.0f}s",
          flush=True)
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()