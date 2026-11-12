#!/usr/bin/env python
"""CASE-ROTATING-DRUM: rotating-drum mixing / resting benchmark (m15329).

Physical case
-------------
A quasi-2-D rotating drum (circular shell, radius ``R``) holds a *well-mixed*
bidisperse bed (small + large disks, equal density).  The protocol has two
phases, mirroring the classical rotary-drum mixing experiment:

* **resting**  -- the shell is at rest and the bed settles under gravity until
  the rms particle speed drops below a threshold (the bed is at rest);
* **mixing**   -- the shell rotates at angular velocity ``omega`` for ``revs``
  revolutions; the cascade/avalanche flow advects the particles and the
  Brazil-nut (percolation) mechanism segregates the coarse species to the
  periphery.

The control parameter is the shell **Froude number**

    Fr = omega^2 R / g ,

the ratio of centrifugal to gravitational acceleration.  Two exact limits from
the Froude-number literature anchor the comparison:

* the **release (detachment) angle** of a particle on the bare shell,
  ``theta_d = arccos(-Fr)`` measured from the 6-o'clock position (Haff 1983);
* the **flow-regime bands** (slipping / rolling / cataracting / centrifuging).

The kernel is the validated ``tensordem.drum_segregation`` 2-D soft-sphere DEM
(p176w2-n08 verification surface); nothing here re-implements the physics.

Usage
-----
    PYTHONPATH=src python benchmarks/rotating_drum_mixing_resting.py \
        --out runs/m15329/cases --jobs 8

    # single case
    PYTHONPATH=src python benchmarks/rotating_drum_mixing_resting.py \
        --one 0.5 --revs 6 --out runs/m15329/cases
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np                                              # noqa: E402

from tensordem.drum_segregation import metrics as _m            # noqa: E402
from tensordem.drum_segregation.kernel2d import DrumConfig, simulate  # noqa: E402

# ---------------------------------------------------------------------------
# reference anchors (Froude-number literature)
# ---------------------------------------------------------------------------
# Release angle of a particle on the bare shell: the normal force
#   N = m (omega^2 R + g cos theta)
# vanishes at theta_d = arccos(-Fr).  Exact; see Haff (1983) JFM 132:401,
# and the drum_flow.regimes module docstring.
def ref_release_angle_deg(fr: float) -> float:
    if fr >= 1.0:
        return 180.0
    return math.degrees(math.acos(-float(fr)))


# Froude bands (slipping / rolling / cataracting / centrifuging) as used by
# the drum_flow phase map (IN32-D01).
FR_BANDS = ((0.0, 0.3, "slipping"), (0.3, 0.7, "rolling"),
            (0.7, 1.0, "cataracting"), (1.0, math.inf, "centrifuging"))


def ref_regime_label(fr: float) -> str:
    for lo, hi, name in FR_BANDS:
        if lo <= fr < hi:
            return name
    return "centrifuging"


def omega_for_froude(fr: float, radius: float, gravity: float = 9.81) -> float:
    return math.sqrt(float(fr) * float(gravity) / float(radius))


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------
def build_cfg(fr: float, revs: float, radius: float = 0.15,
              d_small: float = 0.010, size_ratio: float = 2.0,
              fill: float = 0.30, large_area_fraction: float = 0.30,
              dt: float = 1.0e-4, seed: int = 0) -> DrumConfig:
    omega = omega_for_froude(fr, radius)
    return DrumConfig(
        radius=radius, fill=fill,
        d_small=d_small, d_large=size_ratio * d_small,
        large_area_fraction=large_area_fraction,
        omega=omega, dt=dt,
        settle_steps=6000, revs=revs, sample_revs=0.5,
        n_bins=8, seed=seed,
    )


def _quasi_steady(series, key: str, frac: float = 0.25) -> float:
    arr = np.asarray(series.get(key, []), float)
    if arr.size == 0:
        return float("nan")
    k = max(1, int(round(frac * arr.size)))
    return float(np.mean(arr[-k:]))


def run_case(fr: float, revs: float, radius: float = 0.15,
             d_small: float = 0.010, size_ratio: float = 2.0,
             fill: float = 0.30, large_area_fraction: float = 0.30,
             dt: float = 1.0e-4, seed: int = 0) -> dict:
    cfg = build_cfg(fr, revs, radius=radius, d_small=d_small,
                    size_ratio=size_ratio, fill=fill,
                    large_area_fraction=large_area_fraction, dt=dt, seed=seed)
    t0 = time.time()
    run = simulate(cfg)
    wall = time.time() - t0

    # --- resting-phase evidence -------------------------------------------
    diag = run.diagnostics
    resting_ok = bool(diag["settle_rms_speed"] < 5.0e-3)

    # --- measured release angle -------------------------------------------
    # angle of the outermost wall-contacting particle above the 6 o'clock
    # position, time-averaged over the last quarter of the run.
    pos = run.pos
    r = run.radii
    is_large = run.is_large
    drum_r = cfg.radius
    # final-snapshot wall-contacting particles (within one particle radius of
    # the shell) and their polar angle from the bottom.
    r_pos = np.linalg.norm(pos, axis=1)
    wall_mask = r_pos > (drum_r - 1.5 * r.max())
    if wall_mask.any():
        ang = np.degrees(np.arctan2(pos[wall_mask, 0], pos[wall_mask, 1]))
        # arctan2(x, y): 0 deg at 6 o'clock (y=-R is -90 in atan2(x,y)? no).
        # Use angle from the bottom: theta = atan2(x, -y) in [0, 180].
        theta = np.degrees(np.arctan2(pos[wall_mask, 0], -pos[wall_mask, 1]))
        theta = np.clip(theta, 0.0, 180.0)
        measured_release = float(np.percentile(theta, 95))
    else:
        measured_release = float("nan")

    # --- segregation metrics (final snapshot) -----------------------------
    r_norm = r_pos / drum_r
    prof = _m.radial_profile(pos, r, is_large, drum_r, cfg.n_bins)
    S = _m.concentration_segregation(prof)
    excess = _m.core_shell_excess(prof)
    corr = _m.size_position_correlation(r, r_norm)
    null = _m.permutation_null(pos, r, is_large, drum_r, cfg.n_bins,
                                 n_perm=200, seed=seed)
    mode = _m.classify_mode(null["excess_observed"], null["excess_z"],
                            null["corr_observed"], null["corr_z"])

    # --- mixing index (concentration homogeneity) -------------------------
    # M = 1 - S / S_max ; S_max is the fully-segregated bound (all large
    # species in the outer half).  M = 0 unmixed->segregated, M = 1 mixed.
    # We report S directly and the quasi-steady value.
    S_steady = _quasi_steady(run.series, "segregation")
    excess_steady = _quasi_steady(run.series, "core_shell_excess")
    corr_steady = _quasi_steady(run.series, "size_pos_corr")

    # --- reference comparison ---------------------------------------------
    ref_theta = ref_release_angle_deg(fr)
    ref_regime = ref_regime_label(fr)
    theta_err = (abs(measured_release - ref_theta) / ref_theta
                 if ref_theta > 0 and math.isfinite(measured_release)
                 else float("nan"))

    record = {
        "case": f"fr{fr:g}_rev{revs:g}_s{seed}",
        "fr": float(fr),
        "omega": float(cfg.omega),
        "radius": float(radius),
        "d_small": float(d_small),
        "d_large": float(cfg.d_large),
        "size_ratio": float(size_ratio),
        "fill": float(fill),
        "large_area_fraction": float(large_area_fraction),
        "n_large": int(run.n_large),
        "n_small": int(run.n_small),
        "n_total": int(run.n_large + run.n_small),
        "actual_fill": float(run.actual_fill),
        "actual_large_area_fraction": float(run.actual_large_area_fraction),
        "dt": float(dt),
        "steps": int(diag["steps"]),
        "physical_time_s": float(diag["physical_time"]),
        "wall_time_s": round(wall, 1),
        # resting phase
        "settle_rms_speed": float(diag["settle_rms_speed"]),
        "settle_steps_used": int(diag["settle_steps_used"]),
        "quench_rounds": int(diag["quench_rounds"]),
        "resting_ok": resting_ok,
        # mixing phase
        "S_final": float(S),
        "S_steady": S_steady,
        "excess_final": float(excess),
        "excess_steady": excess_steady,
        "corr_final": float(corr),
        "corr_steady": corr_steady,
        "excess_z": float(null["excess_z"]),
        "corr_z": float(null["corr_z"]),
        "mode": mode,
        "mean_r_large_over_R": float(np.linalg.norm(pos[is_large], axis=1).mean() / drum_r),
        "mean_r_small_over_R": float(np.linalg.norm(pos[~is_large], axis=1).mean() / drum_r),
        # reference comparison
        "ref_release_angle_deg": ref_theta,
        "measured_release_angle_deg": measured_release,
        "release_angle_rel_err": theta_err,
        "ref_regime": ref_regime,
        "froude": float(cfg.froude),
        # series (for figures)
        "series": {k: [float(v) for v in run.series[k]]
                   for k in ("rev", "segregation", "core_shell_excess",
                             "size_pos_corr")},
        "final_pos": pos.tolist(),
        "final_is_large": is_large.tolist(),
        "final_radii": r.tolist(),
    }
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr", type=float, nargs="+", default=[0.5])
    ap.add_argument("--revs", type=float, default=6.0)
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--d-small", type=float, default=0.010)
    ap.add_argument("--size-ratio", type=float, default=2.0)
    ap.add_argument("--fill", type=float, default=0.30)
    ap.add_argument("--large-frac", type=float, default=0.30)
    ap.add_argument("--dt", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/m15329/cases")
    ap.add_argument("--one", type=float, default=None,
                    help="run a single Fr and print JSON")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frs = [args.one] if args.one is not None else args.fr
    records = []
    for fr in frs:
        rec = run_case(fr, args.revs, radius=args.radius, d_small=args.d_small,
                       size_ratio=args.size_ratio, fill=args.fill,
                       large_area_fraction=args.large_frac, dt=args.dt,
                       seed=args.seed)
        records.append(rec)
        if args.one is not None:
            slim = {k: v for k, v in rec.items()
                    if k not in ("final_pos", "final_is_large", "final_radii")}
            print(json.dumps(slim))
            return rec
        # strip the big arrays from the per-case file (kept in memory for merge)
        slim = {k: v for k, v in rec.items()
                if k not in ("final_pos", "final_is_large", "final_radii")}
        (out / f"case_{rec['case']}.json").write_text(json.dumps(slim, indent=1))
        print(f"[drum-mix] {rec['case']}: S={rec['S_final']:.4f} "
              f"excess={rec['excess_final']:+.4f} mode={rec['mode']} "
              f"theta_d={rec['measured_release_angle_deg']:.1f} "
              f"(ref {rec['ref_release_angle_deg']:.1f}) "
              f"wall={rec['wall_time_s']}s", flush=True)

    # merge
    merged = {"cases": records,
               "config": {"fr": frs, "revs": args.revs, "radius": args.radius,
                          "d_small": args.d_small, "size_ratio": args.size_ratio,
                          "fill": args.fill, "large_frac": args.large_frac,
                          "dt": args.dt, "seed": args.seed}}
    (out / "merged.json").write_text(json.dumps(merged, indent=1))
    print(f"[drum-mix] wrote {out/'merged.json'}")
    return merged


if __name__ == "__main__":
    main()
