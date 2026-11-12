#!/usr/bin/env python
"""CASE-ROTATING-DRUM round 2: Froude-number validation of the 2-D drum DEM.

This benchmark replaces the round-1 (m15329) evidence for this case.  It is
built around *measured* observables only:

* the **wall-contact-arc upper end** -- the angular position (from 6 o'clock,
  positive in the direction of rotation) where the bed leaves the shell.  The
  reference is the exact release angle of a particle carried by the shell
  ``theta_d = arccos(-Fr)`` (Haff 1983, JFM 132:401-430; see
  ``tensordem.drum_flow.regimes.detachment_angle_deg``);
* the **centrifuging onset** ``Fr_c = 1`` (exact: the normal force at the top
  of the shell vanishes for Fr = 1);
* the **kinematic regime observables** (wall layer speed ratio, ballistic
  fraction above the drum centre, 5th percentile of r/R) that back the
  qualitative regime labels of the classical Froude-number diagram
  (Mellmann 2001, Powder Technol. 118:251-270);
* the **segregation metrics** of ``tensordem.drum_segregation.metrics`` with
  their permutation null (a ``mixed`` verdict is reported as such).

All acceptance thresholds are frozen in ``prereg/CASE_ROTATING_DRUM_R2.json``.

Usage
-----
    PYTHONPATH=src python benchmarks/rotating_drum_froude_r2.py --subset main \
        --out artifacts/case_rotating_drum_r2/cases

    # one shard of a cluster fan-out (case i of n)
    PYTHONPATH=src python benchmarks/rotating_drum_froude_r2.py --subset main \
        --shard 2 --nshards 9 --out artifacts/case_rotating_drum_r2/cases
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
from tensordem.drum_flow.regimes import (                       # noqa: E402
    MELLMANN_BANDS, classify_measured, detachment_angle_deg)

# ---------------------------------------------------------------------------
# frozen case tables (mirror prereg/CASE_ROTATING_DRUM_R2.json)
# ---------------------------------------------------------------------------
MAIN_FR = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1)

SENSITIVITY = (
    {"name": "rev20", "fr": 0.1, "revs": 20.0, "d_small": 0.010, "dt": 1.0e-4},
    {"name": "dt_half", "fr": 0.3, "revs": 6.0, "d_small": 0.010, "dt": 5.0e-5},
    {"name": "dt_base", "fr": 0.3, "revs": 6.0, "d_small": 0.010, "dt": 1.0e-4},
)


def omega_for_froude(fr: float, radius: float, gravity: float = 9.81) -> float:
    return math.sqrt(float(fr) * float(gravity) / float(radius))


def literature_label(fr: float) -> str:
    for lo, hi, name in MELLMANN_BANDS:
        if lo <= fr < hi:
            return name
    return "centrifuging"


def _steady(series: dict, key: str, frac: float = 0.25) -> float:
    arr = np.asarray(series.get(key, []), float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(round(frac * arr.size)))
    return float(np.mean(arr[-k:]))


def run_case(name: str, fr: float, revs: float, d_small: float, dt: float,
             radius: float = 0.15, fill: float = 0.30,
             large_fraction: float = 0.30, seed: int = 0) -> dict:
    cfg = DrumConfig(
        radius=radius, fill=fill, d_small=d_small, d_large=2.0 * d_small,
        large_area_fraction=large_fraction,
        omega=omega_for_froude(fr, radius), dt=dt,
        revs=revs, sample_revs=0.25, n_bins=8, seed=seed,
    )
    t0 = time.time()
    run = simulate(cfg)
    wall = time.time() - t0
    diag = run.diagnostics
    series = {k: np.asarray(v, float) for k, v in run.series.items()}
    pos, r, is_large = run.pos, run.radii, run.is_large

    # --- segregation (metrics + permutation null) -------------------------
    prof = _m.radial_profile(pos, r, is_large, radius, cfg.n_bins)
    null = _m.permutation_null(pos, r, is_large, radius, cfg.n_bins,
                               n_perm=200, seed=seed)
    mode = _m.classify_mode(null["excess_observed"], null["excess_z"],
                            null["corr_observed"], null["corr_z"])

    # --- measured kinematics (quasi-steady window = last 25 %) -----------
    obs = {
        "wall_velocity_ratio": _steady(series, "wall_velocity_ratio"),
        "upper_flight_fraction": _steady(series, "upper_flight_fraction"),
        "r_p05_over_R": _steady(series, "r_p05_over_R"),
        "airborne_fraction": _steady(series, "airborne_fraction"),
        "wall_contact_arc_upper_deg": _steady(series, "wall_contact_arc_upper_deg"),
        "bed_omega_over_wall": _steady(series, "bed_omega_over_wall"),
    }
    measured_regime = classify_measured(obs)

    # --- reference comparison --------------------------------------------
    theta_ref = detachment_angle_deg(fr)
    theta_got = obs["wall_contact_arc_upper_deg"]
    if math.isfinite(theta_got) and theta_ref > 0.0:
        theta_err = abs(theta_got - theta_ref) / theta_ref
    else:
        theta_err = float("nan")

    # revolution-wise S for the plateau check (nearest samples)
    def S_at(rev_target: float) -> float:
        rev = series.get("rev")
        val = series.get("segregation")
        if rev is None or val is None or rev.size == 0:
            return float("nan")
        i = int(np.argmin(np.abs(rev - rev_target)))
        return float(val[i])

    record = {
        "case": name,
        "fr": float(fr),
        "omega": float(cfg.omega),
        "radius": float(radius),
        "d_small": float(d_small),
        "d_large": float(cfg.d_large),
        "size_ratio": float(cfg.size_ratio),
        "fill": float(fill),
        "large_area_fraction": float(large_fraction),
        "dt": float(dt),
        "revs": float(revs),
        "steps": int(diag["steps"]),
        "physical_time_s": float(diag["physical_time"]),
        "wall_time_s": round(wall, 1),
        "n_large": int(run.n_large),
        "n_small": int(run.n_small),
        "n_total": int(run.n_large + run.n_small),
        "actual_fill": float(run.actual_fill),
        "actual_large_area_fraction": float(run.actual_large_area_fraction),
        # resting phase (must be measured, not asserted)
        "settle_rms_speed": float(diag["settle_rms_speed"]),
        "settle_speed_threshold": float(diag["settle_speed_threshold"]),
        "resting_ok": bool(diag["resting_ok"]),
        "settle_steps_used": int(diag["settle_steps_used"]),
        "quench_rounds": int(diag["quench_rounds"]),
        # measured kinematics
        "observables": obs,
        "measured_regime": measured_regime,
        "surface_slope_deg": float(diag.get("surface_slope_deg", float("nan"))),
        # reference anchors
        "theta_ref_deg": float(theta_ref),
        "theta_got_deg": float(theta_got),
        "theta_rel_err": float(theta_err),
        "literature_regime": literature_label(fr),
        "fr_centrifuging_onset_ref": 1.0,
        "centrifuging_onset_reached": bool(obs["r_p05_over_R"] >= 0.75),
        # segregation
        "S_final": float(_m.concentration_segregation(prof)),
        "excess_final": float(null["excess_observed"]),
        "corr_final": float(null["corr_observed"]),
        "excess_z": float(null["excess_z"]),
        "corr_z": float(null["corr_z"]),
        "mode": mode,
        "S_steady": _steady(series, "segregation"),
        "S_at_rev5": S_at(5.0),
        "S_at_rev10": S_at(10.0),
        "S_at_rev20": S_at(20.0),
        "mean_r_large_over_R": float(np.linalg.norm(pos[is_large], axis=1).mean() / radius),
        "mean_r_small_over_R": float(np.linalg.norm(pos[~is_large], axis=1).mean() / radius),
        "max_overlap_ratio": float(diag["max_overlap_ratio"]),
        "series": {k: [float(v) for v in series[k]] for k in series},
        "final_pos": pos.tolist(),
        "final_is_large": is_large.tolist(),
        "final_radii": r.tolist(),
    }
    return record


def case_table(subset: str) -> list[dict]:
    cases = []
    if subset in ("main", "all"):
        for fr in MAIN_FR:
            cases.append({"name": f"main_fr{fr:g}", "fr": fr, "revs": 10.0,
                          "d_small": 0.010, "dt": 1.0e-4})
    if subset in ("sens", "all"):
        cases.extend(SENSITIVITY)
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=("main", "sens", "all"), default="main")
    ap.add_argument("--out", default="artifacts/case_rotating_drum_r2/cases")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fr", type=float, nargs="+", default=None,
                    help="override: explicit Fr list (smoke runs)")
    ap.add_argument("--revs", type=float, default=None,
                    help="override revolutions (smoke runs)")
    ap.add_argument("--d-small", type=float, default=None,
                    help="override small diameter (smoke runs)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = case_table(args.subset)
    if args.fr is not None:
        cases = [{"name": f"adhoc_fr{fr:g}", "fr": fr,
                  "revs": args.revs if args.revs is not None else 3.0,
                  "d_small": args.d_small if args.d_small is not None else 0.010,
                  "dt": 1.0e-4} for fr in args.fr]
    if args.nshards > 1:
        cases = [c for i, c in enumerate(cases) if i % args.nshards == args.shard]

    for spec in cases:
        rec = run_case(spec["name"], spec["fr"], spec["revs"],
                       spec["d_small"], spec["dt"], seed=args.seed)
        slim = {k: v for k, v in rec.items()
                if k not in ("final_pos", "final_is_large", "final_radii")}
        name = f"case_{spec['name']}.json"
        (out / name).write_text(json.dumps(slim, indent=1))
        # full snapshot (positions) in a separate file for metric recomputation
        full = {k: rec[k] for k in ("case", "fr", "n_total",
                                    "final_pos", "final_is_large", "final_radii")}
        (out / f"pos_{spec['name']}.json").write_text(json.dumps(full))
        print(f"[drum-r2] {spec['name']}: theta={rec['theta_got_deg']:.1f} "
              f"(ref {rec['theta_ref_deg']:.1f}, err {rec['theta_rel_err']:.2%}) "
              f"r_p05={rec['observables']['r_p05_over_R']:.2f} "
              f"regime={rec['measured_regime']} mode={rec['mode']} "
              f"resting_ok={rec['resting_ok']} wall={rec['wall_time_s']}s",
              flush=True)
    return cases


if __name__ == "__main__":
    main()