#!/usr/bin/env python3
"""Measure how fast a 1-ULP ``dt`` difference decorrelates the frozen silo bed.

The archive question (`docs/CASE-DEM-SILO-BEVERLOO_m23184_reproducibility_audit.md`)
is whether the 9.2 × spread of two records of the *same* frozen case and seed is
a code-state difference or an amplification of a sub-precision initial-state
difference.  This probe answers the dynamical half of that question without
needing the full 0.58 s discharge (which costs ~250 s per record):

* build the campaign's frozen bed once (`build_bed`, seed 1, outlet plugged);
* **control**: two copies at the *same* ``dt`` must stay bit-identical
  (``max|Δpos| == 0``) — a non-zero control would mean the kernel is not
  deterministic and the experiment would be void;
* **test**: one copy gets ``dt`` moved by ±1 representable step (or by a finite
  factor) and both are stepped in lockstep; the separation
  ``max|Δpos|/d`` / ``median|Δpos|/d`` is recorded against physical time.

A separation that reaches one particle diameter inside the campaign's 0.5 s
measurement window means a single record's ``W`` cannot be reproduced across
drivers that spell ``dt`` differently.

    PYTHONPATH=src python benchmarks/m23261_ulp_dt/probe_divergence.py \
        --shift-ulps 1 --steps 4000 --out runs/m23261_ulp_dt/divergence_up1.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tensordem.silo_flow.kernel import Material, SiloGeometry  # noqa: E402
from tensordem.silo_flow.packing import build_bed  # noqa: E402

FROZEN = dict(particle_diameter=0.003, silo_radius=0.024, fill_height=0.036,
              kn=5.0e3, mu=0.50, mu_rolling=0.05, wall_mu=0.0, seed=1,
              settle_time=0.06)


def ulp(x: float, n: int) -> float:
    i = struct.unpack(">q", struct.pack(">d", x))[0]
    return struct.unpack(">d", struct.pack(">q", i + int(n)))[0]


def meta() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      cwd=str(ROOT)).strip()
    except Exception:  # pragma: no cover
        sha = "unknown"
    return {"git_sha": sha, "host": platform.node(), "machine": platform.machine(),
            "python": platform.python_version(), "utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def growth_rate(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Least-squares slope of ``ln y`` over the band ``lo <= y <= hi``."""
    m = (y >= lo) & (y <= hi) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    slope = np.polyfit(t[m], np.log(y[m]), 1)[0]
    return float(slope)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shift-ulps", type=int, default=1,
                    help="dt of the perturbed copy, in representable steps")
    ap.add_argument("--finite-factor", type=float, default=None,
                    help="use a relative dt change instead of a ULP shift")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--settle-steps", type=int, default=0,
                    help="extra steps after the outlet opens before recording")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    material = Material(radius=0.5 * FROZEN["particle_diameter"], kn=FROZEN["kn"],
                        mu=FROZEN["mu"], mu_rolling=FROZEN["mu_rolling"])
    geometry = SiloGeometry(cylinder_radius=FROZEN["silo_radius"],
                            cylinder_height=FROZEN["fill_height"] + 0.02,
                            outlet_radius=0.0075)
    t0 = time.time()
    base = build_bed(geometry, material, FROZEN["fill_height"],
                     wall_mu=FROZEN["wall_mu"], seed=FROZEN["seed"],
                     settle_time=FROZEN["settle_time"])
    wall_build = time.time() - t0
    base.plugged = False
    dt0 = base.dt

    # ---- control: two copies, identical dt -> must stay bit-identical ----------
    c1, c2 = copy.deepcopy(base), copy.deepcopy(base)
    for _ in range(300):
        c1.step()
        c2.step()
    control_max = float(np.max(np.abs(c1.pos - c2.pos)))
    control_ok = control_max == 0.0

    # ---- test: one copy with dt shifted --------------------------------------
    a = base
    b = copy.deepcopy(base)
    if args.finite_factor is not None:
        dt1 = dt0 * (1.0 + args.finite_factor)
    else:
        dt1 = ulp(dt0, args.shift_ulps)
    b.dt = float(dt1)
    d = FROZEN["particle_diameter"]

    series = []
    t_phys = 0.0
    for k in range(1, args.steps + 1):
        a.step()
        b.step()
        if k % args.every == 0:
            # the two flows lose particles at (eventually) different steps, so the
            # separation is only defined on the common prefix; a count mismatch is
            # itself the macro-scale divergence we are looking for
            m = min(a.n, b.n)
            dpos = np.linalg.norm(a.pos[:m] - b.pos[:m], axis=1)
            series.append({
                "step": k,
                "time_s": k * dt0,
                "max_over_d": float(dpos.max() / d),
                "median_over_d": float(np.median(dpos) / d),
                "frac_gt_0p1d": float(np.mean(dpos > 0.1 * d)),
                "n_a": int(a.n),
                "n_b": int(b.n),
                "exited_a": int(a.n_exited),
                "exited_b": int(b.n_exited),
                "count_mismatch": int(a.n != b.n),
                "first_count_mismatch_step": next(
                    (s["step"] for s in reversed(series) if s["n_a"] != s["n_b"]), None),
            })
    t_phys = args.steps * dt0

    t_arr = np.array([s["time_s"] for s in series])
    med = np.array([s["median_over_d"] for s in series])
    mx = np.array([s["max_over_d"] for s in series])

    def first_cross(y, thr):
        idx = np.nonzero(y >= thr)[0]
        return None if idx.size == 0 else float(t_arr[idx[0]])

    payload = {
        "probe": "m23261 dt-divergence",
        "frozen": FROZEN,
        "shift_ulps": args.shift_ulps,
        "finite_factor": args.finite_factor,
        "dt0_s": float(dt0),
        "dt1_s": float(dt1),
        "dt0_bits": struct.pack(">d", float(dt0)).hex(),
        "dt1_bits": struct.pack(">d", float(dt1)).hex(),
        "dt_shift_rel": (float(dt1) - float(dt0)) / float(dt0),
        "n_particles": int(a.n_initial),
        "steps": args.steps,
        "sim_time_s": float(t_phys),
        "control_same_dt_max_abs_dpos_m": control_max,
        "control_bit_identical": bool(control_ok),
        "first_cross_0p01d_s": first_cross(med, 0.01),
        "first_cross_0p1d_s": first_cross(med, 0.1),
        "first_cross_1d_s": first_cross(mx, 1.0),
        "growth_rate_per_s": growth_rate(t_arr, med, 1e-6, 1e-1),
        "final_median_over_d": float(med[-1]),
        "final_max_over_d": float(mx[-1]),
        "exited_a": int(a.n_exited),
        "exited_b": int(b.n_exited),
        "n_a_final": int(a.n),
        "n_b_final": int(b.n),
        "first_count_mismatch_s": next(
            (s["time_s"] for s in series if s["n_a"] != s["n_b"]), None),
        "series": series,
        "wall_clock_build_s": wall_build,
        "wall_clock_total_s": time.time() - t0,
        "meta": meta(),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"dt={dt1!r} shift_rel={payload['dt_shift_rel']:+.3e} "
          f"control_bit_identical={control_ok} "
          f"median|dpos|/d final={payload['final_median_over_d']:.3e} "
          f"first_cross_0.01d={payload['first_cross_0p01d_s']} s "
          f"lambda={payload['growth_rate_per_s']:.1f} 1/s "
          f"exited a={a.n_exited} b={b.n_exited} wall={payload['wall_clock_total_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())