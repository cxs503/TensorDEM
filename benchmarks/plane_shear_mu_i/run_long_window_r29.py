#!/usr/bin/env python3
"""Round 29 (m22733): long-window stationarity campaign for CASE-DEM-PLANE-SHEAR-MUIDI.

Why this campaign exists
------------------------
Round 27 (``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R27.md``) showed that every judged
``mu`` in rounds 24-26 comes from a **1-strain** measurement window, and that at
``I = 0.05`` the half-split test rejects that window as a steady state
(``z = -8.9``, the saturation asymptote sitting far from the window mean).
Round 28 then listed "run a long-strain window (gamma = 5-10)" as its **first**
priority and "polydispersity" as the lever round 27 identified for the factor
~2-3 offset.  No round has produced a window longer than ~1.5 strain, so the
ledger currently has no stationarity-certified ``mu`` at all.

This driver runs exactly that missing configuration: the round 25/26 2-D probe
cell (458 free disks, ``P = 10 kPa``, ``mu_p = 0.4``, w = 0.5 polydispersity -
the widest spread for which the archived law is stated) with the measurement
window extended from 1 to **8 strain**, seed ensemble {0,1,2}, at
``I in {0.05, 0.2}`` plus the low-``I`` point ``I = 0.02`` (seed 0), and the
monodisperse crystal control ``w = 0`` at ``I = 0.05``.

Everything except ``strain_measure`` is bit-identical to the archived round 26
cases, so the long window can be compared against the archived 1-strain runs
point by point.  Each run stores its full per-block sample series (``samples``),
which is what ``analyse_long_window_r29.py`` bins by strain.

Usage (the campaign is embarrassingly parallel over combos)::

    python benchmarks/plane_shear_mu_i/run_long_window_r29.py --workers 8
    python benchmarks/plane_shear_mu_i/run_long_window_r29.py --smoke   # 1 tiny run

The cluster submits this file with ``run_compute``; the per-combo JSON lands in
``benchmarks/plane_shear_mu_i/runs/plane_shear_2d/long_r29_*.json``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

import plane_shear_2d as ps  # noqa: E402

RUNS = BENCH / "runs" / "plane_shear_2d"

# (polydispersity w, I_target, seed).  w = 0.5 = round 26's physical symmetry
# break; w = 0.0 = the monodisperse crystal control (round 25).
COMBOS = [
    (0.5, 0.02, 0),
    (0.5, 0.05, 0),
    (0.5, 0.05, 1),
    (0.5, 0.05, 2),
    (0.5, 0.20, 0),
    (0.5, 0.20, 1),
    (0.5, 0.20, 2),
    (0.0, 0.05, 0),
]

STRAIN_MEASURE = 8.0


def case_for(w: float, inertial: float, seed: int, *, smoke: bool = False):
    """The archived round-26 case with a long measurement window."""
    if smoke:
        return ps.PlaneShear2DCase(
            I_target=inertial, n_particles=200, width=0.24, gap=0.12,
            settle_time_units=20.0, shear_settle_strain=0.3, strain_measure=0.6,
            init_mode="random", seed=seed, polydispersity=w)
    return ps.PlaneShear2DCase(
        I_target=inertial, n_particles=458, width=0.30, gap=0.15,
        settle_time_units=40.0, shear_settle_strain=0.5,
        strain_measure=STRAIN_MEASURE,
        init_mode="random", seed=seed, polydispersity=w)


def run_one(task: tuple[float, float, int, bool]) -> dict:
    w, inertial, seed, smoke = task
    case = case_for(w, inertial, seed, smoke=smoke)
    case.validate()
    t0 = time.time()
    out = ps.simulate(case, progress=False)
    out["argv"] = {"w": w, "I_target": inertial, "seed": seed,
                   "strain_measure": case.strain_measure, "smoke": smoke}
    out["round"] = 29
    path = RUNS / f"long_r29_w{w:g}_I{inertial:g}_s{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    line = ("long_r29 w=%.2f I=%.3f seed=%d -> mu_wall %.4f (sd %.4f) "
            "mu_bulk %.4f phi %.4f I_meas %.4f drift_z %.2f dur %.0fs -> %s"
            % (w, inertial, seed, out.get("mu_wall", float("nan")),
               out.get("mu_wall_sd", float("nan")), out.get("mu_bulk", float("nan")),
               out.get("phi_gap", float("nan")), out.get("I_bulk", float("nan")),
               out.get("drift_z_mu_wall", float("nan")), time.time() - t0,
               path.name))
    print(line, flush=True)
    return {"file": str(path.relative_to(REPO)), "w": w, "I_target": inertial,
            "seed": seed, "mu_wall": out.get("mu_wall"),
            "mu_bulk": out.get("mu_bulk"), "phi_gap": out.get("phi_gap"),
            "I_bulk": out.get("I_bulk"),
            "drift_z_mu_wall": out.get("drift_z_mu_wall"),
            "duration_s": out.get("duration_s")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                    help="run a single combo, e.g. 'w=0.5,I=0.05,seed=0'")
    args = ap.parse_args(argv)

    combos = COMBOS
    if args.only:
        parts = dict(p.split("=") for p in args.only.split(","))
        combos = [(float(parts["w"]), float(parts["I"]), int(parts["seed"]))]
    if args.smoke:
        combos = [combos[1]]

    tasks = [(w, inertial, seed, args.smoke) for w, inertial, seed in combos]
    t0 = time.time()
    if args.workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            index = pool.map(run_one, tasks)
    else:
        index = [run_one(t) for t in tasks]

    index_path = RUNS / ("long_r29_index_smoke.json" if args.smoke
                         else "long_r29_index.json")
    index_path.write_text(json.dumps({"round": 29, "strain_measure": STRAIN_MEASURE,
                                      "combos": index}, indent=1))
    print(f"wrote {index_path} ({len(index)} runs, {time.time() - t0:.0f}s total)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())