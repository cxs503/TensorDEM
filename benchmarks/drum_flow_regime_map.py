"""IN32-D01 rotating-drum regime map: Fr x fill sweep, power ledger, free surface.

Usage
-----
    python benchmarks/drum_flow_regime_map.py --out artifacts/drum_flow/phase_map.json \
        --jobs 32 --revolutions 3 --fill 0.1 0.3 0.5 --fr 0.1 0.3 0.5 0.7 0.9 1.2

The sweep is embarrassingly parallel over phase-map cells (multiprocessing); each
cell integrates the drum with :func:`tensordem.drum_flow.regimes.simulate_point`
and reports Fr-band vs measured regime, the free-surface parabola fit and the
power-ledger closure.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.drum_flow.kernel import DrumConfig                       # noqa: E402
from tensordem.drum_flow.regimes import (                               # noqa: E402
    FR_BANDS, REGIMES, cataract_onset_froude, classify_froude, simulate_point,
)


def _cell(args):
    fill, fr, revolutions, measure_last, particle_radius = args
    cfg = DrumConfig(particle_radius=particle_radius, dt=4.0e-5)
    t0 = time.time()
    res = simulate_point(fill=fill, fr=fr, cfg=cfg,
                         revolutions=revolutions, measure_last=measure_last)
    res["wall_time_s"] = round(time.time() - t0, 1)
    return res


def _single(args):
    fill, fr, revolutions, measure_last, radius = args
    cfg = DrumConfig(particle_radius=radius, dt=5.0e-5)
    t0 = time.time()
    res = simulate_point(fill=fill, fr=fr, cfg=cfg, revolutions=revolutions,
                         measure_last=measure_last, sample_every=200)
    res["wall_time_s"] = round(time.time() - t0, 1)
    return res


def merge_cell_files(cell_dir, out, summary_extra):
    rows = [json.loads(p.read_text()) for p in sorted(Path(cell_dir).glob("cell_*.json"))]
    rows.sort(key=lambda r: (r["fill"], r["fr"]))
    closures = [r["ledger"]["closure_rel"] for r in rows if math.isfinite(r["ledger"]["closure_rel"])]
    r2 = [r["surface"]["fit"]["r2"] for r in rows if r.get("surface", {}).get("fit")]
    r2_all = [r["surface"]["r2_all_particles"] for r in rows
              if r.get("surface", {}).get("r2_all_particles") is not None]
    agree = sum(1 for r in rows if r["regime_froude"] == r["regime_kinematic"])
    summary = {
        "n_cells": len(rows),
        "fr_x_fill_agreement": agree / len(rows) if rows else None,
        "closure_median": sorted(closures)[len(closures) // 2] if closures else None,
        "closure_max": max(closures) if closures else None,
        "closure_within_2pct_fraction": (sum(c < 0.02 for c in closures) / len(closures)) if closures else None,
        "surface_r2_min": min(r2_all) if r2_all else None,
        "surface_r2_median": sorted(r2_all)[len(r2_all) // 2] if r2_all else None,
        "surface_r2_bed_min": min(r2) if r2 else None,
        "surface_r2_bed_median": sorted(r2)[len(r2) // 2] if r2 else None,
        "closure_all_below_2pct": all(c < 0.02 for c in closures) if closures else None,
        "cataract_onset_fr_analytic": cataract_onset_froude(135.0),
        "fr_bands": [[lo, hi, name] for lo, hi, name in FR_BANDS],
        "regimes": list(REGIMES),
        "config": summary_extra or {},
    }
    payload = {"summary": summary, "cells": rows}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=1))
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=float, nargs=2, default=None,
                    help="single cell: --one fill fr  (prints one JSON line)")
    ap.add_argument("--cell-out", default=None, help="write the single cell JSON here")
    ap.add_argument("--merge-dir", default=None, help="merge artifacts/drum_flow/cells into --out")
    ap.add_argument("--out", default="artifacts/drum_flow/phase_map.json")
    ap.add_argument("--jobs", type=int, default=min(32, mp.cpu_count()))
    ap.add_argument("--revolutions", type=float, default=3.0)
    ap.add_argument("--measure-last", type=float, default=1.0)
    ap.add_argument("--particle-radius", type=float, default=0.004)
    ap.add_argument("--fill", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--fr", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5])
    args = ap.parse_args(argv)

    if args.merge_dir:
        merge_cell_files(args.merge_dir, args.out,
                         {"revolutions": args.revolutions, "measure_last": args.measure_last,
                          "particle_radius": args.particle_radius, "jobs": args.jobs})
        return None

    if args.one:
        res = _single((args.one[0], args.one[1], args.revolutions,
                       args.measure_last, args.particle_radius))
        line = json.dumps(res)
        if args.cell_out:
            Path(args.cell_out).write_text(line)
        print(line)
        return res

    tasks = [(f, fr, args.revolutions, args.measure_last, args.particle_radius)
             for f, fr in itertools.product(args.fill, args.fr)]
    print(f"[drum] {len(tasks)} cells, {args.jobs} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        rows = pool.map(_cell, tasks, chunksize=1)
    rows.sort(key=lambda r: (r["fill"], r["fr"]))

    agree = sum(1 for r in rows if r["regime_froude"] == r["regime_kinematic"])
    acc = agree / len(rows)
    closures = [r["ledger"]["closure_rel"] for r in rows if math.isfinite(r["ledger"]["closure_rel"])]
    r2 = [r["surface"]["fit"]["r2"] for r in rows if "surface" in r and "fit" in r["surface"]]
    summary = {
        "n_cells": len(rows),
        "wall_time_s": round(time.time() - t0, 1),
        "fr_x_fill_agreement": acc,
        "closure_max": max(closures) if closures else None,
        "closure_median": sorted(closures)[len(closures) // 2] if closures else None,
        "closure_within_2pct_fraction": (sum(c < 0.02 for c in closures) / len(closures)) if closures else None,
        "surface_r2_min": min(r2_all) if r2_all else None,
        "surface_r2_median": sorted(r2_all)[len(r2_all) // 2] if r2_all else None,
        "surface_r2_bed_min": min(r2) if r2 else None,
        "surface_r2_bed_median": sorted(r2)[len(r2) // 2] if r2 else None,
        "closure_all_below_2pct": all(c < 0.02 for c in closures) if closures else None,
        "cataract_onset_fr_analytic": cataract_onset_froude(135.0),
        "fr_bands": [[lo, hi, name] for lo, hi, name in FR_BANDS],
        "regimes": list(REGIMES),
        "config": {"revolutions": args.revolutions, "measure_last": args.measure_last,
                   "particle_radius": args.particle_radius, "jobs": args.jobs},
    }
    payload = {"summary": summary, "cells": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=1))
    return payload


if __name__ == "__main__":
    main()
