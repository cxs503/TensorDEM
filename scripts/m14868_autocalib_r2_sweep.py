"""GAP-DEM-AUTOCALIB round 2, gate 1: is the repose response a *material*
function, or a protocol artefact?

Round 1 (`docs/GAP-DEM-AUTOCALIB_round1.md`) measured the shipped pluviation
heap at a single system size and concluded that the forward model cannot reach
the dry-sand repose band (30-37 deg) and is not monotone in ``mu``.  Round 1's
own round-2 plan then asked for the missing step first: **size convergence** --
without it a calibrated ``theta_r`` is a property of the system size, not of the
material.

This script runs the pre-declared quartet of experiments that decides the
question, all on ``tensordem.dem_calibration``'s pluviation protocol:

``SIZE``  N in {240, 480, 960, 1920} grains, seeds {0, 1}, mu = 0.52 (sand
          twin truth) -> theta_r(N) and the plateau test (round-1's 2 deg
          criterion) plus the seed noise floor.
``DESIGN`` a 3-point factorial at N = 960, seeds {0, 1}:
          (mu, inject_radius_factor) in {(0.30, 3.5), (0.70, 3.5),
          (0.30, 14.0)} -> material sensitivity (grain friction) versus
          protocol-geometry sensitivity (injection footprint), each expressed
          in units of the seed noise.
``REPEAT`` an exact re-run of one cell to prove bitwise determinism on this
          machine (digest equality).

Gates and thresholds live in
:mod:`tensordem.dem_calibration.repose_qualification` (pre-declared, not fitted
after the fact): G1 flank r2, G2 deposit solid fraction, G3 size plateau,
G4 material dominance + signal-to-noise.

Usage::

    PYTHONPATH=src python scripts/m14868_autocalib_r2_sweep.py --workers 6
    PYTHONPATH=src python scripts/m14868_autocalib_r2_sweep.py --evaluate-only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from tensordem.dem_calibration import repose_qualification as rq
from tensordem.dem_calibration.heap import (pluviation_layer_capacity,
                                            run_pluviation)
from tensordem.dem_calibration.materials import spec_by_name

OUT_DIR = os.path.join("artifacts", "gap_dem_autocalib_r2")
CELLS = os.path.join(OUT_DIR, "cells.jsonl")
SUMMARY = os.path.join(OUT_DIR, "qualification.json")

#: Protocol constants of the shipped pluviation experiment (materials._PROTOCOL).
DROP_HEIGHT_FACTOR = 1.5
LAYER_SIZE = 12
N_DESIGN = 960
MU_TRUTH = 0.52
IRF_DEFAULT = 7.0


def _layer_size_for(case, irf: float) -> int:
    """Largest injection layer that fits the injection disk (exact capacity).

    The geometry knob changes the injection radius, and the lattice capacity of
    a small disk is smaller than the historical hexagon-area estimate, so the
    layer has to shrink accordingly (fewer, denser pours).  Uses the exact
    capacity of the lattice generator instead of an area estimate.
    """
    probe = dataclasses.replace(case, inject_radius_factor=float(irf))
    return max(1, min(LAYER_SIZE, pluviation_layer_capacity(probe)))


def design() -> list[dict]:
    """The pre-declared cell list (one forward DEM call per cell)."""
    cells: list[dict] = []
    for n in (240, 480, 960, 1920):
        for seed in (0, 1):
            cells.append({"block": "SIZE", "n_grains": n, "seed": seed,
                          "mu": MU_TRUTH, "inject_radius_factor": IRF_DEFAULT})
    for (mu, irf) in ((0.30, 3.5), (0.70, 3.5), (0.30, 14.0)):
        for seed in (0, 1):
            cells.append({"block": "DESIGN", "n_grains": N_DESIGN, "seed": seed,
                          "mu": mu, "inject_radius_factor": irf})
    # exact re-run of the smallest cell: bitwise determinism witness
    cells.append({"block": "REPEAT", "n_grains": 240, "seed": 0,
                  "mu": MU_TRUTH, "inject_radius_factor": IRF_DEFAULT})
    return cells


def cell_id(cell: dict) -> str:
    return (f"{cell['block']}|N={cell['n_grains']}|seed={cell['seed']}"
            f"|mu={cell['mu']:g}|irf={cell['inject_radius_factor']:g}")


def run_cell(cell: dict) -> dict:
    """Execute one forward DEM repose call and return the response record."""
    sand = spec_by_name("sand")
    case = sand.case("calibration")
    irf = float(cell["inject_radius_factor"])
    layer = _layer_size_for(case, irf)
    case = dataclasses.replace(
        case,
        n_grains=int(cell["n_grains"]),
        seed=int(cell["seed"]),
        mu=float(cell["mu"]),
        inject_radius_factor=irf,
        drop_height_factor=DROP_HEIGHT_FACTOR,
        layer_size=int(layer),
        settle_per_layer=200,
    )
    t0 = time.time()
    res = run_pluviation(case)
    wall = time.time() - t0
    rec = dict(cell)
    rec.update({
        "cell": cell_id(cell),
        "layer_size": int(layer),
        "radius_m": float(case.radius),
        "kn": float(case.kn),
        "mu_rolling": float(case.mu_rolling),
        "theta_r_deg": float(res["theta_r_deg"]),
        "phi_solid": float(res["phi_solid"]) if "phi_solid" in res else float("nan"),
        "r2": float(res["r2"]) if "r2" in res else float("nan"),
        "ke_ratio": float(res["ke_ratio"]) if "ke_ratio" in res else float("nan"),
        "digest": res.get("digest"),
        "wall_s": round(wall, 2),
    })
    return rec


def _load() -> list[dict]:
    if not os.path.exists(CELLS):
        return []
    out = []
    with open(CELLS) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append(rec: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CELLS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def evaluate(cells: list[dict]) -> dict:
    """Evaluate the pre-declared gates on the recorded cells."""
    nominal = [c for c in cells
               if c["block"] in ("SIZE", "DESIGN", "REPEAT")
               and c["mu"] == MU_TRUTH and c["inject_radius_factor"] == IRF_DEFAULT]
    report = rq.evaluate(
        cells,
        size_match={"mu": MU_TRUTH, "inject_radius_factor": IRF_DEFAULT},
        mu_low=0.30, mu_high=0.70, irf_low=3.5, irf_high=14.0,
    )
    repeats = [c for c in cells if c["block"] == "REPEAT"]
    base = [c for c in cells if c["block"] == "SIZE"
            and c["n_grains"] == 240 and c["seed"] == 0]
    determinism = None
    if repeats and base:
        determinism = {
            "digest_cell": base[0]["digest"],
            "digest_repeat": repeats[0]["digest"],
            "bitwise_identical": base[0]["digest"] == repeats[0]["digest"],
            "theta_cell": base[0]["theta_r_deg"],
            "theta_repeat": repeats[0]["theta_r_deg"],
        }
    report["determinism"] = determinism
    report["n_cells"] = len(cells)
    report["cells"] = sorted(cells, key=lambda c: c["cell"])
    report["nominal_cells"] = len(nominal)
    report["environment"] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args(argv)

    cells = design()
    done = {c["cell"] for c in _load()}
    pending = [c for c in cells if cell_id(c) not in done]

    if not args.evaluate_only and pending:
        print(f"{len(pending)} cells pending, {len(done)} already recorded",
              flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = {pool.submit(run_cell, c): c for c in pending}
            for fut in as_completed(futs):
                rec = fut.result()
                _append(rec)
                print(f"[{time.time() - t0:7.1f}s] {rec['cell']:44s} "
                      f"theta={rec['theta_r_deg']:6.2f} deg "
                      f"phi={rec['phi_solid']:.3f} r2={rec['r2']:.3f} "
                      f"ke={rec['ke_ratio']:.3f} ({rec['wall_s']:.0f}s)",
                      flush=True)

    recorded = _load()
    missing = set(cell_id(c) for c in cells) - {c["cell"] for c in recorded}
    report = evaluate(recorded)
    report["missing_cells"] = sorted(missing)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUMMARY, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in
                      ("verdict", "failed_gates", "seed_sigma_deg",
                       "size_sweep")}, indent=2), flush=True)
    print(f"artefact: {SUMMARY}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())