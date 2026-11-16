#!/usr/bin/env python3
"""Sweep driver for CASE-DEM-BRAZIL-NUT-SEGREGATION.

Runs the case points of the Möbius et al. (2001) benchmark with
:func:`run_case.run_case` in a process pool (one process per case, all cases
independent) and writes one JSON per case into ``runs/``.

Grids
-----
``pilot``  three density ratios, few taps -- instrument check only.
``full``   density-ratio family at D/d = 8 (the reference abscissa), a size
           ratio family, a drive (Gamma) sensitivity family and a time-step
           convergence pair.  See ``PREREG`` below (declared before the run).

Every case is a pure function of its arguments, so the grid is reproducible from
this file plus ``run_case.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------- predeclared
PREREG = {
    "reference": {
        "citation": "M.E. Moebius, B.E. Lauderdale, S.R. Nagel & H.M. Jaeger, "
                    "Nature 414, 270 (2001), DOI 10.1038/35104697, Fig. 1",
        "measured": "rising time T_rise vs density ratio rho/rho_m and size ratio D/d",
        "config": {"d_m": 5.0e-4, "rho_m_kg_m3": 2400.0, "fill_height_m": 0.086,
                   "cell_diameter_m": 0.082, "Gamma": 5.0, "f_Hz": 13.0,
                   "D_m": 0.0254, "initial_top_depth_m": 0.046},
    },
    "hypotheses": {
        "H1_direction": "a large intruder denser than the bed (rho/rho_m = 3.24) "
                        "rises (Brazil-nut ascent) at Gamma = 5",
        "H2_nonmonotonic": "T_rise(rho/rho_m) is non-monotonic with an interior "
                           "maximum near rho/rho_m ~ 0.5 (the reference's rho*/rho_m "
                           "~ 0.47-0.5, measured on the ambient-pressure branch)",
        "H3_size": "the rise time decreases as D/d grows (size dependence becomes "
                   "pronounced for large D/d)",
    },
    "metrics": {
        "T_rise": "taps from release until the intruder top reaches the free "
                  "surface (censored at the tap budget)",
        "rise_per_tap": "mean increase of intruder centre height per tap, in units "
                        "of the small-particle diameter d",
        "layering_index": "(z_intruder - z_small_centroid)/H_bed",
    },
    "tolerances": {
        "qualitative": "sign/direction and the ordering of T_rise must match the "
                       "reference trend",
        "quantitative": "T_rise relative error < 20 % for trend descriptors that "
                        "the reference reports numerically",
        "withheld": "absolute T_rise values of the reference are only published "
                    "graphically (Fig. 1, paywalled body); they are marked "
                    "WITHHELD, never digitised by guesswork",
    },
    "scale_gap": {
        "D_over_d": "8 (reference 50.8): a 1:1 reproduction needs ~1e4 particles "
                    "across the bed depth; declared, not hidden",
        "air": "the 2D dry-disk engine has no interstitial gas; the reference's "
               "pressure branch (90 torr) is out of scope for this round",
    },
}


def grid_full() -> list[dict]:
    cases: list[dict] = []
    # (a) density family at fixed size ratio -- the reference's main abscissa
    for rho in (0.25, 0.47, 0.75, 1.0, 1.5, 2.4, 3.24):
        for seed in (0, 1):
            cases.append(dict(size_ratio=8.0, rho_ratio=rho, seed=seed,
                              max_taps=30, settle_time=0.15))
    # (b) size family at the heaviest intruder (steel in the reference inset)
    for sr in (4.0, 6.0, 12.0):
        cases.append(dict(size_ratio=sr, rho_ratio=3.24, seed=0,
                          max_taps=30, settle_time=0.15))
    # (c) drive sensitivity (Gamma = 2, 3, 7 against the reference value 5)
    for gamma in (2.0, 3.0, 7.0):
        cases.append(dict(size_ratio=8.0, rho_ratio=3.24, seed=0, max_taps=30,
                          settle_time=0.15, gamma=gamma))
    # (d) time-step convergence at half the production step
    cases.append(dict(size_ratio=8.0, rho_ratio=3.24, seed=0, max_taps=30,
                      settle_time=0.15, dt=1.0e-5))
    cases.append(dict(size_ratio=8.0, rho_ratio=1.0, seed=0, max_taps=30,
                      settle_time=0.15, dt=1.0e-5))
    return cases


def grid_reduced() -> list[dict]:
    """Reduced grid executed when the cluster queue does not dispatch in budget.

    Same protocol as ``grid_full`` but D/d = 6 (409 particles instead of 730),
    dt = 2.5e-5 s (the engine's own stability limit for kn = 3e4 N/m) and four
    taps.  Recorded as a *deviation* from the pre-registered grid.
    """
    cases: list[dict] = []
    for rho in (0.47, 1.0, 2.0, 3.24):
        cases.append(dict(size_ratio=6.0, rho_ratio=rho, seed=0, max_taps=4,
                          settle_time=0.02, dt=2.5e-5))
    cases.append(dict(size_ratio=4.0, rho_ratio=3.24, seed=0, max_taps=4,
                      settle_time=0.02, dt=2.5e-5))
    cases.append(dict(size_ratio=10.0, rho_ratio=3.24, seed=0, max_taps=4,
                      settle_time=0.02, dt=2.5e-5))
    cases.append(dict(size_ratio=6.0, rho_ratio=3.24, seed=0, max_taps=4,
                      settle_time=0.02, dt=2.5e-5, gamma=7.0))
    cases.append(dict(size_ratio=6.0, rho_ratio=3.24, seed=1, max_taps=4,
                      settle_time=0.02, dt=2.5e-5))
    return cases


def grid_pilot() -> list[dict]:
    return [dict(size_ratio=8.0, rho_ratio=r, seed=0, max_taps=6,
                 settle_time=0.10) for r in (0.47, 1.0, 3.24)]


def case_name(c: dict) -> str:
    extra = ""
    if c.get("dt", 2.0e-5) != 2.0e-5:
        extra += f"_dt{c['dt']:g}"
    if c.get("gamma", 5.0) != 5.0:
        extra += f"_G{c['gamma']:g}"
    return (f"brazil_nut_R{c['size_ratio']:g}_rho{c['rho_ratio']:g}"
            f"_s{c['seed']}{extra}")


def run_one(c: dict, outdir: str) -> dict:
    from run_case import run_case

    path = os.path.join(outdir, case_name(c) + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            return {"case": case_name(c), "status": "cached"}
    rec = run_case(**c)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    return {"case": case_name(c), "status": "ok",
            "wall_clock_s": rec["audit"]["wall_clock_s"],
            "metrics": rec["metrics"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("pilot", "reduced", "full"), default="full")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outdir", default=os.path.join(HERE, "runs"))
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(HERE, "prereg.json"), "w") as fh:
        json.dump(PREREG, fh, indent=1)
    cases = {"pilot": grid_pilot, "reduced": grid_reduced,
             "full": grid_full}[a.grid]()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(run_one, c, a.outdir) for c in cases]
        for f in futs:
            r = f.result()
            results.append(r)
            print(json.dumps(r)[:400], flush=True)
    print(f"done {len(results)} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())