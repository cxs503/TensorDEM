"""Campaign runner for the P70-N03 bidisperse segregation verification face.

Usage (one case per process, so the campaign can be sharded across nodes)::

    python scripts/run_bidisperse_segregation.py --list
    python scripts/run_bidisperse_segregation.py --case s4_g2_phi02_f25_seed1 --out runs/p70n03
    python scripts/run_bidisperse_segregation.py --cases-from runs/p70n03/cases.txt --out runs/p70n03

Every case writes ``<out>/<case>.json`` with the full history, the acceptance
metrics of that case and the stability transmission quantities.  Case names are
stable so that the campaign can be replayed shard by shard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.bidisperse_segregation import (  # noqa: E402
    BedConfig, BidisperseBed, summarize_shake,
)
from tensordem.segregation_stability import (  # noqa: E402
    partition_table, transmission_from_partitions, floor_pressure_report,
)

G = 9.81
DTYPE = "float64"

#: campaign geometry: bed box side (m) and small diameter (m).
BOX_WIDTH_3D = 0.10
BOX_WIDTH_Q2D = 3.0
D_SMALL = 0.008
N_PARTICLES = 4000
HOLD_LENGTH = 12.0          # m, reference hold (stability consumer)
HOLD_BREADTH = 6.0
DISPLACEMENT = 8000.0       # m^3, reference ship
CARGO_MASS_RATIO = 0.35
SETTLE_TIME = 0.06


def case_list() -> list[dict]:
    """Campaign case definitions (name, then BedConfig / drive overrides)."""
    cases: list[dict] = []

    def add(name, *, size_ratio, phi=0.2, geometry="box3d", seeds=(1, 2, 3),
            gamma=2.0, freq=25.0, cycles=30, n=N_PARTICLES, d_small=D_SMALL,
            width=None, drive="shake", top_fraction=0.2, pour_steps=6000,
            kn=800.0):
        for seed in seeds:
            cases.append({
                "name": f"{name}_seed{seed}" if len(seeds) > 1 else name,
                "size_ratio": float(size_ratio),
                "large_volume_fraction": float(phi),
                "geometry": geometry,
                "gamma": float(gamma),
                "frequency": float(freq),
                "cycles": int(cycles),
                "n_particles": int(n),
                "d_small": float(d_small),
                "width": float(width if width is not None else
                               (BOX_WIDTH_3D if geometry == "box3d" else BOX_WIDTH_Q2D)),
                "seed": int(seed),
                "drive": drive,
            "kn": kn,
                "top_fraction": float(top_fraction),
                "pour_steps": int(pour_steps),
            })

    # (a) size-ratio sweep, the monotonicity / exponent scan (3 seeds each)
    for s in (1.0, 1.5, 2.0, 3.0, 4.0):
        add(f"s{s:g}_g2_phi02_f25", size_ratio=s, gamma=2.0)
    # (b) dense coarse fraction, second loading
    for s in (1.0, 2.0, 3.0):
        add(f"s{s:g}_g2_phi04_f25", size_ratio=s, phi=0.4, seeds=(1, 2))
    # (c) shaking-intensity sweep -> threshold and saturation
    for gam in (1.0, 1.5, 2.5, 3.0):
        add(f"s2_g{gam:g}_phi02_f25", size_ratio=2.0, gamma=gam)
    # (d) frequency scan at fixed Gamma
    for f in (15.0, 40.0, 60.0):
        add(f"s2_g2_phi02_f{f:g}", size_ratio=2.0, freq=f)
    # (e) quasi-2D channel configuration
    for s in (1.0, 2.0, 4.0):
        add(f"q2d_s{s:g}_g2_phi02_f25", size_ratio=s, geometry="quasi2d")
    # (f) large-scale confirmation (1e5 particles)
    add("n4e3_s2_g2_phi02_f25", size_ratio=2.0, n=4000, cycles=30, seeds=(1,))
    add("n4e3_s1_g2_phi02_f25", size_ratio=1.0, n=4000, cycles=30, seeds=(1,))
    # (g) pouring drive (heap segregation)
    for s in (1.0, 2.0, 3.0):
        add(f"pour_s{s:g}_phi02", size_ratio=s, drive="pour", seeds=(1, 2))
    return cases


def build_bed(case: dict) -> BidisperseBed:
    cfg = BedConfig(
        size_ratio=case["size_ratio"],
        large_volume_fraction=case["large_volume_fraction"],
        n_particles=case["n_particles"],
        d_small=case["d_small"],
        geometry=case["geometry"],
        width=case["width"],
        kn=case.get("kn"),
        seed=case["seed"],
    )
    bed = BidisperseBed(cfg)
    bed.settle(SETTLE_TIME)
    return bed


def run_case(case: dict) -> dict:
    t_start = time.time()
    bed = build_bed(case)
    cfg = bed.cfg
    bed_start = partition_table(bed, 8, 4)
    initial = bed.measure(case["top_fraction"])
    info = bed.packing_state()

    if case["drive"] == "shake":
        omega = 2.0 * math.pi * case["frequency"]
        amplitude = case["gamma"] * G / omega ** 2
        duration = case["cycles"] / case["frequency"]
        result = bed.run_shaking(duration, amplitude, case["frequency"],
                                 sample_dt=0.04, top_fraction=case["top_fraction"])
        summary = summarize_shake(result, case["top_fraction"])
        history = [{k: v for k, v in h.items()} for h in result["history"]]
    else:
        result = bed.run_pour(case["pour_steps"], nozzle_width=4.0 * cfg.d_small,
                              release_steps=int(0.5 * case["pour_steps"]),
                              sample_dt=0.04)
        hist = result["history"]
        tail = hist[int(0.85 * len(hist)):]
        summary = {
            "S": float(np.mean([h["S_pour"] for h in tail])),
            "S_std": float(np.std([h["S_pour"] for h in tail], ddof=1)),
            "S_core": float(np.mean([h["S_pour_core"] for h in tail])),
            "mass_injected": result["mass_injected"],
            "mass_final": bed.total_mass,
            "mass_rel_error": abs(result["mass_injected"] - bed.total_mass)
            / bed.total_mass,
            "n_released": result["released"],
            "max_overlap": float(max(h["max_overlap"] for h in hist)),
            "n_steps": result["n_steps"],
            "v_up": float("nan"),
        }
        history = [{k: v for k, v in h.items()} for h in hist[:: max(1, len(hist) // 60)]]

    bed_end = partition_table(bed, 8, 4)
    transmission = transmission_from_partitions(
        bed_start, bed_end,
        hold_length=HOLD_LENGTH, hold_breadth=HOLD_BREADTH,
        displacement=DISPLACEMENT, cargo_mass_ratio=CARGO_MASS_RATIO,
        n_x=8, n_z=4)
    if case["drive"] == "shake":
        pressure = floor_pressure_report(summary, bed.Lx, bed.Ly, bed.total_mass)
    else:
        pressure = {}

    # trim the partition tables for the artifact (keep the surface profile)
    out = {
        "case": case,
        "packing": info,
        "initial": initial,
        "summary": summary,
        "transmission": {k: v for k, v in transmission.items()
                         if k not in ("partition_start", "partition_end")},
        "partition_start": bed_start,
        "partition_end": bed_end,
        "floor_pressure": pressure,
        "history": history,
        "final_state": {
            "z_top": float(np.max(bed.pos[:, 2] + bed.r)),
            "z_com": float(np.sum(bed.mass * bed.pos[:, 2]) / bed.total_mass),
            "mass_total": bed.total_mass,
            "solid_volume": float(np.sum(bed.volume)),
            "max_overlap": bed.last_max_overlap,
        },
        "wall_clock_s": time.time() - t_start,
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--cases-from")
    ap.add_argument("--out", default="runs/p70n03")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args(argv)

    cases = {c["name"]: c for c in case_list()}
    if args.list:
        for name in cases:
            if cases[name] is None:
                continue
            sel = list(cases).index(name) % max(args.shards, 1) == args.shard
            if sel:
                print(name)
        return 0

    if args.cases_from:
        with open(args.cases_from) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
    elif args.case:
        names = [args.case]
    else:
        ap.error("need --case or --cases-from")

    os.makedirs(args.out, exist_ok=True)
    for name in names:
        if name not in cases:
            raise SystemExit(f"unknown case {name}")
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            print(f"skip {name} (done)", flush=True)
            continue
        t0 = time.time()
        try:
            out = run_case(cases[name])
        except Exception as exc:                        # noqa: BLE001
            out = {"case": cases[name], "status": "error", "error": repr(exc),
                   "wall_clock_s": time.time() - t0}
            with open(path, "w") as fh:
                json.dump(out, fh, ensure_ascii=False)
            print(f"{name}: ERROR {exc}", flush=True)
            continue
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump(out, fh, ensure_ascii=False)
        os.replace(tmp, path)
        print(f"{name}: {out['wall_clock_s']:.0f}s "
              f"S={out['summary'].get('S', float('nan')):.4f} "
              f"v_up={out['summary'].get('v_up', float('nan')):.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())