#!/usr/bin/env python
"""Anchor validity audit: does the c = 0 assumption hold?

The anchor (docs/CASE-DEM-BIAXIAL-FRICTION.md) converts the peak stress state into a
friction angle with the c = 0 Mohr-Coulomb form

    sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3)

and compares the result with Bolton (1986) phi_ps = 38 deg, a c = 0 friction angle for
dense Hostun sand.  Two things in the anchor configuration are in tension with c = 0:

  * ``freeze_bond_fraction = 0.35`` -- 35 % of near-neighbour contacts are frozen bonds,
    which transmit tension and shear.  A bonded assembly has *cohesion*, so the quantity
    above is an apparent friction angle, not a friction angle.
  * The anchor document pre-declares a confining-pressure sensitivity run
    ("A sensitivity run at sigma_3 = 200 kPa is pre-declared to verify pressure
    independence; its artefact is not yet committed") which has never been run.

This script fills that declared verification debt.  Cells:

    p050 / p100 / p200   confining pressure sweep at fixed material and geometry
    p100_b0              confining 100 kPa with the freeze bonds disabled (c = 0 control)

If phi is a genuine c = 0 friction angle, phi must be pressure independent across
p050/p100/p200.  The pressure dependence, plus the Mohr-Coulomb fit of q_peak vs p_peak,
yields the apparent cohesion c.  The bonded/unbonded pair isolates the cementation share.

Writes runs/m21262/pressure/<tag>.json.  Shardable with --shard/--nshards.
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tensordem.biaxial_friction_case import run_biaxial_friction_case  # noqa: E402

OUT = os.path.join(REPO, "runs", "m21262", "pressure")
STRAIN_TARGET = 0.30

BASE = dict(columns=12, rows=24, strain_rate=0.05, consolidation_steps=4000,
            freeze_bond_fraction=0.35)  # explicit on both sides of the bond control

CELLS: dict[str, dict] = {
    "p050":    dict(BASE, confining_stress=50.0e3),
    "p100":    dict(BASE, confining_stress=100.0e3),
    "p200":    dict(BASE, confining_stress=200.0e3),
    "p100_b0": dict(BASE, confining_stress=100.0e3, freeze_bond_fraction=0.0),  # c = 0 control
}


def run_cell(tag: str, over: dict) -> dict:
    t0 = time.time()
    res = run_biaxial_friction_case(strain_target=STRAIN_TARGET, **over)
    pk = res["peak"]
    rec = {
        "tag": tag,
        "overrides": over,
        "strain_target": STRAIN_TARGET,
        "confining_kPa": over["confining_stress"] / 1e3,
        "freeze_bond_fraction": over.get("freeze_bond_fraction", 0.35),
        "phi_wall_deg": res["friction_angle"]["phi_peak_wall_deg"],
        "phi_internal_deg": res["friction_angle"]["phi_peak_internal_deg"],
        "sigma_v_kPa": pk["sigma_v"] / 1e3,
        "sigma_h_kPa": pk["sigma_h"] / 1e3,
        "q_kPa": pk["deviator"] / 1e3,
        "p_kPa": 0.5 * (pk["sigma_v"] + pk["sigma_h"]) / 1e3,
        "eps_a_peak": pk["axial_strain"],
        "specimen": res["specimen"],
        "loading": res["loading"],
        "elapsed_s": round(time.time() - t0, 2),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(os.environ.get("_PRESS_OUT", OUT), f"{tag}.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    return rec


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--prefix", type=str, default="")
    args = ap.parse_args(argv[1:])

    if args.columns != 12 or args.rows != 24:
        # geometry override: rebuild the cell table so the record is self-describing
        for tag, cfg in CELLS.items():
            cfg["columns"] = args.columns
            cfg["rows"] = args.rows
        globals()["OUT"] = os.path.join(REPO, "runs", "m21262", "pressure_%dx%d"
                                        % (args.columns, args.rows))

    tags = list(CELLS)
    if args.nshards > 1:
        tags = [t for i, t in enumerate(tags) if i % args.nshards == args.shard]

    for tag in tags:
        rec = run_cell(args.prefix + tag if args.prefix else tag, CELLS[tag])
        print(f"{tag:10s} p={rec['confining_kPa']:6.0f} kPa  "
              f"phi_wall={rec['phi_wall_deg']:7.2f}  phi_int={rec['phi_internal_deg']:7.2f}  "
              f"q_peak={rec['q_kPa']:7.1f} kPa  p_peak={rec['p_kPa']:7.1f} kPa  "
              f"{rec['elapsed_s']:6.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))