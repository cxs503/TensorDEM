#!/usr/bin/env python3
"""Run one record of the m23331 branch-probability ensemble (frozen Beverloo case).

Nothing physical is re-implemented: the record is produced by the campaign's own
``tensordem.silo_flow`` kernel (``build_bed`` + ``run_discharge``).  The script
controls exactly two things, the two factors of ``prereg.md`` §2:

* the **``dt`` spelling** — ``F`` = ``0.05·T_c`` (what ``Material.dt`` computes)
  or ``Q`` = ``T_c/20`` (what ``m23114/run_level.py`` computes); the two differ
  by 1 ULP and both are written into the record as IEEE-754 hex;
* the **packing seed** — ``1 … 6``.

    PYTHONPATH=src python benchmarks/m23331_branch_prob/run_case.py --list
    PYTHONPATH=src python benchmarks/m23331_branch_prob/run_case.py \
        --arm F --seed 3 --out runs/m23331_branch_prob/F_s3.json

The compute gate addresses shards by index, so ``--index i --nshards 12`` maps
``i`` onto ``(arm, seed)`` in a fixed, sorted order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

from tensordem.silo_flow.beverloo import (  # noqa: E402
    DischargeCase, beverloo_prediction, run_discharge,
)
from tensordem.silo_flow.kernel import Material, SiloGeometry, contact_time  # noqa: E402
from tensordem.silo_flow.packing import build_bed  # noqa: E402

#: frozen primary case of the campaign (m22885 §2 / m23114 §2 / m23184 /
#: m23261 §2) — no parameter is changed by this mission
FROZEN = dict(
    particle_diameter=0.003,
    orifice_diameter=0.015,
    silo_radius=0.024,
    fill_height=0.036,
    wall_mu=0.0,
    settle_time=0.06,
    discard_time=0.12,
    window_time=0.05,
    n_windows=8,
    mu=0.50,
    mu_rolling=0.05,
    kn=5.0e3,
)

ARMS = ("F", "Q")
SEEDS = (1, 2, 3, 4, 5, 6)


def shard_names() -> list[str]:
    """Fixed shard order ``(arm, seed)`` — used by the compute gate index."""
    return [f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS]


def parse_shard(name: str) -> tuple[str, int]:
    arm, seed = name.split("_s")
    if arm not in ARMS or int(seed) not in SEEDS:
        raise SystemExit(f"unknown shard {name!r}")
    return arm, int(seed)


def dt_specs() -> dict:
    """The two spellings of "20 steps per contact period" and their ULP gap."""
    material = Material(radius=0.5 * FROZEN["particle_diameter"],
                        kn=FROZEN["kn"], mu=FROZEN["mu"],
                        mu_rolling=FROZEN["mu_rolling"])
    t_c = contact_time(material.mass_eff, FROZEN["kn"])
    f_spelling = 0.05 * t_c          # Material.dt
    q_spelling = t_c / 20.0          # m23114 run_level.py
    return {
        "contact_period_s": t_c,
        "dt_F_s": f_spelling,
        "dt_Q_s": q_spelling,
        "dt_F_bits": struct.pack(">d", f_spelling).hex(),
        "dt_Q_bits": struct.pack(">d", q_spelling).hex(),
        "ulp_gap_Q_minus_F": (
            struct.unpack(">q", struct.pack(">d", q_spelling))[0]
            - struct.unpack(">q", struct.pack(">d", f_spelling))[0]),
        "rel_gap": abs(q_spelling - f_spelling) / f_spelling,
        "material": material,
    }


def state_sha(flow) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(flow.pos, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(flow.vel, dtype=np.float64).tobytes())
    return h.hexdigest()


def meta() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      cwd=str(ROOT)).strip()
    except Exception:  # pragma: no cover
        sha = "unknown"
    return {
        "git_sha": sha,
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--shard", help="<arm>_s<seed>, e.g. F_s3")
    ap.add_argument("--index", type=int, help="1-based shard index (compute gate)")
    ap.add_argument("--nshards", type=int, default=len(shard_names()))
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    names = shard_names()
    if args.index is not None:
        if args.nshards != len(names):
            raise SystemExit(f"--nshards must be {len(names)} (one shard per record)")
        args.shard = names[args.index - 1]
    if args.shard:
        args.arm, args.seed = parse_shard(args.shard)
    if args.list or args.arm is None:
        for i, name in enumerate(names, 1):
            print(f"{i:2d}  {name}")
        return 0

    specs = dt_specs()
    material = specs["material"]
    dt = specs["dt_F_s"] if args.arm == "F" else specs["dt_Q_s"]
    case = DischargeCase(seed=args.seed, **FROZEN)
    geometry = SiloGeometry(
        cylinder_radius=case.silo_radius,
        cylinder_height=case.fill_height + 0.02,
        outlet_radius=0.5 * case.orifice_diameter,
    )

    t0 = time.time()
    flow = build_bed(geometry, material, FROZEN["fill_height"],
                     wall_mu=FROZEN["wall_mu"], seed=args.seed,
                     settle_time=FROZEN["settle_time"], dt=dt)
    sha_after_settle = state_sha(flow)
    n_bed = flow.n
    bed_mass = flow.bed_mass
    rho_bulk = flow.bulk_density(FROZEN["fill_height"])
    wall_bed = time.time() - t0

    t1 = time.time()
    res = run_discharge(case, flow=flow)
    wall_discharge = time.time() - t1

    w_ref = beverloo_prediction(case.orifice_diameter, case.particle_diameter, rho_bulk)
    payload = dict(res)
    payload["branch_prob"] = {
        "shard": f"{args.arm}_s{args.seed}",
        "arm": args.arm,
        "seed": args.seed,
        "dt_s": float(dt),
        "dt_bits": struct.pack(">d", float(dt)).hex(),
        "specs": {k: v for k, v in specs.items() if k != "material"},
        "frozen": FROZEN,
        "n_bed": n_bed,
        "bed_mass_kg": bed_mass,
        "rho_bulk_nominal_fill": rho_bulk,
        "bed_state_sha256": sha_after_settle,
        "w_reference_kg_s": w_ref,
        "status": res["discharge_status"]["status"],
        "arrested": bool(res["discharge_status"]["arrested"]),
        "rel_error_vs_beverloo": res["flow_rate_kg_s"] / w_ref - 1.0,
        "wall_clock_bed_s": wall_bed,
        "wall_clock_discharge_s": wall_discharge,
        "meta": meta(),
    }

    out = Path(args.out or f"runs/m23331_branch_prob/{args.arm}_s{args.seed}.json")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    st = res["discharge_status"]
    print(f"{args.arm}_s{args.seed}: dt={float(dt):.17e} W={res['flow_rate_kg_s']:.9f} "
          f"rel={payload['branch_prob']['rel_error_vs_beverloo']:+.2%} "
          f"{st['status']} windows={res['window_particles']} "
          f"bed={sha_after_settle[:12]} wall=({wall_bed:.0f}+{wall_discharge:.0f})s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())