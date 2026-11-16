#!/usr/bin/env python3
"""Run one shard of the m23261 1-ULP-dt experiment (frozen Beverloo case).

The physics is *not* re-implemented: every shard calls the campaign's own
``tensordem.silo_flow`` kernel.  What the shard controls is exactly one number —
the integration step, and whether that number is spelled ``0.05·T_c`` or
``T_c/20`` (which differ by 1 ULP), or moved by ±1 ULP / 0.1 %.

    PYTHONPATH=src python benchmarks/m23261_ulp_dt/run_ulp.py --list
    PYTHONPATH=src python benchmarks/m23261_ulp_dt/run_ulp.py --shard A_arch_bed \
        --out runs/m23261_ulp_dt/A_arch_bed.json

Shards C, D and E reuse the bit-identical settled bed of shard A (deep copy), so
they differ from A in nothing but ``dt``; the bed ``pos``/``vel`` sha256 is
written into every record so the claim can be checked, not trusted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

from tensordem.silo_flow.beverloo import (  # noqa: E402
    DischargeCase, beverloo_prediction, run_discharge,
)
from tensordem.silo_flow.kernel import Material, SiloGeometry, contact_time  # noqa: E402
from tensordem.silo_flow.packing import build_bed  # noqa: E402

#: frozen primary case of the campaign (m22885 §2 / m23114 §2 / m23184) — no
#: parameter is changed here
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
SEED = 1

#: archived values this experiment must reproduce (P1 of prereg.md §1)
ARCHIVE = {
    "A_arch_bed": {  # m22885_beverloo_powered/fill36_seed1.json, git db2bdaf
        "path": "/root/tp-agent/work/m22885/runs/m22885_beverloo_powered/fill36_seed1.json",
        "flow_rate_kg_s": 0.030749793997350797,
    },
    "B_quot_bed": {  # m23114_siloflow_ladder/dt_nom20_s1.json, git 1fb2a46
        "path": "/root/tp-agent/work/m23114/runs/m23114_siloflow_ladder/dt_nom20_s1.json",
        "flow_rate_kg_s": 0.0033577361261474805,
    },
}

#: shard -> (bed dt spelling, discharge dt transform)
SHARDS = {
    "A_arch_bed": ("arch", "identity"),
    "B_quot_bed": ("quot", "identity"),
    "C_up1": ("arch", "up"),
    "D_down1": ("arch", "down"),
    "E_finite": ("arch", "finite"),
}


def ulp(x: float, n: int) -> float:
    """``x`` moved ``n`` representable steps (n may be negative)."""
    i = struct.unpack(">q", struct.pack(">d", x))[0]
    return struct.unpack(">d", struct.pack(">q", i + int(n)))[0]


def dt_specs() -> dict:
    """The two spellings of "20 steps per contact period", and their ULP gap."""
    material = Material(radius=0.5 * FROZEN["particle_diameter"],
                        kn=FROZEN["kn"], mu=FROZEN["mu"],
                        mu_rolling=FROZEN["mu_rolling"])
    t_c = contact_time(material.mass_eff, FROZEN["kn"])
    arch = 0.05 * t_c          # what Material.dt computes (m22885 path)
    quot = t_c / 20.0          # what m23114/run_level.py computes
    return {
        "contact_period_s": t_c,
        "dt_material_property_s": material.dt,
        "dt_arch_spelling_s": arch,
        "dt_quot_spelling_s": quot,
        "spellings_equal": arch == quot,
        "ulp_gap_quot_minus_arch": (
            struct.unpack(">q", struct.pack(">d", quot))[0]
            - struct.unpack(">q", struct.pack(">d", arch))[0]),
        "rel_gap": abs(quot - arch) / arch,
        "dt_bits_arch": struct.pack(">d", arch).hex(),
        "dt_bits_quot": struct.pack(">d", quot).hex(),
        "material": material,
    }


def make_case() -> DischargeCase:
    return DischargeCase(seed=SEED, **FROZEN)


def bed(geometry, material, dt_bed: float):
    """Build the campaign's settled bed (outlet plugged), exactly as run_discharge."""
    return build_bed(geometry, material, FROZEN["fill_height"],
                     wall_mu=FROZEN["wall_mu"], seed=SEED,
                     settle_time=FROZEN["settle_time"], dt=dt_bed)


def state_sha(flow) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(flow.pos, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(flow.vel, dtype=np.float64).tobytes())
    return h.hexdigest()


def initial_plan() -> None:
    for name, (bed_spelling, transform) in SHARDS.items():
        bed_note = "own bed" if name in ("A_arch_bed", "B_quot_bed") else "bed of A (bit-identical)"
        print(f"{name:12s} bed_dt={bed_spelling:5s} discharge_dt={transform:9s} ({bed_note})")


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
    ap.add_argument("--shard", choices=sorted(SHARDS))
    ap.add_argument("--index", type=int, help="1-based shard index (for the compute gate)")
    ap.add_argument("--nshards", type=int, default=len(SHARDS))
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.index is not None:
        names = sorted(SHARDS)
        if args.nshards != len(names):
            raise SystemExit(f"--nshards must be {len(names)} (one shard per dt variant)")
        args.shard = names[args.index - 1]
    if args.list or not args.shard:
        initial_plan()
        return 0

    specs = dt_specs()
    material = specs["material"]
    case = make_case()
    geometry = SiloGeometry(
        cylinder_radius=case.silo_radius,
        cylinder_height=case.fill_height + 0.02,
        outlet_radius=0.5 * case.orifice_diameter,
    )
    bed_spelling, transform = SHARDS[args.shard]
    dt_bed = specs["dt_arch_spelling_s"] if bed_spelling == "arch" else specs["dt_quot_spelling_s"]

    t0 = time.time()
    # shards C/D/E share A's bed: build it once (shard A) and re-use the same
    # deterministic recipe here, which reproduces the identical state because
    # build_bed is a pure function of (geometry, material, seed, dt_bed).
    flow = bed(geometry, material, dt_bed)
    sha_after_settle = state_sha(flow)
    n_bed = flow.n
    bed_mass = flow.bed_mass
    rho_bulk = flow.bulk_density(case.fill_height)
    t_build = time.time() - t0

    dt0 = flow.dt
    if transform == "identity":
        dt_run = dt0
    elif transform == "up":
        dt_run = ulp(dt0, +1)
    elif transform == "down":
        dt_run = ulp(dt0, -1)
    elif transform == "finite":
        dt_run = dt0 * 1.001
    else:  # pragma: no cover
        raise SystemExit(f"unknown transform {transform}")

    if dt_run != dt0:
        flow = copy.deepcopy(flow)
        flow.dt = float(dt_run)

    t1 = time.time()
    res = run_discharge(case, flow=flow)
    wall_discharge = time.time() - t1

    rho_ref = rho_bulk
    w_ref = beverloo_prediction(case.orifice_diameter, case.particle_diameter, rho_ref)
    payload = dict(res)
    payload["archive"] = ARCHIVE.get(args.shard)
    payload["ulp_experiment"] = {
        "shard": args.shard,
        "bed_dt_spelling": bed_spelling,
        "discharge_dt_transform": transform,
        "dt_bed_s": dt_bed,
        "dt_run_s": float(dt_run),
        "dt_run_bits": struct.pack(">d", float(dt_run)).hex(),
        "dt_shift_ulps": (struct.unpack(">q", struct.pack(">d", float(dt_run)))[0]
                          - struct.unpack(">q", struct.pack(">d", dt0))[0]),
        "dt_shift_rel": (float(dt_run) - float(dt0)) / float(dt0),
        "dt0_of_bed_s": float(dt0),
        "spellings": {k: v for k, v in specs.items() if k != "material"},
        "seed": SEED,
        "frozen": FROZEN,
        "n_bed": n_bed,
        "bed_mass_kg": bed_mass,
        "rho_bulk_nominal_fill": rho_ref,
        "bed_state_sha256": sha_after_settle,
        "w_reference_kg_s": w_ref,
        "rel_error_vs_beverloo": res["flow_rate_kg_s"] / w_ref - 1.0,
        "wall_clock_bed_s": t_build,
        "wall_clock_discharge_s": wall_discharge,
        "meta": meta(),
    }
    if args.shard in ARCHIVE:
        wa = ARCHIVE[args.shard]["flow_rate_kg_s"]
        payload["ulp_experiment"]["archive_flow_rate_kg_s"] = wa
        payload["ulp_experiment"]["archive_rel_diff"] = res["flow_rate_kg_s"] / wa - 1.0

    out = Path(args.out or f"runs/m23261_ulp_dt/{args.shard}.json")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    st = res["discharge_status"]
    print(f"{args.shard}: dt={dt_run:.17e} W={res['flow_rate_kg_s']:.9f} kg/s "
          f"rel={payload['ulp_experiment']['rel_error_vs_beverloo']:+.2%} "
          f"{st['status']} windows={res['window_particles']} "
          f"bed={sha_after_settle[:12]} wall={wall_discharge:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())