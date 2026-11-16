#!/usr/bin/env python3
"""Run one level of the m23114 numerical ladder for the `silo_flow` Beverloo kernel.

The physics is *not* re-implemented here: every level calls
``tensordem.silo_flow.beverloo.run_discharge`` with the frozen campaign case of
`prereg.md` §2 and only the integration resolution (``dt``) or the contact
stiffness (``kn``) is varied.

    PYTHONPATH=src python benchmarks/m23114_siloflow_ladder/run_level.py \
        --axis dt --level nom40 --seed 1 --out runs/m23114_siloflow_ladder/dt_nom40_s1.json

Levels:
    dt axis : nom20 (default), nom40, nom80   -- steps per contact period 20/40/80
    kn axis : kn5e3, kn2e4, kn8e4 (N/m)       -- at 20 steps per contact period
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import DischargeCase, run_discharge  # noqa: E402
from tensordem.silo_flow.kernel import Material, contact_time  # noqa: E402

#: frozen case of prereg.md §2 == the campaign's primary case (m22885 §2)
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
)

#: steps per contact period, and stiffness, for every level name
LEVELS = {
    "nom20": dict(axis="dt", steps_per_contact=20, kn=5.0e3),
    "nom40": dict(axis="dt", steps_per_contact=40, kn=5.0e3),
    "nom80": dict(axis="dt", steps_per_contact=80, kn=5.0e3),
    "kn5e3": dict(axis="kn", steps_per_contact=20, kn=5.0e3),
    "kn2e4": dict(axis="kn", steps_per_contact=20, kn=2.0e4),
    "kn8e4": dict(axis="kn", steps_per_contact=20, kn=8.0e4),
}


def overlap_stats(kn: float) -> dict:
    """prereg.md §5: static and free-fall overlap, and the implied rate bias."""
    radius = 0.5 * FROZEN["particle_diameter"]
    mass = 2500.0 * 4.0 / 3.0 * math.pi * radius ** 3
    m_eff = 0.5 * mass
    g = 9.81
    d = FROZEN["particle_diameter"]
    D = FROZEN["orifice_diameter"]
    H = FROZEN["fill_height"]
    k_lit = 1.4
    delta_static = mass * g / kn
    v_o = math.sqrt(2.0 * g * H)
    delta_dyn = v_o * math.sqrt(m_eff / kn)
    eff = D - k_lit * d

    def bias(delta: float) -> float:
        return 2.5 * k_lit * 2.0 * delta / eff

    return {
        "delta_static_m": delta_static,
        "delta_dynamic_m": delta_dyn,
        "delta_static_over_d": delta_static / d,
        "delta_dynamic_over_d": delta_dyn / d,
        "orifice_velocity_m_s": v_o,
        "predicted_bias_static": bias(delta_static),
        "predicted_bias_dynamic": bias(delta_dyn),
        "bias_fraction_of_eff_orifice_static": 2.0 * delta_static / eff,
        "bias_fraction_of_eff_orifice_dynamic": 2.0 * delta_dyn / eff,
    }


def meta() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      cwd=str(ROOT)).strip()
    except Exception:
        sha = "unknown"
    return {
        "git_sha": sha,
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_case(level: str, seed: int) -> tuple[DischargeCase, float, int]:
    spec = LEVELS[level]
    case = DischargeCase(seed=seed, kn=float(spec["kn"]), **FROZEN)
    material = Material(radius=0.5 * case.particle_diameter, kn=case.kn,
                        mu=case.mu, mu_rolling=case.mu_rolling)
    t_c = contact_time(material.mass_eff, case.kn)
    dt = t_c / float(spec["steps_per_contact"])
    return case, dt, int(spec["steps_per_contact"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", required=True, choices=sorted(LEVELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-command", action="store_true",
                    help="print the equivalent module invocation and exit")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    case, dt, steps_per_contact = build_case(args.level, args.seed)
    if args.print_command:
        print(f"level={args.level} seed={args.seed} dt={dt:.6e} "
              f"steps_per_contact={steps_per_contact} kn={case.kn:.3e}")
        return 0

    t0 = time.time()
    res = run_discharge(case, dt=dt)
    wall = time.time() - t0

    payload = dict(res)
    payload["m23114"] = {
        "level": args.level,
        "axis": LEVELS[args.level]["axis"],
        "seed": int(args.seed),
        "steps_per_contact": steps_per_contact,
        "dt_requested_s": dt,
        "dt_used_s": res["dt"],
        "contact_period_s": contact_time(
            Material(radius=0.5 * case.particle_diameter, kn=case.kn).mass_eff, case.kn),
        "frozen": FROZEN,
        "kn": float(case.kn),
        "overlap": overlap_stats(float(case.kn)),
        "wall_clock_s": wall,
        "meta": meta(),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"{args.level} seed {args.seed}: n_bed={payload['n_bed']} "
          f"W={payload['flow_rate_kg_s']:.6f} kg/s dt={res['dt']:.3e} "
          f"steps={payload['steps']} wall={wall:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())