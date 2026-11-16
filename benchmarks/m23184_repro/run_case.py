#!/usr/bin/env python3
"""Re-run one archived Beverloo case at the *current* checkout (m23184 audit).

Motivation (see `prereg.md`): the same frozen case and the same seed are archived
twice in the repository with a factor ~9 disagreement in the discharge rate —

    m22885 `fill36_seed1.json`  W = 0.030750 kg/s  (+5.5 % vs Beverloo, PASS)
    m23114 `dt_nom20_s1.json`   W = 0.003358 kg/s  (-88.5 %, plugged bed)

while the two initial bed masses agree bit for bit (`0.08623671834103983 kg`,
`rho_bulk = 1323.7847222`).  This driver re-executes the case from the current
checkout, records the bed state checksum, and stores enough of the run to decide
whether the archived numbers are reproducible at all.

    PYTHONPATH=src python benchmarks/m23184_repro/run_case.py --seed 1 \
        --out runs/m23184_repro/seed1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import (  # noqa: E402
    DischargeCase, beverloo_prediction, run_discharge,
)
from tensordem.silo_flow.kernel import Material, contact_time  # noqa: E402

#: frozen primary case of the campaign (identical to m22885 run_case.py / m23114
#: prereg §2 — *no* parameter is changed here)
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
    kn=5.0e3,
    mu=0.50,
    mu_rolling=0.05,
)


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    case = DischargeCase(seed=args.seed, **FROZEN)
    material = Material(radius=0.5 * case.particle_diameter, mu=case.mu,
                        mu_rolling=case.mu_rolling, kn=case.kn)
    dt = contact_time(material.mass_eff, case.kn) / 20.0   # campaign resolution T_c/20

    t0 = time.time()
    res = run_discharge(case, dt=dt)
    wall = time.time() - t0

    payload = dict(res)
    payload["w_ref_kg_s"] = beverloo_prediction(
        case.orifice_diameter, case.particle_diameter, res["rho_bulk"])
    payload["w_ref_measured_rho_kg_s"] = beverloo_prediction(
        case.orifice_diameter, case.particle_diameter, res["rho_bulk_measured"])
    payload["rel_error_nominal_rho"] = payload["flow_rate_kg_s"] / payload["w_ref_kg_s"] - 1.0
    payload["rel_error_measured_rho"] = (
        payload["flow_rate_kg_s"] / payload["w_ref_measured_rho_kg_s"] - 1.0)
    payload["m23184"] = {
        "seed": int(args.seed),
        "frozen": FROZEN,
        "dt_s": dt,
        "steps_per_contact": 20.0,
        "wall_clock_s": wall,
        "meta": meta(),
        "reproduces": {
            "m22885_fill36_seed1": 0.030749793997350797,
            "m23114_dt_nom20_s1": 0.0033577361261474805,
        },
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"seed {args.seed}: n_bed={payload['n_bed']} "
          f"W={payload['flow_rate_kg_s']:.6f} kg/s "
          f"(W_ref={payload['w_ref_kg_s']:.6f}) "
          f"rel={payload['rel_error_nominal_rho']:+.4f} "
          f"exited={payload['particles_exited']} wall={wall:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())