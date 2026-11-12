"""Parallel runner for the plane-shear mu(I) sweep (m16427 round 2).

Each inertial-number point is independent, so the six points are farmed out to
separate processes; the merged JSON has the same schema as ``run_sweep.py``.

    python benchmarks/plane_shear_mu_i/run_parallel.py --arm baseline --jobs 6
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_sweep as rs  # noqa: E402


def _point(args):
    I, kn = args
    return rs.run_point(I, kn=kn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(rs.ARMS), default="baseline")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arm = rs.ARMS[a.arm]
    kn = float(arm["kn"])
    out_name = a.out or f"sweep_{a.arm}.json"
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(_point, [(I, kn) for I in rs.I_TARGETS]))
    rows.sort(key=lambda r: r["I_target"])
    payload = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 2,
        "device": "tensordem.mu_i_rheology.simulate_servo_shear",
        "arm": a.arm,
        "arm_label": arm["label"],
        "config": {"P_pa": rs.P, "kn": kn, "mu_contact": rs.MU_CONTACT,
                   "mu_rolling": rs.MU_ROLLING, "e": rs.E, "r": rs.R,
                   "rho": rs.RHO, "box": [rs.W, rs.D, rs.H],
                   "shear_strain": rs.SHEAR_STRAIN, "seed": rs.SEED,
                   "packing_gap": 0.02},
        "reference": {"mu_s": rs.MU_S, "mu_2": rs.MU_2, "I_0": rs.I_0,
                      "phi_c": rs.PHI_C, "phi_j": rs.PHI_J,
                      "source": "da Cruz et al. 2005 PRE 72:021309; "
                                "GDR MiDi 2004 JFM 509:165"},
        "rows": rows,
    }
    (rs.OUT / out_name).write_text(json.dumps(payload, indent=1))
    print("WROTE", rs.OUT / out_name, flush=True)


if __name__ == "__main__":
    main()