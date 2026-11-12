"""Parallel runner for the plane-shear mu(I) sweep (round 3, m16497).

Each inertial-number point is independent, so the points are farmed out to
separate processes; the merged JSON has the same schema as ``run_sweep.py``,
and the payload is also echoed to stdout so a cluster log is self-contained
(the job artefact is not always fetchable).

    python benchmarks/plane_shear_mu_i/run_parallel.py --arm baseline --jobs 6
    python benchmarks/plane_shear_mu_i/run_parallel.py --arm stiff_thick --jobs 3 \
        --points 1e-3,1e-2,1e-1
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
    I, kn, box_depth, fill_height = args
    return rs.run_point(I, kn=kn, box_depth=box_depth, fill_height=fill_height)


def parse_points(text: str | None):
    if not text:
        return list(rs.I_TARGETS)
    return [float(token) for token in text.replace(";", ",").split(",") if token.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=tuple(rs.ARMS), default="baseline")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--points", default=None,
                    help="comma separated inertial-number targets (default: all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arm = rs.ARMS[a.arm]
    kn = float(arm["kn"])
    box_depth = float(arm.get("box_depth", rs.D))
    fill_height = float(arm.get("fill_height", rs.H))
    points = parse_points(a.points)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(_point, [(I, kn, box_depth, fill_height) for I in points]))
    rows.sort(key=lambda r: r["I_target"])
    payload = rs.payload_for(a.arm, rows, points)
    out_name = a.out or f"sweep_{a.arm}.json"
    (rs.OUT / out_name).write_text(json.dumps(payload, indent=1))
    print("WROTE", rs.OUT / out_name, flush=True)
    for r in rows:
        print(f"[{a.arm}] I={r['I_target']:g} I_m={r['I_measured']:.4g} "
              f"mu={r['mu_measured']:.4f} (ref {r['ref_mu']:.4f}, "
              f"{100*(r['mu_measured']/r['ref_mu']-1):+.1f}%) "
              f"p/p_tgt={1.0+r['pressure_control_error']:.3f} "
              f"phi_steady={r['phi_bed_steady']:.4f} (ref {r['ref_phi']:.4f}) "
              f"Z={r['coordination']:.2f} N={r['n_particles']} "
              f"wc={r['wall_clock_s']:.0f}s", flush=True)
    print("PAYLOAD_JSON_BEGIN", flush=True)
    print(json.dumps(payload), flush=True)
    print("PAYLOAD_JSON_END", flush=True)


if __name__ == "__main__":
    main()