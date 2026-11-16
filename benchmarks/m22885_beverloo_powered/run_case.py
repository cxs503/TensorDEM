"""Run one (fill height, seed) shard of the m22885 Beverloo ensemble.

Thin wrapper around the campaign's own DEM driver — the physics is
`tensordem.silo_flow.study beverloo`, nothing is re-implemented here:

    python3 benchmarks/m22885_beverloo_powered/run_case.py --fill 36 --seed 3 \
        --out runs/m22885_beverloo_powered/fill36_seed3.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FROZEN = {
    "particle_diameter": 0.003,
    "ratio": 5.0,            # D/d
    "silo_radius": 0.024,    # silo_D/D = 3.2
    "kn": 5.0e3,
    "mu": 0.50,
    "wall_mu": 0.0,          # flat bottom plate, Beverloo smooth-wall condition
    "settle_time": 0.06,
    "discard_time": 0.12,
    "window_time": 0.05,
    "n_windows": 8,
}


def fill_m(fill_mm: float) -> float:
    return float(fill_mm) / 1000.0


def command(fill_mm: float, seed: int, out: Path) -> list[str]:
    return [
        sys.executable, "-m", "tensordem.silo_flow.study", "beverloo",
        "--ratio", str(FROZEN["ratio"]),
        "--particle-diameter", str(FROZEN["particle_diameter"]),
        "--silo-radius", str(FROZEN["silo_radius"]),
        "--fill-height", str(fill_m(fill_mm)),
        "--kn", str(FROZEN["kn"]),
        "--mu", str(FROZEN["mu"]),
        "--wall-mu", str(FROZEN["wall_mu"]),
        "--settle-time", str(FROZEN["settle_time"]),
        "--discard-time", str(FROZEN["discard_time"]),
        "--window-time", str(FROZEN["window_time"]),
        "--n-windows", str(FROZEN["n_windows"]),
        "--seed", str(int(seed)),
        "--out", str(out),
    ]


def dimensionless(fill_mm: float) -> dict:
    """Beverloo validity checks (prereg.md §2)."""
    d = FROZEN["particle_diameter"]
    D = FROZEN["ratio"] * d
    H = fill_m(fill_mm)
    return {
        "D_over_d": D / d,
        "silo_diameter_over_D": 2.0 * FROZEN["silo_radius"] / D,
        "fill_over_D": H / D,
        "in_domain": bool(D / d >= 4.0 and 2.0 * FROZEN["silo_radius"] / D >= 2.5 and H / D >= 1.5),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fill", type=float, required=True, help="fill height in mm")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-command", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(args.fill, args.seed, out)
    if args.print_command:
        print(" ".join(cmd))
        return 0
    if subprocess.run(cmd, cwd=str(ROOT), env={**__import__("os").environ,
                                               "PYTHONPATH": str(ROOT / "src")}).returncode != 0:
        return 1
    payload = json.loads(out.read_text())
    payload["m22885"] = {"frozen": FROZEN, "domain": dimensionless(args.fill),
                         "fill_height_mm": args.fill, "seed": args.seed}
    out.write_text(json.dumps(payload, indent=1))
    print(f"fill {args.fill} mm seed {args.seed}: n_bed={payload['n_bed']} "
          f"W={payload['flow_rate_kg_s']:.6f} kg/s "
          f"rho_b(meas)={payload['rho_bulk_measured']:.1f} "
          f"in_domain={payload['m22885']['domain']['in_domain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())