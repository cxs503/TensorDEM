#!/usr/bin/env python3
"""Run one packing seed of the frozen m22657 Beverloo ensemble case.

The configuration is frozen in ``prereg.md`` §3; this wrapper exists so the
cluster command is one auditable line per shard and so the parameters cannot
drift between seeds::

    python3 benchmarks/m22657_beverloo_ensemble/run_case.py --seed 3 \
        --out runs/m22657_beverloo_ensemble/seed3.json

It shells out to the campaign kernel driver
(``python -m tensordem.silo_flow.study beverloo``) rather than reimplementing
the case, so the DEM kernel and the JSON schema stay the ones the rest of the
Beverloo campaign already uses.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- frozen configuration (prereg.md §3) -------------------------------------
FROZEN = {
    "ratio": 5.0,            # D/d
    "particle_diameter": 0.003,
    "silo_radius": 0.024,    # silo_D/D = 3.2
    "fill_height": 0.036,    # H/D = 2.4
    "kn": 5000.0,
    "mu": 0.50,
    "wall_mu": 0.0,
    "settle_time": 0.06,
    "discard_time": 0.12,
    "window_time": 0.05,
    "n_windows": 8,
}


def command(seed: int, out: str | Path) -> list[str]:
    cmd = [sys.executable, "-m", "tensordem.silo_flow.study", "beverloo",
           "--ratio", str(FROZEN["ratio"]),
           "--particle-diameter", str(FROZEN["particle_diameter"]),
           "--silo-radius", str(FROZEN["silo_radius"]),
           "--fill-height", str(FROZEN["fill_height"]),
           "--kn", str(FROZEN["kn"]),
           "--mu", str(FROZEN["mu"]),
           "--wall-mu", str(FROZEN["wall_mu"]),
           "--settle-time", str(FROZEN["settle_time"]),
           "--discard-time", str(FROZEN["discard_time"]),
           "--window-time", str(FROZEN["window_time"]),
           "--n-windows", str(FROZEN["n_windows"]),
           "--seed", str(int(seed)),
           "--out", str(out)]
    return cmd


def dimensionless(orifice: float, particle: float, silo_radius: float, fill_height: float) -> dict:
    """Domain checks against Beverloo's stated validity range (prereg.md §3)."""
    return {
        "D_over_d": orifice / particle,
        "silo_diameter_over_D": 2.0 * silo_radius / orifice,
        "fill_over_D": fill_height / orifice,
        "in_domain": bool(orifice / particle >= 4.0
                          and 2.0 * silo_radius / orifice >= 2.5
                          and fill_height / orifice >= 1.5),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-command", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(args.seed, out)
    if args.print_command:
        print(" ".join(cmd))
        return 0
    env_cmd = "PYTHONPATH=%s %s" % (ROOT / "src", " ".join(cmd))
    proc = subprocess.run(env_cmd, shell=True, cwd=str(ROOT))
    if proc.returncode != 0:
        return proc.returncode
    dom = dimensionless(FROZEN["ratio"] * FROZEN["particle_diameter"],
                        FROZEN["particle_diameter"], FROZEN["silo_radius"],
                        FROZEN["fill_height"])
    payload = json.loads(out.read_text())
    payload["m22657"] = {"frozen": FROZEN, "domain": dom, "seed": args.seed}
    out.write_text(json.dumps(payload, indent=1))
    print(f"seed {args.seed}: n_bed={payload['n_bed']} "
          f"W={payload['flow_rate_kg_s']:.6f} kg/s in_domain={dom['in_domain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())