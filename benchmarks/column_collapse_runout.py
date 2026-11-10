"""Criterion (c)(d)(e) runner: granular column collapse runout / mu(I) / ledger.

Usage (from the repository root)::

    python benchmarks/column_collapse_runout.py --aspects 1,2,3,4,6 --width 0.09 \
        --shard-index 0 --shard-count 1 --out runs/column_collapse/collapse_0.json

The script writes a JSON record with, per run, the measured initial geometry, the
dimensionless runout, the full potential-energy / dissipation ledger and the
coarse-grained (I, mu, phi) samples of the flowing layer.  The aggregate fit
(exponent of the runout law, ``mu(I)`` and ``phi(I)`` laws) is computed by
``--aggregate`` over a glob of shard files, so a scan can be sharded over nodes
without any shared state.

Every number in the output is produced here (no hand copying into the report).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tensordem.granular2d.column_collapse import (
    ColumnSpec,
    fit_mu_inertial,
    fit_phi_inertial,
    fit_runout_scaling,
    run_collapse,
)


def _shard(values, index: int, count: int):
    return [v for k, v in enumerate(values) if k % count == index]


def run_scan(aspects, base: dict, *, sample_flow: bool) -> list[dict]:
    out = []
    for a in aspects:
        t0 = time.time()
        spec = ColumnSpec(aspect_ratio=float(a), **base)
        result = run_collapse(spec, sample_flow=sample_flow)
        rec = result.as_dict()
        rec["wall_seconds"] = round(time.time() - t0, 2)
        rec["kn"] = spec.kn
        rec["radius"] = spec.radius
        rec["width"] = spec.width
        rec["mu"] = spec.mu
        rec["mu_rolling"] = spec.mu_rolling
        rec["thickness"] = spec.thickness
        rec["density"] = spec.density
        rec["mu_i_points"] = result.mu_i_points
        rec["phi_i_points"] = result.phi_i_points
        rec["front_history"] = result.front_history
        rec["kinetic_history"] = result.kinetic_history
        out.append(rec)
        print(json.dumps({k: rec[k] for k in
                          ("aspect_ratio", "n_particles", "duration", "runout_dimensionless",
                           "energy_residual_relative", "energy_dissipated_relative",
                           "wall_seconds")}), flush=True)
    return out


def aggregate(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shard files match {pattern}")
    runs: list[dict] = []
    for f in files:
        payload = json.loads(pathlib.Path(f).read_text())
        runs.extend(payload["runs"])
    runs.sort(key=lambda r: r["aspect_ratio"])

    class _Proxy:
        """Minimal stand-in exposing what the fit helpers need."""

        def __init__(self, rec):
            self.aspect_ratio = rec["aspect_ratio"]
            self.runout_dimensionless = rec["runout_dimensionless"]

    proxies = [_Proxy(r) for r in runs]
    out: dict = {"shards": files, "n_runs": len(runs), "runs": runs}
    out["runout_fit"] = fit_runout_scaling(proxies, aspect_min=1.0)
    mu_points = [p for r in runs for p in r.get("mu_i_points", [])]
    phi_points = [p for r in runs for p in r.get("phi_i_points", [])]
    out["n_mu_points"] = len(mu_points)
    out["n_phi_points"] = len(phi_points)
    if len(mu_points) >= 4:
        out["mu_fit"] = fit_mu_inertial(mu_points)
    if len(phi_points) >= 4:
        out["phi_fit"] = fit_phi_inertial(phi_points)
    out["ledger_max_residual_relative"] = max(r["energy_residual_relative"] for r in runs)
    out["ledger_max_abs_residual_relative"] = max(
        abs(r["energy_residual_relative"]) for r in runs)
    out["mass_weighted_dissipation"] = float(np.sum(
        [r["energy_dissipated"] for r in runs]) / np.sum(
        [r["energy_potential_initial"] for r in runs]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspects", default="1,1.5,2,3,4,6")
    ap.add_argument("--width", type=float, default=0.09)
    ap.add_argument("--radius", type=float, default=0.003)
    ap.add_argument("--kn", type=float, default=5.0e3)
    ap.add_argument("--density", type=float, default=2500.0)
    ap.add_argument("--thickness", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--mu-rolling", type=float, default=0.02)
    ap.add_argument("--zeta", type=float, default=0.2)
    ap.add_argument("--max-time", type=float, default=1.5)
    ap.add_argument("--snapshot-interval", type=float, default=0.02)
    ap.add_argument("--sample-flow", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out", default="runs/column_collapse/collapse.json")
    ap.add_argument("--aggregate", default="")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args(argv)

    base = dict(width=args.width, radius=args.radius, kn=args.kn,
                density=args.density, thickness=args.thickness, mu=args.mu,
                mu_rolling=args.mu_rolling, zeta=args.zeta,
                max_time=args.max_time, snapshot_interval=args.snapshot_interval)
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("invalid shard arguments")

    if args.aggregate:
        payload = aggregate(args.aggregate)
    else:
        aspects = [float(s) for s in args.aspects.split(",") if s.strip()]
        aspects = _shard(aspects, args.shard_index, args.shard_count)
        payload = {"tag": args.tag, "spec": base, "shard_index": args.shard_index,
                   "shard_count": args.shard_count, "aspects": aspects}
        payload["runs"] = run_scan(aspects, base, sample_flow=args.sample_flow)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    if args.aggregate:
        summary = {k: payload[k] for k in
                   ("n_runs", "runout_fit", "n_mu_points", "n_phi_points",
                    "ledger_max_abs_residual_relative", "mass_weighted_dissipation")}
        if "mu_fit" in payload:
            summary["mu_fit"] = payload["mu_fit"]
        if "phi_fit" in payload:
            summary["phi_fit"] = payload["phi_fit"]
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())