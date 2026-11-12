"""Rate / budget / size scan of the Lubachevsky-Stillinger-style inflation quench.

Why this scan exists
--------------------
The open anchor ``CASE-DEM-RCP-PACKING-FRACTION`` reports random-close packing but
two independent audits found the deliverable holes:

* ``docs/AUDIT_m16308_CASE-DEM-RCP-PACKING-FRACTION.md``: the *geometric* route
  (``tensordem.bedcoupling.packing.make_random_close_packing``) returns
  ``solid_fraction == 1 - target_porosity`` to 1e-16 -- the "0.53 % agreement with
  0.6366" is a restatement of its input, i.e. a tautological gate.  That audit's
  remedy list asks for a *hard-sphere-inflation* measurement, an unbiased quantity,
  and the two tiers at the anchor's stated 10 000-sphere scale.
* ``docs/P152W1-N04_friction_contact_surface.md``: the inflation route
  (:mod:`tensordem.friction_surface.packing`) measures ``phi = 0.6160`` at
  N = 500/1000, i.e. **3.75 % below** the 0.6366 reference -- the old report passed
  it against a self-declared 5 % tolerance, not the 3 % rule.

The inflation route is honest (it reports an overlap state that is not an input),
so the open question is quantitative, not conceptual: **is the missing 3.75 %
a finite-rate / finite-relaxation-budget artefact of the quench, or a real bias of
the protocol?**  This scan answers it by measuring the two dependencies the
protocol has by construction and extrapolating them away:

* inflation rate  ``rho_growth``  (2e-4 ... 2.5e-5, i.e. 8x slower)
* relaxation budget ``relax_steps`` (30 ... 240, i.e. 8x more work per step)
* finite size     ``n_particles`` (500 ... 8000)
* jamming criterion ``overlap_stop`` (1e-4 ... 1e-2)

Usage
-----
    python3 benchmarks/rcp_ls_extrapolation/run_scan.py --list
    python3 benchmarks/rcp_ls_extrapolation/run_scan.py --shard 3 --shards 17
    python3 benchmarks/rcp_ls_extrapolation/analyse.py

Each configuration writes one JSON to ``benchmarks/rcp_ls_extrapolation/runs/raw/``
and is skipped on re-entry, so the scan is resumable and shardable across cluster
nodes.  Deterministic: the seed is part of the configuration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from tensordem.friction_surface.packing import PackingCase, random_close_packing

HERE = Path(__file__).resolve().parent
RAW = HERE / "runs" / "raw"


def configs() -> list[dict]:
    """The pre-registered scan matrix (written down before any run)."""
    out: list[dict] = []

    def add(tag: str, **kw) -> None:
        base = dict(n_particles=1000, rho_growth=2.0e-4, relax_steps=60, seed=1,
                    overlap_stop=1.0e-3, max_inflation_steps=60000)
        base.update(kw)
        base["tag"] = tag
        out.append(base)

    # A: inflation-rate scan (the quasi-static limit), two seeds
    for rate, rtag in ((2.0e-4, "2e-4"), (1.0e-4, "1e-4"), (5.0e-5, "5e-5"),
                       (2.5e-5, "2.5e-5")):
        for seed in (1, 2):
            add(f"rate_{rtag}_seed{seed}", rho_growth=rate, seed=seed)
    # B: relaxation-budget scan at fixed rate
    for relax in (30, 120, 240):
        add(f"relax_{relax}", relax_steps=relax, rho_growth=1.0e-4)
    # B2: deep-relaxation extension (the budget axis must reach a plateau, otherwise
    # "extrapolate the budget away" is unsupported).  Run at the fast rate so the
    # configuration stays inside one cluster time slice.
    add("relax_400_g2e-4", relax_steps=400)
    add("relax_800_g2e-4", relax_steps=800)
    add("relax_400_g2e-4_seed2", relax_steps=400, seed=2)
    add("relax_400_g1e-4", relax_steps=400, rho_growth=1.0e-4)
    # B3: cheap (N=500) deep-budget points -- N-dependence at fixed protocol was
    # measured to be <= 5e-4 (see the size scan), so the plateau is read at N=500.
    add("relax_400_n500", relax_steps=400, n_particles=500)
    add("relax_800_n500", relax_steps=800, n_particles=500)
    # C: finite-size scan at the fastest rate
    for n in (500, 2000, 4000, 8000, 10000):
        add(f"size_{n}", n_particles=n)
    # D: jamming-criterion sensitivity
    for stop, stag in ((1.0e-4, "1e-4"), (1.0e-2, "1e-2")):
        add(f"stop_{stag}", overlap_stop=stop, rho_growth=1.0e-4)
    return out


def config_id(cfg: dict) -> str:
    return (f"n{cfg['n_particles']}_g{cfg['rho_growth']:.3g}_r{cfg['relax_steps']}"
            f"_s{cfg['seed']}_o{cfg['overlap_stop']:.2g}")


def run_one(cfg: dict) -> dict:
    case = PackingCase(
        n_particles=int(cfg["n_particles"]),
        rho_growth=float(cfg["rho_growth"]),
        relax_steps=int(cfg["relax_steps"]),
        overlap_stop=float(cfg["overlap_stop"]),
        seed=int(cfg["seed"]),
        max_inflation_steps=int(cfg["max_inflation_steps"]),
    )
    t0 = time.time()
    res = random_close_packing(case)
    wall = time.time() - t0
    hist = res.pop("history")
    # Keep the tail of the inflation curve (the jammed region) for the diagnostics.
    tail = hist[-400:]
    res.update(
        config_id=config_id(cfg),
        tag=cfg["tag"],
        wall_seconds=wall,
        seconds_per_inflation_step=wall / max(1, res["inflation_steps"]),
        history_tail=[[float(p), float(o), float(pr)] for p, o, pr in tail],
        host=platform.node(),
        numpy=platform.python_version() + "/np" + np.__version__,
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None,
                    help="restrict to configurations whose config_id contains this string")
    ap.add_argument("--tag", action="append", default=None,
                    help="restrict to these configuration tags (repeatable)")
    args = ap.parse_args()

    cfgs = configs()
    if args.only:
        cfgs = [c for c in cfgs if args.only in config_id(c)]
    if args.tag:
        cfgs = [c for c in cfgs if c["tag"] in set(args.tag)]
    if args.list:
        for c in cfgs:
            print(config_id(c), json.dumps(c, sort_keys=True))
        print(f"# {len(cfgs)} configurations")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    mine = [c for i, c in enumerate(cfgs) if i % max(1, args.shards) == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(mine)}/{len(cfgs)} configurations",
          flush=True)
    for cfg in mine:
        path = RAW / f"{config_id(cfg)}.json"
        if path.exists():
            print(f"skip (cached) {path.name}", flush=True)
            continue
        print(f"run {config_id(cfg)} ...", flush=True)
        try:
            res = run_one(cfg)
        except Exception as exc:  # noqa: BLE001 - one bad config must not kill the shard
            res = dict(config_id=config_id(cfg), tag=cfg["tag"], config=cfg,
                       status="error", error=repr(exc), host=platform.node())
        else:
            res.update(config=cfg, status="ok")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, path)
        if res.get("status") == "ok":
            phi_j = res["jammed_packing_fraction"]
            print(f"  -> phi_J = {phi_j}  steps={res['inflation_steps']}  "
                  f"wall={res['wall_seconds']:.1f}s  "
                  f"overlap={res['mean_overlap_ratio']:.3e}", flush=True)
        else:
            print(f"  -> ERROR {res['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())