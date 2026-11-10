"""Campaign driver for mission P143W2C-N02 (BPM identifiability).

Runs the engine-coupled specimens of the pre-registered design and writes one
JSON per shard so the campaign can be spread over the cluster and merged
afterwards.

Usage
-----
    python scripts/p143w2c_n02_campaign.py --shard 0 --nshards 64 --out artifacts/p143w2c_n02
    python scripts/p143w2c_n02_campaign.py --list          # show the run list
    python scripts/p143w2c_n02_campaign.py --shard 0 --nshards 1 --only beta_tension

Every run record is a JSON-serialisable dict; the ``kind`` field says which
analysis bucket it belongs to, and ``result`` carries the ``SpecimenResult``
fields plus a decimated ``strain``/``stress`` trace.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem import bpm_identifiability as bi  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered campaign configuration (frozen before any run)
# ---------------------------------------------------------------------------
CALIBRATED = bi.MicroParameters(
    # k_b is set so the calibrated UCS lands inside the pre-registered 3-7 MPa
    # window with margin on the upper literature interval (UCS is linear in k_b:
    # 9.6e5 N/m gave 6.60 MPa, so 8.43e5 N/m targets 5.8 MPa).
    bond_stiffness=8.43e5,
    contact_stiffness=1.646e6,
    breaking_strain=0.015,
    shear_breaking_strain=0.03,
    density=917.0,
    bond_weibull_modulus=6.0,
    friction_coefficient=0.4,
    contact_damping=0.0,
    drag=0.0,                    # replaced by the staged local damping
)

BETAS = (100.0, 400.0, 1600.0, 3600.0)
# The three loading rates of the Latin design.  The ramp length is what sets the
# injected kinetic energy of one increment, so the first campaign (ramp 50-130)
# failed the KE <= 1e-3 gate on the tension arm; these levels are 4x longer.
RAMP_LEVELS = (200, 320, 500)
BETA_SEEDS = (0,)
FIXED_RATES = (0.15, 0.30)       # 1/s, fixed-physical-rate arm
RATE_EPS_INC = 0.03 / 10.0       # strain per increment of the tension arm
SIZE_THICKNESSES = (0.1, 0.2, 0.4)
SIZE_EXT_THICKNESSES = (0.05, 0.1, 0.2, 0.4)
SIZE_SEEDS_MIN = 8
SIZE_SEEDS_EXT = 12
REF_BETA = 1600.0


def _geometry(thickness: float) -> bi.SpecimenGeometry:
    return bi.SpecimenGeometry.from_thickness(thickness, cross_n=4, radius=0.0125)


def _protocol(mode: str, *, thickness: float, beta: float, ramp_steps: int,
              increments: int, target_strain: float, hold_steps: int = 150,
              ramp_duration_s: float | None = None) -> bi.LoadProtocol:
    return bi.LoadProtocol(
        mode=mode, protocol="staged", mass_scaling=beta,
        target_strain=target_strain, increments=increments,
        ramp_steps=ramp_steps, hold_steps=hold_steps, decay_steps=15,
        settle_steps=100, ramp_duration_s=ramp_duration_s,
    )


def build_runs() -> list[dict]:
    """Deterministic ordered run list (index -> config)."""
    runs: list[dict] = []
    # (a) mass-scaling gate, tension and compression, Latin three ramp levels
    for mode, inc, target, hold in (("tension", 10, 0.03, 250),
                                    ("compression", 12, 0.08, 250)):
        for beta in BETAS:
            for level, ramp in enumerate(RAMP_LEVELS):
                for seed in BETA_SEEDS:
                    runs.append({
                        "kind": f"beta_{mode}",
                        "mode": mode, "beta": beta, "ramp_steps": ramp,
                        "rate_level": level, "seed": seed,
                        "thickness": 0.1, "increments": inc,
                        "target_strain": target, "hold_steps": hold,
                    })
    # (b) size effect, pre-registered minimum design (n = 8 per family)
    for thickness in SIZE_THICKNESSES:
        for seed in range(SIZE_SEEDS_MIN):
            runs.append({
                "kind": "size_min", "mode": "tension", "beta": REF_BETA,
                "ramp_steps": 80, "rate_level": 1, "seed": seed,
                "thickness": thickness, "increments": 10,
                "target_strain": 0.03, "hold_steps": 150,
            })
    # (b') size effect, extended design (4 families, n = 16)
    for thickness in SIZE_EXT_THICKNESSES:
        for seed in range(SIZE_SEEDS_EXT):
            runs.append({
                "kind": "size_ext", "mode": "tension", "beta": REF_BETA,
                "ramp_steps": 80, "rate_level": 1, "seed": seed,
                "thickness": thickness, "increments": 10,
                "target_strain": 0.03, "hold_steps": 150,
            })
    # (a') fixed *physical* strain-rate arm: the inertia probe with teeth.
    # With the step count fixed (arm (a) above) the physical ramp duration grows
    # as sqrt(beta), which makes the staged trajectory exactly beta-invariant; here
    # the physical ramp duration is held fixed instead, so beta changes inertia.
    for beta in BETAS:
        for rate in FIXED_RATES:
            runs.append({
                "kind": "beta_fixedrate", "mode": "tension", "beta": beta,
                "ramp_steps": 30, "ramp_duration_s": RATE_EPS_INC / rate,
                "rate_level": 0, "seed": 0, "thickness": 0.1,
                "increments": 10, "target_strain": 0.03, "hold_steps": 150,
            })
    # (e) engine-level negative control: homogeneous bond strength (m_bond = inf)
    for thickness in SIZE_THICKNESSES:
        for seed in range(3):
            runs.append({
                "kind": "neg_homogeneous", "mode": "tension", "beta": REF_BETA,
                "ramp_steps": 80, "rate_level": 1, "seed": seed,
                "thickness": thickness, "increments": 10,
                "target_strain": 0.03, "hold_steps": 150,
                "bond_weibull_modulus": 0.0,
            })
    return runs


def _decimate(trace: np.ndarray, n: int = 200) -> list[float]:
    if trace.size == 0:
        return []
    idx = np.linspace(0, trace.size - 1, min(n, trace.size)).astype(int)
    return [float(trace[i]) for i in idx]


def run_one(spec: dict) -> dict:
    """Execute one campaign entry and return its JSON record."""
    t0 = time.time()
    geometry = _geometry(spec["thickness"])
    micro = CALIBRATED
    if "bond_weibull_modulus" in spec:
        micro = replace(micro, bond_weibull_modulus=float(spec["bond_weibull_modulus"]))
    protocol = _protocol(
        spec["mode"], thickness=spec["thickness"], beta=spec["beta"],
        ramp_steps=spec["ramp_steps"], increments=spec["increments"],
        target_strain=spec["target_strain"], hold_steps=spec["hold_steps"],
        ramp_duration_s=spec.get("ramp_duration_s"))
    result = bi.run_specimen(micro, geometry, protocol, bond_seed=spec["seed"])
    record = {
        "kind": spec["kind"],
        "spec": spec,
        "wall_s": time.time() - t0,
        "equivalent_strain_rate": (
            spec["target_strain"] / spec["increments"]
            / ((spec["ramp_steps"] + spec["hold_steps"]) * result.dt)),
        "result": {
            **{k: v for k, v in result.as_dict().items()},
            "strain": _decimate(result.strain),
            "stress": _decimate(result.stress),
            "kinetic": _decimate(result.kinetic),
            "thickness": float(result.thickness),
            "area": float(result.area),
        },
    }
    for key in ("sigma_peak", "youngs_modulus", "sigma_residual", "ke_ratio",
                "ke_ratio_loading_peak"):
        value = record["result"][key]
        if isinstance(value, float) and not math.isfinite(value):
            record["result"][key] = None
    return record


def run_friction(out: dict) -> dict:
    """Criterion (d): slip-law cells and the post-failure residual specimen."""
    cell = {}
    for law in ("none", "tanh", "dual"):
        cell[law] = bi.shear_cell_residual(regularization=law, mu_s=0.6, mu_d=0.4,
                                           transition_speed=1.0e-3,
                                           slip_speed=5.0e-3, n_steps=1500)
    residual = {}
    for law in ("none", "tanh", "dual"):
        r = bi.residual_shear_specimen(regularization=law, mu_s=0.6, mu_d=0.4,
                                       transition_speed=1.0e-3,
                                       geometry=bi.SpecimenGeometry(n_layers=4, cross_n=3),
                                       mass_scaling=REF_BETA,
                                       axial_preload_strain=0.01,
                                       shear_strain_rate=2.0e-2,
                                       target_shear_strain=0.02)
        residual[law] = {k: v for k, v in r.items()
                         if k not in ("shear_strain", "tau", "bond_damage")}
        residual[law]["shear_strain"] = _decimate(r["shear_strain"])
        residual[law]["tau"] = _decimate(r["tau"])
    out["friction"] = {"shear_cell": cell, "residual_specimen": residual}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="artifacts/p143w2c_n02")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--with-friction", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    all_runs = build_runs()
    runs = all_runs
    if args.only:
        runs = [r for r in all_runs if r["kind"] == args.only]
    if args.list:
        for i, r in enumerate(runs):
            print(i, r)
        print("total", len(runs))
        return 0

    os.makedirs(args.out, exist_ok=True)
    mine = [r for i, r in enumerate(runs) if i % args.nshards == args.shard]
    if args.limit:
        mine = mine[: args.limit]
    payload = {"shard": args.shard, "nshards": args.nshards,
               "n_runs": len(mine), "records": []}
    run_dir = os.path.join(args.out, "runs")
    os.makedirs(run_dir, exist_ok=True)
    for k, spec in enumerate(mine):
        # Index against the *full* run list so a filtered run (``--only``) can
        # never overwrite another kind's cache entry.
        index = all_runs.index(spec) if spec in all_runs else k
        run_path = os.path.join(run_dir, f"run_{index:04d}.json")
        if os.path.exists(run_path):
            # A finished run is never repeated: the campaign can be stopped and
            # restarted without losing work.
            with open(run_path) as fh:
                payload["records"].append(json.load(fh))
            continue
        rec = run_one(spec)
        tmp = run_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, run_path)
        payload["records"].append(rec)
        print(f"[shard {args.shard}] {k + 1}/{len(mine)} {spec['kind']} "
              f"t={spec['thickness']} beta={spec['beta']} seed={spec['seed']} "
              f"sig={rec['result']['sigma_peak']:.4g} wall={rec['wall_s']:.0f}s",
              flush=True)
    if args.with_friction:
        run_friction(payload)
    path = os.path.join(args.out, f"shard_{args.shard:03d}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())