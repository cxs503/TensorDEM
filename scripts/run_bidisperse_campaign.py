#!/usr/bin/env python
"""P155-N09 campaign runner: bidisperse granular segregation.

Usage
-----
    python scripts/run_bidisperse_campaign.py --mode pilot --shard 0 --nshards 4 \
        --outdir runs/p155_pilot
    python scripts/run_bidisperse_campaign.py --mode pilot --merge \
        --outdir runs/p155_pilot --report artifacts/bidisperse_campaign.json

Each case is deterministic in its (kind, size ratio, stress/amplitude, seed) key,
so a sharded job and a single-core job produce bitwise identical records.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.bidisperse_segregation import (
    BidisperseConfig,
    ShearBoxCase,
    VibrationBoxCase,
    bootstrap_scaling_law,
    collapse_and_measure_repose,
    compute_segregation_degree,
    consolidation_response,
    fit_scaling_law,
    noise_floor,
    permeability_sensitivity_table,
    run_shear_box,
    run_vibration_box,
    stratified_permeability_profile,
)

RATIOS = (1.2, 1.5, 2.0, 3.0)
STRESSES = (200.0, 1000.0, 4000.0)      # Pa, sets the measured solid fraction
RATES = (3.0, 10.0)
GAMMAS = (2.0, 5.0, 10.0)              # A w^2 / g for the shaker
N_PARTICLES = 200


def case_list(mode: str) -> list[dict]:
    """Deterministic case list for a campaign mode."""
    cases: list[dict] = []
    if mode == "quick":
        for ratio in (1.0, 2.0):
            for seed in (1, 2):
                cases.append(dict(kind="shear", size_ratio=ratio,
                                  normal_stress=1000.0, shear_rate=10.0,
                                  seed=seed, target_strain=1.0,
                                  n_particles=60, compact_steps=4000))
        return cases

    if mode == "pilot":
        for ratio in (1.0, 2.0, 3.0):
            for seed in (1, 2, 3):
                cases.append(dict(kind="shear", size_ratio=ratio,
                                  normal_stress=4000.0, shear_rate=10.0,
                                  seed=seed, target_strain=6.0,
                                  n_particles=N_PARTICLES))
                cases.append(dict(kind="vibration", size_ratio=ratio,
                                  gamma=5.0, seed=seed, n_cycles=30,
                                  n_particles=N_PARTICLES))
        return cases

    if mode == "shear":
        for ratio in RATIOS:
            for sigma in STRESSES:
                for rate in RATES:
                    seeds = (1, 2, 3)
                    for seed in seeds:
                        cases.append(dict(kind="shear", size_ratio=ratio,
                                          normal_stress=sigma, shear_rate=rate,
                                          seed=seed, target_strain=6.0,
                                          n_particles=N_PARTICLES))
        # monodisperse negative control at each stress
        for sigma in STRESSES:
            for seed in (1, 2, 3):
                cases.append(dict(kind="shear", size_ratio=1.0,
                                  normal_stress=sigma, shear_rate=10.0,
                                  seed=seed, target_strain=6.0,
                                  n_particles=N_PARTICLES))
        return cases

    if mode == "vib":
        for ratio in RATIOS:
            for gamma in GAMMAS:
                for seed in (1, 2, 3):
                    cases.append(dict(kind="vibration", size_ratio=ratio,
                                      gamma=gamma, seed=seed, n_cycles=30,
                                      n_particles=N_PARTICLES))
        for gamma in GAMMAS:
            for seed in (1, 2, 3):
                cases.append(dict(kind="vibration", size_ratio=1.0,
                                  gamma=gamma, seed=seed, n_cycles=30,
                                  n_particles=N_PARTICLES))
        return cases

    if mode == "ratios":
        # size-ratio scan at the strongest drivers: Gamma = 10 (shaker) and
        # sigma = 4000 Pa (shear); three seeds each
        for ratio in RATIOS:
            for seed in (1, 2, 3):
                cases.append(dict(kind="vibration", size_ratio=ratio,
                                  gamma=10.0, seed=seed, n_cycles=20,
                                  n_particles=120))
                cases.append(dict(kind="shear", size_ratio=ratio,
                                  normal_stress=4000.0, shear_rate=10.0,
                                  seed=seed, target_strain=6.0,
                                  n_particles=120))
        return cases

    if mode == "probe":
        for ratio in (1.2, 3.0):
            for seed in (1, 2, 3, 4, 5):
                cases.append(dict(kind="shear", size_ratio=ratio,
                                  normal_stress=4000.0, shear_rate=10.0,
                                  seed=seed, target_strain=6.0,
                                  n_particles=120))
                for gamma in (2.0, 10.0):
                    cases.append(dict(kind="vibration", size_ratio=ratio,
                                      gamma=gamma, seed=seed, n_cycles=20,
                                      n_particles=120))
        return cases

    if mode == "deep":      # 10-seed blocks for the direction criterion
        for kind, ratio in (("shear", 1.2), ("shear", 3.0),
                            ("vibration", 1.2), ("vibration", 3.0)):
            for seed in range(1, 11):
                if kind == "shear":
                    cases.append(dict(kind="shear", size_ratio=ratio,
                                      normal_stress=4000.0, shear_rate=10.0,
                                      seed=seed, target_strain=6.0,
                                      n_particles=N_PARTICLES))
                else:
                    cases.append(dict(kind="vibration", size_ratio=ratio,
                                      gamma=5.0, seed=seed, n_cycles=30,
                                      n_particles=N_PARTICLES))
        return cases

    raise ValueError(f"unknown mode {mode!r}")


def case_key(spec: dict) -> str:
    tag = spec["kind"][:3]
    if spec["kind"] == "shear":
        tag += f"_r{spec['size_ratio']}_s{int(spec['normal_stress'])}" \
               f"_g{spec['shear_rate']}_n{spec['seed']}"
    else:
        tag += f"_r{spec['size_ratio']}_G{spec['gamma']}_n{spec['seed']}"
    return tag.replace(".", "p")


def run_one(spec: dict) -> dict:
    kind = spec["kind"]
    r = spec["size_ratio"]
    cfg = BidisperseConfig(d_small=0.010, d_large=r * 0.010,
                           volume_fraction=0.60,
                           n_particles=int(spec.get("n_particles", N_PARTICLES)),
                           seed=int(spec["seed"]))
    if kind == "shear":
        case = ShearBoxCase(
            config=cfg, shear_rate=float(spec["shear_rate"]),
            target_strain=float(spec.get("target_strain", 6.0)),
            normal_stress=float(spec["normal_stress"]),
            compact_steps=int(spec.get("compact_steps", 20000)),
            n_bins=12, sample_every=5000)
        rec = run_shear_box(case)
    elif kind == "vibration":
        gamma = float(spec["gamma"])
        freq = 30.0
        omega = 2.0 * math.pi * freq
        amp = gamma * 9.81 / omega**2
        case = VibrationBoxCase(config=cfg, amplitude=amp, frequency=freq,
                                n_cycles=int(spec.get("n_cycles", 30)),
                                n_bins=12, sample_every=2000)
        rec = run_vibration_box(case)
    else:
        raise ValueError(kind)

    # --- C4: mass / concentration conservation, measured not assumed --------
    areas = math.pi * rec.get("radii", np.array([0.0]))**2 if "radii" in rec else None
    rec["key"] = case_key(spec)
    rec["spec"] = {k: v for k, v in spec.items()}
    rec["mass_check"] = _mass_check(rec)
    return _clean(rec)


def _mass_check(rec: dict) -> dict:
    """Phi normalisation and total solid area, from the reported profile."""
    phi = np.asarray(rec["phi_z"], dtype=float)
    area = np.asarray(rec["area_z"], dtype=float)
    ok = np.isfinite(phi) & (area > 0)
    # weighted mean of Phi over the populated bins must return Phi_0
    mean_phi = float(np.sum(area[ok] * phi[ok]) / np.sum(area[ok])) if ok.any() else None
    resid = abs(mean_phi - rec["phi_0"]) if mean_phi is not None else None
    return {"weighted_phi": mean_phi, "phi_0": rec["phi_0"],
            "residual": resid, "n_bins_populated": int(ok.sum()),
            "total_area": float(np.sum(area))}


def _json_default(obj):
    """JSON fallback: numpy scalars/arrays and anything else via repr."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _clean(rec: dict) -> dict:
    """JSON-safe copy (arrays -> lists, numpy scalars -> python)."""
    out = {}
    for k, v in rec.items():
        if k in ("positions", "radii", "is_large", "history", "history_S",
                 "plate_force_history"):
            continue
        if isinstance(v, np.ndarray):
            out[k] = [None if not np.isfinite(x) else float(x) for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                           else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    if "history_S" in rec:
        out["history_S"] = [[float(h.get("strain", h.get("cycle", 0.0))),
                             float(h["S"])] for h in rec["history_S"]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        return merge(outdir, args.report)

    cases = case_list(args.mode)
    shard = args.shard % max(args.nshards, 1)     # tolerate 1-based {i}
    mine = [c for i, c in enumerate(cases) if i % args.nshards == shard]
    print(f"[shard {shard}/{args.nshards}] {len(mine)} of {len(cases)} cases",
          flush=True)
    for spec in mine:
        path = outdir / f"case_{case_key(spec)}.json"
        if path.exists():
            print("skip (exists)", path.name, flush=True)
            continue
        try:
            rec = run_one(spec)
        except Exception as exc:                      # keep the shard alive
            (outdir / f"FAILED_{case_key(spec)}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n")
            print("FAILED", case_key(spec), type(exc).__name__, exc, flush=True)
            continue
        path.write_text(json.dumps(rec, indent=1, default=float))
        print(f"done {path.name}  S={rec['segregation_degree']:.3f} "
              f"dir={rec['direction']} phi={rec.get('solid_fraction_measured', float('nan')):.3f}",
              flush=True)
    return 0


# ---------------------------------------------------------------------------
# Merge / analysis
# ---------------------------------------------------------------------------

def _load(outdir: pathlib.Path) -> list[dict]:
    recs = []
    for p in sorted(outdir.glob("case_*.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return recs


def _group(recs, keys):
    out = {}
    for r in recs:
        k = tuple(r.get(kk) for kk in keys)
        out.setdefault(k, []).append(r)
    return out


def _consensus(directions: list[int]) -> dict:
    n = len(directions)
    counts = {s: directions.count(s) for s in (-1, 0, 1)}
    best = max((-1, 0, 1), key=lambda s: counts[s])
    decisive = [d for d in directions if d != 0]
    decons = (max(decisive.count(-1), decisive.count(1)) / len(decisive)
              if decisive else 0.0)
    return {"n": n, "counts": counts, "dominant": best,
            "consensus": counts[best] / n if n else 0.0,
            "decisive_consensus": decons, "n_decisive": len(decisive)}


def merge(outdir: pathlib.Path, report_path: str | None) -> int:
    recs = _load(outdir)
    if not recs:
        print("no records", file=sys.stderr)
        return 1
    shear = [r for r in recs if r["kind"] == "shear"]
    vib = [r for r in recs if r["kind"] == "vibration"]

    summary = {}
    for name, group, keys in (
        ("shear", shear, ("size_ratio", "normal_stress", "shear_rate")),
        ("vibration", vib, ("size_ratio", "gamma")),
    ):
        rows = []
        for k, rs in sorted(_group(group, keys).items()):
            S = [float(r["segregation_degree"]) for r in rs]
            dirs = [int(r["direction"]) for r in rs]
            phis = [float(r.get("solid_fraction_measured", float("nan"))) for r in rs]
            row = {"key": dict(zip(keys, k)), "n_seeds": len(rs),
                   "S_mean": float(np.mean(S)), "S_std": float(np.std(S, ddof=1)) if len(S) > 1 else 0.0,
                   "S_values": S, "direction": _consensus(dirs),
                   "phi_mean": float(np.nanmean(phis)) if np.isfinite(phis).any() else None,
                   "S_over_noise": float(np.mean(S) / rs[0]["noise_floor"])}
            rows.append(row)
        summary[name] = rows

    # ---- scaling law S = A (ratio-1)^alpha, per stress / gamma -------------
    scaling = {}
    for name, rows, axis in (("shear", summary["shear"], "normal_stress"),
                             ("vibration", summary["vibration"], "gamma")):
        by_axis = {}
        for row in rows:
            if row["key"]["size_ratio"] == 1.0:
                continue
            by_axis.setdefault(row["key"][axis], []).append(row)
        fits = {}
        for a, rs in by_axis.items():
            rs = sorted(rs, key=lambda x: x["key"]["size_ratio"])
            if len(rs) < 2:
                continue
            ratios = [x["key"]["size_ratio"] for x in rs]
            groups = [x["S_values"] for x in rs]
            try:
                fits[str(a)] = bootstrap_scaling_law(ratios, groups, n_boot=400,
                                                     seed=7)
            except Exception as exc:      # pragma: no cover
                fits[str(a)] = {"error": str(exc)}
        scaling[name] = fits

    # ---- phi -> phi_j suppression -----------------------------------------
    phi_dependence = {}
    for name, rows in summary.items():
        pts = [(r["phi_mean"], r["S_mean"]) for r in rows
               if r["key"]["size_ratio"] == max(x["key"]["size_ratio"] for x in rows)
               and r["phi_mean"]]
        pts = sorted(pts)
        if len(pts) >= 2:
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            slope = float(np.polyfit(x, y, 1)[0])
            phi_dependence[name] = {"points": pts, "slope_dS_dphi": slope,
                                    "monotone_decreasing": bool(slope < 0)}
        else:
            phi_dependence[name] = {"points": pts, "slope_dS_dphi": None}

    # ---- C1 negative control from the ratio-1 cases ------------------------
    mono = [r for r in shear + vib if r["size_ratio"] == 1.0]
    negative_control = {}
    if mono:
        S = [float(r["segregation_degree"]) for r in mono]
        floors = [float(r["noise_floor"]) for r in mono]
        negative_control = {
            "n_runs": len(mono),
            "S_mean": float(np.mean(S)), "S_max": float(np.max(S)),
            "noise_floor_mean": float(np.mean(floors)),
            "S_over_floor": float(np.mean(S) / np.mean(floors)),
            "directions": _consensus([int(r["direction"]) for r in mono]),
            "Sinit_mean": float(np.mean([r["initial_S"] for r in mono])),
        }

    # ---- C4 mass conservation ---------------------------------------------
    mass = [r["mass_check"] for r in recs if r.get("mass_check")]
    mass_conservation = {
        "max_weighted_phi_residual": float(max(m["residual"] for m in mass
                                               if m["residual"] is not None)),
        "n": len(mass),
        "note": "weighted mean of Phi over populated bins vs Phi_0",
    }

    # ---- C6 permeability / excess pore pressure table ----------------------
    best = max(shear, key=lambda r: (r["size_ratio"], r["normal_stress"]) )if shear else None
    permeability = {}
    if shear:
        # strongest segregation that is not noise
        cand = [r for r in shear if r["size_ratio"] > 1.0]
        if cand:
            rep = max(cand, key=lambda r: r["segregation_degree"])
            phi = np.array([np.nan if v is None else v for v in rep["phi_z"]], dtype=float)
            area = np.array([np.nan if v is None else v for v in rep["area_z"]], dtype=float)
            height = rep["plate_height"]
            edges = np.linspace(0.0, height, len(phi) + 1)
            bin_area = np.diff(edges) * rep["box_size"]
            sf = area / bin_area
            table = permeability_sensitivity_table(
                phi, sf, rep["size_ratio"] * 0.010, 0.010, height)
            table["reference_case"] = rep["key"]
            table["phi_z"] = list(phi)
            table["solid_fraction_z"] = list(sf)
            table["k_z"] = list(stratified_permeability_profile(
                phi, sf, rep["size_ratio"] * 0.010, 0.010)["k_z"])
            permeability = table

    out = {
        "meta": {
            "node": "P155-N09",
            "mode_dir": str(outdir),
            "n_records": len(recs),
            "n_shear": len(shear), "n_vibration": len(vib),
            "ratios": sorted({r["size_ratio"] for r in recs}),
        },
        "summary": summary,
        "scaling": scaling,
        "phi_dependence": phi_dependence,
        "negative_control": negative_control,
        "mass_conservation": mass_conservation,
        "permeability": permeability,
    }
    if report_path:
        pathlib.Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(report_path).write_text(
            json.dumps(out, indent=1, default=_json_default))
        print("wrote", report_path)
    print(json.dumps({k: out[k] for k in ("negative_control", "mass_conservation")},
                     indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
