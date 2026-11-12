#!/usr/bin/env python3
"""CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT -- driver (rework round 1).

Runs the homogeneous Lees-Edwards plane-shear sweep of ``tensordem.
elastic_limit_shear`` and writes one JSON per point plus the aggregated scaling.

Usage
-----
    # production sweep shipped with the run artifacts (this is the repro_cmd)
    python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py --production

    # full sweep (kappa ladder + shear-rate sweep), writes artifacts + figure
    python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py --all

    # a single point
    python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py \
        --tag kn1e4 --kn 1e4 --shear-rate 1.0 --strain 2.5

    # re-aggregate existing points only
    python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py --aggregate

Everything is numpy-only; the sweep is embarrassingly parallel over points and
uses a process pool when ``--jobs > 1``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, build_configuration, configuration_report,
    shear_rate_exponent, shear_stress_scaling, simulate_plane_shear)

ART = REPO / "artifacts" / "case_dem_shear_campbell_elastic_limit"
DEFAULT_DT_FACTOR = 0.004

# --- production profile (the sweep behind the shipped artifacts) -------------
# Strain window: 1.0 total, averaged over 0.6 -> 1.0 (0.4 of strain of steady
# sampling at ~10 samples per unit strain).  The window was chosen from the
# steady-drift diagnostic, not from the answer: the half-window drift of tau is
# < 0.25 at this window for every gated point (see point_*.json).
PRODUCTION_STRAIN = 1.0
PRODUCTION_STRAIN_FROM = 0.6
PRODUCTION_N = 400
# kappa ladder at fixed shear rate: kappa = kn / (rho d^3 gdot^2), so kn from
# 1e1 to 1e6 is kappa from 5.8 to 5.8e5 (d_mean = 1.2, rho = 1).
PRODUCTION_LADDER = (1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6)
# direct shear-rate exponent at two stiffnesses (n = d ln tau / d ln gdot)
PRODUCTION_RATE_SWEEPS = {1.0e3: (0.25, 0.5, 2.0), 1.0e4: (0.25, 0.5, 2.0)}
# finite-size check at the stiffest affordable point (the cell must hold >= 20
# particles across its height for a homogeneous shear band)
PRODUCTION_SIZE_CHECK = {1.0e4: (200, 800)}
# soft-particle probe: kappa = 0.58, i.e. the inertial (Bagnold) end, which this
# linear-spring cell can only reach with overlap of order d (see docs)
PRODUCTION_SOFT_PROBE = 1.0e0


def _jsonable(obj):
    """Drop the raw state arrays and numpy scalars; keep numbers and flags."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "final_state"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_point(kn: float, shear_rate: float, n_particles: int, strain: float,
              strain_from: float, dt_factor: float, seed: int,
              tag: str | None = None, outdir: Path | None = None) -> dict:
    case = ElasticLimitShearCase(
        n_particles=n_particles, kn=kn, shear_rate=shear_rate,
        solid_fraction=0.80, dt_factor=dt_factor, seed=seed,
        strain_total=strain, strain_average_from=strain_from)
    cfg = build_configuration(case)
    conf = configuration_report(case, cfg)
    res = simulate_plane_shear(case, cfg)
    res["tag"] = tag or f"kn{kn:.0e}_g{shear_rate:g}_n{n_particles}"
    res["configuration"] = conf
    payload = _jsonable(res)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"point_{res['tag']}.json").write_text(json.dumps(payload, indent=1))
    return payload


def production_specs(args) -> list[dict]:
    """Point list of the production profile (ladder + rate sweeps + size check)."""
    common = dict(strain=PRODUCTION_STRAIN, strain_from=PRODUCTION_STRAIN_FROM,
                  dt_factor=args.dt_factor, seed=args.seed, n_particles=PRODUCTION_N)
    specs = []
    for kn in PRODUCTION_LADDER:
        specs.append(dict(kn=kn, shear_rate=1.0, **common))
    for kn, rates in PRODUCTION_RATE_SWEEPS.items():
        for gd in rates:
            specs.append(dict(kn=kn, shear_rate=gd, **common))
    for kn, sizes in PRODUCTION_SIZE_CHECK.items():
        for n in sizes:
            specs.append(dict(kn=kn, shear_rate=1.0, strain=PRODUCTION_STRAIN,
                              strain_from=PRODUCTION_STRAIN_FROM,
                              dt_factor=args.dt_factor, seed=args.seed,
                              n_particles=n))
    specs.append(dict(kn=PRODUCTION_SOFT_PROBE, shear_rate=1.0, **common))
    return specs


def write_manifest(outdir: Path, profile: str, specs: list[dict],
                   argv: list[str] | None = None) -> dict:
    """Provenance of a sweep: code identity, inputs, environment."""
    import hashlib
    import platform
    import subprocess
    import time as _time

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        rev, dirty = "", ""
    man = {
        "profile": profile,
        "utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "argv": argv or sys.argv,
        "git_head": rev,
        "git_dirty": bool(dirty),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sha256": {
            "impl": sha(REPO / "src" / "tensordem" / "elastic_limit_shear.py"),
            "runner": sha(Path(__file__).resolve()),
        },
        "specs": [{k: v for k, v in sp.items() if k != "outdir"} for sp in specs],
        "defaults": {"strain": PRODUCTION_STRAIN, "strain_from": PRODUCTION_STRAIN_FROM,
                     "n_particles": PRODUCTION_N, "dt_factor": DEFAULT_DT_FACTOR,
                     "solid_fraction": 0.80, "seed": specs[0].get("seed") if specs else None},
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_manifest.json").write_text(json.dumps(_jsonable(man), indent=1))
    return man


def _parallel(jobs_spec, jobs: int) -> list[dict]:
    if jobs <= 1:
        return [run_point(**kw) for kw in jobs_spec]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(jobs, len(jobs_spec))) as pool:
        return pool.starmap(_run_kw, [(kw,) for kw in jobs_spec])


def _run_kw(kw):
    return run_point(**kw)


def re_gate(point: dict) -> dict:
    """Re-evaluate the gates from the stored diagnostics with the current defaults.

    The gate thresholds live in :class:`ElasticLimitShearCase`; a point JSON keeps
    the raw diagnostics, so a threshold change (with a documented reason) can be
    applied without rerunning the DEM.
    """
    d = ElasticLimitShearCase.__dataclass_fields__
    e_gate = d["energy_gate"].default
    ov_gate = d["max_overlap_gate"].default
    ov_mean_gate = d["mean_overlap_gate"].default
    steady_gate = d["steady_gate"].default
    coord_gate = d["coordination_gate"].default
    g = dict(point.get("gates", {}))
    g["energy"] = bool(point["energy"]["relative_defect"] <= e_gate)
    g["overlap"] = bool(point["max_overlap_ratio"] <= ov_gate
                        and point["mean_overlap_ratio"] <= ov_mean_gate)
    g["steady"] = bool(point["steady_drift_tau"] is not None
                       and point["steady_drift_tau"] <= steady_gate)
    g["coordination"] = bool(coord_gate[0] <= point["coordination"] <= coord_gate[1])
    g["pressure_positive"] = bool((point["pressure"] or 0) > 0)
    g["finite"] = bool(all(point.get(k) is not None for k in
                           ("tau", "pressure", "tau_star", "coordination")))
    g["all"] = bool(all(g[k] for k in ("energy", "overlap", "steady",
                                       "coordination", "pressure_positive", "finite")))
    point["gates"] = g
    return point


def aggregate(outdir: Path, min_points: int = 3) -> dict:
    points = []
    for f in sorted(outdir.glob("point_*.json")):
        points.append(re_gate(json.loads(f.read_text())))
    # The kappa ladder is the fixed-shear-rate row at the nominal system size.
    # The finite-size check points and the rate sweeps are reported separately:
    # mixing them into one log-log fit would put two ordinates on one abscissa.
    rate_one = [p for p in points if p["case"]["shear_rate"] == 1.0]
    sizes = [p["case"]["n_particles"] for p in rate_one if p.get("gates", {}).get("all")]
    nominal_n = max(set(sizes), key=sizes.count) if sizes else 400
    ladder = [p for p in rate_one if p["case"]["n_particles"] == nominal_n]
    scaling = shear_stress_scaling(ladder, min_points=min_points)
    scaling["nominal_n_particles"] = nominal_n
    size_check = {}
    for n in sorted({p["case"]["n_particles"] for p in rate_one}):
        gated = [p for p in rate_one
                 if p["case"]["n_particles"] == n and p.get("gates", {}).get("all")]
        if gated:
            size_check[f"n{n}"] = {
                "tau_star": float(np.mean([p["tau_star"] for p in gated])),
                "tau": float(np.mean([p["tau"] for p in gated])),
                "mu_eff": float(np.mean([p["mu_eff"] for p in gated])),
                "n_points": len(gated),
            }
    rates = sorted({p["case"]["shear_rate"] for p in points if p["case"]["shear_rate"] != 1.0})
    sweeps = {}
    for gd in rates:
        subset = [p for p in points if p["case"]["shear_rate"] == gd]
        sweeps[f"g{gd:g}"] = shear_rate_exponent(subset)
    # the rate sweep at fixed kn also gives the direct exponent
    by_kn: dict[float, list[dict]] = {}
    for p in points:
        by_kn.setdefault(p["case"]["kn"], []).append(p)
    rate_sweeps = {}
    for kn, subset in by_kn.items():
        gammas = {p["case"]["shear_rate"] for p in subset}
        if len(gammas) >= 3:
            rate_sweeps[f"kn{kn:.0e}"] = shear_rate_exponent(subset)
    if len(size_check) >= 2:
        base = size_check[f"n{nominal_n}"]["tau_star"]
        for rec in size_check.values():
            rec["rel_dev_vs_nominal"] = float(rec["tau_star"] / base - 1.0)
        scaling["size_check_rel_spread"] = float(
            max(abs(r["rel_dev_vs_nominal"]) for r in size_check.values()))
    out = {
        "scaling_vs_kappa": scaling,
        "size_check": size_check,
        "rate_sweeps": rate_sweeps,
        "n_points": len(points),
        "points": [{k: p[k] for k in ("tag", "kappa", "tau", "pressure", "tau_star",
                                      "mu_eff", "I", "coordination",
                                      "max_overlap_ratio", "mean_overlap_ratio")}
                   for p in points],
        "gates": {p["tag"]: p["gates"] for p in points},
        "energy_audit": {p["tag"]: p["energy"]["relative_defect"] for p in points},
    }
    (outdir / "scaling.json").write_text(json.dumps(_jsonable(out), indent=1))
    return out


def make_figure(outdir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is optional
        return None
    points = [json.loads(f.read_text()) for f in sorted(outdir.glob("point_*.json"))]
    rate_one = [p for p in points if p["case"]["shear_rate"] == 1.0]
    sizes = [p["case"]["n_particles"] for p in rate_one if p.get("gates", {}).get("all")]
    nominal_n = max(set(sizes), key=sizes.count) if sizes else 400
    ladder = sorted([p for p in rate_one if p["case"]["n_particles"] == nominal_n],
                    key=lambda p: p["kappa"])
    if not ladder:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ok = [p for p in ladder if p["gates"]["all"]]
    bad = [p for p in ladder if not p["gates"]["all"]]
    for group, colour, label in ((bad, "#c62828", "gate-rejected"), (ok, "#1565c0", "gated")):
        if group:
            ax[0].loglog([p["kappa"] for p in group], [p["tau_star"] for p in group],
                         "o", color=colour, label=label, ms=7)
    kap = np.array([p["kappa"] for p in ok])
    if kap.size >= 2:
        for slope, ls, lab in ((0.0, "--", r"$\tau^*\propto\kappa^0$ (inertial)"),
                               (1.0, ":", r"$\tau^*\propto\kappa^1$ (elastic)")):
            ref = np.mean([p["tau_star"] / p["kappa"] ** slope for p in ok])
            ax[0].loglog(kap, ref * kap ** slope, ls, color="0.35", label=lab)
    ax[0].set_xlabel(r"$\kappa=k_n/(\rho d^3\dot\gamma^2)$")
    ax[0].set_ylabel(r"$\tau^*=\tau/(\rho d^2\dot\gamma^2)$")
    ax[0].set_title("elastic-limit map (Lees-Edwards cell)")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    ik = [p for p in ok if p["I"] > 0]
    if ik:
        ax[1].loglog([p["I"] for p in ik], [p["mu_eff"] for p in ik], "s-",
                     color="#2e7d32")
        ax[1].set_xlabel(r"$I=\dot\gamma d/\sqrt{P/\rho}$")
        ax[1].set_ylabel(r"$\mu=\tau/P$")
        ax[1].set_title("effective friction of the gated points")
        ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = outdir / "fig_tau_star_kappa.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run the full sweep")
    ap.add_argument("--production", action="store_true",
                    help="run the production profile behind the shipped artifacts")
    ap.add_argument("--aggregate", action="store_true", help="only re-aggregate")
    ap.add_argument("--tag")
    ap.add_argument("--kn", type=float, default=1.0e4)
    ap.add_argument("--shear-rate", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--strain", type=float, default=2.5)
    ap.add_argument("--strain-from", type=float, default=1.0)
    ap.add_argument("--dt-factor", type=float, default=DEFAULT_DT_FACTOR)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--outdir", default=str(ART))
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)

    if args.aggregate:
        out = aggregate(outdir)
        print(json.dumps({k: out[k] for k in ("n_points", "scaling_vs_kappa")}, indent=1))
        print(make_figure(outdir))
        return 0

    if args.production:
        specs = production_specs(args)
        for sp in specs:
            sp["outdir"] = outdir
        t0 = __import__("time").time()
        man = write_manifest(outdir, "production", specs)
        results = _parallel(specs, args.jobs)
        for r in results:
            print(f"{r['tag']:>26s} kappa={r['kappa']:.3e} tau*={r['tau_star']:.4g} "
                  f"mu={r['mu_eff']:.3f} I={r['I']:.2e} ov={r['max_overlap_ratio']:.3f} "
                  f"gates={r['gates']['all']}")
        out = aggregate(outdir)
        print(json.dumps(_jsonable(out["scaling_vs_kappa"]), indent=1))
        print(make_figure(outdir))
        man["wall_seconds"] = __import__("time").time() - t0
        (outdir / "run_manifest.json").write_text(json.dumps(_jsonable(man), indent=1))
        return 0

    if args.all:
        ladder = [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
        specs = []
        for kn in ladder:
            specs.append(dict(kn=kn, shear_rate=1.0, n_particles=args.n,
                              strain=args.strain, strain_from=args.strain_from,
                              dt_factor=args.dt_factor, seed=args.seed,
                              outdir=outdir))
        for kn_sweep in (1.0e0, 1.0e6):
            for gd in (0.25, 0.5, 2.0):
                specs.append(dict(kn=kn_sweep, shear_rate=gd, n_particles=args.n,
                                  strain=args.strain, strain_from=args.strain_from,
                                  dt_factor=args.dt_factor, seed=args.seed,
                                  outdir=outdir))
        results = _parallel(specs, args.jobs)
        for r in results:
            print(f"{r['tag']:>22s} kappa={r['kappa']:.3e} tau*={r['tau_star']:.4g} "
                  f"mu={r['mu_eff']:.3f} I={r['I']:.2e} gates={r['gates']['all']}")
        out = aggregate(outdir)
        print(json.dumps(_jsonable(out["scaling_vs_kappa"]), indent=1))
        print(make_figure(outdir))
        return 0

    res = run_point(kn=args.kn, shear_rate=args.shear_rate, n_particles=args.n,
                    strain=args.strain, strain_from=args.strain_from,
                    dt_factor=args.dt_factor, seed=args.seed, tag=args.tag,
                    outdir=outdir)
    print(json.dumps({k: res[k] for k in ("tag", "kappa", "tau", "pressure", "tau_star",
                                          "mu_eff", "I", "coordination", "gates")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())