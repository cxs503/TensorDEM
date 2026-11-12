#!/usr/bin/env python
"""CASE-GRANULAR-GAS-COOLING: Haff's law benchmark for the DEM dissipation path.

Verification debt being paid
----------------------------
:tensordem.contact_restitution verifies one isolated impact; nothing verified
the *collective* behaviour -- that collision statistics and energy loss are
coupled so that a dilute gas cools as ``theta ~ t**-2`` with the Boltzmann
prefactor.  This driver runs :mod:`tensordem.granular_gas` over a matrix of
restitution, packing fraction, stiffness/timestep and dimension, fits Haff's
law, and writes tables, figures and a report with programmatically injected
numbers (no hand-copied values).

Anchors (external, not fitted)
------------------------------
* Haff (1983) J. Fluid Mech. 132:401 ; Jenkins & Savage (1983) JFM 130:187 --
  ``theta(t) = theta0 (1 + t/tau)**-2``, i.e. ``theta**-1/2`` linear in ``t``.
* Boltzmann / Enskog dilute-gas collision frequency and dissipation, closed form
  in :func:`tensordem.granular_gas.theory_cooling_coefficient`
  (``K = (2/d**2) n S (1-e**2) Gamma((d+3)/2)/Gamma(d/2)``).

Usage
-----
    PYTHONPATH=src python benchmarks/granular_gas_cooling.py --jobs 8 \
        --out runs/m16039_granular_gas
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from tensordem import granular_gas as gg  # noqa: E402

# ---------------------------------------------------------------------------
# acceptance thresholds (pre-registered, see prereg/CASE_GRANULAR_GAS_COOLING.json)
# ---------------------------------------------------------------------------
TOL = {
    "restitution_abs": 0.01,     # |e_eff - e_nom| for e_nom >= 0.8
    "haff_r2": 0.999,            # linearity of theta**-1/2(t)
    "prefactor_rel": 0.15,       # |K_meas/K_theory - 1| per case
    "prefactor_extrap_rel": 0.15,  # |K_meas/K_theory - 1| after phi -> 0 extrapolation
    "rest_scaling_rel": 0.08,    # K/(1-e_eff**2) spread across e
    "kurtosis_abs": 0.15,
    "elastic_drift_rel": 1.0e-2,  # |dE|/E0 for e = 1
    "refinement_rel": 0.10,      # K change under dt/kn refinement
}


def _run_case(spec: dict) -> dict:
    """Run one cooling case and return all derived metrics."""
    cfg = gg.GasConfig(
        dim=spec["dim"], n_particles=spec["n"], packing=spec["phi"],
        restitution=spec["e"], kn=spec["kn"],
        dt_per_contact=spec.get("dt_per_contact", 20.0),
        sample_every=spec.get("sample_every", 200),
        steps=spec.get("steps", 600_000),
        stop_ratio=spec.get("stop_ratio", 8.0),
        seed=spec.get("seed", 0),
    )
    t0 = time.time()
    run = gg.simulate_cooling(cfg)
    wall = time.time() - t0
    fit = run.fit
    local = gg.local_cooling_coefficient(run.times, run.theta)

    e_eff = gg.two_body_restitution(cfg.with_(restitution=spec["e"]))
    k_theory = gg.theory_cooling_coefficient(cfg.dim, cfg.number_density, e_eff)
    k_ratio = fit["K_meas"] / k_theory

    # collision frequency in the middle of the cooling window: every counted
    # contact episode is shared by two particles, hence the factor 2.
    t, imp, th = run.times, run.impacts, run.theta
    lo, hi = int(0.2 * len(t)), int(0.9 * len(t))
    omega_meas = 2.0 * (imp[hi] - imp[lo]) / (cfg.n_particles * (t[hi] - t[lo]))
    theta_mid = float(np.mean(th[lo:hi]))
    omega_theory = gg.theory_collision_frequency(cfg.dim, cfg.number_density, theta_mid)

    return {
        "label": spec["label"], "dim": cfg.dim, "phi": cfg.packing,
        "n": cfg.n_particles, "e_nom": cfg.restitution, "e_eff": e_eff,
        "kn": cfg.kn, "dt": cfg.dt, "contact_time": cfg.contact_time,
        "knudsen": cfg.knudsen, "dt_per_contact": cfg.dt_per_contact,
        "n_density": cfg.number_density,
        "K_meas": fit["K_meas"], "K_theory": k_theory, "K_ratio": k_ratio,
        "K_windows": local["K_windows"],
        "r2": fit["r2"], "exponent_apparent": fit["exponent"],
        "window": fit["window"], "n_points": fit["n_points"],
        "theta_final_ratio": float(run.theta[-1] / run.theta0_measured),
        "omega_ratio": omega_meas / omega_theory if omega_theory else float("nan"),
        "theta_mid": theta_mid,
        "kurtosis": run.kurtosis,
        "max_overlap": run.max_overlap,
        "energy_drift_rel": float(np.ptp(run.kinetic + run.potential)
                                  / run.kinetic[0]),
        "stopped_early": run.stopped_early,
        "wall_s": wall,
        "t": [float(x) for x in run.times[:: max(1, len(run.times) // 400)]],
        "theta": [float(x) for x in run.theta[:: max(1, len(run.times) // 400)]],
    }


def _elastic_control(dt_per_contact: float, steps: int = 40_000) -> dict:
    cfg = gg.GasConfig(dim=2, n_particles=600, packing=0.05, restitution=1.0,
                       kn=500.0, dt_per_contact=dt_per_contact,
                       sample_every=500, steps=steps, stop_ratio=1.0e9,
                       record_velocity=False, seed=7)
    run = gg.simulate_cooling(cfg)
    e_tot = run.kinetic + run.potential
    return {"dt_per_contact": dt_per_contact,
            "relative_drift": float(np.ptp(e_tot) / e_tot[0]),
            "theta_drift": float(np.ptp(run.theta) / run.theta[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "runs" / "m16039_granular_gas"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="small matrix for smoke tests")
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    (outdir / "cases").mkdir(parents=True, exist_ok=True)

    if args.quick:
        specs = [
            {"label": "e_0.90", "dim": 2, "n": 500, "phi": 0.03, "e": 0.90,
             "kn": 500.0, "sample_every": 100, "stop_ratio": 4.0},
            {"label": "phi_0.02", "dim": 2, "n": 500, "phi": 0.02, "e": 0.80,
             "kn": 500.0, "sample_every": 100, "stop_ratio": 4.0},
        ]
    else:
        base = dict(dim=2, n=1200, phi=0.02, kn=1000.0, sample_every=200,
                    stop_ratio=8.0)
        specs = [
            dict(base, label="e_0.60", e=0.60),
            dict(base, label="e_0.80", e=0.80),
            dict(base, label="e_0.90", e=0.90),
            dict(base, label="e_0.95", e=0.95),
            dict(base, label="phi_0.005", e=0.80, phi=0.005, n=2000, stop_ratio=6.0),
            dict(base, label="phi_0.010", e=0.80, phi=0.010, n=1500, stop_ratio=6.0),
            dict(base, label="phi_0.040", e=0.80, phi=0.040, stop_ratio=6.0),
            dict(base, label="refine_kn2000", e=0.90, kn=2000.0),
            dict(base, label="refine_dt40", e=0.90, dt_per_contact=40.0),
            dict(base, label="dim3_0.80", dim=3, n=1000, e=0.80, stop_ratio=6.0),
        ]

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        cases = list(pool.map(_run_case, specs))
        elastic = list(pool.map(_elastic_control, [20.0, 40.0]))

    for c in cases:
        (outdir / "cases" / f"{c['label']}.json").write_text(json.dumps(c, indent=2))

    summary = {
        "case": "CASE-GRANULAR-GAS-COOLING",
        "repo": "TensorDEM",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "tolerances": TOL,
        "cases": cases,
        "elastic_controls": elastic,
        "wall_s": time.time() - t_start,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    for c in cases:
        print(f"{c['label']:>14s} dim={c['dim']} phi={c['phi']:.3f} e={c['e_nom']:.2f} "
              f"e_eff={c['e_eff']:.4f} K_meas={c['K_meas']:.5f} K_th={c['K_theory']:.5f} "
              f"ratio={c['K_ratio']:.3f} R2={c['r2']:.6f} omega={c['omega_ratio']:.3f} "
              f"kurt={c['kurtosis']:+.3f} wall={c['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())