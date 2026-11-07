#!/usr/bin/env python3
"""Pressure-area index campaign: does p_G ~ A**(-1/2) emerge from DEM indentation?

Runs the area scan of :mod:`tensordem.pressure_area_dem` over

* the primary configuration (Weibull-equivalent ``m = 2``, correlation length
  ``l = h/2``) with ``--seeds`` realisations per area,
* a material-statistics family (``m`` sensitivity),
* a correlation-length family ``l -> inf`` = **simultaneous-failure control**,
* a numerical family (contact stiffness, lattice resolution).

All numbers in the emitted JSON are produced programmatically; the figures are
rendered from the same JSON (no hand-written numbers).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tensordem.pressure_area_dem import (PunchConfig, fit_pressure_area_exponent,
                                         run_area_scan, run_punch_indentation)

# A = w * 1 m [m^2].  The fit range is the *wide-indenter* range
# w >= 3.75 h (A >= 0.75 m^2): the pressure-area law is quoted for wide
# indenters, and w < 3 h is a different contact regime (a documented
# asymptotic-range criterion, not a post-hoc data selection).  The full range
# including A = 0.5 m^2 is scanned as a sensitivity check.
# Primary fit range = the full range specified by the campaign brief,
# A in [0.5, 10] m^2.  A second, pre-registered asymptotic fit keeps only
# w >= 3.75 h (A >= 0.75 m^2), the wide-indenter range in which the
# pressure-area law is quoted; both are reported, neither is selected a
# posteriori.
AREAS = [0.5, 0.75, 1.25, 2.0, 3.5, 6.0, 10.0]
AREAS_WIDE = [0.75, 1.25, 2.0, 3.5, 6.0, 10.0]


def _base(**overrides) -> PunchConfig:
    kwargs = dict(punch_width=1.0, sheet_thickness=0.2, spacing=0.04,
                  young_modulus=9.0e9, strength_median=1.0e-3,
                  strength_cov=0.52, correlation_length=0.10,
                  surface_roughness=0.02, roughness_correlation=0.08,
                  contact_stiffness_factor=1.0, max_penetration_fraction=0.03,
                  n_increments=60, newton_iterations=6)
    kwargs.update(overrides)
    return PunchConfig(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default="artifacts/pressure_area_index")
    ap.add_argument("--fast", action="store_true",
                    help="smoke run: 3 seeds, primary scan only")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = 3 if args.fast else args.seeds
    t0 = time.time()

    scans = {}

    def scan(label: str, cfg: PunchConfig, areas=AREAS, seeds_local=None):
        t = time.time()
        result = run_area_scan(areas, seeds_local or seeds, cfg, label=label,
                               n_boot=4000, bootstrap_seed=7,
                               workers=args.workers, progress=False)
        payload = result.to_dict()
        payload["wall_seconds"] = time.time() - t
        (out / f"scan_{label}.json").write_text(json.dumps(payload, indent=1))
        scans[label] = payload
        fit = payload["fit"]
        print(f"[{label}] ex={fit['exponent']:.3f} "
              f"CI=[{fit['ci_low']:.3f},{fit['ci_high']:.3f}] r2={fit['r2']:.3f} "
              f"({payload['wall_seconds']:.0f}s)", flush=True)
        return payload

    primary = _base()
    scan("primary", primary)
    scan("control_uniform", _base(correlation_length=math.inf))
    scan("primary_wide", primary, areas=AREAS_WIDE, seeds_local=seeds)
    scan("control_wide", _base(correlation_length=math.inf), areas=AREAS_WIDE,
         seeds_local=max(6, seeds // 3))
    if not args.fast:
        # material statistics: lognormal COV <-> Weibull modulus m
        for cov in (0.26, 0.36):
            scan(f"m_sensitivity_cov{cov:g}", _base(strength_cov=cov))
        # correlation length family, ending in the simultaneous limit
        for corr in (0.05, 0.20):
            scan(f"l_sensitivity_{corr:g}", _base(correlation_length=corr))
        # numerical family
        scan("numerics_kc0.5", _base(contact_stiffness_factor=0.5))
        scan("numerics_kc2.0", _base(contact_stiffness_factor=2.0))
        scan("numerics_spacing0.05", _base(spacing=0.05, roughness_correlation=0.10),
             seeds_local=max(10, seeds // 3))

    # ---- mechanism diagnostics from the primary scan -----------------------
    prim = scans["primary"]["runs"]
    by_area = {}
    for run in prim:
        by_area.setdefault(run["area_m2"], []).append(run)
    growth = []
    for area in sorted(by_area):
        run = by_area[area][0]
        growth.append({"area_m2": area,
                       "displacement_m": run["displacement_m"],
                       "contact_area_m": run["contact_area_m"],
                       "force_N_per_m": run["force_N_per_m"]})
    (out / "contact_growth.json").write_text(json.dumps(growth, indent=1))

    def stat(label, key):
        vals = [r[key] for r in scans[label]["runs"]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}

    summary = {
        "areas_m2": AREAS,
        "seeds_per_area": seeds,
        "base_config": asdict(_base()),
        "fits": {label: data["fit"] for label, data in scans.items()},
        "mechanism_diagnostics": {
            "primary_non_simultaneity": stat("primary", "non_simultaneity"),
            "uniform_non_simultaneity": stat("control_uniform", "non_simultaneity"),
            "primary_contact_fraction": stat("primary", "contact_fraction_at_peak"),
            "uniform_contact_fraction": stat("control_uniform", "contact_fraction_at_peak"),
            "primary_failed_bonds_at_yield": stat("primary", "broken_bonds_at_peak"),
            "uniform_failed_bonds_at_yield": stat("control_uniform", "broken_bonds_at_peak"),
            "primary_field_cov": stat("primary", "field_cov_realised"),
        },
        "wall_seconds_total": time.time() - t0,
        "python": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v["exponent"] for k, v in summary["fits"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())