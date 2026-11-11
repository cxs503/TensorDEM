"""Benchmark: FEM -> DEM dynamic fracture handoff (campaign p302w2-n03).

Reference-vs-computed table for every quantity the dynamic handoff claims, in one
machine-readable file (``results.json`` next to this script).

Rows come in two families:

**Closed-form references (the continuum side).**  The Kirchhoff-plate FEM is
compared with the beam limit, the Navier double series for a simply supported
square plate, the clamped-plate coefficient and Hetenyi's Winkler-plate point
load.  These are the wave-1 rows, re-run here so the benchmark file is
self-contained.

**Exact identities and analytic limits (the handoff side).**  The rigid fit
reproduces momentum and angular momentum exactly and splits the kinetic energy
exactly; those residuals are compared against zero (round-off is the measured
error).  Two analytic limits pin the projection itself:

* a *rigid* velocity field must be recovered with unit rigid fraction and zero
  residual;
* an *isotropic bending* field ``kappa/2 (x^2 + y^2)`` on a symmetric square has
  vanishing tilt, so the fit collapses to the mean and the rigid fraction is the
  closed form ``mean(u)^2 / mean(u^2)``.

**Physical references (the discrete side).**  Each floe is integrated in heave
with the inherited vertical velocity and the period is compared with
``omega = sqrt(rho_w g / (rho_i t))``.  Added mass is neglected, so a real floe
heaves 5-15 % slower; the benchmark therefore checks an internally consistent
handoff (mass/area/buoyancy agree), not the added-mass-corrected period.

Run:
    python benchmarks/fem_dem_fracture_dynamic/benchmark.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np                                   # noqa: E402

from tensordem.fem_dem_fracture import IcePlateCase    # noqa: E402
from tensordem.fem_dem_fracture.dynamics import (      # noqa: E402
    project_point_masses,
)
from tensordem.fem_dem_fracture.study_dynamic import (  # noqa: E402
    benchmark_table,
    run_dynamic_handoff,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _lattice(n: int = 12, length: float = 1.0):
    coords = (np.arange(n) + 0.5) * (length / n)
    XX, YY = np.meshgrid(coords, coords, indexing="xy")
    return XX.ravel() - length / 2, YY.ravel() - length / 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(HERE, "results.json"))
    parser.add_argument("--pressure-rate", type=float, default=1000.0)
    parser.add_argument("--workdir", default=None,
                        help="keep the artefact bundle here (default: system tmp)")
    args = parser.parse_args(argv)

    import tempfile

    tmp = None if args.workdir else tempfile.TemporaryDirectory()
    workdir = args.workdir or tmp.name
    case = IcePlateCase()
    summary = run_dynamic_handoff(case, workdir,
                                  pressure_rate_pa_s=args.pressure_rate,
                                  tag="benchmark", make_figures=False,
                                  heave_periods=4.0)
    dynamic = None
    from tensordem.fem_dem_fracture import read_dynamic_failure_faces

    dynamic = read_dynamic_failure_faces(summary["artifacts"]["failure_faces_dynamic"])
    intake = json.load(open(summary["artifacts"]["dem_intake"], encoding="utf-8"))
    table = benchmark_table(case, dynamic, intake,
                            linearity=summary["linearity"],
                            quasi_static=summary["quasi_static"])

    results = {
        "campaign": "p302w2-n03-fem-dem-fracture-handoff",
        "pressure_rate_pa_s": args.pressure_rate,
        "case": summary["case"],
        "n_floes": len(summary["per_floe"]),
        "rows": table["rows"],
        "n_rows": table["n_rows"],
        "n_failing": table["n_failing"],
        "all_within_tolerance": table["all_within_tolerance"],
        "criteria": {k: v["pass"] for k, v in summary["criteria"].items()
                     if isinstance(v, dict) and "pass" in v},
        "ledger": summary["ledger"],
        "quasi_static": summary["quasi_static"],
        "repro_cmd": ("python benchmarks/fem_dem_fracture_dynamic/benchmark.py "
                      f"--pressure-rate {args.pressure_rate:g}"),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    width = max(len(r["benchmark"]) for r in results["rows"]) + 2
    print(f"{'benchmark':{width}s} {'quantity':38s} {'reference':>13s} "
          f"{'got':>13s} {'rel err':>10s}  ok")
    for row in results["rows"]:
        ok = "yes" if row["rel_error"] <= row["tolerance"] else "NO"
        print(f"{row['benchmark']:{width}s} {row['quantity']:38s} "
              f"{row['reference']:13.6g} {row['got']:13.6g} "
              f"{row['rel_error']:10.3e}  {ok}")
    print(f"\n{results['n_rows']} rows, {results['n_failing']} outside tolerance; "
          f"all_within_tolerance = {results['all_within_tolerance']}")
    print(f"written: {args.out}")
    if tmp is not None:
        tmp.cleanup()
    return 0 if results["all_within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())