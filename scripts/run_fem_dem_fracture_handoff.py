"""Reproduce the FEM -> DEM fracture handoff artefacts (campaign p302w1-n03).

Usage::

    PYTHONPATH=src python scripts/run_fem_dem_fracture_handoff.py [out_dir]

Writes into ``out_dir`` (default ``artifacts/fem_dem_fracture/p302w1_n03``):

* ``failure_faces.json``        -- the handoff artefact (cracks, fracture
  elements, floe polygons, conservation ledger, provenance hash);
* ``dem_initial_conditions.json`` -- DEM body initial conditions rebuilt from
  that file by the reader;
* ``summary.json``              -- the acceptance numbers (C1..C5);
* ``benchmarks.json``           -- FEM verification against four closed forms;
* ``convergence.json``          -- mesh sensitivity of the floe census;
* ``sensitivity.json``          -- sensitivity to the three crack-model knobs;
* ``figures/*.png``             -- report figures;
* ``report_data.json``          -- everything above in one file, which is what
  the report generator injects (no hand transcription of numbers).
"""
from __future__ import annotations

import json
import os
import sys

from tensordem.fem_dem_fracture import (
    IcePlateCase,
    fem_benchmark_table,
    mesh_convergence,
    run_handoff,
    sensitivity_sweep,
    write_figures,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = argv[0] if argv else "artifacts/fem_dem_fracture/p302w1_n03"
    os.makedirs(out_dir, exist_ok=True)

    case = IcePlateCase()
    summary = run_handoff(case, out_dir)
    benchmarks = fem_benchmark_table()
    convergence = mesh_convergence(case, (20, 30, 40, 50))
    sensitivity = sensitivity_sweep(case)

    figures = write_figures(
        case,
        os.path.join(out_dir, "figures"),
        summary=summary,
        convergence=convergence,
        sensitivity=sensitivity,
        benchmarks=benchmarks,
    )

    for name, payload in (
        ("benchmarks", benchmarks),
        ("convergence", convergence),
        ("sensitivity", sensitivity),
    ):
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    report_data = {
        "summary": summary,
        "benchmarks": benchmarks,
        "convergence": convergence,
        "sensitivity": sensitivity,
        "figures": figures,
    }
    with open(os.path.join(out_dir, "report_data.json"), "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2, sort_keys=True)
        handle.write("\n")

    criteria = summary["criteria"]
    print(json.dumps({k: criteria[k] for k in sorted(criteria)}, indent=2))
    print("\nfigures:")
    for path in figures:
        print("  ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())