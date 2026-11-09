"""Dense CFD--DEM closure verification bench (campaign node
``p50w2-arb7250-n03-dense-dem-cfd``).

Runs the seven machine-checkable criteria, writes ``study.json`` and five
figures under ``artifacts/cfd_dem_dense/`` and prints a summary table.

Usage
-----
    python benchmarks/dense_coupling_verification.py [--outdir DIR] [--quick]

``--quick`` shrinks the sweeps (used by CI); the default settings are the
settings reported in the study JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.cfd_dem_dense import StudyOptions, run_acceptance_study, write_results  # noqa: E402


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e4):
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "cfd_dem_dense"))
    parser.add_argument("--quick", action="store_true", help="reduced sweeps for CI")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    options = (
        StudyOptions(
            rz_points=5,
            n_particles=32,
            n_cells=32,
            grid_cells=(20, 40, 80),
            ledger_dts=(2.0e-5, 1.0e-5),
            sensitivity_closures=False,
        )
        if args.quick
        else StudyOptions()
    )
    results = run_acceptance_study(options)
    written = write_results(results, args.outdir, figures=not args.no_figures)

    print(f"node: {results['node']}")
    print(f"criteria: {sum(c['passed'] for c in results['criteria'].values())}/7 passed")
    print(f"all_criteria_passed: {results['all_criteria_passed']}")
    print()
    crit = results["criteria"]
    print("1 momentum      worst residual        "
          f"{_fmt(crit['1_momentum_conservation']['worst_reaction_residual'])} "
          f"(threshold {THRESHOLD_TEXT(crit['1_momentum_conservation']['threshold'])})")
    rz = crit["2_richardson_zaki"]
    print("2 Richardson-Zaki n dynamic          "
          f"{_fmt(rz['dynamic']['n_fit'])} vs correlation {_fmt(rz['dynamic']['n_correlation'])} "
          f"-> {100*rz['dynamic']['n_deviation_relative']:.3f}% (threshold 5%)")
    eg = crit["3_ergun_pressure_drop"]
    print("3 Ergun         worst deviation      "
          f"{100*eg['worst_relative_error']:.4f}% regimes={eg['regimes_exercised']} (threshold 5%)")
    umf = crit["4_minimum_fluidization"]
    print("4 u_mf          dynamic vs Ergun      "
          f"{100*umf['study']['u_mf_deviation_relative']:.4f}% (threshold 8%)")
    gc = crit["5_grid_convergence"]["study"]
    print("5 grid          error ratios         "
          f"{[round(r, 2) for r in gc['error_ratios']]} (threshold 2.5)")
    co = crit["6_conservation"]
    print("6 conservation  exact identities     "
          f"{_fmt(co['worst_exact_identity_residual'])} mass drift "
          f"{_fmt(co['mass']['relative_drift'])} (threshold 1e-10)")
    nc = crit["7_negative_controls"]
    print("7 negative ctl  NC1/NC2/NC3           "
          f"{100*nc['nc1_reaction_off']['deviation_relative']:.2f}% / "
          f"{100*nc['nc2_drag_doubled']['deviation_relative']:.4f}% / "
          f"{100*nc['nc3_pressure_gradient_off']['deviation_relative']:.2f}%")
    print()
    print(f"json:    {written['json']}")
    for name in written["figures"]:
        print(f"figure:  {Path(args.outdir) / name}")
    print(f"wall clock: {results['wall_clock_s']:.1f} s")
    return 0 if results["all_criteria_passed"] else 1


def THRESHOLD_TEXT(value: float) -> str:
    return f"{value:.0e}"


if __name__ == "__main__":
    raise SystemExit(main())