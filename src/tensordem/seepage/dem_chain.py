"""Closed-loop DEM seepage/piping campaign runner and CLI entry point.

``python -m tensordem.seepage.dem_chain --out artifacts/seepage_piping_dem/chain.json``

runs the measured links of :mod:`tensordem.seepage.dem_measure` (C1 effective
stress, C2 critical gradient, C5 mass audit) and the analytic yardsticks of
:mod:`tensordem.seepage.chain`, and writes a single evidence bundle whose
per-criterion verdicts carry an evidence grade.  Criteria that have no
measuring instrument in this revision are graded ``withheld``; they are never
filled in with an analytic substitute.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

from .critical_gradient import terzaghi_critical_gradient
from .dem_column import ColumnDEMConfig
from .dem_measure import PIPING_GRADIENT_RAMP, run_measured_links
from .kozeny_carman import kozeny_carman_intrinsic, kozeny_carman_conductivity
from .suffusion import SkemptonBroganEnvelope

__all__ = ["run_dem_seepage_chain", "main"]

SCHEMA_VERSION = "1.0"


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=str(pathlib.Path(__file__).resolve().parents[3])
                              ).stdout.strip()
    except Exception:  # pragma: no cover - git always present in CI
        return "unknown"


def run_dem_seepage_chain(config: ColumnDEMConfig | None = None,
                          target_porosity: float = 0.42, layers: int = 2,
                          max_iter: int = 1500,
                          gradients=PIPING_GRADIENT_RAMP) -> dict:
    """Run the full measured chain and return the evidence bundle."""
    cfg = config or ColumnDEMConfig()
    t0 = time.time()
    measured = run_measured_links(cfg, target_porosity=target_porosity,
                                  layers=layers, max_iter=max_iter,
                                  gradients=gradients)
    n = measured["config"]["n_particles"]
    porosity = measured["C1_effective_stress"]["porosity_measured"]
    diameter = cfg.diameter
    # analytic yardsticks at the *measured* porosity (comparison only)
    c3_reference = {
        "porosity": porosity,
        "k_intrinsic_kozeny_carman_m2": kozeny_carman_intrinsic(porosity, diameter),
        "k_hydraulic_kozeny_carman_m_s": kozeny_carman_conductivity(
            porosity, diameter),
        "evidence": "analytic_reference_only",
    }
    env = SkemptonBroganEnvelope()
    i_c_t = terzaghi_critical_gradient(cfg.density, porosity, cfg.rho_w, cfg.gravity)
    c4_reference = {
        "i_c_terzaghi": i_c_t,
        "alpha_skempton_brogan": env.alpha_for(0.20),
        "i_c_suffusion_analytic": env.critical_gradient(i_c_t, 0.20),
        "evidence": "analytic_reference_only",
    }
    verdicts = {
        "C1_effective_stress": {
            "evidence": "REAL",
            "instrument": "DEM contact network (plane-cut stress) vs measured-porosity weight integral",
            "value": measured["C1_effective_stress"]["max_abs_deviation_pct"],
            "threshold": 10.0,
            "pass": measured["C1_effective_stress"]["pass"],
        },
        "C2_critical_gradient": {
            "evidence": "REAL",
            "instrument": "static-equilibrium continuation over the gradient ramp",
            "value": measured["C2_critical_gradient"]["i_c_measured"],
            "reference": measured["C2_critical_gradient"]["i_c_terzaghi"],
            "deviation_pct": measured["C2_critical_gradient"]["deviation_pct"],
            "threshold": 15.0,
            "pass": measured["C2_critical_gradient"]["pass"],
        },
        "C3_permeability": {
            "evidence": "WITHHELD",
            "reason": ("no pore-scale flow solver: deriving k from the packing's "
                       "hydraulic radius is algebraically identical to Kozeny-Carman"),
            "reference": c3_reference,
            "pass": None,
        },
        "C4_suffusion": {
            "evidence": "WITHHELD",
            "reason": ("requires a bidisperse transport run (fine-fraction escape "
                       "count); not executed in this repair round"),
            "reference": c4_reference,
            "pass": None,
        },
        "C5_mass_conservation": {
            "evidence": "REAL",
            "instrument": "solid + pore-fluid ledger of the simulated column",
            "value": measured["C5_mass_conservation"]["total_mass_error_pct"],
            "threshold": 2.0,
            "pass": measured["C5_mass_conservation"]["pass"],
        },
        "fluid_momentum_ledger": {
            "evidence": "REAL",
            "instrument": "slab-wise drag vs excess pressure gradient residual",
            "value": measured["fluid_ledger"]["relative_residual"],
            "threshold": 0.05,
            "pass": measured["fluid_ledger"]["pass"],
        },
    }
    measured_pass = [v["pass"] for v in verdicts.values() if v["pass"] is not None]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "campaign_node": "p299w4-n02-dem-seepage",
        "mission_id": 14594,
        "git_revision": _git_revision(),
        "generated_at_unix": time.time(),
        "wall_time_s": time.time() - t0,
        "model": {
            "kind": "soft-sphere DEM column, static equilibrium by energy minimisation",
            "friction": "none (conservative, lower bound on i_c)",
            "fluid": "quasi-steady 1-D Ergun interstitial flow",
        },
        "config": measured["config"],
        "links": measured,
        "verdicts": verdicts,
        "overall": {
            "criteria_measured": len(measured_pass),
            "criteria_passed": sum(1 for p in measured_pass if p),
            "criteria_withheld": sum(1 for v in verdicts.values() if v["pass"] is None),
            "all_measured_pass": bool(all(measured_pass)),
            "n_particles": n,
        },
    }
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("artifacts/seepage_piping_dem/chain.json"))
    parser.add_argument("--n-particles", type=int, default=600)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--porosity", type=float, default=0.42)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--kn", type=float, default=200.0)
    args = parser.parse_args(argv)

    cfg = ColumnDEMConfig(n_particles=args.n_particles, kn=args.kn)
    bundle = run_dem_seepage_chain(cfg, target_porosity=args.porosity,
                                  layers=args.layers, max_iter=args.max_iter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, default=float) + "\n")
    v = bundle["verdicts"]
    print(f"wrote {args.out}")
    for name, verdict in v.items():
        print(f"  {name:24s} evidence={verdict['evidence']:9s} "
              f"pass={verdict['pass']} value={verdict.get('value')}")
    return 0 if all(bundle["overall"]["all_measured_pass"] for _ in [0]) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())