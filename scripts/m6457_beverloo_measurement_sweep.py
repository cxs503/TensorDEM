"""m6457 — Beverloo discharge *measurement* sweep (wrapper audit evidence).

Runs the real flat-orifice DEM solver through the ``beverloo_discharge`` case
interface and records the measured flow rate, the per-window series, the
recovered Beverloo constant ``C`` and the steadiness statistic.

The point of the sweep is not to re-verify ``silo_discharge`` (m6296 did that),
but to show that the wrapper now reports a *measured* rate that (a) differs from
the correlation prediction, (b) is reproducible from the mass balance, and
(c) exposes the start-up-burst / arching limitation of the underlying solver
instead of hiding it behind ``r2 = 1.0``.

Usage
-----
    PYTHONPATH=src python scripts/m6457_beverloo_measurement_sweep.py --case A
    PYTHONPATH=src python scripts/m6457_beverloo_measurement_sweep.py --quick

Each ``--case`` writes ``runs/m6457_beverloo/<case>.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.beverloo_discharge import (  # noqa: E402
    BeverlooCase, simulate_discharge,
)

OUT = Path("runs/m6457_beverloo")

# (name, kwargs) — baseline is the m6296 D5 configuration, the rest probe the
# disordered-packing / low-friction route to a steady discharge.
CASES: dict[str, dict] = {
    # ordered-lattice control: reproduces the m6296 start-up-burst + arching
    "A_base_D5": dict(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                      orifice_diameter=0.075, mu=0.4, mu_rolling=0.02,
                      packing_gap=0.02, jitter=0.002, settle_time=0.25,
                      window=0.2, n_windows=5, max_time=1.45),
    "F_base_D4": dict(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                      orifice_diameter=0.060, mu=0.4, mu_rolling=0.02,
                      packing_gap=0.02, jitter=0.002, settle_time=0.25,
                      window=0.2, n_windows=5, max_time=1.45),
    # disordered (settling HCP-with-gap) packing, same material, same orifice
    "B_jitter_D5": dict(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                        orifice_diameter=0.075, mu=0.4, mu_rolling=0.02,
                        packing_gap=0.15, jitter=0.08, settle_time=0.25,
                        window=0.2, n_windows=5, max_time=1.45),
    "C_lowmu_D5": dict(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                       orifice_diameter=0.075, mu=0.15, mu_rolling=0.01,
                       packing_gap=0.15, jitter=0.08, settle_time=0.25,
                       window=0.2, n_windows=5, max_time=1.45),
    "D_lowmu_D4": dict(radius=0.0075, silo_radius=0.11, fill_height=0.12,
                       orifice_diameter=0.060, mu=0.15, mu_rolling=0.01,
                       packing_gap=0.15, jitter=0.08, settle_time=0.25,
                       window=0.2, n_windows=5, max_time=1.45),
    # head-independence check at the flowing (disordered) configuration
    "E_lowmu_D5_tall": dict(radius=0.0075, silo_radius=0.11, fill_height=0.20,
                            orifice_diameter=0.075, mu=0.15, mu_rolling=0.01,
                            packing_gap=0.15, jitter=0.08, settle_time=0.25,
                            window=0.2, n_windows=5, max_time=1.45),
}

QUICK = dict(radius=0.01, silo_radius=0.06, fill_height=0.06,
             orifice_diameter=0.06, settle_time=0.05, window=0.04,
             n_windows=5, max_time=0.25)


def _harness_version() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "node": platform.node()}
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        info["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        info["git_sha"] = None
    return info


def run_case(name: str, kwargs: dict) -> dict:
    case = BeverlooCase(**kwargs)
    t0 = time.time()
    res = simulate_discharge(case)
    wall = time.time() - t0

    windows = res["flow_rate_windows_kg_s"]
    nonzero = sum(1 for w in windows if w > 0.0)
    tail = windows[1:] if len(windows) > 1 else windows
    tail_mean = sum(tail) / len(tail) if tail else 0.0
    tail_cv = (math.sqrt(sum((w - tail_mean) ** 2 for w in tail) / len(tail))
               / tail_mean) if tail and tail_mean > 0 else float("inf")

    rec = {
        "case": name,
        "params": {k: (v if not isinstance(v, bool) else v) for k, v in kwargs.items()},
        "n_initial": res["n_initial"],
        "n_discharged": res["n_discharged"],
        "discharged_fraction_of_fill": (res["n_discharged"] / res["n_initial"]
                                       if res["n_initial"] else 0.0),
        "bulk_density_kg_m3": res["bulk_density_kg_m3"],
        "packing_fraction": res["packing_fraction"],
        "W_measured_kg_s": res["flow_rate_kg_s"],
        "W_std_kg_s": res["flow_rate_std_kg_s"],
        "W_windows_kg_s": windows,
        "W_lit_kg_s": res["beverloo_prediction"]["W_kg_s"],
        "rel_err_vs_beverloo": res["rel_err_vs_beverloo"],
        "recovered_C": res["recovered_C"],
        "steady_r2": res["r2"],
        "window_nonzero_fraction": nonzero / len(windows),
        "tail_window_cv": tail_cv,
        "mass_balance_residual_kg": res["mass_balance_residual_kg"],
        "dt_s": res["dt"],
        "steps": res["steps"],
        "wall_clock_s": wall,
        "host": _harness_version(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="*", default=None,
                    help="case names from " + ", ".join(CASES))
    ap.add_argument("--quick", action="store_true",
                    help="small fast case (80 particles) for smoke tests")
    args = ap.parse_args(argv)

    if args.quick:
        rec = run_case("quick", QUICK)
        print(json.dumps({k: rec[k] for k in (
            "n_initial", "n_discharged", "W_measured_kg_s", "W_lit_kg_s",
            "rel_err_vs_beverloo", "steady_r2", "wall_clock_s")}, indent=1))
        return 0

    names = args.case or list(CASES)
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}")
        rec = run_case(name, CASES[name])
        print("RESULT " + json.dumps(rec, default=str), flush=True)
        print(f"{name}: n0={rec['n_initial']} discharged={rec['n_discharged']} "
              f"W={rec['W_measured_kg_s']:.4f} (lit {rec['W_lit_kg_s']:.4f}) "
              f"rel={rec['rel_err_vs_beverloo']:+.1%} r2={rec['steady_r2']:.3f} "
              f"wall={rec['wall_clock_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())