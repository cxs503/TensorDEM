"""Coarse-graining macro-anchor campaign (IN102-N02).

Unified benchmark device
------------------------
``tensordem.mu_i_rheology.simulate_servo_shear`` (constant-stress direct shear
box).  All three macro anchors are read off the same device so that the
comparison across coarse-graining factor k is device-controlled:

  1. rheology      mu(I) at I in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}
  2. coordination  Z = 2 N_c / N in the steady shear state
  3. solid fraction phi of the bulk slab (porosity = 1 - phi)
  4. angle of repose theta_r of a gravity-driven column collapse
     (``simulate_repose_collapse``), plus the quasi-static shear-cell proxy
     theta_r,qs = arctan(mu(I -> 0))

Scaling rule (contact-time invariance, "口径 A" of ``coarse_graining.py``)
--------------------------------------------------------------------------
    radius   R   -> k R
    geometry W,D,H -> k W, k D, k H     (box/particle ratio fixed)
    stiffness kn  -> k^3 kn             (contact time / restitution invariant)
    density, mu, mu_rolling, e, P       invariant
    shear rate gamma_dot -> gamma_dot / k   (keeps I = gamma_dot d / sqrt(P/rho))
    collapse time -> sqrt(k) t          (Froude similarity, g fixed)

Usage
-----
    python benchmarks/coarse_graining/run_macro_campaign.py --k 1
    python benchmarks/coarse_graining/run_macro_campaign.py --aggregate

Every k is an independent shard (one result.json per k); ``--aggregate`` joins
them into the invariance / Pareto / k-bound ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensordem.coarse_graining import (  # noqa: E402
    scale_contact_time_invariant,
    verify_contact_scaling,
)
from tensordem.coarse_graining_macro import (  # noqa: E402
    INERTIAL_TARGETS,
    aggregate,
    build_plan,
    legacy_results,
)
from tensordem.mu_i_rheology import (  # noqa: E402
    ReposeCollapseCase,
    ServoShearCase,
    simulate_repose_collapse,
    simulate_servo_shear,
)

OUT = Path(__file__).resolve().parent
K_VALUES = (1, 2, 4, 8)


def shear_sweep(k: int, recipe: str = "A") -> dict:
    """Run the five-I sweep at coarse-graining factor k for a scaling recipe."""
    plan = build_plan(k, recipe)
    runs = []
    t0 = time.perf_counter()
    for target in INERTIAL_TARGETS:
        case = ServoShearCase(**plan["case"](target))
        res = simulate_servo_shear(case)
        runs.append({
            "inertial_number_target": target,
            "inertial_number": res["inertial_number"],
            "mu_effective": res["mu_effective"],
            "shear_stress_pa": res["shear_stress_pa"],
            "normal_stress_pa": res["normal_stress_measured_pa"],
            "coordination_number": res["coordination_number"],
            "solid_fraction": res["solid_fraction"],
            "porosity": res["porosity"],
            "n_particles": res["n_particles"],
            "shear_steps": res["shear_steps"],
            "settle_steps": res["settle_steps"],
            "dt": res["dt"],
            "plate_height": res["plate_height"],
            "max_velocity": res["max_velocity"],
            "wall_clock_s": res["wall_clock_s"],
        })
        print(f"[k={k}] I_target={target:g} -> I={res['inertial_number']:.4g} "
              f"mu={res['mu_effective']:.4f} Z={res['coordination_number']:.3f} "
              f"phi={res['solid_fraction']:.4f} ({res['wall_clock_s']:.0f}s)",
              flush=True)
    return {
        "k": k,
        "device": "servo_shear_cell",
        "scaling": plan["scaling"],
        "material": plan["material"],
        "runs": runs,
        "wall_clock_s_total": time.perf_counter() - t0,
    }


def repose_anchor(k: int, recipe: str = "A") -> dict:
    plan = build_plan(k, recipe)
    res = simulate_repose_collapse(ReposeCollapseCase(**plan["repose"]), seeds=(0,))
    print(f"[k={k}] repose theta={res['repose_angle_deg']:.3f} deg "
          f"(+-{res['repose_angle_stderr_deg']:.3f}) n={res['n_particles']} "
          f"({res['wall_clock_s']:.0f}s)", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, choices=list(K_VALUES))
    ap.add_argument("--repose-only", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--recipe", default="A", choices=("A", "B"))
    ap.add_argument("--no-repose", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        summary = aggregate(OUT, K_VALUES, args.recipe)
        if args.recipe != "A":
            (OUT / f"summary_{args.recipe}.json").write_text(json.dumps(summary, indent=2))
            print(f"wrote {OUT / f'summary_{args.recipe}.json'}")
            return
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        legacy_path = REPO / "runs" / "cg_scaling" / "results.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(json.dumps(legacy_results(summary), indent=2))
        print(f"wrote {legacy_path}")
        print(json.dumps({key: summary[key] for key in
                          ("max_rheology_dev_pct", "max_micro_dev_pct",
                           "safe_k_upper_bound", "pareto_front")}, indent=2))
        return

    if args.k is None:
        raise SystemExit("--k or --aggregate required")

    kdir = (OUT / f"k{args.k}" if args.recipe == "A" else OUT / f"{args.recipe}_k{args.k}")
    kdir.mkdir(parents=True, exist_ok=True)
    existing = {}
    target_path = kdir / "result.json"
    if target_path.exists():
        existing = json.loads(target_path.read_text())

    if not args.repose_only:
        existing.update(shear_sweep(args.k, args.recipe))
    existing.setdefault("k", args.k)
    if not args.no_repose:
        existing["repose"] = repose_anchor(args.k, args.recipe)
    if "scaling" not in existing:
        existing["scaling"] = build_plan(args.k, args.recipe)["scaling"]
        existing["material"] = build_plan(args.k, args.recipe)["material"]
        existing["recipe"] = args.recipe
    target_path.write_text(json.dumps(existing, indent=2))
    print(f"wrote {target_path}")


if __name__ == "__main__":
    main()