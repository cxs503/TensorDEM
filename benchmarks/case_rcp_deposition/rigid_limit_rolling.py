"""Rigid-limit extrapolation validation with rolling resistance.

Anchor: XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT
Acceptance: <15% error vs reference (owner precision standard)

This script validates that the stiffness_limit extrapolation works correctly
when rolling resistance is present in the deposition. The round-2 method
selection (m17643) validated rolling resistance laws against LIGGGHTS and
analytic solutions, but the deposition rigid-limit extrapolation has never
been validated with rolling resistance.

Physics expectation:
- Rolling resistance makes particles "stickier" in rotation, leading to
  looser packings (lower phi).
- The stiffness_limit extrapolation should still converge to a well-defined
  hard-sphere limit, but the limit value depends on mu_rolling.
- For mu_rolling > 0, the extrapolation order p may differ from p=1
  (linear spring compression) due to the rolling contact history.

Reference:
- Frictionless RCP: phi_rcp = 0.6366 (Scott 1960; Bernal & Mason 1960)
- With rolling resistance: no single literature value; the expected behavior
  is monotone decrease in phi with increasing mu_rolling.

Gates:
- G20: |phi(kn->inf) - phi(kn=256000)| < 0.005 (extrapolation removes most
  of the soft-sphere systematic)
- G21: extrapolation_selfconsistency prediction_error < 0.003
       (4-level leave-one-out self-check)
- G22: phi decreases monotonically with mu_rolling (physics sanity)
- G23: |phi(kn->inf) / phi_ref - 1| < 15% (owner precision standard)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np

from tensordem.deposition_packing import (
    DepositionSpec, run_deposition, stiffness_limit, extrapolation_selfconsistency)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Stiffness levels for extrapolation (4 levels for leave-one-out)
KN_LEVELS = [4000.0, 16000.0, 64000.0, 256000.0]

# Rolling resistance levels to scan
MU_ROLLING_LEVELS = [0.0, 0.1, 0.3]

# Fixed parameters
N_PARTICLES = 2000
MU = 0.1  # inter-particle friction (fixed, we scan mu_rolling)
SEED = 1
RESTITUTION = 0.1
T_MAX = 150.0


def run_one(n, mu, mu_rolling, kn, seed, restitution, t_max):
    """Run a single deposition and return the result."""
    spec = DepositionSpec(
        n_particles=n, mu=mu, mu_rolling=mu_rolling, seed=seed,
        kn=kn, restitution=restitution, t_max=t_max)
    t0 = time.perf_counter()
    res = run_deposition(spec)
    wall = time.perf_counter() - t0
    return res, wall


def tag(n, mu, mu_rolling, kn, seed, e, dt_frac=None):
    """Generate a run tag."""
    t = f"deposition_n{n}_mu{mu:g}_mur{mu_rolling:g}_kn{kn:g}_e{e:g}_seed{seed}"
    if dt_frac is not None and dt_frac != 1.0 / 20.0:
        t += f"_dtf{dt_frac:g}"
    return t


def save_run(res, wall, spec, out_dir, dt_frac=None):
    """Save run results to JSON and NPZ."""
    payload = res.summary()
    payload["wall_clock_total"] = wall
    payload["spec"]["mu_rolling"] = spec.mu_rolling
    t = tag(spec.n_particles, spec.mu, spec.mu_rolling, spec.kn, spec.seed,
            spec.restitution, dt_frac)
    with open(os.path.join(out_dir, t + ".json"), "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    np.savez_compressed(os.path.join(out_dir, t + ".npz"),
                        positions=res.positions, velocities=res.velocities,
                        omegas=res.omegas)
    return t


def main():
    """Run the rigid-limit validation with rolling resistance."""
    results = {}
    gates = {}

    print("=" * 70)
    print("XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT: Rolling resistance validation")
    print("=" * 70)

    # Run all combinations
    for mu_r in MU_ROLLING_LEVELS:
        print(f"\n--- mu_rolling = {mu_r} ---")
        tier_results = []
        for kn in KN_LEVELS:
            t_tag = tag(N_PARTICLES, MU, mu_r, kn, SEED, RESTITUTION)
            json_path = os.path.join(RUNS, t_tag + ".json")
            if os.path.exists(json_path):
                print(f"  kn={kn}: loading existing run")
                d = json.load(open(json_path))
                phi = d["metrics"]["phi_bulk"]
                phi_corr = d["metrics"]["phi_bulk_overlap_corrected"]
                Z = d["metrics"]["coordination_bulk"]
                settled = d.get("settled", True)
            else:
                print(f"  kn={kn}: running...", flush=True)
                spec = DepositionSpec(
                    n_particles=N_PARTICLES, mu=MU, mu_rolling=mu_r, seed=SEED,
                    kn=kn, restitution=RESTITUTION, t_max=T_MAX)
                t0 = time.perf_counter()
                res = run_deposition(spec)
                wall = time.perf_counter() - t0
                save_run(res, wall, spec, RUNS)
                phi = res.metrics["phi_bulk"]
                phi_corr = res.metrics["phi_bulk_overlap_corrected"]
                Z = res.metrics["coordination_bulk"]
                settled = res.settled
                print(f"    phi={phi:.5f}, phi_corr={phi_corr:.5f}, Z={Z:.3f}, "
                      f"settled={settled}, wall={wall:.1f}s", flush=True)
            tier_results.append((kn, phi, phi_corr, Z, settled))
        results[mu_r] = tier_results

    # Stiffness-limit extrapolation for each mu_rolling tier
    print("\n" + "=" * 70)
    print("Stiffness-limit extrapolation")
    print("=" * 70)

    extrapolations = {}
    for mu_r, tier in results.items():
        pts = [(kn, phi) for kn, phi, _, _, _ in tier]
        sl = stiffness_limit(pts)
        extrapolations[mu_r] = sl
        print(f"\nmu_rolling={mu_r}:")
        print(f"  kn levels: {sl['kn']}")
        print(f"  phi values: {[f'{v:.5f}' for v in sl['value']]}")
        print(f"  deltas: {[f'{d:.6f}' for d in sl['delta']]}")
        print(f"  observed order p: {sl['order']}")
        print(f"  order used: {sl['order_used']}")
        print(f"  phi(kn->inf) = {sl['limit']:.6f}")
        print(f"  |phi_prod - phi_inf| = {sl['delta_from_last']:.6f}")

    # Self-consistency check (4-level leave-one-out)
    print("\n" + "=" * 70)
    print("Extrapolation self-consistency (leave-one-out)")
    print("=" * 70)

    selfconsistency = {}
    for mu_r, tier in results.items():
        pts = [(kn, phi) for kn, phi, _, _, _ in tier]
        sc = extrapolation_selfconsistency(pts)
        selfconsistency[mu_r] = sc
        print(f"\nmu_rolling={mu_r}:")
        print(f"  3-level prediction for kn=256000: {sc['predicted']:.6f}")
        print(f"  measured at kn=256000: {sc['measured']:.6f}")
        print(f"  prediction_error: {sc['prediction_error']:.6f}")
        print(f"  limit_shift (3->4 levels): {sc['limit_shift']:.6f}")
        print(f"  pair_orders: {[p['order'] for p in sc['pair_orders']]}")

    # Gates
    print("\n" + "=" * 70)
    print("Gates")
    print("=" * 70)

    phi_ref = 0.6366  # frictionless RCP reference

    for mu_r in MU_ROLLING_LEVELS:
        sl = extrapolations[mu_r]
        sc = selfconsistency[mu_r]
        phi_inf = sl["limit"]
        phi_prod = sl["value"][0]  # kn=4000

        # G20: extrapolation removes most of the soft-sphere systematic
        g20 = abs(phi_prod - phi_inf) < 0.005
        gates[f"G20_extrapolation_removes_systematic_mur{mu_r}"] = {
            "criterion": "|phi(kn=4000) - phi(kn->inf)| < 0.005",
            "got": abs(phi_prod - phi_inf),
            "pass": g20,
        }
        print(f"G20 (mur={mu_r}): |{phi_prod:.5f} - {phi_inf:.5f}| = "
              f"{abs(phi_prod - phi_inf):.6f} < 0.005 -> {'PASS' if g20 else 'FAIL'}")

        # G21: self-consistency prediction error
        g21 = sc["prediction_error"] < 0.003
        gates[f"G21_selfconsistency_mur{mu_r}"] = {
            "criterion": "prediction_error < 0.003",
            "got": sc["prediction_error"],
            "pass": g21,
        }
        print(f"G21 (mur={mu_r}): prediction_error = {sc['prediction_error']:.6f} "
              f"< 0.003 -> {'PASS' if g21 else 'FAIL'}")

        # G23: owner precision standard (<15% vs reference)
        # For mu_rolling > 0, the reference is the frictionless RCP, but the
        # expected value is lower due to rolling resistance. We check that the
        # extrapolated limit is within 15% of the reference.
        rel_err = abs(phi_inf / phi_ref - 1.0)
        g23 = rel_err < 0.15
        gates[f"G23_precision_standard_mur{mu_r}"] = {
            "criterion": "|phi(kn->inf) / phi_ref - 1| < 15%",
            "got": rel_err,
            "pass": g23,
        }
        print(f"G23 (mur={mu_r}): |{phi_inf:.5f} / {phi_ref} - 1| = "
              f"{rel_err:.4f} < 0.15 -> {'PASS' if g23 else 'FAIL'}")

    # G22: monotone decrease with mu_rolling
    phi_infs = [extrapolations[mu_r]["limit"] for mu_r in MU_ROLLING_LEVELS]
    monotone = all(phi_infs[i] >= phi_infs[i + 1] for i in range(len(phi_infs) - 1))
    gates["G22_monotone_with_mu_rolling"] = {
        "criterion": "phi(kn->inf) decreases monotonically with mu_rolling",
        "got": phi_infs,
        "pass": monotone,
    }
    print(f"\nG22: phi_inf vs mu_rolling = {phi_infs}")
    print(f"     monotone decreasing -> {'PASS' if monotone else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    all_pass = all(g["pass"] for g in gates.values())
    print(f"All gates passed: {all_pass}")
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  {name}: {status} (got={g['got']})")

    # Save results
    output = {
        "anchor": "XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT",
        "n_particles": N_PARTICLES,
        "mu": MU,
        "seed": SEED,
        "restitution": RESTITUTION,
        "kn_levels": KN_LEVELS,
        "mu_rolling_levels": MU_ROLLING_LEVELS,
        "results": {str(mu_r): [(kn, phi, phi_corr, Z, settled)
                                 for kn, phi, phi_corr, Z, settled in tier]
                    for mu_r, tier in results.items()},
        "extrapolations": {str(mu_r): sl for mu_r, sl in extrapolations.items()},
        "selfconsistency": {str(mu_r): sc for mu_r, sc in selfconsistency.items()},
        "gates": gates,
        "all_pass": all_pass,
        "phi_ref": phi_ref,
    }
    out_path = os.path.join(RUNS, "rigid_limit_rolling_summary.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(f"\nResults saved to {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
