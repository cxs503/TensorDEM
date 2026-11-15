#!/usr/bin/env python
"""Analyse rolling-resistance critical-state calibration (m21110).

Reads the stress-path JSONs from ``runs/m21110/rolling_cs_*.json``,
extracts the critical-state friction angle using the four-gate detector,
and produces:
1. A summary table (phi_cv vs mu_r)
2. A calibration figure (phi_cv vs mu_r with Rowe limit)
3. A JSON artefact for the platform report

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_rolling_cs.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensordem.biaxial_critical_state import (
    critical_state_from_stress_path,
    judge_against_reference,
)

RUNS_DIR = Path("runs/m21110")
ARTIFACTS_DIR = Path("artifacts/CASE-DEM-BIAXIAL-FRICTION")


def load_stress_paths() -> list[dict]:
    """Load all rolling_cs_*.json files."""
    paths = sorted(RUNS_DIR.glob("rolling_cs_*.json"))
    records = []
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        rec["_file"] = str(p)
        records.append(rec)
    print(f"Loaded {len(records)} stress-path records from {RUNS_DIR}")
    return records


def extract_cs(stress_path: dict, sigma3: float) -> dict:
    """Extract critical-state friction angle from a stress path."""
    try:
        cs = critical_state_from_stress_path(
            stress_path, sigma3_target=sigma3,
            min_strain=0.15,  # relaxed for 16x32 specimens
        )
        return {
            "phi_cv_deg": float(cs.phi_cv_deg),
            "q_cv_Pa": float(cs.q_cv_Pa),
            "sigma_h_cv_Pa": float(cs.sigma_h_cv_Pa),
            "d_eps_v_d_eps_a": float(cs.d_eps_v_d_eps_a),
            "q_drift": float(cs.q_drift),
            "converged": bool(cs.converged),
            "flags": {k: bool(v) for k, v in cs.flags.items()},
            "window": list(cs.window),
            "peak_phi_deg": float(cs.peak_phi_deg),
            "peak_strain": float(cs.peak_strain),
        }
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc), "converged": False}


def analyse(records: list[dict]) -> list[dict]:
    """Analyse all records and return summary."""
    results = []
    for rec in records:
        tag = rec["tag"]
        cfg = rec["config"]
        mu_r = cfg["mu_r"]
        sigma3 = cfg["sigma3_Pa"]
        stress_path = rec.get("stress_path", {})

        cs = extract_cs(stress_path, sigma3)

        # Compute phi from stress ratio in the tail
        eps = np.asarray(stress_path.get("axial_strain", []))
        sv = np.asarray(stress_path.get("sigma_v", []))
        sh = np.asarray(stress_path.get("sigma_h", []))
        if len(eps) > 0:
            tail_mask = eps > 0.20
            if tail_mask.sum() > 10:
                phi_tail = np.degrees(np.arcsin(np.clip(
                    (sv[tail_mask] - sh[tail_mask]) /
                    np.maximum(sv[tail_mask] + sh[tail_mask], 1e-30),
                    -0.95, 0.95)))
                phi_tail_mean = float(phi_tail.mean())
                phi_tail_std = float(phi_tail.std())
            else:
                phi_tail_mean = float("nan")
                phi_tail_std = float("nan")
        else:
            phi_tail_mean = float("nan")
            phi_tail_std = float("nan")

        # Rowe prediction for this mu_r
        mu = cfg["mu"]
        phi_rowe = math.degrees(math.atan(mu))  # Rowe limit is independent of mu_r

        result = {
            "tag": tag,
            "mu_r": mu_r,
            "mu": mu,
            "sigma3_kPa": sigma3 / 1e3,
            "n_particles": rec["specimen"]["n_particles"],
            "final_strain": float(rec["loading"].get("final_strain", eps[-1] if len(eps) > 0 else float("nan"))),
            "peak_phi_deg": rec["peak"]["friction_angle_deg"],
            "phi_cv_deg": cs.get("phi_cv_deg", float("nan")),
            "phi_cv_converged": cs.get("converged", False),
            "phi_tail_mean_deg": phi_tail_mean,
            "phi_tail_std_deg": phi_tail_std,
            "phi_rowe_deg": phi_rowe,
            "cs_flags": cs.get("flags", {}),
            "cs_q_drift": cs.get("q_drift", float("nan")),
            "cs_d_eps_v_d_eps_a": cs.get("d_eps_v_d_eps_a", float("nan")),
            "elapsed_s": rec["elapsed_s"],
        }
        results.append(result)

        status = "✓" if cs.get("converged") else "✗"
        print(f"  {tag}: mu_r={mu_r:.1f} phi_peak={result['peak_phi_deg']:.2f}° "
              f"phi_cv={result['phi_cv_deg']:.2f}° ({status}) "
              f"phi_tail={phi_tail_mean:.2f}±{phi_tail_std:.2f}° "
              f"Rowe={phi_rowe:.2f}°")

    return results


def make_figure(results: list[dict], out_path: str) -> str:
    """Generate calibration figure: phi_cv vs mu_r."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: phi_cv vs mu_r
    ax1 = axes[0]
    mu_rs = [r["mu_r"] for r in results]
    phi_cvs = [r["phi_cv_deg"] for r in results]
    phi_tails = [r["phi_tail_mean_deg"] for r in results]
    phi_tail_stds = [r["phi_tail_std_deg"] for r in results]

    # Rowe limit (horizontal line)
    phi_rowe = results[0]["phi_rowe_deg"] if results else 28.81
    ax1.axhline(phi_rowe, color="green", linestyle="--", alpha=0.7,
                label=f"Rowe (1962) D=0 limit: {phi_rowe:.1f}°")

    # phi_cv points
    ax1.scatter(mu_rs, phi_cvs, c="blue", s=100, zorder=5, label="φ_cv (critical state)")
    for i, (mr, pc) in enumerate(zip(mu_rs, phi_cvs)):
        if math.isfinite(pc):
            ax1.annotate(f"{pc:.1f}°", (mr, pc), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

    # phi_tail points with error bars
    valid_tails = [(mr, pt, pts) for mr, pt, pts in zip(mu_rs, phi_tails, phi_tail_stds)
                   if math.isfinite(pt)]
    if valid_tails:
        tail_mrs, tail_pts, tail_stds = zip(*valid_tails)
        ax1.errorbar(tail_mrs, tail_pts, yerr=tail_stds, fmt="ro", capsize=5,
                     label="φ (tail mean ± std)")

    ax1.set_xlabel("Rolling friction coefficient μ_r")
    ax1.set_ylabel("Friction angle φ [°]")
    ax1.set_title("Critical-State Friction Angle vs Rolling Resistance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: stress paths (q/p vs axial strain)
    ax2 = axes[1]
    colors = ["blue", "orange", "green"]
    for i, rec_file in enumerate(sorted(RUNS_DIR.glob("rolling_cs_*.json"))):
        with open(rec_file) as f:
            rec = json.load(f)
        sp = rec.get("stress_path", {})
        eps = np.asarray(sp.get("axial_strain", []))
        sv = np.asarray(sp.get("sigma_v", []))
        sh = np.asarray(sp.get("sigma_h", []))
        q = sv - sh
        p = 0.5 * (sv + sh)
        q_over_p = q / np.maximum(2 * p, 1e-9)
        mu_r = rec["config"]["mu_r"]
        label = f"μ_r={mu_r:.1f}"
        ax2.plot(eps, q_over_p, color=colors[i % len(colors)], label=label, linewidth=1.5)

    ax2.set_xlabel("Axial strain ε_a")
    ax2.set_ylabel("Stress ratio q/(2p) = sin φ")
    ax2.set_title("Stress Ratio Evolution (Isochoric Shear)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> int:
    records = load_stress_paths()
    if not records:
        print("No stress-path records found. Run the probe first.")
        return 1

    results = analyse(records)

    # Summary table
    print("\n=== Summary ===")
    print(f"{'μ_r':>5} {'φ_peak [°]':>12} {'φ_cv [°]':>10} {'φ_tail [°]':>12} {'Rowe [°]':>10} {'Converged':>10}")
    for r in results:
        cv_str = f"{r['phi_cv_deg']:.2f}" if math.isfinite(r['phi_cv_deg']) else "N/A"
        tail_str = f"{r['phi_tail_mean_deg']:.2f}±{r['phi_tail_std_deg']:.2f}" if math.isfinite(r['phi_tail_mean_deg']) else "N/A"
        print(f"{r['mu_r']:>5.1f} {r['peak_phi_deg']:>12.2f} {cv_str:>10} {tail_str:>12} {r['phi_rowe_deg']:>10.2f} {'✓' if r['phi_cv_converged'] else '✗':>10}")

    # Save artefact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artefact_path = ARTIFACTS_DIR / "rolling_cs_calibration.json"
    with open(artefact_path, "w") as f:
        json.dump({
            "mission": "m21110",
            "anchor": "CASE-DEM-BIAXIAL-FRICTION",
            "results": results,
            "rowe_limit_deg": results[0]["phi_rowe_deg"] if results else 28.81,
        }, f, indent=2)
    print(f"\nArtefact saved to {artefact_path}")

    # Generate figure
    fig_path = ARTIFACTS_DIR / "rolling_cs_calibration.png"
    make_figure(results, str(fig_path))
    print(f"Figure saved to {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
