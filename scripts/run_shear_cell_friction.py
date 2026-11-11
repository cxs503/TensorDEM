#!/usr/bin/env python3
"""Entry point for direct-shear cell macroscopic friction measurement (m13798).

This script runs the shear cell device at multiple normal-stress levels and
generates comprehensive artifacts documenting the measurement attempt.

Mission: m13798 (TensorDEM shear cell dead-code fix + pathological fit repair)
Anchor: GAP-DEM-FRICTION (macroscopic friction coefficient vs experiment <10%)

Status: honest FAIL - physical steady state not achieved (platen loses contact)
        but code-level defects fixed and measurement device is now operational.

Usage:
    python scripts/run_shear_cell_friction.py [--output-dir DIR] [--stress-levels LIST]

Output:
    artifacts/shear_cell_friction/
    ├── shear_cell_diagnosis.json      # Full simulation results + traces
    ├── shear_cell_summary.json        # Executive summary + verdict
    └── shear_cell_run.log             # Execution log
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.shear_cell import (
    ShearCellCase,
    simulate_shear_cell,
    measure_macro_friction,
    _fit_mohr_coulomb,
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "shear_cell_run.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def run_single_stress_level(case: ShearCellCase, sigma_target: float, logger: logging.Logger) -> dict:
    """Run one shear test at a given normal stress level."""
    logger.info(f"Running shear cell at σ_n = {sigma_target:.0f} Pa")
    
    case_level = replace(case, normal_stress=sigma_target)
    result = simulate_shear_cell(case_level)
    
    # Extract key metrics
    summary = {
        "normal_stress_applied_Pa": sigma_target,
        "normal_stress_measured_Pa": result["normal_stress_measured"],
        "normal_stress_sem_Pa": result["normal_stress_sem"],
        "shear_stress_Pa": result["shear_stress"],
        "shear_stress_sem_Pa": result["shear_stress_sem"],
        "mu_macro": result["mu_macro"],
        "friction_angle_deg": result["friction_angle_deg"],
        "valid": result["valid"],
        "invalid_reason": result.get("invalid_reason", ""),
        "n_particles": result["n_particles"],
        "total_steps": result["total_steps"],
        "timestep_s": result["timestep"],
        "shear_rate_1s": result["shear_rate"],
        "inertial_number": result["inertial_number"],
    }
    
    # Extract traces (downsampled for JSON size)
    trace_keys = [
        "shear_strain_history",
        "shear_stress_history",
        "normal_stress_history",
        "bed_top_history",
        "platen_z_history",
    ]
    traces = {}
    for key in trace_keys:
        if key in result:
            hist = result[key]
            # Downsample to ~100 points for JSON
            if len(hist) > 100:
                step = max(1, len(hist) // 100)
                traces[key] = hist[::step]
            else:
                traces[key] = hist
    
    return {"summary": summary, "traces": traces}


def run_campaign(stress_levels: list[float], output_dir: Path, logger: logging.Logger) -> dict:
    """Run full campaign across multiple stress levels."""
    logger.info(f"Starting shear cell campaign: {len(stress_levels)} stress levels")
    logger.info(f"Stress levels (Pa): {stress_levels}")
    
    # Base case parameters
    case = ShearCellCase(
        settle_time=0.02,
        consol_time=0.03,
        shear_strain_target=0.02,  # Short run for diagnosis
        sample_every=20,
    )
    
    runs = {}
    for sigma in stress_levels:
        run_data = run_single_stress_level(case, sigma, logger)
        runs[str(int(sigma))] = run_data
    
    # Attempt Mohr-Coulomb fit across levels
    valid_levels = [
        (sigma, runs[str(int(sigma))]["summary"]["shear_stress_Pa"])
        for sigma in stress_levels
        if runs[str(int(sigma))]["summary"]["valid"]
    ]
    
    fit_result = None
    if len(valid_levels) >= 2:
        sigmas, taus = zip(*valid_levels)
        try:
            fit = _fit_mohr_coulomb(list(sigmas), list(taus))
            fit_result = {
                "cohesion_Pa": fit["cohesion"],
                "friction_angle_deg": fit["friction_angle_deg"],
                "r_squared": fit["r_squared"],
                "n_levels": len(valid_levels),
            }
            logger.info(f"Mohr-Coulomb fit: c={fit['cohesion']:.1f} Pa, φ={fit['friction_angle_deg']:.2f}°, R²={fit['r_squared']:.4f}")
        except ValueError as e:
            logger.warning(f"Mohr-Coulomb fit failed: {e}")
            fit_result = {"error": str(e)}
    else:
        logger.warning(f"Insufficient valid levels for Mohr-Coulomb fit ({len(valid_levels)} < 2)")
    
    return {
        "mission": 13798,
        "anchor": "GAP-DEM-FRICTION",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_stress_levels": len(stress_levels),
        "stress_levels_Pa": stress_levels,
        "runs": runs,
        "mohr_coulomb_fit": fit_result,
    }


def generate_summary(campaign: dict, logger: logging.Logger) -> dict:
    """Generate executive summary from campaign results."""
    runs = campaign["runs"]
    
    # Count valid/invalid
    n_valid = sum(1 for r in runs.values() if r["summary"]["valid"])
    n_invalid = len(runs) - n_valid
    
    # Extract key observations
    observations = []
    for sigma_str, run in runs.items():
        s = run["summary"]
        if not s["valid"]:
            observations.append({
                "stress_level_Pa": s["normal_stress_applied_Pa"],
                "issue": s["invalid_reason"],
                "measured_sigma_Pa": s["normal_stress_measured_Pa"],
            })
    
    # Verdict
    if n_valid == 0:
        verdict = "honest FAIL: all stress levels failed (platen loses contact, σ_n → 0, μ undefined)"
        status = "FAIL"
    elif n_valid < len(runs):
        verdict = f"partial success: {n_valid}/{len(runs)} levels valid, Mohr-Coulomb fit not attempted"
        status = "PARTIAL"
    else:
        verdict = f"success: all {len(runs)} levels valid, Mohr-Coulomb envelope fitted"
        status = "SUCCESS"
    
    logger.info(f"Verdict: {verdict}")
    
    return {
        "mission": campaign["mission"],
        "anchor": campaign["anchor"],
        "timestamp": campaign["timestamp"],
        "status": status,
        "verdict": verdict,
        "n_stress_levels": campaign["n_stress_levels"],
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "observations": observations,
        "mohr_coulomb_fit": campaign.get("mohr_coulomb_fit"),
        "code_defects_fixed": [
            "D1: mass[:, None] TypeError (scalar mass/inertia)",
            "D2: pathological single-level Mohr-Coulomb fit (now cross-level only)",
            "D3: normal stress from Σ|f_z| (now platen reaction force)",
            "D4: dead-load platen detachment (now stress-servo + validity gate)",
        ],
        "next_steps": [
            "Fix bed dilation/platen detachment (bottom-plate damping or denser initial packing)",
            "Extend to Lees-Edwards periodic shear (validated arm in tensordem.mts_contact)",
            "Close GAP-DEM-FRICTION anchor with experimental comparison",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run shear cell friction measurement campaign")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/shear_cell_friction"),
        help="Output directory for artifacts (default: artifacts/shear_cell_friction)",
    )
    parser.add_argument(
        "--stress-levels",
        type=float,
        nargs="+",
        default=[400.0, 800.0, 1600.0],
        help="Normal stress levels in Pa (default: 400 800 1600)",
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("TensorDEM Shear Cell Friction Measurement (m13798)")
    logger.info("=" * 70)
    
    # Run campaign
    campaign = run_campaign(args.stress_levels, output_dir, logger)
    
    # Save full campaign data
    campaign_file = output_dir / "shear_cell_diagnosis.json"
    with open(campaign_file, "w") as f:
        json.dump(campaign, f, indent=2)
    logger.info(f"Wrote {campaign_file}")
    
    # Generate and save summary
    summary = generate_summary(campaign, logger)
    summary_file = output_dir / "shear_cell_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {summary_file}")
    
    logger.info("=" * 70)
    logger.info(f"Status: {summary['status']}")
    logger.info(f"Verdict: {summary['verdict']}")
    logger.info("=" * 70)
    
    # Exit code reflects status
    if summary["status"] == "FAIL":
        sys.exit(1)
    elif summary["status"] == "PARTIAL":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
