#!/usr/bin/env python3
"""Parameter sweep for freeze-bond strength and failure mode transition.

Scans temperature T ∈ [-20, -1] °C × contact pressure P ∈ [0.01, 1] MPa ×
contact time t ∈ [1, 1000] s and classifies failure modes, locates the
transition temperature, and quantifies the peak load jump.

Usage:
    python tools/run_freeze_bond_sweep.py [--output results.json]

Output:
    JSON file with sweep results, transition analysis, and validation metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.freeze_bond import (
    FreezeBondConfig,
    FreezeBondResult,
    T_FREEZE_FRESH,
    freeze_bond_strength,
    failure_mode,
    transition_temperature,
    peak_load_jump,
    sastry_metzger_error,
    sigma_bond_temperature,
    sigma_ice_temperature,
    SASTRY_METZGER_1986_DATA,
)


def _json_safe(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_sweep(
    t_range: tuple[float, float, int] = (-20.0, -1.0, 40),
    p_range: tuple[float, float, int] = (0.01, 1.0, 20),
    time_range: tuple[float, float, int] = (1.0, 1000.0, 10),
    config: FreezeBondConfig | None = None,
) -> dict:
    """Run the full parameter sweep.

    Args:
        t_range: (T_min, T_max, n_points) in °C.
        p_range: (P_min, P_max, n_points) in MPa.
        time_range: (t_min, t_max, n_points) in seconds (log-spaced).
        config: Freeze-bond configuration.

    Returns:
        Dict with sweep results and analysis.
    """
    cfg = config if config is not None else FreezeBondConfig()

    t_vals = np.linspace(t_range[0], t_range[1], t_range[2])
    p_vals = np.linspace(p_range[0], p_range[1], p_range[2])
    time_vals = np.logspace(
        math.log10(time_range[0]), math.log10(time_range[1]), time_range[2]
    )

    results = []
    mode_counts = {"cohesive": 0, "adhesive": 0, "no_bond": 0}

    for t in t_vals:
        for p in p_vals:
            for tm in time_vals:
                res = freeze_bond_strength(float(t), float(p), float(tm), cfg)
                results.append({
                    "temperature_c": round(float(t), 4),
                    "pressure_mpa": round(float(p), 4),
                    "contact_time_s": round(float(tm), 4),
                    "sigma_bond_mpa": round(float(res.sigma_bond_mpa), 6),
                    "sigma_ice_mpa": round(float(res.sigma_ice_mpa), 6),
                    "failure_mode": res.failure_mode,
                    "is_bonded": bool(res.is_bonded),
                })
                mode_counts[res.failure_mode] += 1

    # Transition analysis at reference conditions
    t_trans = transition_temperature(cfg, contact_pressure_mpa=0.1, contact_time_s=100.0)
    jump = peak_load_jump(cfg, contact_pressure_mpa=0.1, contact_time_s=100.0)

    # Sastry & Metzger validation
    sm_err = sastry_metzger_error()

    # Contact time exponent validation
    # sigma_bond ∝ t^alpha => log(sigma) = alpha * log(t) + const
    t_ref_time = -10.0  # °C
    p_ref_time = 0.1  # MPa
    time_test = np.logspace(0, 3, 20)  # 1 to 1000 s
    sigma_times = []
    for tm in time_test:
        sb = sigma_bond_temperature(t_ref_time, cfg.t_freeze, cfg.sigma_0, cfg.beta,
                                    cfg.gamma, cfg.t_gamma)
        sb = sb * (tm / cfg.t_ref) ** cfg.alpha
        sigma_times.append(sb)

    # Fit alpha from log-log slope
    log_t = np.log(time_test)
    log_s = np.log([s for s in sigma_times if s > 0])
    log_t_valid = log_t[:len(log_s)]
    if len(log_s) > 1:
        alpha_fitted = float(np.polyfit(log_t_valid, log_s, 1)[0])
    else:
        alpha_fitted = cfg.alpha

    alpha_error = abs(alpha_fitted - cfg.alpha) / cfg.alpha if cfg.alpha > 0 else 0.0

    return {
        "config": {
            "t_freeze": cfg.t_freeze,
            "sigma_0": cfg.sigma_0,
            "beta": cfg.beta,
            "gamma": cfg.gamma,
            "t_gamma": cfg.t_gamma,
            "alpha": cfg.alpha,
            "t_ref": cfg.t_ref,
            "sigma_ice_0": cfg.sigma_ice_0,
            "beta_ice": cfg.beta_ice,
            "pressure_exponent": cfg.pressure_exponent,
            "p_ref": cfg.p_ref,
        },
        "sweep_ranges": {
            "temperature_c": list(t_range),
            "pressure_mpa": list(p_range),
            "contact_time_s": list(time_range),
        },
        "n_samples": len(results),
        "mode_counts": mode_counts,
        "transition_temperature_c": round(float(t_trans), 4),
        "peak_load_jump": {
            "t_transition_c": round(float(jump["t_transition_c"]), 4),
            "peak_load_below_mpa": round(float(jump["peak_load_below_mpa"]), 6),
            "peak_load_above_mpa": round(float(jump["peak_load_above_mpa"]), 6),
            "delta_f_over_f_mean": round(float(jump["delta_f_over_f_mean"]), 4),
            "failure_below": jump["failure_below"],
            "failure_above": jump["failure_above"],
        },
        "sastry_metzger_validation": {
            "mean_relative_error": round(float(sm_err["mean_relative_error"]), 4),
            "max_relative_error": round(float(sm_err["max_relative_error"]), 4),
            "fraction_within_15pct": round(float(sm_err["fraction_within_15pct"]), 4),
            "n_points": sm_err["n_points"],
            "per_point": [
                {
                    "temperature_c": p["temperature_c"],
                    "experimental_mpa": p["experimental_mpa"],
                    "predicted_mpa": round(float(p["predicted_mpa"]), 4),
                    "relative_error": round(float(p["relative_error"]), 4),
                }
                for p in sm_err["points"]
            ],
        },
        "contact_time_exponent": {
            "alpha_input": cfg.alpha,
            "alpha_fitted": round(float(alpha_fitted), 4),
            "relative_error": round(float(alpha_error), 4),
        },
        "negative_control": {
            "description": "T > T_freeze => sigma_bond = 0",
            "sigma_bond_at_1C": float(sigma_bond_temperature(1.0, cfg.t_freeze, cfg.sigma_0, cfg.beta, cfg.gamma, cfg.t_gamma)),
            "sigma_bond_at_0C": float(sigma_bond_temperature(0.0, cfg.t_freeze, cfg.sigma_0, cfg.beta, cfg.gamma, cfg.t_gamma)),
            "sigma_bond_at_minus_0_1C": float(sigma_bond_temperature(-0.1, cfg.t_freeze, cfg.sigma_0, cfg.beta, cfg.gamma, cfg.t_gamma)),
        },
        # Store a subset of results for plotting
        "sample_results": results[::max(1, len(results) // 100)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze-bond parameter sweep")
    parser.add_argument("--output", "-o", default="runs/freeze_bond_sweep.json",
                        help="Output JSON file path")
    parser.add_argument("--t-min", type=float, default=-20.0, help="Min temperature (°C)")
    parser.add_argument("--t-max", type=float, default=-1.0, help="Max temperature (°C)")
    parser.add_argument("--t-n", type=int, default=40, help="Number of temperature points")
    parser.add_argument("--p-min", type=float, default=0.01, help="Min pressure (MPa)")
    parser.add_argument("--p-max", type=float, default=1.0, help="Max pressure (MPa)")
    parser.add_argument("--p-n", type=int, default=20, help="Number of pressure points")
    parser.add_argument("--t-min-time", type=float, default=1.0, help="Min contact time (s)")
    parser.add_argument("--t-max-time", type=float, default=1000.0, help="Max contact time (s)")
    parser.add_argument("--t-n-time", type=int, default=10, help="Number of time points")
    args = parser.parse_args()

    result = run_sweep(
        t_range=(args.t_min, args.t_max, args.t_n),
        p_range=(args.p_min, args.p_max, args.p_n),
        time_range=(args.t_min_time, args.t_max_time, args.t_n_time),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Sweep complete: {result['n_samples']} samples")
    print(f"  Mode counts: {result['mode_counts']}")
    print(f"  Transition temperature: {result['transition_temperature_c']:.2f} °C")
    print(f"  Peak load jump (ΔF/F_mean): {result['peak_load_jump']['delta_f_over_f_mean']:.2%}")
    print(f"  Sastry & Metzger MAE: {result['sastry_metzger_validation']['mean_relative_error']:.2%}")
    print(f"  Contact time exponent error: {result['contact_time_exponent']['relative_error']:.2%}")
    print(f"  Negative control (T>0°C): σ_bond = {result['negative_control']['sigma_bond_at_1C']:.6f} MPa")
    print(f"  Results written to: {output_path}")


if __name__ == "__main__":
    main()
