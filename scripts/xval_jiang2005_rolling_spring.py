#!/usr/bin/env python
"""Cross-validation: TensorDEM EPSD rolling spring vs Jiang-2005 (kr-225-mur2-kn-R2).

Jiang et al. (2005) Comput. Geotech. 32, 340-357 propose the rolling stiffness
rule:

    k_r = 2.25 * mu_r^2 * k_n * R^2          (Jiang-2005)

TensorDEM currently uses:

    k_r = 2.25 * mu_r * k_n * R^2            (TensorDEM)

The difference is mu_r vs mu_r^2.  For mu_r < 1 the Jiang rule gives a softer
rolling spring, which affects the transient dynamics (oscillation frequency,
settling time) but NOT the steady-state rolling acceleration (which is set by
the plastic limit mu_r * F_n * r_eff, independent of k_r).

This script runs the same incline-harness tests with both rules and reports:
1. Acceleration parity (should be identical — steady state is k_r-independent)
2. Transient dynamics (oscillation amplitude, settling time — will differ)
3. Static threshold (should be identical — threshold is set by mu_r, not k_r)
4. Torque evolution curves (will differ in the elastic branch)

Usage:
    PYTHONPATH=src python scripts/xval_jiang2005_rolling_spring.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.rolling_resistance import (
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    fit_acceleration,
    incline_rolling_acceleration,
    pure_rolling_feasible,
    rolling_resistance_torque,
    simulate_sphere_on_incline,
)

# ---------------------------------------------------------------------------
# Harness parameters (match validate_rolling_resistance.py)
# ---------------------------------------------------------------------------
DT = 1.0e-4
K_N = 2.0e4
RADIUS = 0.01
DURATION = 0.10
SETTLE = 0.03
SLIDE_MU = 0.9
DENSITY = 900  # kg/m^3

# ---------------------------------------------------------------------------
# Rolling stiffness rules
# ---------------------------------------------------------------------------

def k_r_tensordem(mu_r: float, k_n: float = K_N, radius: float = RADIUS) -> float:
    """TensorDEM rule: k_r = 2.25 * mu_r * k_n * R^2."""
    return 2.25 * mu_r * k_n * radius ** 2


def k_r_jiang2005(mu_r: float, k_n: float = K_N, radius: float = RADIUS) -> float:
    """Jiang-2005 rule: k_r = 2.25 * mu_r^2 * k_n * R^2."""
    return 2.25 * (mu_r ** 2) * k_n * radius ** 2


def _epsd_tensordem(mu_r: float) -> RollingResistanceConfig:
    return RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r_tensordem(mu_r))


def _epsd_jiang2005(mu_r: float) -> RollingResistanceConfig:
    return RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r_jiang2005(mu_r))


def _harness() -> SphereInclineConfig:
    return SphereInclineConfig(radius=RADIUS, k_n=K_N, dt=DT)


# ---------------------------------------------------------------------------
# Study 1: Acceleration parity (steady state — should be identical)
# ---------------------------------------------------------------------------

def study_acceleration_parity(mu_r_values: Sequence[float]) -> list[dict[str, Any]]:
    """Compare simulated acceleration for both rules against analytic reference.

    The steady-state rolling acceleration is independent of k_r, so both rules
    should give the same result (within numerical tolerance).
    """
    rows: list[dict[str, Any]] = []
    for mu_r in mu_r_values:
        theta_c = critical_incline_angle_deg(mu_r)
        angles = sorted({
            1.25 * theta_c, 1.5 * theta_c, 2.0 * theta_c,
            max(3.0 * theta_c, 8.0), 12.0, 18.0,
        })
        for label, config_fn in [("tensordem", _epsd_tensordem), ("jiang2005", _epsd_jiang2005)]:
            result = simulate_sphere_on_incline(
                angles, config_fn(mu_r), config=_harness(),
                duration=DURATION, settle_time=SETTLE,
            )
            accel, r_squared = fit_acceleration(result)
            for index, theta in enumerate(angles):
                if theta <= theta_c:
                    continue
                rolling_ok = pure_rolling_feasible(theta, mu_r, SLIDE_MU)
                analytic = incline_rolling_acceleration(theta, mu_r) if rolling_ok else None
                simulated = float(accel[index])
                rows.append({
                    "rule": label,
                    "mu_r": mu_r,
                    "theta_deg": theta,
                    "theta_over_theta_c": theta / theta_c,
                    "regime": "rolling" if rolling_ok else "sliding",
                    "a_simulated_m_s2": simulated,
                    "a_analytic_m_s2": analytic,
                    "relative_error_pct": (
                        abs(simulated - analytic) / analytic * 100 if analytic and analytic > 0.05 else None
                    ),
                    "fit_r_squared": float(r_squared[index]),
                })
    return rows


# ---------------------------------------------------------------------------
# Study 2: Transient dynamics (oscillation amplitude, settling time)
# ---------------------------------------------------------------------------

def study_transient_dynamics(mu_r_values: Sequence[float]) -> list[dict[str, Any]]:
    """Compare transient oscillation characteristics.

    The rolling spring stiffness k_r sets the oscillation frequency:
        omega_s = sqrt((1 + 1/f) * k_r / m)
    where f = I/(mR^2) = 2/5 for a solid sphere.

    Jiang-2005's softer spring (mu_r^2 vs mu_r) gives lower frequency and
    smaller oscillation amplitude.
    """
    rows: list[dict[str, Any]] = []
    mass = 4.0 / 3.0 * math.pi * RADIUS ** 3 * DENSITY
    f_inertia = 0.4  # I/(mR^2) for solid sphere

    for mu_r in mu_r_values:
        theta_c = critical_incline_angle_deg(mu_r)
        theta = 1.5 * theta_c  # supercritical but not too steep

        for label, config_fn in [("tensordem", _epsd_tensordem), ("jiang2005", _epsd_jiang2005)]:
            config = config_fn(mu_r)
            result = simulate_sphere_on_incline(
                [theta], config, config=_harness(),
                duration=0.20, settle_time=0.0,
            )
            v = result["v"][:, 0].numpy()
            time_arr = result["time"].numpy()

            # Compute oscillation envelope (deviation from mean acceleration)
            a_mean = (v[-1] - v[0]) / (time_arr[-1] - time_arr[0])
            v_linear = v[0] + a_mean * time_arr
            deviation = v - v_linear
            osc_amplitude = float(deviation.max() - deviation.min()) / 2

            # Analytic oscillation frequency
            k_r_val = config.k_r
            omega_s_analytic = math.sqrt((1 + 1/f_inertia) * k_r_val / mass)
            freq_analytic_hz = omega_s_analytic / (2 * math.pi)

            # Period of oscillation
            period_analytic = 1.0 / freq_analytic_hz

            # Count oscillations in the run
            n_oscillations = 0.20 / period_analytic if period_analytic > 0 else 0

            rows.append({
                "rule": label,
                "mu_r": mu_r,
                "theta_deg": theta,
                "k_r_Nm_rad": k_r_val,
                "osc_amplitude_m_s": osc_amplitude,
                "freq_analytic_hz": freq_analytic_hz,
                "period_analytic_s": period_analytic,
                "n_oscillations_in_run": n_oscillations,
            })
    return rows


# ---------------------------------------------------------------------------
# Study 3: Static threshold (should be identical)
# ---------------------------------------------------------------------------

def study_static_threshold(mu_r_values: Sequence[float]) -> list[dict[str, Any]]:
    """Compare static threshold angle for both rules.

    The threshold theta_c = arctan(mu_r) is independent of k_r, so both rules
    should give the same result.
    """
    rows: list[dict[str, Any]] = []
    for mu_r in mu_r_values:
        theta_c = critical_incline_angle_deg(mu_r)
        for label, config_fn in [("tensordem", _epsd_tensordem), ("jiang2005", _epsd_jiang2005)]:
            # Test at theta_c * 0.95 (should be static) and theta_c * 1.05 (should roll)
            for factor, expected in [(0.95, "static"), (1.05, "rolling")]:
                theta = theta_c * factor
                result = simulate_sphere_on_incline(
                    [theta], config_fn(mu_r), config=_harness(),
                    duration=0.15, settle_time=0.05,
                )
                v_final = float(result["v"][-1, 0])
                is_moving = abs(v_final) > 1e-5
                rows.append({
                    "rule": label,
                    "mu_r": mu_r,
                    "theta_deg": theta,
                    "theta_over_theta_c": factor,
                    "expected": expected,
                    "v_final_m_s": v_final,
                    "is_moving": is_moving,
                    "correct": (expected == "static" and not is_moving) or (expected == "rolling" and is_moving),
                })
    return rows


# ---------------------------------------------------------------------------
# Study 4: Torque evolution curves
# ---------------------------------------------------------------------------

def study_torque_evolution(mu_r: float, omega_values: Sequence[float]) -> list[dict[str, Any]]:
    """Compare torque buildup for both rules under prescribed spin.

    The EPSD model integrates: M^{n+1} = M^n - k_r * omega * dt
    So the torque reaches the plastic limit faster with higher k_r.
    """
    rows: list[dict[str, Any]] = []
    f_n = torch.tensor([1.5], dtype=torch.float64)
    r_eff = torch.tensor([RADIUS], dtype=torch.float64)
    limit = mu_r * float(f_n) * RADIUS

    for omega_val in omega_values:
        omega = torch.tensor([[0.0, 0.0, -omega_val]], dtype=torch.float64)
        for label, config_fn in [("tensordem", _epsd_tensordem), ("jiang2005", _epsd_jiang2005)]:
            config = config_fn(mu_r)
            state = torch.zeros(1, 3, dtype=torch.float64)
            # Run until torque reaches 99% of limit or max steps
            steps_to_99 = None
            for step in range(1, 5001):
                torque, state = rolling_resistance_torque(
                    omega, f_n, r_eff, config, torque_state=state, dt=1e-4
                )
                mag = float(torque.norm())
                if steps_to_99 is None and mag >= 0.99 * limit:
                    steps_to_99 = step
            rows.append({
                "rule": label,
                "mu_r": mu_r,
                "omega_rad_s": omega_val,
                "k_r_Nm_rad": config.k_r,
                "limit_Nm": limit,
                "final_torque_Nm": float(state.norm()),
                "steps_to_99pct_limit": steps_to_99,
                "time_to_99pct_s": steps_to_99 * 1e-4 if steps_to_99 else None,
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt_val(v, digits=4):
    """Format a value, handling None."""
    if v is None:
        return "N/A"
    return f"{v:.{digits}g}"


def main() -> int:
    mu_r_values = [0.05, 0.1, 0.2, 0.3, 0.5]

    print("=" * 70)
    print("XVAL: TensorDEM vs Jiang-2005 EPSD Rolling Spring")
    print("=" * 70)
    print()

    # Print stiffness rules comparison
    print("Rolling stiffness rules:")
    print(f"  TensorDEM:  k_r = 2.25 * mu_r * k_n * R^2")
    print(f"  Jiang-2005: k_r = 2.25 * mu_r^2 * k_n * R^2")
    print()
    print("Stiffness values for test mu_r:")
    for mu_r in mu_r_values:
        kt = k_r_tensordem(mu_r)
        kj = k_r_jiang2005(mu_r)
        ratio = kj / kt if kt > 0 else 0
        print(f"  mu_r={mu_r:.2f}:  TensorDEM k_r={kt:.4f}, Jiang-2005 k_r={kj:.4f}, ratio={ratio:.4f}")
    print()

    # Study 1: Acceleration parity
    print("-" * 70)
    print("Study 1: Acceleration parity (steady state)")
    print("-" * 70)
    accel_rows = study_acceleration_parity(mu_r_values)
    for row in accel_rows[:12]:
        print(f"  {row['rule']:12s} mu_r={row['mu_r']:.2f} theta={row['theta_deg']:.2f} "
              f"a_sim={row['a_simulated_m_s2']:.6f} a_ref={fmt_val(row['a_analytic_m_s2'])} "
              f"err={fmt_val(row['relative_error_pct'])}%")
    print(f"  ... ({len(accel_rows)} total rows)")
    print()

    # Study 2: Transient dynamics
    print("-" * 70)
    print("Study 2: Transient dynamics (oscillation)")
    print("-" * 70)
    transient_rows = study_transient_dynamics(mu_r_values)
    for row in transient_rows:
        print(f"  {row['rule']:12s} mu_r={row['mu_r']:.2f} k_r={row['k_r_Nm_rad']:.4f} "
              f"osc_amp={row['osc_amplitude_m_s']:.6e} freq_ref={row['freq_analytic_hz']:.2f}Hz "
              f"period={row['period_analytic_s']:.4f}s n_osc={row['n_oscillations_in_run']:.1f}")
    print()

    # Study 3: Static threshold
    print("-" * 70)
    print("Study 3: Static threshold")
    print("-" * 70)
    threshold_rows = study_static_threshold(mu_r_values)
    for row in threshold_rows:
        status = "OK" if row["correct"] else "FAIL"
        print(f"  {row['rule']:12s} mu_r={row['mu_r']:.2f} theta/theta_c={row['theta_over_theta_c']:.2f} "
              f"v_final={row['v_final_m_s']:.6e} expected={row['expected']} [{status}]")
    print()

    # Study 4: Torque evolution
    print("-" * 70)
    print("Study 4: Torque evolution (steps to 99% limit)")
    print("-" * 70)
    omega_values = [0.1, 0.5, 1.0, 2.0]
    torque_rows = study_torque_evolution(0.2, omega_values)
    for row in torque_rows:
        print(f"  {row['rule']:12s} mu_r={row['mu_r']:.2f} omega={row['omega_rad_s']:.1f} "
              f"k_r={row['k_r_Nm_rad']:.4f} steps_to_99%={row['steps_to_99pct_limit']} "
              f"time_to_99%={fmt_val(row['time_to_99pct_s'])}s")
    print()

    # -----------------------------------------------------------------------
    # Write artifacts
    # -----------------------------------------------------------------------
    out_dir = ROOT / "artifacts" / "xval_jiang2005_rolling_spring"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "title": "XVAL: TensorDEM vs Jiang-2005 EPSD Rolling Spring",
        "reference": "Jiang, M.J., Yu, H.-S., Harris, D. (2005). Comput. Geotech. 32, 340-357.",
        "stiffness_rules": {
            "tensordem": "k_r = 2.25 * mu_r * k_n * R^2",
            "jiang2005": "k_r = 2.25 * mu_r^2 * k_n * R^2",
        },
        "harness": {
            "radius_m": RADIUS,
            "k_n_N_m": K_N,
            "dt_s": DT,
            "density_kg_m3": DENSITY,
        },
        "acceleration_parity": accel_rows,
        "transient_dynamics": transient_rows,
        "static_threshold": threshold_rows,
        "torque_evolution": torque_rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write CSVs
    for name, rows in [
        ("acceleration_parity.csv", accel_rows),
        ("transient_dynamics.csv", transient_rows),
        ("static_threshold.csv", threshold_rows),
        ("torque_evolution.csv", torque_rows),
    ]:
        if rows:
            with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    print(f"Artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
