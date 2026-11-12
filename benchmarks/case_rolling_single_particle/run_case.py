#!/usr/bin/env python
"""CASE-DEM-ROLLING-SINGLE-PARTICLE -- single sphere rolling down an incline.

Benchmark case of the DEM case pool (owner 2026-11-12 算例池令): the classical
rigid-body check of a sphere rolling without slip on a plane of inclination
``theta``,

    a = g sin(theta) / (1 + I / (m R^2))            alpha = a / R

(``I / (m R^2) = 2/5`` for a solid sphere), with the rolling-resistance branch
of Ai et al. (2011) Powder Technology 206, 269-282 activated as a secondary
point (``mu_r > 0`` -> ``a = g (sin t - mu_r cos t) / (1 + I/(m R^2))``,
critical angle ``arctan(mu_r)``).

The case *reuses* the existing TensorDEM capability
(:func:`tensordem.rolling_resistance.simulate_sphere_on_incline`, Cundall-Strack
tangential contact + CDT rolling-resistance branch); no new solver is written
here.  What this file adds is the reproducible case driver: declared parameter
points, the estimator definitions, the in-code analytic reference (no
hand-typed constants), negative controls, and a JSON artifact.

Usage::

    PYTHONPATH=src python benchmarks/case_rolling_single_particle/run_case.py \
        --out artifacts/case_rolling_single_particle/result.json

Design (thresholds, controls, stopping rule) is pre-registered in
``prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from tensordem.rolling_resistance import (  # noqa: E402
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    incline_rolling_acceleration,
    incline_sliding_acceleration,
    pure_rolling_feasible,
    simulate_sphere_on_incline,
)

CASE_ID = "CASE-DEM-ROLLING-SINGLE-PARTICLE"
THETA_DEG = 15.0
TOL = 0.02                                   # declared acceptance tolerance
DURATION = 0.12                              # s
SETTLE = 0.03                                # s, excluded from the fits (>65 tangential periods)
KT_NONE = None                               # k_t defaults to k_n


# ----------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------
def ls_slope(time: torch.Tensor, series: torch.Tensor) -> tuple[float, float]:
    """Least-squares slope of ``series(t)`` and the coefficient of determination."""
    if series.dim() == 2:
        series = series[:, 0]
    t_mean = time.mean()
    y_mean = series.mean()
    dt = time - t_mean
    slope = (dt * (series - y_mean)).sum() / (dt ** 2).sum()
    residual = series - (y_mean + dt * slope)
    ss_res = (residual ** 2).sum()
    ss_tot = ((series - y_mean) ** 2).sum()
    r2 = 1.0 - float(ss_res / ss_tot.clamp_min(torch.finfo(torch.float64).tiny))
    return float(slope), r2


def analytic(theta_deg: float, mu_r: float, inertia_factor: float, g: float, radius: float) -> dict:
    """Closed-form reference, computed in-code from the sphere's inertia factor.

    Sign convention of the harness (see ``simulate_sphere_on_incline``):
    ``v > 0`` down-slope, ``omega > 0`` counter-clockwise, so rolling without
    slip means ``v + R omega = 0`` and therefore ``alpha_ref = -a_ref / R``.

    In addition the undamped Cundall-Strack tangential spring makes the
    *instantaneous* acceleration oscillate about ``a_ref``: with
    ``f = I/(m R^2)`` the exact solution of the slip oscillator gives

        a(t) = g sin(theta) (1 + f cos(omega_s t)) / (1 + f),
        omega_s = sqrt((1 + 1/f) k_t / m),

    i.e. ``a_min = g sin(theta)(1-f)/(1+f)``, ``a_max = g sin(theta)``,
    amplitude ``f g sin(theta)/(1+f)``.  These closed-form bounds are the
    reference for the measured instantaneous envelope.
    """
    theta = math.radians(theta_deg)
    gs = g * math.sin(theta)
    a = g * (math.sin(theta) - mu_r * math.cos(theta)) / (1.0 + inertia_factor)
    f = inertia_factor
    return {
        "a_ref": a,
        "alpha_ref": -a / radius,
        "alpha_magnitude_ref": abs(a) / radius,
        "sliding_without_inertia_a": g * (math.sin(theta) - mu_r * math.cos(theta)),
        "oscillation": {
            "a_max": gs,
            "a_min": gs * (1.0 - f) / (1.0 + f),
            "amplitude": f * gs / (1.0 + f),
            "std_of_sine": f * gs / (1.0 + f) / math.sqrt(2.0),
        },
        "theta_deg": theta_deg,
        "mu_r": mu_r,
        "critical_angle_deg": critical_incline_angle_deg(mu_r),
        "formula": "a = g (sin(theta) - mu_r cos(theta)) / (1 + I/(m R^2)); alpha = -a / R (harness sign convention)",
    }


def rel_error(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref) if ref != 0.0 else abs(got)


# ----------------------------------------------------------------------
# one parameter point
# ----------------------------------------------------------------------
def run_point(
    *,
    theta_deg: float = THETA_DEG,
    mu_r: float = 0.0,
    k_n: float = 2.0e5,
    dt: float = 2.0e-5,
    mu: float = 0.9,
    duration: float = DURATION,
    settle: float = SETTLE,
    keep_trace: bool = False,
    label: str = "",
) -> dict:
    cfg = SphereInclineConfig(k_n=k_n, dt=dt, mu=mu)
    rolling = RollingResistanceConfig(model="cdt", mu_r=mu_r)
    t0 = time.time()
    res = simulate_sphere_on_incline(
        [theta_deg], rolling, config=cfg, duration=duration, settle_time=settle
    )
    wall = time.time() - t0

    mask = res["steady_mask"]
    t_steady = res["time"][mask]
    v = res["v"][mask]
    omega = res["omega"][mask]
    slip = res["slip_velocity"][mask]

    a_t, r2_v = ls_slope(t_steady, v)
    alpha, r2_w = ls_slope(t_steady, omega)

    ref = analytic(theta_deg, mu_r, cfg.inertia_factor, cfg.gravity, cfg.radius)

    # instantaneous acceleration diagnostic (dt-centred difference on the raw trace)
    a_inst = (res["v"][1:, 0] - res["v"][:-1, 0]) / dt
    a_inst_steady = a_inst[int(settle / dt):]

    out = {
        "label": label,
        "inputs": {
            "theta_deg": theta_deg, "mu_r": mu_r, "k_n_N_per_m": k_n,
            "k_t_N_per_m": k_n if KT_NONE is None else KT_NONE, "mu": mu, "dt_s": dt,
            "duration_s": duration, "settle_s": settle,
            "radius_m": cfg.radius, "density_kg_per_m3": cfg.density,
            "mass_kg": cfg.mass, "inertia_kg_m2": cfg.inertia,
            "inertia_factor_I_over_mR2": cfg.inertia_factor,
            "gravity_m_per_s2": cfg.gravity, "steps": int(len(res["time"])),
        },
        "dimensionless": {
            "I_over_mR2": cfg.inertia_factor,
            "omega_n_dt": math.sqrt(k_n / cfg.mass) * dt,
            "omega_slip_dt": math.sqrt((1.0 + 1.0 / cfg.inertia_factor) * k_n / cfg.mass) * dt,
            "stiffness_ratio_kn_over_mg_over_R": k_n / (cfg.mass * cfg.gravity * math.cos(math.radians(theta_deg)) / cfg.radius),
            "friction_demand_over_cone": cfg.inertia_factor * math.tan(math.radians(theta_deg)) / (1.0 + cfg.inertia_factor) / mu,
            "pure_rolling_feasible": pure_rolling_feasible(theta_deg, mu_r, mu),
            "a_over_g_sin_theta": ref["a_ref"] / (cfg.gravity * math.sin(math.radians(theta_deg))),
        },
        "reference": ref,
        "got": {
            "a_t_m_per_s2": a_t,
            "alpha_rad_per_s2": alpha,
            "r2_velocity_fit": r2_v,
            "r2_spin_fit": r2_w,
            "slip_rms_m_per_s": float(slip.pow(2).mean().sqrt()),
            "slip_max_abs_m_per_s": float(slip.abs().max()),
            "a_instantaneous_mean": float(a_inst_steady.mean()),
            "a_instantaneous_std": float(a_inst_steady.std()),
            "a_instantaneous_min": float(a_inst_steady.min()),
            "a_instantaneous_max": float(a_inst_steady.max()),
            "wall_clock_s": round(wall, 2),
        },
    }
    osc = ref["oscillation"]
    measured_amplitude = 0.5 * (out["got"]["a_instantaneous_max"] - out["got"]["a_instantaneous_min"])
    out["got"]["oscillation_amplitude_measured"] = measured_amplitude
    out["got"]["oscillation_amplitude_ref"] = osc["amplitude"]
    closure = abs(a_t + cfg.radius * alpha) / abs(a_t) if abs(a_t) > 1e-3 else float("nan")
    out["error"] = {
        "a_rel_error": rel_error(a_t, ref["a_ref"]),
        "alpha_rel_error": rel_error(alpha, ref["alpha_ref"]),
        "alpha_sign_matches_convention": bool((alpha < 0) == (ref["alpha_ref"] < 0)) if ref["alpha_ref"] != 0 else True,
        "no_slip_closure": closure,
        "oscillation_amplitude_rel_error": rel_error(measured_amplitude, osc["amplitude"]) if abs(a_t) > 1e-3 else float("nan"),
        "oscillation_std_rel_error": rel_error(out["got"]["a_instantaneous_std"], osc["std_of_sine"]) if abs(a_t) > 1e-3 else float("nan"),
        "pass_a": rel_error(a_t, ref["a_ref"]) <= TOL,
        "pass_alpha": rel_error(alpha, ref["alpha_ref"]) <= TOL,
    }
    out["error"]["pass"] = bool(
        out["error"]["pass_a"] and out["error"]["pass_alpha"]
        and (closure != closure or closure <= TOL)  # NaN (static hold) is not a slip failure
        and out["error"]["alpha_sign_matches_convention"]
    )

    if keep_trace:
        stride = max(1, int(round(duration / dt / 600.0)))
        out["trace"] = {
            "stride": stride,
            "time": res["time"][::stride].tolist(),
            "v": res["v"][::stride, 0].tolist(),
            "omega": res["omega"][::stride, 0].tolist(),
            "slip": res["slip_velocity"][::stride, 0].tolist(),
        }
        # one oscillation period at full resolution, for the detail panel
        n_period = int(round(1.0 / (math.sqrt((1.0 + 1.0 / cfg.inertia_factor) * k_n / cfg.mass)) / dt * 3))
        out["trace_detail"] = {
            "dt": dt,
            "a_inst": a_inst[int(settle / dt): int(settle / dt) + n_period].tolist(),
        }
    return out


# ----------------------------------------------------------------------
# case assembly
# ----------------------------------------------------------------------
def build_case() -> dict:
    points: dict[str, dict] = {}

    # --- base point: the acceptance point of the case -------------------
    points["A_base"] = run_point(label="base: theta=15, mu_r=0, kn=2e5, dt=2e-5", keep_trace=True)

    # --- M1/M2 convergence: time-step refinement ------------------------
    convergence = []
    for dt in (4e-5, 2e-5, 1e-5, 5e-6):
        p = points["A_base"] if dt == 2e-5 else run_point(dt=dt, label=f"dt={dt:g}")
        if dt != 2e-5:
            points[f"dt_{dt:g}"] = p
        convergence.append({
            "dt_s": dt,
            "steps": p["inputs"]["steps"],
            "omega_slip_dt": p["dimensionless"]["omega_slip_dt"],
            "a_t": p["got"]["a_t_m_per_s2"],
            "alpha": p["got"]["alpha_rad_per_s2"],
            "a_rel_error": p["error"]["a_rel_error"],
            "alpha_rel_error": p["error"]["alpha_rel_error"],
            "a_instantaneous_std": p["got"]["a_instantaneous_std"],
            "oscillation_amplitude": p["got"]["oscillation_amplitude_measured"],
            "oscillation_amplitude_rel_error": p["error"]["oscillation_amplitude_rel_error"],
            "wall_clock_s": p["got"]["wall_clock_s"],
        })

    # --- window sensitivity: fit window must not matter -------------------
    windows = []
    for duration, settle, lab in ((0.06, 0.015, "window 0.045 s"), (0.12, 0.03, "base window 0.09 s"),
                                  (0.24, 0.06, "window 0.18 s")):
        if (duration, settle) == (DURATION, SETTLE):
            p = points["A_base"]
        else:
            p = run_point(duration=duration, settle=settle, label=lab)
            points[f"win_{duration:g}"] = p
        windows.append({
            "duration_s": duration,
            "settle_s": settle,
            "fit_window_s": duration - settle,
            "a_t": p["got"]["a_t_m_per_s2"],
            "alpha": p["got"]["alpha_rad_per_s2"],
            "a_rel_error": p["error"]["a_rel_error"],
            "alpha_rel_error": p["error"]["alpha_rel_error"],
        })

    # --- M3 stiffness invariance ----------------------------------------
    stiffness = []
    for k_n in (5e4, 2e5, 8e5):
        p = points["A_base"] if k_n == 2e5 else run_point(k_n=k_n, label=f"kn={k_n:g}")
        if k_n != 2e5:
            points[f"kn_{k_n:g}"] = p
        stiffness.append({
            "k_n_N_per_m": k_n,
            "a_t": p["got"]["a_t_m_per_s2"],
            "a_rel_error": p["error"]["a_rel_error"],
            "a_instantaneous_std": p["got"]["a_instantaneous_std"],
        })

    # --- M5 rolling-resistance branch (Ai et al. 2011) -------------------
    points["B_rolling"] = run_point(
        mu_r=0.1, label="rolling resistance mu_r=0.1 (Ai et al. 2011 CDT branch)"
    )
    pB = points["B_rolling"]

    # --- M6 critical angle (below theta_c the sphere must not accelerate)
    points["C_below_critical"] = run_point(
        theta_deg=5.0, mu_r=0.1, label="theta=5 deg < arctan(mu_r)=5.71 deg: static hold test"
    )
    pC = points["C_below_critical"]
    pC["error"]["creep_over_g_sin_theta"] = abs(pC["got"]["a_t_m_per_s2"]) / (
        pC["inputs"]["gravity_m_per_s2"] * math.sin(math.radians(5.0))
    )

    # --- negative control 1: friction cone below the pure-rolling need ---
    points["NEG_low_friction"] = run_point(
        mu=0.01, label="negative control: mu=0.01 < required 0.051 -> must slip and fail"
    )
    pN = points["NEG_low_friction"]
    pN["error"]["pass"] = bool(pN["error"]["a_rel_error"] <= TOL and pN["error"]["alpha_rel_error"] <= TOL)
    pN["sliding_reference"] = {
        "a_sliding_ref": incline_sliding_acceleration(THETA_DEG, 0.01),
        "note": "steady sliding branch a = g (sin t - mu cos t)",
    }
    pN["error"]["a_rel_error_vs_sliding_ref"] = rel_error(
        pN["got"]["a_t_m_per_s2"], pN["sliding_reference"]["a_sliding_ref"])

    # --- negative control 2: wrong reference without inertia correction --
    neg2_wrong = points["A_base"]["reference"]["sliding_without_inertia_a"]
    neg2_err = rel_error(points["A_base"]["got"]["a_t_m_per_s2"], neg2_wrong)

    summary = {
        "tolerance": TOL,
        "a_ref": points["A_base"]["reference"]["a_ref"],
        "alpha_ref": points["A_base"]["reference"]["alpha_ref"],
        "a_got": points["A_base"]["got"]["a_t_m_per_s2"],
        "alpha_got": points["A_base"]["got"]["alpha_rad_per_s2"],
        "a_rel_error": points["A_base"]["error"]["a_rel_error"],
        "alpha_rel_error": points["A_base"]["error"]["alpha_rel_error"],
        "no_slip_closure": points["A_base"]["error"]["no_slip_closure"],
        "verdict": "PASS" if points["A_base"]["error"]["pass"] else "FAIL",
        "rolling_resistance_verdict": "PASS" if pB["error"]["pass"] else "FAIL",
        "rolling_resistance_a_rel_error": pB["error"]["a_rel_error"],
        "negative_control_slip": {
            "mu": 0.01,
            "a_rel_error": pN["error"]["a_rel_error"],
            "rejected": not pN["error"]["pass"],
        },
        "oscillation_amplitude_ref": points["A_base"]["reference"]["oscillation"]["amplitude"],
        "oscillation_amplitude_measured": points["A_base"]["got"]["oscillation_amplitude_measured"],
        "oscillation_amplitude_rel_error": points["A_base"]["error"]["oscillation_amplitude_rel_error"],
        "sliding_branch_a_rel_error": pN["error"]["a_rel_error_vs_sliding_ref"],
        "stable_hold_creep_over_g_sin_theta": pC["error"]["creep_over_g_sin_theta"],
        "negative_control_no_inertia_reference": {
            "wrong_a": neg2_wrong,
            "rel_error_vs_wrong": neg2_err,
            "rejected": not (neg2_err <= TOL),
        },
    }

    return {
        "case_id": CASE_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "script": os.path.abspath(__file__),
            "argv": sys.argv,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "prereg": "prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json",
            "capability": "tensordem.rolling_resistance.simulate_sphere_on_incline",
            "reference": {
                "primary": "classical rigid-body dynamics a = g sin(theta)/(1+I/(mR^2)), alpha = a/R",
                "secondary": "Ai, Chen, Rotter, Ooi (2011) Powder Technology 206, 269-282, doi:10.1016/j.powtec.2010.11.003 (rolling-resistance models: M_r = mu_r |F_n| r_eff, critical angle arctan(mu_r)); implemented in src/tensordem/rolling_resistance.py",
                "access_note": "the Ai et al. 2011 PDF is NOT available in this environment; the numbers compared here are the closed-form relations (derived in-code), not transcribed table values",
            },
        },
        "summary": summary,
        "points": points,
        "convergence_dt": convergence,
        "window_sensitivity": windows,
        "stiffness_invariance": stiffness,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="artifacts/case_rolling_single_particle/result.json")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    case = build_case()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(case, fh, indent=1)

    s = case["summary"]
    print(json.dumps({
        "case": s["verdict"],
        "a_ref": round(s["a_ref"], 6), "a_got": round(s["a_got"], 6),
        "a_err_%": round(100 * s["a_rel_error"], 4),
        "alpha_ref": round(s["alpha_ref"], 4), "alpha_got": round(s["alpha_got"], 4),
        "alpha_err_%": round(100 * s["alpha_rel_error"], 4),
        "no_slip_closure": f"{s['no_slip_closure']:.2e}",
        "mu_r_branch": f"{s['rolling_resistance_verdict']} ({100*s['rolling_resistance_a_rel_error']:.3f}%)",
        "osc_amp": f"{s['oscillation_amplitude_measured']:.4f} vs {s['oscillation_amplitude_ref']:.4f}",
        "sliding_branch_err_%": round(100 * s["sliding_branch_a_rel_error"], 4),
        "neg_ctl_slip_rejected": s["negative_control_slip"]["rejected"],
        "neg_ctl_no_inertia_rejected": s["negative_control_no_inertia_reference"]["rejected"],
        "out": out,
    }, indent=1))
    return 0 if s["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())