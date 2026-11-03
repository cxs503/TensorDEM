"""Generate the rotational-DOF verification figures and the measured-numbers JSON.

Outputs (all under ``artifacts/``):

* ``dem3d_rotation_spindown.png``      spin histories of the three rolling laws
* ``dem3d_rotation_plane.png``         sliding-to-rolling and rolling deceleration
* ``dem3d_rotation_errors.png``        relative error against the closed-form references
* ``dem3d_rotation_metrics.json``      programmatically injected numbers for the docs/report

Run: ``python scripts/make_dem3d_rotation_figures.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.rotation_verification import (  # noqa: E402
    PlaneCaseConfig,
    SpinDownCaseConfig,
    build_plane_case,
    build_spin_down_case,
    cdt_spin,
    epsd_clamp_peak,
    epsd_stop_time,
    epsd_spin,
    legacy_rolling_deceleration,
    rolling_deceleration,
    slip_to_roll_speed_ratio,
    viscous_spin,
    viscous_time_constant,
)

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def relative_error(measured: float, reference: float) -> float:
    return abs(measured - reference) / abs(reference)


def _fit_slope(time: torch.Tensor, values: torch.Tensor) -> float:
    centred = time - time.mean()
    return float((centred * (values - values.mean())).sum() / (centred ** 2).sum())


# ---------------------------------------------------------------------------
# 1. Spin-down of the three rolling-resistance laws
# ---------------------------------------------------------------------------
def spin_down_study() -> dict:
    spin = 10.0
    duration = 0.6
    results: dict[str, dict] = {}
    cases = {
        "cdt": SpinDownCaseConfig(mu_r=0.05, rolling_model="cdt", dt=1.0e-4),
        "viscous": SpinDownCaseConfig(mu_r=0.05, rolling_model="viscous",
                                      rolling_omega_ref=1.0, dt=1.0e-4),
        "epsd": SpinDownCaseConfig(mu_r=0.05, rolling_model="epsd",
                                   rolling_stiffness=5.0e-5, dt=1.0e-4),
    }
    for name, config in cases.items():
        dem, loads = build_spin_down_case(config, spin=spin)
        steps = int(round(duration / config.dt))
        times, spins = [], []
        for step in range(steps):
            dem.step(loads)
            times.append((step + 1) * config.dt)
            spins.append(float(dem.omega[1, 2]))
        time = torch.tensor(times)
        measured = torch.tensor(spins)
        if name == "cdt":
            reference = torch.tensor([
                cdt_spin(spin, config.mu_r, config.normal_force, config.effective_radius,
                         config.inertia, float(t)) for t in time])
        elif name == "viscous":
            reference = torch.tensor([
                viscous_spin(spin, config.mu_r, config.normal_force, config.effective_radius,
                             config.inertia, config.rolling_omega_ref, float(t)) for t in time])
        else:
            stop = epsd_stop_time(spin, config.rolling_stiffness, config.inertia)
            reference = torch.tensor([
                epsd_spin(spin, config.rolling_stiffness, config.inertia, float(t))
                if t <= stop else 0.0 for t in time])
        # Compare over the window where the reference is still above 20 % of spin,
        # i.e. away from the zero crossing where the relative error is meaningless.
        mask = reference > 0.2 * spin
        error = relative_error(float(measured[mask][-1]), float(reference[mask][-1]))
        results[name] = {
            "time": time.tolist(),
            "measured": measured.tolist(),
            "reference": reference.tolist(),
            "error_at_last_above_20pct": error,
            "normal_force_N": config.normal_force,
            "effective_radius_m": config.effective_radius,
            "limit_torque_Nm": config.mu_r * config.normal_force * config.effective_radius,
            "epsd_elastic_peak_Nm": (epsd_clamp_peak(spin, config.rolling_stiffness,
                                                     config.inertia) if name == "epsd" else None),
            "epsd_stop_time_s": (epsd_stop_time(spin, config.rolling_stiffness, config.inertia)
                                 if name == "epsd" else None),
            "viscous_time_constant_s": (viscous_time_constant(
                config.mu_r, config.normal_force, config.effective_radius,
                config.inertia, config.rolling_omega_ref) if name == "viscous" else None),
        }
    return {"spin0": spin, "duration": duration, "models": results}


# ---------------------------------------------------------------------------
# 2. Sphere on a plane: sliding -> rolling, then pure-rolling deceleration
# ---------------------------------------------------------------------------
def plane_study() -> dict:
    v0 = 0.5
    config = PlaneCaseConfig(mu_r=0.0)
    dem, loads = build_plane_case(config)
    dem.velocities[0, 0] = v0
    times, speeds, spins, slip = [], [], [], []
    first_rolling = None
    first_rolling_time = None
    for step in range(4000):
        dem.step(loads)
        speed = float(dem.velocities[0, 0])
        spin = float(dem.omega[0, 1])
        times.append((step + 1) * config.dt)
        speeds.append(speed)
        spins.append(spin)
        slip.append(speed - spin * config.radius)
        if first_rolling is None and step > 50 and slip[-1] <= 0.0:
            first_rolling, first_rolling_time = speed, times[-1]
    time = torch.tensor(times)
    speed = torch.tensor(speeds)
    spin = torch.tensor(spins)
    sliding_slope = -_fit_slope(time[50:700], speed[50:700])
    sliding_reference = config.mu_platen * config.gravity
    rolling_reference = v0 * slip_to_roll_speed_ratio()

    # Pure-rolling deceleration with the rolling-resistance torque switched on.
    mu_r = 0.05
    decel_config = PlaneCaseConfig(mu_r=mu_r, dt=2.0e-5)
    decel_dem, decel_loads = build_plane_case(decel_config)
    decel_v0 = 0.2
    decel_dem.velocities[0, 0] = decel_v0
    decel_dem.omega[0, 1] = decel_v0 / decel_config.radius
    decel_times, decel_speeds = [], []
    for step in range(2500):
        decel_dem.step(decel_loads)
        decel_times.append((step + 1) * decel_config.dt)
        decel_speeds.append(float(decel_dem.velocities[0, 0]))
    decel_time = torch.tensor(decel_times[500:])
    decel_speed = torch.tensor(decel_speeds[500:])
    measured_decel = -_fit_slope(decel_time, decel_speed)
    analytic_decel = rolling_deceleration(mu_r, decel_config.gravity)
    legacy_decel = legacy_rolling_deceleration(mu_r, decel_config.gravity)

    # Legacy (translation-only) path on the same pure-rolling initial state.
    legacy_config = PlaneCaseConfig(mu_r=mu_r, dt=2.0e-5, enable_rotation=False)
    legacy_dem, legacy_loads = build_plane_case(legacy_config)
    legacy_dem.velocities[0, 0] = decel_v0
    legacy_times, legacy_speeds = [], []
    for step in range(1200):
        legacy_dem.step(legacy_loads)
        legacy_times.append((step + 1) * legacy_config.dt)
        legacy_speeds.append(float(legacy_dem.velocities[0, 0]))
    legacy_measured = -_fit_slope(torch.tensor(legacy_times[200:]),
                                  torch.tensor(legacy_speeds[200:]))

    return {
        "v0": v0,
        "sliding": {
            "time": times, "speed": speeds, "spin_radius": (config.radius * spin).tolist(),
            "slip": slip, "first_rolling_speed": first_rolling,
            "first_rolling_time": first_rolling_time,
            "reference_speed": rolling_reference,
            "sliding_deceleration": sliding_slope,
            "sliding_reference": sliding_reference,
        },
        "rolling": {
            "mu_r": mu_r, "measured_deceleration": measured_decel,
            "analytic_deceleration": analytic_decel,
            "legacy_analytic_deceleration": legacy_decel,
            "legacy_measured_deceleration": legacy_measured,
            "time": decel_times[500:], "speed": decel_speeds[500:],
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_spin_down(study: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"cdt": ("tab:blue", "CDT (saturated)"),
              "viscous": ("tab:orange", "viscous"),
              "epsd": ("tab:green", "EPSD (elastic-plastic)")}
    for name, data in study["models"].items():
        colour, label = styles[name]
        time = data["time"]
        axes[0].plot(time, data["reference"], color=colour, lw=2.2, alpha=0.35,
                     label=f"{label} analytic")
        axes[0].plot(time, data["measured"], color=colour, lw=1.1, ls="--",
                     label=f"{label} solver")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel(r"spin $\omega_z$ [rad/s]")
    axes[0].set_title("Constant normal force spin-down (F_n = %.2f N)" %
                      next(iter(study["models"].values()))["normal_force_N"])
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    for name, data in study["models"].items():
        colour, label = styles[name]
        axes[1].plot(data["time"], data["measured"], color=colour, lw=1.2, label=label)
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel(r"spin $\omega_z$ [rad/s]")
    axes[1].set_title("Solver spin: elastic hold (EPSD) vs saturating decay")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "dem3d_rotation_spindown.png", dpi=115)
    plt.close(fig)


def figure_plane(study: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sliding = study["sliding"]
    time = sliding["time"]
    axes[0].plot(time, sliding["speed"], color="tab:blue", lw=1.2, label="v (solver)")
    axes[0].plot(time, sliding["spin_radius"], color="tab:red", lw=1.2, ls="--",
                 label=r"$\omega R$ (solver)")
    axes[0].axhline(sliding["reference_speed"], color="k", lw=1.0, ls=":",
                    label=r"$5/7\,v_0$ = %.4f" % sliding["reference_speed"])
    if sliding["first_rolling_time"] is not None:
        axes[0].axvline(sliding["first_rolling_time"], color="grey", lw=0.8)
        axes[0].annotate("first pure rolling\nv = %.4f m/s" % sliding["first_rolling_speed"],
                         xy=(sliding["first_rolling_time"], sliding["reference_speed"]),
                         xytext=(0.35, 0.42), fontsize=8,
                         arrowprops=dict(arrowstyle="->", lw=0.7))
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("speed [m/s]")
    axes[0].set_title(r"Slip $\to$ rolling transition ($\mu$ = %.2f, $\mu_r$ = 0)" %
                      PlaneCaseConfig().mu_platen)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    rolling = study["rolling"]
    axes[1].plot(rolling["time"], rolling["speed"], color="tab:blue", lw=1.2,
                 label="solver (rotation on)")
    t = torch.tensor(rolling["time"])
    v_start = rolling["speed"][0]
    axes[1].plot(rolling["time"],
                 (v_start - rolling["analytic_deceleration"] * (t - t[0])).tolist(),
                 color="k", lw=1.0, ls="--",
                 label="analytic $\\mu_r g/1.4$ = %.3f m/s$^2$" % rolling["analytic_deceleration"])
    axes[1].plot(rolling["time"],
                 (v_start - rolling["legacy_analytic_deceleration"] * (t - t[0])).tolist(),
                 color="tab:red", lw=1.0, ls=":",
                 label="legacy surrogate $\\mu_r g$ = %.3f m/s$^2$"
                       % rolling["legacy_analytic_deceleration"])
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("speed [m/s]")
    axes[1].set_title("Pure-rolling deceleration ($\\mu_r$ = %.2f)" % rolling["mu_r"])
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "dem3d_rotation_plane.png", dpi=115)
    plt.close(fig)


def figure_errors(study: dict, plane: dict) -> None:
    labels, errors = [], []
    for name, data in study["models"].items():
        labels.append(f"spin-down {name}")
        errors.append(data["error_at_last_above_20pct"])
    labels.append("sliding decel. $\\mu g$")
    errors.append(relative_error(plane["sliding"]["sliding_deceleration"],
                                 plane["sliding"]["sliding_reference"]))
    labels.append("rolling speed $5/7 v_0$")
    errors.append(relative_error(plane["sliding"]["first_rolling_speed"],
                                 plane["sliding"]["reference_speed"]))
    labels.append("rolling decel. (rotation)")
    errors.append(relative_error(plane["rolling"]["measured_deceleration"],
                                 plane["rolling"]["analytic_deceleration"]))
    labels.append("rolling decel. (legacy)")
    errors.append(relative_error(plane["rolling"]["legacy_measured_deceleration"],
                                 plane["rolling"]["analytic_deceleration"]))

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    colours = ["tab:red" if value > 0.03 else "tab:green" for value in errors]
    bars = ax.barh(labels, [100 * value for value in errors], color=colours, alpha=0.85)
    ax.axvline(3.0, color="k", ls="--", lw=1.0, label="3 % acceptance")
    ax.set_xlabel("relative error vs closed-form reference [%]")
    ax.set_title("Rotational-DOF verification: error ledger")
    ax.set_xscale("symlog")
    for bar, value in zip(bars, errors):
        ax.text(100 * value * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{100 * value:.3g}%", va="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "dem3d_rotation_errors.png", dpi=115)
    plt.close(fig)


def main() -> dict:
    spin = spin_down_study()
    plane = plane_study()
    figure_spin_down(spin)
    figure_plane(plane)
    figure_errors(spin, plane)

    metrics = {
        "spin_down": {
            name: {key: value for key, value in data.items()
                   if key not in ("time", "measured", "reference")}
            for name, data in spin["models"].items()
        },
        "spin_down_spin0": spin["spin0"],
        "plane": {
            "v0": plane["v0"],
            "first_rolling_speed": plane["sliding"]["first_rolling_speed"],
            "first_rolling_time": plane["sliding"]["first_rolling_time"],
            "reference_speed_5_7": plane["sliding"]["reference_speed"],
            "sliding_deceleration": plane["sliding"]["sliding_deceleration"],
            "sliding_reference": plane["sliding"]["sliding_reference"],
            "measured_rolling_deceleration": plane["rolling"]["measured_deceleration"],
            "analytic_rolling_deceleration": plane["rolling"]["analytic_deceleration"],
            "legacy_measured_deceleration": plane["rolling"]["legacy_measured_deceleration"],
            "legacy_analytic_deceleration": plane["rolling"]["legacy_analytic_deceleration"],
        },
    }
    plane_data = {
        "sliding": {key: plane["sliding"][key] for key in
                    ("time", "speed", "spin_radius", "slip")},
        "rolling": {key: plane["rolling"][key] for key in ("time", "speed")},
    }
    (ARTIFACTS / "dem3d_rotation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "dem3d_rotation_series.json").write_text(json.dumps(plane_data, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()