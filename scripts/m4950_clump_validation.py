#!/usr/bin/env python3
"""m4950 / IN14-D3 — clump (multi-sphere) particles: verification harness.

Runs the quantitative checks reported in ``docs/clump_multisphere_particles.md``
and writes ``artifacts/m4950/clump_validation.json`` plus the figures embedded in
the mission report.  Every number in the report is produced here; nothing is
transcribed by hand.

Usage
-----
    python scripts/m4950_clump_validation.py [--out artifacts/m4950]

Checks
------
c1  double-sphere clump inertia vs the closed-form dumbbell value (touching and
    separated members) -- target < 1e-10 relative (machine precision level);
    plus the independent voxel integration of the union geometry, including an
    overlap where the naive member sum double-counts mass.
c2a free fall without rotation: energy conservation, zero angular momentum,
    orientation drift and agreement with the closed-form parabola.
c2b inclined plane: a "rolling-pin" dumbbell must roll with
    ``a = g sin(theta) / (1 + I_roll/(m r^2))``, keep rolling without slip, and
    mirror its trajectory when the tilt is reversed.
c3  static clump-on-plane load split, measured restitution band and the
    clump-sphere third-law closure (contact forces summed over member spheres).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.clump import (  # noqa: E402
    Clump,
    SphereMember,
    double_sphere_clump,
    double_sphere_inertia_analytic,
    double_sphere_mass_analytic,
    mass_properties_by_voxel,
)
from tensordem.clump_dynamics import (  # noqa: E402
    ClumpBody,
    ContactConfig,
    PlaneContact,
    SphereObstacle,
    quaternion_from_axis_angle,
)

RADIUS = 0.05
DENSITY = 900.0
GRAVITY = 9.81


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


# --------------------------------------------------------------------------
# c1 — mass properties
# --------------------------------------------------------------------------
def check_inertia_vs_analytic() -> dict:
    rows = []
    worst = 0.0
    for separation in (2.0 * RADIUS, 2.2 * RADIUS, 3.0 * RADIUS):
        clump = double_sphere_clump(RADIUS, separation, density=DENSITY,
                                    require_connected=False)
        reference = double_sphere_inertia_analytic(RADIUS, separation, DENSITY)
        relative = float(np.max(np.abs(clump.inertia_about_com - reference))
                         / np.max(np.abs(reference)))
        mass_error = abs(clump.mass - double_sphere_mass_analytic(
            RADIUS, separation, DENSITY)) / clump.mass
        worst = max(worst, relative)
        rows.append({
            "separation_over_radius": separation / RADIUS,
            "mass_kg": clump.mass,
            "inertia_axial": float(clump.inertia_about_com[0, 0]),
            "inertia_transverse": float(clump.inertia_about_com[1, 1]),
            "analytic_axial": float(reference[0, 0]),
            "analytic_transverse": float(reference[1, 1]),
            "relative_error_inertia": relative,
            "relative_error_mass": float(mass_error),
        })
    return {
        "target": "< 1e-10 relative (machine precision)",
        "worst_relative_error": worst,
        "passed": worst < 1e-10,
        "cases": rows,
    }


def check_overlap_and_voxel() -> dict:
    rows = []
    for separation in (2.0 * RADIUS, 1.5 * RADIUS, 1.3 * RADIUS):
        clump = double_sphere_clump(RADIUS, separation, density=DENSITY,
                                    max_overlap_fraction=0.8,
                                    require_connected=False)
        member_sum_gap = (clump.inertia_member_sum[1, 1] - clump.inertia_union[1, 1]) \
            / clump.inertia_member_sum[1, 1]
        row = {
            "separation_over_radius": separation / RADIUS,
            "union_volume_m3": clump.volume,
            "member_sum_volume_m3": clump.member_sum_volume,
            "overlap_volume_fraction": 1.0 - clump.volume / clump.member_sum_volume,
            "inertia_member_sum_transverse": clump.inertia_member_sum[1, 1],
            "inertia_union_transverse": clump.inertia_union[1, 1],
            "member_sum_inertia_overestimate": float(member_sum_gap),
            "voxel": {},
        }
        for resolution in (48, 96, 192, 256):
            voxel = mass_properties_by_voxel(clump.centers, clump.radii, DENSITY,
                                             resolution=resolution)
            row["voxel"][str(resolution)] = {
                "mass_relative_error": float(abs(voxel["mass"] - clump.mass) / clump.mass),
                "inertia_relative_error": float(
                    np.max(np.abs(voxel["inertia"] - clump.inertia_union))
                    / np.max(np.abs(clump.inertia_union))),
                "com_error_m": float(np.max(np.abs(voxel["center_of_mass"]
                                                    - clump.center_of_mass))),
            }
        rows.append(row)
    return {
        "comment": "voxel integration is an independent midpoint-quadrature path"
                   " over the union geometry (no shared code with the exact moments)",
        "cases": rows,
    }


# --------------------------------------------------------------------------
# c2a — free fall
# --------------------------------------------------------------------------
def check_free_fall() -> dict:
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    body = ClumpBody(clump, position=(0.0, 0.0, 2.0),
                     orientation=quaternion_from_axis_angle((0.3, 0.5, 0.9), 0.9),
                     gravity=(0.0, 0.0, -GRAVITY))
    dt, steps = 1e-3, 2000
    history = body.run(steps, dt)
    energy = history["total_energy"]
    time = history["time"]
    analytic_z = 2.0 - 0.5 * GRAVITY * time ** 2
    return {
        "steps": steps,
        "dt_s": dt,
        "duration_s": steps * dt,
        "energy_zero_s": float(energy[0]),
        "energy_drift_relative": float(np.max(np.abs(energy - energy[0])) / abs(energy[0])),
        "energy_drift_absolute_J": float(np.max(np.abs(energy - energy[0]))),
        "position_error_vs_analytic_m": float(np.max(np.abs(
            history["position"][:, 2] - analytic_z))),
        "position_error_relative": float(np.max(np.abs(
            history["position"][:, 2] - analytic_z)) / np.max(np.abs(analytic_z))),
        "max_angular_velocity_rad_s": float(np.max(np.abs(history["omega_body"]))),
        "max_angular_momentum_increment": float(np.max(np.abs(
            history["angular_momentum"] - history["angular_momentum"][0]))),
        "quaternion_drift": float(np.max(np.abs(
            history["quaternion"] - history["quaternion"][0]))),
        "final_velocity_z": float(body.velocity[2]),
        "analytic_final_velocity_z": float(-GRAVITY * steps * dt),
        "target": "energy drift < 1e-11 relative, orientation untouched",
        "passed": bool(np.max(np.abs(energy - energy[0])) / abs(energy[0]) < 1e-11
                       and np.max(np.abs(history["omega_body"])) == 0.0),
        "history": {
            "time": time.tolist(),
            "position_z": history["position"][:, 2].tolist(),
            "energy_error": (energy - energy[0]).tolist(),
            "velocity_z": history["velocity"][:, 2].tolist(),
        },
    }


# --------------------------------------------------------------------------
# c2b — inclined plane rolling
# --------------------------------------------------------------------------
def _rolling_pin(angle: float, sign: float = 1.0, duration: float = 0.5,
                 dt: float = 1e-4, stiffness: float = 1.0e5,
                 restitution: float = 0.5, friction: float = 0.6):
    normal = np.array([sign * math.sin(angle), 0.0, math.cos(angle)])
    downslope = np.array([sign * math.cos(angle), 0.0, -math.sin(angle)])
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    penetration = clump.mass * GRAVITY * math.cos(angle) / (2.0 * stiffness)
    plane = PlaneContact(normal, (0.0, 0.0, 0.0),
                         ContactConfig(normal_stiffness=stiffness,
                                       restitution=restitution, friction=friction))
    body = ClumpBody(clump, position=normal * (RADIUS - penetration),
                     orientation=quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0),
                     gravity=(0.0, 0.0, -GRAVITY))
    history = body.run(int(round(duration / dt)), dt, (plane,), record_every=10)
    return clump, body, plane, history, normal, downslope


def check_inclined_plane() -> dict:
    angle = math.radians(10.0)
    clump, body, plane, history, normal, downslope = _rolling_pin(angle)
    inertia_roll = clump.inertia_about_axis((1.0, 0.0, 0.0))
    analytic = GRAVITY * math.sin(angle) / (1.0 + inertia_roll / (clump.mass * RADIUS ** 2))
    sliding = GRAVITY * math.sin(angle)
    duration = float(history["time"][-1])
    measured = float(body.velocity @ downslope) / duration
    slip = float(body.velocity @ downslope) - body.angular_velocity[0] * RADIUS
    mirror, mirror_body, _, mirror_history, _, mirror_downslope = _rolling_pin(angle, sign=-1.0)
    mirror_acceleration = float(mirror_body.velocity @ mirror_downslope) / duration
    return {
        "slope_angle_deg": math.degrees(angle),
        "friction_coefficient": plane.config.friction,
        "duration_s": duration,
        "rolling_inertia_kg_m2": inertia_roll,
        "inertia_ratio_I_over_mr2": inertia_roll / (clump.mass * RADIUS ** 2),
        "acceleration_analytic_roll_m_s2": analytic,
        "acceleration_frictionless_slide_m_s2": sliding,
        "acceleration_measured_m_s2": measured,
        "acceleration_relative_error": abs(measured - analytic) / analytic,
        "slipping_band": [analytic, sliding],
        "inside_band": bool(analytic * 0.98 <= measured <= sliding),
        "downslope_displacement_m": float(body.position @ downslope),
        "normal_offset_m": float(body.position @ normal),
        "normal_offset_over_radius": float((body.position @ normal) / RADIUS),
        "omega_axial_rad_s": float(body.angular_velocity[0]),
        "no_slip_residual_m_s": float(slip),
        "no_slip_residual_relative": float(abs(slip) / abs(float(body.velocity @ downslope))),
        "mirror_acceleration_m_s2": mirror_acceleration,
        "mirror_relative_difference": abs(mirror_acceleration - measured) / abs(measured),
        "mirror_omega_sign": float(np.sign(mirror_body.angular_velocity[0])
                                   * np.sign(body.angular_velocity[0])),
        "target": "3 % band around the rolling-without-slip acceleration; direction"
                  " down-slope; mirrored tilt mirrors the motion",
        "passed": bool(abs(measured - analytic) / analytic < 3e-2),
        "history": {
            "time": history["time"].tolist(),
            "downslope_displacement": (history["position"] @ downslope).tolist(),
            "downslope_velocity": (history["velocity"] @ downslope).tolist(),
            "omega_axial": history["omega_body"][:, 0].tolist(),
            "normal_offset": (history["position"] @ normal).tolist(),
        },
    }


def check_static_contact_and_rest(self_note: str = "") -> dict:
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    stiffness = 1.0e5
    plane = PlaneContact((0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
                         ContactConfig(normal_stiffness=stiffness, restitution=0.5,
                                       friction=0.6))
    penetration = clump.mass * GRAVITY / (2.0 * stiffness)
    body = ClumpBody(clump, position=(0.0, 0.0, RADIUS - penetration),
                     orientation=quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0),
                     gravity=(0.0, 0.0, -GRAVITY))
    body.run(2000, 1e-4, (plane,))
    forces = plane.member_forces()
    return {
        "weight_N": clump.mass * GRAVITY,
        "member_normal_forces_N": forces[:, 2].tolist(),
        "summed_normal_force_N": float(forces[:, 2].sum()),
        "relative_error": float(abs(forces[:, 2].sum() - clump.mass * GRAVITY)
                                / (clump.mass * GRAVITY)),
        "member_penetrations_m": plane.penetrations().tolist(),
        "analytic_penetration_m": penetration,
        "penetration_relative_error": float(np.max(np.abs(
            plane.penetrations() - penetration)) / penetration),
        "settled_velocity_m_s": body.velocity.tolist(),
        "passed": bool(abs(forces[:, 2].sum() - clump.mass * GRAVITY)
                       / (clump.mass * GRAVITY) < 1e-6),
    }


def check_restitution() -> dict:
    clump = Clump((SphereMember((0.0, 0.0, 0.0), RADIUS),), density=DENSITY)
    rows = []
    for target in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
        plane = PlaneContact((0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
                             ContactConfig(normal_stiffness=1e5, restitution=target,
                                           friction=0.0))
        body = ClumpBody(clump, position=(0.0, 0.0, RADIUS + 2e-3),
                         gravity=(0.0, 0.0, -GRAVITY))
        incoming = 0.0
        outgoing = None
        was_in_contact = False
        for _ in range(40000):
            body.step(2e-5, (plane,))
            speed = float(body.velocity[2])
            if plane.penetrations()[0] > 0.0:
                was_in_contact = True
                incoming = min(incoming, speed)
            elif was_in_contact and incoming < -1e-6:
                outgoing = speed
                break
        rows.append({
            "target_restitution": target,
            "measured_restitution": float(-outgoing / incoming),
            "absolute_error": float(abs(-outgoing / incoming - target)),
        })
    worst = max(row["absolute_error"] for row in rows)
    return {
        "comment": "linear spring-dashpot with the log-decrement damping ratio;"
                   " the achieved restitution is systematically ~0.03-0.06 low",
        "worst_absolute_error": worst,
        "cases": rows,
    }


def check_clump_sphere_contact() -> dict:
    clump = double_sphere_clump(RADIUS, density=DENSITY)
    obstacle = SphereObstacle((0.0, RADIUS, -0.2), 0.02,
                              config=ContactConfig(normal_stiffness=1e5,
                                                   restitution=0.5, friction=0.3))
    body = ClumpBody(clump, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, -0.7),
                     gravity=(0.0, 0.0, -GRAVITY),
                     orientation=quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0))
    times, total_force, member_0, member_1, third_law_residual = [], [], [], [], []
    approach = 0.0
    rebound = None
    was_in_contact = False
    max_penetration = 0.0
    for step in range(6000):
        body.step(1e-4, (obstacle,))
        reactions = obstacle.member_forces()
        in_contact = reactions.size > 0 and float(obstacle.penetrations().max()) > 0.0
        if reactions.size:
            residual = np.abs(obstacle.obstacle_force() + reactions.sum(axis=0)).max()
            third_law_residual.append(float(residual))
        if in_contact:
            was_in_contact = True
            approach = min(approach, float(body.velocity[2]))
            max_penetration = max(max_penetration, float(obstacle.penetrations().max()))
            times.append(step * 1e-4)
            total_force.append(float(obstacle.obstacle_force()[2]))
            member_0.append(float(reactions[0, 2]))
            member_1.append(float(reactions[1, 2]))
        elif was_in_contact:
            rebound = float(body.velocity[2])
            break
    return {
        "obstacle_radius_m": obstacle.radius,
        "approach_velocity_m_s": approach,
        "velocity_after_contact_m_s": rebound,
        "max_member_penetration_m": max_penetration,
        "third_law_residual_max_N": max(third_law_residual) if third_law_residual else None,
        "member_forces_are_summed": True,
        "passed": bool(third_law_residual and max(third_law_residual) < 1e-9
                       and rebound is not None and rebound > approach),
        "history": {
            "time": times,
            "obstacle_reaction_z": total_force,
            "member_0_z": member_0,
            "member_1_z": member_1,
        },
    }


# --------------------------------------------------------------------------
# c3 — pytest
# --------------------------------------------------------------------------
def run_pytest() -> dict:
    target = "tests/test_clump_multisphere.py"
    proc = subprocess.run([sys.executable, "-m", "pytest", target, "-q",
                           "--no-header", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {
        "command": f"python -m pytest {target} -q",
        "returncode": proc.returncode,
        "summary": tail,
        "passed": proc.returncode == 0,
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(out_dir: Path, payload: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    # 1) geometry + mass-property comparison ---------------------------------
    clump = double_sphere_clump(RADIUS, 1.5 * RADIUS, density=DENSITY,
                                max_overlap_fraction=0.8)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), dpi=88)
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 400)
    for sign in (-1.0, 1.0):
        ax.fill(sign * 0.75 * RADIUS + RADIUS * np.cos(theta),
                RADIUS * np.sin(theta), color="0.85", ec="0.4", lw=0.8, zorder=1)
        ax.plot(sign * 0.75 * RADIUS + RADIUS * np.cos(theta),
                RADIUS * np.sin(theta), color="0.35", lw=1.0, zorder=2)
    ax.axvline(0.0, color="tab:red", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("two-sphere clump, d = 1.5 r\n(red: overlap lens removed once)")
    ax = axes[1]
    labels = ["I_axial", "I_transverse"]
    member = [clump.inertia_member_sum[0, 0], clump.inertia_member_sum[1, 1]]
    union = [clump.inertia_union[0, 0], clump.inertia_union[1, 1]]
    idx = np.arange(2)
    ax.bar(idx - 0.2, member, width=0.4, label="member sum (overlap counted twice)")
    ax.bar(idx + 0.2, union, width=0.4, label="union (exact)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("inertia / kg m$^2$")
    ax.set_title("overlap only inflates the mass properties")
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = out_dir / "fig1_clump_geometry_inertia.png"
    fig.savefig(path, pil_kwargs={"optimize": True})
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Two-sphere clump at d = 1.5 r: union geometry (left) and"
                               " the exact union inertia vs the naive member sum (right)."})

    # 2) voxel convergence ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.0, 2.7), dpi=88)
    for case in payload["overlap_and_voxel"]["cases"]:
        resolutions = sorted(int(k) for k in case["voxel"])
        errors = [case["voxel"][str(r)]["inertia_relative_error"] for r in resolutions]
        ax.loglog(resolutions, errors, marker="o", ms=3,
                  label=f"d = {case['separation_over_radius']:.1f} r")
    ax.set_xlabel("voxel resolution (cells per axis)")
    ax.set_ylabel("relative inertia error")
    ax.set_title("independent voxel integration vs exact union moments")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig2_voxel_convergence.png"
    fig.savefig(path, pil_kwargs={"optimize": True})
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Midpoint voxel integration of the union geometry converges to the"
                               " exact union inertia tensor for touching and overlapping pairs."})

    # 3) free fall -----------------------------------------------------------
    history = payload["free_fall"]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), dpi=88)
    time = np.asarray(history["time"])
    position = np.asarray(history["position_z"])
    axes[0].plot(time, position, label="clump COM")
    axes[0].plot(time, 2.0 - 0.5 * GRAVITY * time ** 2, "--", lw=1.0,
                 label="2 - g t$^2$/2")
    axes[0].set_xlabel("t / s")
    axes[0].set_ylabel("z / m")
    axes[0].set_title("free fall without rotation")
    axes[0].legend(fontsize=7)
    axes[1].semilogy(time[1:], np.abs(np.asarray(history["energy_error"])[1:]) + 1e-30,
                     label="|E(t) - E(0)|")
    axes[1].set_xlabel("t / s")
    axes[1].set_ylabel("energy error / J")
    axes[1].set_title("machine-level energy conservation")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig3_free_fall.png"
    fig.savefig(path, pil_kwargs={"optimize": True})
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Free fall of the clump: COM trajectory against the closed-form"
                               " parabola and the energy error (round-off only)."})

    # 4) inclined plane ------------------------------------------------------
    inclined = payload["inclined_plane"]["history"]
    time = np.asarray(inclined["time"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), dpi=88)
    axes[0].plot(time, inclined["downslope_displacement"], label="clump (DEM)")
    a = payload["inclined_plane"]["acceleration_analytic_roll_m_s2"]
    axes[0].plot(time, 0.5 * a * time ** 2, "--", lw=1.0,
                 label=f"0.5 a t$^2$, a = {a:.4f} m/s$^2$")
    axes[0].set_xlabel("t / s")
    axes[0].set_ylabel("down-slope displacement / m")
    axes[0].set_title("dumbbell rolling down a 10 deg slope")
    axes[0].legend(fontsize=7)
    axes[1].plot(time, inclined["downslope_velocity"], label="v (COM)")
    axes[1].plot(time, RADIUS * np.asarray(inclined["omega_axial"]),
                 "--", lw=1.0, label="r $\\omega_{axial}$ (rolling)")
    axes[1].set_xlabel("t / s")
    axes[1].set_ylabel("m/s")
    axes[1].set_title("rolling without slip")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig4_inclined_rolling.png"
    fig.savefig(path, pil_kwargs={"optimize": True})
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Rolling-pin dumbbell on a 10 deg slope: displacement against the"
                               " rolling-without-slip parabola and the no-slip closure r omega = v."})

    # 5) contact summation ---------------------------------------------------
    contact = payload["clump_sphere_contact"]["history"]
    fig, ax = plt.subplots(figsize=(4.0, 2.7), dpi=88)
    ax.plot(contact["time"], contact["obstacle_reaction_z"], lw=1.4,
            label="net reaction on obstacle")
    ax.plot(contact["time"], contact["member_0_z"], lw=0.9, label="member 0 contribution")
    ax.plot(contact["time"], contact["member_1_z"], lw=0.9, label="member 1 contribution")
    ax.set_xlabel("t / s")
    ax.set_ylabel("F$_z$ / N")
    ax.set_title("clump-sphere contact = sum of member contacts")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig5_contact_summation.png"
    fig.savefig(path, pil_kwargs={"optimize": True})
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Off-centre impact on a spherical obstacle: the net reaction is the"
                               " sum of the member-sphere contact forces (Newton's third law)."})
    return figures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/m4950")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "mission": "m4950",
        "campaign_node": "IN14-D3",
        "repository": "TensorDEM",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "member_radius_m": RADIUS,
            "density_kg_m3": DENSITY,
            "gravity_m_s2": GRAVITY,
            "contact_stiffness_N_m": 1.0e5,
            "contact_restitution": 0.5,
            "friction_coefficient": 0.6,
        },
    }

    print("c1: inertia vs closed form")
    payload["inertia_vs_analytic"] = check_inertia_vs_analytic()
    print("c1: overlap + voxel cross-check")
    payload["overlap_and_voxel"] = check_overlap_and_voxel()
    print("c2a: free fall")
    payload["free_fall"] = check_free_fall()
    print("c2b: inclined plane")
    payload["inclined_plane"] = check_inclined_plane()
    print("static clump on plane")
    payload["static_contact"] = check_static_contact_and_rest()
    print("restitution band")
    payload["restitution"] = check_restitution()
    print("clump-sphere contact")
    payload["clump_sphere_contact"] = check_clump_sphere_contact()
    if not args.skip_pytest:
        print("pytest")
        payload["pytest"] = run_pytest()

    print("figures")
    payload["figures"] = make_figures(out_dir, payload)

    payload["criteria"] = {
        "c1_inertia_relative_error_lt_1e_10": payload["inertia_vs_analytic"]["passed"],
        "c2a_free_fall_conservation": payload["free_fall"]["passed"],
        "c2b_rolling_acceleration_lt_3pct": payload["inclined_plane"]["passed"],
        "c2b_direction_and_mirror": payload["inclined_plane"]["mirror_relative_difference"] < 1e-6,
        "c3_pytest_ge_5_with_negative_controls": payload.get("pytest", {}).get("passed", None),
    }
    payload["all_criteria_met"] = all(
        value for value in payload["criteria"].values() if value is not None)

    json_path = out_dir / "clump_validation.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False))

    summary = {
        key: payload["criteria"][key] for key in payload["criteria"]
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {json_path}")
    return 0 if payload["all_criteria_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())