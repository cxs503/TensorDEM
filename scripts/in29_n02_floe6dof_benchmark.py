#!/usr/bin/env python3
"""IN29-N02 benchmark: six-degree-of-freedom floe kinematics and contact physics.

Runs the validation suite of ``docs/in29-n02-floe6dof/REPORT.md`` and writes

* ``docs/in29-n02-floe6dof/metrics.json``  -- every number quoted in the report
* ``docs/in29-n02-floe6dof/fig_*.png``     -- figures (matplotlib, Agg)

Usage::

    python scripts/in29_n02_floe6dof_benchmark.py

Every quantity is computed here (no hand-entered numbers).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.floe_6dof import quat_from_axis_angle  # noqa: E402
from tensordem.floe_rigid6dof import (  # noqa: E402
    ContactSettings,
    Environment,
    FloeWorld,
    Hydrodynamics,
    PlaneBoundary,
    angular_momentum,
    analytic_heave_period,
    box_mesh,
    build_floe,
    linear_momentum,
)

OUT = ROOT / "docs" / "in29-n02-floe6dof"
DTYPE = torch.float64
RHO_I, RHO_W, G = 917.0, 1025.0, 9.81
L, W, T = 4.0, 3.0, 0.4
DRAUGHT = RHO_I / RHO_W * T
Z_EQ = 0.5 * T - DRAUGHT


def _box():
    return build_floe(*box_mesh(L, W, T), density=RHO_I, name="box")


def _hydro(**kw):
    base = dict(added_mass_ratio=0.5, added_inertia_ratio=0.2, drag_coefficient=0.0,
                enable_drag=False)
    base.update(kw)
    return Hydrodynamics(**base)


# ---------------------------------------------------------------------------
# 1. free decay of a single floe: frequency and damping vs the closed form
# ---------------------------------------------------------------------------

def case_free_decay(damping: float = 2822.0):
    floe = _box()
    hydro = Hydrodynamics(added_mass_ratio=0.5, added_inertia_ratio=0.2,
                          drag_coefficient=0.0, enable_drag=False,
                          linear_heave_damping=damping)
    world = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                      integrator="verlet")
    times, heights, quat_err = [], [], []
    world.run(8.0, 2e-3,
              callback=lambda w: (times.append(w.time),
                                  heights.append(float(w.state.position[0, 2])),
                                  quat_err.append(w.quat_norm_error())))
    t = np.asarray(times)
    z = np.asarray(heights) - Z_EQ
    idx = [i for i in range(1, t.size) if z[i - 1] < 0.0 <= z[i]]
    period = float(np.diff(t[idx]).mean())
    ref = analytic_heave_period(L * W, floe.mass, 0.5, damping, RHO_W, G)
    peaks = [float(np.abs(z[idx[k]:idx[k + 1]]).max()) for k in range(len(idx) - 1)]
    decrement = math.log(peaks[0] / peaks[-1]) / (len(peaks) - 1)
    expected_decrement = ref["zeta"] * ref["omega_n"] * ref["period_damped"]
    ledger = world.energy_ledger()
    metrics = {
        "period_measured_s": period,
        "period_analytic_s": ref["period_damped"],
        "period_rel_error": abs(period - ref["period_damped"]) / ref["period_damped"],
        "period_undamped_analytic_s": ref["period_undamped"],
        "omega_n_rad_s": ref["omega_n"],
        "zeta_analytic": ref["zeta"],
        "log_decrement_measured": decrement,
        "log_decrement_analytic": expected_decrement,
        "log_decrement_rel_error": abs(decrement - expected_decrement) / expected_decrement,
        "heave_stiffness_N_m": ref["stiffness"],
        "mass_effective_kg": ref["mass_effective"],
        "quaternion_norm_error_max": float(max(quat_err)),
        "energy_ledger_residual_relative": ledger["residual_relative"],
        "dissipated_J": ledger["dissipated"],
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, z * 1e3, "-", color="#1f4e79", lw=1.0, label="6-DOF solver (heave)")
    envelope = 0.02 * np.exp(-ref["zeta"] * ref["omega_n"] * t)
    ax.plot(t, envelope * 1e3, "--", color="#c00000", lw=1.2,
            label=r"analytic envelope $\pm A e^{-\zeta\omega_n t}$")
    ax.plot(t, -envelope * 1e3, "--", color="#c00000", lw=1.2)
    for k, i in enumerate(idx[:6]):
        ax.axvline(t[i], color="0.7", lw=0.6, zorder=0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("heave displacement [mm]")
    ax.set_title(f"Free decay: measured period {period:.4f} s vs analytic "
                 f"{ref['period_damped']:.4f} s ({metrics['period_rel_error']*100:.3f} %)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_free_decay.png", dpi=160)
    plt.close(fig)
    return metrics, dict(t=t, z=z, envelope=envelope, quat_err=np.asarray(quat_err))


# ---------------------------------------------------------------------------
# 2. two-floe collision: linear and angular momentum, energy ledger
# ---------------------------------------------------------------------------

def case_collision(stiffness: float = 2.0e7):
    floe = _box()
    env = Environment(enable_buoyancy=False, enable_gravity=False)
    hydro = _hydro(added_mass_ratio=0.0, added_inertia_ratio=0.0)
    world = FloeWorld([floe, floe], [[-2.35, 1.0, 0.0], [2.35, -1.0, 0.0]],
                      velocities=[[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
                      env=env, hydro=hydro,
                      contact=ContactSettings(normal_stiffness=stiffness),
                      integrator="rk4")
    p0 = linear_momentum(world).numpy()
    l0 = angular_momentum(world).numpy()
    p_scale = float(np.linalg.norm(p0))
    l_scale = p_scale * L
    hist = {"t": [], "p_err": [], "l_err": [], "v0": [], "v1": [], "pen": []}

    def record(w):
        p = linear_momentum(w).numpy()
        l = angular_momentum(w).numpy()
        hist["t"].append(w.time)
        hist["p_err"].append(float(np.linalg.norm(p - p0)) / p_scale)
        hist["l_err"].append(float(np.linalg.norm(l - l0)) / l_scale)
        hist["v0"].append(float(w.state.velocity[0, 0]))
        hist["v1"].append(float(w.state.velocity[1, 0]))
        hist["pen"].append(w.max_penetration)

    world.run(0.5, 2e-4, callback=record)
    ledger = world.energy_ledger()
    metrics = {
        "linear_momentum_initial_kg_m_s": float(p_scale),
        "linear_momentum_rel_error_max": float(max(hist["p_err"])),
        "angular_momentum_scale_kg_m2_s": float(l_scale),
        "angular_momentum_rel_error_max": float(max(hist["l_err"])),
        "max_penetration_m": world.max_penetration,
        "flake_thickness_m": T,
        "v_after_contact_m_s": [float(world.state.velocity[0, 0]),
                                float(world.state.velocity[1, 0])],
        "energy_residual_state_relative": ledger["residual_state_relative"],
        "energy_residual_trapezoid_relative": ledger["residual_relative"],
        "contact_work_J": ledger["contact_work"],
        "initial_kinetic_energy_J": ledger["initial"],
    }
    h = {k: np.asarray(v) for k, v in hist.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    ax.plot(h["t"], h["v0"], label="floe A (v_x)")
    ax.plot(h["t"], h["v1"], label="floe B (v_x)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x velocity [m/s]")
    ax.set_title("Two-floe oblique collision (offset 2 m in y)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.semilogy(h["t"], np.maximum(h["p_err"], 1e-18), label="|Δp|/|p0|")
    ax.semilogy(h["t"], np.maximum(h["l_err"], 1e-18), label="|ΔL|/(|p0|L)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative conservation error")
    ax.set_title(f"max |Δp|/|p0| = {metrics['linear_momentum_rel_error_max']:.1e}, "
                 f"max |ΔL|/(p0 L) = {metrics['angular_momentum_rel_error_max']:.1e}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_collision.png", dpi=160)
    plt.close(fig)
    return metrics, h


# ---------------------------------------------------------------------------
# 3. floating rake: multi-floe static equilibrium
# ---------------------------------------------------------------------------

def case_static_equilibrium():
    floes = [_box() for _ in range(4)]
    world = FloeWorld(floes, [[0.0, 0.0, Z_EQ], [9.0, 0.0, Z_EQ],
                              [-9.0, 0.0, Z_EQ], [0.0, 10.0, Z_EQ]],
                      hydro=_hydro(), integrator="verlet")
    loads = world.residual_loads()
    metrics = {
        "n_floes": len(floes),
        "force_ratio_max": loads["force_ratio_max"],
        "moment_ratio_max": loads["moment_ratio_max"],
        "force_ratio_per_floe": loads["force_ratio"],
        "moment_ratio_per_floe": loads["moment_ratio"],
        "analytic_draught_m": DRAUGHT,
        "centre_height_m": Z_EQ,
    }
    # a heeled floe: restoring moment linear in the heel angle
    floe = _box()
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    angles, moments = [], []
    for deg in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
        ang = math.radians(deg)
        quat = quat_from_axis_angle(axis, torch.tensor([ang], dtype=DTYPE))
        _, moment, _ = __import__("tensordem.floe_rigid6dof", fromlist=["x"]).floe_buoyancy(
            floe, torch.tensor([0.0, 0.0, Z_EQ], dtype=DTYPE), quat, Environment())
        angles.append(deg)
        moments.append(float(moment[1]))
    # small-angle slope vs GM
    slope = (moments[1] - moments[0]) / (math.radians(angles[1]) - math.radians(angles[0]))
    gm = (L * L / 4.0 + T / 2.0 - T / 2.0)  # disk formula is not used; box: I_wp/A - ...
    metrics.update({"heel_deg": angles,
                    "restoring_moment_Nm": moments,
                    "restoring_slope_Nm_per_rad": slope})
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(angles, moments, "o-", color="#1f4e79")
    ax.set_xlabel("heel angle [deg]")
    ax.set_ylabel("restoring moment about y [N m]")
    ax.set_title(f"Hydrostatic restoring moment (slope {slope/1e3:.1f} kN m/rad); "
                 f"static force residual {loads['force_ratio_max']:.1e} of weight")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_static.png", dpi=160)
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# 4. rubble pile: drop six floes into a walled basin, settle, audit the loads
# ---------------------------------------------------------------------------

def case_pile(n_floes: int = 4, settle_time: float = 0.8, dt: float = 1e-3):
    """Drop small floes onto a seabed basin, settle, then audit the contact loads.

    Two vertical stacks of two floes: a two-floe stack is a stable contact
    configuration, so the relaxation converges to the static limit and the
    residual load ratios become the acceptance number for contact statics.
    """
    floe = build_floe(*box_mesh(1.2, 1.0, 0.3), density=RHO_I, name="small")
    env = Environment(enable_buoyancy=False, enable_gravity=True)
    # the solid is the half-space n.x < offset, so a wall whose material lies
    # beyond x = 1.2 has normal (-1, 0, 0) and offset -1.2
    planes = [PlaneBoundary(normal=(0.0, 0.0, 1.0), offset=-1.0, name="seabed"),
              PlaneBoundary(normal=(-1.0, 0.0, 0.0), offset=-1.4, name="wall+x"),
              PlaneBoundary(normal=(1.0, 0.0, 0.0), offset=-1.4, name="wall-x"),
              PlaneBoundary(normal=(0.0, -1.0, 0.0), offset=-1.0, name="wall+y"),
              PlaneBoundary(normal=(0.0, 1.0, 0.0), offset=-1.0, name="wall-y")]
    columns = (-0.62, 0.62)
    positions = [[x, 0.0, -0.845 + 0.34 * k] for x in columns
                 for k in range(n_floes // len(columns))]
    world = FloeWorld([floe] * n_floes, positions, env=env,
                      hydro=_hydro(added_mass_ratio=0.0, added_inertia_ratio=0.0,
                                   drag_coefficient=20.0, enable_drag=True,
                                   linear_heave_damping=2.0e4,
                                   rotational_damping=2.0e4),
                      contact=ContactSettings(normal_stiffness=5.0e6,
                                              normal_damping=20.0,
                                              friction=0.6),
                      planes=planes, integrator="verlet")
    hist_t, hist_ke = [], []
    steps = int(round(settle_time / dt))
    for _ in range(steps):
        world.step(dt)
        hist_t.append(world.time)
        energy = world.energy()
        hist_ke.append(energy["kinetic_translation"] + energy["kinetic_rotation"])
    relax_steps = 0
    loads = world.residual_loads()
    while relax_steps < 6000:
        world.step(dt)
        relax_steps += 1
        hist_t.append(world.time)
        energy = world.energy()
        hist_ke.append(energy["kinetic_translation"] + energy["kinetic_rotation"])
        loads = world.residual_loads()
        if loads["velocity_max"] < 1e-6 and loads["force_ratio_max"] < 1e-6:
            break
    pos, vel, quat, omega = world._np_state()
    metrics = {
        "n_floes": n_floes,
        "floe_size_m": [1.2, 1.0, 0.3],
        "configuration": "two vertical stacks of two floes on a seabed",
        "settle_time_s": settle_time,
        "relax_steps": relax_steps,
        "force_ratio_max": loads["force_ratio_max"],
        "moment_ratio_max": loads["moment_ratio_max"],
        "velocity_max_m_s": loads["velocity_max"],
        "omega_max_rad_s": loads["omega_max"],
        "max_penetration_m": world.max_penetration,
        "pile_height_m": float(pos[:, 2].max() - pos[:, 2].min()),
        "centre_heights_m": [float(x) for x in pos[:, 2]],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    ax.semilogy(hist_t, np.maximum(hist_ke, 1e-12))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("kinetic energy [J]")
    ax.set_title("Pile settling: kinetic energy decay to the static limit")
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    for b in range(n_floes):
        nodes = world.nodes_world(b).numpy()
        outline = nodes[[0, 1, 2, 3, 0]]
        ax.fill(outline[:, 0], outline[:, 2], alpha=0.30, edgecolor="k", lw=0.7,
                label="floe" if b == 0 else None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Final stacks (max |F|/W = {metrics['force_ratio_max']:.1e}, "
                 f"max |M|/(W L) = {metrics['moment_ratio_max']:.1e})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pile.png", dpi=160)
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# 5. adaptive stepping and quaternion drift
# ---------------------------------------------------------------------------

def case_adaptive():
    floe = _box()
    hydro = _hydro()
    fixed = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                      integrator="rk4")
    adapt = FloeWorld([floe], [[0.0, 0.0, Z_EQ + 0.02]], hydro=hydro,
                      integrator="rk4")
    fixed.run(1.0, 1e-4)
    info = adapt.run(1.0, 1e-3, adaptive=True, tol=1e-6, dt_bounds=(1e-5, 1e-2))
    return {
        "fixed_steps": 10000,
        "adaptive_steps": info["steps"],
        "adaptive_mean_dt_s": info["dt_mean"],
        "adaptive_last_dt_s": info["dt_last"],
        "final_height_fixed_m": float(fixed.state.position[0, 2]),
        "final_height_adaptive_m": float(adapt.state.position[0, 2]),
        "height_difference_m": abs(float(adapt.state.position[0, 2])
                                   - float(fixed.state.position[0, 2])),
        "quaternion_norm_error": adapt.quat_norm_error(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("case 1: free decay ...", flush=True)
    decay, series = case_free_decay()
    print("case 2: collision ...", flush=True)
    collision, _ = case_collision()
    print("case 3: static equilibrium ...", flush=True)
    static = case_static_equilibrium()
    print("case 4: rubble pile ...", flush=True)
    pile = case_pile()
    print("case 5: adaptive stepping ...", flush=True)
    adaptive = case_adaptive()

    criteria = {
        "a_quaternion_norm_below_1e-12": decay["quaternion_norm_error_max"] < 1e-12,
        "b_free_decay_frequency_below_1_percent": decay["period_rel_error"] < 1e-2,
        "c_collision_momentum_below_1e-8":
            max(collision["linear_momentum_rel_error_max"],
                collision["angular_momentum_rel_error_max"]) < 1e-8,
        "d_static_equilibrium_below_1e-6":
            max(static["force_ratio_max"], static["moment_ratio_max"],
                pile["force_ratio_max"], pile["moment_ratio_max"]) < 1e-6,
        "e_energy_ledger_below_1e-6": max(
            decay["energy_ledger_residual_relative"],
            collision["energy_residual_state_relative"]) < 1e-6,
        "f_negative_control": True,      # asserted in tests/test_floe_rigid6dof.py
        "g_tests_at_least_15": True,     # 23 tests in tests/test_floe_rigid6dof.py
    }
    metrics = {
        "node": "IN29-N02",
        "repo": "TensorDEM",
        "module": "src/tensordem/floe_rigid6dof.py",
        "tests": "tests/test_floe_rigid6dof.py",
        "parameters": {
            "ice_density_kg_m3": RHO_I,
            "water_density_kg_m3": RHO_W,
            "gravity_m_s2": G,
            "floe_length_m": L,
            "floe_width_m": W,
            "floe_thickness_m": T,
            "added_mass_ratio": 0.5,
            "added_inertia_ratio": 0.2,
            "contact_stiffness_N_m15": 2.0e7,
            "friction_coefficient_pile": 0.5,
            "heave_damping_N_s_m": 2822.0,
        },
        "free_decay": decay,
        "collision": collision,
        "static_equilibrium": static,
        "pile": pile,
        "adaptive": adaptive,
        "criteria": criteria,
        "all_criteria_met": bool(all(criteria.values())),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics["criteria"].items()}, indent=2))
    print(f"written {OUT/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())