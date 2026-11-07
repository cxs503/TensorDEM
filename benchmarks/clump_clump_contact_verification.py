"""Clump-clump contact verification campaign (P16-N3): criteria, numbers, figures.

Runs the six acceptance criteria of the campaign node on fixed, documented
configurations, writes ``artifacts/clump_clump_contact/criteria.json`` and the
figures used by ``docs/clump_clump_contact.md``.

Command
-------
    python benchmarks/clump_clump_contact_verification.py [--out DIR] [--steps N]

``--steps`` defaults to 100000 (the C6 stability run); lower it only for a smoke
run, and the JSON records the value actually used.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensordem import clump_clump as cc                       # noqa: E402
from tensordem.clump import Clump, SphereMember                # noqa: E402
from tensordem.clump_dynamics import (                         # noqa: E402
    ClumpBody,
    ContactConfig as WallContactConfig,
    PlaneContact,
)

RADIUS = 0.05
DENSITY = 900.0
YOUNG = 1.0e8
POISSON = 0.3
GRAVITY = 9.81


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sphere_clump(radius: float = RADIUS) -> Clump:
    return Clump([SphereMember((0.0, 0.0, 0.0), radius)], density=DENSITY)


def dumbbell(separation: float = 0.1, radius: float = RADIUS) -> Clump:
    return Clump([SphereMember((-0.5 * separation, 0.0, 0.0), radius),
                  SphereMember((0.5 * separation, 0.0, 0.0), radius)],
                 density=DENSITY, max_overlap_fraction=2.0)


def compact(separation: float = 0.02, radius: float = RADIUS) -> Clump:
    return Clump([SphereMember((-0.5 * separation, 0.0, 0.0), radius),
                  SphereMember((0.5 * separation, 0.0, 0.0), radius)],
                 density=DENSITY, max_overlap_fraction=2.0)


def config(**kwargs) -> cc.ClumpClumpConfig:
    base = dict(young_modulus=YOUNG, poisson_ratio=POISSON)
    base.update(kwargs)
    return cc.ClumpClumpConfig(**base)


def hertz_reference(penetration: float, radius_eff: float, e_star: float) -> float:
    return (4.0 / 3.0) * e_star * math.sqrt(radius_eff) * penetration ** 1.5


def reference_moduli() -> tuple[float, float]:
    shear = YOUNG / (2.0 * (1.0 + POISSON))
    e_star = YOUNG / (2.0 * (1.0 - POISSON ** 2))
    g_star = 1.0 / (2.0 * (2.0 - POISSON) / shear)
    return e_star, g_star


def restitution_scan(targets, v0=0.1, dt=1.0e-5):
    rows = []
    for target in targets:
        body_a = ClumpBody(sphere_clump(), position=(-(RADIUS + 2e-4), 0, 0),
                           velocity=(v0, 0, 0), gravity=(0.0, 0.0, 0.0))
        body_b = ClumpBody(sphere_clump(), position=((RADIUS + 2e-4), 0, 0),
                           velocity=(-v0, 0, 0), gravity=(0.0, 0.0, 0.0))
        contact = cc.ClumpClumpContact(config(restitution=target, friction=0.0))
        touched = False
        measured = float("nan")
        peak = 0.0
        for _ in range(6000):
            contact.step(body_a, body_b, dt)
            active = (contact.last.normal_forces.size
                      and float(contact.last.normal_forces.max()) > 0.0)
            if active:
                touched = True
                peak = max(peak, float(contact.last.normal_forces.max()))
            elif touched:
                measured = (body_b.velocity[0] - body_a.velocity[0]) / (2.0 * v0)
                break
        rows.append({"target": target, "measured": measured,
                     "deviation": measured - target, "peak_force": peak})
    return rows


# --------------------------------------------------------------------------
# criteria
# --------------------------------------------------------------------------
def criterion_c1() -> dict:
    """Single-member clump versus sphere-sphere closed form."""
    cfg = config(restitution=0.7)
    e_star, _ = reference_moduli()
    worst = 0.0
    rows = []
    for penetration in (1e-6, 1e-5, 1e-4, 1e-3):
        for approach in (0.0, 0.01, 0.1):
            body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
            body_b = ClumpBody(sphere_clump(),
                               position=(2.0 * RADIUS - penetration, 0.0, 0.0),
                               velocity=(approach, 0.0, 0.0),
                               gravity=(0.0, 0.0, 0.0))
            response = cc.ClumpClumpContact(cfg).forces(body_a, body_b,
                                                        commit=False)
            delta = float(response.joints[0].penetration)
            mass_eff = 1.0 / (1.0 / body_a.clump.mass + 1.0 / body_b.clump.mass)
            ln_e = math.log(0.7)
            zeta = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
            k_n = 2.0 * e_star * math.sqrt(0.5 * RADIUS * delta)
            c_n = 2.0 * zeta * math.sqrt(mass_eff * k_n)
            elastic = hertz_reference(delta, 0.5 * RADIUS, e_star)
            reference = max(elastic - c_n * approach, 0.0)
            measured = float(response.normal_forces[0])
            # at the smallest penetrations a fast approach clips the contact, so
            # normalise by the elastic scale rather than by the clipped value
            relative = abs(measured - reference) / max(reference, elastic)
            worst = max(worst, relative)
            rows.append({"penetration": penetration, "approach": approach,
                         "measured": measured, "reference": reference,
                         "relative_error": relative})
    # integrator consistency in free flight
    clump = dumbbell()
    solo = ClumpBody(clump, position=(0.0, 0.0, 3.0), velocity=(0.4, -0.2, 0.1),
                     angular_velocity=(0.3, -0.5, 0.2))
    body_a = ClumpBody(clump, position=(0.0, 0.0, 3.0), velocity=(0.4, -0.2, 0.1),
                       angular_velocity=(0.3, -0.5, 0.2))
    body_b = ClumpBody(clump, position=(0.0, 0.0, 6.0), velocity=(0.0, 0.1, 0.0))
    contact = cc.ClumpClumpContact(config())
    for _ in range(50):
        solo.step(1.0e-4)
        contact.step(body_a, body_b, 1.0e-4)
    integrator = float(np.abs(body_a.position - solo.position).max())
    return {
        "id": "C1",
        "criterion": "single-member clump == sphere-sphere, rel. error <= 1e-12",
        "worst_relative_error": worst,
        "pair_integrator_max_abs_difference_m": integrator,
        "passed": bool(worst <= 1e-12 and integrator <= 1e-15),
        "rows": rows,
    }


def criterion_c2() -> dict:
    """Two-sphere clump contact versus an independent geometric estimate."""
    e_star, _ = reference_moduli()
    # (i) parallel dumbbells: two junctions, penetration from the geometry alone
    separation, delta = 0.1, 1.1e-2
    body_a = ClumpBody(dumbbell(separation),
                       position=(0.0, 0.0, 2.0 * RADIUS - delta),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell(separation), gravity=(0.0, 0.0, 0.0))
    cfg = config(restitution=1.0)
    response = cc.ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    expected = 2.0 * hertz_reference(delta, 0.5 * RADIUS, e_star)
    two_junction = {
        "configured_penetration": delta,
        "joints": len(response.joints),
        "candidates": response.candidates,
        "measured_force": response.force_magnitude,
        "independent_estimate": expected,
        "relative_error": abs(response.force_magnitude - expected) / expected,
    }
    # (ii) deep parallel contact: four overlapping member pairs
    separation2, height = 0.06, 0.078
    body_a2 = ClumpBody(dumbbell(separation2), position=(0.0, 0.0, height),
                        gravity=(0.0, 0.0, 0.0))
    body_b2 = ClumpBody(dumbbell(separation2), gravity=(0.0, 0.0, 0.0))
    primary = 2.0 * RADIUS - height
    diagonal = 2.0 * RADIUS - math.hypot(separation2, height)
    estimate4 = (2.0 * hertz_reference(primary, 0.5 * RADIUS, e_star)
                 + 2.0 * hertz_reference(diagonal, 0.5 * RADIUS, e_star))
    response4 = cc.ClumpClumpContact(cfg).forces(body_a2, body_b2, commit=False)
    four_pair = {
        "primary_penetration": primary,
        "diagonal_penetration": diagonal,
        "candidates": response4.candidates,
        "joints": len(response4.joints),
        "measured_force": response4.force_magnitude,
        "independent_estimate": estimate4,
        "relative_error": abs(response4.force_magnitude - estimate4) / estimate4,
    }
    worst = max(two_junction["relative_error"], four_pair["relative_error"])
    return {
        "id": "C2",
        "criterion": "two-sphere clump contact within 5% of the geometric estimate",
        "worst_relative_error": worst,
        "passed": bool(worst <= 0.05),
        "two_junction": two_junction,
        "four_pair": four_pair,
    }


def criterion_c3() -> dict:
    """Duplicate-counting negative control."""
    cfg = config(restitution=1.0)
    e_star, _ = reference_moduli()
    distance, member_offset = 0.088, 0.01
    body_a = ClumpBody(compact(2.0 * member_offset), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(0.0, distance, 0.0),
                       gravity=(0.0, 0.0, 0.0))
    candidates = cc.member_pair_candidates(body_a, body_b)
    joints = cc.clump_clump_joints(body_a, body_b, cfg)
    penetration = 2.0 * RADIUS - math.hypot(member_offset, distance)
    unique = hertz_reference(penetration, 0.5 * RADIUS, e_star)
    merged = cc.ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    ratio = cc.duplicate_force_ratio(body_a, body_b, cfg)
    sweep = []
    for factor in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        merged_cfg = config(restitution=1.0, merge_factor=factor)
        joints_f = cc.clump_clump_joints(body_a, body_b, merged_cfg)
        response_f = cc.ClumpClumpContact(merged_cfg).forces(body_a, body_b,
                                                             commit=False)
        sweep.append({"merge_factor": factor, "joints": len(joints_f),
                      "force": response_f.force_magnitude,
                      "force_over_unique": response_f.force_magnitude / unique})
    return {
        "id": "C3",
        "criterion": ("merged force <= 1.05x unique-junction force; naive member "
                      "sum ~2x"),
        "candidate_pairs": len(candidates),
        "joints": len(joints),
        "unique_junction_force": unique,
        "merged_force": merged.force_magnitude,
        "merged_over_unique": merged.force_magnitude / unique,
        "unmerged_force": ratio["unmerged_force"],
        "unmerged_over_merged": ratio["ratio"],
        "passed": bool(merged.force_magnitude <= 1.05 * unique
                       and 1.9 <= ratio["ratio"] <= 2.1),
        "merge_factor_sweep": sweep,
    }


def criterion_c4() -> dict:
    """Newton's third law and net-moment closure, plus momentum in a collision."""
    cfg = config(restitution=0.9, friction=0.4)
    body_a = ClumpBody(dumbbell(), position=(-0.049, 0.004, 0.001),
                       velocity=(0.1, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell(), position=(0.049, 0.0, 0.0),
                       velocity=(-0.1, 0.0, 0.0), gravity=(0.0, 0.0, 0.0))
    forward = cc.ClumpClumpContact(cfg).forces(body_a, body_b, commit=False)
    swapped = cc.ClumpClumpContact(cfg).forces(body_b, body_a, commit=False)
    scale = float(np.linalg.norm(forward.force_a))
    swap_residual = float(np.linalg.norm(forward.force_a + swapped.force_a)) / scale
    partner_residual = float(np.abs(forward.force_a + forward.force_b).max()) / scale
    moment = (np.cross(body_a.position, forward.force_a) + forward.torque_a
              + np.cross(body_b.position, forward.force_b) + forward.torque_b)
    moment_residual = float(np.linalg.norm(moment)) / (scale * 0.1)

    # momentum of an asymmetric, frictional, spinning collision
    cfg2 = config(restitution=0.75, friction=0.3)
    a2 = ClumpBody(dumbbell(), position=(-0.09, 0.013, 0.0),
                   velocity=(0.2, 0.0, 0.0), angular_velocity=(0.0, 0.0, 2.0),
                   gravity=(0.0, 0.0, 0.0))
    b2 = ClumpBody(dumbbell(), position=(0.09, 0.0, 0.0),
                   velocity=(-0.05, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    contact = cc.ClumpClumpContact(cfg2)
    p0 = cc.total_linear_momentum(a2, b2)
    l0 = cc.total_angular_momentum(a2, b2)
    scale_p, scale_l = float(np.linalg.norm(p0)), float(np.linalg.norm(l0))
    for _ in range(4000):
        contact.step(a2, b2, 5.0e-6)
    momentum_residual = float(
        np.linalg.norm(cc.total_linear_momentum(a2, b2) - p0)) / scale_p
    angular_residual = float(
        np.linalg.norm(cc.total_angular_momentum(a2, b2) - l0)) / scale_l
    return {
        "id": "C4",
        "criterion": "third-law residual <= 1e-12 (force) / <= 1e-12 (momentum)",
        "swap_relative_residual": swap_residual,
        "partner_relative_residual": partner_residual,
        "net_moment_relative_residual": moment_residual,
        "linear_momentum_relative_residual": momentum_residual,
        "angular_momentum_relative_residual": angular_residual,
        "passed": bool(swap_residual <= 1e-12 and partner_residual <= 1e-12
                       and moment_residual <= 1e-12
                       and momentum_residual <= 1e-12
                       and angular_residual <= 1e-10),
    }


def criterion_c5() -> dict:
    """Angular momentum in an oblique frictional collision and restitution."""
    rows = restitution_scan((0.5, 0.7, 0.9, 0.95))
    worst = max(abs(row["deviation"]) for row in rows)

    cfg = config(restitution=0.8, friction=0.4)
    body_a = ClumpBody(dumbbell(), position=(-0.09, 0.013, 0.0),
                       velocity=(0.2, 0.0, 0.0), angular_velocity=(0.0, 0.0, 2.0),
                       gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(dumbbell(), position=(0.09, 0.0, 0.0),
                       velocity=(-0.05, 0.01, 0.0), gravity=(0.0, 0.0, 0.0))
    contact = cc.ClumpClumpContact(cfg)
    l0 = cc.total_angular_momentum(body_a, body_b)
    scale = float(np.linalg.norm(l0))
    log = []
    for step in range(4000):
        contact.step(body_a, body_b, 5.0e-6)
        if step % 100 == 0:
            log.append({"step": step,
                        "angular_momentum_residual":
                            float(np.linalg.norm(
                                cc.total_angular_momentum(body_a, body_b) - l0))
                            / scale,
                        "force": contact.last.force_magnitude})
    angular = max(row["angular_momentum_residual"] for row in log)
    return {
        "id": "C5",
        "criterion": ("angular momentum residual <= 1e-10; restitution deviation "
                      "<= 0.05"),
        "angular_momentum_relative_residual": angular,
        "restitution_worst_deviation": worst,
        "restitution_rows": rows,
        "passed": bool(angular <= 1e-10 and worst <= 0.05),
        "history": log,
    }


def criterion_c6(steps: int) -> dict:
    """Resting clump stack: 1e5 steps, COM drift and energy audit."""
    wall = WallContactConfig(normal_stiffness=1.0e7, restitution=0.9, friction=0.4)
    floor = PlaneContact((0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0), config=wall)
    cfg = config(restitution=0.9, friction=0.4)
    e_star, _ = reference_moduli()
    clump = dumbbell()
    weight = clump.mass * GRAVITY
    delta_plane = weight / wall.normal_stiffness
    hertz_coefficient = (4.0 / 3.0) * e_star * math.sqrt(0.5 * RADIUS)
    delta_pair = (weight / (2.0 * hertz_coefficient)) ** (2.0 / 3.0)
    z_b = RADIUS - delta_plane
    body_a = ClumpBody(clump, position=(0.0, 0.0, z_b + 2.0 * RADIUS - delta_pair))
    body_b = ClumpBody(clump, position=(0.0, 0.0, z_b))
    contact = cc.ClumpClumpContact(cfg)
    dt = 1.0e-5

    def total_energy(pair_contact: "cc.ClumpClumpContact") -> float:
        plane = 0.5 * wall.normal_stiffness * float(np.sum(floor.penetrations() ** 2))
        return (body_a.energy() + body_b.energy() + pair_contact.stored_energy()
                + plane)

    contact.forces(body_a, body_b, dt=0.0, commit=False)
    floor.forces(body_b, dt=0.0, commit=False)
    start_a, start_b = body_a.position.copy(), body_b.position.copy()
    energy_start = total_energy(contact)
    drift_log = []
    wall_start = time.time()
    for step in range(steps + 1):
        if step % max(steps // 200, 1) == 0:
            drift_log.append({
                "step": step,
                "drift": float(max(np.linalg.norm(body_a.position - start_a),
                                   np.linalg.norm(body_b.position - start_b))),
                "energy": total_energy(contact)})
        if step == steps:
            break
        contact.step(body_a, body_b, dt, extra_b=(floor,))
    elapsed = time.time() - wall_start
    drift = float(max(np.linalg.norm(body_a.position - start_a),
                      np.linalg.norm(body_b.position - start_b)))
    energy_end = total_energy(contact)

    # supplementary: start 5% deeper than equilibrium and let the stack settle,
    # so the run also exercises the dissipative path instead of a pure fixed point
    body_a = ClumpBody(clump, position=(0.0, 0.0, z_b + 2.0 * RADIUS
                                        - 1.05 * delta_pair))
    body_b = ClumpBody(clump, position=(0.0, 0.0, z_b))
    contact_s = cc.ClumpClumpContact(cfg)
    contact_s.forces(body_a, body_b, dt=0.0, commit=False)
    floor.forces(body_b, dt=0.0, commit=False)
    equilibrium_a = z_b + 2.0 * RADIUS - delta_pair
    energy_s0 = total_energy(contact_s)
    settling_energy = []
    settle_steps = min(steps, 20_000)
    for step in range(settle_steps):
        contact_s.step(body_a, body_b, dt, extra_b=(floor,))
        if step % max(settle_steps // 50, 1) == 0:
            settling_energy.append(total_energy(contact_s))
    settling = {
        "steps": settle_steps,
        "initial_offset_m": 0.05 * delta_pair,
        "final_offset_m": float(body_a.position[2] - equilibrium_a),
        "energy_start": energy_s0,
        "energy_end": total_energy(contact_s),
        "energy_monotone_decreasing": bool(
            np.all(np.diff(settling_energy) <= 1e-9 * abs(energy_s0))),
        "energy_history": settling_energy,
    }
    return {
        "id": "C6",
        "criterion": "COM drift <= 1e-9 m over the resting run; energy non-increasing",
        "steps": steps,
        "dt": dt,
        "equilibrium_plane_overlap": delta_plane,
        "equilibrium_pair_overlap": delta_pair,
        "drift": drift,
        "energy_start": energy_start,
        "energy_end": energy_end,
        "relative_energy_change": (energy_end - energy_start) / abs(energy_start),
        "joints": len(contact.last.joints),
        "wall_seconds": elapsed,
        "passed": bool(drift <= 1e-9 and energy_end <= energy_start + 1e-9),
        "log": drift_log,
        "settling": settling,
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(out: Path, c1: dict, c2: dict, c3: dict, c5: dict,
                 c6: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    # 1 -- Hertz force law versus the module
    e_star, _ = reference_moduli()
    penetrations = np.logspace(-6.5, -2.5, 40)
    analytic = [hertz_reference(d, 0.5 * RADIUS, e_star) for d in penetrations]
    measured = []
    for delta in penetrations:
        body_a = ClumpBody(sphere_clump(), gravity=(0.0, 0.0, 0.0))
        body_b = ClumpBody(sphere_clump(),
                           position=(2.0 * RADIUS - float(delta), 0.0, 0.0),
                           gravity=(0.0, 0.0, 0.0))
        response = cc.ClumpClumpContact(config(restitution=1.0)).forces(
            body_a, body_b, commit=False)
        measured.append(float(response.normal_forces[0]))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(penetrations, analytic, "-", label="Hertz analytic  (4/3)E*√R* δ^1.5")
    ax.loglog(penetrations, measured, "o", ms=3.5, label="clump-clump junction force")
    ax.set_xlabel("penetration δ [m]")
    ax.set_ylabel("normal force F_n [N]")
    ax.set_title("C1: single-member clump reproduces sphere–sphere Hertz")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out / "c1_hertz_force.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Hertz normal force versus penetration: analytic "
                               "curve and the clump-clump junction force "
                               "(max relative error "
                               f"{c1['worst_relative_error']:.2e})."})

    # 2 -- duplicate counting control
    sweep = c3["merge_factor_sweep"]
    factors = [row["merge_factor"] for row in sweep]
    ratios = [row["force_over_unique"] for row in sweep]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(np.maximum(factors, 1e-2), ratios, "o-", label="clump-clump force")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="unique junction force")
    ax.axhline(1.05, color="tab:red", ls=":", lw=1, label="1.05 acceptance band")
    ax.axhline(c3["unmerged_over_merged"], color="tab:orange", ls="-.",
               lw=1, label=f"naive member sum = {c3['unmerged_over_merged']:.3f}x")
    ax.set_xlabel("merge_factor  (merge length = merge_factor · (a_p + a_q))")
    ax.set_ylabel("force / unique-junction force")
    ax.set_title("C3: duplicate-counting negative control")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "c3_duplicate_control.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Duplicate-counting negative control: the merged "
                               "junction force stays on the unique-junction level "
                               "for merge_factor >= 0.5, while the naive member "
                               f"sum is {c3['unmerged_over_merged']:.3f}x "
                               "(two symmetric member pairs)."})

    # 3 -- restitution
    rows = c5["restitution_rows"]
    targets = [row["target"] for row in rows]
    measured_e = [row["measured"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    grid = np.linspace(0.4, 1.0, 10)
    ax.plot(grid, grid, "k--", lw=1, label="target = measured")
    ax.fill_between(grid, grid - 0.05, grid + 0.05, color="tab:green", alpha=0.15,
                    label="±0.05 band")
    ax.plot(targets, measured_e, "o", color="tab:blue", label="measured")
    ax.set_xlabel("requested restitution e")
    ax.set_ylabel("measured restitution")
    ax.set_title("C5: restitution calibration (Hertz + dashpot)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "c5_restitution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Measured coefficient of restitution versus the "
                               "requested value; worst deviation "
                               f"{c5['restitution_worst_deviation']:+.4f} "
                               "(within the ±0.05 band)."})

    # 4 -- stability (drift + energy over 1e5 steps)
    log = c6["log"]
    step_axis = np.array([row["step"] for row in log])
    drift_axis = np.array([row["drift"] for row in log])
    energy_axis = np.array([row["energy"] for row in log])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax1.semilogy(step_axis, np.maximum(drift_axis, 1e-18), "-")
    ax1.axhline(1e-9, color="tab:red", ls=":", label="1e-9 m acceptance")
    ax1.set_xlabel("step")
    ax1.set_ylabel("centre-of-mass drift [m]")
    ax1.set_title("C6: resting clump stack drift")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.plot(step_axis, (energy_axis - energy_axis[0]), "-")
    ax2.set_xlabel("step")
    ax2.set_ylabel("total energy change [J]")
    ax2.set_title("C6: energy budget (KE + gravity + elastic)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "c6_stability.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Resting clump stack over "
                               f"{c6['steps']} velocity-Verlet steps: COM drift "
                               f"{c6['drift']:.2e} m (limit 1e-9 m) and total "
                               "energy change including the contact elastic "
                               "energy."})

    # 5 -- force versus penetration: merged versus naive member sum
    e_star, _ = reference_moduli()
    distances = np.linspace(0.0995, 0.078, 60)      # B centre line distance
    merged_force, naive_force, unique_force, pair_count = [], [], [], []
    for distance in distances:
        body_a = ClumpBody(compact(0.02), gravity=(0.0, 0.0, 0.0))
        body_b = ClumpBody(sphere_clump(), position=(0.0, float(distance), 0.0),
                           gravity=(0.0, 0.0, 0.0))
        merged_force.append(cc.ClumpClumpContact(config(restitution=1.0)).forces(
            body_a, body_b, commit=False).force_magnitude)
        naive_force.append(cc.ClumpClumpContact(
            config(restitution=1.0, merging=False)).forces(
                body_a, body_b, commit=False).force_magnitude)
        pair_count.append(len(cc.member_pair_candidates(body_a, body_b)))
        penetration = 2.0 * RADIUS - math.hypot(0.01, float(distance))
        unique_force.append(hertz_reference(penetration, 0.5 * RADIUS, e_star)
                            if penetration > 0.0 else 0.0)
    penetration_axis = np.array([2.0 * RADIUS - math.hypot(0.01, d)
                                 for d in distances])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax.plot(penetration_axis * 1e3, np.array(merged_force) / 1e3, "-",
            label="merged junction (this work)")
    ax.plot(penetration_axis * 1e3, np.array(naive_force) / 1e3, "--",
            label="naive member sum")
    ax.plot(penetration_axis * 1e3, np.array(unique_force) / 1e3, ":", color="k",
            label="single-junction reference")
    ax.set_xlabel("penetration of the near member pair [mm]")
    ax.set_ylabel("normal force [kN]")
    ax.set_title("Duplicate sampling: merged vs naive member sum")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax2.plot(penetration_axis * 1e3, np.array(naive_force)
             / np.maximum(np.array(merged_force), 1e-30), "-")
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax2.axhline(2.0, color="tab:orange", ls="--", lw=1)
    ax2.set_xlabel("penetration of the near member pair [mm]")
    ax2.set_ylabel("naive sum / merged force")
    ax2.set_title("Overcount factor (2 = both member pairs active)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "duplicate_overcount_vs_penetration.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Quasi-static compression of the duplicate-counting "
                               "configuration: once the second (nearly concentric) "
                               "member pair activates, the naive member sum rises "
                               "to about twice the merged junction force while the "
                               "merged phase stays on the single-junction reference."})

    # 6 -- merge geometry schematic
    e_star, _ = reference_moduli()
    body_a = ClumpBody(compact(0.02), gravity=(0.0, 0.0, 0.0))
    body_b = ClumpBody(sphere_clump(), position=(0.0, 0.088, 0.0),
                       gravity=(0.0, 0.0, 0.0))
    candidates = cc.member_pair_candidates(body_a, body_b)
    joints = cc.clump_clump_joints(body_a, body_b, config())
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    theta = np.linspace(0.0, math.pi, 200)
    for i, centre in enumerate(body_a.member_centers()):
        ax.plot(centre[0] + RADIUS * np.cos(theta), centre[1] + RADIUS * np.sin(theta),
                "b-", lw=1.2, label="clump A members" if i == 0 else None)
        ax.plot(centre[0], centre[1], "b+")
    centre_b = body_b.member_centers()[0]
    ax.plot(centre_b[0] + RADIUS * np.cos(theta), centre_b[1] + RADIUS * np.sin(theta),
            "g-", lw=1.2, label="clump B member")
    for candidate in candidates:
        ax.plot(candidate.point[0], candidate.point[1], "r.", ms=7,
                label="candidate contact" if candidate is candidates[0] else None)
        circle = plt.Circle((candidate.point[0], candidate.point[1]),
                            candidate.patch_radius, fill=False, color="tab:red",
                            ls=":", lw=1)
        ax.add_patch(circle)
    for joint in joints:
        ax.plot(joint.point[0], joint.point[1], "k*", ms=14,
                label="merged junction")
        ax.annotate("", xy=(joint.point[0], joint.point[1] + 0.02),
                    xytext=(joint.point[0], joint.point[1]),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title("Two duplicate member pairs merge into one junction")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = out / "merge_geometry.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Geometry of the duplicate-counting control: two "
                               "nearly concentric members of clump A sample the "
                               "same junction of clump B; the dotted circles are "
                               "the Hertz patch radii and the star is the single "
                               "merged junction that carries the force."})
    return figures


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "clump_clump_contact"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--reuse-json", default=None,
                        help="regenerate only the figures from an existing "
                             "criteria.json (no integrations are re-run)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if args.reuse_json:
        payload = json.loads(Path(args.reuse_json).read_text())
        by_id = {item["id"]: item for item in payload["criteria"]}
        figures = make_figures(out, by_id["C1"], by_id["C2"], by_id["C3"],
                               by_id["C5"], by_id["C6"])
        payload["figures"] = figures
        payload.setdefault("environment", {})["figures_regenerated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (out / "criteria.json").write_text(json.dumps(payload, indent=2))
        for figure in figures:
            print(f"figure {figure['name']}: {figure['caption']}")
        print(f"artifacts: {out}")
        return 0 if payload.get("all_passed") else 1

    c1 = criterion_c1()
    c2 = criterion_c2()
    c3 = criterion_c3()
    c4 = criterion_c4()
    c5 = criterion_c5()
    c6 = criterion_c6(args.steps)
    criteria = [c1, c2, c3, c4, c5, c6]
    figures = [] if args.no_figures else make_figures(out, c1, c2, c3, c5, c6)

    payload = {
        "mission": "P16-N3 / m5186",
        "module": "tensordem.clump_clump",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_seconds": round(time.time() - started, 2),
        },
        "parameters": {
            "radius_m": RADIUS, "density_kg_m3": DENSITY,
            "young_modulus_Pa": YOUNG, "poisson_ratio": POISSON,
            "e_star_Pa": reference_moduli()[0], "g_star_Pa": reference_moduli()[1],
            "gravity_m_s2": GRAVITY,
        },
        "criteria": criteria,
        "all_passed": all(item["passed"] for item in criteria),
        "figures": figures,
    }
    (out / "criteria.json").write_text(json.dumps(payload, indent=2))

    for item in criteria:
        flag = "PASS" if item["passed"] else "FAIL"
        print(f"[{flag}] {item['id']}: {item['criterion']}")
        if item["id"] == "C1":
            print(f"        worst relative error {item['worst_relative_error']:.3e}")
        elif item["id"] == "C2":
            print(f"        worst relative error {item['worst_relative_error']:.3e}")
        elif item["id"] == "C3":
            print(f"        merged/unique {item['merged_over_unique']:.6f}, "
                  f"unmerged/merged {item['unmerged_over_merged']:.4f}")
        elif item["id"] == "C4":
            print(f"        swap {item['swap_relative_residual']:.3e}, "
                  f"moment {item['net_moment_relative_residual']:.3e}, "
                  f"P {item['linear_momentum_relative_residual']:.3e}")
        elif item["id"] == "C5":
            print(f"        angular momentum {item['angular_momentum_relative_residual']:.3e}, "
                  f"restitution deviation {item['restitution_worst_deviation']:+.4f}")
        elif item["id"] == "C6":
            print(f"        drift {item['drift']:.3e} m over {item['steps']} steps "
                  f"in {item['wall_seconds']:.1f} s")
    print(f"artifacts: {out}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())