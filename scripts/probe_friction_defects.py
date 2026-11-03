"""Probe: quantify friction / rolling-resistance behaviour of the DEM3D contact kernel.

Read-only diagnostic. Each probe drives a two-sphere contact kinematically
(positions restored every step, relative velocity prescribed) so the normal
force is an analytic constant and the friction response can be compared
against the Cundall-Strack closed form.

Run: PYTHONPATH=src python scripts/probe_friction_defects.py
"""
import math

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

R = 0.025
KN = 5000.0
OVERLAP = 0.1 * 2 * R          # 5 mm overlap -> F_n = 25 N


def build(mu=0.0, mu_r=0.0, dt=1e-4):
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=R, contact_stiffness=KN, contact_damping=0.0,
        drag=0.0, friction_coefficient=mu, rolling_resistance_coefficient=mu_r,
        restitution_coefficient=1.0, dt=dt, fix_x_edges=False, fix_bottom=False,
        tool_speed=1e-9, breaking_strain=10.0, shear_breaking_strain=10.0,
    )
    sim = IceDEM3D(cfg)
    sim.alive[:] = False          # isolate contact physics, no bond springs
    place(sim, normal_axis=0)
    return sim


def place(sim, normal_axis=0):
    """Restore the overlapping pair, normal along ``normal_axis``."""
    separation = 2 * R - OVERLAP
    p0 = torch.zeros(3, dtype=sim.dtype)
    p1 = torch.zeros(3, dtype=sim.dtype)
    p1[normal_axis] = separation
    sim.positions[0] = -p1 / 2
    sim.positions[1] = +p1 / 2
    # park the remaining two lattice particles far away (bonds are dead)
    for k in range(2, len(sim.positions)):
        sim.positions[k] = torch.tensor([0.0, 0.0, 1.0 + k], dtype=sim.dtype)
    sim.initial_positions.copy_(sim.positions)


def drive(sim, speed, steps, tangent_axis=1, normal_axis=0, record=None):
    cfg = sim.config
    out = []
    for _ in range(steps):
        v = torch.zeros(3, dtype=sim.dtype)
        v[tangent_axis] = speed
        sim.velocities[0] = -0.5 * v
        sim.velocities[1] = +0.5 * v
        force, _ = sim.forces()
        # force on particle 1; remove the symmetric half (spring on i is +F_t,
        # on j is -F_t for the Coulomb path, so read the raw components)
        out.append({
            "f_normal": float(force[1, normal_axis]),
            "f_tangent": float(force[1, tangent_axis]),
            "state": float(torch.linalg.vector_norm(
                torch.stack([v2[0] for v2 in sim._tangential_state.values()])
            )) if sim._tangential_state else 0.0,
            "n_states": len(sim._tangential_state),
        })
        place(sim, normal_axis)
    return out


def probe_A():
    """Sliding then reversal: friction sign must follow the velocity."""
    mu = 0.1
    sim = build(mu=mu)
    f_n = KN * OVERLAP
    f_limit = mu * f_n
    a_stick = f_limit / KN
    hist = drive(sim, +1.0, 100)
    recorded = hist[-1]["state"]
    rev = drive(sim, -1.0, 20)
    flip_step = next((k + 1 for k, h in enumerate(rev) if h["f_tangent"] < 0), None)
    print(f"[A] F_n={f_n:.3f} N  F_limit=mu*F_n={f_limit:.4f} N  a_stick={a_stick:.3e} m")
    print(f"[A] driven 0.1 m/s x 0.1 s -> sliding distance 0.010 m = {0.01 / a_stick:.1f} x a_stick")
    print(f"[A] stored tangential displacement |delta| = {recorded:.6e} m "
          f"(expected clamped {a_stick:.6e} m, ratio {recorded / a_stick:.1f})")
    print(f"[A] sliding |F_t| = {hist[-1]['f_tangent']:.4f} N (expect {f_limit:.4f})")
    print(f"[A] steps after velocity reversal until F_t changes sign: {flip_step}")
    print(f"[A] F_t during first 3 reversal steps: "
          f"{[round(h['f_tangent'], 4) for h in rev[:3]]}")
    return {"f_n": f_n, "f_limit": f_limit, "a_stick": a_stick,
            "stored_delta": recorded, "flip_step": flip_step}


def probe_B():
    """Contact normal rotates: tangential memory must be re-projected."""
    mu = 0.2
    sim = build(mu=mu)
    drive(sim, +1.0, 100, tangent_axis=1, normal_axis=0)
    # rotate the contact normal by 90 deg: now y is the normal
    place(sim, normal_axis=1)
    sim.velocities[:] = 0.0
    force, _ = sim.forces()
    fn = float(force[1, 1])
    print(f"[B] after normal rotation (x->y) with zero relative velocity:")
    print(f"[B]   force on particle 1 along new normal = {fn:.6f} N "
          f"(friction-limited max {mu * KN * OVERLAP:.4f} N)")
    return {"spurious_normal_force": fn}


def probe_C():
    """Separation must clear tangential memory."""
    sim = build(mu=0.2)
    drive(sim, +1.0, 100)
    sep = 3 * R
    for _ in range(50):
        sim.positions[0] = torch.tensor([-sep / 2, 0.0, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([+sep / 2, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[:] = 0.0
        sim.forces()
    n_after_separation = len(sim._tangential_state)
    place(sim, normal_axis=0)
    sim.velocities[:] = 0.0
    force, _ = sim.forces()
    ft = float(force[1, 1])
    print(f"[C] contact pairs retained in state after 50 separated steps: {n_after_separation}")
    print(f"[C] tangential force on re-contact with zero relative velocity = {ft:.6f} N")
    return {"states_retained": n_after_separation, "recontact_force": ft}


def probe_D():
    """Rolling resistance must oppose tangential motion (dissipative)."""
    mu_r = 0.05
    sim = build(mu=0.0, mu_r=mu_r)
    hist = drive(sim, +1.0, 5)
    f_n = KN * OVERLAP
    fy = hist[-1]["f_tangent"]
    fz = float(0.0)
    sim2 = build(mu=0.0, mu_r=mu_r)
    v = torch.zeros(3, dtype=sim2.dtype); v[1] = 1.0
    sim2.velocities[0] = -0.5 * v
    sim2.velocities[1] = +0.5 * v
    f2, _ = sim2.forces()
    fz = float(f2[1, 2])
    power = float(f2[1, 1] * 1.0)
    print(f"[D] mu_r={mu_r}, mu=0, F_n={f_n:.3f} N -> mu_r*F_n = {mu_r * f_n:.4f} N")
    print(f"[D] force on particle 1: F_y(tangential, opposing motion expected -) = {fy:.6f} N, "
          f"F_z(spurious lateral, expected 0) = {fz:.6f} N")
    print(f"[D] dissipative power F_t*v_t = {power:.6f} W "
          f"(expect -{mu_r * f_n * 1.0:.6f} W)")
    return {"f_tangential": fy, "f_lateral": fz, "power": power}


if __name__ == "__main__":
    print("=" * 70)
    a = probe_A()
    print("=" * 70)
    b = probe_B()
    print("=" * 70)
    c = probe_C()
    print("=" * 70)
    d = probe_D()
    print("=" * 70)