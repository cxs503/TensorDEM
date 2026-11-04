"""Reproduce the verification of the in-solver Cundall-Strack friction frame.

Generates the numbers and figures of ``docs/dem3d_tangential_frame_transport.md``
without any hand-copied value:

* ramp / Coulomb-cone series from the production ``IceDEM3D.forces``;
* normal-leakage sweep over rigid rotations of a contact pair, against the
  closed-form prediction for a fixed-frame (naive) implementation;
* contact-loss test: friction force carried by a re-formed contact.

Run from the repository root::

    python scripts/verify_dem3d_tangential_frame.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

RADIUS = 0.025
KN = 5000.0
OVERLAP = 2.0e-3
OUT = Path("docs/dem3d-tangential-frame")


def build_pair(mu: float) -> IceDEM3D:
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=917.0,
        contact_stiffness=KN, contact_damping=0.0, contact_damping_model="absolute",
        drag=0.0, restitution_coefficient=1.0, friction_coefficient=mu,
        rolling_resistance_coefficient=0.0, fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0,
    )
    sim = IceDEM3D(cfg)
    sim.positions[:] = 0.0
    for idx in range(len(sim.positions)):
        sim.positions[idx] = torch.tensor([10.0 + 0.5 * idx, 10.0, 10.0], dtype=sim.dtype)
    sim.velocities[:] = 0.0
    gap = 2.0 * RADIUS - OVERLAP
    sim.positions[0] = torch.tensor([-0.5 * gap, 0.0, 0.0], dtype=sim.dtype)
    sim.positions[1] = torch.tensor([+0.5 * gap, 0.0, 0.0], dtype=sim.dtype)
    sim.alive[:] = False
    sim.bonded[:] = False
    return sim


def normal(sim: IceDEM3D) -> torch.Tensor:
    d = sim.positions[1] - sim.positions[0]
    return d / torch.linalg.vector_norm(d)


def ramp(mu: float, v_t: float, steps: int) -> list[dict]:
    sim = build_pair(mu)
    fn = KN * OVERLAP
    rows = []
    for step in range(1, steps + 1):
        sim.velocities[1] = torch.tensor([0.0, v_t, 0.0], dtype=sim.dtype)
        sim.forces(update_fracture=False, update_history=True)
        sim.velocities[1] = torch.zeros(3, dtype=sim.dtype)
        force, _ = sim.forces(update_fracture=False, update_history=False)
        row_normal = normal(sim)
        friction = force[0] + fn * row_normal
        slip = step * v_t * sim.dt
        rows.append({
            "step": step,
            "slip_m": slip,
            "friction_measured_N": float(torch.linalg.vector_norm(friction)),
            "friction_analytic_N": float(min(KN * slip, mu * fn)),
            "normal_component_N": float(torch.dot(force[0], row_normal)),
        })
    return rows


def leakage_sweep(mu: float, steps: int = 4, v_t: float = 1.0e-3) -> list[dict]:
    """Rotate a pair rigidly after freezing a slip history; measure the leak."""
    fn = KN * OVERLAP
    rows = []
    for deg in (0, 15, 30, 45, 60, 90, 120, 180):
        sim = build_pair(mu)
        sim.velocities[1] = torch.tensor([0.0, v_t, 0.0], dtype=sim.dtype)
        for _ in range(steps):
            sim.forces(update_fracture=False, update_history=True)
        sim.velocities[1] = torch.zeros(3, dtype=sim.dtype)
        sim.velocities[0] = torch.zeros(3, dtype=sim.dtype)
        delta0 = float(KN * steps * v_t * sim.dt)  # spring force magnitude before rotation
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        half = 0.5 * (2.0 * RADIUS - OVERLAP)
        sim.positions[0] = torch.tensor([-c * half, -s * half, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([+c * half, +s * half, 0.0], dtype=sim.dtype)
        force, _ = sim.forces(update_fracture=False, update_history=False)
        n = normal(sim)
        leak = float(torch.dot(force[0], n)) + fn
        friction = force[0] + fn * n
        rows.append({
            "rotation_deg": deg,
            "leak_measured_N": leak,
            "leak_naive_closed_form_N": delta0 * abs(math.sin(th)),
            "friction_magnitude_measured_N": float(torch.linalg.vector_norm(friction)),
            "friction_magnitude_naive_closed_form_N": delta0 * abs(math.cos(th)),
            "friction_magnitude_exact_N": delta0,
        })
    return rows


def contact_loss(mu: float = 0.3, v_t: float = 5.0, steps: int = 4) -> dict:
    sim = build_pair(mu)
    fn = KN * OVERLAP
    sim.velocities[1] = torch.tensor([0.0, v_t, 0.0], dtype=sim.dtype)
    for _ in range(steps):
        sim.forces(update_fracture=False, update_history=True)
    sim.velocities[1] = torch.zeros(3, dtype=sim.dtype)
    saturated = float(torch.linalg.vector_norm(
        sim.forces(update_fracture=False, update_history=False)[0][0]
        + fn * normal(sim)))
    # separate well beyond the contact diameter
    sim.positions[0] = torch.tensor([-1.5 * RADIUS, 0.0, 0.0], dtype=sim.dtype)
    sim.positions[1] = torch.tensor([+1.5 * RADIUS, 0.0, 0.0], dtype=sim.dtype)
    sim.forces(update_fracture=False, update_history=True)
    keys_after_separation = len(sim._tangential_state)
    # re-contact at the original overlap with no relative velocity
    gap = 2.0 * RADIUS - OVERLAP
    sim.positions[0] = torch.tensor([-0.5 * gap, 0.0, 0.0], dtype=sim.dtype)
    sim.positions[1] = torch.tensor([+0.5 * gap, 0.0, 0.0], dtype=sim.dtype)
    force, _ = sim.forces(update_fracture=False, update_history=True)
    inherited = float(torch.linalg.vector_norm(force[0] + fn * normal(sim)))
    return {
        "saturated_friction_before_loss_N": saturated,
        "mu_times_fn_N": mu * fn,
        "history_entries_after_separation": keys_after_separation,
        "inherited_friction_after_recontact_N": inherited,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mu = 0.3
    ramp_rows = ramp(mu=mu, v_t=5.0, steps=5)
    leak_rows = leakage_sweep(mu=1.0)
    loss = contact_loss()

    payload = {
        "config": {"radius_m": RADIUS, "contact_stiffness_N_per_m": KN,
                   "overlap_m": OVERLAP, "mu_ramp": mu, "mu_leak": 1.0},
        "ramp": ramp_rows,
        "leakage": leak_rows,
        "contact_loss": loss,
    }
    (OUT / "tangential_frame_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - figure is optional
        print("matplotlib unavailable; JSON written only")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    slips = [r["slip_m"] for r in ramp_rows]
    ax[0].plot(slips, [r["friction_measured_N"] for r in ramp_rows], "o-",
               label="IceDEM3D.forces (measured)")
    ax[0].plot(slips, [r["friction_analytic_N"] for r in ramp_rows], "s--",
               label=r"analytic min($k_t\Delta s$, $\mu|F_n|$)")
    ax[0].axhline(mu * KN * OVERLAP, color="k", lw=1, ls=":",
                  label=r"Coulomb cone $\mu|F_n|$")
    ax[0].set_xlabel("accumulated slip $|v_t|\\,t$ [m]")
    ax[0].set_ylabel("$|F_t|$ [N]")
    ax[0].set_title("tangential spring ramp and saturation")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    degs = [r["rotation_deg"] for r in leak_rows]
    ax[1].plot(degs, [r["leak_measured_N"] for r in leak_rows], "o-",
               label="this implementation (measured)")
    ax[1].plot(degs, [r["leak_naive_closed_form_N"] for r in leak_rows], "s--",
               label=r"fixed-frame (naive) $k_t\,\Delta s\,|\sin\theta|$")
    ax[1].set_xlabel("rigid rotation of the contact pair [deg]")
    ax[1].set_ylabel("spurious normal component [N]")
    ax[1].set_title("normal leakage after rigid rotation")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "tangential_frame_verification.png", dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()