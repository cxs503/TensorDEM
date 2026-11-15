"""Energy-ledger identity gates for the ridge (rotationless disk) contact model.

Background (m22314): the isochoric biaxial cell of ``CASE-DEM-BIAXIAL-FRICTION``
reported a contact-layer friction dissipation power that exceeded the boundary
work rate ``q * eps_dot * V`` by an order of magnitude (m22106).  Before that
number can be read as a physics result two bookkeeping questions have to be
answered and pinned by tests:

1. Is the friction meter's power the *same* number as the independently
   accumulated ``dissipation['contact_friction']`` of the force routine?
2. Does the closed-system energy ledger (KE + contact-spring energy + integrated
   dissipation) close for a two-body sliding pair and for a disk sliding on a
   plane, i.e. is the accounting consistent *when the dynamics is integrated*
   (the m22106 identity test pinned the velocities, so it could not see this)?

Both are cheap (a few thousand steps of 1-2 bodies) and they are the two
preconditions for attributing the cell-level residual to the time step instead
of to the meter.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.ridge_dem import (  # noqa: E402
    Plane,
    RidgeAssembly,
    RidgeMaterial,
    _advance_planes,
)


def _spring_state(assembly) -> tuple[float, float]:
    """(kinetic, contact-spring) energy of the current state."""
    pos, radii = assembly.pos, assembly.radii
    n = int(pos.shape[0])
    kn = assembly.material.contact_stiffness
    u = 0.0
    if n > 1:
        i, j = torch.triu_indices(n, n, offset=1)
        dist = torch.linalg.vector_norm(pos[j] - pos[i], dim=1)
        overlap = (radii[i] + radii[j] - dist).clamp_min(0.0)
        u = float(0.5 * kn * (overlap ** 2).sum())
    ke = float(0.5 * (assembly.mass * assembly.vel.square().sum(dim=1)).sum())
    return ke, u


def _pair(velocity=(0.0, 0.5)) -> RidgeAssembly:
    material = RidgeMaterial(particle_radius=0.75, friction_coefficient=0.55,
                             freeze_bond_fraction=0.0, contact_damping=0.0)
    pos = torch.tensor([[0.0, 0.0], [1.40, 0.0]], dtype=torch.float64)
    radii = torch.tensor([0.75, 0.75], dtype=torch.float64)
    mass = torch.full((2,), material.particle_mass, dtype=torch.float64)
    assembly = RidgeAssembly(pos, radii, mass,
                             torch.zeros((0, 2), dtype=torch.long), material,
                             planes=[], thickness=1.0)
    assembly.vel = torch.tensor([list(velocity), [0.0, 0.0]], dtype=torch.float64)
    return assembly


def test_meter_power_equals_ledger_contact_friction():
    """The friction meter must reproduce ``dissipation['contact_friction']`` exactly."""
    assembly = _pair()
    assembly.enable_friction_meter()
    dt = assembly.material.recommended_dt
    worst = 0.0
    for _ in range(200):
        out = assembly.step(dt)
        meter = assembly.read_friction_meter(reset=True)
        ledger = float(out["dissipation"]["contact_friction"])
        worst = max(worst, abs(meter["sum_friction_power"] - ledger))
    assert worst < 1e-6, worst


def test_closed_pair_energy_ledger_closes():
    """Two sliding disks, no walls, no drive: KE + U + integral(dissipation) = const."""
    assembly = _pair()
    dt = assembly.material.recommended_dt
    ke0, u0 = _spring_state(assembly)
    e0 = ke0 + u0
    dissipated = 0.0
    for _ in range(4000):
        out = assembly.step(dt)
        dissipated += sum(float(v) for v in out["dissipation"].values()) * dt
    ke, u = _spring_state(assembly)
    drift = (ke + u) + dissipated - e0
    assert abs(drift / e0) < 0.05, (drift, e0)
    # the tangential law must actually have removed the relative sliding
    assert abs(float(assembly.vel[0, 1] - assembly.vel[1, 1])) < 1e-2


def test_wall_slide_energy_ledger_closes():
    """Disk settling on a frictional plane: exchange - dissipation = dKE (+ small)."""
    material = RidgeMaterial(particle_radius=0.75, friction_coefficient=0.55,
                             freeze_bond_fraction=0.0, contact_damping=0.0)
    dt = material.recommended_dt
    pos = torch.tensor([[0.0, 0.70]], dtype=torch.float64)
    radii = torch.tensor([0.75], dtype=torch.float64)
    mass = torch.full((1,), material.particle_mass, dtype=torch.float64)
    plane = Plane((0.0, 1.0), (0.0, 0.0), stiffness=material.contact_stiffness,
                  damping=0.0, friction=0.55, velocity=1.0)
    assembly = RidgeAssembly(pos, radii, mass,
                             torch.zeros((0, 2), dtype=torch.long), material,
                             planes=[plane], thickness=1.0)
    ke0, _ = _spring_state(assembly)
    exchange = dissipated = 0.0
    for _ in range(2000):
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
        exchange += float(out["exchange"]["plane"]) * dt
        dissipated += sum(float(v) for v in out["dissipation"].values()) * dt
    ke, _ = _spring_state(assembly)
    residual = (ke - ke0) - exchange + dissipated
    scale = max(abs(exchange), abs(dissipated), 1.0)
    assert abs(residual / scale) < 0.10, (residual, exchange, dissipated)