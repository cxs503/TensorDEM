#!/usr/bin/env python
"""m22314: dynamic energy-closure probe of the contact model (2-body, no walls, no gravity).

The m22106/m22314 ledger of the isochoric biaxial cell shows the friction
dissipation integrated over the run (100.3 MJ) exceeding the boundary work
measured three independent ways (wall force . particle velocity 14.4 MJ,
wall force . wall velocity 14.7 MJ, macro q*epsdot*V 14.0 MJ) by a factor 6.3,
while dKE (28 kJ) and the contact-spring energy change (4.1 MJ) are far too
small to buffer it.  Either the contact-law dissipation bookkeeping or the
boundary bookkeeping is wrong, or the integrator is not energy consistent in
the *dynamic* regime.

R8 (m22106) verified the meter's power identity on a *prescribed* sliding pair
(velocities held fixed, 27 500 W in = 27 500 W out).  A prescribed-velocity
check cannot detect a bookkeeping error that cancels only when the dynamics is
integrated.  This probe does the integrated check on a closed two-body system:

    E_tot = KE + U_contact  (+ U_wall when walls are present)
    dE_tot/dt = - sum(dissipation)

A drift of E_tot + int(sum(dissipation) dt) is a direct measure of the ledger's
inconsistency, with no boundary and no drive involved.

Usage: PYTHONPATH=src python scripts/m22314_energy_identity_probe.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.ridge_dem import Plane, RidgeAssembly, RidgeMaterial, _advance_planes  # noqa: E402

OUT_SUBDIR = os.path.join("artifacts", "CASE-DEM-BIAXIAL-FRICTION", "energy_ledger")


def _state(assembly) -> tuple[float, float]:
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


def run_pair(*, mu=0.55, vt0=0.5, steps=4000, damping_pair=False):
    material = RidgeMaterial(particle_radius=0.75, friction_coefficient=mu,
                             freeze_bond_fraction=0.0)
    if not damping_pair:
        material = RidgeMaterial(particle_radius=0.75, friction_coefficient=mu,
                                 freeze_bond_fraction=0.0, contact_damping=0.0)
    dt = material.recommended_dt
    pos = torch.tensor([[0.0, 0.0], [1.40, 0.0]], dtype=torch.float64)  # overlap 0.10
    radii = torch.tensor([0.75, 0.75], dtype=torch.float64)
    mass = torch.full((2,), material.particle_mass, dtype=torch.float64)
    assembly = RidgeAssembly(pos, radii, mass, torch.zeros((0, 2), dtype=torch.long),
                             material, planes=[], thickness=1.0)
    mass_per = material.particle_mass
    assembly.vel = torch.tensor([[0.0, vt0], [0.0, 0.0]], dtype=torch.float64)
    ke0, u0 = _state(assembly)
    e0 = ke0 + u0
    dissip = 0.0
    traces = []
    for k in range(steps):
        out = assembly.step(dt)
        dissip += sum(float(v) for v in out["dissipation"].values()) * dt
        if (k + 1) % 500 == 0:
            ke, u = _state(assembly)
            rel_vt = float(assembly.vel[0, 1] - assembly.vel[1, 1])
            traces.append({"step": k + 1, "KE": ke, "U": u,
                           "rel_vt": rel_vt,
                           "dissipation_J": dissip,
                           "drift_J": (ke + u) + dissip - e0})
    ke, u = _state(assembly)
    return {
        "specimen_mass_kg": mass_per,
        "dt": dt, "steps": steps,
        "mu": mu, "vt0": vt0,
        "E0_J": e0,
        "KE_final": ke, "U_final": u,
        "dissipation_J": dissip,
        "ledger_drift_J": (ke + u) + dissip - e0,
        "ledger_drift_fraction": ((ke + u) + dissip - e0) / e0 if e0 else float("nan"),
        "rel_vt_final": float(assembly.vel[0, 1] - assembly.vel[1, 1]),
        "traces": traces,
    }


def run_wall_slide(*, mu_wall=0.0, steps=4000, wall_velocity=0.0):
    """One disk on a moving plane: wall exchange vs dissipation, closed system."""
    material = RidgeMaterial(particle_radius=0.75, friction_coefficient=0.55,
                             freeze_bond_fraction=0.0, contact_damping=0.0)
    dt = material.recommended_dt
    pos = torch.tensor([[0.0, 0.70]], dtype=torch.float64)   # 0.05 overlap with y=0 plane
    radii = torch.tensor([0.75], dtype=torch.float64)
    mass = torch.full((1,), material.particle_mass, dtype=torch.float64)
    plane = Plane((0.0, 1.0), (0.0, 0.0), stiffness=material.contact_stiffness * 1.0,
                  damping=0.0, friction=mu_wall, velocity=wall_velocity)
    assembly = RidgeAssembly(pos, radii, mass, torch.zeros((0, 2), dtype=torch.long),
                             material, planes=[plane], thickness=1.0)
    ke0 = float(0.5 * (assembly.mass * assembly.vel.square().sum(dim=1)).sum())
    diss = 0.0
    exchange = 0.0
    for _ in range(steps):
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
        diss += sum(float(v) for v in out["dissipation"].values()) * dt
        exchange += float(out["exchange"]["plane"]) * dt
    ke = float(0.5 * (assembly.mass * assembly.vel.square().sum(dim=1)).sum())
    return {"mu_wall": mu_wall, "wall_velocity": wall_velocity,
            "KE0": ke0, "KE_final": ke, "exchange_J": exchange,
            "dissipation_J": diss,
            "ledger_drift_J": (ke - ke0) - exchange + diss}


def main() -> int:
    torch.set_num_threads(1)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "module": "CASE-DEM-BIAXIAL-FRICTION",
        "purpose": "closed-system (no drive, no walls) energy ledger of the contact model",
        "pair_frictionless_damping": run_pair(damping_pair=False),
        "pair_with_damping": run_pair(damping_pair=True),
        "wall_sliding_free": run_wall_slide(mu_wall=0.0),
        "wall_sliding_frictional": run_wall_slide(mu_wall=0.55, wall_velocity=1.0),
    }
    path = os.path.join(out_dir, "contact_energy_identity.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    for key, value in record.items():
        if isinstance(value, dict):
            print(key, {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in value.items() if k != "traces"})
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())