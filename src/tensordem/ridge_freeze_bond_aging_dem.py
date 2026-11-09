"""DEM keel-tip penetration probe: peak platen resistance of a bonded/aged keel.

Used by ``scripts/run_ridge_freeze_bond_aging.py`` and by
``tests/test_ridge_freeze_bond_aging.py`` to measure ``R_keel(t)`` *directly in
the DEM*, not just from the closed-form keel model.
"""
from __future__ import annotations

import math

import torch

from .ridge_dem import RidgeAssembly, RidgeKeel, RidgeMaterial, generate_keel_assembly

__all__ = ["dem_keel_penetration", "dem_bond_element_capacity"]


def dem_keel_penetration(assembly: RidgeAssembly, *,
                         platen_velocity: float = 0.5,
                         stroke: float = 1.0,
                         settle_steps: int = 200,
                         max_steps: int = 40000) -> dict:
    """Push the kinematic top row of a keel assembly down and peak the reaction.

    Parameters
    ----------
    assembly
        Keel assembly from :func:`tensordem.ridge_dem.generate_keel_assembly`.
    platen_velocity
        Downward velocity of the kinematic top row [m/s], positive down.
    stroke
        Penetration depth to be reached [m].
    settle_steps
        Steps to let the assembly relax under buoyancy before the drive starts.

    Returns
    -------
    dict with the peak and mean reaction per unit thickness [N/m], the
    penetration series and the number of broken bonds.
    """
    for name, value in (("platen_velocity", platen_velocity), ("stroke", stroke)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    dt = assembly.material.recommended_dt
    driven = assembly.driven
    if not bool(driven.any()):
        raise ValueError("assembly has no driven (platen) particles")
    assembly.set_platen_velocity(torch.zeros(2, dtype=torch.float64))
    for _ in range(int(settle_steps)):
        assembly.step(dt)
    start = float(assembly.pos[:, 1].max())
    baseline = float(assembly.forces(None, update_memory=False)["force"][driven][:, 1].sum())
    assembly.set_platen_velocity(torch.tensor([0.0, -float(platen_velocity)], dtype=torch.float64))
    displacement = []
    reaction = []
    for _ in range(int(max_steps)):
        out = assembly.step(dt)
        force = out["force"]
        # resistance = upward reaction of the rubble on the downward-driven platen,
        # measured relative to the settled state so that buoyancy cancels
        push = float(force[driven][:, 1].sum()) - baseline
        depth = start - float(assembly.pos[:, 1].max())
        displacement.append(depth)
        reaction.append(push)
        if depth >= stroke:
            break
    depth_t = torch.tensor(displacement, dtype=torch.float64)
    force_t = torch.tensor(reaction, dtype=torch.float64)
    tail = max(1, force_t.numel() // 10)
    return {
        "peak_resistance_N_per_m": float(force_t.max()) / assembly.thickness,
        "resistance_at_max_penetration_N_per_m": float(force_t[-1]) / assembly.thickness,
        "mean_resistance_last_decile_N_per_m": float(force_t[-tail:].mean()) / assembly.thickness,
        "max_penetration_m": float(depth_t.max()),
        "steps": int(force_t.numel()),
        "dt": dt,
        "broken_bonds": assembly.broken_bonds,
        "live_bonds": int(assembly.bond_alive.sum()),
        "penetration_m": displacement,
        "resistance_N_per_m": reaction,
    }


def dem_bond_element_capacity(material: RidgeMaterial, *, tolerance: float = 1e-6) -> dict:
    """Break one DEM freeze bond in tension *quasi-statically* and report the threshold.

    Two disks joined by a single bond are placed at a prescribed axial extension
    and the force state is evaluated without time integration
    (``RidgeAssembly.forces(dt=None, update_memory=False)``), so the test probes
    the bond law itself and not the explicit integrator.  The bond must survive
    at ``(1 - tolerance) * sigma_b A / k`` and fail at ``(1 + tolerance)`` of it:
    this is what links ``ridge_dem``'s ``bond_tensile_strength`` to the aging law.
    """
    radius = material.particle_radius
    threshold = material.bond_tensile_strength * material.bond_area / material.bond_stiffness

    def evaluate(extension: float) -> dict:
        # the rest length is taken from the positions at construction, so the
        # extension is applied afterwards: that is the state the bond is in.
        positions = torch.tensor([[0.0, 0.0], [2.0 * radius, 0.0]], dtype=torch.float64)
        masses = torch.full((2,), material.particle_mass, dtype=torch.float64)
        assembly = RidgeAssembly(positions, torch.full((2,), radius, dtype=torch.float64),
                                 masses, torch.tensor([[0, 1]]), material)
        assembly.pos[1, 0] = assembly.pos[1, 0] + float(extension)
        out = assembly.forces(None, update_memory=False)
        return {
            "force_N": abs(float(out["force"][1][0])),
            "broken": bool(assembly.broken_bonds) or not bool(assembly.bond_alive[0]),
            "live": int(assembly.bond_alive.sum()),
        }

    below = evaluate((1.0 - tolerance) * threshold)
    above = evaluate((1.0 + tolerance) * threshold)
    return {
        "bond_area_m2": material.bond_area,
        "threshold_extension_m": threshold,
        "expected_failure_force_N": material.bond_tensile_strength * material.bond_area,
        "force_below_threshold_N": below["force_N"],
        "force_above_threshold_N": above["force_N"],
        "bond_alive_below": not below["broken"],
        "bond_broken_above": above["broken"],
        "rel_error": abs(below["force_N"]
                         - (1.0 - tolerance) * material.bond_tensile_strength
                         * material.bond_area) / (material.bond_tensile_strength
                                                  * material.bond_area),
    }
