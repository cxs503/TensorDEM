"""Static (relaxed) elastic field of the Brazilian disc lattice.

Why this script exists
----------------------
`docs/dem3d_platen_contact_mask_audit.md` §6 reports the diametral (Brazilian)
centre field as **WITHHELD**: after 900 explicit steps the centre virial stress
is still ``-0.06 Pa`` against the Hondros point-load value ``2 P / (pi D t)``,
and the argument was that a wave transit across ``D`` alone needs ~3.3e3 steps
and the viscoelastic relaxation longer, so a campaign of order 2e4 steps would
have to be run before the physics checks mean anything.

A long transient is only *one* of two possible explanations.  The other is
structural: the specimen is a central-force (normal-only) bond network on a
cubic-ish lattice, and such a lattice has ``C12 = C44 = 0`` in the Born
(Affine) limit, i.e. zero shear rigidity and zero Poisson ratio, so it is not
the isotropic continuum assumed by the Hondros solution at all.

This script settles the question without a transient.  It solves the *static*
equilibrium of the very same bond network at the very same platen travel with a
linearised stiffness matrix:

    K(x) u = -F(0),      K = K_bonds + K_platen(active set)

with ``F(0)`` the engine's own unbalanced force at zero velocity.  The result is
the ``t -> infinity, no damping`` limit of the dynamic run at that travel.  If
the static field also misses Hondros, the miss is structural (lattice
anisotropy / finite contact arc), not a matter of running longer.

Scope: numerical verification of the apparatus.  Nothing here calibrates an ice
material or certifies a tensile strength.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import (  # noqa: E402  (script-level import)
    MANIFEST,
    _build,
    axial_profile,
    centre_field,
    contact_arc,
    specimen_geometry,
    virial_stress,
)
from tensordem.lattice_elasticity3d import (  # noqa: E402
    periodic_elastic_constants,
    derived_moduli,
)


def _engine_force(sim: "Any", travel: float, speed: float) -> np.ndarray:
    """Engine net force per particle at zero velocity for a given platen travel.

    The platens are positioned through ``sim.time`` because that is the only
    handle the engine exposes; a zero velocity field removes the drag and
    dashpot contributions so the result is the pure elastic residual ``F(0)``.
    """
    sim.time = travel / speed
    sim.velocities.zero_()
    force, _ = sim.forces(update_fracture=False)
    return force.detach().cpu().numpy().astype(float).reshape(-1)


def lattice_energy_force(positions: np.ndarray, pairs: np.ndarray, rest: np.ndarray,
                         alive: np.ndarray, k_bond: float, k_platen: float,
                         radius: float, top_z: float, bottom_z: float
                         ) -> tuple[float, np.ndarray]:
    """Total elastic potential and its gradient for the Brazilian specimen.

    Mirrors the engine's own force law exactly (linear bond springs on the
    current normal, ``k_bond`` independent of rest length; infinite platens with
    ``k_platen`` acting on the radial overlap), so the gradient is the negative
    of the engine force.  ``tests/test_dem3d_brazilian_static.py`` asserts that
    equivalence, which is what licenses using this cheaper numpy form inside the
    minimiser.
    """
    i, j = pairs[:, 0], pairs[:, 1]
    delta = positions[j] - positions[i]
    distance = np.linalg.norm(delta, axis=1)
    safe = np.where(distance > 0.0, distance, 1.0)
    normal = delta / safe[:, None]
    extension = np.where(alive, distance - rest, 0.0)
    bond_energy = 0.5 * k_bond * np.sum(extension**2)
    bond_force = (k_bond * extension)[:, None] * normal
    force = np.zeros_like(positions)
    np.add.at(force, i, bond_force)
    np.add.at(force, j, -bond_force)

    top_overlap = np.clip(positions[:, 2] + radius - top_z, 0.0, None)
    bottom_overlap = np.clip(bottom_z + radius - positions[:, 2], 0.0, None)
    platen_energy = 0.5 * k_platen * (np.sum(top_overlap**2) + np.sum(bottom_overlap**2))
    force[:, 2] += -k_platen * top_overlap
    force[:, 2] += k_platen * bottom_overlap
    return float(bond_energy + platen_energy), force


def solve_static_equilibrium(case: dict[str, Any], travel: float, *,
                             increments: int = 20,
                             max_iterations: int = 4000,
                             tolerance_Pa: float = 1e-4,
                             return_configuration: bool = False) -> dict[str, Any]:
    """Relaxed configuration of the bond network at a prescribed platen travel.

    Static equilibrium is found by minimising the elastic potential with L-BFGS
    along a load continuation: the travel is applied in ``increments`` steps so
    that each minimisation starts close to its solution.  A minimiser is used
    instead of a Newton solve because the engine exposes only the bond law
    (material tangent); the geometric stiffness that large rotations need is not
    available, and damping a Newton loop that lacks it diverges.

    ``tolerance_Pa`` is the convergence criterion expressed as a bond-stress
    equivalent (``|F|_inf / (spacing^2)``), so it is a physical tolerance and not
    a machine-epsilon one.

    ``return_configuration`` additionally returns the relaxed ``positions`` and
    the ``sim``/``geometry`` pair under a ``"configuration"`` key.  Those objects
    are deliberately kept out of the default result because it must stay
    JSON-serialisable; they exist so that one expensive relaxation can be reused
    to re-measure the field (see ``scripts/brazilian_hondros_estimator.py``).
    """
    sim, geometry = _build(case)
    p = case["config"]
    radius = float(p["radius_m"])
    speed = float(p["loading_speed_m_s"])
    spacing = 2.0 * radius
    k_bond = float(p["bond_stiffness_N_m"])
    k_platen = float(p["platen_stiffness_N_m"])
    diameter = geometry["diameter_m"]
    thickness = geometry["thickness_m"]

    x0 = sim.positions.detach().cpu().numpy().astype(float).copy()
    pairs = sim.pairs.detach().cpu().numpy()
    rest = sim.rest_lengths.detach().cpu().numpy().astype(float)
    alive = (sim.bonded & sim.alive).detach().cpu().numpy()
    n = len(x0)

    force_tolerance = tolerance_Pa * spacing**2
    positions = x0.copy()
    history: list[dict[str, Any]] = []
    total_iterations = 0
    converged = True
    for increment in range(1, increments + 1):
        current_travel = travel * increment / increments
        top_z = sim.top_platen_start_z - current_travel
        bottom_z = sim.bottom_platen_start_z + current_travel

        def objective(flat: np.ndarray):
            energy, force = lattice_energy_force(
                flat.reshape(n, 3), pairs, rest, alive, k_bond, k_platen,
                radius, top_z, bottom_z)
            return energy, -force.reshape(-1)

        result = minimize(objective, positions.reshape(-1), jac=True, method="L-BFGS-B",
                          options={"maxiter": max_iterations, "maxfun": 4 * max_iterations,
                                   "gtol": force_tolerance, "ftol": 1e-18})
        positions = result.x.reshape(n, 3)
        total_iterations += int(result.nit)
        _, force = lattice_energy_force(positions, pairs, rest, alive, k_bond,
                                        k_platen, radius,
                                        sim.top_platen_start_z - current_travel,
                                        sim.bottom_platen_start_z + current_travel)
        residual_max = float(np.abs(force).max())
        history.append({
            "increment": increment,
            "travel_m": current_travel,
            "total_platen_travel_m": travel * increment / increments,
            "lbfgs_iterations": int(result.nit),
            "energy_J": float(result.fun),
            "residual_max_N": residual_max,
            "residual_tolerance_N": force_tolerance,
            "converged": bool(residual_max <= force_tolerance),
        })
        converged = converged and bool(residual_max <= force_tolerance)

    final_travel = travel
    _, force_final = lattice_energy_force(
        positions, pairs, rest, alive, k_bond, k_platen, radius,
        sim.top_platen_start_z - final_travel, sim.bottom_platen_start_z + final_travel)
    residual_max = float(np.abs(force_final).max())
    extension = np.linalg.norm(positions[pairs[:, 1]] - positions[pairs[:, 0]], axis=1) - rest
    bond_force_scale = float(np.abs(k_bond * extension[alive]).max()) if alive.any() else 0.0

    sim.positions.copy_(torch.as_tensor(positions, dtype=sim.dtype))
    sim.velocities.zero_()
    sim.time = final_travel / speed
    stress = virial_stress(sim, radius)
    centre = centre_field(sim, stress, diameter)
    profile = axial_profile(sim, stress, diameter)
    top_z = sim.top_platen_start_z - final_travel
    bottom_z = sim.bottom_platen_start_z + final_travel
    arc = contact_arc(sim, radius, diameter, top_z)

    top_overlap = np.clip(positions[:, 2] + radius - top_z, 0.0, None)
    bottom_overlap = np.clip(bottom_z + radius - positions[:, 2], 0.0, None)
    top_load = float(k_platen * top_overlap.sum())
    bottom_load = float(k_platen * bottom_overlap.sum())
    load = 0.5 * (top_load + bottom_load)
    split = 2.0 * load / (math.pi * diameter * thickness)
    ratio = (centre["centre_sigma_zz_Pa"] / centre["centre_sigma_xx_Pa"]
             if centre["centre_sigma_xx_Pa"] else float("nan"))
    rel_error = (centre["centre_sigma_xx_Pa"] - split) / split if split else float("nan")
    prof_values = [entry["sigma_xx_Pa"] for entry in profile]
    mean_profile = sum(prof_values) / len(prof_values) if prof_values else 0.0
    spread = (math.sqrt(sum((v - mean_profile) ** 2 for v in prof_values) / len(prof_values))
              if prof_values else float("inf"))
    uniformity = spread / abs(mean_profile) if mean_profile else float("inf")
    displacement = positions - x0

    return {
        "case_id": case["id"],
        "method": ("static elastic equilibrium by L-BFGS minimisation of the bond+platen "
                   "potential along a 20-step load continuation"),
        "platen_travel_m": travel,
        "increments": increments,
        "lbfgs_iterations_total": total_iterations,
        "equilibrium_converged": bool(converged),
        "force_tolerance_N": force_tolerance,
        "specimen_diameter_m": diameter,
        "specimen_thickness_m": thickness,
        "particle_count": int(n),
        "bond_count": int(alive.sum()),
        "platen_stiffness_N_m": k_platen,
        "bond_stiffness_N_m": k_bond,
        "top_contact_particles": int((top_overlap > 0).sum()),
        "bottom_contact_particles": int((bottom_overlap > 0).sum()),
        "contact_arc_ratio": arc["contact_arc_ratio"],
        "top_platen_load_N": top_load,
        "bottom_platen_load_N": bottom_load,
        "mean_platen_load_N": load,
        "load_balance_rel": (abs(top_load - bottom_load) / load if load else float("nan")),
        "split_stress_Pa": split,
        "analytic_centre_sigma_xx_Pa": split,
        **centre,
        "centre_principal_ratio": ratio,
        "centre_tension_rel_error": rel_error,
        "axial_uniformity_rel": uniformity,
        "axial_profile": profile,
        "equilibrium_residual_max_N": residual_max,
        "bond_force_scale_N": bond_force_scale,
        "equilibrium_residual_rel": (residual_max / bond_force_scale if bond_force_scale else None),
        "displacement_max_m": float(np.abs(displacement).max()),
        "axial_shortening_m": float(-displacement[:, 2].max() + displacement[:, 2].min()),
        "history": history,
        **({"configuration": {"sim": sim, "geometry": geometry,
                              "positions": positions}} if return_configuration else {}),
    }


def lattice_moduli(case: dict[str, Any]) -> dict[str, float]:
    """Born (affine) elastic constants of the same bond network on its lattice."""
    p = case["config"]
    radius = float(p["radius_m"])
    spacing = 2.0 * radius
    constants = periodic_elastic_constants(float(p["bond_stiffness_N_m"]), spacing)
    derived = derived_moduli(constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"])
    return {**constants, **derived, "analysis": "Born/affine, periodic lattice"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="brazil_elastic_field")
    parser.add_argument("--travel", type=float, default=None,
                        help="platen travel in metres (default: the manifest ramp end)")
    parser.add_argument("--steps", type=int, default=None,
                        help="convert a step count to travel using the case dt")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dem3d_brazilian_static"))
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["id"] == args.case)
    p = case["config"]
    _, geometry = _build(case)
    dt = float(geometry["dt_s"])
    if args.travel is not None:
        travel = float(args.travel)
    else:
        steps = int(args.steps if args.steps is not None else case.get("steps", 40000))
        travel = steps * dt * float(p["loading_speed_m_s"])

    result = solve_static_equilibrium(case, travel)
    result["dt_s"] = dt
    result["manifest_steps"] = int(case.get("steps", 40000))
    result["manifest_travel_m"] = float(case.get("steps", 40000)) * dt * float(p["loading_speed_m_s"])
    result["reference"] = manifest["reference"]
    result["acceptance"] = manifest["acceptance"]
    result["hondros_check"] = {
        "centre_principal_ratio_target": float(manifest["reference"]["centre_principal_stress_ratio"]),
        "centre_principal_ratio_rel_error": (
            abs(result["centre_principal_ratio"] - float(
                manifest["reference"]["centre_principal_stress_ratio"]))
            / abs(float(manifest["reference"]["centre_principal_stress_ratio"]))),
        "centre_tension_rel_error": result["centre_tension_rel_error"],
    }
    result["lattice_moduli"] = lattice_moduli(case)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{case['id']}_static.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_id": result["case_id"],
        "travel_m": travel,
        "mean_platen_load_N": result["mean_platen_load_N"],
        "split_stress_Pa": result["split_stress_Pa"],
        "centre_sigma_xx_Pa": result["centre_sigma_xx_Pa"],
        "centre_sigma_zz_Pa": result["centre_sigma_zz_Pa"],
        "centre_principal_ratio": result["centre_principal_ratio"],
        "centre_tension_rel_error": result["centre_tension_rel_error"],
        "axial_uniformity_rel": result["axial_uniformity_rel"],
        "equilibrium_residual_rel": result["equilibrium_residual_rel"],
        "load_balance_rel": result["load_balance_rel"],
        "C11_Pa": result["lattice_moduli"]["C11_Pa"],
        "C12_Pa": result["lattice_moduli"]["C12_Pa"],
        "C44_Pa": result["lattice_moduli"]["C44_Pa"],
        "youngs_modulus_Pa": result["lattice_moduli"].get("youngs_modulus_Pa"),
        "poisson_ratio": result["lattice_moduli"].get("poisson_ratio"),
    }, indent=2))


if __name__ == "__main__":
    main()