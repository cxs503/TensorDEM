"""Verification drivers for the ice-particle Mohr-Coulomb surface.

Four experiments, each with a machine-checkable criterion:

C1/C2/C6  :func:`run_uniaxial`      -- cylindrical bonded specimen (D = 50 mm,
                                       H = 100 mm) loaded by two rigid grips;
                                       peak stress vs ``2c cos(phi)/(1-sin phi)``,
                                       measured ``E`` vs the affine lattice
                                       prediction, energy ledger closure.
C3        :func:`run_indentation`   -- flat punch of projected area ``A`` pressed
                                       into a bonded ice layer; peak pressure vs
                                       ``A`` gives the Korzhavin exponent.
C4        :func:`run_ice_resistance`-- ice rubble pushed by a wall; resistance
                                       ``R_ice`` at ``mu = 0.1`` vs ``0.3``.
C5        :func:`force_chain_stats` -- contact-force PDF tail from the compacted
                                       rubble, exponential fit for ``f > <f>``.

Every number returned by these drivers is computed from the simulation state
(control-section force, punch reaction, wall reaction, contact forces) -- none is
copied from the input parameters.  The only *inputs* are material and numerical
parameters, which is what makes the comparisons non-circular.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from . import analysis
from .bond import (MohrCoulombStrength, affine_stiffness_matrix,
                   affine_uniaxial_modulus, mohr_coulomb_ucs)
from .engine import AssemblyConfig, IceParticleAssembly, PlaneWall
from .packing import (Specimen, cylinder_specimen, ice_layer, orientation_coverage,
                      validate_cylinder)

__all__ = ["run_uniaxial", "run_indentation", "run_ice_resistance",
           "affine_fabric_strength",
           "force_chain_stats", "affine_modulus_prediction"]

def _pin_threads() -> None:
    """Small models: intra-op parallelism costs far more than it saves here.

    Measured (m9179): 3.5 ms/step at one thread vs 98 ms/step at the default 128
    threads for the 264-particle specimen.  Pinned inside the drivers so that a
    caller cannot accidentally run the verification 30x slower.
    """
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)


GRAVITY_OFF_NOTE = ("gravity is off: the specimen is load controlled through "
                    "the grips and the layer through the punch, so the "
                    "self-weight of a 50 mm core is negligible by design")


def affine_fabric_strength(specimen: Specimen, config: AssemblyConfig,
                           cohesion_pa: float | None = None) -> dict[str, Any]:
    """Analytic uniaxial strength of *this* bond fabric (affine strain field).

    The continuum Mohr-Coulomb formula ``2c cos(phi)/(1-sin phi)`` is the strength
    of a continuum whose every plane carries the *macroscopic* traction.  A bond
    set does not: in an affine field the traction a single bond transmits is
    ``E_bond * (projective strain on that bond)``, which differs from the macro
    traction on the same plane by the fabric factor (the same factor that makes
    ``E_macro = E_bond / g``).  The physically meaningful prediction for a bonded
    particle specimen is therefore the macro stress at which the *first bond*
    reaches its Mohr-Coulomb limit:

        sigma_pred = min_b  c_b / ( tau_b - max(sigma_n,b, 0) tan(phi_b) )

    with ``tau_b`` and ``sigma_n,b`` evaluated for a unit macro uniaxial stress.
    ``sigma_pred`` is exactly linear in ``c_b``, so a target macro cohesion maps
    to the bond cohesion in closed form -- this is the standard BPM micro-macro
    calibration, made explicit instead of hidden in a trial-and-error loop.
    """
    from .bond import bond_areas, normal_stiffness_from_modulus, \
        shear_stiffness_from_normal
    cohesion = config.cohesion_pa if cohesion_pa is None else float(cohesion_pa)
    area = bond_areas(specimen.volume, specimen.n_bonds, specimen.bond_length)
    kn = normal_stiffness_from_modulus(config.bond_modulus, area, specimen.bond_length)
    ks = shear_stiffness_from_normal(kn, config.shear_stiffness_ratio)
    normals = (specimen.positions[specimen.bond_j] - specimen.positions[specimen.bond_i])
    normals = normals / normals.norm(dim=1, keepdim=True)
    stiffness = affine_stiffness_matrix(normals, specimen.bond_length, kn, ks,
                                        specimen.volume)
    modulus = affine_uniaxial_modulus(stiffness, axis=2)
    compliance = torch.linalg.inv(stiffness)
    strain = compliance @ torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                       dtype=torch.float64)
    tensor = torch.zeros(3, 3, dtype=torch.float64)
    tensor[0, 0], tensor[1, 1], tensor[2, 2] = strain[0], strain[1], strain[2]
    tensor[1, 2] = tensor[2, 1] = 0.5 * strain[3]
    tensor[0, 2] = tensor[2, 0] = 0.5 * strain[4]
    tensor[0, 1] = tensor[1, 0] = 0.5 * strain[5]
    projected = normals @ tensor.T
    normal_strain = (projected * normals).sum(dim=1)
    tangential = projected - normal_strain[:, None] * normals
    # bond traction for a unit macro stress (compression positive normal stress)
    sigma_n = config.bond_modulus * normal_strain
    tau = config.bond_modulus * tangential.norm(dim=1)
    denominator = tau - torch.clamp(sigma_n, min=0.0) * config.strength.tan_phi
    usable = denominator > 1e-30
    if not bool(usable.any()):
        return {"sigma_ucs_Pa": float("inf"), "E_affine_Pa": modulus,
                "fabric_amplification": float("nan")}
    ratio = torch.where(usable, denominator, torch.zeros_like(denominator))
    index = int(torch.argmax(ratio))
    ucs = cohesion / float(ratio[index])
    continuum = mohr_coulomb_ucs(cohesion, config.friction_angle_deg)
    from .bond import mohr_coulomb_critical_normal_fraction
    return {
        "sigma_ucs_Pa": float(ucs),
        "E_affine_Pa": modulus,
        "poisson_ratio_fabric": float(stiffness[1, 0] / stiffness[2, 2]),
        "fabric_amplification": float(continuum / ucs) if ucs else float("nan"),
        "critical_bond_n_z_squared": float(normals[index, 2] ** 2),
        "critical_bond_sigma_n_Pa": float(sigma_n[index]),
        "critical_bond_tau_Pa": float(tau[index]),
        "critical_n_z_squared_continuum": mohr_coulomb_critical_normal_fraction(
            config.friction_angle_deg),
        "cohesion_pa": float(cohesion),
    }


def affine_modulus_prediction(specimen: Specimen, config: AssemblyConfig) -> float:
    """Macroscopic ``E`` predicted from the bond stiffness and the bond fabric."""
    from .bond import bond_areas, normal_stiffness_from_modulus, \
        shear_stiffness_from_normal
    area = bond_areas(specimen.volume, specimen.n_bonds, specimen.bond_length)
    kn = normal_stiffness_from_modulus(config.bond_modulus, area, specimen.bond_length)
    ks = shear_stiffness_from_normal(kn, config.shear_stiffness_ratio)
    normals = (specimen.positions[specimen.bond_j]
               - specimen.positions[specimen.bond_i])
    normals = normals / normals.norm(dim=1, keepdim=True)
    stiffness = affine_stiffness_matrix(normals, specimen.bond_length, kn, ks,
                                        specimen.volume)
    return affine_uniaxial_modulus(stiffness, axis=2)


# ----------------------------------------------------------------------
# C1 / C2 / C6 -- uniaxial compression of a D = 50 mm, H = 100 mm core
# ----------------------------------------------------------------------
@dataclass
class UniaxialSpec:
    cohesion_pa: float
    friction_angle_deg: float
    tensile_strength_pa: float = 1.0e6
    seed: int = 0
    diameter: float = 0.05
    height: float = 0.10
    particle_radius: float = 0.004
    bond_modulus: float = 9.0e9
    shear_stiffness_ratio: float = 0.4
    viscous_damping: float = 2.0e3        # 1/s, wave tamer (energy accounted)
    step_displacement: float = 5.0e-8     # m per step of platen closure
    ramp_steps: int = 300                 # smooth start: no shock wave
    strain_target: float = 4.0e-3
    boundary_strengthening: float = 3.0
    density_scale: float = 1.0
    bond_gap_fraction: float = 0.25   # bond reach / (r_i + r_j): sets coordination


def run_uniaxial(spec: UniaxialSpec) -> dict[str, Any]:
    """Displacement-controlled uniaxial compression; returns the measured curve.

    The *bond* cohesion is calibrated (closed form, see
    :func:`affine_fabric_strength`) so that the specimen's analytic uniaxial
    strength equals the requested macro cohesion's Mohr-Coulomb value
    ``2 c cos(phi)/(1 - sin(phi))``.  The friction angle is *not* calibrated:
    ``phi_b = phi`` is a transfer assumption, which is what the three-group test
    then checks.  The measured peak is a dynamical, non-affine number -- it is
    not imposed anywhere.
    """
    _pin_threads()
    specimen = cylinder_specimen(diameter=spec.diameter, height=spec.height,
                                 particle_radius=spec.particle_radius,
                                 spacing_fraction=1.10,
                                 jitter_fraction=0.50,      # quasi-random bond fabric
                                 bond_gap_fraction=spec.bond_gap_fraction,
                                 seed=spec.seed)
    probe = AssemblyConfig(
        bond_modulus=spec.bond_modulus,
        shear_stiffness_ratio=spec.shear_stiffness_ratio,
        cohesion_pa=spec.cohesion_pa,
        friction_angle_deg=spec.friction_angle_deg,
        tensile_strength_pa=spec.tensile_strength_pa)
    target_ucs = mohr_coulomb_ucs(spec.cohesion_pa, spec.friction_angle_deg)
    probe_strength = affine_fabric_strength(specimen, probe, cohesion_pa=spec.cohesion_pa)
    bond_cohesion = spec.cohesion_pa * target_ucs / probe_strength["sigma_ucs_Pa"]
    config = AssemblyConfig(
        bond_modulus=spec.bond_modulus,
        shear_stiffness_ratio=spec.shear_stiffness_ratio,
        cohesion_pa=bond_cohesion,
        friction_angle_deg=spec.friction_angle_deg,
        tensile_strength_pa=spec.tensile_strength_pa,
        boundary_strengthening=spec.boundary_strengthening,
        boundary_layer_thickness=2.0 * spec.particle_radius,
        viscous_damping=spec.viscous_damping,
        density_scale=spec.density_scale,
        store_history=False,
    )
    assembly = IceParticleAssembly(specimen, config)

    z = assembly.positions[:, 2]
    grip = 1.4 * spec.particle_radius
    top = z >= z.max() - grip
    bottom = z <= z.min() + grip
    dt = assembly.dt
    speed = spec.step_displacement / dt
    velocity = torch.zeros(assembly.n, 3, dtype=torch.float64)
    velocity[top, 2] = -speed
    assembly.kinematic = top | bottom
    assembly.kinematic_velocity = velocity.clone()

    z_top0 = float(assembly.positions[top, 2].mean())
    z_bot0 = float(assembly.positions[bottom, 2].mean())
    h0 = z_top0 - z_bot0
    z_cut = 0.5 * (z_top0 + z_bot0)
    area0 = math.pi * (spec.diameter / 2.0) ** 2
    strain_target = spec.strain_target
    strain, stress, times, killed = [], [], [], []
    n_steps = int(round(strain_target * h0 / spec.step_displacement))
    peak = 0.0
    sigma_mc_guess = mohr_coulomb_ucs(spec.cohesion_pa, spec.friction_angle_deg)
    for step in range(n_steps):
        if step < spec.ramp_steps:
            scale = step / float(spec.ramp_steps)
            assembly.kinematic_velocity[top, 2] = -speed * scale
        assembly.step()
        closure = z_top0 - float(assembly.positions[top, 2].mean())
        eps = closure / h0
        sigma = assembly.section_force(z_cut) / area0
        strain.append(eps)
        stress.append(sigma)
        times.append(assembly.time)
        killed.append(assembly.killed_bonds)
        peak = max(peak, sigma)
        if assembly.killed_bonds > 0 and peak > 0.3 * sigma_mc_guess and sigma < 0.5 * peak:
            break

    strain = np.asarray(strain)
    stress = np.asarray(stress)
    smooth = np.convolve(stress, np.ones(5) / 5.0, mode="same")
    sigma_peak = float(np.max(smooth))
    sigma_mc = target_ucs
    modulus = analysis.fit_linear_modulus(strain, stress)
    e_pred = affine_modulus_prediction(specimen, config)
    diag = assembly.diagnostics()
    first_break = int(np.argmax(np.asarray(killed) > 0)) if max(killed) > 0 else None
    return {
        "spec": spec.__dict__,
        "geometry": validate_cylinder(specimen.positions, specimen.radii,
                                      spec.diameter, specimen.height),
        "n_particles": specimen.n_particles,
        "n_bonds": specimen.n_bonds,
        "orientation": orientation_coverage(specimen, spec.friction_angle_deg),
        "coordination": float(torch.bincount(
            torch.cat([specimen.bond_i, specimen.bond_j]),
            minlength=specimen.n_particles).double().mean()),
        "sigma_peak_Pa": sigma_peak,
        "sigma_peak_raw_Pa": float(np.max(stress)),
        "sigma_mc_Pa": sigma_mc,
        "bond_cohesion_pa": bond_cohesion,
        "fabric_mc": probe_strength,
        "sigma_error": (abs(sigma_peak - sigma_mc) / sigma_mc) if sigma_mc else float("nan"),
        "E_measured_Pa": modulus["E_Pa"],
        "E_r2": modulus["r2"],
        "E_affine_Pa": e_pred,
        "E_error": (abs(modulus["E_Pa"] - e_pred) / e_pred) if e_pred else float("nan"),
        "strain_at_peak": float(strain[int(np.argmax(smooth))]),
        "first_break_step": first_break,
        "killed_bonds": assembly.killed_bonds,
        "dt": dt,
        "n_steps": len(strain),
        "energy": diag,
        "energy_residual_relative": diag["energy_residual_relative"],
        "strain": strain.tolist(),
        "stress": stress.tolist(),
        "smoothed_stress": smooth.tolist(),
        "time": times,
        "killed_history": killed,
    }


# ----------------------------------------------------------------------
# C3 -- pressure-area law from punch indentation
# ----------------------------------------------------------------------
@dataclass
class IndentationSpec:
    punch_widths: tuple[float, ...] = (0.04, 0.08, 0.12, 0.16)
    layer_width: float = 0.28
    layer_depth: float = 0.28
    layer_height: float = 0.05
    particle_radius: float = 0.010
    bond_modulus: float = 9.0e9
    cohesion_pa: float = 1.2e6
    friction_angle_deg: float = 30.0
    tensile_strength_pa: float = 0.9e6
    weibull_modulus: float = 4.0
    seed: int = 0
    penetration: float = 0.02
    step_displacement: float = 1.0e-6
    ramp_steps: int = 300
    density_scale: float = 10.0
    punch_stiffness: float = 5.0e9
    punch_friction: float = 0.0
    floor_friction: float = 0.0
    scale: float = 1.0            # geometric scale factor (self-similarity check)


def _punch_case(spec: IndentationSpec, width: float,
                contact_friction: float = 0.2) -> dict[str, Any]:
    """Flat punch indentation with a *kinematic* punch (no penalty wall).

    The punch is the block of particles inside the punch footprint on the top
    surface, driven at a prescribed velocity (the same kinematic path the
    uniaxial grips use, which is stable by construction); the bottom layer is
    fixed.  The reaction is read from the bond forces the interior exerts on the
    punch particles, so the peak pressure is measured, not prescribed.
    """
    _pin_threads()
    lam = spec.scale
    layer = ice_layer(width=spec.layer_width * lam,
                      depth=spec.layer_depth * lam,
                      height=spec.layer_height * lam,
                      particle_radius=spec.particle_radius * lam,
                      seed=spec.seed)
    config = AssemblyConfig(
        bond_modulus=spec.bond_modulus,
        cohesion_pa=spec.cohesion_pa,
        friction_angle_deg=spec.friction_angle_deg,
        tensile_strength_pa=spec.tensile_strength_pa,
        bond_strength_weibull_modulus=spec.weibull_modulus,
        bond_strength_seed=spec.seed,
        density_scale=spec.density_scale,
        contact_friction=contact_friction,
        store_history=False,
    )
    assembly = IceParticleAssembly(layer, config)
    pos = assembly.positions
    cx = 0.5 * (pos[:, 0].min() + pos[:, 0].max())
    cy = 0.5 * (pos[:, 1].min() + pos[:, 1].max())
    half = 0.5 * width * lam
    z = pos[:, 2]
    footprint = ((pos[:, 0] - cx).abs() <= half) & ((pos[:, 1] - cy).abs() <= half)
    top_band = z >= z.max() - 1.5 * float(layer.radii.max())
    punch = footprint & top_band
    substrate = z <= z.min() + 1.5 * float(layer.radii.max())
    velocity = torch.zeros(assembly.n, 3, dtype=torch.float64)
    speed = spec.step_displacement * lam / assembly.dt
    velocity[punch, 2] = -speed
    assembly.kinematic = punch | substrate
    assembly.kinematic_velocity = velocity
    punch_area = (2.0 * half) ** 2
    depth_target = spec.penetration * lam
    n_steps = int(round(depth_target / (spec.step_displacement * lam)))
    peak = 0.0
    curve = []
    z0 = float(pos[punch, 2].mean())
    ramp_steps = int(spec.ramp_steps)
    for step in range(n_steps):
        if step < ramp_steps:                      # smooth start: no shock wave
            assembly.kinematic_velocity[punch, 2] = -speed * step / float(ramp_steps)
        assembly.step()
        forces = assembly.forces()
        reaction = abs(float(forces[punch][:, 2].sum()))
        pressure = reaction / punch_area
        depth = z0 - float(assembly.positions[punch, 2].mean())
        peak = max(peak, pressure)
        curve.append((depth, pressure, reaction))
        if (depth > 0.2 * depth_target and assembly.killed_bonds > 0
                and pressure < 0.5 * peak):
            break
    return {
        "punch_width_m": float(2.0 * half),
        "punch_area_m2": punch_area,
        "peak_pressure_Pa": peak,
        "peak_force_N": max(c[2] for c in curve) if curve else 0.0,
        "n_particles": layer.n_particles,
        "n_bonds": layer.n_bonds,
        "n_punch_particles": int(punch.sum()),
        "killed_bonds": assembly.killed_bonds,
        "contact_friction": contact_friction,
        "energy_residual_relative": assembly.diagnostics()["energy_residual_relative"],
        "n_steps": len(curve),
        "curve": [(float(a), float(b), float(c)) for a, b, c in curve],
        "geometry": {"layer_height_m": layer.height, "layer_volume_m3": layer.volume},
    }


def run_indentation(spec: IndentationSpec = IndentationSpec()) -> dict[str, Any]:
    """Pressure-area law from flat-punch indentation at several punch areas."""
    cases = [_punch_case(spec, w) for w in spec.punch_widths]
    areas = [c["punch_area_m2"] for c in cases]
    pressures = [c["peak_pressure_Pa"] for c in cases]
    fit = analysis.pressure_area_exponent(areas, pressures)
    return {"spec": {k: v for k, v in spec.__dict__.items()},
            "cases": cases, "areas": areas, "pressures": pressures, "fit": fit,
            "exponent": fit["exponent"], "exponent_error_vs_m05": fit["exponent_error"],
            "r2": fit["r2"]}


@dataclass
class ResistanceSpec:
    """Friction sensitivity of the ice crushing resistance (indentation proxy).

    ``R_ind`` is the peak indentation force of a fixed punch into the bonded ice
    layer; ``mu`` is the Coulomb friction of the *ruptured* contacts (the layer is
    bonded, so friction acts only after the bonds fail).  This is an indentation
    resistance, not a ship-scale ice resistance R_ice -- the difference is stated
    in the report, and the sensitivity (a ratio) is the transferred quantity.
    """

    punch_width: float = 0.10
    layer_width: float = 0.24
    layer_depth: float = 0.24
    layer_height: float = 0.05
    particle_radius: float = 0.010
    bond_modulus: float = 9.0e9
    cohesion_pa: float = 1.2e6
    friction_angle_deg: float = 30.0
    tensile_strength_pa: float = 0.9e6
    weibull_modulus: float = 4.0
    seed: int = 0
    penetration: float = 0.010
    step_displacement: float = 2.0e-6
    density_scale: float = 10.0
    friction_coefficients: tuple[float, ...] = (0.1, 0.3, 0.5)


def run_ice_resistance(spec: ResistanceSpec = ResistanceSpec()) -> dict[str, Any]:
    """Run the indentation proxy at several contact friction coefficients."""
    cases = []
    for mu in spec.friction_coefficients:
        result = _punch_case(IndentationSpec(
            punch_widths=(spec.punch_width,),
            layer_width=spec.layer_width, layer_depth=spec.layer_depth,
            layer_height=spec.layer_height, particle_radius=spec.particle_radius,
            bond_modulus=spec.bond_modulus, cohesion_pa=spec.cohesion_pa,
            friction_angle_deg=spec.friction_angle_deg,
            tensile_strength_pa=spec.tensile_strength_pa,
            weibull_modulus=spec.weibull_modulus, seed=spec.seed,
            penetration=spec.penetration, step_displacement=spec.step_displacement,
            density_scale=spec.density_scale), spec.punch_width, contact_friction=mu)
        result["mu"] = mu
        result["R_ind_N"] = result["peak_force_N"]
        cases.append(result)
    base, top = cases[0], cases[-1]
    change = (abs(top["R_ind_N"] - base["R_ind_N"]) / base["R_ind_N"]
              if base["R_ind_N"] else float("nan"))
    fit = analysis.fit_power_law(spec.friction_coefficients[:len(cases)],
                                 [c["R_ind_N"] for c in cases])
    return {"spec": spec.__dict__, "cases": cases,
            "base_mu": base["mu"], "top_mu": top["mu"],
            "R_ind_base_N": base["R_ind_N"], "R_ind_top_N": top["R_ind_N"],
            "R_ind_change": change, "trend_fit": fit}


def force_chain_stats(forces: np.ndarray) -> dict[str, float]:
    """Exponential-tail statistics of a contact-force sample."""
    return analysis.exponential_tail_fit(forces)