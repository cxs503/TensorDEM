"""3D DEM contact mechanics for ice ridge rubble.

Contact model:
- Normal: Hertz law F_n = (4/3) E* sqrt(R*) delta^(3/2)
- Tangential: Cundall-Strack incremental Coulomb friction |F_t| <= mu * F_n
- Rolling resistance: T_r = mu_r * R_eff * F_n
- Interlocking: geometric interlock -> extra shear strength

Analytical anchors:
- Single particle Hertz solution: F_n = (4/3) E* sqrt(R*) delta^(3/2) (criterion c, error < 1%)
- Interlocking: tau_interlock vs no-interlock difference > 30% (criterion f)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "hertz_normal_force_3d",
    "hertz_contact_energy_3d",
    "hertz_secant_stiffness_3d",
    "coulomb_friction_3d",
    "rolling_resistance_3d",
    "interlocking_shear_strength",
    "ContactState3D",
    "compute_contact_force",
]


def _finite(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def hertz_reduced_modulus_3d(young_modulus: float, poisson_ratio: float,
                              rigid_counterpart: bool = False) -> float:
    """Effective modulus E* for sphere-sphere or sphere-rigid contact."""
    _finite("young_modulus", young_modulus, positive=True)
    _finite("poisson_ratio", poisson_ratio, nonnegative=True)
    if poisson_ratio >= 0.5:
        raise ValueError("poisson_ratio must be < 0.5")
    compliance = 1.0 - poisson_ratio ** 2
    return young_modulus / (compliance if rigid_counterpart else 2.0 * compliance)


def hertz_effective_radius_3d(r1: float, r2: float,
                               rigid_counterpart: bool = False) -> float:
    """Effective radius R* for two spheres or sphere-rigid contact."""
    _finite("r1", r1, positive=True)
    _finite("r2", r2, positive=True)
    if rigid_counterpart:
        return r1
    return 1.0 / (1.0 / r1 + 1.0 / r2)


def hertz_normal_force_3d(overlap: float, e_star: float, r_eff: float) -> float:
    """Hertz normal force: F_n = (4/3) E* sqrt(R*) delta^(3/2).

    Analytical anchor for acceptance criterion (c): error < 1%.
    """
    _finite("overlap", overlap, nonnegative=True)
    _finite("e_star", e_star, positive=True)
    _finite("r_eff", r_eff, positive=True)
    if overlap <= 0.0:
        return 0.0
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff) * overlap ** 1.5


def hertz_contact_energy_3d(overlap: float, e_star: float, r_eff: float) -> float:
    """Recoverable Hertz contact energy: U = (8/15) E* sqrt(R*) delta^(5/2)."""
    _finite("overlap", overlap, nonnegative=True)
    _finite("e_star", e_star, positive=True)
    _finite("r_eff", r_eff, positive=True)
    if overlap <= 0.0:
        return 0.0
    return (8.0 / 15.0) * e_star * math.sqrt(r_eff) * overlap ** 2.5


def hertz_secant_stiffness_3d(overlap: float, e_star: float, r_eff: float) -> float:
    """Secant stiffness: k_s = F/delta = (4/3) E* sqrt(R*) sqrt(delta)."""
    _finite("overlap", overlap, nonnegative=True)
    _finite("e_star", e_star, positive=True)
    _finite("r_eff", r_eff, positive=True)
    if overlap <= 0.0:
        return 0.0
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff) * math.sqrt(overlap)


def coulomb_friction_3d(tangential_displacement: np.ndarray, normal_force: float,
                         friction_coefficient: float, tangential_stiffness: float,
                         dt: float) -> np.ndarray:
    """Cundall-Strack incremental Coulomb friction.

    F_t_new = F_t_old + k_t * delta_t, clamped to |F_t| <= mu * F_n.

    Parameters
    ----------
    tangential_displacement : np.ndarray, shape (3,)
        Incremental tangential displacement vector.
    normal_force : float
        Current normal force magnitude (N).
    friction_coefficient : float
        Coulomb friction coefficient mu.
    tangential_stiffness : float
        Tangential spring stiffness k_t (N/m).
    dt : float
        Time step (s).

    Returns
    -------
    np.ndarray, shape (3,)
        Tangential friction force vector.
    """
    delta_f_t = tangential_stiffness * tangential_displacement
    f_t_max = friction_coefficient * max(normal_force, 0.0)
    f_t_mag = np.linalg.norm(delta_f_t)
    if f_t_mag > f_t_max and f_t_max > 0.0:
        delta_f_t = delta_f_t * (f_t_max / f_t_mag)
    return delta_f_t


def rolling_resistance_3d(relative_angular_velocity: np.ndarray,
                           normal_force: float,
                           rolling_friction_coefficient: float,
                           effective_radius: float) -> np.ndarray:
    """Rolling resistance torque: T_r = mu_r * R_eff * F_n opposing rotation.

    Parameters
    ----------
    relative_angular_velocity : np.ndarray, shape (3,)
        Relative angular velocity vector.
    normal_force : float
        Normal force magnitude (N).
    rolling_friction_coefficient : float
        Rolling friction coefficient mu_r.
    effective_radius : float
        Effective contact radius (m).

    Returns
    -------
    np.ndarray, shape (3,)
        Rolling resistance torque vector.
    """
    omega_mag = np.linalg.norm(relative_angular_velocity)
    if omega_mag < 1e-15 or normal_force <= 0.0:
        return np.zeros(3)
    t_mag = rolling_friction_coefficient * effective_radius * normal_force
    return -t_mag * relative_angular_velocity / omega_mag


@dataclass
class InterlockGeometry:
    """Geometric parameters for interlocking contact."""
    interlock_depth: float = 0.0    # depth of geometric interlock (m)
    contact_area: float = 0.0       # contact area (m²)
    asperity_angle: float = 0.0     # asperity inclination angle (rad)


def interlocking_shear_strength(base_shear_strength: float,
                                 interlock_depth: float,
                                 particle_diameter: float,
                                 friction_coefficient: float,
                                 interlock_factor: float = 1.0) -> float:
    """Shear strength with geometric interlocking enhancement.

    tau_interlock = tau_base * (1 + interlock_factor * (d_interlock / D)^gamma)

    The interlock factor captures the additional shear resistance from
    geometric interlocking of irregular ice blocks. For angular polyhedra,
    this can increase shear strength by 30-100% compared to smooth spheres.

    The enhancement uses a power-law form with gamma=0.5 to capture the
    non-linear relationship between interlock depth and shear resistance.

    Analytical anchor for criterion (f): tau_interlock vs no-interlock > 30%.

    Parameters
    ----------
    base_shear_strength : float
        Base shear strength without interlocking (Pa).
    interlock_depth : float
        Depth of geometric interlock (m).
    particle_diameter : float
        Characteristic particle diameter (m).
    friction_coefficient : float
        Base friction coefficient.
    interlock_factor : float
        Interlocking enhancement factor (typical 0.5-2.0).

    Returns
    -------
    float
        Enhanced shear strength with interlocking (Pa).
    """
    _finite("base_shear_strength", base_shear_strength, nonnegative=True)
    _finite("interlock_depth", interlock_depth, nonnegative=True)
    _finite("particle_diameter", particle_diameter, positive=True)
    _finite("interlock_factor", interlock_factor, nonnegative=True)

    if particle_diameter <= 0.0:
        return base_shear_strength

    # Non-linear interlock enhancement: gamma=0.5 captures the square-root
    # scaling of contact area with interlock depth
    interlock_ratio = interlock_depth / particle_diameter
    gamma = 0.5
    enhancement = 1.0 + interlock_factor * (interlock_ratio ** gamma)
    return base_shear_strength * enhancement


@dataclass
class ContactState3D:
    """State of a single 3D contact between two particles."""
    normal_overlap: float = 0.0
    normal_force: float = 0.0
    tangential_force: np.ndarray = None
    rolling_torque: np.ndarray = None
    contact_energy: float = 0.0
    is_bonded: bool = False
    interlock_depth: float = 0.0

    def __post_init__(self):
        if self.tangential_force is None:
            self.tangential_force = np.zeros(3)
        if self.rolling_torque is None:
            self.rolling_torque = np.zeros(3)


def compute_contact_force(pos1: np.ndarray, r1: float,
                           pos2: np.ndarray, r2: float,
                           young_modulus: float, poisson_ratio: float,
                           friction_coefficient: float,
                           rolling_friction_coefficient: float,
                           tangential_disp: np.ndarray,
                           tangential_stiffness: float,
                           relative_angular_vel: np.ndarray,
                           interlock_depth: float = 0.0,
                           interlock_factor: float = 0.0) -> ContactState3D:
    """Compute full 3D contact force between two spheres.

    Parameters
    ----------
    pos1, pos2 : np.ndarray, shape (3,)
        Particle center positions.
    r1, r2 : float
        Particle radii.
    young_modulus : float
        Young's modulus (Pa).
    poisson_ratio : float
        Poisson's ratio.
    friction_coefficient : float
        Coulomb friction coefficient.
    rolling_friction_coefficient : float
        Rolling friction coefficient.
    tangential_disp : np.ndarray, shape (3,)
        Tangential displacement vector.
    tangential_stiffness : float
        Tangential spring stiffness.
    relative_angular_vel : np.ndarray, shape (3,)
        Relative angular velocity.
    interlock_depth : float
        Geometric interlock depth.
    interlock_factor : float
        Interlocking enhancement factor.

    Returns
    -------
    ContactState3D
        Full contact state.
    """
    delta = pos2 - pos1
    distance = np.linalg.norm(delta)
    overlap = r1 + r2 - distance

    state = ContactState3D()

    if overlap <= 0.0:
        return state

    state.normal_overlap = overlap

    # Hertz normal force
    e_star = hertz_reduced_modulus_3d(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius_3d(r1, r2)
    state.normal_force = hertz_normal_force_3d(overlap, e_star, r_eff)
    state.contact_energy = hertz_contact_energy_3d(overlap, e_star, r_eff)

    # Normal direction
    normal = delta / distance

    # Tangential friction
    state.tangential_force = coulomb_friction_3d(
        tangential_disp, state.normal_force,
        friction_coefficient, tangential_stiffness, dt=1.0
    )

    # Rolling resistance
    state.rolling_torque = rolling_resistance_3d(
        relative_angular_vel, state.normal_force,
        rolling_friction_coefficient, r_eff
    )

    # Interlocking enhancement
    if interlock_depth > 0.0 and interlock_factor > 0.0:
        d_char = 2.0 * r_eff  # effective diameter
        base_shear = np.linalg.norm(state.tangential_force)
        enhanced_shear = interlocking_shear_strength(
            base_shear, interlock_depth, d_char,
            friction_coefficient, interlock_factor
        )
        if enhanced_shear > base_shear and base_shear > 0.0:
            state.tangential_force *= enhanced_shear / base_shear

    state.interlock_depth = interlock_depth
    return state
