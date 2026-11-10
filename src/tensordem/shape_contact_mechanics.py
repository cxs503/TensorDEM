"""Shape-aware contact mechanics for polyhedra DEM.

Implements contact force computation using GJK/EPA for polyhedra-polyhedra
and polyhedra-wall contacts, with:
- Normal force (Hertzian or linear elastic with crushing cap)
- Tangential force (Coulomb friction with regularisation)
- Rolling resistance torque (geometric or viscous)

This module bridges the gap between the generic GJK/EPA contact geometry
(convex_contact.py) and the DEM dynamics engine, providing force-level
outputs that can be compared between sphere and polyhedra models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .convex_contact import (
    ContactGeometry,
    ConvexShape,
    Polytope,
    Sphere,
    contact,
    gjk_distance,
    epa_penetration,
)
from .polyhedra_shapes import (
    PolyhedronSpec,
    ShapeDescriptor,
    compute_shape_descriptor,
    scale_to_volume,
)

__all__ = [
    "ContactModel",
    "ContactForce",
    "ContactResult",
    "ShapeContactEngine",
    "compute_contact_forces",
    "polyhedron_from_spec",
    "sphere_from_volume",
]


@dataclass
class ContactModel:
    """Contact model parameters (SI units)."""
    # Normal
    k_n: float = 1.0e5          # Normal stiffness [N/m]
    zeta_n: float = 0.3         # Normal damping ratio [-]
    sigma_crush: float = 5.0e6  # Crushing strength cap [Pa]
    use_hertz: bool = False     # Use Hertzian contact (vs linear)
    E_star: float = 5.0e9       # Effective modulus for Hertz [Pa]

    # Tangential
    mu: float = 0.4             # Friction coefficient [-]
    k_t: float = 0.0            # Tangential stiffness (0 = use k_n) [N/m]
    v_slip: float = 0.01        # Regularisation velocity [m/s]

    # Rolling
    mu_r: float = 0.0           # Rolling friction coefficient [-]
    rolling_model: str = "viscous"  # "viscous" or "geometric"
    omega_smooth: float = 1.0   # Rolling regularisation [rad/s]

    # Material
    rho: float = 917.0          # Density [kg/m^3]

    def validate(self) -> None:
        for name in ("k_n", "sigma_crush", "rho"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("zeta_n", "mu", "mu_r"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be non-negative and finite")


@dataclass
class ContactForce:
    """Contact force decomposition at a single contact point."""
    # Geometry
    gap: float = 0.0            # Gap (negative = penetration) [m]
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Contact normal
    point: np.ndarray = field(default_factory=lambda: np.zeros(3))   # Contact point
    area: float = 0.0           # Contact area [m^2]

    # Forces
    f_normal: float = 0.0       # Normal force magnitude [N]
    f_tangential: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Tangential force [N]
    f_total: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Total force [N]

    # Torques
    torque_rolling: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Rolling torque [N·m]

    # Diagnostics
    is_contacting: bool = False
    penetration_depth: float = 0.0  # |gap| when gap < 0


@dataclass
class ContactResult:
    """Full contact result between two bodies."""
    force_a: ContactForce = field(default_factory=ContactForce)
    force_b: ContactForce = field(default_factory=ContactForce)
    # Summary scalars
    normal_force_mag: float = 0.0
    tangential_force_mag: float = 0.0
    rolling_torque_mag: float = 0.0
    total_force_mag: float = 0.0


def polyhedron_from_spec(
    spec: PolyhedronSpec,
    target_volume: float,
    center: np.ndarray = None,
    orientation: np.ndarray = None,
) -> Polytope:
    """Create a Polytope shape from a PolyhedronSpec, scaled to target volume."""
    if center is None:
        center = np.zeros(3)
    verts = scale_to_volume(spec.vertices, target_volume)
    # Centre at origin (spec vertices are already centred)
    verts = verts - verts.mean(axis=0)
    # Apply orientation
    if orientation is not None:
        verts = verts @ orientation.T
    # Translate to center
    verts = verts + center
    return Polytope(vertices=verts)


def sphere_from_volume(
    volume: float,
    center: np.ndarray = None,
) -> Sphere:
    """Create a Sphere shape from volume."""
    if center is None:
        center = np.zeros(3)
    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    return Sphere(center=center, radius=radius)


def _contact_area_linear(penetration: float, r_eff: float) -> float:
    """Linear contact area: A = π * r_eff * δ"""
    return math.pi * r_eff * max(penetration, 0.0)


def _contact_area_hertz(penetration: float, r_eff: float) -> float:
    """Hertzian contact area: A = π * r_eff * δ"""
    return math.pi * r_eff * max(penetration, 0.0)


def _effective_radius(r_a: float, r_b: float) -> float:
    """Effective radius for two spheres: 1/R = 1/R_a + 1/R_b"""
    if r_a <= 0 or r_b <= 0:
        return max(r_a, r_b)
    return r_a * r_b / (r_a + r_b)


def compute_contact_forces(
    shape_a: ConvexShape,
    shape_b: ConvexShape,
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    omega_a: np.ndarray,
    omega_b: np.ndarray,
    mass_a: float,
    mass_b: float,
    model: ContactModel,
    tangential_history: Optional[np.ndarray] = None,
) -> ContactResult:
    """Compute contact forces between two convex shapes.

    Uses GJK for gap detection and EPA for penetration when overlapping.
    Forces are computed using linear elastic + damping model with crushing cap.

    Parameters
    ----------
    shape_a, shape_b : ConvexShape
        The two bodies in contact.
    vel_a, vel_b : ndarray (3,)
        Linear velocities of body centres.
    omega_a, omega_b : ndarray (3,)
        Angular velocities of bodies.
    mass_a, mass_b : float
        Masses of bodies [kg].
    model : ContactModel
        Contact model parameters.
    tangential_history : ndarray (3,), optional
        Previous tangential displacement for elastic tangential model.

    Returns
    -------
    ContactResult
        Full contact force decomposition.
    """
    result = ContactResult()

    # Step 1: Contact geometry via GJK
    geom = contact(shape_a, shape_b)

    if not not geom.separated and geom.distance > 1e-10:
        # No contact
        return result

    # Step 2: Penetration depth
    if not geom.separated:
        penetration = geom.depth
    else:
        penetration = -geom.distance

    if penetration <= 0:
        return result

    # Step 3: Effective properties
    r_a = shape_a.bounding_radius()
    r_b = shape_b.bounding_radius()
    r_eff = _effective_radius(r_a, r_b)

    # Contact area (linear model)
    area = _contact_area_linear(penetration, r_eff)

    # Step 4: Normal force
    # Relative velocity at contact point
    r_ca = geom.point_a - shape_a.centroid()
    r_cb = geom.point_b - shape_b.centroid()
    v_contact_a = vel_a + np.cross(omega_a, r_ca)
    v_contact_b = vel_b + np.cross(omega_b, r_cb)
    v_rel = v_contact_a - v_contact_b

    normal = geom.normal
    v_n = float(np.dot(v_rel, normal))  # Normal relative velocity (positive = separating)

    # Effective mass
    m_eff = mass_a * mass_b / (mass_a + mass_b) if (mass_a + mass_b) > 0 else mass_a

    # Critical damping coefficient
    c_crit = 2.0 * math.sqrt(model.k_n * m_eff)
    c_n = 2.0 * model.zeta_n * math.sqrt(model.k_n * m_eff)

    # Linear elastic + damping
    f_n_elastic = model.k_n * penetration
    f_n_damping = c_n * max(-v_n, 0.0)  # Only resist compression
    f_n = f_n_elastic + f_n_damping

    # Apply crushing cap (pressure-based)
    if area > 0:
        pressure = f_n / area
        if pressure > model.sigma_crush:
            f_n = model.sigma_crush * area

    # Step 5: Tangential force (regularised Coulomb)
    v_t = v_rel - v_n * normal
    v_t_mag = float(np.linalg.norm(v_t))

    if v_t_mag > 1e-14:
        t_dir = v_t / v_t_mag
        # Regularised friction: F_t = μ F_n tanh(|v_t| / v_slip)
        f_t_mag = model.mu * f_n * math.tanh(v_t_mag / model.v_slip)
        f_tangential = -f_t_mag * t_dir
    else:
        f_tangential = np.zeros(3)

    # Step 6: Rolling resistance torque
    omega_rel = omega_a - omega_b
    omega_mag = float(np.linalg.norm(omega_rel))

    if omega_mag > 1e-14 and model.mu_r > 0:
        omega_dir = omega_rel / omega_mag
        # Rolling torque: M_r = -μ_r * R * F_n * ω̂_rel
        torque_mag = model.mu_r * r_eff * f_n * math.tanh(omega_mag / model.omega_smooth)
        torque_rolling = -torque_mag * omega_dir
    else:
        torque_rolling = np.zeros(3)

    # Step 7: Assemble result
    f_normal_vec = f_n * normal
    f_total = f_normal_vec + f_tangential

    result.force_a = ContactForce(
        gap=geom.distance,
        normal=normal,
        point=geom.point_a,
        area=area,
        f_normal=f_n,
        f_tangential=f_tangential,
        f_total=f_total,
        torque_rolling=torque_rolling,
        is_contacting=True,
        penetration_depth=penetration,
    )
    result.force_b = ContactForce(
        gap=geom.distance,
        normal=-normal,
        point=geom.point_b,
        area=area,
        f_normal=f_n,
        f_tangential=-f_tangential,
        f_total=-f_total,
        torque_rolling=-torque_rolling,
        is_contacting=True,
        penetration_depth=penetration,
    )

    result.normal_force_mag = f_n
    result.tangential_force_mag = float(np.linalg.norm(f_tangential))
    result.rolling_torque_mag = float(np.linalg.norm(torque_rolling))
    result.total_force_mag = float(np.linalg.norm(f_total))

    return result


class ShapeContactEngine:
    """High-level engine for polyhedra DEM with shape-aware contacts.

    This engine manages a collection of particles (spheres or polyhedra) and
    computes all pairwise contacts using GJK/EPA.
    """

    def __init__(
        self,
        shapes: list[ConvexShape],
        masses: np.ndarray,
        model: ContactModel = None,
    ):
        self.shapes = list(shapes)
        self.n = len(self.shapes)
        self.masses = np.asarray(masses, dtype=np.float64)
        self.model = model or ContactModel()
        self.model.validate()

        # State
        self.positions = np.array([s.centroid() for s in self.shapes])
        self.velocities = np.zeros((self.n, 3))
        self.omegas = np.zeros((self.n, 3))
        self.forces = np.zeros((self.n, 3))
        self.torques = np.zeros((self.n, 3))

        # Contact history
        self.contact_forces: list[ContactResult] = []
        self.total_normal_force = 0.0
        self.total_tangential_force = 0.0
        self.total_rolling_torque = 0.0

    def compute_all_contacts(self) -> list[ContactResult]:
        """Compute all pairwise contacts and accumulate forces."""
        self.forces[:] = 0.0
        self.torques[:] = 0.0
        self.contact_forces = []
        self.total_normal_force = 0.0
        self.total_tangential_force = 0.0
        self.total_rolling_torque = 0.0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Broad phase: bounding sphere check
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                r_sum = self.shapes[i].bounding_radius() + self.shapes[j].bounding_radius()
                if dist > r_sum * 1.1:  # 10% skin
                    continue

                # Narrow phase: GJK/EPA
                result = compute_contact_forces(
                    self.shapes[i], self.shapes[j],
                    self.velocities[i], self.velocities[j],
                    self.omegas[i], self.omegas[j],
                    self.masses[i], self.masses[j],
                    self.model,
                )

                if result.force_a.is_contacting:
                    self.forces[i] += result.force_a.f_total
                    self.forces[j] += result.force_b.f_total
                    self.torques[i] += result.force_a.torque_rolling
                    self.torques[j] += result.force_b.torque_rolling
                    self.contact_forces.append(result)
                    self.total_normal_force += result.normal_force_mag
                    self.total_tangential_force += result.tangential_force_mag
                    self.total_rolling_torque += result.rolling_torque_mag

        return self.contact_forces

    def step(self, dt: float, gravity: np.ndarray = None) -> None:
        """Advance one time step with velocity Verlet integration."""
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])

        # Compute contacts
        self.compute_all_contacts()

        # Add gravity
        for i in range(self.n):
            self.forces[i] += self.masses[i] * gravity

        # Update velocities (half step)
        for i in range(self.n):
            self.velocities[i] += 0.5 * dt * self.forces[i] / self.masses[i]
            self.omegas[i] += 0.5 * dt * self.torques[i] / (self.masses[i] * 0.4 * self.shapes[i].bounding_radius()**2)

        # Update positions
        self.positions += dt * self.velocities

        # Rebuild shapes at new positions
        self._rebuild_shapes()

        # Recompute contacts for second half-step
        self.compute_all_contacts()

        # Complete velocity update
        for i in range(self.n):
            self.velocities[i] += 0.5 * dt * self.forces[i] / self.masses[i]
            self.omegas[i] += 0.5 * dt * self.torques[i] / (self.masses[i] * 0.4 * self.shapes[i].bounding_radius()**2)

    def _rebuild_shapes(self) -> None:
        """Rebuild shape objects at current positions (for spheres/polytopes)."""
        for i, shape in enumerate(self.shapes):
            if isinstance(shape, Sphere):
                self.shapes[i] = Sphere(center=self.positions[i], radius=shape.radius)
            elif isinstance(shape, Polytope):
                # Translate vertices
                old_centroid = shape.vertices.mean(axis=0)
                new_verts = shape.vertices - old_centroid + self.positions[i]
                self.shapes[i] = Polytope(vertices=new_verts)
