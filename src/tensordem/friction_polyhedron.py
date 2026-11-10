"""Coulomb friction + rolling resistance contact model for polyhedron DEM.

This module unifies the existing TensorDEM contact primitives into a single
contact model suitable for arbitrary convex polyhedra:

1. **Normal contact**: linear spring-dashpot (Hertz-like with effective
   curvature from GJK/EPA penetration depth).
2. **Tangential contact**: Cundall-Strack incremental spring with Coulomb
   saturation |F_t| <= mu * |F_n|.
3. **Rolling resistance**: EPSD (elastic-plastic spring-damper) model from
   Ai et al. 2011, with torque cap mu_r * r_eff * |F_n|.

The geometry (gap, penetration depth, contact normal, contact point) is
provided by the GJK/EPA engine in ``convex_contact``.

Physics background
------------------
For granular materials (ice rubble, rock, sand), the macroscopic angle of
repose theta_r is controlled by the microscopic friction coefficients:
    tan(theta_r) ~ mu  (sliding friction)
    tan(theta_r) ~ mu_r (rolling resistance, for spherical particles)

For polyhedra, the shape interlocking adds an effective friction that depends
on the particle geometry. The GJK/EPA contact detection captures this
geometrically without any shape-specific tuning.

References
----------
- Ai et al. (2011). Assessment of rolling resistance models in DEM.
  Powder Technology 206, 269-282.
- Cundall & Strack (1979). A discrete numerical model for granular assemblies.
  Geotechnique 29, 47-65.
- Gilbert, Johnson & Keerthi (1988). A fast procedure for computing the
  distance between complex objects in three-dimensional space.
- van den Bergen (2003). A fast and robust GJK implementation for collision
  detection of convex objects.

All quantities SI: m, kg, s, N, Pa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .convex_contact import (
    ConvexShape,
    Polytope,
    Sphere,
    Box,
    gjk_distance,
    epa_penetration,
    contact as gjk_epa_contact,
)

__all__ = [
    "FrictionPolyhedronConfig",
    "ContactState",
    "compute_contact_force",
    "compute_wall_contact_force",
    "PolyhedronParticle",
    "PolyhedronDEMSystem",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrictionPolyhedronConfig:
    """Contact model parameters for Coulomb friction + rolling resistance.

    Attributes:
        k_n: Normal stiffness (N/m).
        gamma_n: Normal damping coefficient (N*s/m).
        k_t: Tangential stiffness (N/m). Typically k_t ~ (2/7)*k_n.
        mu: Coulomb friction coefficient (dimensionless).
        mu_r: Rolling resistance coefficient (dimensionless).
        k_r: Rolling spring stiffness (N*m/rad). Required for EPSD model.
        e: Coefficient of restitution (alternative to gamma_n).
        density: Particle density (kg/m^3), used for mass computation.
    """
    k_n: float = 1.0e5
    gamma_n: float = 100.0
    k_t: float = 2.0e4  # ~ (2/7) * k_n
    mu: float = 0.3
    mu_r: float = 0.1
    k_r: float = 1.0e2
    e: Optional[float] = None
    density: float = 2500.0  # rock/ice typical

    def __post_init__(self):
        if self.k_n <= 0:
            raise ValueError("k_n must be positive")
        if self.mu < 0:
            raise ValueError("mu must be non-negative")
        if self.mu_r < 0:
            raise ValueError("mu_r must be non-negative")


# ---------------------------------------------------------------------------
# Contact state (per-pair tangential + rolling history)
# ---------------------------------------------------------------------------
@dataclass
class ContactState:
    """Mutable state for one contact pair (tangential displacement + rolling spring)."""
    delta_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rolling_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = True

    def reset(self):
        self.delta_t[:] = 0.0
        self.rolling_moment[:] = 0.0
        self.active = True


# ---------------------------------------------------------------------------
# Contact force computation (particle-particle)
# ---------------------------------------------------------------------------
def compute_contact_force(
    shape_a: ConvexShape,
    shape_b: ConvexShape,
    v_a: np.ndarray,
    v_b: np.ndarray,
    omega_a: np.ndarray,
    omega_b: np.ndarray,
    m_a: float,
    m_b: float,
    I_a: np.ndarray,
    I_b: np.ndarray,
    state: ContactState,
    config: FrictionPolyhedronConfig,
    dt: float,
    r_eff: Optional[float] = None,
) -> Tuple[dict, bool]:
    """Compute contact force and torque for one pair of convex shapes.

    Uses GJK for gap detection and EPA for penetration when overlapping.

    Returns:
        (result_dict, is_contacting)
        result_dict contains:
            - 'force_a': force on particle a (3,)
            - 'force_b': force on particle b (3,)
            - 'torque_a': torque on particle a (3,)
            - 'torque_b': torque on particle b (3,)
            - 'normal': contact normal from a to b (3,)
            - 'overlap': penetration depth (m)
            - 'f_n': normal force magnitude (N)
            - 'f_t': tangential force magnitude (N)
            - 'm_r': rolling resistance moment magnitude (N*m)
            - 'contact_point': point of contact (3,)
            - 'energy_elastic': elastic potential energy stored (J)
            - 'energy_dissipated': energy dissipated this step (J)
    """
    # --- Geometry via GJK/EPA ---
    geom = gjk_epa_contact(shape_a, shape_b)

    if geom.separated and geom.distance > 0.0:
        state.reset()
        state.active = False
        return {
            'force_a': np.zeros(3), 'force_b': np.zeros(3),
            'torque_a': np.zeros(3), 'torque_b': np.zeros(3),
            'normal': geom.normal, 'overlap': 0.0,
            'f_n': 0.0, 'f_t': 0.0, 'm_r': 0.0,
            'contact_point': 0.5 * (geom.point_a + geom.point_b),
            'energy_elastic': 0.0, 'energy_dissipated': 0.0,
        }, False

    # Overlapping: get penetration info
    overlap = geom.depth if not geom.separated else 0.0
    normal = geom.normal  # points from a to b

    # Effective radius for rolling resistance
    if r_eff is None:
        r_eff = max(shape_a.bounding_radius(), shape_b.bounding_radius()) * 0.5

    # Contact point (midpoint of closest pair)
    contact_pt = 0.5 * (geom.point_a + geom.point_b)

    # --- Relative velocity at contact point ---
    r_a = contact_pt - shape_a.centroid()
    r_b = contact_pt - shape_b.centroid()
    v_contact_a = v_a + np.cross(omega_a, r_a)
    v_contact_b = v_b + np.cross(omega_b, r_b)
    v_rel = v_contact_b - v_contact_a

    # Normal and tangential components
    v_n = np.dot(v_rel, normal) * normal
    v_t = v_rel - v_n

    # --- Normal force (spring-dashpot) ---
    f_n_spring = config.k_n * overlap
    f_n_damping = config.gamma_n * max(0.0, -np.dot(v_rel, normal))  # only during approach
    f_n_mag = f_n_spring + f_n_damping
    f_n_vec = f_n_mag * normal  # repulsive, pushes a away from b

    # --- Tangential force (Cundall-Strack) ---
    # Integrate tangential displacement
    state.delta_t += v_t * dt
    # Project onto tangent plane
    state.delta_t -= np.dot(state.delta_t, normal) * normal

    f_t_vec = -config.k_t * state.delta_t
    # Coulomb saturation
    f_t_mag = np.linalg.norm(f_t_vec)
    f_t_limit = config.mu * f_n_mag
    if f_t_mag > f_t_limit:
        # Plastic slip: project onto yield surface
        if f_t_limit > 0:
            f_t_vec = f_t_vec * (f_t_limit / f_t_mag)
        else:
            f_t_vec = np.zeros(3)  # mu=0: no tangential force
        # Update delta_t to be consistent with projected force
        if config.k_t > 0:
            state.delta_t = -f_t_vec / config.k_t

    # --- Rolling resistance (EPSD) ---
    omega_rel = omega_b - omega_a
    # Rolling component (perpendicular to normal)
    omega_roll = omega_rel - np.dot(omega_rel, normal) * normal

    # EPSD update: M_r^{n+1} = M_r^n - k_r * omega_roll * dt
    state.rolling_moment -= config.k_r * omega_roll * dt
    # Plastic clamp
    m_r_mag = np.linalg.norm(state.rolling_moment)
    m_r_limit = config.mu_r * r_eff * f_n_mag
    if m_r_limit > 0 and m_r_mag > m_r_limit:
        state.rolling_moment *= (m_r_limit / m_r_mag)

    # Rolling torque acts on b; -torque on a
    torque_r = state.rolling_moment

    # --- Assemble forces ---
    force_on_a = -f_n_vec + f_t_vec  # normal pushes a away, tangential opposes slip
    force_on_b = f_n_vec - f_t_vec
    torque_on_a = np.cross(r_a, -f_n_vec + f_t_vec) - torque_r
    torque_on_b = np.cross(r_b, f_n_vec - f_t_vec) + torque_r

    # --- Energy accounting ---
    energy_elastic = 0.5 * config.k_n * overlap**2 + 0.5 * config.k_t * np.dot(state.delta_t, state.delta_t)
    energy_dissipated = f_n_damping * max(0.0, -np.dot(v_rel, normal)) * dt

    state.active = True

    return {
        'force_a': force_on_a,
        'force_b': force_on_b,
        'torque_a': torque_on_a,
        'torque_b': torque_on_b,
        'normal': normal,
        'overlap': overlap,
        'f_n': f_n_mag,
        'f_t': np.linalg.norm(f_t_vec),
        'm_r': np.linalg.norm(torque_r),
        'contact_point': contact_pt,
        'energy_elastic': energy_elastic,
        'energy_dissipated': energy_dissipated,
    }, True


# ---------------------------------------------------------------------------
# Wall contact (simplified: direct plane distance, no GJK)
# ---------------------------------------------------------------------------
def compute_wall_contact_force(
    shape: ConvexShape,
    v: np.ndarray,
    omega: np.ndarray,
    m: float,
    I: np.ndarray,
    wall_normal: np.ndarray,
    wall_offset: float,
    state: ContactState,
    config: FrictionPolyhedronConfig,
    dt: float,
) -> Tuple[dict, bool]:
    """Compute contact force between a particle and a planar wall.

    Uses direct distance computation (no GJK) for efficiency.

    Args:
        shape: Particle shape.
        v: Particle velocity.
        omega: Particle angular velocity.
        m: Particle mass.
        I: Particle inertia tensor.
        wall_normal: Wall normal (pointing into domain).
        wall_offset: Wall plane offset (n.x = offset).
        state: Contact state.
        config: Contact parameters.
        dt: Time step.

    Returns:
        (result_dict, is_contacting)
    """
    n = wall_normal
    center = shape.centroid()
    r_bound = shape.bounding_radius()

    # Distance from particle center to wall
    dist = np.dot(center, n) - wall_offset
    overlap = r_bound - dist

    if overlap <= 0:
        state.reset()
        state.active = False
        return {
            'force': np.zeros(3), 'torque': np.zeros(3),
            'normal': n, 'overlap': 0.0,
            'f_n': 0.0, 'f_t': 0.0, 'm_r': 0.0,
            'contact_point': center - r_bound * n,
            'energy_elastic': 0.0, 'energy_dissipated': 0.0,
        }, False

    # Contact point
    contact_pt = center - (dist) * n

    # Relative velocity at contact (wall is stationary)
    r = contact_pt - center
    v_contact = v + np.cross(omega, r)

    # Normal and tangential components
    v_n = np.dot(v_contact, n) * n
    v_t = v_contact - v_n

    # Normal force
    f_n_spring = config.k_n * overlap
    f_n_damping = config.gamma_n * max(0.0, -np.dot(v_contact, n))
    f_n_mag = f_n_spring + f_n_damping
    f_n_vec = f_n_mag * n

    # Tangential force (Cundall-Strack)
    state.delta_t += v_t * dt
    state.delta_t -= np.dot(state.delta_t, n) * n
    f_t_vec = -config.k_t * state.delta_t
    f_t_mag = np.linalg.norm(f_t_vec)
    f_t_limit = config.mu * f_n_mag
    if f_t_limit > 0 and f_t_mag > f_t_limit:
        f_t_vec = f_t_vec * (f_t_limit / f_t_mag)
        state.delta_t = -f_t_vec / config.k_t

    # Rolling resistance
    omega_roll = omega - np.dot(omega, n) * n
    state.rolling_moment -= config.k_r * omega_roll * dt
    m_r_mag = np.linalg.norm(state.rolling_moment)
    m_r_limit = config.mu_r * r_bound * f_n_mag
    if m_r_limit > 0 and m_r_mag > m_r_limit:
        state.rolling_moment *= (m_r_limit / m_r_mag)
    torque_r = state.rolling_moment

    # Assemble forces (normal pushes particle away from wall, i.e., in direction of n)
    force = f_n_vec + f_t_vec
    torque = np.cross(r, force) - torque_r

    # Energy
    energy_elastic = 0.5 * config.k_n * overlap**2 + 0.5 * config.k_t * np.dot(state.delta_t, state.delta_t)
    energy_dissipated = f_n_damping * max(0.0, -np.dot(v_contact, n)) * dt

    state.active = True

    return {
        'force': force,
        'torque': torque,
        'normal': n,
        'overlap': overlap,
        'f_n': f_n_mag,
        'f_t': np.linalg.norm(f_t_vec),
        'm_r': np.linalg.norm(torque_r),
        'contact_point': contact_pt,
        'energy_elastic': energy_elastic,
        'energy_dissipated': energy_dissipated,
    }, True


# ---------------------------------------------------------------------------
# Polyhedron particle
# ---------------------------------------------------------------------------
@dataclass
class PolyhedronParticle:
    """A single polyhedron particle with state.

    Attributes:
        shape: ConvexShape (Polytope, Sphere, or Box).
        mass: Particle mass (kg).
        inertia: Inertia tensor (3,3) in world frame.
        x: Position (3,).
        v: Velocity (3,).
        omega: Angular velocity (3,).
        label: Optional identifier.
    """
    shape: ConvexShape
    mass: float
    inertia: np.ndarray
    x: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    label: str = ""

    @classmethod
    def from_polytope(cls, vertices: np.ndarray, density: float = 2500.0,
                      position: Optional[np.ndarray] = None,
                      label: str = "") -> 'PolyhedronParticle':
        """Create a polyhedron particle from vertices."""
        verts = np.asarray(vertices, dtype=np.float64)
        shape = Polytope(verts)

        # Compute mass properties from convex hull
        volume = _convex_hull_volume(verts)
        mass = density * volume
        inertia = _convex_hull_inertia(verts, mass)

        x = np.asarray(position if position is not None else shape.centroid(), dtype=np.float64)
        return cls(shape=shape, mass=mass, inertia=inertia,
                   x=x, v=np.zeros(3), omega=np.zeros(3), label=label)

    @classmethod
    def from_sphere(cls, radius: float, density: float = 2500.0,
                    position: Optional[np.ndarray] = None,
                    label: str = "") -> 'PolyhedronParticle':
        """Create a sphere particle."""
        center = np.asarray(position if position is not None else [0, 0, 0], dtype=np.float64)
        shape = Sphere(center.tolist(), radius)
        volume = (4.0 / 3.0) * math.pi * radius**3
        mass = density * volume
        I_diag = 0.4 * mass * radius**2
        inertia = np.diag([I_diag, I_diag, I_diag])
        return cls(shape=shape, mass=mass, inertia=inertia,
                   x=center.copy(), v=np.zeros(3), omega=np.zeros(3), label=label)

    @classmethod
    def from_box(cls, half_extents: np.ndarray, density: float = 2500.0,
                 position: Optional[np.ndarray] = None,
                 orientation: Optional[np.ndarray] = None,
                 label: str = "") -> 'PolyhedronParticle':
        """Create a box particle."""
        he = np.asarray(half_extents, dtype=np.float64)
        center = np.asarray(position if position is not None else [0, 0, 0], dtype=np.float64)
        shape = Box(center.tolist(), he.tolist(), orientation)
        volume = 8.0 * np.prod(he)
        mass = density * volume
        I_xx = mass / 3.0 * (he[1]**2 + he[2]**2)
        I_yy = mass / 3.0 * (he[0]**2 + he[2]**2)
        I_zz = mass / 3.0 * (he[0]**2 + he[1]**2)
        inertia = np.diag([I_xx, I_yy, I_zz])
        return cls(shape=shape, mass=mass, inertia=inertia,
                   x=center.copy(), v=np.zeros(3), omega=np.zeros(3), label=label)


def _convex_hull_volume(vertices: np.ndarray) -> float:
    """Compute volume of convex hull via divergence theorem."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except ImportError:
        # Fallback: bounding box estimate (underestimates)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        return float(np.prod(extents) * 0.5)  # rough


def _convex_hull_inertia(vertices: np.ndarray, mass: float) -> np.ndarray:
    """Compute inertia tensor of convex hull (approximate)."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        # Use vertex-based approximation
        center = vertices.mean(axis=0)
        r2 = np.sum((vertices - center)**2, axis=1)
        I_diag = mass * np.mean(r2) * 0.4  # ~ solid sphere equivalent
        return np.diag([I_diag, I_diag, I_diag])
    except ImportError:
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        r2 = np.sum(extents**2)
        I_diag = mass * r2 * 0.1
        return np.diag([I_diag, I_diag, I_diag])


# ---------------------------------------------------------------------------
# DEM System
# ---------------------------------------------------------------------------
class PolyhedronDEMSystem:
    """DEM system for polyhedron particles with Coulomb friction + rolling resistance.

    Integrates Newton-Euler equations with contact forces from GJK/EPA geometry.
    """

    def __init__(self, particles: list, config: FrictionPolyhedronConfig,
                 gravity: np.ndarray = None):
        self.particles = particles
        self.config = config
        self.gravity = np.asarray(gravity if gravity is not None else [0, 0, -9.81])
        self.contact_states: dict = {}  # (i, j) -> ContactState
        self.walls: list = []  # list of (normal, offset) for planar walls
        self.wall_states: dict = {}  # (i, wall_idx) -> ContactState

        # Energy ledger
        self.ke_trans = 0.0
        self.ke_rot = 0.0
        self.pe_gravity = 0.0
        self.pe_elastic = 0.0
        self.energy_dissipated = 0.0
        self.energy_initial = 0.0

    def add_wall(self, normal: np.ndarray, offset: float):
        """Add a planar wall: normal . x = offset (particles on normal side)."""
        self.walls.append((np.asarray(normal, dtype=np.float64), float(offset)))

    def add_box_container(self, center: np.ndarray, half_extents: np.ndarray):
        """Add 6 walls forming a box container."""
        c = np.asarray(center)
        h = np.asarray(half_extents)
        for axis in range(3):
            n = np.zeros(3)
            n[axis] = 1.0
            self.add_wall(n, float(c[axis] - h[axis]))  # lower wall
            self.add_wall(-n, float(-(c[axis] + h[axis])))  # upper wall

    def _get_contact_state(self, key: tuple) -> ContactState:
        if key not in self.contact_states:
            self.contact_states[key] = ContactState()
        return self.contact_states[key]

    def _get_wall_state(self, key: tuple) -> ContactState:
        if key not in self.wall_states:
            self.wall_states[key] = ContactState()
        return self.wall_states[key]

    def step(self, dt: float):
        """Advance one timestep with velocity Verlet integration."""
        n_part = len(self.particles)

        # --- Compute forces ---
        forces = [np.zeros(3) for _ in range(n_part)]
        torques = [np.zeros(3) for _ in range(n_part)]
        total_elastic_energy = 0.0
        step_dissipation = 0.0

        # Particle-particle contacts
        for i in range(n_part):
            for j in range(i + 1, n_part):
                pi, pj = self.particles[i], self.particles[j]
                state = self._get_contact_state((i, j))
                result, contacting = compute_contact_force(
                    pi.shape, pj.shape,
                    pi.v, pj.v, pi.omega, pj.omega,
                    pi.mass, pj.mass, pi.inertia, pj.inertia,
                    state, self.config, dt,
                )
                if contacting:
                    forces[i] += result['force_a']
                    forces[j] += result['force_b']
                    torques[i] += result['torque_a']
                    torques[j] += result['torque_b']
                    total_elastic_energy += result['energy_elastic']
                    step_dissipation += result['energy_dissipated']

        # Wall contacts (using direct plane distance, no GJK)
        for i in range(n_part):
            pi = self.particles[i]
            for w_idx, (n_wall, d_wall) in enumerate(self.walls):
                state = self._get_wall_state((i, w_idx))
                result, contacting = compute_wall_contact_force(
                    pi.shape, pi.v, pi.omega,
                    pi.mass, pi.inertia,
                    n_wall, d_wall,
                    state, self.config, dt,
                )
                if contacting:
                    forces[i] += result['force']
                    torques[i] += result['torque']
                    total_elastic_energy += result['energy_elastic']
                    step_dissipation += result['energy_dissipated']

        # --- Integrate (velocity Verlet) ---
        for i, p in enumerate(self.particles):
            # Acceleration
            a = forces[i] / p.mass + self.gravity
            alpha = np.linalg.solve(p.inertia, torques[i])

            # Update velocity
            p.v += a * dt
            p.omega += alpha * dt

            # Update position
            p.x += p.v * dt

            # Update shape centroid (for Sphere/Box, update center)
            if isinstance(p.shape, Sphere):
                p.shape = Sphere(p.x.tolist(), p.shape.radius)
            elif isinstance(p.shape, Box):
                p.shape = Box(p.x.tolist(), p.shape.half_extents.tolist(), p.shape.orientation)
            elif isinstance(p.shape, Polytope):
                # Translate vertices
                delta = p.v * dt
                new_verts = p.shape.vertices + delta
                p.shape = Polytope(new_verts)

        # --- Energy ledger ---
        self.ke_trans = sum(0.5 * p.mass * np.dot(p.v, p.v) for p in self.particles)
        self.ke_rot = sum(0.5 * np.dot(p.omega, p.inertia @ p.omega) for p in self.particles)
        self.pe_gravity = sum(-p.mass * np.dot(self.gravity, p.x) for p in self.particles)
        self.pe_elastic = total_elastic_energy
        self.energy_dissipated += step_dissipation

    def total_energy(self) -> float:
        """Total mechanical energy (should be conserved minus dissipation)."""
        return self.ke_trans + self.ke_rot + self.pe_gravity + self.pe_elastic + self.energy_dissipated

    def energy_balance_error(self) -> float:
        """Relative energy balance error: |E_total - E_initial| / |E_initial|."""
        e_total = self.total_energy()
        if abs(self.energy_initial) < 1e-15:
            return 0.0
        return abs(e_total - self.energy_initial) / abs(self.energy_initial)

    def record_initial_energy(self):
        """Record current total energy as reference for balance check."""
        self.energy_initial = self.total_energy()


# ---------------------------------------------------------------------------
# Utility: generate random convex polyhedra
# ---------------------------------------------------------------------------
def random_polyhedron(center: np.ndarray, mean_radius: float,
                      n_vertices: int = 12, seed: Optional[int] = None) -> np.ndarray:
    """Generate vertices of a random convex polyhedron.

    Uses random points on a sphere scaled by mean_radius, then takes convex hull.
    """
    rng = np.random.default_rng(seed)
    # Random directions on unit sphere
    directions = rng.standard_normal((n_vertices, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Scale by radius with some variation
    radii = mean_radius * (0.7 + 0.6 * rng.random(n_vertices))
    vertices = center + directions * radii[:, np.newaxis]
    return vertices


def regular_polyhedron(name: str, center: np.ndarray, size: float) -> np.ndarray:
    """Generate vertices of a regular polyhedron.

    Supported: 'tetrahedron', 'cube', 'octahedron', 'icosahedron', 'dodecahedron'.
    """
    c = np.asarray(center, dtype=np.float64)
    s = float(size)

    if name == 'tetrahedron':
        verts = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=np.float64) * s / np.sqrt(3)
    elif name == 'cube':
        verts = np.array(list(__import__('itertools').product([-1, 1], repeat=3)),
                         dtype=np.float64) * s
    elif name == 'octahedron':
        verts = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ], dtype=np.float64) * s
    elif name == 'icosahedron':
        phi = (1 + math.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        ], dtype=np.float64) * s / math.sqrt(1 + phi**2)
    elif name == 'dodecahedron':
        phi = (1 + math.sqrt(5)) / 2
        inv_phi = 1.0 / phi
        verts = []
        # Cube vertices
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    verts.append([sx, sy, sz])
        # Rectangle vertices
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                verts.append([0, s1 * inv_phi, s2 * phi])
                verts.append([s1 * inv_phi, s2 * phi, 0])
                verts.append([s2 * phi, 0, s1 * inv_phi])
        verts = np.array(verts, dtype=np.float64) * s / math.sqrt(3)
    else:
        raise ValueError(f"Unknown polyhedron: {name}")

    return verts + c
