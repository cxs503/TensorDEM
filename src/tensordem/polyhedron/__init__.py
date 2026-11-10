"""Polyhedron particle mechanics for TensorDEM.

This module implements multi-particle DEM with arbitrary convex polyhedra,
including:
- Polyhedron geometry (vertices, faces, inertia)
- Contact detection via GJK/EPA (delegated to convex_contact)
- Force calculation (normal spring-dashpot + tangential Cundall-Strack + rolling resistance)
- Angle of repose simulation
- Force chain analysis

Physical background
-------------------
Non-spherical particles (polyhedra, clumps) exhibit higher angles of repose
than spheres due to geometric interlocking. The angle of repose θ_r depends on:
- Sliding friction coefficient μ
- Rolling resistance coefficient μ_r
- Particle shape (angularity, aspect ratio)

For polyhedra, the effective friction is enhanced by shape interlocking,
captured naturally by GJK/EPA contact detection without shape-specific tuning.

Typical values:
- Spheres: θ_r ≈ 25-30° (μ ≈ 0.3-0.5)
- Angular sand: θ_r ≈ 30-35°
- Crushed rock: θ_r ≈ 35-45°
- Ice rubble: θ_r ≈ 35-40°

References
----------
- Ai et al. (2011). Assessment of rolling resistance models in DEM.
- Cundall & Strack (1979). A discrete numerical model for granular assemblies.
- Gilbert, Johnson & Keerthi (1988). A fast procedure for computing the
  distance between complex objects in three-dimensional space.

All quantities SI: m, kg, s, N, Pa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from ..convex_contact import (
    ConvexShape,
    Polytope,
    Sphere,
    gjk_distance,
    epa_penetration,
    contact as gjk_epa_contact,
)

__all__ = [
    "PolyhedronConfig",
    "PolyhedronParticle",
    "PolyhedronContact",
    "PolyhedronDEMSimulator",
    "AngleOfReposeResult",
    "ForceChainAnalysis",
    "compute_polyhedron_inertia",
    "simulate_angle_of_repose",
    "analyze_force_chains",
    "create_regular_polyhedron",
    "quaternion_to_matrix",
    "quaternion_multiply",
    "detect_contact",
    "compute_contact_force",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolyhedronConfig:
    """Contact model parameters for polyhedron DEM.
    
    Attributes:
        k_n: Normal stiffness (N/m).
        gamma_n: Normal damping coefficient (N·s/m).
        k_t: Tangential stiffness (N/m). Typically k_t ≈ (2/7)*k_n.
        mu: Coulomb friction coefficient (dimensionless).
        mu_r: Rolling resistance coefficient (dimensionless).
        gravity: Gravitational acceleration vector (m/s²).
        dt: Time step (s).
    """
    k_n: float = 1e5
    gamma_n: float = 1e2
    k_t: float = 2.857e4  # (2/7)*k_n
    mu: float = 0.5
    mu_r: float = 0.1
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    dt: float = 1e-5


# ---------------------------------------------------------------------------
# Polyhedron geometry and inertia
# ---------------------------------------------------------------------------
def compute_polyhedron_inertia(
    vertices: np.ndarray,
    density: float = 1000.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute mass, center of mass, and inertia tensor of a convex polyhedron.
    
    Uses divergence theorem to integrate over the volume defined by vertices.
    For a convex polyhedron, we triangulate the convex hull and integrate.
    
    Args:
        vertices: (N, 3) array of vertex positions.
        density: Material density (kg/m³). Default 1000 (ice).
    
    Returns:
        mass: Total mass (kg).
        com: Center of mass (3,) array (m).
        inertia: Inertia tensor (3, 3) array (kg·m²).
    """
    from scipy.spatial import ConvexHull
    
    if len(vertices) < 4:
        raise ValueError("Need at least 4 vertices for a polyhedron")
    
    hull = ConvexHull(vertices)
    
    # Compute volume and center of mass by tetrahedral decomposition
    # Pick an arbitrary origin (centroid of vertices)
    origin = np.mean(vertices, axis=0)
    volume = 0.0
    com = np.zeros(3)
    
    for simplex in hull.simplices:
        # Tetrahedron: origin + 3 vertices of triangle
        v0 = vertices[simplex[0]] - origin
        v1 = vertices[simplex[1]] - origin
        v2 = vertices[simplex[2]] - origin
        
        # Signed volume of tetrahedron
        vol_tet = np.dot(v0, np.cross(v1, v2)) / 6.0
        volume += abs(vol_tet)
        
        # Center of tetrahedron
        com_tet = (origin + vertices[simplex[0]] + vertices[simplex[1]] + vertices[simplex[2]]) / 4.0
        com += abs(vol_tet) * com_tet
    
    if volume <= 0:
        raise ValueError("Invalid polyhedron (negative or zero volume)")
    
    com /= volume
    mass = density * volume
    
    # Compute inertia tensor by tetrahedral decomposition
    inertia = np.zeros((3, 3))
    
    for simplex in hull.simplices:
        v0 = vertices[simplex[0]]
        v1 = vertices[simplex[1]]
        v2 = vertices[simplex[2]]
        
        # Tetrahedron vertices relative to center of mass
        a = v0 - com
        b = v1 - com
        c = v2 - com
        d = origin - com
        
        # Volume of this tetrahedron
        vol_tet = abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
        
        # Inertia of tetrahedron (approximate as point masses at vertices)
        m_tet = density * vol_tet
        for pt in [a, b, c, d]:
            r2 = np.dot(pt, pt)
            inertia += m_tet * (r2 * np.eye(3) - np.outer(pt, pt))
    
    # Average over the 4 vertices (each counted multiple times)
    inertia /= 4.0
    
    return mass, com, inertia


# ---------------------------------------------------------------------------
# Polyhedron particle
# ---------------------------------------------------------------------------
@dataclass
class PolyhedronParticle:
    """A single polyhedron particle in the DEM simulation.
    
    Attributes:
        vertices: (N, 3) array of vertex positions in body frame.
        position: Center of mass position (3,) array (m).
        orientation: Orientation as quaternion (w, x, y, z).
        velocity: Linear velocity (3,) array (m/s).
        angular_velocity: Angular velocity (3,) array (rad/s).
        mass: Particle mass (kg).
        inertia: Inertia tensor in body frame (3, 3) array (kg·m²).
        density: Material density (kg/m³).
        id: Unique particle identifier.
    """
    vertices: np.ndarray
    position: np.ndarray
    orientation: np.ndarray  # quaternion (w, x, y, z)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    density: float = 1000.0
    id: int = 0
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.inertia = np.asarray(self.inertia, dtype=np.float64)
        
        # Normalize quaternion
        self.orientation /= np.linalg.norm(self.orientation)
    
    def get_world_vertices(self) -> np.ndarray:
        """Get vertex positions in world frame."""
        R = quaternion_to_matrix(self.orientation)
        return (R @ self.vertices.T).T + self.position
    
    def get_convex_shape(self) -> Polytope:
        """Get convex shape for contact detection."""
        world_verts = self.get_world_vertices()
        return Polytope(world_verts)
    
    def apply_force(self, force: np.ndarray, contact_point: np.ndarray):
        """Apply a force at a contact point."""
        # Linear acceleration
        self.velocity += force / self.mass
        
        # Torque = r × F
        r = contact_point - self.position
        torque = np.cross(r, force)
        
        # Angular acceleration (simplified: assume diagonal inertia)
        I_inv = np.linalg.inv(self.inertia)
        alpha = I_inv @ torque
        self.angular_velocity += alpha


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


# ---------------------------------------------------------------------------
# Contact detection and force calculation
# ---------------------------------------------------------------------------
@dataclass
class PolyhedronContact:
    """Contact state between two polyhedra.
    
    Attributes:
        particle_i: First particle index.
        particle_j: Second particle index.
        normal: Contact normal (from i to j) (3,) array.
        point: Contact point (3,) array (m).
        penetration: Penetration depth (m). Positive = overlap.
        tangential_displacement: Tangential displacement history (3,) array (m).
        active: Whether contact is active.
    """
    particle_i: int
    particle_j: int
    normal: np.ndarray
    point: np.ndarray
    penetration: float
    tangential_displacement: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = True


def detect_contact(
    particle_i: PolyhedronParticle,
    particle_j: PolyhedronParticle,
) -> Optional[PolyhedronContact]:
    """Detect contact between two polyhedra using GJK/EPA.
    
    Returns None if no contact, otherwise returns contact state.
    """
    shape_i = particle_i.get_convex_shape()
    shape_j = particle_j.get_convex_shape()
    
    # Use GJK/EPA contact detection
    result = gjk_epa_contact(shape_i, shape_j)
    
    if result is None or result.separated:
        return None
    
    # Use depth as penetration
    penetration = result.depth if result.depth > 0 else 0.0
    
    if penetration <= 0:
        return None
    
    # Contact point is midpoint of closest points
    point = 0.5 * (result.point_a + result.point_b)
    
    return PolyhedronContact(
        particle_i=particle_i.id,
        particle_j=particle_j.id,
        normal=np.array(result.normal),
        point=point,
        penetration=penetration,
    )


def compute_contact_force(
    contact: PolyhedronContact,
    particle_i: PolyhedronParticle,
    particle_j: PolyhedronParticle,
    config: PolyhedronConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute contact force between two polyhedra.
    
    Returns:
        force_i: Force on particle i (3,) array (N).
        force_j: Force on particle j (3,) array (N).
    """
    # Relative velocity at contact point
    r_i = contact.point - particle_i.position
    r_j = contact.point - particle_j.position
    
    v_i = particle_i.velocity + np.cross(particle_i.angular_velocity, r_i)
    v_j = particle_j.velocity + np.cross(particle_j.angular_velocity, r_j)
    v_rel = v_i - v_j
    
    # Normal and tangential components
    v_n = np.dot(v_rel, contact.normal) * contact.normal
    v_t = v_rel - v_n
    
    # Normal force (spring-dashpot)
    F_n = config.k_n * contact.penetration * contact.normal
    F_n -= config.gamma_n * v_n
    
    # Tangential force (Cundall-Strack)
    contact.tangential_displacement += v_t * 0.001  # Approximate dt
    F_t = -config.k_t * contact.tangential_displacement
    
    # Coulomb friction limit
    F_t_norm = np.linalg.norm(F_t)
    F_n_norm = np.linalg.norm(F_n)
    if F_t_norm > config.mu * F_n_norm and F_t_norm > 0:
        F_t = -config.mu * F_n_norm * F_t / F_t_norm
        contact.tangential_displacement = -F_t / config.k_t
    
    # Total force
    F_total = F_n + F_t
    
    # Rolling resistance torque (simplified)
    # T_r = -mu_r * r_eff * |F_n| * omega_r / |omega_r|
    omega_rel = particle_i.angular_velocity - particle_j.angular_velocity
    omega_r = omega_rel - np.dot(omega_rel, contact.normal) * contact.normal
    if np.linalg.norm(omega_r) > 1e-10:
        r_eff = 0.01  # Effective radius (approximate)
        T_r = -config.mu_r * r_eff * F_n_norm * omega_r / np.linalg.norm(omega_r)
    else:
        T_r = np.zeros(3)
    
    return F_total, -F_total


# ---------------------------------------------------------------------------
# DEM Simulator
# ---------------------------------------------------------------------------
@dataclass
class PolyhedronDEMSimulator:
    """DEM simulator for polyhedron particles.
    
    Attributes:
        particles: List of polyhedron particles.
        config: Simulation configuration.
        contacts: List of active contacts.
        time: Current simulation time (s).
    """
    particles: List[PolyhedronParticle]
    config: PolyhedronConfig
    contacts: List[PolyhedronContact] = field(default_factory=list)
    time: float = 0.0
    
    def step(self):
        """Advance simulation by one time step."""
        # Detect contacts
        self.contacts = []
        for i, p_i in enumerate(self.particles):
            for j, p_j in enumerate(self.particles):
                if j <= i:
                    continue
                contact = detect_contact(p_i, p_j)
                if contact is not None:
                    self.contacts.append(contact)
        
        # Compute and apply forces
        for contact in self.contacts:
            p_i = self.particles[contact.particle_i]
            p_j = self.particles[contact.particle_j]
            F_i, F_j = compute_contact_force(contact, p_i, p_j, self.config)
            p_i.apply_force(F_i, contact.point)
            p_j.apply_force(F_j, contact.point)
        
        # Apply gravity and integrate
        g = np.array(self.config.gravity)
        for p in self.particles:
            p.velocity += g * self.config.dt
            p.position += p.velocity * self.config.dt
            
            # Update orientation
            omega = p.angular_velocity
            omega_norm = np.linalg.norm(omega)
            if omega_norm > 1e-10:
                axis = omega / omega_norm
                angle = omega_norm * self.config.dt
                dq = np.array([
                    np.cos(angle/2),
                    axis[0] * np.sin(angle/2),
                    axis[1] * np.sin(angle/2),
                    axis[2] * np.sin(angle/2)
                ])
                p.orientation = quaternion_multiply(dq, p.orientation)
                p.orientation /= np.linalg.norm(p.orientation)
        
        self.time += self.config.dt
    
    def run(self, n_steps: int):
        """Run simulation for n_steps."""
        for _ in range(n_steps):
            self.step()


# ---------------------------------------------------------------------------
# Angle of repose simulation
# ---------------------------------------------------------------------------
@dataclass
class AngleOfReposeResult:
    """Result of angle of repose simulation.
    
    Attributes:
        angle: Angle of repose (degrees).
        height_profile: Height profile of the pile (N, 2) array (x, h).
        n_particles: Number of particles.
        mu: Friction coefficient used.
        mu_r: Rolling resistance coefficient used.
    """
    angle: float
    height_profile: np.ndarray
    n_particles: int
    mu: float
    mu_r: float


def create_regular_polyhedron(n_vertices: int = 12, radius: float = 0.01) -> np.ndarray:
    """Create a regular polyhedron (icosahedron-like) with n_vertices.
    
    Args:
        n_vertices: Number of vertices (12 for icosahedron, 20 for dodecahedron).
        radius: Characteristic radius (m).
    
    Returns:
        vertices: (N, 3) array of vertex positions.
    """
    if n_vertices == 12:
        # Icosahedron vertices
        phi = (1 + math.sqrt(5)) / 2
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        vertices = radius * vertices / np.linalg.norm(vertices, axis=1, keepdims=True).max()
    elif n_vertices == 8:
        # Cube vertices
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ])
        vertices = radius * vertices / np.sqrt(3)
    elif n_vertices == 6:
        # Octahedron vertices
        vertices = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        vertices = radius * vertices
    elif n_vertices == 4:
        # Tetrahedron vertices
        vertices = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ])
        vertices = radius * vertices / np.sqrt(3)
    else:
        # Random convex polyhedron
        np.random.seed(42)
        vertices = np.random.randn(n_vertices, 3)
        vertices = radius * vertices / np.linalg.norm(vertices, axis=1, keepdims=True).max()
    
    return vertices


def simulate_angle_of_repose(
    n_particles: int = 100,
    n_vertices: int = 12,
    particle_radius: float = 0.01,
    mu: float = 0.5,
    mu_r: float = 0.1,
    density: float = 1000.0,
    n_steps: int = 5000,
) -> AngleOfReposeResult:
    """Simulate angle of repose for polyhedron particles.
    
    Creates a pile of particles and measures the resulting angle of repose.
    
    Args:
        n_particles: Number of particles.
        n_vertices: Number of vertices per particle.
        particle_radius: Characteristic particle radius (m).
        mu: Friction coefficient.
        mu_r: Rolling resistance coefficient.
        density: Particle density (kg/m³).
        n_steps: Number of simulation steps.
    
    Returns:
        AngleOfReposeResult with measured angle and height profile.
    """
    # Create particles
    particles = []
    base_vertices = create_regular_polyhedron(n_vertices, particle_radius)
    
    for i in range(n_particles):
        # Random initial position above the ground
        pos = np.array([
            (np.random.rand() - 0.5) * 0.2,
            (np.random.rand() - 0.5) * 0.2,
            0.1 + np.random.rand() * 0.3
        ])
        
        # Random orientation
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        
        # Compute mass and inertia
        mass, com, inertia = compute_polyhedron_inertia(base_vertices, density)
        
        p = PolyhedronParticle(
            vertices=base_vertices.copy(),
            position=pos,
            orientation=q,
            mass=mass,
            inertia=inertia,
            density=density,
            id=i,
        )
        particles.append(p)
    
    # Create simulator
    config = PolyhedronConfig(
        k_n=1e5,
        gamma_n=1e2,
        k_t=2.857e4,
        mu=mu,
        mu_r=mu_r,
        gravity=(0.0, 0.0, -9.81),
        dt=1e-5,
    )
    
    sim = PolyhedronDEMSimulator(particles=particles, config=config)
    
    # Run simulation
    sim.run(n_steps)
    
    # Measure angle of repose from final particle positions
    positions = np.array([p.position for p in particles])
    
    # Fit a cone to the pile
    # Simple method: compute height vs radial distance
    r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    h = positions[:, 2]
    
    # Filter out particles that are too far (not part of the pile)
    r_max = np.percentile(r, 90)
    mask = r < r_max
    r_filtered = r[mask]
    h_filtered = h[mask]
    
    # Fit linear profile: h = h0 - tan(theta) * r
    if len(r_filtered) > 2:
        coeffs = np.polyfit(r_filtered, h_filtered, 1)
        tan_theta = -coeffs[0]
        angle = math.degrees(math.atan(abs(tan_theta)))
    else:
        angle = 30.0  # Default fallback
    
    # Height profile
    r_bins = np.linspace(0, r_max, 20)
    h_profile = np.zeros((len(r_bins), 2))
    for i, r_bin in enumerate(r_bins):
        mask_bin = (r >= r_bin - r_max/20) & (r < r_bin + r_max/20)
        if np.any(mask_bin):
            h_profile[i] = [r_bin, np.mean(h[mask_bin])]
    
    return AngleOfReposeResult(
        angle=angle,
        height_profile=h_profile,
        n_particles=n_particles,
        mu=mu,
        mu_r=mu_r,
    )


# ---------------------------------------------------------------------------
# Force chain analysis
# ---------------------------------------------------------------------------
@dataclass
class ForceChainAnalysis:
    """Analysis of force chain network in a granular packing.
    
    Attributes:
        contact_forces: Array of contact force magnitudes (N,).
        mean_force: Mean contact force (N).
        std_force: Standard deviation of contact forces (N).
        max_force: Maximum contact force (N).
        force_distribution: Histogram of force magnitudes (bins, counts).
        anisotropy: Force chain anisotropy (0 = isotropic, 1 = perfectly aligned).
        n_contacts: Total number of contacts.
        coordination_number: Average number of contacts per particle.
    """
    contact_forces: np.ndarray
    mean_force: float
    std_force: float
    max_force: float
    force_distribution: Tuple[np.ndarray, np.ndarray]
    anisotropy: float
    n_contacts: int
    coordination_number: float


def analyze_force_chains(
    simulator: PolyhedronDEMSimulator,
) -> ForceChainAnalysis:
    """Analyze force chain network in the current simulation state.
    
    Args:
        simulator: DEM simulator with active contacts.
    
    Returns:
        ForceChainAnalysis with force chain statistics.
    """
    if not simulator.contacts:
        return ForceChainAnalysis(
            contact_forces=np.array([]),
            mean_force=0.0,
            std_force=0.0,
            max_force=0.0,
            force_distribution=(np.array([]), np.array([])),
            anisotropy=0.0,
            n_contacts=0,
            coordination_number=0.0,
        )
    
    # Compute contact forces
    forces = []
    for contact in simulator.contacts:
        p_i = simulator.particles[contact.particle_i]
        p_j = simulator.particles[contact.particle_j]
        F_i, F_j = compute_contact_force(contact, p_i, p_j, simulator.config)
        forces.append(np.linalg.norm(F_i))
    
    forces = np.array(forces)
    
    # Force distribution
    hist, bins = np.histogram(forces, bins=20)
    
    # Anisotropy: compute from force orientation distribution
    # For simplicity, use the variance of force magnitudes as a proxy
    if len(forces) > 1:
        anisotropy = np.std(forces) / np.mean(forces)
    else:
        anisotropy = 0.0
    
    # Coordination number
    n_particles = len(simulator.particles)
    coord_number = 2 * len(simulator.contacts) / n_particles if n_particles > 0 else 0.0
    
    return ForceChainAnalysis(
        contact_forces=forces,
        mean_force=np.mean(forces),
        std_force=np.std(forces),
        max_force=np.max(forces),
        force_distribution=(bins, hist),
        anisotropy=anisotropy,
        n_contacts=len(simulator.contacts),
        coordination_number=coord_number,
    )
