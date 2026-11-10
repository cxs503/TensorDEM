"""Polyhedral particle repose angle simulation and validation.

Why this module exists
----------------------
The DEM engine in ``friction_polyhedron`` provides GJK/EPA contact detection,
Coulomb friction, and EPSD rolling resistance for arbitrary convex shapes.
This module wraps those primitives into a standardised **repose-angle protocol**
for polyhedral particles, providing the macro-scale validation anchor for the
micro-scale contact laws.

Protocol
--------
1. Generate N random convex polyhedra (or use regular Platonic solids).
2. Place them in a column above a flat frictional floor inside a cylindrical
   container (or box).
3. Let them fall and settle under gravity.
4. Measure:
   - Repose angle θ from the free-surface profile
   - Volume fraction φ of the deposit
   - Energy balance closure
   - Velocity-time history for comparison with analytical free-fall

Literature references
---------------------
- Zhou et al. (2002). Discrete element simulation of angular particle behavior.
  Powder Technology 125, 45-56. (θ ≈ 30-40° for angular particles, μ=0.3-0.5)
- Zhong et al. (2016). DEM study of repose angle of non-spherical particles.
  Advanced Powder Technology 27, 1077-1086.
- Guo et al. (2019). DEM investigation of the repose angle of irregular particles.
  Granular Matter 21, 42.
- Volume fraction: φ ≈ 0.55-0.65 for random packing of convex polyhedra
  (Williams & Philipse, 2003; Donev et al., 2004).

All quantities SI: m, kg, s, N, Pa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any

import numpy as np

from .friction_polyhedron import (
    FrictionPolyhedronConfig,
    PolyhedronParticle,
    PolyhedronDEMSystem,
    random_polyhedron,
    regular_polyhedron,
)
from .convex_contact import Polytope, Sphere


__all__ = [
    "PolyhedraReposeConfig",
    "PolyhedraReposeSimulation",
    "measure_repose_angle",
    "measure_volume_fraction",
    "run_repose_simulation",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PolyhedraReposeConfig:
    """Configuration for polyhedral repose angle simulation.

    Attributes:
        n_particles: Number of particles to generate.
        particle_size: Mean particle radius (m).
        container_radius: Radius of cylindrical container (m).
        container_height: Height of container (m).
        pour_height: Height from which particles are released (m).
        mu: Coulomb friction coefficient.
        mu_r: Rolling resistance coefficient.
        k_n: Normal stiffness (N/m).
        gamma_n: Normal damping (N*s/m).
        density: Particle density (kg/m^3).
        dt: Time step (s).
        settle_time: Total simulation time for settling (s).
        seed: Random seed for reproducibility.
        shape_type: 'random' for random polyhedra, 'icosahedron', 'cube', etc.
        n_vertices: Number of vertices for random polyhedra.
    """
    n_particles: int = 100
    particle_size: float = 0.005  # 5 mm
    container_radius: float = 0.04  # 40 mm
    container_height: float = 0.15  # 150 mm
    pour_height: float = 0.10  # 100 mm above floor
    mu: float = 0.4
    mu_r: float = 0.05
    k_n: float = 5.0e4
    gamma_n: float = 50.0
    density: float = 2500.0
    dt: float = 1.0e-5
    settle_time: float = 2.0
    seed: int = 42
    shape_type: str = "random"
    n_vertices: int = 12


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------
def measure_repose_angle(
    particles: List[PolyhedronParticle],
    container_radius: float,
    n_radial_bins: int = 10,
) -> dict:
    """Measure the repose angle from the settled particle positions.

    Uses the free-surface profile method: bin particles radially, find the
    maximum height in each bin, fit a line to the flank, θ = atan(-slope).

    Returns:
        dict with keys:
            - 'theta_deg': repose angle in degrees
            - 'slope': fitted slope (dz/dr)
            - 'r_bins': radial bin centers
            - 'h_max': maximum height in each bin
            - 'pile_height': height of pile apex
            - 'pile_radius': radius of pile base
    """
    if not particles:
        return {
            'theta_deg': 0.0, 'slope': 0.0,
            'r_bins': [], 'h_max': [],
            'pile_height': 0.0, 'pile_radius': 0.0,
        }

    # Collect positions and bounding radii
    positions = []
    radii = []
    for p in particles:
        positions.append(p.x.copy())
        radii.append(p.shape.bounding_radius())
    positions = np.array(positions)
    radii = np.array(radii)

    # Use particle centers + bounding radius for surface elevation
    z_surface = positions[:, 2] + radii

    # Radial distance from center (assuming pile centered at origin)
    r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)

    # Find the extent of the pile
    r_max = np.max(r[r < container_radius * 0.95])  # exclude edge effects
    z_min = np.min(z_surface)
    z_max = np.max(z_surface)

    if r_max < 1e-6 or (z_max - z_min) < 1e-6:
        return {
            'theta_deg': 0.0,
            'slope': 0.0,
            'r_bins': [],
            'h_max': [],
            'pile_height': 0.0,
            'pile_radius': 0.0,
        }

    # Bin radially
    r_bins = np.linspace(0, r_max, n_radial_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    h_max = np.zeros(n_radial_bins)

    for i in range(n_radial_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            h_max[i] = np.max(z_surface[mask])
        else:
            h_max[i] = np.nan

    # Fit line to the flank (25%-85% of pile radius)
    valid = ~np.isnan(h_max)
    r_valid = r_centers[valid]
    h_valid = h_max[valid]

    if len(r_valid) < 3:
        return {
            'theta_deg': 0.0,
            'slope': 0.0,
            'r_bins': r_centers.tolist(),
            'h_max': h_max.tolist(),
            'pile_height': float(z_max),
            'pile_radius': float(r_max),
        }

    # Select flank region
    r_flank_min = 0.2 * r_max
    r_flank_max = 0.85 * r_max
    flank_mask = (r_valid >= r_flank_min) & (r_valid <= r_flank_max)

    if np.sum(flank_mask) < 2:
        # Use all valid points
        flank_mask = np.ones(len(r_valid), dtype=bool)

    r_flank = r_valid[flank_mask]
    h_flank = h_valid[flank_mask]

    # Linear fit: h = a + b*r
    if len(r_flank) >= 2:
        coeffs = np.polyfit(r_flank, h_flank, 1)
        slope = coeffs[0]
        theta_rad = math.atan(abs(slope))
        theta_deg = math.degrees(theta_rad)
    else:
        slope = 0.0
        theta_deg = 0.0

    return {
        'theta_deg': theta_deg,
        'slope': float(slope),
        'r_bins': r_centers.tolist(),
        'h_max': h_max.tolist(),
        'pile_height': float(z_max),
        'pile_radius': float(r_max),
    }


def measure_volume_fraction(
    particles: List[PolyhedronParticle],
    container_radius: float,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> dict:
    """Measure the volume fraction of the settled deposit.

    φ = V_particles / V_container_in_deposit_region

    Returns:
        dict with keys:
            - 'phi': volume fraction
            - 'v_particles': total particle volume (m^3)
            - 'v_container': container volume in deposit region (m^3)
            - 'deposit_height': height of deposit region (m)
    """
    # Compute particle volumes (convex hull)
    from scipy.spatial import ConvexHull

    v_particles = 0.0
    z_positions = []
    for p in particles:
        verts = p.shape.vertices
        try:
            hull = ConvexHull(verts)
            v_particles += hull.volume
        except Exception:
            # Fallback: sphere approximation
            r = p.shape.bounding_radius()
            v_particles += (4.0 / 3.0) * math.pi * r**3
        z_positions.append(p.x[2])

    z_positions = np.array(z_positions)

    # Determine deposit region
    if z_min is None:
        z_min = float(np.min(z_positions)) - np.mean([p.shape.bounding_radius() for p in particles])
    if z_max is None:
        z_max = float(np.max(z_positions)) + np.mean([p.shape.bounding_radius() for p in particles])

    deposit_height = z_max - z_min
    v_container = math.pi * container_radius**2 * deposit_height

    phi = v_particles / v_container if v_container > 0 else 0.0

    return {
        'phi': float(phi),
        'v_particles': float(v_particles),
        'v_container': float(v_container),
        'deposit_height': float(deposit_height),
    }


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------
class PolyhedraReposeSimulation:
    """Run a polyhedral repose angle simulation.

    This class manages the full workflow:
    1. Generate particles
    2. Create DEM system with container walls
    3. Run settling simulation
    4. Measure repose angle, volume fraction, energy balance
    """

    def __init__(self, config: PolyhedraReposeConfig):
        self.config = config
        self.particles: List[PolyhedronParticle] = []
        self.system: Optional[PolyhedronDEMSystem] = None
        self.history: dict = {
            'time': [],
            'ke': [],
            'pe': [],
            'dissipated': [],
            'total_energy': [],
            'max_velocity': [],
            'mean_velocity': [],
        }
        self.result: dict = {}

    def generate_particles(self) -> List[PolyhedronParticle]:
        """Generate polyhedral particles arranged in a column above the floor."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        particles = []

        for i in range(cfg.n_particles):
            # Random position in a column above the pour point
            r_pos = cfg.container_radius * 0.3 * np.sqrt(rng.random())
            theta = 2 * math.pi * rng.random()
            x = r_pos * math.cos(theta)
            y = r_pos * math.sin(theta)
            z = cfg.pour_height + i * cfg.particle_size * 3.0  # 3x to avoid overlap

            center = np.array([x, y, z])

            if cfg.shape_type == "random":
                verts = random_polyhedron(
                    center, cfg.particle_size,
                    n_vertices=cfg.n_vertices,
                    seed=cfg.seed + i
                )
            elif cfg.shape_type in ("icosahedron", "cube", "octahedron",
                                     "tetrahedron", "dodecahedron"):
                verts = regular_polyhedron(cfg.shape_type, center, cfg.particle_size)
            else:
                raise ValueError(f"Unknown shape_type: {cfg.shape_type}")

            # Random initial orientation
            from .convex_contact import Polytope
            angle = rng.random() * 2 * math.pi
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            # Rotation matrix (Rodrigues)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
            verts_rot = (R @ (verts - center).T).T + center

            particle = PolyhedronParticle.from_polytope(verts_rot, density=cfg.density)
            particles.append(particle)

        self.particles = particles
        return particles

    def setup_system(self) -> PolyhedronDEMSystem:
        """Create the DEM system with container walls."""
        cfg = self.config

        # Contact model config
        contact_cfg = FrictionPolyhedronConfig(
            k_n=cfg.k_n,
            gamma_n=cfg.gamma_n,
            mu=cfg.mu,
            mu_r=cfg.mu_r,
            density=cfg.density,
        )

        # Create system
        system = PolyhedronDEMSystem(self.particles, contact_cfg, gravity=-9.81)

        # Add floor (z = 0, normal pointing up)
        system.add_wall(normal=np.array([0, 0, 1]), offset=0.0)

        # Add cylindrical wall approximation (flat walls forming a polygon)
        # Normals point INWARD (toward center), offset is the wall distance from origin
        n_wall_segments = 12
        for i in range(n_wall_segments):
            angle = 2 * math.pi * i / n_wall_segments
            # Normal points inward (toward center)
            nx = -math.cos(angle)
            ny = -math.sin(angle)
            # Offset is the distance from origin to the wall plane
            offset = -cfg.container_radius * math.cos(math.pi / n_wall_segments)
            system.add_wall(normal=np.array([nx, ny, 0]), offset=offset)

        self.system = system
        return system

    def run(self, record_interval: int = 100) -> dict:
        """Run the full simulation.

        Args:
            record_interval: Record history every N steps.

        Returns:
            dict with simulation results.
        """
        cfg = self.config

        # Generate particles if not done
        if not self.particles:
            self.generate_particles()

        # Setup system
        if self.system is None:
            self.setup_system()

        system = self.system
        system.record_initial_energy()

        # Time integration
        n_steps = int(cfg.settle_time / cfg.dt)
        step = 0
        current_time = 0.0

        while step < n_steps:
            system.step(cfg.dt)
            step += 1
            current_time += cfg.dt

            if step % record_interval == 0:
                self.history['time'].append(current_time)
                self.history['ke'].append(system.ke_trans + system.ke_rot)
                self.history['pe'].append(system.pe_gravity)
                self.history['dissipated'].append(system.energy_dissipated)
                self.history['total_energy'].append(system.total_energy())

                velocities = [np.linalg.norm(p.v) for p in system.particles]
                self.history['max_velocity'].append(max(velocities))
                self.history['mean_velocity'].append(np.mean(velocities))

        # Measure results
        repose = measure_repose_angle(system.particles, cfg.container_radius)
        packing = measure_volume_fraction(system.particles, cfg.container_radius)
        energy_error = system.energy_balance_error()

        self.result = {
            'repose': repose,
            'packing': packing,
            'energy_error': energy_error,
            'n_particles': len(system.particles),
            'simulated_time': current_time,
            'n_steps': step,
            'final_ke': system.ke_trans + system.ke_rot,
            'final_pe': system.pe_gravity,
            'total_dissipated': system.energy_dissipated,
            'history': self.history,
        }

        return self.result


def run_repose_simulation(
    n_particles: int = 100,
    mu: float = 0.4,
    mu_r: float = 0.05,
    shape_type: str = "random",
    seed: int = 42,
    **kwargs,
) -> dict:
    """Convenience function to run a repose angle simulation.

    Args:
        n_particles: Number of particles.
        mu: Friction coefficient.
        mu_r: Rolling resistance coefficient.
        shape_type: Particle shape type.
        seed: Random seed.
        **kwargs: Additional config overrides.

    Returns:
        dict with simulation results.
    """
    config = PolyhedraReposeConfig(
        n_particles=n_particles,
        mu=mu,
        mu_r=mu_r,
        shape_type=shape_type,
        seed=seed,
        **kwargs,
    )
    sim = PolyhedraReposeSimulation(config)
    return sim.run()
