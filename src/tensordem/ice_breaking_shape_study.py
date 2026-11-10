"""Ice breaking resistance study: sphere vs polyhedra comparison.

Simplified quasi-static approach: particles are pushed against a hull wall
and the reaction force is measured. This avoids expensive dynamic simulation
while still capturing the shape effect on resistance.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .convex_contact import Polytope, Sphere, contact
from .polyhedra_shapes import (
    SHAPE_NAMES,
    PolyhedronSpec,
    ShapeDescriptor,
    compute_shape_descriptor,
    make_polyhedron_by_name,
    scale_to_volume,
)

__all__ = [
    "IceBreakingConfig",
    "IceBreakingResult",
    "run_ice_breaking_case",
    "run_shape_comparison",
    "export_results_json",
]


@dataclass
class IceBreakingConfig:
    """Configuration for ice breaking simulation."""
    n_particles: int = 20
    particle_volume: float = 1e-6  # m^3
    domain_size: tuple = (0.05, 0.03, 0.03)  # (Lx, Ly, Lz) in m
    indentation_depth: float = 0.005  # m (how far particles are pushed into wall)
    k_n: float = 1e6  # N/m (wall contact stiffness)
    mu: float = 0.4  # friction coefficient
    rho: float = 917.0  # kg/m^3
    shape_name: str = "sphere"
    seed: int = 42


@dataclass
class IceBreakingResult:
    """Results from an ice breaking simulation."""
    config: IceBreakingConfig
    shape_descriptor: Optional[ShapeDescriptor] = None

    # Summary scalars
    mean_resistance: float = 0.0  # Mean hull reaction force [N]
    peak_resistance: float = 0.0  # Peak hull reaction force [N]
    mean_tangential: float = 0.0  # Mean tangential force [N]
    n_contacts: int = 0  # Number of wall contacts
    mean_penetration: float = 0.0  # Mean penetration depth [m]

    # Per-particle data
    particle_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    particle_positions: np.ndarray = field(default_factory=lambda: np.array([]))


def _generate_particle_positions(
    rng: np.random.Generator,
    n: int,
    volume: float,
    domain: tuple,
    r_bound: float,
) -> np.ndarray:
    """Generate random particle positions within domain."""
    lx, ly, lz = domain
    positions = np.zeros((n, 3))
    for i in range(n):
        for _ in range(200):  # Max attempts
            pos = np.array([
                rng.uniform(r_bound, lx * 0.6),  # Particles in front half
                rng.uniform(r_bound + 0.001, ly - r_bound - 0.001),
                rng.uniform(r_bound + 0.001, lz - r_bound - 0.001),
            ])
            # Check overlap with existing particles
            if i == 0:
                break
            dists = np.linalg.norm(positions[:i] - pos, axis=1)
            min_sep = 2.05 * r_bound
            if np.all(dists > min_sep):
                break
        positions[i] = pos
    return positions


def run_ice_breaking_case(config: IceBreakingConfig) -> IceBreakingResult:
    """Run a quasi-static ice breaking case.

    Particles are placed randomly, then pushed against a wall at x = Lx.
    The wall reaction force is computed from contact mechanics.
    """
    rng = np.random.default_rng(config.seed)
    lx, ly, lz = config.domain_size

    # Create shapes
    shapes = []
    masses = []
    r_bounds = []

    if config.shape_name == "sphere":
        for i in range(config.n_particles):
            vol = config.particle_volume * (1.0 + 0.05 * rng.standard_normal())
            vol = max(vol, 0.8 * config.particle_volume)
            r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
            center = np.array([
                rng.uniform(lx * 0.7, lx - r - 0.001),
                rng.uniform(r + 0.001, ly - r - 0.001),
                rng.uniform(r + 0.001, lz - r - 0.001),
            ])
            shapes.append(Sphere(center=center, radius=r))
            masses.append(config.rho * vol)
            r_bounds.append(r)
    else:
        spec = make_polyhedron_by_name(config.shape_name)
        descriptor = compute_shape_descriptor(config.shape_name, spec.vertices)

        for i in range(config.n_particles):
            vol = config.particle_volume * (1.0 + 0.05 * rng.standard_normal())
            vol = max(vol, 0.8 * config.particle_volume)
            verts = scale_to_volume(spec.vertices, vol)
            # Random rotation
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            rot = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ])
            verts = verts @ rot.T
            # Compute bounding radius
            centroid = verts.mean(axis=0)
            verts = verts - centroid
            r_bound = float(np.max(np.linalg.norm(verts, axis=1)))

            center = np.array([
                rng.uniform(lx * 0.7, lx - r_bound - 0.001),
                rng.uniform(r_bound + 0.001, ly - r_bound - 0.001),
                rng.uniform(r_bound + 0.001, lz - r_bound - 0.001),
            ])
            verts = verts + center
            shapes.append(Polytope(vertices=verts))
            masses.append(config.rho * vol)
            r_bounds.append(r_bound)

    masses = np.array(masses)
    r_bounds = np.array(r_bounds)

    # Quasi-static indentation: push particles towards wall
    wall_x = lx
    indentation = config.indentation_depth

    # Compute wall contact forces
    wall_forces = np.zeros(config.n_particles)
    tangential_forces = np.zeros(config.n_particles)
    penetrations = np.zeros(config.n_particles)
    n_wall_contacts = 0

    for i, shape in enumerate(shapes):
        # Move particle towards wall
        r_bound = r_bounds[i]
        original_x = shapes[i].centroid()[0] if hasattr(shapes[i], 'centroid') else shapes[i].center[0]

        # Check if particle would contact wall after indentation
        new_x = original_x + indentation
        if isinstance(shape, Sphere):
            new_shape = Sphere(center=np.array([new_x, shape.center[1], shape.center[2]]), radius=shape.radius)
            penetration = new_x + shape.radius - wall_x
        else:
            # For polytope, translate vertices
            new_verts = shape.vertices.copy()
            new_verts[:, 0] += indentation
            new_shape = Polytope(vertices=new_verts)
            # Penetration = max x of vertices - wall_x
            penetration = float(np.max(new_verts[:, 0])) - wall_x

        if penetration > 0:
            n_wall_contacts += 1
            penetrations[i] = penetration

            # Normal force
            f_n = config.k_n * penetration
            wall_forces[i] = f_n

            # Tangential force (friction) - simplified
            # Assume particles have some lateral velocity
            v_t = 0.1  # m/s (typical sliding speed)
            f_t = config.mu * f_n * min(v_t / 0.01, 1.0)  # Regularised Coulomb
            tangential_forces[i] = f_t

    # Summary statistics
    total_resistance = np.sum(wall_forces)
    mean_resistance = total_resistance / max(config.n_particles, 1)
    peak_resistance = np.max(wall_forces) if n_wall_contacts > 0 else 0.0
    mean_tangential = np.mean(tangential_forces)
    mean_penetration = np.mean(penetrations[penetrations > 0]) if n_wall_contacts > 0 else 0.0

    # Shape descriptor
    shape_desc = None
    if config.shape_name != "sphere":
        spec = make_polyhedron_by_name(config.shape_name)
        shape_desc = compute_shape_descriptor(config.shape_name, spec.vertices)

    return IceBreakingResult(
        config=config,
        shape_descriptor=shape_desc,
        mean_resistance=float(mean_resistance),
        peak_resistance=float(peak_resistance),
        mean_tangential=float(mean_tangential),
        n_contacts=n_wall_contacts,
        mean_penetration=float(mean_penetration),
        particle_forces=wall_forces,
        particle_positions=np.array([s.centroid() if hasattr(s, 'centroid') else s.center for s in shapes]),
    )


def run_shape_comparison(
    shape_names: list[str] = None,
    config_base: IceBreakingConfig = None,
) -> dict:
    """Run ice breaking for multiple shapes and compare results."""
    if shape_names is None:
        shape_names = ["sphere", "dodecahedron", "cube", "plate", "column"]

    if config_base is None:
        config_base = IceBreakingConfig()

    results = {}
    for name in shape_names:
        config = IceBreakingConfig(
            n_particles=config_base.n_particles,
            particle_volume=config_base.particle_volume,
            domain_size=config_base.domain_size,
            indentation_depth=config_base.indentation_depth,
            k_n=config_base.k_n,
            mu=config_base.mu,
            rho=config_base.rho,
            shape_name=name,
            seed=config_base.seed,
        )
        results[name] = run_ice_breaking_case(config)

    # Compute comparison metrics
    sphere_result = results.get("sphere")
    comparison = {}
    if sphere_result is not None:
        for name, res in results.items():
            if name == "sphere":
                continue
            resistance_ratio = res.mean_resistance / max(sphere_result.mean_resistance, 1e-10)
            comparison[name] = {
                "resistance_ratio": resistance_ratio,
                "resistance_diff_pct": 100.0 * (resistance_ratio - 1.0),
            }

    return {
        "results": results,
        "comparison": comparison,
        "sphere_baseline": {
            "mean_resistance": sphere_result.mean_resistance if sphere_result else 0.0,
            "n_contacts": sphere_result.n_contacts if sphere_result else 0,
        },
    }


def export_results_json(
    comparison: dict,
    output_path: str = "artifacts/shape_study_results.json",
) -> dict:
    """Export results to JSON with sha256 hash for audit trail."""
    data = {
        "study": "ice_breaking_shape_comparison",
        "timestamp": str(np.datetime64("now")),
        "sphere_baseline": comparison["sphere_baseline"],
        "shapes": {},
        "comparison": comparison["comparison"],
    }

    for name, result in comparison["results"].items():
        shape_data = {
            "mean_resistance_N": result.mean_resistance,
            "peak_resistance_N": result.peak_resistance,
            "mean_tangential_N": result.mean_tangential,
            "n_contacts": result.n_contacts,
            "mean_penetration_m": result.mean_penetration,
        }
        if result.shape_descriptor is not None:
            sd = result.shape_descriptor
            shape_data["shape_descriptor"] = {
                "sphericity": sd.sphericity,
                "angularity": sd.angularity,
                "aspect_ratio": sd.aspect_ratio,
                "n_vertices": sd.n_vertices,
                "n_faces": sd.n_faces,
            }
        data["shapes"][name] = shape_data

    # Compute sha256
    json_str = json.dumps(data, sort_keys=True, indent=2)
    sha256 = hashlib.sha256(json_str.encode()).hexdigest()
    data["sha256"] = sha256

    # Write to file
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return data
