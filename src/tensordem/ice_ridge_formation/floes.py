"""Polygonal ice floe population with power-law size distribution.

Ice floes in nature have irregular polygonal shapes with a size distribution
that follows a power law: n(D) ~ D^(-alpha), where alpha is typically 2-3.

This module generates polygonal floes using Voronoi tessellation and assigns
sizes from a power-law distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import Voronoi

__all__ = [
    "PolygonalFloe",
    "FloePopulation",
    "generate_power_law_floes",
    "FloePopulationConfig",
]


@dataclass(frozen=True)
class PolygonalFloe:
    """A single polygonal ice floe.
    
    Attributes
    ----------
    vertices : torch.Tensor, shape (n_vertices, 2)
        Polygon vertices in counter-clockwise order (metres).
    centre : torch.Tensor, shape (2,)
        Centre of mass (metres).
    thickness : float
        Ice thickness (metres).
    density : float
        Ice density (kg/m^3).
    bending_strength : float
        Bending strength (Pa).
    floe_id : int
        Unique identifier.
    """
    vertices: torch.Tensor
    centre: torch.Tensor
    thickness: float
    density: float
    bending_strength: float
    floe_id: int
    
    @property
    def area(self) -> float:
        """Area of the floe (m^2) using shoelace formula."""
        v = self.vertices
        n = v.shape[0]
        x = v[:, 0]
        y = v[:, 1]
        return 0.5 * float(torch.abs(torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y)))
    
    @property
    def volume(self) -> float:
        """Volume of the floe (m^3)."""
        return self.area * self.thickness
    
    @property
    def mass(self) -> float:
        """Mass of the floe (kg)."""
        return self.volume * self.density
    
    @property
    def equivalent_diameter(self) -> float:
        """Diameter of a circle with the same area (metres)."""
        return 2.0 * math.sqrt(self.area / math.pi)
    
    def contains_point(self, point: torch.Tensor) -> bool:
        """Check if a point is inside the polygon (ray casting algorithm)."""
        x, y = float(point[0]), float(point[1])
        n = self.vertices.shape[0]
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = float(self.vertices[i, 0]), float(self.vertices[i, 1])
            xj, yj = float(self.vertices[j, 0]), float(self.vertices[j, 1])
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def to_disks(self, n_disks: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Approximate the polygon with a set of disks for DEM simulation.
        
        Returns
        -------
        positions : torch.Tensor, shape (n_disks, 2)
            Disk centre positions.
        radii : torch.Tensor, shape (n_disks,)
            Disk radii.
        """
        # Simple approach: place disks at the centre and around the perimeter
        positions = [self.centre.unsqueeze(0)]
        radii = [self.equivalent_diameter * 0.3]
        
        # Add perimeter disks
        perimeter_length = 0.0
        for i in range(self.vertices.shape[0]):
            j = (i + 1) % self.vertices.shape[0]
            perimeter_length += torch.norm(self.vertices[j] - self.vertices[i])
        
        disk_spacing = perimeter_length / (n_disks - 1)
        for i in range(self.vertices.shape[0]):
            j = (i + 1) % self.vertices.shape[0]
            edge = self.vertices[j] - self.vertices[i]
            edge_length = torch.norm(edge)
            n_edge_disks = max(1, int(edge_length / disk_spacing))
            for k in range(n_edge_disks):
                t = (k + 0.5) / n_edge_disks
                pos = self.vertices[i] + t * edge
                # Move slightly inward
                normal = torch.tensor([-edge[1], edge[0]]) / edge_length
                pos = pos - 0.1 * self.equivalent_diameter * normal
                positions.append(pos.unsqueeze(0))
                radii.append(self.equivalent_diameter * 0.15)
        
        positions = torch.cat(positions, dim=0)[:n_disks]
        radii = torch.tensor(radii[:n_disks], dtype=torch.float64)
        return positions, radii


@dataclass(frozen=True)
class FloePopulationConfig:
    """Configuration for generating a floe population.
    
    Attributes
    ----------
    n_floes : int
        Number of floes to generate.
    domain_size : Tuple[float, float]
        Domain size (width, height) in metres.
    thickness : float
        Ice thickness (metres).
    density : float
        Ice density (kg/m^3).
    water_density : float
        Water density (kg/m^3).
    bending_strength : float
        Bending strength (Pa).
    power_law_exponent : float
        Power-law exponent alpha for size distribution n(D) ~ D^(-alpha).
    min_diameter : float
        Minimum floe diameter (metres).
    max_diameter : float
        Maximum floe diameter (metres).
    seed : int
        Random seed for reproducibility.
    """
    n_floes: int = 50
    domain_size: Tuple[float, float] = (20.0, 10.0)
    thickness: float = 1.0
    density: float = 917.0
    water_density: float = 1025.0
    bending_strength: float = 5.0e5
    power_law_exponent: float = 2.5
    min_diameter: float = 0.5
    max_diameter: float = 3.0
    seed: int = 20261110
    
    def __post_init__(self) -> None:
        if self.n_floes < 1:
            raise ValueError("n_floes must be at least 1")
        if self.domain_size[0] <= 0 or self.domain_size[1] <= 0:
            raise ValueError("domain_size must be positive")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.water_density <= 0:
            raise ValueError("water_density must be positive")
        if self.bending_strength <= 0:
            raise ValueError("bending_strength must be positive")
        if self.power_law_exponent <= 0:
            raise ValueError("power_law_exponent must be positive")
        if self.min_diameter <= 0 or self.max_diameter <= self.min_diameter:
            raise ValueError("invalid diameter range")


class FloePopulation:
    """A population of polygonal ice floes.
    
    Attributes
    ----------
    floes : List[PolygonalFloe]
        List of floes in the population.
    config : FloePopulationConfig
        Configuration used to generate the population.
    """
    
    def __init__(self, floes: List[PolygonalFloe], config: FloePopulationConfig):
        self.floes = floes
        self.config = config
    
    @property
    def total_area(self) -> float:
        """Total area of all floes (m^2)."""
        return sum(floe.area for floe in self.floes)
    
    @property
    def total_volume(self) -> float:
        """Total volume of all floes (m^3)."""
        return sum(floe.volume for floe in self.floes)
    
    @property
    def total_mass(self) -> float:
        """Total mass of all floes (kg)."""
        return sum(floe.mass for floe in self.floes)
    
    @property
    def size_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Size distribution: diameters and counts.
        
        Returns
        -------
        diameters : np.ndarray
            Floe diameters (metres).
        counts : np.ndarray
            Number of floes in each size bin.
        """
        diameters = np.array([floe.equivalent_diameter for floe in self.floes])
        return diameters, np.ones_like(diameters)
    
    def to_dem_particles(self, particles_per_floe: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert all floes to DEM particles.
        
        Returns
        -------
        positions : torch.Tensor, shape (n_particles, 2)
            Particle positions.
        radii : torch.Tensor, shape (n_particles,)
            Particle radii.
        floe_ids : torch.Tensor, shape (n_particles,)
            Floe ID for each particle.
        """
        all_positions = []
        all_radii = []
        all_floe_ids = []
        
        for floe in self.floes:
            positions, radii = floe.to_disks(particles_per_floe)
            all_positions.append(positions)
            all_radii.append(radii)
            all_floe_ids.append(torch.full((positions.shape[0],), floe.floe_id, dtype=torch.long))
        
        positions = torch.cat(all_positions, dim=0)
        radii = torch.cat(all_radii, dim=0)
        floe_ids = torch.cat(all_floe_ids, dim=0)
        
        return positions, radii, floe_ids


def generate_power_law_floes(config: FloePopulationConfig) -> FloePopulation:
    """Generate a population of polygonal floes with power-law size distribution.
    
    The size distribution follows n(D) ~ D^(-alpha), where alpha is the
    power-law exponent. Floes are generated using Voronoi tessellation.
    
    Parameters
    ----------
    config : FloePopulationConfig
        Configuration for the floe population.
    
    Returns
    -------
    FloePopulation
        Population of polygonal floes.
    """
    rng = np.random.default_rng(config.seed)
    
    # Generate floe diameters from power-law distribution
    # n(D) ~ D^(-alpha) => CDF: F(D) = (D^(-alpha+1) - D_min^(-alpha+1)) / (D_max^(-alpha+1) - D_min^(-alpha+1))
    alpha = config.power_law_exponent
    d_min = config.min_diameter
    d_max = config.max_diameter
    
    # Inverse CDF sampling
    u = rng.uniform(0, 1, config.n_floes)
    if abs(alpha - 1.0) < 1e-6:
        # Special case: alpha = 1
        diameters = d_min * (d_max / d_min) ** u
    else:
        # General case
        exponent = 1.0 - alpha
        diameters = (d_min ** exponent + u * (d_max ** exponent - d_min ** exponent)) ** (1.0 / exponent)
    
    # Compute buoyancy equilibrium: submerged fraction = rho_ice / rho_water
    submerged_fraction = config.density / config.water_density
    draft = submerged_fraction * config.thickness
    freeboard = config.thickness - draft
    
    # Generate floe positions (avoid overlap)
    # Floes should be positioned so their centre is at the waterline minus draft
    positions = []
    max_attempts = 1000
    for i in range(config.n_floes):
        for attempt in range(max_attempts):
            x = rng.uniform(0, config.domain_size[0])
            # Position floe centre at waterline minus half the draft
            # This ensures the floe floats with correct buoyancy
            y = -draft / 2  # Centre is below waterline by half the draft
            
            # Check for overlap with existing floes
            overlap = False
            for j, (px, py, d) in enumerate(positions):
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < 0.5 * (diameters[i] + d):
                    overlap = True
                    break
            
            if not overlap:
                positions.append((x, y, diameters[i]))
                break
        else:
            # If we couldn't place the floe, skip it
            continue
    
    # Generate polygonal shapes using Voronoi tessellation
    floes = []
    for i, (x, y, d) in enumerate(positions):
        # Generate random points around the centre
        n_points = max(5, int(d * 3))  # More points for larger floes
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radii = d * 0.5 * (1.0 + 0.2 * rng.standard_normal(n_points))
        
        # Create polygon vertices
        vertices_x = x + radii * np.cos(angles)
        vertices_y = y + radii * np.sin(angles)
        vertices = torch.tensor(np.stack([vertices_x, vertices_y], axis=1), dtype=torch.float64)
        
        # Ensure counter-clockwise ordering
        centre = torch.tensor([x, y], dtype=torch.float64)
        floe = PolygonalFloe(
            vertices=vertices,
            centre=centre,
            thickness=config.thickness,
            density=config.density,
            bending_strength=config.bending_strength,
            floe_id=i,
        )
        floes.append(floe)
    
    return FloePopulation(floes, config)
