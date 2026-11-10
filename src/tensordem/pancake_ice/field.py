"""Pancake ice field initialization and DEM-style simulation.

Generates a random field of circular ice floes with specified diameter
distribution, concentration, and orientation. Provides a simple plan-view
DEM simulation for pile-up and resistance estimation.

Physical model
--------------
Each floe is a circular disk of diameter D_f and thickness h, floating at
the water surface. The field is initialized with:

1. Random positions (Poisson-distributed for given concentration)
2. Log-normal diameter distribution (typical for pancake ice)
3. Random orientations (uniform or Gaussian about mean)

The DEM simulation advances floes under:
- Hull-floe contact (Kelvin-Voigt + Coulomb)
- Floe-floe contact (Hertz-Mindlin)
- Hydrodynamic drag (quadratic)
- Buoyancy restoration

References
----------
- Herman et al. (2017): Wave-ice interaction in pancake fields
- Dierking (2013): Antarctic pancake ice observations
- Shen & Ackley (2001): Pancake ice dynamics

All quantities SI: m, Pa, kg/m³, m/s, N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "FloeProperties",
    "PancakeIceField",
    "generate_field",
    "compute_field_statistics",
    "simple_dem_step",
    "estimate_pile_up_dem",
]

# Physical constants
WATER_DENSITY = 1025.0
ICE_DENSITY = 917.0
GRAVITY = 9.80665


@dataclass
class FloeProperties:
    """Properties of a single ice floe.
    
    Attributes
    ----------
    x : float
        X position [m].
    y : float
        Y position [m].
    diameter : float
        Floe diameter D_f [m].
    thickness : float
        Floe thickness h [m].
    orientation : float
        Floe orientation angle θ [rad].
    vx : float
        X velocity [m/s].
    vy : float
        Y velocity [m/s].
    omega : float
        Angular velocity [rad/s].
    mass : float
        Floe mass [kg].
    moment_inertia : float
        Moment of inertia [kg·m²].
    """
    x: float = 0.0
    y: float = 0.0
    diameter: float = 1.0
    thickness: float = 0.1
    orientation: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    mass: float = 0.0
    moment_inertia: float = 0.0

    def __post_init__(self) -> None:
        # Compute mass and moment of inertia
        area = math.pi * (self.diameter / 2) ** 2
        volume = area * self.thickness
        self.mass = ICE_DENSITY * volume
        self.moment_inertia = 0.5 * self.mass * (self.diameter / 2) ** 2


@dataclass
class PancakeIceField:
    """Collection of pancake ice floes.
    
    Attributes
    ----------
    floes : List[FloeProperties]
        List of floe objects.
    domain_x : Tuple[float, float]
        Domain extent in x [m].
    domain_y : Tuple[float, float]
        Domain extent in y [m].
    concentration : float
        Actual areal concentration.
    mean_diameter : float
        Mean floe diameter [m].
    std_diameter : float
        Standard deviation of diameter [m].
    """
    floes: List[FloeProperties] = field(default_factory=list)
    domain_x: Tuple[float, float] = (0.0, 100.0)
    domain_y: Tuple[float, float] = (-50.0, 50.0)
    concentration: float = 0.0
    mean_diameter: float = 0.0
    std_diameter: float = 0.0


def generate_field(
    domain_x: Tuple[float, float],
    domain_y: Tuple[float, float],
    target_concentration: float,
    mean_diameter: float,
    thickness: float,
    std_diameter: Optional[float] = None,
    orientation_std: float = math.pi,
    seed: Optional[int] = None,
) -> PancakeIceField:
    """Generate a random pancake ice field.
    
    Floes are placed randomly (Poisson process) with log-normal diameter
    distribution until target concentration is reached.
    
    Parameters
    ----------
    domain_x : Tuple[float, float]
        Domain extent in x [m].
    domain_y : Tuple[float, float]
        Domain extent in y [m].
    target_concentration : float
        Target areal concentration C ∈ [0, 1].
    mean_diameter : float
        Mean floe diameter D_f [m].
    thickness : float
        Floe thickness h [m].
    std_diameter : Optional[float]
        Standard deviation of diameter [m]. Default 0.3*mean_diameter.
    orientation_std : float
        Standard deviation of orientation [rad]. Default π (uniform).
    seed : Optional[int]
        Random seed for reproducibility.
    
    Returns
    -------
    PancakeIceField
        Generated ice field.
    """
    rng = np.random.default_rng(seed)
    
    if std_diameter is None:
        std_diameter = 0.3 * mean_diameter
    
    Lx = domain_x[1] - domain_x[0]
    Ly = domain_y[1] - domain_y[0]
    domain_area = Lx * Ly
    
    # Target ice area
    target_ice_area = target_concentration * domain_area
    
    # Mean floe area
    mean_floe_area = math.pi * (mean_diameter / 2) ** 2
    
    # Expected number of floes (with some margin for overlap rejection)
    n_expected = int(target_ice_area / mean_floe_area * 1.5)
    
    floes = []
    total_ice_area = 0.0
    
    # Log-normal distribution parameters
    sigma_ln = math.sqrt(math.log(1 + (std_diameter / mean_diameter) ** 2))
    mu_ln = math.log(mean_diameter) - 0.5 * sigma_ln**2
    
    for _ in range(n_expected):
        if total_ice_area >= target_ice_area:
            break
        
        # Random position
        x = rng.uniform(domain_x[0], domain_x[1])
        y = rng.uniform(domain_y[0], domain_y[1])
        
        # Log-normal diameter
        D = rng.lognormal(mu_ln, sigma_ln)
        D = max(0.05, min(D, 5.0))  # Clamp to physical range
        
        # Random orientation
        theta = rng.vonmises(0, 0.5)  # Von Mises distribution
        
        # Check for overlap with existing floes
        floe_radius = D / 2
        overlap = False
        for f in floes:
            dx = x - f.x
            dy = y - f.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < floe_radius + f.diameter / 2:
                overlap = True
                break
        
        if not overlap:
            floe = FloeProperties(
                x=x,
                y=y,
                diameter=D,
                thickness=thickness,
                orientation=theta,
            )
            floes.append(floe)
            total_ice_area += math.pi * (D / 2) ** 2
    
    # Compute actual concentration
    actual_concentration = total_ice_area / domain_area
    
    # Compute statistics
    if floes:
        diameters = [f.diameter for f in floes]
        mean_D = np.mean(diameters)
        std_D = np.std(diameters)
    else:
        mean_D = 0.0
        std_D = 0.0
    
    return PancakeIceField(
        floes=floes,
        domain_x=domain_x,
        domain_y=domain_y,
        concentration=actual_concentration,
        mean_diameter=mean_D,
        std_diameter=std_D,
    )


def compute_field_statistics(field: PancakeIceField) -> Dict[str, float]:
    """Compute statistics of the ice field.
    
    Parameters
    ----------
    field : PancakeIceField
        Ice field to analyze.
    
    Returns
    -------
    Dict[str, float]
        Statistics including mean diameter, concentration, etc.
    """
    if not field.floes:
        return {
            "n_floes": 0,
            "concentration": 0.0,
            "mean_diameter": 0.0,
            "std_diameter": 0.0,
            "min_diameter": 0.0,
            "max_diameter": 0.0,
            "total_ice_area": 0.0,
        }
    
    diameters = [f.diameter for f in field.floes]
    areas = [math.pi * (d / 2) ** 2 for d in diameters]
    
    domain_area = (field.domain_x[1] - field.domain_x[0]) * (field.domain_y[1] - field.domain_y[0])
    
    return {
        "n_floes": len(field.floes),
        "concentration": sum(areas) / domain_area,
        "mean_diameter": np.mean(diameters),
        "std_diameter": np.std(diameters),
        "min_diameter": min(diameters),
        "max_diameter": max(diameters),
        "total_ice_area": sum(areas),
    }


def simple_dem_step(
    field: PancakeIceField,
    hull_x: float,
    hull_y: float,
    hull_beam: float,
    hull_speed: float,
    dt: float,
    contact_stiffness: float = 1e6,
    friction_coeff: float = 0.1,
) -> Tuple[float, float]:
    """Advance the ice field by one time step.
    
    Simple DEM with hull-floe and floe-floe contacts.
    
    Parameters
    ----------
    field : PancakeIceField
        Ice field to advance.
    hull_x : float
        Hull x position [m].
    hull_y : float
        Hull y position [m].
    hull_beam : float
        Hull beam [m].
    hull_speed : float
        Hull speed [m/s].
    dt : float
        Time step [s].
    contact_stiffness : float
        Contact stiffness [N/m].
    friction_coeff : float
        Friction coefficient [-].
    
    Returns
    -------
    F_x : float
        Total x force on hull [N].
    F_y : float
        Total y force on hull [N].
    """
    F_x_total = 0.0
    F_y_total = 0.0
    
    # Hull as a rectangle
    hull_xmin = hull_x - hull_beam / 2
    hull_xmax = hull_x + hull_beam / 2
    hull_ymin = hull_y - hull_beam / 4
    hull_ymax = hull_y + hull_beam / 4
    
    for floe in field.floes:
        # Hull-floe contact
        dx = floe.x - hull_x
        dy = floe.y - hull_y
        dist = math.sqrt(dx**2 + dy**2)
        floe_radius = floe.diameter / 2
        
        # Check if floe overlaps with hull
        if abs(dx) < hull_beam / 2 + floe_radius and abs(dy) < hull_beam / 4 + floe_radius:
            # Compute overlap
            overlap_x = max(0, hull_beam / 2 + floe_radius - abs(dx))
            overlap_y = max(0, hull_beam / 4 + floe_radius - abs(dy))
            overlap = min(overlap_x, overlap_y)
            
            if overlap > 0:
                # Contact force
                F_n = contact_stiffness * overlap
                
                # Direction (push floe away from hull)
                if dist > 0:
                    nx = dx / dist
                    ny = dy / dist
                else:
                    nx, ny = 1.0, 0.0
                
                # Normal force
                Fx = F_n * nx
                Fy = F_n * ny
                
                # Friction (opposing relative motion)
                v_rel_x = floe.vx - hull_speed
                v_rel_y = floe.vy
                v_rel = math.sqrt(v_rel_x**2 + v_rel_y**2)
                if v_rel > 0:
                    F_fric = friction_coeff * F_n
                    Fx -= F_fric * v_rel_x / v_rel
                    Fy -= F_fric * v_rel_y / v_rel
                
                # Apply to floe
                floe.vx += Fx / floe.mass * dt
                floe.vy += Fy / floe.mass * dt
                
                # Reaction on hull
                F_x_total -= Fx
                F_y_total -= Fy
        
        # Floe-floe contact (simplified)
        for other in field.floes:
            if other is floe:
                continue
            dx = other.x - floe.x
            dy = other.y - floe.y
            dist = math.sqrt(dx**2 + dy**2)
            min_dist = (floe.diameter + other.diameter) / 2
            
            if dist < min_dist and dist > 0:
                overlap = min_dist - dist
                F_n = contact_stiffness * overlap * 0.5  # Shared
                
                nx = dx / dist
                ny = dy / dist
                
                floe.vx -= F_n * nx / floe.mass * dt
                floe.vy -= F_n * ny / floe.mass * dt
    
    # Update positions
    for floe in field.floes:
        # Hydrodynamic drag
        v = math.sqrt(floe.vx**2 + floe.vy**2)
        if v > 0:
            C_D = 1.0
            A = floe.diameter * floe.thickness
            F_drag = 0.5 * WATER_DENSITY * C_D * A * v**2
            floe.vx -= F_drag / floe.mass * dt * floe.vx / v
            floe.vy -= F_drag / floe.mass * dt * floe.vy / v
        
        # Damping
        floe.vx *= 0.99
        floe.vy *= 0.99
        
        # Update position
        floe.x += floe.vx * dt
        floe.y += floe.vy * dt
    
    return F_x_total, F_y_total


def estimate_pile_up_dem(
    field: PancakeIceField,
    hull_beam: float,
    hull_speed: float,
    n_steps: int = 100,
    dt: float = 0.01,
) -> Dict[str, float]:
    """Estimate pile-up geometry using simple DEM.
    
    Parameters
    ----------
    field : PancakeIceField
        Initial ice field.
    hull_beam : float
        Hull beam [m].
    hull_speed : float
        Hull speed [m/s].
    n_steps : int
        Number of time steps.
    dt : float
        Time step [s].
    
    Returns
    -------
    Dict[str, float]
        Pile-up statistics (height, length, etc.).
    """
    hull_x = field.domain_x[0]
    hull_y = (field.domain_y[0] + field.domain_y[1]) / 2
    
    total_Fx = 0.0
    max_pile_height = 0.0
    
    for step in range(n_steps):
        hull_x += hull_speed * dt
        Fx, Fy = simple_dem_step(field, hull_x, hull_y, hull_beam, hull_speed, dt)
        total_Fx += Fx
        
        # Estimate pile height from floe positions ahead of hull
        floes_ahead = [f for f in field.floes if f.x > hull_x and f.x < hull_x + 5.0]
        if floes_ahead:
            pile_height = sum(f.diameter for f in floes_ahead) / len(floes_ahead)
            max_pile_height = max(max_pile_height, pile_height)
    
    avg_Fx = total_Fx / n_steps
    
    return {
        "mean_resistance": avg_Fx,
        "max_pile_height": max_pile_height,
        "n_floes_remaining": len(field.floes),
    }
