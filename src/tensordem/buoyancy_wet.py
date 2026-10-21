"""Particle buoyancy model and linear-damping wet resistance for 2-D DEM.

Provides:
- Buoyancy force per particle based on submerged fraction computed from the
  particle centre height relative to a configurable water-surface elevation.
  For a circular disk of radius R, the submerged area is obtained analytically
  from circular-segment geometry, giving the submerged fraction f_sub in [0, 1].
  The buoyancy force is then  F_b = rho_w * g * V_sub * f_sub  (per-unit-depth
  equivalent: rho_w * g * A_sub where A_sub = pi*R^2 * f_sub).

- Linear viscous wet drag  F_wet = -C_wet * v  applied only to the submerged
  portion of each particle, using the same f_sub as a weight.

All quantities use SI units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Circular-segment submerged-fraction helper (scalar, CPU)
# ---------------------------------------------------------------------------

def _submerged_fraction_2d(h: float, R: float) -> float:
    """Submerged area fraction of a circle of radius R whose centre is at
    height *h* relative to the free surface (positive h = centre above water).

    h >=  R  => fully out of water  => f=0
    h <= -R  => fully submerged       => f=1
    otherwise use the circular-segment formula.
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 1.0
    # h is in (-R, R): partial submersion.
    # The submerged cap angle theta satisfies cos(theta/2) = h/R.
    theta = 2.0 * math.acos(max(-1.0, min(1.0, h / R)))
    # Submerged area of the circular segment:
    #   A_sub = 0.5 * R^2 * (theta - sin(theta))
    # Total area = pi * R^2, so
    #   f_sub = (theta - sin(theta)) / (2 * pi)
    return (theta - math.sin(theta)) / (2.0 * math.pi)


def submerged_fraction_tensor(
    y_centre: torch.Tensor, radius: float, water_level: float = 0.0
) -> torch.Tensor:
    """Vectorised submerged fraction for a batch of particle y-coordinates.

    Args:
        y_centre:  (N,) tensor of particle centre y-coordinates.
        radius:    particle radius (scalar, m).
        water_level: free-surface elevation (scalar, m).  Default 0.

    Returns:
        (N,) tensor of submerged fractions in [0, 1].
    """
    # h = distance of centre *above* water level
    h = y_centre - water_level
    # Clamp h to [-R, R] for safe acos
    h_clamped = torch.clamp(h, -radius, radius)
    # theta = 2 * acos(h/R); for fully in/out cases theta is set correctly
    theta = 2.0 * torch.acos(h_clamped / radius)
    f_sub = (theta - torch.sin(theta)) / (2.0 * math.pi)
    # Clamp to [0, 1] for numerical safety
    f_sub = torch.clamp(f_sub, 0.0, 1.0)
    # Pure above-water or fully-submerged: overwrite with exact values
    f_sub = torch.where(h >= radius, torch.zeros_like(f_sub), f_sub)
    f_sub = torch.where(h <= -radius, torch.ones_like(f_sub), f_sub)
    return f_sub


# ---------------------------------------------------------------------------
# Wet-coupling configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WetCouplingConfig:
    """Configuration for buoyancy and wet drag on 2-D DEM particles."""
    enabled: bool = False
    water_density: float = 1025.0      # rho_w, kg/m^3 (sea water default)
    gravity: float = 9.81              # g, m/s^2
    water_level: float = 0.0           # free-surface elevation, m
    wet_drag_coeff: float = 0.0        # C_wet, N·s/m per particle (linear)


# ---------------------------------------------------------------------------
# Force computation (stand-alone, injectable into the DEM force loop)
# ---------------------------------------------------------------------------

def compute_buoyancy_force(
    positions: torch.Tensor,
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Return buoyancy forces for all particles (shape N×2, same device as positions).

    The buoyancy force acts purely in +y (vertical).  Per-unit-depth formulation
    is converted to 3-D equivalent by multiplying with *thickness*.

    For a particle whose centre is at (x, y), the submerged fraction f_sub is
    computed from y relative to cfg.water_level and the particle radius.

        F_b_y = rho_w * g * (pi * R^2 * f_sub) * thickness

    Args:
        positions: (N, 2) tensor of particle positions (x, y).
        radius:    particle radius, m.
        thickness: extrusion thickness, m.
        density_ice: ice density (used only to sanity-check; not needed for F_b).
        cfg:       WetCouplingConfig instance.

    Returns:
        (N, 2) buoyancy force tensor.
    """
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    if not cfg.enabled:
        return torch.zeros_like(positions)

    y_centre = positions[:, 1]
    f_sub = submerged_fraction_tensor(y_centre, radius, cfg.water_level)

    # V_sub = A_sub * thickness = pi * R^2 * f_sub * thickness
    vol_sub = math.pi * radius ** 2 * f_sub * thickness
    # F_b_y = rho_w * g * V_sub
    Fb_y = cfg.water_density * cfg.gravity * vol_sub

    force = torch.zeros((N, 2), device=device, dtype=dtype)
    force[:, 1] = Fb_y
    return force


def compute_wet_drag_force(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Return linear wet-damping forces (shape N×2).

        F_wet = -C_wet * f_sub * v

    The drag is proportional to the submerged fraction so that a fully dry
    particle feels no wet drag (the existing dry drag remains separate).

    Args:
        positions:  (N, 2) particle positions.
        velocities: (N, 2) particle velocities.
        radius:     particle radius, m.
        cfg:        WetCouplingConfig instance.

    Returns:
        (N, 2) wet drag force tensor.
    """
    if not cfg.enabled or cfg.wet_drag_coeff <= 0.0:
        return torch.zeros_like(velocities)

    y_centre = positions[:, 1]
    f_sub = submerged_fraction_tensor(y_centre, radius, cfg.water_level)
    # F_wet = -C_wet * f_sub * v  (applied to both x and y components)
    return -cfg.wet_drag_coeff * f_sub[:, None] * velocities


def compute_wet_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined buoyancy + wet drag.

    Returns:
        (buoyancy_force, wet_drag_force), each (N, 2).
    """
    buoyancy = compute_buoyancy_force(positions, radius, thickness, density_ice, cfg)
    wet_drag = compute_wet_drag_force(positions, velocities, radius, cfg)
    return buoyancy, wet_drag
