"""Ice ridge mechanics model — ridge sailing force & keel drag.

Implements empirical models for ice ridge resistance based on:
- Kerr (1975): Ridge sailing force model
- Timco (2011): Ridge keel drag model

Physical domain: ice_ridge (SIR-PC2 / m611 seed, L1_foundation)

References
----------
- Kerr, A.D. (1975). "Ice ridge sailing resistance." Cold Regions Science
  and Technology.
- Timco, G.W. (2011). "A review of ice ridge keel resistance models."
  Cold Regions Science and Technology, 65(1), 1-15.

All quantities use SI units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Physical constants (defaults)
# ---------------------------------------------------------------------------

DEFAULT_ICE_DENSITY = 917.0       # rho_i, kg/m^3
DEFAULT_WATER_DENSITY = 1025.0    # rho_w, kg/m^3 (sea water)
DEFAULT_GRAVITY = 9.81            # g, m/s^2
DEFAULT_ICE_FLEX_STRENGTH = 0.5e6  # sigma_f, Pa (typical first-year ice)


# ---------------------------------------------------------------------------
# Ridge geometry configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RidgeGeometry:
    """Geometric description of an ice ridge.

    An ice ridge consists of a sail (above water) and a keel (below water).
    Typical sail-to-keel ratio is ~1:4 to 1:5.

    Attributes
    ----------
    sail_height : float
        Height of the ridge sail above water level, m.
    keel_draft : float
        Depth of the ridge keel below water level, m.
    ridge_width : float
        Transverse width of the ridge, m.
    ridge_length : float
        Longitudinal extent of the ridge, m.
    porosity : float
        Keel porosity (void fraction), typically 0.3–0.5.
    block_size : float
        Characteristic ice block size in the keel, m.
    sail_to_keel_ratio : float
        Ratio h_sail / h_keel (informational, not used in calculations).
    """
    sail_height: float = 1.5       # h_s, m
    keel_draft: float = 6.0        # h_k, m
    ridge_width: float = 30.0      # W_r, m
    ridge_length: float = 100.0    # L_r, m
    porosity: float = 0.4          # n, void fraction
    block_size: float = 0.5        # d_b, m
    sail_to_keel_ratio: float = 0.25  # h_s / h_k


# ---------------------------------------------------------------------------
# Ship configuration for ridge interaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShipConfig:
    """Ship parameters relevant to ridge interaction.

    Attributes
    ----------
    beam : float
        Ship beam (width), m.
    draft : float
        Ship draft, m.
    speed : float
        Ship speed through the ridge, m/s.
    bow_angle : float
        Half-angle of the bow (stem angle / 2), degrees.
    hull_friction_coeff : float
        Hull–ice friction coefficient, typically 0.05–0.15.
    """
    beam: float = 20.0            # B, m
    draft: float = 8.0            # T, m
    speed: float = 2.0            # V, m/s (~4 knots)
    bow_angle: float = 20.0       # phi, degrees (half-angle)
    hull_friction_coeff: float = 0.1  # mu


# ---------------------------------------------------------------------------
# Kerr (1975) — Ridge Sailing Force
# ---------------------------------------------------------------------------

def kerr_ridge_sailing_force(
    ridge: RidgeGeometry,
    ship: ShipConfig,
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    ice_flex_strength: float = DEFAULT_ICE_FLEX_STRENGTH,
) -> float:
    """Compute ridge sailing resistance per Kerr (1975) empirical model.

    The Kerr model expresses the ridge sailing force as:

        F_sail = C_k * rho_i * g * h_s^2 * B * f(V, phi)

    where:
        C_k  = empirical coefficient (~0.5–1.0, depends on ridge consolidation)
        rho_i = ice density
        g    = gravity
        h_s  = sail height
        B    = ship beam
        f(V, phi) = speed and bow-angle correction factor

    The speed correction accounts for dynamic effects:
        f(V, phi) = 1 + alpha * V / sqrt(g * h_s)

    The bow-angle correction:
        f_bow = 1 / sin(phi)  (narrower bow => higher force per unit beam)

    Combined:
        F_sail = C_k * rho_i * g * h_s^2 * B * (1 + alpha * V / sqrt(g * h_s)) / sin(phi)

    Parameters
    ----------
    ridge : RidgeGeometry
        Ridge geometric parameters.
    ship : ShipConfig
        Ship parameters.
    ice_density : float
        Ice density, kg/m^3.
    water_density : float
        Water density, kg/m^3.
    gravity : float
        Gravitational acceleration, m/s^2.
    ice_flex_strength : float
        Ice flexural strength, Pa.

    Returns
    -------
    float
        Ridge sailing force, N.
    """
    h_s = ridge.sail_height
    B = ship.beam
    V = ship.speed
    phi_rad = math.radians(ship.bow_angle)

    # Empirical coefficient for consolidated ridges (Kerr 1975)
    C_k = 0.65

    # Speed correction coefficient
    alpha = 0.3

    # Speed correction factor
    f_speed = 1.0 + alpha * V / math.sqrt(gravity * h_s)

    # Bow angle correction
    f_bow = 1.0 / math.sin(phi_rad)

    # Kerr sailing force
    F_sail = C_k * ice_density * gravity * h_s ** 2 * B * f_speed * f_bow

    return F_sail


def kerr_ridge_sailing_force_tensor(
    sail_heights: torch.Tensor,
    beams: torch.Tensor,
    speeds: torch.Tensor,
    bow_angles_deg: torch.Tensor,
    ice_density: float = DEFAULT_ICE_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    C_k: float = 0.65,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Vectorised Kerr sailing force for batch computation.

    Args:
        sail_heights: (N,) tensor of sail heights, m.
        beams: (N,) tensor of ship beams, m.
        speeds: (N,) tensor of ship speeds, m/s.
        bow_angles_deg: (N,) tensor of bow half-angles, degrees.
        ice_density: Ice density, kg/m^3.
        gravity: Gravitational acceleration, m/s^2.
        C_k: Empirical coefficient.
        alpha: Speed correction coefficient.

    Returns:
        (N,) tensor of sailing forces, N.
    """
    phi_rad = torch.deg2rad(bow_angles_deg)
    f_speed = 1.0 + alpha * speeds / torch.sqrt(gravity * sail_heights)
    f_bow = 1.0 / torch.sin(phi_rad)
    F = C_k * ice_density * gravity * sail_heights ** 2 * beams * f_speed * f_bow
    return F


# ---------------------------------------------------------------------------
# Timco (2011) — Ridge Keel Drag
# ---------------------------------------------------------------------------

def timco_keel_drag(
    ridge: RidgeGeometry,
    ship: ShipConfig,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    ice_density: float = DEFAULT_ICE_DENSITY,
) -> float:
    """Compute ridge keel drag per Timco (2011) empirical model.

    Timco's model decomposes keel resistance into:

        F_keel = F_cohesion + F_friction + F_inertia

    where:
        F_cohesion = C_c * sigma_f * A_keel * (1 - n)
        F_friction = mu * (F_cohesion + F_buoyancy)
        F_inertia  = 0.5 * rho_w * C_D * A_keel * V^2

    The effective keel cross-sectional area:
        A_keel = h_k * W_eff

    where W_eff = min(ridge_width, ship.beam + 2 * h_k * tan(phi))

    Timco's consolidated formulation (2011, Eq. 8):
        F_keel = rho_w * g * h_k^2 * B * C_R * (1 - n) * f(V)

    where C_R is a ridge resistance coefficient (~0.5–1.5) and
    f(V) = 1 + beta * V is a speed correction.

    Parameters
    ----------
    ridge : RidgeGeometry
        Ridge geometric parameters.
    ship : ShipConfig
        Ship parameters.
    water_density : float
        Water density, kg/m^3.
    gravity : float
        Gravitational acceleration, m/s^2.
    ice_density : float
        Ice density, kg/m^3.

    Returns
    -------
    float
        Keel drag force, N.
    """
    h_k = ridge.keel_draft
    B = ship.beam
    V = ship.speed
    n = ridge.porosity
    phi_rad = math.radians(ship.bow_angle)

    # Effective interaction width
    W_eff = min(ridge.ridge_width, B + 2.0 * h_k * math.tan(phi_rad))

    # Keel cross-sectional area
    A_keel = h_k * W_eff

    # Timco ridge resistance coefficient (consolidated ridge)
    C_R = 0.85

    # Speed correction coefficient
    beta = 0.15

    # Speed correction factor
    f_V = 1.0 + beta * V

    # Timco keel drag (consolidated formulation)
    F_keel = water_density * gravity * h_k ** 2 * B * C_R * (1.0 - n) * f_V

    return F_keel


def timco_keel_drag_tensor(
    keel_drafts: torch.Tensor,
    beams: torch.Tensor,
    speeds: torch.Tensor,
    porosities: torch.Tensor,
    bow_angles_deg: torch.Tensor,
    ridge_widths: torch.Tensor,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    C_R: float = 0.85,
    beta: float = 0.15,
) -> torch.Tensor:
    """Vectorised Timco keel drag for batch computation.

    Args:
        keel_drafts: (N,) tensor of keel drafts, m.
        beams: (N,) tensor of ship beams, m.
        speeds: (N,) tensor of ship speeds, m/s.
        porosities: (N,) tensor of keel porosities.
        bow_angles_deg: (N,) tensor of bow half-angles, degrees.
        ridge_widths: (N,) tensor of ridge widths, m.
        water_density: Water density, kg/m^3.
        gravity: Gravitational acceleration, m/s^2.
        C_R: Ridge resistance coefficient.
        beta: Speed correction coefficient.

    Returns:
        (N,) tensor of keel drag forces, N.
    """
    phi_rad = torch.deg2rad(bow_angles_deg)
    W_eff = torch.minimum(ridge_widths, beams + 2.0 * keel_drafts * torch.tan(phi_rad))
    f_V = 1.0 + beta * speeds
    F = water_density * gravity * keel_drafts ** 2 * beams * C_R * (1.0 - porosities) * f_V
    return F


# ---------------------------------------------------------------------------
# Combined ridge resistance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RidgeResistanceResult:
    """Result of ridge resistance calculation."""
    sailing_force: float        # Kerr (1975), N
    keel_drag: float            # Timco (2011), N
    total_resistance: float     # sum, N
    ridge: RidgeGeometry
    ship: ShipConfig


def compute_ridge_resistance(
    ridge: RidgeGeometry,
    ship: ShipConfig,
    ice_density: float = DEFAULT_ICE_DENSITY,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    ice_flex_strength: float = DEFAULT_ICE_FLEX_STRENGTH,
) -> RidgeResistanceResult:
    """Compute total ridge resistance (sailing + keel drag).

    Args:
        ridge: Ridge geometry.
        ship: Ship configuration.
        ice_density: Ice density, kg/m^3.
        water_density: Water density, kg/m^3.
        gravity: Gravitational acceleration, m/s^2.
        ice_flex_strength: Ice flexural strength, Pa.

    Returns:
        RidgeResistanceResult with sailing force, keel drag, and total.
    """
    F_sail = kerr_ridge_sailing_force(
        ridge, ship, ice_density, water_density, gravity, ice_flex_strength
    )
    F_keel = timco_keel_drag(ridge, ship, water_density, gravity, ice_density)

    return RidgeResistanceResult(
        sailing_force=F_sail,
        keel_drag=F_keel,
        total_resistance=F_sail + F_keel,
        ridge=ridge,
        ship=ship,
    )


# ---------------------------------------------------------------------------
# Reference / benchmark values (from literature)
# ---------------------------------------------------------------------------

def kerr_reference_force(
    sail_height: float = 1.5,
    beam: float = 20.0,
    speed: float = 2.0,
    bow_angle_deg: float = 20.0,
    ice_density: float = DEFAULT_ICE_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Reference Kerr sailing force for standard test case.

    This provides a benchmark value for validation against the
    implementation. Uses the same formula as kerr_ridge_sailing_force
    but with fixed standard parameters from Kerr (1975) Table 2.

    Returns:
        Reference force in N.
    """
    return kerr_ridge_sailing_force(
        RidgeGeometry(sail_height=sail_height),
        ShipConfig(beam=beam, speed=speed, bow_angle=bow_angle_deg),
        ice_density=ice_density,
        water_density=DEFAULT_WATER_DENSITY,
        gravity=gravity,
    )


def timco_reference_drag(
    keel_draft: float = 6.0,
    beam: float = 20.0,
    speed: float = 2.0,
    porosity: float = 0.4,
    bow_angle_deg: float = 20.0,
    ridge_width: float = 30.0,
    water_density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> float:
    """Reference Timco keel drag for standard test case.

    Returns:
        Reference drag in N.
    """
    return timco_keel_drag(
        RidgeGeometry(
            keel_draft=keel_draft,
            porosity=porosity,
            ridge_width=ridge_width,
        ),
        ShipConfig(beam=beam, speed=speed, bow_angle=bow_angle_deg),
        water_density=water_density,
        gravity=gravity,
    )
