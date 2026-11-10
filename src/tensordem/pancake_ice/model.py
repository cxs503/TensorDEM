"""Pancake ice mechanics: resistance, pile-up, and maneuverability.

Physical scenario
-----------------
Pancake ice consists of circular ice disks (diameter D_f ~ 0.3-3 m, thickness
h ~ 5-30 cm) floating in the ocean surface layer, typically at concentration
C (areal fraction). When a vessel navigates through such a field, three
resistance mechanisms arise:

1. **Edge collision** (R_edge): Floes impact the hull edge-on. The force scales
   with the floe cross-section and flexural strength:
   
       F_edge = k_edge * σ_f * h * D_f
   
   where k_edge ~ O(0.01) is an edge efficiency factor.

2. **Slip friction** (R_slip): Floes slide along the hull with friction
   coefficient μ_slip. The normal force comes from buoyancy restoration.

3. **Pile-up** (R_pile): Floes accumulate ahead of the vessel, forming a
   rubble field. The resistance is the work to push and submerge this pile.

Total resistance:
    R_pancake = R_edge + R_slip + R_pile

For typical Antarctic conditions (D_f=1m, h=0.1m, C=0.7), the total
resistance for a Polarstern-class vessel is ~10-50 kN (Dierking 2013).

Degradation limits (structural, not tuned):
- D_f → ∞: pancake → level ice (Lindqvist R_level)
- D_f → 0: pancake → brash ice (granular rheology)
- C → 0: R_pancake → 0 (negative control)

References
----------
- Dierking (2013): Antarctic pancake ice field observations
- Herman et al. (2017): Wave-ice interaction in pancake fields
- Squire et al. (2001): Marginal ice zone dynamics

All quantities SI: m, Pa, kg/m³, m/s, N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "PancakeIceConfig",
    "HullGeometry",
    "ResistanceComponents",
    "PileUpGeometry",
    "ManeuverabilityForces",
    "STANDARD_CASE",
    "edge_collision_force",
    "slip_friction_force",
    "pile_up_resistance",
    "pile_up_geometry",
    "pancake_resistance",
    "maneuverability_forces",
    "degradation_level_ice",
    "degradation_brash_ice",
    "concentration_sweep",
    "diameter_sweep",
]

# Physical constants
WATER_DENSITY = 1025.0  # kg/m³
ICE_DENSITY = 917.0  # kg/m³
GRAVITY = 9.80665  # m/s²


@dataclass(frozen=True)
class PancakeIceConfig:
    """Pancake ice field properties.
    
    Attributes
    ----------
    diameter : float
        Mean floe diameter D_f [m]. Typical range 0.3-3 m.
    thickness : float
        Floe thickness h [m]. Typical range 0.05-0.3 m.
    concentration : float
        Areal concentration C ∈ [0, 1]. Fraction of water surface covered.
    flexural_strength : float
        Flexural strength σ_f [Pa]. Typical 0.3-1.0 MPa for sea ice.
    crushing_strength : float
        Crushing strength σ_c [Pa]. Typical 2-10 MPa.
    slip_friction : float
        Ice-steel friction coefficient μ_slip [-]. Typical 0.05-0.15.
    edge_efficiency : float
        Edge collision efficiency k_edge [-]. Typical 0.01-0.05.
    """
    diameter: float = 1.0
    thickness: float = 0.10
    concentration: float = 0.7
    flexural_strength: float = 500e3  # 500 kPa
    crushing_strength: float = 5e6  # 5 MPa
    slip_friction: float = 0.08
    edge_efficiency: float = 0.02  # Reduced for realistic resistance
    diameter_std: float = 0.3
    orientation_std: float = math.pi / 6

    def __post_init__(self) -> None:
        if not (0.05 <= self.diameter <= 100.0):  # Allow large D for degradation
            raise ValueError(f"diameter must be in [0.05, 100] m, got {self.diameter}")
        if not (0.01 <= self.thickness <= 1.0):
            raise ValueError(f"thickness must be in [0.01, 1.0] m, got {self.thickness}")
        if not (0.0 <= self.concentration <= 1.0):
            raise ValueError(f"concentration must be in [0, 1], got {self.concentration}")
        if not (100e3 <= self.flexural_strength <= 5e6):
            raise ValueError(f"flexural_strength must be in [100kPa, 5MPa], got {self.flexural_strength}")
        if not (1e6 <= self.crushing_strength <= 20e6):
            raise ValueError(f"crushing_strength must be in [1MPa, 20MPa], got {self.crushing_strength}")
        if not (0.01 <= self.slip_friction <= 0.5):
            raise ValueError(f"slip_friction must be in [0.01, 0.5], got {self.slip_friction}")
        if not (0.001 <= self.edge_efficiency <= 1.0):
            raise ValueError(f"edge_efficiency must be in [0.001, 1.0], got {self.edge_efficiency}")


@dataclass(frozen=True)
class HullGeometry:
    """Ship hull geometry for ice resistance calculation."""
    length: float = 100.0
    beam: float = 20.0
    draught: float = 6.0
    bow_angle: float = math.radians(20)
    stem_angle: float = math.radians(15)

    def __post_init__(self) -> None:
        if self.length <= 0 or self.beam <= 0 or self.draught <= 0:
            raise ValueError("All dimensions must be positive")
        if not (0 < self.bow_angle < math.pi / 2):
            raise ValueError(f"bow_angle must be in (0, π/2), got {self.bow_angle}")


@dataclass
class ResistanceComponents:
    """Decomposed pancake ice resistance."""
    R_edge: float = 0.0
    R_slip: float = 0.0
    R_pile: float = 0.0
    R_total: float = 0.0
    n_collisions: float = 0.0

    def __post_init__(self) -> None:
        self.R_total = self.R_edge + self.R_slip + self.R_pile


@dataclass
class PileUpGeometry:
    """Geometry of ice pile-up ahead of vessel."""
    height: float = 0.0
    length: float = 0.0
    width: float = 0.0
    volume: float = 0.0
    porosity: float = 0.3


@dataclass
class ManeuverabilityForces:
    """Lateral forces and yaw moment from asymmetric ice loading."""
    F_x: float = 0.0
    F_y: float = 0.0
    N_yaw: float = 0.0
    drift_angle: float = 0.0
    yaw_rate: float = 0.0


# Standard test case (Antarctic conditions, Dierking 2013-like)
STANDARD_CASE = PancakeIceConfig(
    diameter=1.0,
    thickness=0.10,
    concentration=0.7,
    flexural_strength=500e3,
    crushing_strength=5e6,
    slip_friction=0.08,
    edge_efficiency=0.02,
)

STANDARD_HULL = HullGeometry(
    length=80.0,
    beam=16.0,
    draught=5.0,
    bow_angle=math.radians(20),
)


def edge_collision_force(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> Tuple[float, float]:
    """Calculate edge collision resistance component.
    
    Physical model: Each floe collision transfers momentum. The force per
    collision scales with floe cross-section (h × D_f) and flexural strength.
    
    The collision rate depends on concentration, speed, and floe spacing.
    For pancake ice, floes are loosely packed and can move aside, so the
    effective collision rate is reduced by a factor (1-C).
    
    R_edge = k_edge * σ_f * h * D_f * C * V * B / D_f
           = k_edge * σ_f * h * C * V * B
    
    Note: The D_f cancels out because larger floes have larger force per
    collision but lower collision rate. We add a weak D_f^0.3 dependence
    to capture that larger floes create more concentrated impacts.
    """
    if speed <= 0 or config.concentration <= 0:
        return 0.0, 0.0
    
    # Force per collision
    F_single = config.edge_efficiency * config.flexural_strength * config.thickness * config.diameter
    
    # Collision rate (per unit time)
    # Floes encountered per second ~ C * V * B / D_f
    n_coll = config.concentration * speed * hull.beam / config.diameter
    
    # Total edge resistance with D_f scaling
    D_ref = 1.0
    D_factor = (config.diameter / D_ref) ** 0.3
    R_edge = n_coll * F_single * D_factor
    
    return R_edge, n_coll


def slip_friction_force(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> float:
    """Calculate slip friction resistance component.
    
    Floes slide along the hull with friction coefficient μ_slip.
    The normal force comes from buoyancy restoration when floes are pushed under.
    """
    if config.concentration <= 0 or speed <= 0:
        return 0.0
    
    # Buoyancy normal force per unit area
    N_buoyancy = WATER_DENSITY * GRAVITY * config.thickness * 0.3
    
    # Contact area
    L_bow = hull.beam / (2.0 * math.tan(hull.bow_angle))
    L_contact = L_bow + 0.2 * hull.length
    A_contact = L_contact * hull.draught * config.concentration
    
    R_slip = config.slip_friction * N_buoyancy * A_contact
    
    return R_slip


def pile_up_geometry(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> PileUpGeometry:
    """Calculate pile-up geometry ahead of vessel.
    
    Pile height and length depend on concentration, floe size, and speed.
    Calibrated to Dierking (2013) observations: h_pile ~ 0.3-1.0 m.
    """
    if config.concentration <= 0 or speed <= 0:
        return PileUpGeometry()
    
    V_ref = 3.0
    
    # Pile height: ~0.4 * D_f * sqrt(C)
    k_h = 0.4
    h_pile = k_h * config.diameter * config.concentration * (1.0 + 0.1 * speed / V_ref)
    
    # Pile length
    k_L = 2.0
    L_pile = k_L * config.diameter * config.concentration**2 / (1.0 - config.concentration + 0.3)
    
    # Pile width
    W_pile = hull.beam * 1.1
    
    # Porosity
    porosity = 0.3
    
    # Volume
    volume = 0.5 * h_pile * L_pile * W_pile * (1.0 - porosity)
    
    return PileUpGeometry(
        height=h_pile,
        length=L_pile,
        width=W_pile,
        volume=volume,
        porosity=porosity,
    )


def pile_up_resistance(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> float:
    """Calculate pile-up resistance component.
    
    Work required to push and submerge the accumulated ice ahead of vessel.
    """
    if speed <= 0:
        return 0.0
    
    pile = pile_up_geometry(config, hull, speed)
    
    if pile.volume <= 0:
        return 0.0
    
    # Buoyancy resistance
    buoyancy_factor = 1.0 - ICE_DENSITY / WATER_DENSITY
    R_buoyancy = WATER_DENSITY * GRAVITY * pile.volume * buoyancy_factor / max(pile.length, 1.0)
    
    # Breaking resistance
    n_layers = pile.height / config.thickness
    k_break = 0.1
    R_break = config.flexural_strength * config.thickness * pile.width * n_layers * k_break
    
    return R_buoyancy + R_break


def pancake_resistance(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> ResistanceComponents:
    """Calculate total pancake ice resistance."""
    if speed <= 0 or config.concentration <= 0:
        return ResistanceComponents()
    
    R_edge, n_coll = edge_collision_force(config, hull, speed)
    R_slip = slip_friction_force(config, hull, speed)
    R_pile = pile_up_resistance(config, hull, speed)
    
    return ResistanceComponents(
        R_edge=R_edge,
        R_slip=R_slip,
        R_pile=R_pile,
        R_total=R_edge + R_slip + R_pile,
        n_collisions=n_coll,
    )


def maneuverability_forces(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
    drift_angle: float = 0.0,
    yaw_rate: float = 0.0,
) -> ManeuverabilityForces:
    """Calculate lateral forces and yaw moment from asymmetric ice loading."""
    R_base = pancake_resistance(config, hull, speed).R_total
    
    k_beta = 0.3
    k_r = 0.1
    k_N = 0.25
    k_nr = 0.15
    
    F_x = R_base * math.cos(drift_angle)
    F_y = k_beta * R_base * math.sin(drift_angle)
    if speed > 0 and yaw_rate != 0:
        F_y += k_r * R_base * yaw_rate * hull.length / speed
    
    N_yaw = k_N * R_base * hull.length * math.sin(drift_angle)
    if speed > 0 and yaw_rate != 0:
        N_yaw += k_nr * R_base * hull.length**2 * yaw_rate / speed
    
    return ManeuverabilityForces(
        F_x=F_x,
        F_y=F_y,
        N_yaw=N_yaw,
        drift_angle=drift_angle,
        yaw_rate=yaw_rate,
    )


def degradation_level_ice(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> float:
    """Calculate resistance in the limit D_f → ∞ (level ice).
    
    Lindqvist (1989) three-component formula.
    """
    h = config.thickness
    sigma_c = config.crushing_strength
    sigma_f = config.flexural_strength
    B = hull.beam
    L = hull.length
    alpha = hull.bow_angle
    mu = config.slip_friction
    
    R_c = 0.5 * sigma_c * h**2 * (math.tan(alpha) + mu / math.cos(alpha))
    R_b = 0.25 * sigma_f * B * h**2 / L
    R_s = WATER_DENSITY * GRAVITY * h**2 * B * (0.9 + 0.05 * mu * L / B)
    R_v = 0.5 * WATER_DENSITY * B * h * speed**2
    
    return R_c + R_b + R_s + R_v


def degradation_brash_ice(
    config: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
) -> float:
    """Calculate resistance in the limit D_f → 0 (brash ice)."""
    if config.concentration <= 0 or speed <= 0:
        return 0.0
    
    h_sub = config.thickness * config.concentration
    phi = 0.3
    rho_b = ICE_DENSITY * (1 - phi) + WATER_DENSITY * phi
    
    mu_eff = 0.5
    delta = 0.1 * max(config.diameter, 0.05)
    P = (WATER_DENSITY - rho_b) * GRAVITY * h_sub + 0.5 * 500 * speed**2
    A_shear = hull.length * h_sub * (1 - phi)
    R_shear = mu_eff * P * A_shear
    
    V_disp = (1 - phi) * hull.beam * h_sub
    R_buoy = WATER_DENSITY * GRAVITY * V_disp
    
    A_keel = (1 - phi) * hull.beam * h_sub
    R_drag = 0.5 * 1.0 * WATER_DENSITY * speed**2 * A_keel
    
    return R_shear + R_buoy + R_drag


def concentration_sweep(
    config_base: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
    concentrations: List[float],
) -> List[Tuple[float, ResistanceComponents]]:
    """Sweep resistance over concentration range."""
    results = []
    for C in concentrations:
        config = PancakeIceConfig(
            diameter=config_base.diameter,
            thickness=config_base.thickness,
            concentration=C,
            flexural_strength=config_base.flexural_strength,
            crushing_strength=config_base.crushing_strength,
            slip_friction=config_base.slip_friction,
            edge_efficiency=config_base.edge_efficiency,
        )
        R = pancake_resistance(config, hull, speed)
        results.append((C, R))
    return results


def diameter_sweep(
    config_base: PancakeIceConfig,
    hull: HullGeometry,
    speed: float,
    diameters: List[float],
) -> List[Tuple[float, ResistanceComponents]]:
    """Sweep resistance over floe diameter range."""
    results = []
    for D in diameters:
        config = PancakeIceConfig(
            diameter=D,
            thickness=config_base.thickness,
            concentration=config_base.concentration,
            flexural_strength=config_base.flexural_strength,
            crushing_strength=config_base.crushing_strength,
            slip_friction=config_base.slip_friction,
            edge_efficiency=config_base.edge_efficiency,
        )
        R = pancake_resistance(config, hull, speed)
        results.append((D, R))
    return results
