"""M3: Broken ice channel clearing rate and Lindqvist comparison.

Implements and verifies:
- Broken ice field simulation (C_i=20-80%, h_i=0.5-2m)
- Channel clearing rate: eta_clear = L_clear / L_total
- Ice resistance: R_ice = sum(F_contact)
- Verification: R_ice vs Lindqvist empirical formula (≤30%)
- Negative control: C_i=0 → R_ice=0

References:
- Lindqvist, M. (1989): "A direct method for ice resistance prediction"
- Riska, K. et al. (2003): "Ice resistance components"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class IceFloe:
    """Single ice floe in the broken ice field."""
    x: float           # center x, m
    y: float           # center y, m
    radius: float      # m
    thickness: float   # m
    vx: float = 0.0    # velocity x, m/s
    vy: float = 0.0    # velocity y, m/s
    mass: float = 0.0  # kg (computed from geometry)
    
    def __post_init__(self):
        if self.mass == 0.0:
            self.mass = 917.0 * math.pi * self.radius**2 * self.thickness


@dataclass
class ShipTrajectory:
    """Ship trajectory through ice field."""
    x: float           # bow x, m
    y: float           # center y, m
    speed: float       # m/s
    beam: float        # m
    entry_angle_deg: float  # degrees
    bow_radius: float  # m
    
    @property
    def entry_angle_rad(self) -> float:
        return math.radians(self.entry_angle_deg)


@dataclass
class BrokenIceFieldConfig:
    """Configuration for broken ice field simulation."""
    domain_length: float = 500.0    # m
    domain_width: float = 100.0     # m
    concentration: float = 0.5      # 0-1
    mean_floe_radius: float = 3.0   # m
    radius_std: float = 1.0         # m
    ice_thickness: float = 1.0      # m
    compressive_strength: float = 5.0e6  # Pa
    friction_coefficient: float = 0.1
    ship_speed: float = 3.0         # m/s (~6 kn)
    ship_beam: float = 26.0         # m
    ship_entry_angle: float = 18.0  # degrees
    ship_bow_radius: float = 15.0   # m
    seed: int = 42


@dataclass
class ChannelClearingResult:
    """Result of channel clearing simulation."""
    time: np.ndarray              # s
    ship_x: np.ndarray            # ship bow x position, m
    L_clear: np.ndarray           # cleared length, m
    eta_clear: np.ndarray         # clearing rate, dimensionless
    R_ice: np.ndarray             # ice resistance, N
    R_ice_mean: float             # mean ice resistance, N
    R_ice_std: float              # std ice resistance, N
    R_ice_peak: float             # peak ice resistance, N
    floe_positions: List[np.ndarray]  # floe positions at each time step


def generate_ice_field(config: BrokenIceFieldConfig) -> List[IceFloe]:
    """Generate random broken ice field.
    
    Args:
        config: Field configuration
    
    Returns:
        List of IceFloe objects
    """
    rng = np.random.RandomState(config.seed)
    
    # Domain area
    A_domain = config.domain_length * config.domain_width
    
    # Mean floe area
    A_floe_mean = math.pi * config.mean_floe_radius**2
    
    # Number of floes from concentration
    N_floes = int(config.concentration * A_domain / A_floe_mean)
    
    floes = []
    for _ in range(N_floes):
        # Random radius (truncated normal)
        r = rng.normal(config.mean_floe_radius, config.radius_std)
        r = max(0.5 * config.mean_floe_radius, min(2.0 * config.mean_floe_radius, r))
        
        # Random position
        x = rng.uniform(0, config.domain_length)
        y = rng.uniform(0, config.domain_width)
        
        floe = IceFloe(
            x=x,
            y=y,
            radius=r,
            thickness=config.ice_thickness,
        )
        floes.append(floe)
    
    return floes


def ship_hull_signed_distance(
    ship: ShipTrajectory,
    x: float,
    y: float,
) -> float:
    """Signed distance from point to ship hull (negative = inside).
    
    Simplified model: wedge bow + parallel body.
    
    Args:
        ship: Ship trajectory
        x: Point x coordinate, m
        y: Point y coordinate, m
    
    Returns:
        Signed distance, m (negative inside hull)
    """
    # Parallel body (rectangle)
    half_beam = ship.beam / 2.0
    body_length = 3.0 * ship.beam  # simplified
    
    # Check if in parallel body region
    if ship.x - body_length <= x <= ship.x:
        if abs(y - ship.y) <= half_beam:
            return -min(half_beam - abs(y - ship.y), ship.x - x)
    
    # Wedge bow (triangle)
    bow_length = ship.beam / (2.0 * math.tan(ship.entry_angle_rad))
    if ship.x <= x <= ship.x + bow_length:
        # Half-width at this x
        frac = (x - ship.x) / bow_length
        half_width = half_beam * (1.0 - frac)
        if abs(y - ship.y) <= half_width:
            return -min(half_width - abs(y - ship.y), (x - ship.x) * 0.1)
    
    # Outside hull
    return 1.0


def floe_ship_contact_force(
    floe: IceFloe,
    ship: ShipTrajectory,
    sigma_c: float,
    mu: float,
) -> Tuple[float, float]:
    """Compute floe-ship contact force.
    
    Crushing-limited: F_n = sigma_c * h * w(delta)
    where w = 2*sqrt(2*R*delta) is the Hertzian chord width.
    
    Args:
        floe: Ice floe
        ship: Ship trajectory
        sigma_c: Ice compressive strength, Pa
        mu: Friction coefficient
    
    Returns:
        (F_n, F_t): Normal and tangential forces, N
    """
    # Signed distance
    d = ship_hull_signed_distance(ship, floe.x, floe.y)
    
    if d >= 0:
        return 0.0, 0.0
    
    # Overlap
    delta = -d
    
    # Effective radius (floe as disk, ship bow as cylinder)
    R_eff = floe.radius
    
    # Contact width (Hertzian chord)
    w = min(2.0 * math.sqrt(2.0 * R_eff * delta), 2.0 * floe.radius)
    
    # Normal force (crushing-limited)
    F_n = sigma_c * floe.thickness * w
    
    # Tangential force (Coulomb friction)
    F_t = mu * F_n
    
    return F_n, F_t


def broken_ice_field_simulation(
    config: BrokenIceFieldConfig,
    dt: float = 0.01,
    t_total: float = 60.0,
) -> ChannelClearingResult:
    """Simulate ship transit through broken ice field.
    
    Args:
        config: Field configuration
        dt: Time step, s
        t_total: Total simulation time, s
    
    Returns:
        ChannelClearingResult with time histories
    """
    # Generate ice field
    floes = generate_ice_field(config)
    
    # Ship trajectory
    ship = ShipTrajectory(
        x=-config.ship_beam,
        y=config.domain_width / 2.0,
        speed=config.ship_speed,
        beam=config.ship_beam,
        entry_angle_deg=config.ship_entry_angle,
        bow_radius=config.ship_bow_radius,
    )
    
    # Time integration
    n_steps = int(t_total / dt)
    time = np.linspace(0, t_total, n_steps)
    
    ship_x = np.zeros(n_steps)
    L_clear = np.zeros(n_steps)
    eta_clear = np.zeros(n_steps)
    R_ice = np.zeros(n_steps)
    floe_positions = []
    
    x_start = ship.x
    
    for i in range(n_steps):
        # Update ship position
        ship.x += config.ship_speed * dt
        ship_x[i] = ship.x
        
        # Compute ice resistance
        F_total = 0.0
        for floe in floes:
            F_n, F_t = floe_ship_contact_force(
                floe, ship,
                config.compressive_strength,
                config.friction_coefficient,
            )
            # Project force onto ship direction
            F_total += F_n + F_t
        
        R_ice[i] = F_total
        
        # Update floe positions (simplified: push away from ship)
        for floe in floes:
            d = ship_hull_signed_distance(ship, floe.x, floe.y)
            if d < 0:
                # Push floe sideways
                if floe.y < ship.y:
                    floe.y -= config.ship_speed * dt * 0.5
                else:
                    floe.y += config.ship_speed * dt * 0.5
        
        # Channel clearing: count floes cleared (moved past ship)
        n_cleared = sum(1 for f in floes if f.x < ship.x - 2.0 * ship.beam)
        L_clear[i] = ship.x - x_start
        eta_clear[i] = n_cleared / len(floes) if len(floes) > 0 else 0.0
        
        # Store floe positions (every 10 steps to save memory)
        if i % 10 == 0:
            pos = np.array([[f.x, f.y, f.radius] for f in floes])
            floe_positions.append(pos)
    
    # Statistics (steady state: last 50% of simulation)
    i_steady = n_steps // 2
    R_ice_mean = np.mean(R_ice[i_steady:])
    R_ice_std = np.std(R_ice[i_steady:])
    R_ice_peak = np.max(R_ice[i_steady:])
    
    return ChannelClearingResult(
        time=time,
        ship_x=ship_x,
        L_clear=L_clear,
        eta_clear=eta_clear,
        R_ice=R_ice,
        R_ice_mean=R_ice_mean,
        R_ice_std=R_ice_std,
        R_ice_peak=R_ice_peak,
        floe_positions=floe_positions,
    )


def lindqvist_ice_resistance(
    ship_beam: float,
    ship_entry_angle_deg: float,
    ice_thickness: float,
    ice_flexural_strength: float,
    ice_compressive_strength: float,
    ship_speed: float,
    concentration: float = 0.5,
) -> float:
    """Lindqvist (1989) ice resistance formula for broken ice.
    
    R_ice = C_b * sigma_f * h^2 * B * (1 + v/v_ref) * C_i
    
    where C_b is an empirical coefficient for broken ice (~0.3-0.5).
    
    Args:
        ship_beam: Ship beam, m
        ship_entry_angle_deg: Waterline entry angle, degrees
        ice_thickness: Ice thickness, m
        ice_flexural_strength: Flexural strength, Pa
        ice_compressive_strength: Compressive strength, Pa
        ship_speed: Ship speed, m/s
        concentration: Ice concentration, 0-1
    
    Returns:
        R_ice: Ice resistance, N
    """
    # Empirical coefficient for broken ice (calibrated to MT Uikku)
    C_b = 0.35
    
    # Reference speed
    v_ref = 1.0  # m/s
    
    # Broken ice resistance (Lindqvist-style)
    # R = C_b * sigma_f * h^2 * B * (1 + v/v_ref) * C_i
    R_ice = C_b * ice_flexural_strength * ice_thickness**2 * ship_beam * \
            (1.0 + ship_speed / v_ref) * concentration
    
    return R_ice


def channel_clearing_rate(result: ChannelClearingResult) -> float:
    """Compute final channel clearing rate.
    
    eta_clear = L_clear / L_total
    
    Args:
        result: Simulation result
    
    Returns:
        eta_clear: Clearing rate, dimensionless (0-1)
    """
    return result.eta_clear[-1]


def ice_resistance_lindqvist_comparison(
    config: BrokenIceFieldConfig,
) -> dict:
    """Compare DEM ice resistance with Lindqvist empirical formula.
    
    Criteria:
    - R_ice vs Lindqvist deviation ≤ 30%
    - Negative control: C_i=0 → R_ice=0
    
    Args:
        config: Field configuration
    
    Returns:
        dict with comparison results
    """
    # DEM simulation
    result = broken_ice_field_simulation(config, dt=0.01, t_total=60.0)
    
    # Lindqvist prediction
    R_lindqvist = lindqvist_ice_resistance(
        ship_beam=config.ship_beam,
        ship_entry_angle_deg=config.ship_entry_angle,
        ice_thickness=config.ice_thickness,
        ice_flexural_strength=1.0e6,  # 1 MPa typical
        ice_compressive_strength=config.compressive_strength,
        ship_speed=config.ship_speed,
        concentration=config.concentration,
    )
    
    # Comparison
    R_dem = result.R_ice_mean
    deviation = abs(R_dem - R_lindqvist) / R_lindqvist if R_lindqvist > 0 else 0.0
    
    # Negative control: C_i=0 → R_ice=0
    config_zero = BrokenIceFieldConfig(
        concentration=0.0,
        seed=config.seed,
    )
    result_zero = broken_ice_field_simulation(config_zero, dt=0.01, t_total=10.0)
    negative_control_pass = (result_zero.R_ice_mean < 1.0)  # < 1 N
    
    return {
        "R_dem_mean_N": R_dem,
        "R_dem_std_N": result.R_ice_std,
        "R_dem_peak_N": result.R_ice_peak,
        "R_lindqvist_N": R_lindqvist,
        "deviation": deviation,
        "deviation_criterion_pass": deviation <= 0.30,
        "eta_clear_final": channel_clearing_rate(result),
        "negative_control_pass": negative_control_pass,
        "R_zero_concentration_N": result_zero.R_ice_mean,
    }


def verify_channel_clearing(
    concentration: float = 0.5,
    ice_thickness: float = 1.0,
    ship_speed: float = 3.0,
) -> dict:
    """Verify channel clearing with Lindqvist comparison.
    
    Args:
        concentration: Ice concentration, 0-1
        ice_thickness: Ice thickness, m
        ship_speed: Ship speed, m/s
    
    Returns:
        dict with verification results
    """
    config = BrokenIceFieldConfig(
        concentration=concentration,
        ice_thickness=ice_thickness,
        ship_speed=ship_speed,
        seed=42,
    )
    
    return ice_resistance_lindqvist_comparison(config)
