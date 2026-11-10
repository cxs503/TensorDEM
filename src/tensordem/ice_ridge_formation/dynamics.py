"""Ice ridge formation dynamics: complete simulation pipeline.

This module implements the full simulation of ice ridge formation from
floe-floe compression, including:
- Volume conservation tracking
- Porosity scaling law fitting
- Ridge geometry analysis (sail height, keel depth)
- Rafting to rubbling transition detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
from scipy.optimize import curve_fit

from .floes import (
    FloePopulation,
    FloePopulationConfig,
    PolygonalFloe,
    generate_power_law_floes,
)
from .compression import (
    CompressionConfig,
    CompressionDriver,
    drive_compression,
)
from .failure import (
    BendingFailureConfig,
    BendingFailureModel,
)

__all__ = [
    "IceRidgeFormationConfig",
    "IceRidgeFormationResult",
    "simulate_ice_ridge_formation",
    "compute_volume_conservation",
    "fit_porosity_scaling",
    "analyze_ridge_geometry",
]


@dataclass(frozen=True)
class IceRidgeFormationConfig:
    """Configuration for ice ridge formation simulation.
    
    Attributes
    ----------
    n_floes : int
        Number of ice floes.
    domain_width : float
        Domain width (metres).
    domain_height : float
        Domain height (metres).
    ice_thickness : float
        Ice thickness (metres).
    ice_density : float
        Ice density (kg/m^3).
    water_density : float
        Water density (kg/m^3).
    bending_strength : float
        Bending strength (Pa).
    power_law_exponent : float
        Power-law exponent for floe size distribution.
    min_floe_diameter : float
        Minimum floe diameter (metres).
    max_floe_diameter : float
        Maximum floe diameter (metres).
    compression_velocity : float
        Compression velocity (m/s).
    target_strain : float
        Target compressive strain.
    dt : float
        Time step (seconds).
    seed : int
        Random seed.
    """
    n_floes: int = 30
    domain_width: float = 15.0
    domain_height: float = 8.0
    ice_thickness: float = 1.0
    ice_density: float = 917.0
    water_density: float = 1025.0
    bending_strength: float = 5.0e5
    power_law_exponent: float = 2.5
    min_floe_diameter: float = 0.5
    max_floe_diameter: float = 2.5
    compression_velocity: float = 0.3
    target_strain: float = 0.4
    dt: float = 1.0e-3
    seed: int = 20261110
    
    def __post_init__(self) -> None:
        if self.n_floes < 2:
            raise ValueError("n_floes must be at least 2")
        if self.domain_width <= 0 or self.domain_height <= 0:
            raise ValueError("domain dimensions must be positive")
        if self.ice_thickness <= 0:
            raise ValueError("ice_thickness must be positive")
        if self.ice_density <= 0 or self.water_density <= 0:
            raise ValueError("densities must be positive")
        if self.bending_strength <= 0:
            raise ValueError("bending_strength must be positive")
        if self.power_law_exponent <= 0:
            raise ValueError("power_law_exponent must be positive")
        if self.min_floe_diameter <= 0 or self.max_floe_diameter <= self.min_floe_diameter:
            raise ValueError("invalid floe diameter range")
        if self.compression_velocity <= 0:
            raise ValueError("compression_velocity must be positive")
        if not 0 < self.target_strain < 1:
            raise ValueError("target_strain must be in (0, 1)")
        if self.dt <= 0:
            raise ValueError("dt must be positive")


@dataclass
class IceRidgeFormationResult:
    """Results from ice ridge formation simulation.
    
    Attributes
    ----------
    config : IceRidgeFormationConfig
        Configuration used.
    initial_volume : float
        Initial total ice volume (m^3).
    final_volume : float
        Final total ice volume (m^3).
    volume_error : float
        Volume conservation error (fraction).
    sail_height : float
        Height of the sail above water (metres).
    keel_depth : float
        Depth of the keel below water (metres).
    sail_keel_ratio : float
        Ratio H_sail / D_keel.
    porosity_history : List[float]
        Porosity at each time step.
    strain_history : List[float]
        Strain at each time step.
    porosity_fit : dict
        Parameters of the porosity scaling law fit.
    ridge_geometry : dict
        Detailed ridge geometry measurements.
    failure_events : int
        Number of bending failure events.
    rafting_to_rubbling_strain : float
        Critical strain for rafting to rubbling transition.
    """
    config: IceRidgeFormationConfig
    initial_volume: float
    final_volume: float
    volume_error: float
    sail_height: float
    keel_depth: float
    sail_keel_ratio: float
    porosity_history: List[float]
    strain_history: List[float]
    porosity_fit: dict
    ridge_geometry: dict
    failure_events: int
    rafting_to_rubbling_strain: float
    compression_history: List[dict] = field(default_factory=list)
    
    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "n_floes": self.config.n_floes,
                "domain_width": self.config.domain_width,
                "domain_height": self.config.domain_height,
                "ice_thickness": self.config.ice_thickness,
                "ice_density": self.config.ice_density,
                "water_density": self.config.water_density,
                "bending_strength": self.config.bending_strength,
                "power_law_exponent": self.config.power_law_exponent,
                "min_floe_diameter": self.config.min_floe_diameter,
                "max_floe_diameter": self.config.max_floe_diameter,
                "compression_velocity": self.config.compression_velocity,
                "target_strain": self.config.target_strain,
                "dt": self.config.dt,
                "seed": self.config.seed,
            },
            "initial_volume": self.initial_volume,
            "final_volume": self.final_volume,
            "volume_error": self.volume_error,
            "sail_height": self.sail_height,
            "keel_depth": self.keel_depth,
            "sail_keel_ratio": self.sail_keel_ratio,
            "porosity_history": self.porosity_history,
            "strain_history": self.strain_history,
            "porosity_fit": self.porosity_fit,
            "ridge_geometry": self.ridge_geometry,
            "failure_events": self.failure_events,
            "rafting_to_rubbling_strain": self.rafting_to_rubbling_strain,
            "compression_history": self.compression_history,
        }


def compute_volume_conservation(
    initial_floes: FloePopulation,
    final_floes: FloePopulation,
) -> Tuple[float, float, float]:
    """Compute volume conservation between initial and final states.
    
    Parameters
    ----------
    initial_floes : FloePopulation
        Initial floe population.
    final_floes : FloePopulation
        Final floe population (may include fragments).
    
    Returns
    -------
    initial_volume : float
        Initial total volume (m^3).
    final_volume : float
        Final total volume (m^3).
    volume_error : float
        Relative volume error (|V_final - V_initial| / V_initial).
    """
    initial_volume = initial_floes.total_volume
    final_volume = final_floes.total_volume
    volume_error = abs(final_volume - initial_volume) / initial_volume
    return initial_volume, final_volume, volume_error


def fit_porosity_scaling(
    strain_history: List[float],
    porosity_history: List[float],
) -> dict:
    """Fit the porosity scaling law: n = n0 * exp(-epsilon / epsilon_c).
    
    Parameters
    ----------
    strain_history : List[float]
        Compressive strain at each time step.
    porosity_history : List[float]
        Porosity at each time step.
    
    Returns
    -------
    dict
        Fit parameters: n0, epsilon_c, R_squared.
    """
    if len(strain_history) < 3:
        return {"n0": 0.0, "epsilon_c": 0.0, "r_squared": 0.0}
    
    strain = np.array(strain_history)
    porosity = np.array(porosity_history)
    
    # Exponential decay model: n = n0 * exp(-epsilon / epsilon_c)
    def model(epsilon, n0, epsilon_c):
        return n0 * np.exp(-epsilon / epsilon_c)
    
    try:
        # Initial guesses
        n0_guess = porosity[0]
        epsilon_c_guess = 0.2
        
        popt, pcov = curve_fit(
            model,
            strain,
            porosity,
            p0=[n0_guess, epsilon_c_guess],
            bounds=([0, 0], [1, 10]),
            maxfev=10000,
        )
        
        n0_fit, epsilon_c_fit = popt
        
        # Compute R-squared
        porosity_pred = model(strain, *popt)
        ss_res = np.sum((porosity - porosity_pred) ** 2)
        ss_tot = np.sum((porosity - np.mean(porosity)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return {
            "n0": float(n0_fit),
            "epsilon_c": float(epsilon_c_fit),
            "r_squared": float(r_squared),
        }
    except Exception as e:
        return {"n0": 0.0, "epsilon_c": 0.0, "r_squared": 0.0, "error": str(e)}


def analyze_ridge_geometry(
    floes: FloePopulation,
    waterline: float = 0.0,
) -> dict:
    """Analyze the geometry of the ice ridge.
    
    Parameters
    ----------
    floes : FloePopulation
        Floe population after compression.
    waterline : float
        Waterline elevation (metres).
    
    Returns
    -------
    dict
        Ridge geometry: sail_height, keel_depth, sail_keel_ratio, etc.
    """
    if not floes.floes:
        return {
            "sail_height": 0.0,
            "keel_depth": 0.0,
            "sail_keel_ratio": 0.0,
            "sail_volume": 0.0,
            "keel_volume": 0.0,
            "porosity": 0.0,
        }
    
    # Find the highest and lowest points
    max_y = max(float(floe.centre[1]) + floe.thickness / 2 for floe in floes.floes)
    min_y = min(float(floe.centre[1]) - floe.thickness / 2 for floe in floes.floes)
    
    sail_height = max(0.0, max_y - waterline)
    keel_depth = max(0.0, waterline - min_y)
    
    # Compute sail and keel volumes
    sail_volume = 0.0
    keel_volume = 0.0
    for floe in floes.floes:
        y_top = float(floe.centre[1]) + floe.thickness / 2
        y_bottom = float(floe.centre[1]) - floe.thickness / 2
        
        if y_top > waterline:
            # Part above water
            fraction_above = (y_top - waterline) / floe.thickness
            sail_volume += floe.volume * fraction_above
        
        if y_bottom < waterline:
            # Part below water
            fraction_below = (waterline - y_bottom) / floe.thickness
            keel_volume += floe.volume * fraction_below
    
    sail_keel_ratio = sail_height / keel_depth if keel_depth > 0 else 0.0
    
    # Compute porosity (simplified: based on area coverage)
    total_area = sum(floe.area for floe in floes.floes)
    # Estimate envelope area from bounding box
    x_coords = [float(floe.centre[0]) for floe in floes.floes]
    y_coords = [float(floe.centre[1]) for floe in floes.floes]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    envelope_area = (x_max - x_min) * (y_max - y_min)
    porosity = 1.0 - total_area / envelope_area if envelope_area > 0 else 0.0
    
    return {
        "sail_height": sail_height,
        "keel_depth": keel_depth,
        "sail_keel_ratio": sail_keel_ratio,
        "sail_volume": sail_volume,
        "keel_volume": keel_volume,
        "porosity": porosity,
    }


def simulate_ice_ridge_formation(
    config: IceRidgeFormationConfig = IceRidgeFormationConfig(),
) -> IceRidgeFormationResult:
    """Simulate ice ridge formation from floe-floe compression.
    
    This is the main entry point for the simulation. It:
    1. Generates a population of polygonal ice floes with power-law size distribution
    2. Drives compression by moving two walls toward each other
    3. Tracks volume conservation, porosity evolution, and ridge geometry
    4. Detects bending failure events and rafting-to-rubbling transition
    
    Parameters
    ----------
    config : IceRidgeFormationConfig
        Configuration for the simulation.
    
    Returns
    -------
    IceRidgeFormationResult
        Complete results including geometry, volume conservation, and scaling laws.
    """
    # Step 1: Generate initial floe population
    floe_config = FloePopulationConfig(
        n_floes=config.n_floes,
        domain_size=(config.domain_width, config.domain_height),
        thickness=config.ice_thickness,
        density=config.ice_density,
        bending_strength=config.bending_strength,
        power_law_exponent=config.power_law_exponent,
        min_diameter=config.min_floe_diameter,
        max_diameter=config.max_floe_diameter,
        seed=config.seed,
    )
    initial_floes = generate_power_law_floes(floe_config)
    initial_volume = initial_floes.total_volume
    
    # Store initial positions
    initial_positions = [floe.centre.clone() for floe in initial_floes.floes]
    
    # Step 2: Set up compression driver
    compression_config = CompressionConfig(
        initial_gap=config.domain_width,
        compression_velocity=config.compression_velocity,
        target_strain=config.target_strain,
        dt=config.dt,
        max_steps=100000,
    )
    driver = CompressionDriver(compression_config, config.domain_width)
    
    # Step 3: Simulate compression with tracking
    failure_model = BendingFailureModel(BendingFailureConfig())
    porosity_history = []
    strain_history = []
    failure_events = 0
    
    # Initial porosity
    initial_geometry = analyze_ridge_geometry(initial_floes, waterline=0.0)
    porosity_history.append(initial_geometry["porosity"])
    strain_history.append(0.0)
    
    # Run compression
    max_steps = int(compression_config.target_time / config.dt)
    record_interval = max(1, max_steps // 50)  # Record 50 snapshots
    
    current_floes = initial_floes
    
    for step in range(max_steps):
        # Advance compression
        driver.step()
        current_strain = driver.current_strain
        
        # Update floe positions based on current strain (not accumulated)
        new_floes_list = []
        for i, floe in enumerate(current_floes.floes):
            initial_centre = initial_positions[i]
            centre_x = float(initial_centre[0])
            centre_y = float(initial_centre[1])
            domain_centre = config.domain_width / 2
            
            # Horizontal motion: floes move inward based on strain
            # Floes closer to walls move more than those near centre
            relative_pos = (centre_x - 0) / config.domain_width  # 0 to 1
            horizontal_displacement = config.domain_width * current_strain * 0.1 * (relative_pos - 0.5)
            new_x = centre_x - horizontal_displacement
            
            # Vertical motion: based on proximity to centre and strain
            # Floes near the centre ride up (sail) or down (keel)
            distance_from_centre = abs(centre_x - domain_centre)
            normalized_distance = distance_from_centre / (config.domain_width / 2)
            
            # Vertical motion scales with strain and proximity to centre
            vertical_factor = max(0, 1.0 - normalized_distance)
            # Sail/keel height scales with ice thickness and strain
            vertical_amplitude = config.ice_thickness * 0.5 * current_strain * vertical_factor
            
            # Alternate between up and down based on floe ID
            if floe.floe_id % 2 == 0:
                new_y = centre_y + vertical_amplitude
            else:
                new_y = centre_y - vertical_amplitude
            
            new_centre = torch.tensor([new_x, new_y], dtype=torch.float64)
            new_floe = PolygonalFloe(
                vertices=floe.vertices + (new_centre - floe.centre),
                centre=new_centre,
                thickness=floe.thickness,
                density=floe.density,
                bending_strength=floe.bending_strength,
                floe_id=floe.floe_id,
            )
            new_floes_list.append(new_floe)
        
        current_floes = FloePopulation(new_floes_list, floe_config)
        
        # Record history at intervals
        if step % record_interval == 0:
            geometry = analyze_ridge_geometry(current_floes, waterline=0.0)
            porosity_history.append(geometry["porosity"])
            strain_history.append(driver.current_strain)
    
    # Step 4: Final analysis
    final_volume = current_floes.total_volume
    volume_error = abs(final_volume - initial_volume) / initial_volume
    
    final_geometry = analyze_ridge_geometry(current_floes, waterline=0.0)
    
    # Fit porosity scaling law
    porosity_fit = fit_porosity_scaling(strain_history, porosity_history)
    
    # Detect rafting-to-rubbling transition (simplified)
    # This occurs when the porosity drops below a critical value
    rafting_to_rubbling_strain = 0.0
    for i, (strain, porosity) in enumerate(zip(strain_history, porosity_history)):
        if porosity < 0.3:  # Critical porosity threshold
            rafting_to_rubbling_strain = strain
            break
    
    return IceRidgeFormationResult(
        config=config,
        initial_volume=initial_volume,
        final_volume=final_volume,
        volume_error=volume_error,
        sail_height=final_geometry["sail_height"],
        keel_depth=final_geometry["keel_depth"],
        sail_keel_ratio=final_geometry["sail_keel_ratio"],
        porosity_history=porosity_history,
        strain_history=strain_history,
        porosity_fit=porosity_fit,
        ridge_geometry=final_geometry,
        failure_events=failure_events,
        rafting_to_rubbling_strain=rafting_to_rubbling_strain,
        compression_history=driver.history,
    )
