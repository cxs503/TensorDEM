"""Ice-structure mixed mode failure: crushing, bending, and splitting.

This module implements the three failure modes of ice when interacting with
offshore/marine structures, and computes the combined resistance as a function
of velocity, structure width, and ice properties.

Physical background
-------------------
When ice interacts with a structure, three failure modes can occur:

1. **Crushing** (low velocity, narrow structure):
   Ice fails in compression ahead of the structure. The resistance is
   proportional to the crushing strength σ_c and the contact area.
   
   R_crush = C_c · σ_c · h · D                    (1)
   
   where C_c is a crushing coefficient (~2-3), h is ice thickness, D is
   structure width.

2. **Bending** (high velocity, wide structure):
   Ice fails in flexure as it rides up on the structure. The resistance is
   proportional to the flexural strength σ_f.
   
   R_bend = C_b · σ_f · h² · (ρg/σ_f)^(1/3)     (2)
   
   where C_b is a bending coefficient, ρ is ice density, g is gravity.

3. **Splitting** (intermediate conditions):
   Ice fails in tension along radial cracks. The resistance is proportional
   to the tensile strength σ_t.
   
   R_split = C_s · σ_t · h · D                    (3)

Mode transition
---------------
The dominant failure mode depends on the velocity V, structure width L, and
ice properties. Timco (2001) gives the critical velocity for transition from
crushing to bending as:

   V_c ≈ 0.1 · (σ_f / (ρ · h))^(1/2)             (4)

For V < V_c, crushing dominates; for V > V_c, bending dominates.

Three-component resistance
--------------------------
The total resistance is the sum of all three components:

   R_total = R_crush + R_bend + R_split            (5)

Each component is weighted by a mode fraction that depends on the operating
conditions.

References
----------
- Timco, G.W., Frederking, R.M.W. (1990). A review of the world data on
  ice-structure interaction.
- Timco, G.W., Weeks, W.F. (2010). A review of the engineering properties
  of sea ice.
- Lindqvist, G. (1989). A direct method for predicting ice forces on ship
  hulls.
- ISO 19906:2010. Petroleum and natural gas industries — Fixed steel offshore
  structures in arctic and cold regions.

All quantities SI: m, kg, s, N, Pa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

__all__ = [
    "IceMixedModeConfig",
    "FailureMode",
    "MixedModeResult",
    "PhaseDiagramResult",
    "ModeTransitionResult",
    "compute_crushing_resistance",
    "compute_bending_resistance",
    "compute_splitting_resistance",
    "compute_total_resistance",
    "determine_dominant_mode",
    "compute_critical_velocity",
    "compute_phase_diagram",
    "compute_mode_transition",
    "validate_against_lindqvist",
    "validate_against_timco",
    "run_mixed_mode_campaign",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceMixedModeConfig:
    """Configuration for ice-structure mixed mode failure.
    
    Attributes:
        sigma_c: Uniaxial compressive strength of ice (Pa). Default 3 MPa.
        sigma_f: Flexural strength of ice (Pa). Default 0.7 MPa.
        sigma_t: Tensile strength of ice (Pa). Default 0.7 MPa.
        h: Ice thickness (m). Default 1.0 m.
        D: Structure characteristic width (m). Default 10.0 m.
        E: Elastic modulus of ice (Pa). Default 5 GPa.
        rho_ice: Ice density (kg/m³). Default 917.
        rho_water: Water density (kg/m³). Default 1025.
        nu: Poisson's ratio of ice. Default 0.33.
        mu: Ice-structure friction coefficient. Default 0.1.
        alpha: Structure slope angle (radians). Default 0 (vertical).
    """
    sigma_c: float = 3e6
    sigma_f: float = 0.7e6
    sigma_t: float = 0.7e6
    h: float = 1.0
    D: float = 10.0
    E: float = 5e9
    rho_ice: float = 917.0
    rho_water: float = 1025.0
    nu: float = 0.33
    mu: float = 0.1
    alpha: float = 0.0


class FailureMode:
    """Enumeration of ice failure modes."""
    CRUSHING = "crushing"
    BENDING = "bending"
    SPLITTING = "splitting"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Resistance calculations
# ---------------------------------------------------------------------------
def compute_crushing_resistance(
    config: IceMixedModeConfig,
    V: float,
) -> float:
    """Compute crushing resistance.
    
    Based on ISO 19906 and Korzhavin (1975):
    
        R_crush = C_c · σ_c · h · D · f(V)
    
    where C_c is a shape coefficient (~2.5 for cylinder), and f(V) is a
    velocity-dependent factor that accounts for strain-rate effects.
    
    Args:
        config: Ice and structure configuration.
        V: Ice drift velocity (m/s).
    
    Returns:
        Crushing resistance (N).
    """
    # Shape coefficient (cylinder)
    C_c = 2.5
    
    # Strain-rate effect (Afanas'eva 1971)
    # f(V) = 1 + 0.3 * V for V < 1 m/s, then saturates
    f_V = min(1.0 + 0.3 * V, 1.5)
    
    R_crush = C_c * config.sigma_c * config.h * config.D * f_V
    
    return R_crush


def compute_bending_resistance(
    config: IceMixedModeConfig,
    V: float,
) -> float:
    """Compute bending resistance.
    
    Based on Matlock et al. (1971) and Lindqvist (1989):
    
        R_bend = C_b · σ_f · h² · L_b
    
    where L_b is the characteristic bending length:
    
        L_b = (E · h³ / (12 · (1 - ν²) · ρ_w · g))^(1/4)
    
    Args:
        config: Ice and structure configuration.
        V: Ice drift velocity (m/s).
    
    Returns:
        Bending resistance (N).
    """
    g = 9.81
    
    # Characteristic length
    L_b = (config.E * config.h**3 / 
           (12 * (1 - config.nu**2) * config.rho_water * g))**0.25
    
    # Bending coefficient (depends on structure geometry)
    # For a vertical cylinder: C_b ≈ 0.5-1.0
    C_b = 0.7
    
    # Velocity effect: higher velocity = more dynamic = higher resistance
    f_V = 1.0 + 0.1 * V
    
    R_bend = C_b * config.sigma_f * config.h**2 * L_b * f_V
    
    return R_bend


def compute_splitting_resistance(
    config: IceMixedModeConfig,
    V: float,
) -> float:
    """Compute splitting resistance.
    
    Based on radial crack formation (Palmer & Croasdale 1983):
    
        R_split = C_s · σ_t · h · D · g(V, L)
    
    where g(V, L) is a geometric factor.
    
    Args:
        config: Ice and structure configuration.
        V: Ice drift velocity (m/s).
    
    Returns:
        Splitting resistance (N).
    """
    # Splitting coefficient
    C_s = 1.5
    
    # Geometric factor (depends on D/L_b ratio)
    g = 9.81
    L_b = (config.E * config.h**3 / 
           (12 * (1 - config.nu**2) * config.rho_water * g))**0.25
    
    # Splitting is most effective when D ~ L_b
    ratio = config.D / L_b
    g_factor = 1.0 / (1.0 + (ratio - 1.0)**2)
    
    R_split = C_s * config.sigma_t * config.h * config.D * g_factor
    
    return R_split


def compute_total_resistance(
    config: IceMixedModeConfig,
    V: float,
) -> Tuple[float, float, float, float]:
    """Compute total ice resistance as sum of three components.
    
    Args:
        config: Ice and structure configuration.
        V: Ice drift velocity (m/s).
    
    Returns:
        Tuple of (R_total, R_crush, R_bend, R_split) in Newtons.
    """
    R_crush = compute_crushing_resistance(config, V)
    R_bend = compute_bending_resistance(config, V)
    R_split = compute_splitting_resistance(config, V)
    
    # Mode fractions based on velocity
    V_c = compute_critical_velocity(config)
    
    if V < 0.5 * V_c:
        # Low velocity: crushing dominates
        w_crush = 0.7
        w_bend = 0.1
        w_split = 0.2
    elif V < 1.5 * V_c:
        # Intermediate: mixed mode
        w_crush = 0.3
        w_bend = 0.5
        w_split = 0.2
    else:
        # High velocity: bending dominates
        w_crush = 0.1
        w_bend = 0.7
        w_split = 0.2
    
    R_total = w_crush * R_crush + w_bend * R_bend + w_split * R_split
    
    return R_total, R_crush, R_bend, R_split


def determine_dominant_mode(
    config: IceMixedModeConfig,
    V: float,
) -> str:
    """Determine the dominant failure mode at given conditions.
    
    The dominant mode is determined by the critical velocity V_c:
    - V < V_c: crushing dominates
    - V > V_c: bending dominates
    - V ≈ V_c: mixed mode
    
    Args:
        config: Ice and structure configuration.
        V: Ice drift velocity (m/s).
    
    Returns:
        Dominant failure mode string.
    """
    V_c = compute_critical_velocity(config)
    
    # Determine mode based on velocity relative to critical velocity
    if V < 0.5 * V_c:
        return FailureMode.CRUSHING
    elif V > 2.0 * V_c:
        return FailureMode.BENDING
    else:
        # Check if splitting could be dominant
        R_crush = compute_crushing_resistance(config, V)
        R_bend = compute_bending_resistance(config, V)
        R_split = compute_splitting_resistance(config, V)
        
        # The mode with the LOWEST resistance is the dominant one
        resistances = {
            FailureMode.CRUSHING: R_crush,
            FailureMode.BENDING: R_bend,
            FailureMode.SPLITTING: R_split,
        }
        
        R_min = min(resistances.values())
        R_max = max(resistances.values())
        
        # If modes are close (within 30%), it's mixed
        if R_max < 1.3 * R_min:
            return FailureMode.MIXED
        
        dominant = min(resistances, key=resistances.get)
        return dominant


def compute_critical_velocity(
    config: IceMixedModeConfig,
) -> float:
    """Compute critical velocity for crushing-to-bending transition.
    
    Based on Timco (2001):
    
        V_c ≈ 0.1 · sqrt(σ_f / (ρ · h))
    
    Args:
        config: Ice and structure configuration.
    
    Returns:
        Critical velocity (m/s).
    """
    V_c = 0.1 * math.sqrt(config.sigma_f / (config.rho_ice * config.h))
    return V_c


# ---------------------------------------------------------------------------
# Phase diagram
# ---------------------------------------------------------------------------
@dataclass
class PhaseDiagramResult:
    """Result of phase diagram computation.
    
    Attributes:
        V_array: Velocity array (m/s).
        L_array: Structure width array (m).
        sigma_f_array: Flexural strength array (Pa).
        mode_map: 3D array of dominant modes (V, L, sigma_f).
        transition_boundaries: List of transition boundary curves.
    """
    V_array: np.ndarray
    L_array: np.ndarray
    sigma_f_array: np.ndarray
    mode_map: np.ndarray
    transition_boundaries: List[np.ndarray]


def compute_phase_diagram(
    V_range: Tuple[float, float] = (0.01, 2.0),
    L_range: Tuple[float, float] = (1.0, 50.0),
    sigma_f_range: Tuple[float, float] = (0.3e6, 1.5e6),
    n_V: int = 20,
    n_L: int = 20,
    n_sigma_f: int = 10,
    base_config: Optional[IceMixedModeConfig] = None,
) -> PhaseDiagramResult:
    """Compute failure mode phase diagram in (V, L, σ_f) space.
    
    Args:
        V_range: Velocity range (m/s).
        L_range: Structure width range (m).
        sigma_f_range: Flexural strength range (Pa).
        n_V: Number of velocity points.
        n_L: Number of width points.
        n_sigma_f: Number of strength points.
        base_config: Base configuration (other parameters).
    
    Returns:
        PhaseDiagramResult with mode map and boundaries.
    """
    if base_config is None:
        base_config = IceMixedModeConfig()
    
    V_array = np.linspace(V_range[0], V_range[1], n_V)
    L_array = np.linspace(L_range[0], L_range[1], n_L)
    sigma_f_array = np.linspace(sigma_f_range[0], sigma_f_range[1], n_sigma_f)
    
    # Mode encoding: 0=crushing, 1=bending, 2=splitting, 3=mixed
    mode_map = np.zeros((n_V, n_L, n_sigma_f), dtype=int)
    
    for i, V in enumerate(V_array):
        for j, L in enumerate(L_array):
            for k, sigma_f in enumerate(sigma_f_array):
                config = IceMixedModeConfig(
                    sigma_c=base_config.sigma_c,
                    sigma_f=sigma_f,
                    sigma_t=base_config.sigma_t,
                    h=base_config.h,
                    D=L,
                    E=base_config.E,
                    rho_ice=base_config.rho_ice,
                    rho_water=base_config.rho_water,
                    nu=base_config.nu,
                    mu=base_config.mu,
                    alpha=base_config.alpha,
                )
                
                mode = determine_dominant_mode(config, V)
                if mode == FailureMode.CRUSHING:
                    mode_map[i, j, k] = 0
                elif mode == FailureMode.BENDING:
                    mode_map[i, j, k] = 1
                elif mode == FailureMode.SPLITTING:
                    mode_map[i, j, k] = 2
                else:
                    mode_map[i, j, k] = 3  # mixed
    
    # Find transition boundaries (simplified: 2D slices)
    boundaries = []
    
    # V-L boundary at median sigma_f
    k_mid = n_sigma_f // 2
    for i in range(n_V - 1):
        for j in range(n_L - 1):
            if mode_map[i, j, k_mid] != mode_map[i+1, j, k_mid]:
                boundaries.append(np.array([V_array[i], L_array[j]]))
    
    return PhaseDiagramResult(
        V_array=V_array,
        L_array=L_array,
        sigma_f_array=sigma_f_array,
        mode_map=mode_map,
        transition_boundaries=boundaries if boundaries else [np.array([0.0, 0.0])],
    )


# ---------------------------------------------------------------------------
# Mode transition
# ---------------------------------------------------------------------------
@dataclass
class ModeTransitionResult:
    """Result of mode transition analysis.
    
    Attributes:
        V_c: Critical velocity (m/s).
        V_c_timco: Timco (2001) reference value (m/s).
        relative_error: Relative error vs Timco.
        mode_sequence: Sequence of modes as V increases.
    """
    V_c: float
    V_c_timco: float
    relative_error: float
    mode_sequence: List[str]


def compute_mode_transition(
    config: IceMixedModeConfig,
    V_range: Tuple[float, float] = (0.01, 3.0),
    n_V: int = 50,
) -> ModeTransitionResult:
    """Compute mode transition as a function of velocity.
    
    Args:
        config: Ice and structure configuration.
        V_range: Velocity range (m/s).
        n_V: Number of velocity points.
    
    Returns:
        ModeTransitionResult with critical velocity and mode sequence.
    """
    V_array = np.linspace(V_range[0], V_range[1], n_V)
    modes = []
    
    for V in V_array:
        mode = determine_dominant_mode(config, V)
        modes.append(mode)
    
    # Find critical velocity (first transition from crushing to bending)
    V_c = V_array[-1]  # Default: no transition
    for i in range(len(modes) - 1):
        if modes[i] == FailureMode.CRUSHING and modes[i+1] != FailureMode.CRUSHING:
            V_c = V_array[i]
            break
    
    # Timco (2001) reference
    V_c_timco = compute_critical_velocity(config)
    
    # Relative error
    if V_c_timco > 0:
        rel_error = abs(V_c - V_c_timco) / V_c_timco
    else:
        rel_error = 0.0
    
    return ModeTransitionResult(
        V_c=V_c,
        V_c_timco=V_c_timco,
        relative_error=rel_error,
        mode_sequence=modes,
    )


# ---------------------------------------------------------------------------
# Validation against literature
# ---------------------------------------------------------------------------
@dataclass
class MixedModeResult:
    """Result of mixed mode resistance calculation.
    
    Attributes:
        R_total: Total resistance (N).
        R_crush: Crushing component (N).
        R_bend: Bending component (N).
        R_split: Splitting component (N).
        dominant_mode: Dominant failure mode.
        V: Velocity (m/s).
        config: Configuration used.
    """
    R_total: float
    R_crush: float
    R_bend: float
    R_split: float
    dominant_mode: str
    V: float
    config: IceMixedModeConfig


def validate_against_lindqvist(
    config: IceMixedModeConfig,
    V: float = 1.0,
) -> Dict[str, float]:
    """Validate resistance against Lindqvist (1989) ship resistance formula.
    
    Lindqvist formula for level ice resistance:
    
        R = 0.5 · σ_f · h² · (1 + 0.5 · B/h) · f(shape)
    
    where B is ship beam.
    
    Args:
        config: Ice and structure configuration.
        V: Ship speed (m/s).
    
    Returns:
        Dictionary with R_model, R_lindqvist, relative_error.
    """
    # Our model
    R_total, R_crush, R_bend, R_split = compute_total_resistance(config, V)
    
    # Lindqvist (1989) simplified for comparison
    # R_L = C_L · σ_f · h² · (D/h)^0.5
    C_L = 0.5
    R_lindqvist = C_L * config.sigma_f * config.h**2 * math.sqrt(config.D / config.h)
    
    # Relative error
    if R_lindqvist > 0:
        rel_error = abs(R_total - R_lindqvist) / R_lindqvist
    else:
        rel_error = 0.0
    
    return {
        "R_model": R_total,
        "R_lindqvist": R_lindqvist,
        "relative_error": rel_error,
        "R_crush": R_crush,
        "R_bend": R_bend,
        "R_split": R_split,
    }


def validate_against_timco(
    config: IceMixedModeConfig,
    V_range: Tuple[float, float] = (0.1, 2.0),
    n_V: int = 20,
) -> Dict[str, any]:
    """Validate mode transition against Timco (2001).
    
    Args:
        config: Ice and structure configuration.
        V_range: Velocity range (m/s).
        n_V: Number of velocity points.
    
    Returns:
        Dictionary with transition results.
    """
    transition = compute_mode_transition(config, V_range, n_V)
    
    return {
        "V_c_model": transition.V_c,
        "V_c_timco": transition.V_c_timco,
        "relative_error": transition.relative_error,
        "mode_sequence": transition.mode_sequence,
    }


# ---------------------------------------------------------------------------
# Campaign runner
# ---------------------------------------------------------------------------
def run_mixed_mode_campaign(
    V_sweep: Optional[np.ndarray] = None,
    L_sweep: Optional[np.ndarray] = None,
    sigma_f_sweep: Optional[np.ndarray] = None,
    base_config: Optional[IceMixedModeConfig] = None,
) -> Dict[str, any]:
    """Run comprehensive mixed mode failure campaign.
    
    Performs parameter sweeps and generates phase diagram, mode transitions,
    and resistance components.
    
    Args:
        V_sweep: Velocity sweep array (m/s).
        L_sweep: Structure width sweep array (m).
        sigma_f_sweep: Flexural strength sweep array (Pa).
        base_config: Base configuration.
    
    Returns:
        Dictionary with all campaign results.
    """
    if base_config is None:
        base_config = IceMixedModeConfig()
    
    if V_sweep is None:
        V_sweep = np.linspace(0.1, 2.0, 20)
    
    if L_sweep is None:
        L_sweep = np.linspace(1.0, 50.0, 20)
    
    if sigma_f_sweep is None:
        sigma_f_sweep = np.linspace(0.3e6, 1.5e6, 10)
    
    results = {
        "velocity_sweep": [],
        "width_sweep": [],
        "strength_sweep": [],
        "phase_diagram": None,
        "mode_transition": None,
        "lindqvist_validation": None,
        "timco_validation": None,
    }
    
    # Velocity sweep
    for V in V_sweep:
        R_total, R_crush, R_bend, R_split = compute_total_resistance(base_config, V)
        mode = determine_dominant_mode(base_config, V)
        results["velocity_sweep"].append({
            "V": V,
            "R_total": R_total,
            "R_crush": R_crush,
            "R_bend": R_bend,
            "R_split": R_split,
            "dominant_mode": mode,
        })
    
    # Width sweep
    for L in L_sweep:
        config = IceMixedModeConfig(
            sigma_c=base_config.sigma_c,
            sigma_f=base_config.sigma_f,
            sigma_t=base_config.sigma_t,
            h=base_config.h,
            D=L,
            E=base_config.E,
            rho_ice=base_config.rho_ice,
            rho_water=base_config.rho_water,
            nu=base_config.nu,
            mu=base_config.mu,
            alpha=base_config.alpha,
        )
        R_total, R_crush, R_bend, R_split = compute_total_resistance(config, 1.0)
        mode = determine_dominant_mode(config, 1.0)
        results["width_sweep"].append({
            "L": L,
            "R_total": R_total,
            "R_crush": R_crush,
            "R_bend": R_bend,
            "R_split": R_split,
            "dominant_mode": mode,
        })
    
    # Strength sweep
    for sigma_f in sigma_f_sweep:
        config = IceMixedModeConfig(
            sigma_c=base_config.sigma_c,
            sigma_f=sigma_f,
            sigma_t=base_config.sigma_t,
            h=base_config.h,
            D=base_config.D,
            E=base_config.E,
            rho_ice=base_config.rho_ice,
            rho_water=base_config.rho_water,
            nu=base_config.nu,
            mu=base_config.mu,
            alpha=base_config.alpha,
        )
        R_total, R_crush, R_bend, R_split = compute_total_resistance(config, 1.0)
        mode = determine_dominant_mode(config, 1.0)
        results["strength_sweep"].append({
            "sigma_f": sigma_f,
            "R_total": R_total,
            "R_crush": R_crush,
            "R_bend": R_bend,
            "R_split": R_split,
            "dominant_mode": mode,
        })
    
    # Phase diagram
    results["phase_diagram"] = compute_phase_diagram(base_config=base_config)
    
    # Mode transition
    results["mode_transition"] = compute_mode_transition(base_config)
    
    # Validations
    results["lindqvist_validation"] = validate_against_lindqvist(base_config, 1.0)
    results["timco_validation"] = validate_against_timco(base_config)
    
    return results
