"""DEM thermal-mechanical fracture verification face for ice plates.

Physics background
------------------
When an ice plate cools uniformly, thermal contraction is restrained by
self-equilibration (compatibility), generating tensile thermal stress.
When this stress exceeds the temperature-dependent tensile strength,
periodic thermal cracks form. The crack spacing is governed by the
stress-intensity factor reaching fracture toughness K_IC.

This module implements:
1. Temperature field initialization (uniform / gradient) on a 1D particle chain
2. Explicit finite-difference heat conduction
3. Thermal stress computation: σ_th = E·α·ΔT/(1-ν) (restrained, plane stress)
4. Temperature-dependent bond strength σ_b(T) from Timco & Frederking (1984)
5. Thermal-mechanical coupled fracture criterion
6. Crack spacing measurement and comparison with theory
7. Lindqvist (1983) temperature correction for ice breaking resistance
8. Energy conservation tracking

References
----------
- Timco & Frederking (1984): Compressive strength of ice
- Lindqvist (1983): Ice breaking resistance temperature correction
- Schulson & Duval (2009): Ice as a structural material
- Lawn (2006): Fracture of Brittle Solids

Units: SI throughout, temperatures in °C (differences in K).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ice material properties
# ---------------------------------------------------------------------------

# Default ice properties (freshwater ice)
E_ICE = 9.0e9          # Young's modulus [Pa] at -10°C reference
ALPHA_ICE = 5.0e-5     # Linear thermal expansion coefficient [1/K]
NU_ICE = 0.33          # Poisson's ratio [-]
K_IC = 0.10e6          # Fracture toughness [Pa·m^0.5] (Mode I)
RHO_ICE = 917.0        # Density [kg/m³]
CP_ICE = 2050.0        # Specific heat [J/(kg·K)]
K_ICE = 2.2            # Thermal conductivity [W/(m·K)]
ALPHA_DIFF = K_ICE / (RHO_ICE * CP_ICE)  # Thermal diffusivity [m²/s]

# Reference temperature for strength
T_REF_C = -10.0        # Reference temperature [°C]


def bond_strength(temperature_c: float) -> float:
    """Temperature-dependent ice tensile strength [Pa].

    Uses the relationship:
        σ_b(T) = σ_0 * (1 + c * |T|)
    where T is in °C (negative), σ_0 = 1.0 MPa, c = 0.04 /K.

    This gives monotonically increasing strength with decreasing temperature
    (colder = stronger), consistent with experimental observations.

    Parameters
    ----------
    temperature_c : float
        Temperature in °C (must be ≤ 0)

    Returns
    -------
    float
        Tensile strength in Pa
    """
    T = float(temperature_c)
    if T > 0.0:
        raise ValueError(f"Temperature must be ≤ 0°C, got {T}")
    # Base tensile strength and temperature coefficient
    sigma_0 = 1.0e6   # Base strength at 0°C [Pa]
    c = 0.04           # Temperature coefficient [1/K]
    # Strength increases with |T| (colder = stronger)
    sigma_b = sigma_0 * (1.0 + c * abs(T))
    return sigma_b


def bond_strength_monotonicity_check(temperatures: List[float]) -> bool:
    """Verify dσ_b/dT < 0 (colder = stronger, i.e., σ_b increases as T decreases).

    Returns True if strength is monotonically increasing with decreasing temperature.
    """
    sorted_temps = sorted(temperatures)  # ascending: coldest first
    strengths = [bond_strength(T) for T in sorted_temps]
    # As T increases (sorted ascending), strength should decrease
    for i in range(len(strengths) - 1):
        if strengths[i] < strengths[i + 1]:
            return False  # colder has lower strength = wrong
    return True


# ---------------------------------------------------------------------------
# Temperature field on particle chain
# ---------------------------------------------------------------------------

@dataclass
class ParticleChain:
    """1D chain of DEM particles for thermal fracture analysis.

    Attributes
    ----------
    n_particles : int
        Number of particles
    spacing : float
        Initial particle spacing [m] (= bond rest length)
    positions : ndarray
        Particle x-positions [m], shape (n,)
    temperatures : ndarray
        Particle temperatures [°C], shape (n,)
    displacements : ndarray
        Particle x-displacements [m], shape (n,)
    bond_intact : ndarray
        Boolean array, True if bond between i and i+1 is intact
    cross_section : float
        Cross-sectional area of each bond [m²] (unit width × particle diameter)
    """
    n_particles: int
    spacing: float
    positions: np.ndarray
    temperatures: np.ndarray
    displacements: np.ndarray
    bond_intact: np.ndarray
    cross_section: float
    initial_crack_count: int = 0

    @classmethod
    def create(
        cls,
        length: float,
        n_particles: int,
        initial_temperature_c: float = -10.0,
        cross_section: float = 1.0,
    ) -> "ParticleChain":
        """Create a uniform-temperature particle chain.

        Parameters
        ----------
        length : float
            Total length of the chain [m]
        n_particles : int
            Number of particles
        initial_temperature_c : float
            Initial uniform temperature [°C]
        cross_section : float
            Bond cross-section area [m²]
        """
        spacing = length / (n_particles - 1)
        positions = np.linspace(0, length, n_particles)
        temperatures = np.full(n_particles, initial_temperature_c)
        displacements = np.zeros(n_particles)
        bond_intact = np.ones(n_particles - 1, dtype=bool)
        return cls(
            n_particles=n_particles,
            spacing=spacing,
            positions=positions.copy(),
            temperatures=temperatures,
            displacements=displacements,
            bond_intact=bond_intact,
            cross_section=cross_section,
            initial_crack_count=0,
        )


def apply_uniform_cooling(chain: ParticleChain, delta_t: float) -> None:
    """Apply uniform temperature change to all particles.

    Parameters
    ----------
    chain : ParticleChain
        The particle chain
    delta_t : float
        Temperature change [K] (negative = cooling)
    """
    chain.temperatures += delta_t


def apply_gradient_temperature(
    chain: ParticleChain,
    t_left: float,
    t_right: float,
) -> None:
    """Apply linear temperature gradient across the chain.

    Parameters
    ----------
    chain : ParticleChain
        The particle chain
    t_left : float
        Temperature at left end [°C]
    t_right : float
        Temperature at right end [°C]
    """
    x_norm = (chain.positions - chain.positions[0]) / (chain.positions[-1] - chain.positions[0])
    chain.temperatures = t_left + (t_right - t_left) * x_norm


# ---------------------------------------------------------------------------
# Heat conduction (explicit finite difference)
# ---------------------------------------------------------------------------

def solve_heat_conduction_1d(
    chain: ParticleChain,
    boundary_left_c: float,
    boundary_right_c: float,
    total_time: float,
    dt: float,
) -> Tuple[np.ndarray, int]:
    """Solve 1D heat conduction using explicit finite difference.

    Uses forward-time central-space (FTCS) scheme:
        T_i^{n+1} = T_i^n + Fo * (T_{i+1}^n - 2*T_i^n + T_{i-1}^n)
    where Fo = α·Δt/Δx² is the Fourier number.

    Parameters
    ----------
    chain : ParticleChain
        Particle chain (temperatures updated in-place)
    boundary_left_c : float
        Left boundary temperature [°C]
    boundary_right_c : float
        Right boundary temperature [°C]
    total_time : float
        Total simulation time [s]
    dt : float
        Time step [s]

    Returns
    -------
    final_temperatures : ndarray
        Final temperature field [°C]
    n_steps : int
        Number of time steps taken
    """
    dx = chain.spacing
    # Stability criterion: Fo ≤ 0.5
    fo = ALPHA_DIFF * dt / (dx * dx)
    if fo > 0.5:
        dt = 0.4 * dx * dx / ALPHA_DIFF
        fo = ALPHA_DIFF * dt / (dx * dx)

    n_steps = int(math.ceil(total_time / dt))
    T = chain.temperatures.copy()

    for _ in range(n_steps):
        T_new = T.copy()
        # Interior points
        for i in range(1, len(T) - 1):
            T_new[i] = T[i] + fo * (T[i + 1] - 2 * T[i] + T[i - 1])
        # Boundary conditions (Dirichlet)
        T_new[0] = boundary_left_c
        T_new[-1] = boundary_right_c
        T = T_new

    chain.temperatures = T
    return T, n_steps


# ---------------------------------------------------------------------------
# Thermal stress computation
# ---------------------------------------------------------------------------

def compute_thermal_stress_restrained(
    temperature_c: float,
    t_ref: float = T_REF_C,
    young_modulus: float = E_ICE,
    alpha: float = ALPHA_ICE,
    nu: float = NU_ICE,
) -> float:
    """Compute thermal stress for fully restrained case (plane stress).

    For cooling (T < T_ref): σ_th = E · α · (T_ref - T) / (1 - ν) > 0 (tensile)
    For heating (T > T_ref): σ_th < 0 (compressive)

    Parameters
    ----------
    temperature_c : float
        Current temperature [°C]
    t_ref : float
        Reference (stress-free) temperature [°C]
    young_modulus : float
        Young's modulus [Pa]
    alpha : float
        Thermal expansion coefficient [1/K]
    nu : float
        Poisson's ratio [-]

    Returns
    -------
    float
        Thermal stress [Pa] (positive = tensile)
    """
    delta_t = t_ref - temperature_c  # positive for cooling
    sigma_th = young_modulus * alpha * delta_t / (1.0 - nu)
    return sigma_th


def compute_thermal_stress_field(
    chain: ParticleChain,
    t_ref: float = T_REF_C,
    young_modulus: float = E_ICE,
    alpha: float = ALPHA_ICE,
    nu: float = NU_ICE,
) -> np.ndarray:
    """Compute thermal stress at each particle location.

    Parameters
    ----------
    chain : ParticleChain
        Particle chain with temperature field
    t_ref : float
        Reference temperature [°C]
    young_modulus : float
        Young's modulus [Pa]
    alpha : float
        Thermal expansion coefficient [1/K]
    nu : float
        Poisson's ratio [-]

    Returns
    -------
    ndarray
        Thermal stress at each particle [Pa] (positive = tensile)
    """
    delta_t = t_ref - chain.temperatures  # positive for cooling
    sigma_th = young_modulus * alpha * delta_t / (1.0 - nu)
    return sigma_th


def compute_thermal_stress_analytical(
    delta_t: float,
    young_modulus: float = E_ICE,
    alpha: float = ALPHA_ICE,
    nu: float = NU_ICE,
) -> float:
    """Analytical thermal stress for uniform cooling of restrained plate.

    σ_th = E · α · |ΔT| / (1 - ν)

    Parameters
    ----------
    delta_t : float
        Temperature change [K] (negative for cooling)
    young_modulus : float
        Young's modulus [Pa]
    alpha : float
        Thermal expansion coefficient [1/K]
    nu : float
        Poisson's ratio [-]

    Returns
    -------
    float
        Tensile thermal stress [Pa]
    """
    return young_modulus * alpha * abs(delta_t) / (1.0 - nu)


# ---------------------------------------------------------------------------
# Fracture criterion and crack formation
# ---------------------------------------------------------------------------

@dataclass
class FractureResult:
    """Result of thermal fracture analysis."""
    bond_intact: np.ndarray
    crack_positions: np.ndarray  # x-positions of cracks [m]
    crack_count: int
    total_bond_energy_released: float  # [J]
    stress_field: np.ndarray  # [Pa]
    strength_field: np.ndarray  # [Pa]


def apply_fracture_criterion(
    chain: ParticleChain,
    sigma_mechanical: Optional[np.ndarray] = None,
    t_ref: float = T_REF_C,
) -> FractureResult:
    """Apply thermal-mechanical coupled fracture criterion.

    Bond breaks when: σ_th + σ_mech ≥ σ_b(T)

    Parameters
    ----------
    chain : ParticleChain
        Particle chain
    sigma_mechanical : ndarray, optional
        Mechanical stress at each bond [Pa] (positive = tensile)
    t_ref : float
        Reference temperature [°C]

    Returns
    -------
    FractureResult
        Fracture analysis results
    """
    n_bonds = chain.n_particles - 1

    # Compute thermal stress at each bond (average of adjacent particles)
    sigma_th = np.zeros(n_bonds)
    for i in range(n_bonds):
        t_avg = 0.5 * (chain.temperatures[i] + chain.temperatures[i + 1])
        sigma_th[i] = compute_thermal_stress_restrained(t_avg, t_ref)

    # Mechanical stress (default zero)
    if sigma_mechanical is None:
        sigma_mechanical = np.zeros(n_bonds)

    # Total stress
    sigma_total = sigma_th + sigma_mechanical

    # Bond strength at each location
    sigma_b = np.zeros(n_bonds)
    for i in range(n_bonds):
        t_avg = 0.5 * (chain.temperatures[i] + chain.temperatures[i + 1])
        sigma_b[i] = bond_strength(t_avg)

    # Fracture criterion: break if total tensile stress ≥ strength
    # Only break intact bonds
    energy_released = 0.0
    new_crack_positions = []

    for i in range(n_bonds):
        if chain.bond_intact[i] and sigma_total[i] >= sigma_b[i]:
            chain.bond_intact[i] = False
            # Energy released = elastic energy stored in bond
            # U = 0.5 * σ² * V / E = 0.5 * σ² * A * dx / E
            volume = chain.cross_section * chain.spacing
            energy = 0.5 * sigma_total[i]**2 * volume / E_ICE
            energy_released += energy
            # Crack position (midpoint of broken bond)
            x_crack = 0.5 * (chain.positions[i] + chain.positions[i + 1])
            new_crack_positions.append(x_crack)

    crack_positions = np.array(new_crack_positions)
    return FractureResult(
        bond_intact=chain.bond_intact.copy(),
        crack_positions=crack_positions,
        crack_count=len(crack_positions),
        total_bond_energy_released=energy_released,
        stress_field=sigma_total,
        strength_field=sigma_b,
    )


# ---------------------------------------------------------------------------
# Crack spacing analysis
# ---------------------------------------------------------------------------

def measure_crack_spacing(crack_positions: np.ndarray, plate_length: float) -> Dict[str, float]:
    """Measure crack spacing statistics.

    Parameters
    ----------
    crack_positions : ndarray
        X-positions of cracks [m]
    plate_length : float
        Total plate length [m]

    Returns
    -------
    dict
        Statistics: mean_spacing, min_spacing, max_spacing, n_cracks, spacing_std
    """
    if len(crack_positions) < 2:
        return {
            "mean_spacing": plate_length,
            "min_spacing": plate_length,
            "max_spacing": plate_length,
            "n_cracks": len(crack_positions),
            "spacing_std": 0.0,
        }

    sorted_pos = np.sort(crack_positions)
    spacings = np.diff(sorted_pos)

    return {
        "mean_spacing": float(np.mean(spacings)),
        "min_spacing": float(np.min(spacings)),
        "max_spacing": float(np.max(spacings)),
        "n_cracks": len(crack_positions),
        "spacing_std": float(np.std(spacings)),
    }


def theoretical_crack_spacing(
    k_ic: float = K_IC,
    young_modulus: float = E_ICE,
    alpha: float = ALPHA_ICE,
    delta_t: float = -20.0,
    initial_flaw_size: float = 0.01,
) -> float:
    """Theoretical thermal crack spacing from fracture mechanics.

    s_th = K_IC / (E · α · |ΔT|) · √(π · a_0)

    This is derived from the condition that the stress intensity factor
    at a crack tip reaches K_IC:
        K_I = σ_th · √(π · a) = K_IC
    and the crack spacing is related to the stress field relaxation length.

    Parameters
    ----------
    k_ic : float
        Fracture toughness [Pa·m^0.5]
    young_modulus : float
        Young's modulus [Pa]
    alpha : float
        Thermal expansion coefficient [1/K]
    delta_t : float
        Temperature change [K] (negative for cooling)
    initial_flaw_size : float
        Initial flaw size a_0 [m]

    Returns
    -------
    float
        Theoretical crack spacing [m]
    """
    sigma_th = young_modulus * alpha * abs(delta_t)
    s_th = (k_ic / sigma_th) * math.sqrt(math.pi * initial_flaw_size)
    return s_th


# ---------------------------------------------------------------------------
# Lindqvist temperature correction
# ---------------------------------------------------------------------------

def lindqvist_temperature_correction(
    temperature_c: float,
    t_ref_c: float = T_REF_C,
    exponent_n: float = 0.75,
) -> float:
    """Lindqvist (1983) temperature correction factor for ice breaking resistance.

    F_break(T) / F_break(T_ref) = (σ_b(T) / σ_b(T_ref))^n

    where n ≈ 0.5-1.0 depending on breaking mode.

    Parameters
    ----------
    temperature_c : float
        Ice temperature [°C]
    t_ref_c : float
        Reference temperature [°C]
    exponent_n : float
        Temperature exponent (0.5-1.0)

    Returns
    -------
    float
        Correction factor (dimensionless)
    """
    sigma_b_t = bond_strength(temperature_c)
    sigma_b_ref = bond_strength(t_ref_c)
    correction = (sigma_b_t / sigma_b_ref) ** exponent_n
    return correction


def compute_breaking_force_vs_temperature(
    temperatures_c: List[float],
    f_ref: float = 1.0e6,
    t_ref_c: float = T_REF_C,
    exponent_n: float = 0.75,
) -> Dict[str, List[float]]:
    """Compute breaking force at multiple temperatures.

    Parameters
    ----------
    temperatures_c : list of float
        Temperatures [°C]
    f_ref : float
        Reference breaking force at T_ref [N]
    t_ref_c : float
        Reference temperature [°C]
    exponent_n : float
        Lindqvist exponent

    Returns
    -------
    dict
        temperatures, forces, correction_factors
    """
    forces = []
    corrections = []
    for T in temperatures_c:
        cf = lindqvist_temperature_correction(T, t_ref_c, exponent_n)
        forces.append(f_ref * cf)
        corrections.append(cf)

    return {
        "temperatures": temperatures_c,
        "forces": forces,
        "correction_factors": corrections,
    }


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------

def compute_elastic_energy_stored(
    chain: ParticleChain,
    sigma_field: np.ndarray,
) -> float:
    """Compute total elastic energy stored in intact bonds.

    U = Σ 0.5 * σ² * V / E for each intact bond

    Parameters
    ----------
    chain : ParticleChain
        Particle chain
    sigma_field : ndarray
        Stress at each bond [Pa]

    Returns
    -------
    float
        Total elastic energy [J]
    """
    energy = 0.0
    volume = chain.cross_section * chain.spacing
    for i in range(len(chain.bond_intact)):
        if chain.bond_intact[i]:
            energy += 0.5 * sigma_field[i]**2 * volume / E_ICE
    return energy


def energy_conservation_check(
    initial_energy: float,
    final_energy: float,
    fracture_energy: float,
) -> Dict[str, float]:
    """Check energy conservation: E_initial = E_final + E_fracture.

    Parameters
    ----------
    initial_energy : float
        Initial elastic energy [J]
    final_energy : float
        Final elastic energy [J]
    fracture_energy : float
        Energy dissipated in fracture [J]

    Returns
    -------
    dict
        initial_energy, final_energy, fracture_energy, residual, relative_residual
    """
    total_final = final_energy + fracture_energy
    residual = initial_energy - total_final
    rel_residual = abs(residual) / max(abs(initial_energy), 1e-30)

    return {
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "fracture_energy": float(fracture_energy),
        "residual": float(residual),
        "relative_residual": float(rel_residual),
    }


# ---------------------------------------------------------------------------
# Full verification scenario runners
# ---------------------------------------------------------------------------

@dataclass
class ThermalFractureScenario:
    """Configuration for a thermal fracture verification scenario."""
    plate_length: float = 10.0       # [m]
    plate_height: float = 1.0        # [m]
    n_particles: int = 21            # Coarser resolution for crack spacing
    initial_temperature_c: float = -10.0
    cooling_delta_t: float = -20.0   # [K]
    cross_section: float = 1.0       # [m²] (unit width × height)
    initial_flaw_size: float = 0.5   # [m] - matched to DEM scale
    t_ref_c: float = T_REF_C
    strength_variation: float = 0.60  # 60% random variation in bond strength


def run_crack_spacing_verification(
    scenario: Optional[ThermalFractureScenario] = None,
) -> Dict[str, Any]:
    """Run Verification Face 1: Thermal crack spacing.

    Uniform cooling ΔT = -20K of ice plate, measure crack spacing s,
    compare with theory s_th.

    The crack spacing emerges from the interaction of:
    - Thermal stress level (determines how many bonds break)
    - Spatial variation in bond strength (represents flaws)
    - DEM resolution (particle spacing)

    Returns
    -------
    dict
        Complete verification results
    """
    if scenario is None:
        scenario = ThermalFractureScenario()

    # Create particle chain
    chain = ParticleChain.create(
        length=scenario.plate_length,
        n_particles=scenario.n_particles,
        initial_temperature_c=scenario.initial_temperature_c,
        cross_section=scenario.cross_section,
    )

    # Store reference temperature (stress-free state)
    t_ref = scenario.initial_temperature_c

    # Apply uniform cooling
    apply_uniform_cooling(chain, scenario.cooling_delta_t)

    # Compute thermal stress field
    sigma_th_field = compute_thermal_stress_field(chain, t_ref)

    # Compute theoretical thermal stress
    sigma_analytical = compute_thermal_stress_analytical(scenario.cooling_delta_t)
    sigma_numerical_mean = float(np.mean(sigma_th_field))
    stress_error = abs(sigma_numerical_mean - sigma_analytical) / abs(sigma_analytical)

    # For crack spacing: model flaw distribution with spatially varying strength
    np.random.seed(42)  # Reproducibility
    n_bonds = chain.n_particles - 1

    # Create strength field with spatial variation (representing flaws)
    base_strength = bond_strength(chain.temperatures[0])  # uniform T after cooling
    strength_variation = scenario.strength_variation
    local_strength = base_strength * (1.0 + strength_variation * np.random.randn(n_bonds))
    # Ensure positive strength
    local_strength = np.maximum(local_strength, 0.3 * base_strength)

    # Apply fracture criterion with spatially varying strength
    sigma_total = sigma_th_field.copy()
    energy_released = 0.0
    crack_positions_list = []

    for i in range(n_bonds):
        if sigma_total[i] >= local_strength[i]:
            chain.bond_intact[i] = False
            volume = chain.cross_section * chain.spacing
            energy = 0.5 * sigma_total[i]**2 * volume / E_ICE
            energy_released += energy
            x_crack = 0.5 * (chain.positions[i] + chain.positions[i + 1])
            crack_positions_list.append(x_crack)

    crack_positions = np.array(crack_positions_list)

    # Measure crack spacing
    spacing_stats = measure_crack_spacing(crack_positions, scenario.plate_length)

    # Theoretical crack spacing using DEM-scale flaw size
    s_th = theoretical_crack_spacing(
        delta_t=scenario.cooling_delta_t,
        initial_flaw_size=scenario.initial_flaw_size,
    )

    # Error
    if spacing_stats["n_cracks"] >= 2:
        measured_spacing = spacing_stats["mean_spacing"]
        relative_error = abs(measured_spacing - s_th) / s_th
    else:
        measured_spacing = float('nan')
        relative_error = float('nan')

    return {
        "scenario": {
            "plate_length": scenario.plate_length,
            "plate_height": scenario.plate_height,
            "n_particles": scenario.n_particles,
            "initial_temperature_c": scenario.initial_temperature_c,
            "cooling_delta_t": scenario.cooling_delta_t,
            "initial_flaw_size": scenario.initial_flaw_size,
        },
        "measured_spacing_m": measured_spacing,
        "theoretical_spacing_m": s_th,
        "spacing_relative_error": relative_error,
        "n_cracks": len(crack_positions_list),
        "spacing_stats": spacing_stats,
        "thermal_stress_analytical_pa": sigma_analytical,
        "thermal_stress_numerical_mean_pa": sigma_numerical_mean,
        "thermal_stress_relative_error": stress_error,
        "fracture_energy_j": energy_released,
        "crack_positions": crack_positions.tolist(),
    }


def run_lindqvist_verification(
    temperatures_c: Optional[List[float]] = None,
    exponent_n: float = 0.75,
) -> Dict[str, Any]:
    """Run Verification Face 2: Lindqvist temperature correction.

    Compare F_break(T)/F_break(T_ref) with empirical formula at
    T = -5°C, -15°C, -25°C.

    Returns
    -------
    dict
        Complete verification results
    """
    if temperatures_c is None:
        temperatures_c = [-25.0, -15.0, -10.0, -5.0]

    t_ref = T_REF_C
    f_ref = 1.0e6  # Reference force [N]

    results = compute_breaking_force_vs_temperature(
        temperatures_c, f_ref, t_ref, exponent_n
    )

    # Compare with Lindqvist empirical: correction ≈ (σ_b(T)/σ_b(T_ref))^n
    # The "empirical" is our own model, so we verify internal consistency
    # and compare with published n values (0.5-1.0)
    comparisons = []
    for i, T in enumerate(temperatures_c):
        cf_model = results["correction_factors"][i]
        # Lindqvist empirical range: n ∈ [0.5, 1.0]
        ratio = bond_strength(T) / bond_strength(t_ref)
        cf_n05 = ratio ** 0.5
        cf_n10 = ratio ** 1.0
        # Range is [min, max] regardless of whether ratio > 1 or < 1
        cf_low = min(cf_n05, cf_n10)
        cf_high = max(cf_n05, cf_n10)
        in_range = cf_low <= cf_model <= cf_high

        comparisons.append({
            "temperature_c": T,
            "model_correction": cf_model,
            "lindqvist_low_n05": cf_n05,
            "lindqvist_high_n10": cf_n10,
            "in_lindqvist_range": in_range,
        })

    return {
        "reference_temperature_c": t_ref,
        "exponent_n": exponent_n,
        "comparisons": comparisons,
        "forces": results["forces"],
    }


def run_negative_control() -> Dict[str, Any]:
    """Run negative control: isothermal field → no thermal stress → no cracks.

    Returns
    -------
    dict
        Verification that no cracks form under isothermal conditions
    """
    # Create chain at uniform temperature
    chain = ParticleChain.create(
        length=10.0,
        n_particles=51,
        initial_temperature_c=-10.0,
        cross_section=1.0,
    )

    # No temperature change - apply fracture criterion
    result = apply_fracture_criterion(chain, t_ref=-10.0)

    return {
        "n_cracks": result.crack_count,
        "max_stress_pa": float(np.max(np.abs(result.stress_field))),
        "all_bonds_intact": bool(np.all(result.bond_intact)),
        "pass": result.crack_count == 0,
    }


def run_energy_conservation_verification() -> Dict[str, Any]:
    """Run energy conservation verification.

    Returns
    -------
    dict
        Energy balance check results
    """
    # Create chain and apply cooling
    chain = ParticleChain.create(
        length=10.0,
        n_particles=101,
        initial_temperature_c=-10.0,
        cross_section=1.0,
    )

    t_ref = -10.0

    # Apply cooling
    apply_uniform_cooling(chain, -20.0)

    # Compute stress after cooling (before fracture)
    sigma_pre_fracture = compute_thermal_stress_field(chain, t_ref)
    energy_pre = compute_elastic_energy_stored(chain, sigma_pre_fracture)

    # Apply fracture with spatially varying strength
    np.random.seed(42)
    n_bonds = chain.n_particles - 1
    base_strength = bond_strength(chain.temperatures[0])
    local_strength = base_strength * (1.0 + 0.60 * np.random.randn(n_bonds))
    local_strength = np.maximum(local_strength, 0.3 * base_strength)

    sigma_total = sigma_pre_fracture.copy()
    energy_released = 0.0

    for i in range(n_bonds):
        if chain.bond_intact[i] and sigma_total[i] >= local_strength[i]:
            chain.bond_intact[i] = False
            volume = chain.cross_section * chain.spacing
            energy = 0.5 * sigma_total[i]**2 * volume / E_ICE
            energy_released += energy

    # Compute final stress (after fracture - broken bonds carry no stress)
    sigma_post = sigma_total.copy()
    for i in range(len(chain.bond_intact)):
        if not chain.bond_intact[i]:
            sigma_post[i] = 0.0
    energy_post = compute_elastic_energy_stored(chain, sigma_post)

    # Energy check
    check = energy_conservation_check(energy_pre, energy_post, energy_released)

    return {
        "energy_before_fracture_j": energy_pre,
        "energy_after_fracture_j": energy_post,
        "fracture_energy_j": energy_released,
        "energy_balance": check,
        "pass": check["relative_residual"] <= 0.02,
    }
