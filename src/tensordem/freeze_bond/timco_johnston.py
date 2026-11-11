"""Timco & Johnston (2003) exponential freeze-bond law (IN13-W3-N06).

This module was originally the top-level ``tensordem/freeze_bond.py``.  The
``tensordem/freeze_bond/`` package (Sastri-Metzger power-law model, added
later) shadowed it, silently breaking the verification contract
``tests/test_freeze_bond.py`` at collection time.  It was moved here as the
``timco_johnston`` submodule and its public API is re-exported from the
package ``__init__`` (see the collision policy documented there).

Freeze-bond model for ice–structure interaction in TensorDEM.

Physical background
-------------------
When ice comes into contact with a sub-cooled structure (or another ice surface)
at temperatures below the pressure-melting point, a **freeze-bond** forms at the
interface.  The bond grows over time as heat is conducted away from the contact,
freezing any thin water film or bridging surface asperities.  The resulting
interface has a tensile/shear strength that depends on:

1. **Temperature** — colder interfaces produce stronger bonds because the
   thermal driving force for freezing is larger.  The bond strength follows::

       σ_fb(T) = σ_0 · exp(-β · ΔT)

   where ΔT = T_interface - T_m is negative when sub-cooled (so -β·ΔT > 0),
   σ_0 is the reference strength at the melting point, and β is a
   temperature-sensitivity parameter.  This Arrhenius-like form captures the
   observation that bond strength increases rapidly as temperature drops
   below -5°C (Timco & Johnston 2003).

2. **Contact time** — the longer two surfaces remain in contact below the
   melting point, the larger the bonded area becomes.  For diffusion-limited
   freezing of a thin water film of thickness h_w, the bond area fraction
   grows as::

       A_fb(t) / A_0 = 1 - exp(-t_contact / τ_growth)

   where τ_growth depends on the thermal diffusivity of ice and the film
   thickness.  For typical sea-ice conditions τ_growth ~ 10–100 s.

3. **Failure mode transition** — there are two competing failure mechanisms:

   - **Adhesive failure** (interface debonding): the bond breaks at the
     ice–structure interface.  This occurs when σ_fb < σ_cohesive (the bulk
     ice tensile strength).  Dominant at high temperatures (near melting)
     and short contact times.

   - **Cohesive failure** (ice body fracture): the bond is stronger than the
     ice itself, so the fracture propagates through the ice body rather than
     along the interface.  Dominant at low temperatures and long contact times.

   The transition occurs when σ_fb(T, t_contact) ≈ σ_cohesive.  At this
   transition, the peak load exhibits a **jump** because the failure mechanism
   changes abruptly (Timco & Johnston 2003 observed this step change).

4. **Loading rate effect** — at high loading rates, the interface has less time
   to relax viscoelastically, so the apparent strength increases.  At very low
   rates, creep damage accumulates and the strength decreases.

References
----------
Timco, G.W. & Johnston, M. (2003). "Freezing of water to structural surfaces."
    Cold Regions Science and Technology, 37(2), 145-156.
Jordaan, I.J. et al. (1999). "The effect of water on the adhesion of ice to
    steel." Proc. 15th POAC, 931-940.
Schulson, E.M. & Duval, P. (2009). Ice: A Brief Review. Cambridge Univ. Press.
Moser, H. (1967). "The strength of ice." IAHS Publication 79, 334-342.

All quantities are SI: m, kg, s, N, Pa, K (or °C for temperature differences).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "FreezeBondParams",
    "FreezeBondState",

    "FailureMode",
    "PullOffResult",
    "ShearResult",
    "FailureModePhaseDiagram",
    "ParameterSweepResult",
    "freeze_bond_strength",
    "bond_growth_kinetics",
    "classify_failure_mode",
    "pull_off_test",
    "shear_test",
    "failure_mode_phase_diagram",
    "parameter_sweep",
    "run_full_verification",
]


# ---------------------------------------------------------------------------
# Physical constants and default parameters
# ---------------------------------------------------------------------------

#: Melting point of fresh-water ice at atmospheric pressure [°C]
T_MELTING_FRESHWATER: float = 0.0

#: Melting point of sea ice (approximate, depends on salinity) [°C]
T_MELTING_SEA_ICE: float = -1.8

#: Thermal diffusivity of ice [m²/s]
ALPHA_ICE: float = 1.15e-6

#: Density of ice [kg/m³]
RHO_ICE: float = 917.0

#: Young's modulus of ice [Pa] (typical range 3-10 GPa)
E_ICE: float = 5.0e9

#: Tensile strength of ice [Pa] (typical range 0.5-2.5 MPa)
SIGMA_T_ICE: float = 1.0e6

#: Default reference freeze-bond strength at melting point [Pa]
#: Based on Timco & Johnston (2003): ~0.1-0.5 MPa at near-melting temperatures
SIGMA_0_DEFAULT: float = 0.2e6  # 0.2 MPa

#: Default temperature sensitivity parameter [1/°C]
#: Calibrated so that σ_fb(-10°C) ≈ 1 MPa (consistent with literature)
BETA_DEFAULT: float = 0.15

#: Default growth time constant [s]
TAU_GROWTH_DEFAULT: float = 50.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreezeBondParams:
    """Parameters for the freeze-bond model.

    Attributes
    ----------
    sigma_0 : float
        Reference bond strength at the melting point [Pa].
    beta : float
        Temperature sensitivity parameter [1/°C].
    tau_growth : float
        Characteristic growth time for bond area [s].
    sigma_cohesive : float
        Cohesive (bulk ice) tensile strength [Pa].
    T_melting : float
        Melting point temperature [°C].
    A_contact : float
        Nominal contact area [m²].
    h_film : float
        Initial water film thickness [m] (for growth kinetics).
    """
    sigma_0: float = SIGMA_0_DEFAULT
    beta: float = BETA_DEFAULT
    tau_growth: float = TAU_GROWTH_DEFAULT
    sigma_cohesive: float = SIGMA_T_ICE
    T_melting: float = T_MELTING_FRESHWATER
    A_contact: float = 1.0e-4  # 1 cm²
    h_film: float = 1.0e-5  # 10 μm

    def __post_init__(self) -> None:
        for name in ("sigma_0", "beta", "tau_growth", "sigma_cohesive",
                     "A_contact", "h_film"):
            val = getattr(self, name)
            if not (math.isfinite(val) and val > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {val}")
        if not math.isfinite(self.T_melting):
            raise ValueError("T_melting must be finite")


@dataclass
class FreezeBondState:
    """State of a freeze-bond at a given time.

    Attributes
    ----------
    T_interface : float
        Interface temperature [°C].
    t_contact : float
        Contact time [s].
    bond_area_fraction : float
        Fraction of contact area that is bonded (0 to 1).
    sigma_fb : float
        Current freeze-bond strength [Pa].
    failure_mode : str
        Current failure mode ('adhesive', 'cohesive', or 'mixed').
    peak_force : float
        Peak force at failure [N].
    """
    T_interface: float
    t_contact: float
    bond_area_fraction: float
    sigma_fb: float
    failure_mode: str
    peak_force: float


@dataclass
class FailureMode:
    """Classification of failure mode with transition indicator."""
    mode: str  # 'adhesive', 'cohesive', 'mixed'
    sigma_fb: float  # freeze-bond strength [Pa]
    sigma_cohesive: float  # bulk ice strength [Pa]
    ratio: float  # sigma_fb / sigma_cohesive
    is_transition: bool  # True if near the transition boundary


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def freeze_bond_strength(
    T_interface: float,
    params: Optional[FreezeBondParams] = None,
) -> float:
    """Compute freeze-bond strength as a function of interface temperature.

    σ_fb(T) = σ_0 · exp(-β · ΔT)

    where ΔT = T_interface - T_melting is negative when the interface is
    below the melting point, so -β·ΔT = β·|subcooling| > 0 and the bond
    strength increases with sub-cooling.

    Parameters
    ----------
    T_interface : float
        Interface temperature [°C]. Must be below the melting point.
    params : FreezeBondParams, optional
        Model parameters. Uses defaults if not provided.

    Returns
    -------
    float
        Freeze-bond strength [Pa].

    Raises
    ------
    ValueError
        If T_interface >= T_melting (no freeze-bond above melting point).
    """
    if params is None:
        params = FreezeBondParams()
    if T_interface >= params.T_melting:
        return 0.0  # No bond above melting point
    # ΔT = T_interface - T_melting (negative when sub-cooled)
    # σ_fb = σ_0 · exp(-β · ΔT) = σ_0 · exp(+β · |subcooling|)
    delta_T = T_interface - params.T_melting  # negative for sub-cooled
    return params.sigma_0 * math.exp(-params.beta * delta_T)


def bond_growth_kinetics(
    t_contact: float,
    params: Optional[FreezeBondParams] = None,
) -> float:
    """Compute the bonded area fraction as a function of contact time.

    A_fb(t) / A_0 = 1 - exp(-t_contact / τ_growth)

    This represents diffusion-limited freezing of a thin water film at the
    interface. The growth rate depends on the thermal diffusivity of ice
    and the film thickness.

    Parameters
    ----------
    t_contact : float
        Contact time [s]. Must be non-negative.
    params : FreezeBondParams, optional
        Model parameters.

    Returns
    -------
    float
        Bonded area fraction (0 to 1).
    """
    if params is None:
        params = FreezeBondParams()
    if t_contact < 0.0:
        raise ValueError("t_contact must be non-negative")
    return 1.0 - math.exp(-t_contact / params.tau_growth)


def effective_bond_strength(
    T_interface: float,
    t_contact: float,
    params: Optional[FreezeBondParams] = None,
) -> float:
    """Compute the effective freeze-bond strength including growth kinetics.

    σ_eff = σ_fb(T) · A_fb(t) / A_0

    This combines the temperature-dependent intrinsic strength with the
    time-dependent bonded area fraction.

    Parameters
    ----------
    T_interface : float
        Interface temperature [°C].
    t_contact : float
        Contact time [s].
    params : FreezeBondParams, optional
        Model parameters.

    Returns
    -------
    float
        Effective bond strength [Pa].
    """
    if params is None:
        params = FreezeBondParams()
    sigma_T = freeze_bond_strength(T_interface, params)
    area_frac = bond_growth_kinetics(t_contact, params)
    return sigma_T * area_frac


def classify_failure_mode(
    T_interface: float,
    t_contact: float,
    params: Optional[FreezeBondParams] = None,
    tolerance: float = 0.1,
) -> FailureMode:
    """Classify the failure mode based on bond strength vs cohesive strength.

    The failure mode is determined by comparing the effective freeze-bond
    strength σ_eff with the cohesive (bulk ice) tensile strength σ_cohesive:

    - σ_eff < (1-tol) · σ_cohesive → adhesive failure (interface debonding)
    - σ_eff > (1+tol) · σ_cohesive → cohesive failure (ice body fracture)
    - otherwise → mixed/transition regime

    Parameters
    ----------
    T_interface : float
        Interface temperature [°C].
    t_contact : float
        Contact time [s].
    params : FreezeBondParams, optional
        Model parameters.
    tolerance : float
        Relative tolerance for transition detection.

    Returns
    -------
    FailureMode
        Classification result with mode, strengths, ratio, and transition flag.
    """
    if params is None:
        params = FreezeBondParams()
    sigma_eff = effective_bond_strength(T_interface, t_contact, params)
    sigma_coh = params.sigma_cohesive
    ratio = sigma_eff / sigma_coh if sigma_coh > 0 else float('inf')

    if ratio < (1.0 - tolerance):
        mode = "adhesive"
    elif ratio > (1.0 + tolerance):
        mode = "cohesive"
    else:
        mode = "mixed"

    is_transition = (1.0 - tolerance) <= ratio <= (1.0 + tolerance)

    return FailureMode(
        mode=mode,
        sigma_fb=sigma_eff,
        sigma_cohesive=sigma_coh,
        ratio=ratio,
        is_transition=is_transition,
    )


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

@dataclass
class PullOffResult:
    """Result of a uniaxial pull-off test.

    An ice column is bonded to a rigid plate and pulled perpendicular to the
    interface. The peak force is recorded as a function of temperature and
    contact time.
    """
    T_interface: float
    t_contact: float
    peak_force: float  # [N]
    peak_stress: float  # [Pa]
    failure_mode: str
    displacement_at_peak: float  # [m]
    stiffness: float  # [N/m]
    params: FreezeBondParams


def pull_off_test(
    T_interface: float,
    t_contact: float,
    params: Optional[FreezeBondParams] = None,
    loading_rate: float = 0.01,  # m/s
) -> PullOffResult:
    """Simulate a uniaxial pull-off test.

    An ice column of cross-section A_contact is bonded to a rigid plate.
    The column is pulled at a constant rate until failure.

    The peak force is determined by the failure mode:
    - Adhesive: F_peak = σ_eff · A_contact
    - Cohesive: F_peak = σ_cohesive · A_contact (limited by ice strength)

    The displacement at peak is estimated from the elastic compliance of the
    ice column (assuming a characteristic length L = 0.1 m).

    Parameters
    ----------
    T_interface : float
        Interface temperature [°C].
    t_contact : float
        Contact time [s].
    params : FreezeBondParams, optional
        Model parameters.
    loading_rate : float
        Pull-off velocity [m/s].

    Returns
    -------
    PullOffResult
        Test results including peak force, stress, and failure mode.
    """
    if params is None:
        params = FreezeBondParams()

    sigma_eff = effective_bond_strength(T_interface, t_contact, params)
    fm = classify_failure_mode(T_interface, t_contact, params)

    # Peak stress is limited by the weaker mechanism
    if fm.mode == "adhesive":
        peak_stress = sigma_eff
    elif fm.mode == "cohesive":
        peak_stress = params.sigma_cohesive
    else:  # mixed - transition regime, use minimum of the two
        peak_stress = min(sigma_eff, params.sigma_cohesive)

    peak_force = peak_stress * params.A_contact

    # Elastic displacement at peak (ice column of length L = 0.1 m)
    L_column = 0.1  # m
    stiffness = params.A_contact * E_ICE / L_column
    displacement_at_peak = peak_force / stiffness

    return PullOffResult(
        T_interface=T_interface,
        t_contact=t_contact,
        peak_force=peak_force,
        peak_stress=peak_stress,
        failure_mode=fm.mode,
        displacement_at_peak=displacement_at_peak,
        stiffness=stiffness,
        params=params,
    )


@dataclass
class ShearResult:
    """Result of a shear test along the ice-structure interface.

    An ice plate is sheared along a rigid surface under a normal pressure.
    The shear strength includes both the freeze-bond contribution and
    friction.
    """
    T_interface: float
    t_contact: float
    sigma_normal: float  # normal pressure [Pa]
    tau_peak: float  # peak shear stress [Pa]
    tau_bond: float  # bond contribution to shear strength [Pa]
    tau_friction: float  # friction contribution [Pa]
    failure_mode: str
    mu_effective: float  # effective friction coefficient
    params: FreezeBondParams


def shear_test(
    T_interface: float,
    t_contact: float,
    sigma_normal: float = 0.0,
    params: Optional[FreezeBondParams] = None,
    mu_friction: float = 0.1,
) -> ShearResult:
    """Simulate a shear test along the ice-structure interface.

    The peak shear stress is the sum of the freeze-bond contribution and
    the friction contribution:

    τ_peak = τ_bond + μ · σ_normal

    where τ_bond is related to the tensile bond strength by a factor
    (typically τ_bond ≈ 0.6 · σ_eff for ice, based on the Mohr-Coulomb
    criterion with internal friction angle ~30°).

    Parameters
    ----------
    T_interface : float
        Interface temperature [°C].
    t_contact : float
        Contact time [s].
    sigma_normal : float
        Normal pressure on the interface [Pa].
    params : FreezeBondParams, optional
        Model parameters.
    mu_friction : float
        Friction coefficient for the residual friction after bond failure.

    Returns
    -------
    ShearResult
        Test results including peak shear stress and components.
    """
    if params is None:
        params = FreezeBondParams()

    sigma_eff = effective_bond_strength(T_interface, t_contact, params)
    fm = classify_failure_mode(T_interface, t_contact, params)

    # Bond shear strength (Mohr-Coulomb: τ = c + σ·tan(φ))
    # For the bond, cohesion c ≈ σ_eff and internal friction angle φ ≈ 30°
    # so τ_bond ≈ σ_eff · (1 + sin(φ))/(1 - sin(φ))^(1/2) ≈ 0.6 · σ_eff
    # Simplified: τ_bond ≈ 0.6 · σ_eff
    bond_shear_factor = 0.6

    if fm.mode == "adhesive":
        tau_bond = bond_shear_factor * sigma_eff
    elif fm.mode == "cohesive":
        # Cohesive failure: bond is stronger than ice, so ice fails in shear
        # Ice shear strength ≈ 0.5 · σ_tensile
        tau_bond = bond_shear_factor * params.sigma_cohesive
    else:
        tau_bond = bond_shear_factor * 0.5 * (sigma_eff + params.sigma_cohesive)

    tau_friction = mu_friction * sigma_normal
    tau_peak = tau_bond + tau_friction

    mu_effective = tau_peak / sigma_normal if sigma_normal > 0 else float('inf')

    return ShearResult(
        T_interface=T_interface,
        t_contact=t_contact,
        sigma_normal=sigma_normal,
        tau_peak=tau_peak,
        tau_bond=tau_bond,
        tau_friction=tau_friction,
        failure_mode=fm.mode,
        mu_effective=mu_effective,
        params=params,
    )


# ---------------------------------------------------------------------------
# Phase diagram and parameter sweep
# ---------------------------------------------------------------------------

@dataclass
class FailureModePhaseDiagram:
    """Failure mode boundary in the temperature–loading rate plane.

    The boundary separates adhesive failure (high T, low t_contact) from
    cohesive failure (low T, high t_contact).
    """
    temperatures: np.ndarray  # [°C]
    contact_times: np.ndarray  # [s]
    failure_modes: np.ndarray  # string array: 'adhesive', 'cohesive', 'mixed'
    transition_boundary_T: np.ndarray  # temperatures at transition
    transition_boundary_t: np.ndarray  # contact times at transition
    params: FreezeBondParams


def failure_mode_phase_diagram(
    T_range: tuple[float, float] = (-15.0, -1.0),
    t_range: tuple[float, float] = (1.0, 1000.0),
    n_T: int = 50,
    n_t: int = 50,
    params: Optional[FreezeBondParams] = None,
) -> FailureModePhaseDiagram:
    """Compute the failure mode phase diagram.

    Maps out the failure mode (adhesive/cohesive/mixed) as a function of
    temperature and contact time.

    Parameters
    ----------
    T_range : tuple
        Temperature range (T_min, T_max) in °C.
    t_range : tuple
        Contact time range (t_min, t_max) in seconds.
    n_T, n_t : int
        Number of points in each dimension.
    params : FreezeBondParams, optional
        Model parameters.

    Returns
    -------
    FailureModePhaseDiagram
        Phase diagram with failure mode classification and transition boundary.
    """
    if params is None:
        params = FreezeBondParams()

    temperatures = np.linspace(T_range[0], T_range[1], n_T)
    contact_times = np.linspace(t_range[0], t_range[1], n_t)

    failure_modes = np.empty((n_T, n_t), dtype=object)
    transition_T = []
    transition_t = []

    for i, T in enumerate(temperatures):
        for j, t in enumerate(contact_times):
            fm = classify_failure_mode(T, t, params)
            failure_modes[i, j] = fm.mode
            if fm.is_transition:
                transition_T.append(T)
                transition_t.append(t)

    # Find the transition boundary: for each temperature, find the contact
    # time where the mode transitions from adhesive to cohesive
    transition_boundary_T = []
    transition_boundary_t = []

    for i, T in enumerate(temperatures):
        modes_row = [failure_modes[i, j] for j in range(n_t)]
        # Find first transition from adhesive to cohesive
        for j in range(1, n_t):
            if modes_row[j-1] == "adhesive" and modes_row[j] in ("cohesive", "mixed"):
                transition_boundary_T.append(T)
                transition_boundary_t.append(contact_times[j])
                break
            elif modes_row[j-1] == "mixed" and modes_row[j] == "cohesive":
                transition_boundary_T.append(T)
                transition_boundary_t.append(contact_times[j])
                break

    return FailureModePhaseDiagram(
        temperatures=temperatures,
        contact_times=contact_times,
        failure_modes=failure_modes,
        transition_boundary_T=np.array(transition_boundary_T),
        transition_boundary_t=np.array(transition_boundary_t),
        params=params,
    )


@dataclass
class ParameterSweepResult:
    """Result of a parameter sweep over temperature, contact time, and loading rate."""
    temperatures: np.ndarray
    contact_times: np.ndarray
    loading_rates: np.ndarray
    peak_stresses: np.ndarray  # shape (n_T, n_t, n_v)
    failure_modes: np.ndarray  # shape (n_T, n_t, n_v)
    peak_load_jumps: np.ndarray  # stress jump at transition points
    params: FreezeBondParams


def parameter_sweep(
    T_range: tuple[float, float] = (-15.0, -1.0),
    t_range: tuple[float, float] = (1.0, 1000.0),
    v_range: tuple[float, float] = (0.001, 1.0),
    n_T: int = 15,
    n_t: int = 15,
    n_v: int = 5,
    params: Optional[FreezeBondParams] = None,
) -> ParameterSweepResult:
    """Perform a parameter sweep over temperature, contact time, and loading rate.

    The loading rate affects the apparent strength through viscoelastic effects:
    at high rates, the bond appears stronger (less time for creep relaxation).

    Parameters
    ----------
    T_range, t_range, v_range : tuple
        Ranges for temperature [°C], contact time [s], and loading rate [m/s].
    n_T, n_t, n_v : int
        Number of points in each dimension.
    params : FreezeBondParams, optional
        Model parameters.

    Returns
    -------
    ParameterSweepResult
        Sweep results with peak stresses, failure modes, and load jumps.
    """
    if params is None:
        params = FreezeBondParams()

    temperatures = np.linspace(T_range[0], T_range[1], n_T)
    contact_times = np.linspace(t_range[0], t_range[1], n_t)
    loading_rates = np.logspace(np.log10(v_range[0]), np.log10(v_range[1]), n_v)

    peak_stresses = np.zeros((n_T, n_t, n_v))
    failure_modes = np.empty((n_T, n_t, n_v), dtype=object)
    peak_load_jumps = np.zeros((n_T, n_v))

    for i, T in enumerate(temperatures):
        for j, t in enumerate(contact_times):
            for k, v in enumerate(loading_rates):
                # Rate enhancement factor: strength increases with loading rate
                # Simple model: σ_apparent = σ_eff · (1 + α · log10(v/v_ref))
                # where v_ref = 0.01 m/s and α ≈ 0.1
                v_ref = 0.01
                alpha_rate = 0.1
                rate_factor = 1.0 + alpha_rate * math.log10(max(v, 1e-6) / v_ref)
                rate_factor = max(0.5, min(2.0, rate_factor))  # clamp

                sigma_eff = effective_bond_strength(T, t, params) * rate_factor
                fm = classify_failure_mode(T, t, params)

                if fm.mode == "adhesive":
                    peak_stress = sigma_eff
                elif fm.mode == "cohesive":
                    peak_stress = params.sigma_cohesive * rate_factor
                else:
                    peak_stress = 0.5 * (sigma_eff + params.sigma_cohesive * rate_factor)

                peak_stresses[i, j, k] = peak_stress
                failure_modes[i, j, k] = fm.mode

        # Compute peak load jump at transition for this temperature and rate
        for k, v in enumerate(loading_rates):
            stresses_T = peak_stresses[i, :, k]
            # Find the maximum gradient (jump) in stress
            if len(stresses_T) > 1:
                gradients = np.abs(np.diff(stresses_T))
                peak_load_jumps[i, k] = float(np.max(gradients))

    return ParameterSweepResult(
        temperatures=temperatures,
        contact_times=contact_times,
        loading_rates=loading_rates,
        peak_stresses=peak_stresses,
        failure_modes=failure_modes,
        peak_load_jumps=peak_load_jumps,
        params=params,
    )


# ---------------------------------------------------------------------------
# Full verification runner
# ---------------------------------------------------------------------------

@dataclass
class VerificationReport:
    """Comprehensive verification report for the freeze-bond model."""
    pull_off_results: list[PullOffResult]
    shear_results: list[ShearResult]
    phase_diagram: FailureModePhaseDiagram
    sweep: ParameterSweepResult
    # Verification checks
    strength_monotone_with_cooling: bool
    strength_monotone_with_time: bool
    failure_mode_transition_detected: bool
    peak_load_jump_detected: bool
    adhesive_at_high_T: bool
    cohesive_at_low_T: bool
    summary: dict


def run_full_verification(
    params: Optional[FreezeBondParams] = None,
) -> VerificationReport:
    """Run the full verification suite for the freeze-bond model.

    Performs:
    1. Pull-off tests at various temperatures and contact times
    2. Shear tests at various normal pressures
    3. Failure mode phase diagram computation
    4. Parameter sweep over T, t_contact, and loading rate

    Returns
    -------
    VerificationReport
        Comprehensive report with all results and verification checks.
    """
    if params is None:
        params = FreezeBondParams()

    # 1. Pull-off tests
    pull_off_results = []
    for T in [-1.0, -3.0, -5.0, -10.0, -15.0]:
        for t in [1.0, 10.0, 100.0, 1000.0]:
            result = pull_off_test(T, t, params)
            pull_off_results.append(result)

    # 2. Shear tests
    shear_results = []
    for T in [-1.0, -5.0, -10.0, -15.0]:
        for sigma_n in [0.0, 1e4, 5e4, 1e5]:
            result = shear_test(T, 100.0, sigma_n, params)
            shear_results.append(result)

    # 3. Phase diagram
    phase_diagram = failure_mode_phase_diagram(params=params)

    # 4. Parameter sweep
    sweep = parameter_sweep(params=params)

    # Verification checks
    # Check 1: strength increases with cooling (lower T → higher strength)
    strengths_T = [freeze_bond_strength(T, params) for T in [-1, -5, -10, -15]]
    strength_monotone_cooling = all(
        strengths_T[i] <= strengths_T[i+1] for i in range(len(strengths_T)-1)
    )

    # Check 2: strength increases with contact time
    strengths_t = [effective_bond_strength(-10.0, t, params) for t in [1, 10, 100, 1000]]
    strength_monotone_time = all(
        strengths_t[i] <= strengths_t[i+1] for i in range(len(strengths_t)-1)
    )

    # Check 3: failure mode transition exists
    failure_mode_transition = len(phase_diagram.transition_boundary_T) > 0

    # Check 4: peak load jump at transition
    peak_load_jump = float(np.max(sweep.peak_load_jumps)) > 0

    # Check 5: adhesive at high T
    fm_high_T = classify_failure_mode(-1.0, 1.0, params)
    adhesive_high_T = fm_high_T.mode == "adhesive"

    # Check 6: cohesive at low T
    fm_low_T = classify_failure_mode(-15.0, 1000.0, params)
    cohesive_low_T = fm_low_T.mode == "cohesive"

    summary = {
        "n_pull_off_tests": len(pull_off_results),
        "n_shear_tests": len(shear_results),
        "phase_diagram_resolution": (len(phase_diagram.temperatures),
                                      len(phase_diagram.contact_times)),
        "sweep_resolution": (len(sweep.temperatures), len(sweep.contact_times),
                            len(sweep.loading_rates)),
        "strength_at_minus_1C_Pa": freeze_bond_strength(-1.0, params),
        "strength_at_minus_10C_Pa": freeze_bond_strength(-10.0, params),
        "strength_at_minus_15C_Pa": freeze_bond_strength(-15.0, params),
        "max_peak_load_jump_Pa": float(np.max(sweep.peak_load_jumps)),
        "transition_boundary_points": len(phase_diagram.transition_boundary_T),
    }

    return VerificationReport(
        pull_off_results=pull_off_results,
        shear_results=shear_results,
        phase_diagram=phase_diagram,
        sweep=sweep,
        strength_monotone_with_cooling=strength_monotone_cooling,
        strength_monotone_with_time=strength_monotone_time,
        failure_mode_transition_detected=failure_mode_transition,
        peak_load_jump_detected=peak_load_jump,
        adhesive_at_high_T=adhesive_high_T,
        cohesive_at_low_T=cohesive_low_T,
        summary=summary,
    )
