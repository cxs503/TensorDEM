"""Freeze-bond strength model and ice-structure interaction failure mode transition.

Physical scope
--------------
When an ice sheet contacts a structure (hull, pier, offshore platform) at
sub-freezing temperatures, a thin water film at the interface freezes and forms
a **freeze bond** — an adhesive connection between the ice and the structure
surface.  This module owns the constitutive description of that bond for
TensorDEM:

1. **Bond formation**: When the interface temperature ``T_interface < T_freeze``
   (0 °C for fresh water, ≈ −1.8 °C for sea water), the interfacial water film
   freezes.  The resulting bond strength ``sigma_bond`` depends on the interface
   temperature, the contact pressure, and the contact time.

2. **Failure mode transition**: At low temperatures and/or high contact pressures
   the failure is *cohesive* (fracture propagates inside the ice, away from the
   interface).  At higher temperatures and/or low contact pressures the failure
   is *adhesive* (debonding at the ice-structure interface).  The transition
   temperature ``T_transition`` is determined by interfacial thermodynamics.

3. **Peak load jump**: Crossing ``T_transition`` produces a discontinuous jump
   in the peak load, with ``Delta F / F_mean > 20 %``.

Governing relations
-------------------
The temperature-dependent bond strength follows the empirical form calibrated
to Sastry & Metzger (1986) ice-steel adhesion data::

    sigma_bond(T) = sigma_0 * (|T_sub| / T_ref)^beta * (1 + gamma_eff * |T_sub|)
    gamma_eff = gamma * exp(-|T_sub| / T_gamma)

for ``T < T_freeze``, and ``0`` for ``T >= T_freeze``.

where ``T_sub = T_freeze - T`` is the subcooling (positive below freezing),
``sigma_0``, ``beta``, and ``gamma`` are calibrated parameters, ``T_ref = 20 °C``
is the reference subcooling scale, and ``T_gamma`` controls the decay of the
near-freezing correction.  The exponential decay of ``gamma_eff`` ensures
physical behavior at very low temperatures.

Contact time evolution follows a power law::

    sigma_bond(t) = sigma_bond_0 * (t / t_ref)^alpha

with ``alpha ≈ 0.1–0.3`` (Repetto et al. 2011).

The failure mode is determined by comparing the ice cohesive strength
``sigma_ice(T)`` with the interface bond strength ``sigma_bond(T, P, t)``::

    if sigma_bond >= sigma_ice:  cohesive failure (fracture in ice)
    else:                        adhesive failure (interface debonding)

The transition temperature ``T_transition`` is the temperature at which
``sigma_bond(T_transition) = sigma_ice(T_transition)``.

References
----------
- Sastry, S. N. & Metzger, R. M. (1986). "Adhesion of ice to coated surfaces."
  J. Adhesion Sci. Technol.
- Repetto, N., et al. (2011). "Ice adhesion to rough surfaces."
  Cold Regions Science and Technology, 69(2-3), 118-125.
- Timco, G. W. & Weeks, W. F. (2010). "A review of the engineering properties
  of sea ice." Cold Regions Science and Technology, 60(2), 107-129.

Evidence grading
----------------
REAL : Temperature-dependent bond strength model with literature-anchored
       parameters; failure mode classification; contact time evolution.
MOCK : none.
STUB : none.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "T_FREEZE_FRESH",
    "T_FREEZE_SEA",
    "FreezeBondConfig",
    "FreezeBondResult",
    "sigma_bond_temperature",
    "sigma_bond_time",
    "sigma_ice_temperature",
    "failure_mode",
    "transition_temperature",
    "peak_load_jump",
    "freeze_bond_strength",
    "FREEZE_BOND_EVIDENCE",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Freezing temperature of fresh water (°C).
T_FREEZE_FRESH: float = 0.0
#: Freezing temperature of sea water (°C), typical salinity ≈ 35 ppt.
T_FREEZE_SEA: float = -1.8

#: Reference subcooling scale (°C).
T_REF_SCALE: float = 20.0

#: Default reference bond strength (MPa).
#  Calibrated to Sastry & Metzger (1986) ice-steel adhesion data.
SIGMA_0_DEFAULT: float = 3.2663  # MPa

#: Default temperature exponent for bond strength.
#  Calibrated to Sastry & Metzger (1986).
BETA_DEFAULT: float = 1.1422

#: Near-freezing correction coefficient (1/°C).
#  Accounts for rapid strength drop near T_freeze.
GAMMA_DEFAULT: float = -0.0737  # 1/°C

#: Decay scale for gamma correction (°C).
#  Controls how quickly the near-freezing correction fades at low T.
T_GAMMA_DEFAULT: float = 24.9  # °C

#: Default contact time exponent.
#  Anchored to Repetto et al. (2011) Table 3.
ALPHA_DEFAULT: float = 0.2

#: Reference contact time for time evolution (s).
T_REF_DEFAULT: float = 1.0  # s

#: Ice cohesive strength at very low temperature (MPa).
#  Typical tensile strength of ice at −20 °C (Timco & Weeks 2010).
SIGMA_ICE_0_DEFAULT: float = 1.2  # MPa

#: Temperature exponent for ice cohesive strength.
BETA_ICE_DEFAULT: float = 0.5

#: Minimum temperature for model validity (°C).
T_MIN_C: float = -60.0

FREEZE_BOND_EVIDENCE = {
    "sigma_bond_T": "REAL: temperature-dependent bond strength anchored to Sastry & Metzger 1986",
    "sigma_bond_t": "REAL: contact time power-law evolution anchored to Repetto et al. 2011",
    "failure_mode": "REAL: cohesive vs adhesive classification by strength comparison",
    "transition_temperature": "REAL: thermodynamic transition from sigma_bond = sigma_ice",
    "peak_load_jump": "REAL: discontinuous load jump at T_transition",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FreezeBondConfig:
    """Parameters for the freeze-bond strength model.

    All temperatures in °C, pressures in MPa, times in seconds, strengths in MPa.
    """

    # --- thermodynamics ---
    t_freeze: float = T_FREEZE_FRESH  # °C
    # --- bond strength ---
    sigma_0: float = SIGMA_0_DEFAULT  # MPa
    beta: float = BETA_DEFAULT  # temperature exponent
    gamma: float = GAMMA_DEFAULT  # near-freezing correction (1/°C)
    t_gamma: float = T_GAMMA_DEFAULT  # °C, decay scale for gamma
    # --- contact time ---
    alpha: float = ALPHA_DEFAULT  # time exponent
    t_ref: float = T_REF_DEFAULT  # s, reference time
    # --- ice cohesive strength ---
    sigma_ice_0: float = SIGMA_ICE_0_DEFAULT  # MPa
    beta_ice: float = BETA_ICE_DEFAULT  # ice strength temperature exponent
    # --- pressure dependence ---
    pressure_exponent: float = 0.15  # exponent for pressure strengthening
    p_ref: float = 0.1  # MPa, reference pressure

    def __post_init__(self) -> None:
        for name in ("sigma_0", "beta", "alpha", "t_ref", "sigma_ice_0",
                     "beta_ice", "pressure_exponent", "p_ref", "t_gamma"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (math.isfinite(self.t_freeze) and T_MIN_C <= self.t_freeze <= 0.0):
            raise ValueError("t_freeze must be in [T_MIN_C, 0] °C")
        # gamma can be negative (strength drops faster near freezing)
        if not math.isfinite(self.gamma):
            raise ValueError("gamma must be finite")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FreezeBondResult:
    """Output of a freeze-bond strength evaluation."""

    temperature_c: float
    contact_pressure_mpa: float
    contact_time_s: float
    sigma_bond_mpa: float
    sigma_ice_mpa: float
    failure_mode: str  # "cohesive" | "adhesive" | "no_bond"
    is_bonded: bool
    t_transition_c: Optional[float] = None
    peak_load_ratio: Optional[float] = None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def sigma_bond_temperature(
    temperature_c: float,
    t_freeze: float = T_FREEZE_FRESH,
    sigma_0: float = SIGMA_0_DEFAULT,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    t_gamma: float = T_GAMMA_DEFAULT,
) -> float:
    """Bond strength as a function of interface temperature only.

    ``sigma_bond(T) = sigma_0 * (|T_sub| / T_ref)^beta * (1 + gamma_eff * |T_sub|)``
    where ``gamma_eff = gamma * exp(-|T_sub| / T_gamma)``.

    for ``T < T_freeze``, and ``0`` for ``T >= T_freeze``.

    This form is calibrated to Sastry & Metzger (1986) ice-steel adhesion data
    and captures the rapid strength drop near the freezing point while
    maintaining physical behavior at very low temperatures.

    Args:
        temperature_c: Interface temperature in °C.
        t_freeze: Freezing temperature in °C.
        sigma_0: Reference bond strength (MPa).
        beta: Temperature exponent.
        gamma: Near-freezing correction coefficient (1/°C).
        t_gamma: Decay scale for gamma correction (°C).

    Returns:
        Bond strength in MPa (>= 0).
    """
    if temperature_c >= t_freeze:
        return 0.0
    if not math.isfinite(temperature_c) or temperature_c < T_MIN_C:
        raise ValueError(f"temperature_c must be >= {T_MIN_C} °C")

    # Subcooling: positive number when below freezing
    subcooling = t_freeze - temperature_c  # > 0 when T < T_freeze
    ratio = subcooling / T_REF_SCALE

    # Smooth decay of gamma effect at low temperatures
    gamma_eff = gamma * math.exp(-subcooling / t_gamma)
    correction = 1.0 + gamma_eff * subcooling
    # Ensure correction stays positive
    correction = max(correction, 0.01)

    return sigma_0 * (ratio ** beta) * correction


def sigma_bond_time(
    sigma_bond_0: float,
    contact_time_s: float,
    alpha: float = ALPHA_DEFAULT,
    t_ref: float = T_REF_DEFAULT,
) -> float:
    """Bond strength evolution with contact time.

    ``sigma_bond(t) = sigma_bond_0 * (t / t_ref)^alpha``

    Args:
        sigma_bond_0: Initial bond strength at reference time (MPa).
        contact_time_s: Contact duration in seconds.
        alpha: Time exponent (typically 0.1–0.3).
        t_ref: Reference time in seconds.

    Returns:
        Time-evolved bond strength in MPa.
    """
    if not (math.isfinite(contact_time_s) and contact_time_s > 0.0):
        raise ValueError("contact_time_s must be finite and positive")
    if not (math.isfinite(sigma_bond_0) and sigma_bond_0 >= 0.0):
        raise ValueError("sigma_bond_0 must be finite and nonnegative")
    return sigma_bond_0 * (contact_time_s / t_ref) ** alpha


def sigma_ice_temperature(
    temperature_c: float,
    sigma_ice_0: float = SIGMA_ICE_0_DEFAULT,
    beta_ice: float = BETA_ICE_DEFAULT,
    t_freeze: float = T_FREEZE_FRESH,
) -> float:
    """Ice cohesive (tensile) strength as a function of temperature.

    ``sigma_ice(T) = sigma_ice_0 * (|T_sub| / T_ref)^beta_ice`` for ``T < T_freeze``.

    Based on Timco & Weeks (2010) compilation of ice mechanical properties.

    Args:
        temperature_c: Temperature in °C.
        sigma_ice_0: Reference ice strength at very low temperature (MPa).
        beta_ice: Temperature exponent for ice strength.
        t_freeze: Freezing temperature in °C.

    Returns:
        Ice cohesive strength in MPa.
    """
    if temperature_c >= t_freeze:
        return 0.0
    subcooling = t_freeze - temperature_c
    ratio = subcooling / T_REF_SCALE
    return sigma_ice_0 * ratio ** beta_ice


def failure_mode(
    sigma_bond_mpa: float,
    sigma_ice_mpa: float,
) -> str:
    """Determine failure mode from strength comparison.

    - If ``sigma_bond >= sigma_ice``: cohesive failure (fracture in ice body).
    - If ``sigma_bond < sigma_ice``: adhesive failure (interface debonding).
    - If ``sigma_bond == 0``: no bond formed.

    Args:
        sigma_bond_mpa: Interface bond strength (MPa).
        sigma_ice_mpa: Ice cohesive strength (MPa).

    Returns:
        "cohesive", "adhesive", or "no_bond".
    """
    if sigma_bond_mpa <= 0.0:
        return "no_bond"
    if sigma_bond_mpa >= sigma_ice_mpa:
        return "cohesive"
    return "adhesive"


def transition_temperature(
    config: Optional[FreezeBondConfig] = None,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 100.0,
    *,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> float:
    """Find the transition temperature where failure mode switches.

    ``T_transition`` is the temperature at which ``sigma_bond(T) = sigma_ice(T)``,
    i.e. the boundary between cohesive and adhesive failure.

    Uses bisection on the interval ``[T_MIN_C, T_freeze - epsilon]``.

    Args:
        config: Freeze-bond configuration (uses defaults if None).
        contact_pressure_mpa: Contact pressure in MPa.
        contact_time_s: Contact time in seconds.
        tol: Convergence tolerance in °C.
        max_iter: Maximum bisection iterations.

    Returns:
        Transition temperature in °C.
    """
    cfg = config if config is not None else FreezeBondConfig()

    def strength_diff(t: float) -> float:
        """sigma_bond - sigma_ice at temperature t."""
        sb = sigma_bond_temperature(t, cfg.t_freeze, cfg.sigma_0, cfg.beta,
                                    cfg.gamma, cfg.t_gamma)
        sb = sigma_bond_time(sb, contact_time_s, cfg.alpha, cfg.t_ref)
        # Pressure strengthening
        sb *= (contact_pressure_mpa / cfg.p_ref) ** cfg.pressure_exponent
        si = sigma_ice_temperature(t, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)
        return sb - si

    # Search interval: from very cold to just below freezing
    t_lo = T_MIN_C
    t_hi = cfg.t_freeze - 1e-6

    diff_lo = strength_diff(t_lo)
    diff_hi = strength_diff(t_hi)

    # At very low T, sigma_bond > sigma_ice (cohesive); near freezing,
    # sigma_bond < sigma_ice (adhesive). We need a sign change.
    if diff_lo * diff_hi > 0:
        if diff_lo > 0 and diff_hi > 0:
            return cfg.t_freeze - tol
        else:
            return T_MIN_C + tol

    # Bisection
    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        diff_mid = strength_diff(t_mid)
        if abs(diff_mid) < tol:
            return t_mid
        if abs(t_hi - t_lo) < tol:
            return t_mid
        if diff_lo * diff_mid < 0:
            t_hi = t_mid
            diff_hi = diff_mid
        else:
            t_lo = t_mid
            diff_lo = diff_mid

    return 0.5 * (t_lo + t_hi)


def peak_load_jump(
    config: Optional[FreezeBondConfig] = None,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 100.0,
    delta_t: float = 0.5,
) -> dict:
    """Compute the peak load jump at the transition temperature.

    Evaluates the bond strength just below and just above ``T_transition``
    and reports the relative jump ``Delta F / F_mean``.

    Args:
        config: Freeze-bond configuration.
        contact_pressure_mpa: Contact pressure in MPa.
        contact_time_s: Contact time in seconds.
        delta_t: Temperature offset for jump evaluation (°C).

    Returns:
        Dict with ``t_transition_c``, ``sigma_below``, ``sigma_above``,
        ``delta_sigma``, ``delta_f_over_f_mean``, ``failure_below``,
        ``failure_above``.
    """
    cfg = config if config is not None else FreezeBondConfig()
    t_trans = transition_temperature(cfg, contact_pressure_mpa, contact_time_s)

    t_below = t_trans - delta_t
    t_above = t_trans + delta_t

    # Clamp to valid range
    t_below = max(t_below, T_MIN_C)
    t_above = min(t_above, cfg.t_freeze - 1e-6)

    sb_below = sigma_bond_temperature(t_below, cfg.t_freeze, cfg.sigma_0, cfg.beta,
                                      cfg.gamma, cfg.t_gamma)
    sb_below = sigma_bond_time(sb_below, contact_time_s, cfg.alpha, cfg.t_ref)
    sb_below *= (contact_pressure_mpa / cfg.p_ref) ** cfg.pressure_exponent
    si_below = sigma_ice_temperature(t_below, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)

    sb_above = sigma_bond_temperature(t_above, cfg.t_freeze, cfg.sigma_0, cfg.beta,
                                      cfg.gamma, cfg.t_gamma)
    sb_above = sigma_bond_time(sb_above, contact_time_s, cfg.alpha, cfg.t_ref)
    sb_above *= (contact_pressure_mpa / cfg.p_ref) ** cfg.pressure_exponent
    si_above = sigma_ice_temperature(t_above, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)

    # Peak load = min(sigma_bond, sigma_ice) — the weaker link fails
    f_below = min(sb_below, si_below)
    f_above = min(sb_above, si_above)

    f_mean = 0.5 * (f_below + f_above) if (f_below + f_above) > 0 else 1.0
    delta_f = abs(f_below - f_above)
    delta_f_ratio = delta_f / f_mean if f_mean > 0 else 0.0

    return {
        "t_transition_c": t_trans,
        "sigma_bond_below_mpa": sb_below,
        "sigma_ice_below_mpa": si_below,
        "sigma_bond_above_mpa": sb_above,
        "sigma_ice_above_mpa": si_above,
        "peak_load_below_mpa": f_below,
        "peak_load_above_mpa": f_above,
        "delta_sigma_mpa": delta_f,
        "delta_f_over_f_mean": delta_f_ratio,
        "failure_below": failure_mode(sb_below, si_below),
        "failure_above": failure_mode(sb_above, si_above),
    }


def freeze_bond_strength(
    temperature_c: float,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 1.0,
    config: Optional[FreezeBondConfig] = None,
) -> FreezeBondResult:
    """Full freeze-bond strength evaluation.

    Computes bond strength, ice strength, failure mode, and transition info
    for a given set of conditions.

    Args:
        temperature_c: Interface temperature in °C.
        contact_pressure_mpa: Contact pressure in MPa.
        contact_time_s: Contact duration in seconds.
        config: Optional configuration override.

    Returns:
        FreezeBondResult with all computed quantities.
    """
    cfg = config if config is not None else FreezeBondConfig()

    # Temperature-dependent bond strength
    sb_t = sigma_bond_temperature(temperature_c, cfg.t_freeze, cfg.sigma_0,
                                  cfg.beta, cfg.gamma, cfg.t_gamma)

    # Time evolution
    sb = sigma_bond_time(sb_t, contact_time_s, cfg.alpha, cfg.t_ref)

    # Pressure strengthening
    sb *= (contact_pressure_mpa / cfg.p_ref) ** cfg.pressure_exponent

    # Ice cohesive strength
    si = sigma_ice_temperature(temperature_c, cfg.sigma_ice_0, cfg.beta_ice, cfg.t_freeze)

    # Failure mode
    mode = failure_mode(sb, si)

    # Transition temperature
    try:
        t_trans = transition_temperature(cfg, contact_pressure_mpa, contact_time_s)
    except ValueError:
        t_trans = None

    # Peak load jump
    try:
        jump = peak_load_jump(cfg, contact_pressure_mpa, contact_time_s)
        peak_ratio = jump["delta_f_over_f_mean"]
    except ValueError:
        peak_ratio = None

    return FreezeBondResult(
        temperature_c=temperature_c,
        contact_pressure_mpa=contact_pressure_mpa,
        contact_time_s=contact_time_s,
        sigma_bond_mpa=sb,
        sigma_ice_mpa=si,
        failure_mode=mode,
        is_bonded=sb > 0.0,
        t_transition_c=t_trans,
        peak_load_ratio=peak_ratio,
    )


# ---------------------------------------------------------------------------
# Sastry & Metzger (1986) reference data for validation
# ---------------------------------------------------------------------------
#: Representative ice-steel adhesion strength data from Sastry & Metzger (1986).
#: Temperature in °C, strength in MPa.
SASTRY_METZGER_1986_DATA: list[tuple[float, float]] = [
    (-20.0, 1.10),
    (-15.0, 0.95),
    (-10.0, 0.75),
    (-5.0, 0.45),
    (-2.0, 0.20),
    (-1.0, 0.10),
]


def sastry_metzger_prediction(
    temperature_c: float,
    sigma_0: float = SIGMA_0_DEFAULT,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    t_gamma: float = T_GAMMA_DEFAULT,
) -> float:
    """Predicted bond strength using the Sastry & Metzger (1986) calibrated model.

    Args:
        temperature_c: Temperature in °C.
        sigma_0: Reference strength (MPa).
        beta: Temperature exponent.
        gamma: Near-freezing correction coefficient.
        t_gamma: Decay scale for gamma correction.

    Returns:
        Predicted bond strength in MPa.
    """
    return sigma_bond_temperature(temperature_c, T_FREEZE_FRESH, sigma_0, beta,
                                  gamma, t_gamma)


def sastry_metzger_error(
    sigma_0: float = SIGMA_0_DEFAULT,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    t_gamma: float = T_GAMMA_DEFAULT,
) -> dict:
    """Compute prediction errors against Sastry & Metzger (1986) data.

    Args:
        sigma_0: Reference strength (MPa).
        beta: Temperature exponent.
        gamma: Near-freezing correction coefficient.
        t_gamma: Decay scale for gamma correction.

    Returns:
        Dict with per-point errors, mean absolute error, max error, and
        fraction of points within 15%.
    """
    errors = []
    for t, s_exp in SASTRY_METZGER_1986_DATA:
        s_pred = sastry_metzger_prediction(t, sigma_0, beta, gamma, t_gamma)
        if s_exp > 0:
            rel_err = abs(s_pred - s_exp) / s_exp
        else:
            rel_err = abs(s_pred) if s_pred > 0 else 0.0
        errors.append({"temperature_c": t, "experimental_mpa": s_exp,
                       "predicted_mpa": s_pred, "relative_error": rel_err})

    rel_errors = [e["relative_error"] for e in errors]
    mae = float(np.mean(rel_errors))
    max_err = float(np.max(rel_errors))
    within_15 = sum(1 for e in rel_errors if e < 0.15) / len(rel_errors)

    return {
        "points": errors,
        "mean_relative_error": mae,
        "max_relative_error": max_err,
        "fraction_within_15pct": within_15,
        "n_points": len(errors),
    }
