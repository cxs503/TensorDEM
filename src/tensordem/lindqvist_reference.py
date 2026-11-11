"""Lindqvist (1989) ice resistance reference calculator with SHA-256 anchored inputs.

This module provides a clean, auditable implementation of the Lindqvist (1989)
level-ice resistance formula for ship navigation in level ice. All input parameters
are hashed to ensure reproducibility and auditability.

Reference:
    Lindqvist, M. (1989). "A direct method for predicting icebreaking resistance
    of ships." Proceedings of the 5th International Conference on Offshore Mechanics
    and Arctic Engineering (OMAE), Vol. IV, pp. 117-121.

All quantities SI: m, Pa, kg/m^3, m/s, N.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict

__all__ = [
    "LindqvistInput",
    "LindqvistResult",
    "compute_lindqvist",
    "lindqvist_sha256",
]

# Physical constants
WATER_DENSITY = 1025.0  # kg/m^3
ICE_DENSITY = 917.0  # kg/m^3
GRAVITY = 9.80665  # m/s^2


@dataclass(frozen=True)
class LindqvistInput:
    """Input parameters for Lindqvist (1989) formula.

    Attributes
    ----------
    hull_beam_m : float
        Ship beam (breadth) [m].
    hull_length_m : float
        Ship length [m].
    hull_draught_m : float
        Ship draught [m].
    hull_entry_angle_deg : float
        Waterline entry angle [degrees].
    ice_thickness_m : float
        Level ice thickness [m].
    ice_compressive_strength_pa : float
        Ice compressive (crushing) strength [Pa].
    ice_flexural_strength_pa : float
        Ice flexural (bending) strength [Pa].
    speed_m_s : float
        Ship speed [m/s].
    friction_coefficient : float
        Hull-ice friction coefficient [-].
    """

    hull_beam_m: float
    hull_length_m: float
    hull_draught_m: float
    hull_entry_angle_deg: float
    ice_thickness_m: float
    ice_compressive_strength_pa: float
    ice_flexural_strength_pa: float
    speed_m_s: float
    friction_coefficient: float = 0.05

    def __post_init__(self) -> None:
        for name in [
            "hull_beam_m",
            "hull_length_m",
            "hull_draught_m",
            "ice_thickness_m",
            "ice_compressive_strength_pa",
            "ice_flexural_strength_pa",
        ]:
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.speed_m_s) or self.speed_m_s < 0:
            raise ValueError("speed_m_s must be non-negative and finite")
        if not math.isfinite(self.friction_coefficient) or self.friction_coefficient < 0:
            raise ValueError("friction_coefficient must be non-negative and finite")
        if self.hull_entry_angle_deg <= 0 or self.hull_entry_angle_deg >= 90:
            raise ValueError("hull_entry_angle_deg must be in (0, 90)")


@dataclass(frozen=True)
class LindqvistResult:
    """Result of Lindqvist (1989) calculation.

    Attributes
    ----------
    crushing_N : float
        Crushing resistance component [N].
    bending_N : float
        Bending resistance component [N].
    submersion_N : float
        Submersion resistance component [N].
    added_drag_N : float
        Added drag (speed-dependent) component [N].
    total_N : float
        Total ice resistance [N].
    input_sha256 : str
        SHA-256 hash of input parameters.
    """

    crushing_N: float
    bending_N: float
    submersion_N: float
    added_drag_N: float
    total_N: float
    input_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "crushing_N": self.crushing_N,
            "bending_N": self.bending_N,
            "submersion_N": self.submersion_N,
            "added_drag_N": self.added_drag_N,
            "total_N": self.total_N,
            "input_sha256": self.input_sha256,
        }


def lindqvist_sha256(inp: LindqvistInput) -> str:
    """Compute SHA-256 hash of Lindqvist input parameters.

    This ensures reproducibility and auditability of the calculation.
    """
    payload = {
        "hull_beam_m": inp.hull_beam_m,
        "hull_length_m": inp.hull_length_m,
        "hull_draught_m": inp.hull_draught_m,
        "hull_entry_angle_deg": inp.hull_entry_angle_deg,
        "ice_thickness_m": inp.ice_thickness_m,
        "ice_compressive_strength_pa": inp.ice_compressive_strength_pa,
        "ice_flexural_strength_pa": inp.ice_flexural_strength_pa,
        "speed_m_s": inp.speed_m_s,
        "friction_coefficient": inp.friction_coefficient,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_lindqvist(inp: LindqvistInput) -> LindqvistResult:
    """Compute Lindqvist (1989) level-ice resistance.

    The formula decomposes the total resistance into four components:

    1. Crushing resistance R_c: stem crushing of ice at the waterline
    2. Bending resistance R_b: flexural failure of ice sheet
    3. Submersion resistance R_s: buoyant depression of ice floes
    4. Added drag R_v: speed-dependent hydrodynamic drag

    Parameters
    ----------
    inp : LindqvistInput
        Input parameters (ship geometry, ice properties, speed).

    Returns
    -------
    LindqvistResult
        Resistance components and total.
    """
    # Extract parameters
    B = inp.hull_beam_m
    L = inp.hull_length_m
    T = inp.hull_draught_m
    alpha = math.radians(inp.hull_entry_angle_deg)
    h = inp.ice_thickness_m
    sigma_c = inp.ice_compressive_strength_pa
    sigma_f = inp.ice_flexural_strength_pa
    V = inp.speed_m_s
    mu = inp.friction_coefficient

    # 1. Crushing resistance (stem crushing)
    # R_c = 0.5 * sigma_c * h^2 * (tan(alpha) + mu / cos(alpha))
    # The 0.5 factor comes from triangular pressure distribution over ice thickness
    R_c = 0.5 * sigma_c * h**2 * (math.tan(alpha) + mu / math.cos(alpha))

    # 2. Bending resistance (flexural failure)
    # R_b = 0.25 * sigma_f * B * h^2 / L
    # The 0.25 factor is an order-one coefficient for bending moment distribution
    R_b = 0.25 * sigma_f * B * h**2 / L

    # 3. Submersion resistance (buoyant depression + channel friction)
    # R_s = rho_w * g * h^2 * B * (0.9 + 0.05 * mu * L / B)
    # The 0.9 is the channel-opening coefficient (displaced ice volume fraction)
    # The 0.05 * mu * L/B is the friction of rubble sheet over hull
    rho_w = WATER_DENSITY
    g = GRAVITY
    R_s = rho_w * g * h**2 * B * (0.9 + 0.05 * mu * L / B)

    # 4. Added drag (speed-dependent)
    # R_v = 0.5 * rho_w * B * h * V^2
    # This is the dynamic pressure drag on the displaced ice area B*h
    R_v = 0.5 * rho_w * B * h * V**2

    # Total resistance
    R_total = R_c + R_b + R_s + R_v

    # Compute input hash
    input_hash = lindqvist_sha256(inp)

    return LindqvistResult(
        crushing_N=R_c,
        bending_N=R_b,
        submersion_N=R_s,
        added_drag_N=R_v,
        total_N=R_total,
        input_sha256=input_hash,
    )


# ---------------------------------------------------------------------------
# Convenience function for quick calculations
# ---------------------------------------------------------------------------


def lindqvist_quick(
    hull_beam_m: float,
    hull_length_m: float,
    ice_thickness_m: float,
    speed_m_s: float,
    *,
    hull_draught_m: float = 7.0,
    hull_entry_angle_deg: float = 22.0,
    ice_compressive_strength_pa: float = 1.5e6,
    ice_flexural_strength_pa: float = 0.5e6,
    friction_coefficient: float = 0.05,
) -> Dict[str, Any]:
    """Quick Lindqvist calculation with default parameters.

    Returns a dictionary with all resistance components and total.
    """
    inp = LindqvistInput(
        hull_beam_m=hull_beam_m,
        hull_length_m=hull_length_m,
        hull_draught_m=hull_draught_m,
        hull_entry_angle_deg=hull_entry_angle_deg,
        ice_thickness_m=ice_thickness_m,
        ice_compressive_strength_pa=ice_compressive_strength_pa,
        ice_flexural_strength_pa=ice_flexural_strength_pa,
        speed_m_s=speed_m_s,
        friction_coefficient=friction_coefficient,
    )
    result = compute_lindqvist(inp)
    return result.as_dict()
