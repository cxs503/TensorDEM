"""Multi-year (MY) ice parameterization and mixed ice field support.

Why this module exists
----------------------
The original Lindqvist (1989) model was calibrated for first-year (FY) level
ice only.  Arctic shipping routes increasingly encounter multi-year (MY) ice
and mixed FY/MY fields.  MY ice has significantly higher mechanical properties:

* Compressive strength σ_c:  FY ≈ 1-3 MPa,  MY ≈ 5-10 MPa
* Flexural strength σ_f:     FY ≈ 0.4-0.7 MPa, MY ≈ 1.0-2.0 MPa
* Elastic modulus E:         FY ≈ 4-6 GPa,    MY ≈ 7-12 GPa
* Fracture toughness K_IC:   FY ≈ 0.1-0.3 MPa·m^0.5, MY ≈ 0.3-0.8 MPa·m^0.5

This module provides:
1. MY ice material property models with literature-based ranges
2. Mixed ice field FSD (Floe Size Distribution) sampling with bimodal
   size-strength distribution
3. Lindqvist MY correction factor k_MY(σ_c, h_i) for R_rot and R_clear terms
4. A unified ``lindqvist_my_components`` that handles pure FY, pure MY,
   and mixed ice fields.

All quantities are SI: m, Pa, kg/m^3, m/s, N.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .formula import (
    GRAVITY,
    ICE_DENSITY,
    WATER_DENSITY,
    HullForm,
    IceState,
    _finite,
    added_drag,
    bending_resistance,
    crushing_resistance,
    submersion_resistance,
)

# ---------------------------------------------------------------------------
# MY ice material property ranges (literature-based)
# ---------------------------------------------------------------------------
# References (summarised from standard ice mechanics texts):
#   - Timco & Weeks (2010) "A review of the mechanical properties of sea ice"
#   - Schulson & Duval (2009) "Structure and Mechanics of Ice"
#   - IACS Unified Requirements for Polar Ships (UR I)

#: FY ice typical ranges (compressive strength, flexural strength, E, K_IC)
FY_RANGES = {
    "compressive_strength_pa": (1.0e6, 3.0e6),
    "flexural_strength_pa": (0.4e6, 0.7e6),
    "elastic_modulus_pa": (4.0e9, 6.0e9),
    "fracture_toughness_pa_m05": (0.1e6, 0.3e6),
}

#: MY ice typical ranges (compressive strength, flexural strength, E, K_IC)
MY_RANGES = {
    "compressive_strength_pa": (5.0e6, 10.0e6),
    "flexural_strength_pa": (1.0e6, 2.0e6),
    "elastic_modulus_pa": (7.0e9, 12.0e9),
    "fracture_toughness_pa_m05": (0.3e6, 0.8e6),
}

#: Default FY ice properties (mid-range)
FY_DEFAULT = {
    "compressive_strength_pa": 1.5e6,
    "flexural_strength_pa": 0.5e6,
    "elastic_modulus_pa": 5.0e9,
    "fracture_toughness_pa_m05": 0.2e6,
}

#: Default MY ice properties (mid-range)
MY_DEFAULT = {
    "compressive_strength_pa": 7.5e6,
    "flexural_strength_pa": 1.5e6,
    "elastic_modulus_pa": 9.5e9,
    "fracture_toughness_pa_m05": 0.55e6,
}


@dataclass(frozen=True)
class IceAgeProperties:
    """Ice age class with property ranges and defaults."""

    label: str  # "FY" or "MY"
    compressive_strength_pa: float
    flexural_strength_pa: float
    elastic_modulus_pa: float
    fracture_toughness_pa_m05: float
    density_kg_m3: float = ICE_DENSITY

    @classmethod
    def fy(cls, *, sigma_c: float | None = None, sigma_f: float | None = None,
           E: float | None = None, K_IC: float | None = None) -> "IceAgeProperties":
        """First-year ice properties."""
        return cls(
            label="FY",
            compressive_strength_pa=sigma_c if sigma_c is not None else FY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=sigma_f if sigma_f is not None else FY_DEFAULT["flexural_strength_pa"],
            elastic_modulus_pa=E if E is not None else FY_DEFAULT["elastic_modulus_pa"],
            fracture_toughness_pa_m05=K_IC if K_IC is not None else FY_DEFAULT["fracture_toughness_pa_m05"],
        )

    @classmethod
    def my(cls, *, sigma_c: float | None = None, sigma_f: float | None = None,
           E: float | None = None, K_IC: float | None = None) -> "IceAgeProperties":
        """Multi-year ice properties."""
        return cls(
            label="MY",
            compressive_strength_pa=sigma_c if sigma_c is not None else MY_DEFAULT["compressive_strength_pa"],
            flexural_strength_pa=sigma_f if sigma_f is not None else MY_DEFAULT["flexural_strength_pa"],
            elastic_modulus_pa=E if E is not None else MY_DEFAULT["elastic_modulus_pa"],
            fracture_toughness_pa_m05=K_IC if K_IC is not None else MY_DEFAULT["fracture_toughness_pa_m05"],
        )

    def to_ice_state(self, thickness_m: float) -> IceState:
        """Convert to the standard IceState for use with existing formulas."""
        return IceState(
            thickness_m=thickness_m,
            compressive_strength_pa=self.compressive_strength_pa,
            flexural_strength_pa=self.flexural_strength_pa,
            elastic_modulus_pa=self.elastic_modulus_pa,
            density_kg_m3=self.density_kg_m3,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "compressive_strength_pa": self.compressive_strength_pa,
            "flexural_strength_pa": self.flexural_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "fracture_toughness_pa_m05": self.fracture_toughness_pa_m05,
            "density_kg_m3": self.density_kg_m3,
        }


# ---------------------------------------------------------------------------
# MY correction factor k_MY(σ_c, h_i)
# ---------------------------------------------------------------------------
# The Lindqvist model decomposes resistance as:
#   R_ice = R_sub + R_rot + R_clear
# where R_sub is submersion, R_rot is rotation/bending, R_clear is clearing.
#
# For MY ice, the rotation and clearing terms must be scaled because MY ice
# has higher strength and produces larger floes that require more energy to
# break and clear.  We introduce a correction factor:
#
#   k_MY = (σ_c_MY / σ_c_FY)^α · (h_i / h_ref)^β
#
# where α ≈ 0.5-0.7 (sub-linear because fracture is not purely strength-driven)
# and β ≈ 0.3-0.5 (thickness effect on floe size).
#
# This factor multiplies the R_rot and R_clear terms only; R_sub is primarily
# buoyancy-driven and is less sensitive to ice strength.

MY_CORRECTION_ALPHA = 0.6   # strength exponent
MY_CORRECTION_BETA = 0.4    # thickness exponent
MY_REFERENCE_THICKNESS = 1.0  # h_ref = 1 m


def my_correction_factor(
    sigma_c_pa: float,
    thickness_m: float,
    *,
    sigma_c_fy: float = FY_DEFAULT["compressive_strength_pa"],
    alpha: float = MY_CORRECTION_ALPHA,
    beta: float = MY_CORRECTION_BETA,
    h_ref: float = MY_REFERENCE_THICKNESS,
) -> float:
    """MY ice strength correction factor k_MY(σ_c, h_i).

    Returns a multiplicative factor >= 1.0 that scales the rotation and
    clearing resistance terms for ice stronger than the FY baseline.

    Parameters
    ----------
    sigma_c_pa : float
        Compressive strength of the ice [Pa].
    thickness_m : float
        Ice thickness [m].
    sigma_c_fy : float
        Reference FY compressive strength [Pa].
    alpha : float
        Strength exponent (default 0.6).
    beta : float
        Thickness exponent (default 0.4).
    h_ref : float
        Reference thickness [m] (default 1.0).
    """
    sigma_c_pa = _finite("sigma_c_pa", sigma_c_pa, positive=True)
    thickness_m = _finite("thickness_m", thickness_m, positive=True)
    sigma_c_fy = _finite("sigma_c_fy", sigma_c_fy, positive=True)

    strength_ratio = sigma_c_pa / sigma_c_fy
    thickness_ratio = thickness_m / h_ref

    k_my = (strength_ratio ** alpha) * (thickness_ratio ** beta)
    return max(1.0, k_my)  # never reduce below FY baseline


# ---------------------------------------------------------------------------
# Mixed ice field FSD sampling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MixedIceField:
    """A mixed FY/MY ice field with bimodal floe size distribution.

    The field is characterised by:
    - my_fraction: proportion of MY ice by area (0 = pure FY, 1 = pure MY)
    - thickness_m: nominal ice thickness (applies to both types)
    - fy_properties / my_properties: material properties for each type
    - fsd_params: floe size distribution parameters (power-law)

    The bimodal FSD is generated by sampling floe sizes from two power-law
    distributions with different characteristic sizes:
    - FY floes: smaller, D_char ≈ 5-20 m
    - MY floes: larger, D_char ≈ 20-100 m (due to rafting/ridging history)
    """

    my_fraction: float  # 0..1
    thickness_m: float
    fy_properties: IceAgeProperties
    my_properties: IceAgeProperties
    fy_char_floe_size_m: float = 10.0
    my_char_floe_size_m: float = 40.0
    fsd_exponent: float = -2.0  # power-law exponent for FSD

    def __post_init__(self) -> None:
        _finite("my_fraction", self.my_fraction, nonnegative=True)
        if self.my_fraction > 1.0:
            raise ValueError("my_fraction must be <= 1.0")
        _finite("thickness_m", self.thickness_m, positive=True)

    def effective_strength(self) -> dict[str, float]:
        """Area-weighted effective strength properties for the mixed field."""
        fy = self.fy_fraction
        my = self.my_fraction
        return {
            "compressive_strength_pa": (
                fy * self.fy_properties.compressive_strength_pa
                + my * self.my_properties.compressive_strength_pa
            ),
            "flexural_strength_pa": (
                fy * self.fy_properties.flexural_strength_pa
                + my * self.my_properties.flexural_strength_pa
            ),
            "elastic_modulus_pa": (
                fy * self.fy_properties.elastic_modulus_pa
                + my * self.my_properties.elastic_modulus_pa
            ),
            "fracture_toughness_pa_m05": (
                fy * self.fy_properties.fracture_toughness_pa_m05
                + my * self.my_properties.fracture_toughness_pa_m05
            ),
        }

    @property
    def fy_fraction(self) -> float:
        return 1.0 - self.my_fraction

    def sample_fsd(self, n_floes: int, seed: int = 42) -> np.ndarray:
        """Sample floe sizes from the bimodal FSD.

        Returns array of floe diameters [m].
        Uses inverse transform sampling for truncated power-law:
        p(D) ∝ D^exp for D in [D_min, D_max]
        """
        rng = np.random.default_rng(seed)
        n_my = int(round(n_floes * self.my_fraction))
        n_fy = n_floes - n_my

        def _sample_powerlaw(n: int, d_char: float, exp: float) -> np.ndarray:
            """Sample from power-law p(D) ∝ D^exp on [d_char*0.1, d_char*3]."""
            if n == 0:
                return np.array([])
            d_min = d_char * 0.1
            d_max = d_char * 3.0
            # CDF: F(D) = (D^(exp+1) - d_min^(exp+1)) / (d_max^(exp+1) - d_min^(exp+1))
            # Inverse: D = [u * (d_max^(exp+1) - d_min^(exp+1)) + d_min^(exp+1)]^(1/(exp+1))
            u = rng.uniform(0.0, 1.0, size=n)
            exp1 = exp + 1.0
            d_min_p = d_min ** exp1
            d_max_p = d_max ** exp1
            return ((u * (d_max_p - d_min_p) + d_min_p)) ** (1.0 / exp1)

        fy_sizes = _sample_powerlaw(n_fy, self.fy_char_floe_size_m, self.fsd_exponent)
        my_sizes = _sample_powerlaw(n_my, self.my_char_floe_size_m, self.fsd_exponent)

        return np.concatenate([fy_sizes, my_sizes])

    def to_ice_state(self) -> IceState:
        """Convert to a single IceState with area-weighted properties."""
        eff = self.effective_strength()
        return IceState(
            thickness_m=self.thickness_m,
            compressive_strength_pa=eff["compressive_strength_pa"],
            flexural_strength_pa=eff["flexural_strength_pa"],
            elastic_modulus_pa=eff["elastic_modulus_pa"],
            density_kg_m3=ICE_DENSITY,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "my_fraction": self.my_fraction,
            "thickness_m": self.thickness_m,
            "fy_properties": self.fy_properties.as_dict(),
            "my_properties": self.my_properties.as_dict(),
            "fy_char_floe_size_m": self.fy_char_floe_size_m,
            "my_char_floe_size_m": self.my_char_floe_size_m,
            "fsd_exponent": self.fsd_exponent,
            "effective_strength": self.effective_strength(),
        }


# ---------------------------------------------------------------------------
# Lindqvist MY-extended model
# ---------------------------------------------------------------------------

def lindqvist_my_components(
    ice: IceState,
    hull: HullForm,
    speed_m_s: float,
    *,
    ice_age: str = "FY",
    my_fraction: float = 0.0,
    water_density_kg_m3: float = WATER_DENSITY,
    gravity_m_s2: float = GRAVITY,
) -> dict[str, Any]:
    """Lindqvist resistance with MY ice correction.

    For pure FY ice (ice_age="FY", my_fraction=0), this reduces to the
    standard three-component Lindqvist model.

    For MY or mixed ice, the rotation (bending) and clearing (submersion)
    terms are scaled by k_MY(σ_c, h_i).

    Parameters
    ----------
    ice : IceState
        Ice state (thickness and properties). For "FY" and "MY" modes,
        the ice state's actual properties are used. For "mixed" mode,
        the effective properties are computed from FY/MY defaults weighted
        by my_fraction.
    hull : HullForm
        Hull geometry.
    speed_m_s : float
        Ship speed [m/s].
    ice_age : str
        "FY" for first-year, "MY" for multi-year, "mixed" for mixed field.
    my_fraction : float
        Fraction of MY ice by area (0..1). Only used when ice_age="mixed".
    """
    speed = _finite("speed_m_s", speed_m_s, nonnegative=True)
    _finite("my_fraction", my_fraction, nonnegative=True)
    if my_fraction > 1.0:
        raise ValueError("my_fraction must be <= 1.0")

    # Base components (using the ice state's properties)
    r_c = crushing_resistance(ice, hull)
    r_b = bending_resistance(ice, hull)
    r_s = submersion_resistance(ice, hull, water_density_kg_m3, gravity_m_s2)
    r_v = added_drag(ice, hull, speed, water_density_kg_m3)

    # Determine effective strength for k_MY calculation
    if ice_age == "MY":
        # Use the ice state's actual properties
        effective_sigma_c = ice.compressive_strength_pa
        eff_my_frac = 1.0
    elif ice_age == "mixed":
        # Weighted average of FY and MY defaults
        fy_sigma = FY_DEFAULT["compressive_strength_pa"]
        my_sigma = MY_DEFAULT["compressive_strength_pa"]
        effective_sigma_c = (1.0 - my_fraction) * fy_sigma + my_fraction * my_sigma
        eff_my_frac = my_fraction
    else:  # FY
        effective_sigma_c = ice.compressive_strength_pa
        eff_my_frac = 0.0

    # MY correction factor (applies to rotation/bending and clearing/submersion)
    k_my = my_correction_factor(effective_sigma_c, ice.thickness_m)

    # For mixed fields, interpolate between FY (k=1) and MY (k=k_my)
    if ice_age == "mixed":
        k_eff = 1.0 + eff_my_frac * (k_my - 1.0)
    elif ice_age == "MY":
        k_eff = k_my
    else:  # FY
        k_eff = 1.0

    # Apply correction to bending (rotation) and submersion (clearing) terms
    r_b_corrected = r_b * k_eff
    r_s_corrected = r_s * k_eff

    total = r_c + r_b_corrected + r_s_corrected + r_v

    return {
        "model": "lindqvist-my-extended",
        "ice_age": ice_age,
        "my_fraction": my_fraction if ice_age == "mixed" else (1.0 if ice_age == "MY" else 0.0),
        "hull": hull.as_dict(),
        "ice": ice.as_dict(),
        "speed_m_s": speed,
        "k_MY": k_my,
        "k_effective": k_eff,
        "crushing_N": r_c,
        "bending_N": r_b,
        "bending_corrected_N": r_b_corrected,
        "submersion_N": r_s,
        "submersion_corrected_N": r_s_corrected,
        "added_drag_N": r_v,
        "static_N": r_c + r_b_corrected + r_s_corrected,
        "total_N": total,
        "crushing_fraction": r_c / total if total > 0 else 0.0,
        "bending_fraction": r_b_corrected / total if total > 0 else 0.0,
        "submersion_fraction": r_s_corrected / total if total > 0 else 0.0,
        "drag_fraction": r_v / total if total > 0 else 0.0,
    }


def lindqvist_my_sweep(
    hull: HullForm,
    *,
    my_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    thicknesses_m: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    speeds_kn: Sequence[float] = (2.0, 4.0, 6.0, 8.0, 10.0),
) -> list[dict[str, Any]]:
    """Parameter sweep over MY fraction, ice thickness, and ship speed.

    Returns a list of result dicts, one per combination.
    """
    results = []
    for my_frac in my_fractions:
        for h in thicknesses_m:
            for v_kn in speeds_kn:
                v_ms = v_kn * 0.514444  # knots to m/s

                # Build mixed ice field with appropriate properties
                mixed = MixedIceField(
                    my_fraction=my_frac,
                    thickness_m=h,
                    fy_properties=IceAgeProperties.fy(),
                    my_properties=IceAgeProperties.my(),
                )
                ice_state = mixed.to_ice_state()

                result = lindqvist_my_components(
                    ice_state, hull, v_ms,
                    ice_age="mixed", my_fraction=my_frac,
                )
                result["sweep_params"] = {
                    "my_fraction": my_frac,
                    "thickness_m": h,
                    "speed_kn": v_kn,
                    "speed_m_s": v_ms,
                }
                results.append(result)
    return results


# ---------------------------------------------------------------------------
# MY coefficient ledger (for provenance tracking)
# ---------------------------------------------------------------------------

MY_COEFFICIENT_LEDGER: dict[str, tuple[float, str, str]] = {
    "my_correction_alpha": (
        MY_CORRECTION_ALPHA,
        "DERIVED",
        "Sub-linear strength exponent from fracture mechanics: "
        "crack propagation energy scales as K_IC^2 ~ σ_c · a, giving "
        "effective exponent ~0.5-0.7 for sea ice.",
    ),
    "my_correction_beta": (
        MY_CORRECTION_BETA,
        "CONVENTION",
        "Thickness exponent for floe size scaling: MY ice forms larger "
        "floe sizes due to rafting/ridging history, exponent ~0.3-0.5.",
    ),
    "my_reference_thickness_m": (
        MY_REFERENCE_THICKNESS,
        "CONVENTION",
        "Reference thickness for k_MY normalisation (1 m, typical FY).",
    ),
    "fy_sigma_c_default_pa": (
        FY_DEFAULT["compressive_strength_pa"],
        "CONVENTION",
        "Default FY compressive strength (mid-range of 1-3 MPa).",
    ),
    "my_sigma_c_default_pa": (
        MY_DEFAULT["compressive_strength_pa"],
        "CONVENTION",
        "Default MY compressive strength (mid-range of 5-10 MPa).",
    ),
}


def my_ledger_sha256() -> str:
    """SHA-256 over the MY coefficient ledger."""
    payload = {
        name: [value, grade, text]
        for name, (value, grade, text) in sorted(MY_COEFFICIENT_LEDGER.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
