"""Age -> sea-ice material state: the multi-year (MY) parameterisation.

Physics
-------
Multi-year ice differs from first-year (FY) ice in three coupled ways, and the
parameterisation below keeps them separate so each can be audited:

1. **Desalination.**  Bulk salinity drops from a first-year 4-6 permille to a
   1-2 permille asymptote as brine drains through the summer.  Modelled as an
   exponential decay in age with time constant ``DESALINATION_TIME_A``.
2. **Thermal state.**  Thicker, older ice has a colder mean temperature (the
   winter cold wave penetrates a thicker slab, and the summer melt does not
   reach its core).  Modelled as an exponential approach to
   ``MY_TEMPERATURE_ASYMPTOTE_C``.
3. **Thickness.**  Survival through one melt season, then growth governed by
   the same conductive limit, gives an asymptotic equilibrium thickness
   (observed MY ice is 2-4 m; ``MY_THICKNESS_ASYMPTOTE_M = 3.2 m``).  Modelled
   as an exponential approach from ``FY_THICKNESS_M``.

Point 1 and 2 together determine the **brine volume fraction** through the
classical freezing-point relation

    v_b = S * (1 + 49.185 / |T|) / 1000,

with ``S`` the bulk salinity in permille and ``T`` in degrees Celsius, which is
the gas-free form of Frankenstein & Garner (1967) as used by Cox & Weeks (1983).
Brine volume is the single state variable that carries the mechanical
consequences: liquid brine pockets do not carry load, so v_b sets pore-induced
degradation of strength, stiffness and toughness.

The three degradation laws, all written as functions of ``v_b`` only:

**Flexural strength** (the one closing relation taken from the literature)

    sigma_f = 1.76 MPa * exp(-5.88 * sqrt(v_b)),

Timco & O'Brien (1994) as reviewed by Timco & Weeks (2010), valid for
``v_b <= 0.27``, ``S = 0-10`` permille, ``T = -2 .. -30`` degC, strain rate near
1e-3 1/s.  Its limits are machine-checked: ``v_b -> 0`` returns 1.76 MPa (the
near-pure-ice flexural strength) and ``v_b = 0.27`` returns 0.08 MPa.

**Compressive strength and elastic modulus** use one pinned percolation form

    X(v_b) = X_solid * (1 - v_b / v_p) ** n_X,

with ``v_p = 0.28`` (pinned) the brine volume at which the load-bearing ice
platelet network is taken to lose continuity.  ``X_solid`` is *calibrated* --
not fitted -- so that the FY reference state returns the values this repository
already uses for first-year ice (``IceState.compressive_strength_pa = 1.5 MPa``,
``SheetConfig.elastic_modulus_pa = 5.0 GPa``).  That makes the MY extension
exactly FY-consistent at age 1 by construction, and makes the MY/FY *ratio*
the only new content.  Exponents ``n_sigma = 2`` (brittle strength scales with
the square of the solid fraction near a percolation threshold) and ``n_E = 1``
(dilute-pore modulus depression) are pinned.

**Fracture toughness** uses the same form with the same ``n = 1`` exponent and
the near-pure-ice value 0.12 MPa m^(1/2).  It is parameterised for interface
completeness only: the bonded-particle sheet of
:mod:`tensordem.level_ice.sheet` uses strength-only bonds (no fracture energy),
so ``K_Ic`` has **zero influence** on the resistance numbers of this campaign.
That is stated in the artifact as ``consumed_by_dem: false`` instead of being
hidden.

Evidence grades are in :data:`COEFFICIENT_LEDGER`; ``literature_agreement`` is
``WITHHELD`` because the primary sources are not readable offline and nothing
here is calibrated against measured MY-ice resistance.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "COEFFICIENT_LEDGER",
    "IceAgeState",
    "aging_ledger_sha256",
    "brine_volume_fraction",
    "compressive_strength_pa",
    "density_kg_m3",
    "elastic_modulus_pa",
    "first_year_state",
    "flexural_strength_pa",
    "fracture_toughness_pa_sqrt_m",
    "material_state",
    "material_state_arrays",
]

# ---------------------------------------------------------------------------
# Pinned field / material constants
# ---------------------------------------------------------------------------
#: first-year bulk salinity [permille] (published FY range 4-6 permille).
FY_SALINITY_PERMILLE = 5.0
#: desalinated asymptote [permille] (published MY range 1-2 permille).
MY_SALINITY_ASYMPTOTE_PERMILLE = 1.5
#: desalination time constant [years].
DESALINATION_TIME_A = 0.7
#: first-year mean ice temperature [degC].
FY_TEMPERATURE_C = -5.0
#: multi-year mean ice temperature asymptote [degC].
MY_TEMPERATURE_ASYMPTOTE_C = -15.0
#: thermal relaxation time constant [years].
TEMPERATURE_TIME_A = 1.5
#: first-year mean level-ice thickness [m].
FY_THICKNESS_M = 1.2
#: multi-year equilibrium thickness [m] (observed MY ice 2-4 m).
MY_THICKNESS_ASYMPTOTE_M = 3.2
#: thickness relaxation time constant [years].
THICKNESS_TIME_A = 1.5

#: percolation brine volume at which the ice network is taken to disconnect.
PERCOLATION_BRINE_VOLUME = 0.28
#: exponent of the compressive-strength degradation law.
COMPRESSIVE_EXPONENT = 2.0
#: exponent of the temperature amplification of the ice-matrix compressive
#: strength, ``sigma_c ~ (|T| / |T_FY|)**m`` (see the coefficient ledger).
COMPRESSIVE_TEMPERATURE_EXPONENT = 0.35
#: sensitivity band swept for the temperature exponent.
COMPRESSIVE_TEMPERATURE_EXPONENT_BAND = (0.20, 0.50)
#: exponent of the elastic-modulus degradation law.
MODULUS_EXPONENT = 1.0
#: exponent of the fracture-toughness degradation law.
TOUGHNESS_EXPONENT = 1.0
#: near-pure-ice fracture toughness [Pa m^0.5] (literature 0.05-0.15).
TOUGHNESS_SOLID_PA_SQRT_M = 0.12e6
#: Timco & O'Brien flexural-strength prefix [MPa].
FLEXURAL_PREFIX_MPA = 1.76
#: Timco & O'Brien flexural-strength decay coefficient.
FLEXURAL_DECAY = 5.88
#: temperature used for the pure-ice modulus reference [degC] (documentation).
PURE_ICE_MODULUS_PA = 9.7e9
#: pure ice density [kg/m^3].
PURE_ICE_DENSITY_KG_M3 = 917.0
#: brine density base [kg/m^3] and salinity slope [kg/m^3 per permille].
BRINE_DENSITY_BASE_KG_M3 = 1000.0
BRINE_DENSITY_SLOPE = 0.8

#: Repository first-year reference values that the calibration reproduces.
FY_COMPRESSIVE_REFERENCE_PA = 1.5e6
FY_MODULUS_REFERENCE_PA = 5.0e9

COEFFICIENT_LEDGER: dict[str, tuple[float, str, str]] = {
    "brine_volume_law": (
        49.185,
        "CITED",
        "v_b = S (1 + 49.185/|T|)/1000, gas-free Frankenstein & Garner (1967) "
        "form as used by Cox & Weeks (1983).  Not re-readable offline; the "
        "45-degree freezing-point slope is the only coefficient.",
    ),
    "flexural_strength_prefix_mpa": (
        FLEXURAL_PREFIX_MPA,
        "CITED",
        "Timco & O'Brien (1994) / Timco & Weeks (2010) review: "
        "sigma_f = 1.76 exp(-5.88 sqrt(v_b)) MPa for v_b <= 0.27.",
    ),
    "flexural_strength_decay": (
        FLEXURAL_DECAY,
        "CITED",
        "Same relation as above; sqrt(v_b) porosity argument.",
    ),
    "percolation_brine_volume": (
        PERCOLATION_BRINE_VOLUME,
        "CONVENTION",
        "Pinned brine volume at which the ice platelet network is taken to "
        "lose continuity; the strength/modulus/toughness laws vanish there. "
        "Sensitivity band 0.24-0.32 is swept by the campaign.",
    ),
    "compressive_exponent": (
        COMPRESSIVE_EXPONENT,
        "CONVENTION",
        "sigma_c ~ (1 - v_b/v_p)^2, the brittle-strength scaling near a "
        "percolation threshold.  Calibrated so the FY reference state returns "
        "the repository value 1.5 MPa, hence FY-consistent by construction.",
    ),
    "compressive_temperature_exponent": (
        COMPRESSIVE_TEMPERATURE_EXPONENT,
        "CONVENTION",
        "sigma_c ~ (|T|/|T_FY|)**m with m in the swept band 0.20-0.50; the "
        "literature trend (brittle compressive strength rising as sea ice "
        "cools, Schulson & Duval 2009 ch. 7) fixes the sign and order of "
        "magnitude, not the exact exponent.  Normalised at the FY reference "
        "temperature, so age 1 is unchanged for any m.",
    ),
    "modulus_exponent": (
        MODULUS_EXPONENT,
        "CONVENTION",
        "E ~ (1 - v_b/v_p), linear dilute-pore depression; calibrated so the "
        "FY reference state returns SheetConfig.elastic_modulus_pa = 5.0 GPa.",
    ),
    "toughness_exponent": (
        TOUGHNESS_EXPONENT,
        "CONVENTION",
        "K_Ic ~ (1 - v_b/v_p) with a 0.12 MPa m^0.5 near-pure-ice value "
        "(literature 0.05-0.15).  NOT consumed by the current DEM (strength-"
        "only bonds): interface contract only.",
    ),
    "desalination_time_a": (
        DESALINATION_TIME_A,
        "CONVENTION",
        "Exponential desalination, time constant 0.7 a; endpoints anchored on "
        "the published FY (4-6 permille) and MY (1-2 permille) ranges.",
    ),
    "temperature_time_a": (
        TEMPERATURE_TIME_A,
        "CONVENTION",
        "Exponential approach from -5 degC (FY mean) to -15 degC (MY mean), "
        "time constant 1.5 a; both endpoints are published means.",
    ),
    "thickness_time_a": (
        THICKNESS_TIME_A,
        "CONVENTION",
        "Exponential approach from 1.2 m (FY mean) to a 3.2 m equilibrium "
        "(observed MY ice is 2-4 m); time constant 1.5 a.",
    ),
    "literature_agreement": (
        float("nan"),
        "WITHHELD",
        "No measured MY-ice resistance series is available offline, so nothing "
        "here is validated against data at the absolute level.  The MY/FY "
        "*ratio* and its sensitivity are the only claims made.",
    ),
}


def aging_ledger_sha256() -> str:
    """SHA-256 over the graded coefficient ledger (stable JSON dump)."""
    payload = {
        name: [value if math.isfinite(value) else "nan", grade, text]
        for name, (value, grade, text) in sorted(COEFFICIENT_LEDGER.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# State variable
# ---------------------------------------------------------------------------
def _finite(name: str, value: Any, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _as_float_array(name: str, value: Any, *, positive: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    return arr


def brine_volume_fraction(salinity_permille: Any, temperature_c: Any) -> np.ndarray | float:
    """Gas-free brine volume fraction ``v_b`` (dimensionless).

    ``v_b = S (1 + 49.185/|T|) / 1000`` with ``S`` in permille and ``T`` in
    degC (< 0).  Returns a float for scalar input, an array otherwise.
    """
    s = _as_float_array("salinity_permille", salinity_permille, positive=False)
    t = np.asarray(temperature_c, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("temperature_c must be finite")
    if np.any(t >= 0.0):
        raise ValueError("temperature_c must be below the freezing point (< 0)")
    if np.any(s < 0.0):
        raise ValueError("salinity_permille must be nonnegative")
    v_b = s * (1.0 + 49.185 / np.abs(t)) / 1000.0
    if np.any(v_b >= PERCOLATION_BRINE_VOLUME):
        raise ValueError(
            "brine volume %.4f reaches the percolation threshold %.3f"
            % (float(np.max(v_b)), PERCOLATION_BRINE_VOLUME)
        )
    out = v_b
    return float(out) if out.ndim == 0 else out


def flexural_strength_pa(v_b: Any) -> np.ndarray | float:
    """Timco & O'Brien flexural strength [Pa] as a function of brine volume."""
    v = _as_float_array("brine_volume_fraction", v_b, positive=False)
    if np.any(v > 0.27):
        raise ValueError("flexural relation is defined for v_b <= 0.27")
    out = FLEXURAL_PREFIX_MPA * 1.0e6 * np.exp(-FLEXURAL_DECAY * np.sqrt(v))
    return float(out) if out.ndim == 0 else out


def _solid_calibration(reference_pa: float, exponent: float,
                       v_b_ref: float) -> float:
    return reference_pa / (1.0 - v_b_ref / PERCOLATION_BRINE_VOLUME) ** exponent


def compressive_strength_pa(
    v_b: Any,
    temperature_c_value: Any | None = None,
    *,
    temperature_exponent: float = COMPRESSIVE_TEMPERATURE_EXPONENT,
) -> np.ndarray | float:
    """Compressive strength [Pa]: percolation law times temperature amplification.

    ``sigma_c = sigma_solid (|T|/|T_FY|)**m (1 - v_b/v_p)**2``

    The percolation factor alone reproduces the repository first-year anchor
    (1.5 MPa at age 1) and grows only by the brine-volume loss on desalination,
    which yields an MY/FY ratio of ~1.47 -- below the observed band.  The
    amplification factor carries the *matrix* (pure-ice) part of the temperature
    dependence that the brine-volume argument cannot supply: the ice matrix is
    both colder and much less saline in MY ice, and brittle compressive strength
    rises as the temperature falls.  The factor is exactly 1.0 at the FY
    reference temperature, so the first-year degeneration is exact for any
    ``m``, and it saturates because the mean temperature saturates.

    ``temperature_c_value=None`` reproduces the pre-extension (porosity-only)
    law, which is kept for backward compatibility of stored artifacts.
    """
    v = _as_float_array("brine_volume_fraction", v_b, positive=False)
    _guard_porosity(v)
    ref = brine_volume_fraction(FY_SALINITY_PERMILLE, FY_TEMPERATURE_C)
    solid = _solid_calibration(FY_COMPRESSIVE_REFERENCE_PA, COMPRESSIVE_EXPONENT, ref)
    out = solid * (1.0 - v / PERCOLATION_BRINE_VOLUME) ** COMPRESSIVE_EXPONENT
    if temperature_c_value is not None:
        m = _finite("temperature_exponent", temperature_exponent)
        if m < 0.0:
            raise ValueError("temperature_exponent must be nonnegative")
        t = np.abs(_as_float_array("temperature_c", temperature_c_value,
                                  positive=False))
        if np.any(t <= 0.0):
            raise ValueError("temperature must be strictly below 0 degC")
        out = out * (t / abs(FY_TEMPERATURE_C)) ** m
    return float(out) if np.ndim(out) == 0 else out


def elastic_modulus_pa(v_b: Any) -> np.ndarray | float:
    """Elastic modulus [Pa] from the pinned percolation law."""
    v = _as_float_array("brine_volume_fraction", v_b, positive=False)
    _guard_porosity(v)
    ref = brine_volume_fraction(FY_SALINITY_PERMILLE, FY_TEMPERATURE_C)
    solid = _solid_calibration(FY_MODULUS_REFERENCE_PA, MODULUS_EXPONENT, ref)
    out = solid * (1.0 - v / PERCOLATION_BRINE_VOLUME) ** MODULUS_EXPONENT
    return float(out) if out.ndim == 0 else out


def fracture_toughness_pa_sqrt_m(v_b: Any) -> np.ndarray | float:
    """Fracture toughness [Pa m^0.5] from the pinned percolation law."""
    v = _as_float_array("brine_volume_fraction", v_b, positive=False)
    _guard_porosity(v)
    out = TOUGHNESS_SOLID_PA_SQRT_M * (1.0 - v / PERCOLATION_BRINE_VOLUME) ** TOUGHNESS_EXPONENT
    return float(out) if out.ndim == 0 else out


def density_kg_m3(v_b: Any, salinity_permille: Any) -> np.ndarray | float:
    """Gas-free ice density [kg/m^3]: pure ice plus brine, no air."""
    v = _as_float_array("brine_volume_fraction", v_b, positive=False)
    s = _as_float_array("salinity_permille", salinity_permille, positive=False)
    rho_brine = BRINE_DENSITY_BASE_KG_M3 + BRINE_DENSITY_SLOPE * s
    out = PURE_ICE_DENSITY_KG_M3 * (1.0 - v) + rho_brine * v
    return float(out) if out.ndim == 0 else out


def _guard_porosity(v: np.ndarray) -> None:
    if np.any(v >= PERCOLATION_BRINE_VOLUME):
        raise ValueError("brine volume at/above the percolation threshold")


# ---------------------------------------------------------------------------
# Ageing trajectory
# ---------------------------------------------------------------------------
def salinity_permille(age_years: Any) -> np.ndarray | float:
    """Bulk salinity [permille] versus age [years]."""
    a = _as_float_array("age_years", age_years)
    if np.any(a < 1.0):
        raise ValueError("age_years must be >= 1 (first-year ice is 1 year old)")
    out = MY_SALINITY_ASYMPTOTE_PERMILLE + (
        FY_SALINITY_PERMILLE - MY_SALINITY_ASYMPTOTE_PERMILLE
    ) * np.exp(-(a - 1.0) / DESALINATION_TIME_A)
    return float(out) if out.ndim == 0 else out


def temperature_c(age_years: Any) -> np.ndarray | float:
    """Mean ice temperature [degC] versus age [years]."""
    a = _as_float_array("age_years", age_years)
    if np.any(a < 1.0):
        raise ValueError("age_years must be >= 1")
    out = FY_TEMPERATURE_C - (FY_TEMPERATURE_C - MY_TEMPERATURE_ASYMPTOTE_C) * (
        1.0 - np.exp(-(a - 1.0) / TEMPERATURE_TIME_A)
    )
    return float(out) if out.ndim == 0 else out


def class_thickness_m(age_years: Any) -> np.ndarray | float:
    """Mean ice thickness [m] of an age class."""
    a = _as_float_array("age_years", age_years)
    if np.any(a < 1.0):
        raise ValueError("age_years must be >= 1")
    out = MY_THICKNESS_ASYMPTOTE_M - (MY_THICKNESS_ASYMPTOTE_M - FY_THICKNESS_M) * np.exp(
        -(a - 1.0) / THICKNESS_TIME_A
    )
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class IceAgeState:
    """Material state of one age class (SI)."""

    age_years: float
    ice_class: str
    salinity_permille: float
    temperature_c: float
    thickness_m: float
    brine_volume_fraction: float
    compressive_strength_pa: float
    flexural_strength_pa: float
    elastic_modulus_pa: float
    fracture_toughness_pa_sqrt_m: float
    density_kg_m3: float

    def __post_init__(self) -> None:
        _finite("age_years", self.age_years, positive=True)
        if self.ice_class not in ("FY", "MY"):
            raise ValueError("ice_class must be 'FY' or 'MY'")
        _finite("thickness_m", self.thickness_m, positive=True)
        _finite("compressive_strength_pa", self.compressive_strength_pa, positive=True)
        _finite("flexural_strength_pa", self.flexural_strength_pa, positive=True)
        _finite("elastic_modulus_pa", self.elastic_modulus_pa, positive=True)
        _finite("fracture_toughness_pa_sqrt_m", self.fracture_toughness_pa_sqrt_m, positive=True)
        _finite("brine_volume_fraction", self.brine_volume_fraction, nonnegative=True)
        _finite("salinity_permille", self.salinity_permille, nonnegative=True)
        _finite("temperature_c", self.temperature_c, positive=False)

    # -- interfaces -------------------------------------------------------
    def as_ice_state(self, thickness_m: float | None = None):
        """Bridge to :class:`tensordem.level_ice.formula.IceState`."""
        from tensordem.level_ice.formula import IceState

        return IceState(
            thickness_m=self.thickness_m if thickness_m is None else thickness_m,
            compressive_strength_pa=self.compressive_strength_pa,
            flexural_strength_pa=self.flexural_strength_pa,
            elastic_modulus_pa=self.elastic_modulus_pa,
            density_kg_m3=self.density_kg_m3,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "age_years": self.age_years,
            "ice_class": self.ice_class,
            "salinity_permille": self.salinity_permille,
            "temperature_c": self.temperature_c,
            "thickness_m": self.thickness_m,
            "brine_volume_fraction": self.brine_volume_fraction,
            "compressive_strength_pa": self.compressive_strength_pa,
            "flexural_strength_pa": self.flexural_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "fracture_toughness_pa_sqrt_m": self.fracture_toughness_pa_sqrt_m,
            "density_kg_m3": self.density_kg_m3,
            "consumed_by_dem": {
                "compressive_strength_pa": True,
                "flexural_strength_pa": True,
                "elastic_modulus_pa": True,
                "density_kg_m3": True,
                "fracture_toughness_pa_sqrt_m": False,
            },
            "compressive_temperature_exponent": COMPRESSIVE_TEMPERATURE_EXPONENT,
            "aging_ledger_sha256": aging_ledger_sha256(),
        }


def material_state(age_years: float) -> IceAgeState:
    """Material state of an ice of the given age [years]."""
    age = _finite("age_years", age_years)
    if age < 1.0:
        raise ValueError("age_years must be >= 1")
    s = salinity_permille(age)
    t = temperature_c(age)
    v_b = brine_volume_fraction(s, t)
    return IceAgeState(
        age_years=age,
        ice_class="FY" if age <= 1.0 else "MY",
        salinity_permille=s,
        temperature_c=t,
        thickness_m=class_thickness_m(age),
        brine_volume_fraction=v_b,
        compressive_strength_pa=compressive_strength_pa(v_b, t),
        flexural_strength_pa=flexural_strength_pa(v_b),
        elastic_modulus_pa=elastic_modulus_pa(v_b),
        fracture_toughness_pa_sqrt_m=fracture_toughness_pa_sqrt_m(v_b),
        density_kg_m3=density_kg_m3(v_b, s),
    )


def material_state_arrays(ages: Any) -> dict[str, np.ndarray]:
    """Vectorised material state for an age array (used by the FSD sampler)."""
    a = np.asarray(ages, dtype=float)
    if np.any(~np.isfinite(a)) or np.any(a < 1.0):
        raise ValueError("ages must be finite and >= 1")
    s = salinity_permille(a)
    t = temperature_c(a)
    v_b = brine_volume_fraction(s, t)
    return {
        "age_years": a,
        "salinity_permille": np.asarray(s, dtype=float),
        "temperature_c": np.asarray(t, dtype=float),
        "thickness_m": np.asarray(class_thickness_m(a), dtype=float),
        "brine_volume_fraction": np.asarray(v_b, dtype=float),
        "compressive_strength_pa": np.asarray(
            compressive_strength_pa(v_b, t), dtype=float
        ),
        "flexural_strength_pa": np.asarray(flexural_strength_pa(v_b), dtype=float),
        "elastic_modulus_pa": np.asarray(elastic_modulus_pa(v_b), dtype=float),
        "fracture_toughness_pa_sqrt_m": np.asarray(
            fracture_toughness_pa_sqrt_m(v_b), dtype=float
        ),
        "density_kg_m3": np.asarray(density_kg_m3(v_b, s), dtype=float),
    }


def first_year_state() -> IceAgeState:
    """The first-year reference state (age 1 year)."""
    return material_state(1.0)