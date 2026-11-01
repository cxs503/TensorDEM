"""Level-ice resistance baselines used as reference gates for the DEM coupling.

The module implements two independent engineering baselines for the mean
resistance of a ship advancing through level ice:

* ``lindqvist_resistance`` -- three-component (crushing / bending / submersion)
  decomposition in the spirit of Lindqvist (1989).
* ``riska_resistance`` -- speed-linear model ``R = C1 + C2 V`` in the spirit of
  Riska (1989), with the C1/C2 terms assembled from the same physical
  components.

Provenance caveat (kept explicit on purpose, see
``docs/threefield/FORMULAS.md``): the component *structure* follows the
published models, but the primary papers were not reachable from the offline
sandbox used to build this module, so the coefficient set is the one written
out in the documentation and guarded by the structural self-tests in
``tests/test_threefield_resistance.py``.  The two baselines are therefore used
*as a band* (``resistance_band``) rather than as individually authoritative
numbers, and any DEM-versus-formula error reported by this repository is
reported against both baselines separately as well as against the band centre.

All quantities are SI: m, Pa, kg/m^3, m/s, N.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

GRAVITY = 9.80665
WATER_DENSITY = 1025.0
ICE_DENSITY = 917.0


def _finite(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class IceProperties:
    """Level-ice material state (all SI)."""

    thickness_m: float = 0.60
    compressive_strength_pa: float = 1.5e6
    flexural_strength_pa: float = 0.5e6
    elastic_modulus_pa: float = 5.0e9
    density_kg_m3: float = ICE_DENSITY

    def __post_init__(self) -> None:
        _finite("thickness_m", self.thickness_m, positive=True)
        _finite("compressive_strength_pa", self.compressive_strength_pa, positive=True)
        _finite("flexural_strength_pa", self.flexural_strength_pa, positive=True)
        _finite("elastic_modulus_pa", self.elastic_modulus_pa, positive=True)
        _finite("density_kg_m3", self.density_kg_m3, positive=True)

    def as_dict(self) -> dict[str, float]:
        return {
            "thickness_m": self.thickness_m,
            "compressive_strength_pa": self.compressive_strength_pa,
            "flexural_strength_pa": self.flexural_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "density_kg_m3": self.density_kg_m3,
        }


@dataclass(frozen=True)
class HullGeometry:
    """Hull particulars used by the engineering baselines (all SI / deg)."""

    length_m: float = 120.0
    beam_m: float = 20.0
    draught_m: float = 7.0
    waterline_entry_angle_deg: float = 22.0
    stem_angle_deg: float = 30.0
    buttock_angle_deg: float = 20.0
    friction_coefficient: float = 0.05

    def __post_init__(self) -> None:
        _finite("length_m", self.length_m, positive=True)
        _finite("beam_m", self.beam_m, positive=True)
        _finite("draught_m", self.draught_m, positive=True)
        for name in ("waterline_entry_angle_deg", "stem_angle_deg", "buttock_angle_deg"):
            value = _finite(name, getattr(self, name), positive=True)
            if value >= 90.0:
                raise ValueError(f"{name} must be < 90 degrees")
        _finite("friction_coefficient", self.friction_coefficient, nonnegative=True)

    def as_dict(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "beam_m": self.beam_m,
            "draught_m": self.draught_m,
            "waterline_entry_angle_deg": self.waterline_entry_angle_deg,
            "stem_angle_deg": self.stem_angle_deg,
            "buttock_angle_deg": self.buttock_angle_deg,
            "friction_coefficient": self.friction_coefficient,
        }


@dataclass(frozen=True)
class Environment:
    """Ambient water properties."""

    water_density_kg_m3: float = WATER_DENSITY
    gravity_m_s2: float = GRAVITY

    def __post_init__(self) -> None:
        _finite("water_density_kg_m3", self.water_density_kg_m3, positive=True)
        _finite("gravity_m_s2", self.gravity_m_s2, positive=True)


#: Dimensionless coefficient of the buoyant-depression (channel opening) term of
#: the submersion resistance: the broken channel floats the displaced ice volume
#: ``c_b * B * h^2`` at the hull waterline.
SUBMERSION_BUOYANCY_COEFFICIENT = 0.9
#: Friction-speed correction of the submersion term, ``mu * c_f * L / B``.
SUBMERSION_FRICTION_COEFFICIENT = 0.05
#: Bending-efficiency coefficient of the Lindqvist-style breaking term.
BENDING_EFFICIENCY = 0.25
#: Crushing-efficiency coefficient (contact area of the crushed wedge).
CRUSHING_EFFICIENCY = 0.5


def _entry_terms(hull: HullGeometry) -> tuple[float, float, float]:
    alpha = math.radians(hull.waterline_entry_angle_deg)
    phi = math.radians(hull.stem_angle_deg)
    psi = math.radians(hull.buttock_angle_deg)
    tan_alpha = math.tan(alpha)
    mu = hull.friction_coefficient
    if tan_alpha <= 0.0:
        raise ValueError("waterline entry angle produces a degenerate tangent")
    return tan_alpha, mu / max(math.cos(alpha), 1e-9), mu / max(math.sin(psi), 1e-9) + math.cos(phi)


def crushing_resistance(ice: IceProperties, hull: HullGeometry) -> float:
    """Crushing term, per unit width, integrated over the bow contact height."""
    tan_alpha, mu_over_cos, _ = _entry_terms(hull)
    return (
        CRUSHING_EFFICIENCY
        * ice.compressive_strength_pa
        * ice.thickness_m**2
        * (tan_alpha + mu_over_cos)
    )


def bending_resistance(ice: IceProperties, hull: HullGeometry) -> float:
    """Bending (flexural failure) term of the Lindqvist-style decomposition."""
    return (
        BENDING_EFFICIENCY
        * ice.flexural_strength_pa
        * hull.beam_m
        * ice.thickness_m**2
        / hull.length_m
    )


def submersion_resistance(
    ice: IceProperties, hull: HullGeometry, environment: Environment = Environment()
) -> float:
    """Submersion term: buoyant depression plus hull friction over the channel."""
    rho = environment.water_density_kg_m3
    g = environment.gravity_m_s2
    h = ice.thickness_m
    mu = hull.friction_coefficient
    buoyant = SUBMERSION_BUOYANCY_COEFFICIENT
    frictional = SUBMERSION_FRICTION_COEFFICIENT * mu * hull.length_m / hull.beam_m
    return rho * g * h * h * hull.beam_m * (buoyant + frictional)


def lindqvist_resistance(
    ice: IceProperties,
    hull: HullGeometry,
    speed_m_s: float,
    environment: Environment = Environment(),
) -> dict[str, Any]:
    """Lindqvist-style three-component level-ice resistance.

    ``R = R_crush + R_bend + R_sub``. Only ``R_sub`` carries the (weak) speed
    dependence through the friction correction of the displaced ice sheet; the
    model is therefore nearly speed independent at model scale, which matches
    the behaviour of the published baselines for low ice drift speeds.
    """
    speed = _finite("speed_m_s", speed_m_s, nonnegative=True)
    r_crush = crushing_resistance(ice, hull)
    r_bend = bending_resistance(ice, hull)
    r_sub = submersion_resistance(ice, hull, environment)
    # Friction of the ice rubble sliding along the hull scales with speed
    # through the added drag of the displaced sheet; keep it explicit.
    r_speed = 0.5 * environment.water_density_kg_m3 * hull.beam_m * ice.thickness_m * speed**2
    total = r_crush + r_bend + r_sub + r_speed
    return {
        "model": "lindqvist",
        "speed_m_s": speed,
        "crushing_N": r_crush,
        "bending_N": r_bend,
        "submersion_N": r_sub,
        "speed_dependent_N": r_speed,
        "total_N": total,
    }


def riska_resistance(
    ice: IceProperties,
    hull: HullGeometry,
    speed_m_s: float,
    environment: Environment = Environment(),
) -> dict[str, Any]:
    """Riska-style speed-linear level-ice resistance ``R = C1 + C2 V``."""
    speed = _finite("speed_m_s", speed_m_s, nonnegative=True)
    phi = hull.stem_angle_deg
    r_crush = crushing_resistance(ice, hull)
    r_bend = bending_resistance(ice, hull)
    r_sub = submersion_resistance(ice, hull, environment)
    c1 = (1.0 + 0.021 * phi) * (r_crush + r_sub) + r_bend
    wave_speed = math.sqrt(environment.gravity_m_s2 * ice.thickness_m)
    c2 = (1.0 + 0.063 * phi) * (
        0.5 * hull.friction_coefficient * environment.water_density_kg_m3
        * hull.beam_m * ice.thickness_m * wave_speed
        + ice.flexural_strength_pa * hull.beam_m * ice.thickness_m**2
        / (3.0 * hull.length_m * wave_speed)
    )
    total = c1 + c2 * speed
    return {
        "model": "riska",
        "speed_m_s": speed,
        "c1_N": c1,
        "c2_N_per_m_s": c2,
        "crushing_reference_N": r_crush,
        "bending_reference_N": r_bend,
        "submersion_reference_N": r_sub,
        "total_N": total,
    }


def resistance_band(
    ice: IceProperties,
    hull: HullGeometry,
    speed_m_s: float,
    environment: Environment = Environment(),
) -> dict[str, Any]:
    """Both baselines plus the band they span (used as the reference gate)."""
    lindqvist = lindqvist_resistance(ice, hull, speed_m_s, environment)
    riska = riska_resistance(ice, hull, speed_m_s, environment)
    low = min(lindqvist["total_N"], riska["total_N"])
    high = max(lindqvist["total_N"], riska["total_N"])
    return {
        "lindqvist": lindqvist,
        "riska": riska,
        "band_low_N": low,
        "band_high_N": high,
        "band_centre_N": 0.5 * (low + high),
        "band_spread_pct": 100.0 * (high - low) / max(0.5 * (low + high), 1e-12),
    }


def relative_error_pct(predicted: float, reference: float) -> float:
    """Absolute percentage error of ``predicted`` against ``reference``."""
    reference = _finite("reference", reference, positive=True)
    predicted = _finite("predicted", predicted)
    return abs(predicted - reference) / reference * 100.0