"""Level-ice resistance of a ship, decomposed for channel/formation use.

The resistance is built from first principles rather than from a fitted
regression, because a formation study must evaluate *different* ice thicknesses
across the same hull (channel centre, ridge edge, undisturbed parent sheet).

Bending failure
    A semi-infinite ice plate on an elastic foundation (the buoyant water) fails
    at a free edge under a line load.  The classical marginal-load solution gives
    the failure load per unit edge length

        ``q_b = 4 M_f beta``,  ``M_f = sigma_f h^2 / 6``,
        ``beta = (rho_w g / (4 D))^(1/4)``,  ``D = E h^3 / (12 (1 - nu^2))``

    so ``q_b = (2/3) sigma_f h^2 beta`` (N/m).  The per-unit-width load is capped by
    the crushing limit ``sigma_c h`` of the ice edge, which is evaluated with
    ``sigma_c = 3 sigma_f`` (a standard sea-ice ratio).

Submergence
    Broken ice must be pushed down and carried under the hull:
    ``q_s = C_s (rho_w - rho_i) g h^2`` (N/m), the buoyancy of a displaced ice
    slab of thickness ``h`` and unit width, with ``C_s = O(1)``.

Friction
    Hull-ice friction acts on the resultant normal load:
    ``q_f = mu (q_b + q_s)``.

The hull is split into transverse strips, each with its own ice thickness, so a
ship following in a refrozen channel is evaluated strip by strip (channel centre,
ridge zone, parent sheet).  With a uniform ice cover the strip sum reduces to the
single-ship baseline, which keeps the formation framework's ``N = 1`` case exactly
degenerate.

Absolute level
    The law is used in two ways: (i) absolute resistance with
    :attr:`IceResistanceModel.scale_factor` calibrated on the MT Uikku level-ice
    reference cases that ship this repository
    (``benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv``), using a
    leave-out split so the reported error is out-of-sample; and (ii) relative
    quantities (interference coefficients, follower/leader ratios) that do not
    depend on the scale factor at all.  The calibration is a *scale* calibration of
    the derived law; it is not a hull-form validation.

All quantities are SI (m, N, Pa, kg/m3).  Quasi-static: the ice terms are
speed-independent, so the model does not reproduce the measured speed
dependence of level-ice resistance (reported as a limitation).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

GRAVITY = 9.80665


def _number(value: float, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class IceSheet:
    """Mechanical and thermo-physical properties of the ice cover."""

    flexural_strength: float = 5.0e5
    elastic_modulus: float = 5.0e9
    poisson_ratio: float = 0.3
    ice_density: float = 917.0
    water_density: float = 1025.0
    friction: float = 0.12
    submergence_factor: float = 1.0
    crushing_to_flexural_ratio: float = 3.0
    brash_strength_factor: float = 0.5

    def __post_init__(self) -> None:
        for name in ("flexural_strength", "elastic_modulus"):
            _number(getattr(self, name), name, positive=True)
        for name in ("ice_density", "water_density"):
            _number(getattr(self, name), name, positive=True)
        if self.water_density <= self.ice_density:
            raise ValueError("water_density must exceed ice_density")
        for name in ("poisson_ratio", "friction", "submergence_factor",
                     "crushing_to_flexural_ratio", "brash_strength_factor"):
            _number(getattr(self, name), name, nonnegative=True)
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")

    @property
    def compressive_strength(self) -> float:
        return self.crushing_to_flexural_ratio * self.flexural_strength

    def characteristic_length(self, thickness: float) -> float:
        """Plate-on-elastic-foundation parameter ``beta`` (1/m)."""
        thickness = _number(thickness, "thickness", positive=True)
        rigidity = (self.elastic_modulus * thickness ** 3
                    / (12.0 * (1.0 - self.poisson_ratio ** 2)))
        return (self.water_density * GRAVITY / (4.0 * rigidity)) ** 0.25

    def line_load(self, thickness: float, *, brash: bool = False) -> tuple[float, float, float]:
        """Breaking, submergence and friction line loads (N/m) at ``thickness``."""
        thickness = _number(thickness, "thickness", nonnegative=True)
        if thickness <= 0.0:
            return 0.0, 0.0, 0.0
        strength = self.flexural_strength * (self.brash_strength_factor if brash else 1.0)
        beta = self.characteristic_length(thickness)
        bending = (2.0 / 3.0) * strength * thickness * thickness * beta
        crushing = (self.compressive_strength
                    * (self.brash_strength_factor if brash else 1.0)) * thickness
        breaking = min(bending, crushing)
        submergence = (self.submergence_factor * (self.water_density - self.ice_density)
                       * GRAVITY * thickness * thickness)
        friction = self.friction * (breaking + submergence)
        return breaking, submergence, friction


@dataclass(frozen=True)
class IceResistanceModel:
    """Ship-level application of :class:`IceSheet` over transverse strips."""

    sheet: IceSheet = IceSheet()
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        _number(self.scale_factor, "scale_factor", positive=True)


@dataclass(frozen=True)
class IcebreakerShip:
    """Principal particulars of an ice-going ship plus strip discretisation."""

    name: str = "reference icebreaker"
    length: float = 100.0
    beam: float = 20.0
    draft: float = 8.0
    block_coefficient: float = 0.70
    contact_width_factor: float = 0.85
    strips: int = 9

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        for name in ("length", "beam", "draft"):
            _number(getattr(self, name), name, positive=True)
        for name in ("block_coefficient", "contact_width_factor"):
            _number(getattr(self, name), name, positive=True)
        if not 3.0 <= self.length / self.beam <= 12.0:
            raise ValueError("length/beam ratio must be in [3, 12] for this model")
        if not 2.0 <= self.beam / self.draft <= 6.0:
            raise ValueError("beam/draft ratio must be in [2, 6] for this model")
        if not 0.40 <= self.block_coefficient <= 0.90:
            raise ValueError("block_coefficient must be in [0.40, 0.90]")
        if not 0.3 <= self.contact_width_factor <= 1.0:
            raise ValueError("contact_width_factor must be in [0.3, 1.0]")
        if isinstance(self.strips, bool) or not isinstance(self.strips, int) or self.strips < 1:
            raise ValueError("strips must be a positive integer")

    @property
    def displacement_volume(self) -> float:
        """Displacement volume implied by ``Cb L B T`` (m^3)."""
        return self.block_coefficient * self.length * self.beam * self.draft

    @property
    def contact_width(self) -> float:
        """Transverse width of the icebreaking contact patch (m)."""
        return self.contact_width_factor * self.beam

    def strip_centres(self) -> list[float]:
        """Transverse positions of the strip centres relative to the centreline."""
        width = self.contact_width / self.strips
        return [(-0.5 * self.contact_width) + (index + 0.5) * width
                for index in range(self.strips)]

    def geometry_report(self) -> dict[str, float]:
        """Geometry checks required before any resistance number is trusted."""
        return {
            "length_m": self.length,
            "beam_m": self.beam,
            "draft_m": self.draft,
            "length_beam_ratio": self.length / self.beam,
            "beam_draft_ratio": self.beam / self.draft,
            "block_coefficient": self.block_coefficient,
            "displacement_volume_m3": self.displacement_volume,
            "displacement_mass_t": self.displacement_volume * 1025.0 / 1000.0,
            "contact_width_m": self.contact_width,
            "strips": float(self.strips),
        }


def strip_resistance(ship: IcebreakerShip, model: IceResistanceModel,
                     thicknesses: Sequence[float],
                     *, brash_flags: Sequence[bool] | None = None) -> dict[str, object]:
    """Total ice resistance from a per-strip thickness distribution.

    ``thicknesses`` must provide one level-ice-equivalent thickness per strip
    (:meth:`IcebreakerShip.strip_centres` ordering).  ``brash_flags`` marks strips
    inside the cleared channel, where the ice is a refrozen brash layer rather than
    a continuous plate.
    """
    if len(thicknesses) != ship.strips:
        raise ValueError(f"expected {ship.strips} thickness values, got {len(thicknesses)}")
    flags = [False] * ship.strips if brash_flags is None else list(brash_flags)
    if len(flags) != ship.strips:
        raise ValueError("brash_flags must match the number of strips")
    strip_width = ship.contact_width / ship.strips
    rows: list[dict[str, float]] = []
    total_breaking = 0.0
    total_submergence = 0.0
    total_friction = 0.0
    for index, (thickness, brash) in enumerate(zip(thicknesses, flags)):
        breaking, submergence, friction = model.sheet.line_load(float(thickness), brash=bool(brash))
        rows.append({
            "strip_index": float(index),
            "transverse_position_m": ship.strip_centres()[index],
            "width_m": strip_width,
            "thickness_m": float(thickness),
            "brash": bool(brash),
            "breaking_resistance_N": breaking * strip_width,
            "submergence_resistance_N": submergence * strip_width,
            "friction_resistance_N": friction * strip_width,
            "total_resistance_N": (breaking + submergence + friction) * strip_width,
        })
        total_breaking += breaking * strip_width
        total_submergence += submergence * strip_width
        total_friction += friction * strip_width
    scale = model.scale_factor
    return {
        "strips": rows,
        "breaking_resistance_N": total_breaking * scale,
        "submergence_resistance_N": total_submergence * scale,
        "friction_resistance_N": total_friction * scale,
        "total_resistance_N": (total_breaking + total_submergence + total_friction) * scale,
        "scale_factor": scale,
        "mean_thickness_m": sum(float(t) for t in thicknesses) / ship.strips,
    }


def level_ice_resistance(ship: IcebreakerShip, model: IceResistanceModel,
                         thickness: float, *, brash: bool = False) -> dict[str, object]:
    """Single-ship level-ice resistance baseline (uniform thickness)."""
    thickness = _number(thickness, "thickness", nonnegative=True)
    return strip_resistance(ship, model, [thickness] * ship.strips,
                            brash_flags=[brash] * ship.strips)


def ice_properties_from_reference_row(row: dict[str, str]) -> IceSheet:
    """Build an :class:`IceSheet` from a row of the MT Uikku reference CSV."""
    def value(key: str) -> float:
        text = row.get(key, "")
        if text is None or str(text).strip() == "":
            raise ValueError(f"reference row has no value for {key}")
        return float(text)

    return IceSheet(
        flexural_strength=value("flexural_strength_kPa") * 1.0e3,
        elastic_modulus=value("ice_elastic_modulus_MPa") * 1.0e6,
    )


def calibrate_ice_resistance(
    rows: Sequence[dict[str, str]],
    ship: IcebreakerShip,
    *,
    base_model: IceResistanceModel | None = None,
    holdout: int = 3,
) -> dict[str, object]:
    """Single-parameter scale calibration with a held-out validation subset.

    The scale factor minimises the ordinary least squares (ratio form) error over
    the training cases, ``s = sum(R_meas R_pred) / sum(R_pred^2)``, and is then
    frozen for the held-out cases.  Returns training and held-out metrics so the
    reported accuracy is out-of-sample.
    """
    base = base_model or IceResistanceModel()
    usable = []
    for row in rows:
        measured = float(row["experimental_reference_table8_kN"]) * 1.0e3
        thickness = float(row["ice_thickness_m"])
        if measured <= 0 or thickness <= 0:
            continue
        sheet = ice_properties_from_reference_row(row)
        prediction = level_ice_resistance(ship, replace(base, sheet=sheet, scale_factor=1.0), thickness)
        usable.append({
            "test_id": row.get("test_id", ""),
            "measured_N": measured,
            "thickness_m": thickness,
            "unscaled_prediction_N": float(prediction["total_resistance_N"]),
        })
    if len(usable) < holdout + 3:
        raise ValueError("not enough reference cases for a leave-out calibration")
    usable.sort(key=lambda item: item["test_id"])
    train = usable[holdout:]
    validate = usable[:holdout]
    numerator = sum(item["measured_N"] * item["unscaled_prediction_N"] for item in train)
    denominator = sum(item["unscaled_prediction_N"] ** 2 for item in train)
    scale = numerator / denominator

    def metrics(items: list[dict[str, float]]) -> list[dict[str, float]]:
        output = []
        for item in items:
            predicted = scale * item["unscaled_prediction_N"]
            output.append({
                "test_id": item["test_id"],
                "thickness_m": item["thickness_m"],
                "measured_kN": item["measured_N"] / 1000.0,
                "predicted_kN": predicted / 1000.0,
                "relative_error_percent": (predicted - item["measured_N"]) / item["measured_N"] * 100.0,
            })
        return output

    train_rows = metrics(train)
    validate_rows = metrics(validate)

    def mape(items: list[dict[str, float]]) -> float:
        return sum(abs(item["relative_error_percent"]) for item in items) / len(items)

    return {
        "scale_factor": scale,
        "training": train_rows,
        "holdout": validate_rows,
        "training_mape_percent": mape(train_rows),
        "holdout_mape_percent": mape(validate_rows),
        "holdout_count": len(validate_rows),
        "ship": ship.geometry_report(),
        "model": {
            "friction": base.sheet.friction,
            "submergence_factor": base.sheet.submergence_factor,
            "brash_strength_factor": base.sheet.brash_strength_factor,
        },
    }