"""Parameter ledger, standard case and provenance hash for the broken-ice surface.

Every number that enters a reported resistance is either

``DERIVED``     follows from a stated assumption / geometry and is re-derived by
                the test suite,
``CITED``       transcribed from a source that ships with the repository (the
                MT Uikku model-test table, the ice-steel friction calibration),
``CONVENTION``  a pinned modelling choice, or
``WITHHELD``    literature agreement cannot be established in this offline
                sandbox.

The ledger is hashed verbatim into every payload, so a reported resistance can
be traced to the exact constants that produced it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

#: Standard broken-ice field material.  Strength / modulus are the *measured*
#: properties of MT Uikku ice sheet III (model tests 301-303,
#: ``benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv``), i.e. the
#: sheet whose measured thickness (1.04 m) is closest to the campaign standard
#: 1.0 m.  Using a measured sheet instead of a textbook default keeps the
#: standard case anchored to the same physical ice the reference table reports.
MATERIAL: dict[str, Any] = {
    "ice_density_kg_m3": 917.0,
    "water_density_kg_m3": 1025.0,
    "gravity_m_s2": 9.80665,
    "compressive_strength_pa": 2.477e6,
    "flexural_strength_pa": 0.540e6,
    "elastic_modulus_pa": 1.474e9,
    "source": "MT Uikku ice sheet III (tests 301-303), "
              "benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv",
}

#: Graded parameter ledger.  ``name -> (value, grade, justification)``.
COEFFICIENT_LEDGER: dict[str, tuple[float, str, str]] = {
    "friction_coefficient": (
        0.10, "CITED",
        "Engineering default for a steel hull on cold sea ice, pinned by the "
        "repository's ice-steel friction calibration "
        "(docs/ICE_STEEL_FRICTION_LAW.md: mu = 0.1 sits at the cold, slow "
        "corner; the same document flags ice_channel's mu = 0.3 as twice the "
        "measured cold-ice value)."),
    "water_drag_coefficient": (
        1.0, "CONVENTION",
        "Form drag of a thin floe moving in its own plane, referenced to the "
        "submerged side-wall area 2 R h; order-one bluff-body value used by "
        "tensordem.ice_channel."),
    "normal_damping_ratio": (
        0.2, "CONVENTION",
        "Contact damping at critical damping of the local contact oscillator; "
        "same value as tensordem.ice_channel.ContactParameters."),
    "chord_width_cap_diameters": (
        1.0, "DERIVED",
        "The Hertzian contact chord w = 2 sqrt(2 R delta) cannot exceed the floe "
        "diameter 2 R: beyond delta = R/2 the floe is fully engaged and the "
        "pressure acts over its whole width."),
    "chord_regularisation_m": (
        5.0e-3, "DERIVED",
        "The chord width is evaluated at delta + eps with eps = 5 mm so that the "
        "contact tangent stiffness dF/ddelta = sigma_c h sqrt(2R/(delta+eps)) "
        "stays finite at incipient contact (local elastic compliance of the ice "
        "edge before crushing). eps is 0.1 % of the standard 5 m floe diameter "
        "and is reported with every case; the contact law's own limit eps -> 0 "
        "is exercised by the unit tests."),
    "mean_floe_radius_m": (
        5.0, "CONVENTION",
        "Characteristic half-size of a broken-channel floe field; the "
        "sensitivity of the result to this choice is reported explicitly "
        "(artefact 'floe_size_sensitivity')."),
    "floe_radius_spread": (
        0.35, "CONVENTION",
        "Uniform +/-35 % radius spread about the characteristic radius "
        "(polydisperse field; monodisperse packings jam earlier)."),
    "floe_radius_min_m": (2.0, "CONVENTION", "Lower truncation of the floe radius range."),
    "floe_radius_max_m": (8.0, "CONVENTION", "Upper truncation of the floe radius range."),
    "field_placement": (
        float("nan"), "DERIVED",
        "Jittered triangular lattice from tensordem.ice_channel.from_field: the "
        "realised concentration equals the requested one to < 1 % and there is "
        "no initial floe overlap, so the reported resistance contains no "
        "placement artefact."),
    "ship_form": (
        float("nan"), "CITED",
        "MT Uikku equivalent transport hull from "
        "tensordem.level_ice.formula.TRANSPORT (L = 105 m, B = 26 m, T = 8 m, "
        "entry angle 18 deg, stem angle 26 deg), the offline hull fit to the "
        "paper's own Lindqvist column."),
    "literature_agreement": (
        float("nan"), "WITHHELD",
        "The MT Uikku paper's broken-ice channel tests (304-306, 404-406) are "
        "NOT transcribed in this repository (only the ten level-ice cases are), "
        "so no broken-ice experimental agreement is claimed here."),
}


def ledger_sha256() -> str:
    """SHA-256 over the graded ledger plus the material state."""
    payload = json.dumps(
        {
            "material": {k: (v if not isinstance(v, float) or math.isfinite(v)
                             else "nan") for k, v in MATERIAL.items()},
            "ledger": {name: [(value if math.isfinite(value) else "nan"), grade, text]
                       for name, (value, grade, text) in sorted(COEFFICIENT_LEDGER.items())},
        },
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class MaterialState:
    """Floe material state (SI)."""

    thickness_m: float = 1.0
    compressive_strength_pa: float = 2.477e6
    flexural_strength_pa: float = 0.540e6
    elastic_modulus_pa: float = 1.474e9
    ice_density_kg_m3: float = 917.0
    water_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        for name in ("thickness_m", "compressive_strength_pa", "flexural_strength_pa",
                     "elastic_modulus_pa", "ice_density_kg_m3", "water_density_kg_m3",
                     "gravity_m_s2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive (got {value!r})")
        if self.ice_density_kg_m3 >= self.water_density_kg_m3:
            raise ValueError("ice density must be below water density for floating floes")

    def as_dict(self) -> dict[str, float]:
        return {
            "thickness_m": self.thickness_m,
            "compressive_strength_pa": self.compressive_strength_pa,
            "flexural_strength_pa": self.flexural_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "ice_density_kg_m3": self.ice_density_kg_m3,
            "water_density_kg_m3": self.water_density_kg_m3,
            "gravity_m_s2": self.gravity_m_s2,
        }


@dataclass(frozen=True)
class CaseSpec:
    """One direct-navigation transit case."""

    name: str = "standard"
    concentration: float = 0.6
    thickness_m: float = 1.0
    speed_m_s: float = 3.0
    seed: int = 7
    mean_floe_radius_m: float = 5.0
    domain_length_m: float = 200.0
    domain_width_m: float = 100.0
    time_step_s: float = 4.0e-3
    law: str = "crushing"          # "crushing" | "penalty"
    hull_form: str = "ship"        # "ship" | "block"
    friction_coefficient: float = 0.10
    record_every: int = 20
    steady_fraction: float = 0.5
    compressive_strength_pa: float = MATERIAL["compressive_strength_pa"]
    null_field: bool = False      # field-free control: one distant micro floe

    def __post_init__(self) -> None:
        if not 0.0 <= self.concentration <= 1.0:
            raise ValueError("concentration must lie in [0, 1]")
        for name in ("thickness_m", "speed_m_s", "mean_floe_radius_m",
                     "domain_length_m", "domain_width_m", "time_step_s",
                     "compressive_strength_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive (got {value!r})")
        if self.law not in ("crushing", "penalty"):
            raise ValueError("law must be 'crushing' or 'penalty'")
        if self.hull_form not in ("ship", "block"):
            raise ValueError("hull_form must be 'ship' or 'block'")
        if not 0.0 <= self.friction_coefficient < 1.5:
            raise ValueError("friction_coefficient must lie in [0, 1.5)")
        if not 0.0 < self.steady_fraction < 1.0:
            raise ValueError("steady_fraction must lie in (0, 1)")

    @property
    def material(self) -> MaterialState:
        return MaterialState(thickness_m=self.thickness_m,
                             compressive_strength_pa=self.compressive_strength_pa)

    def with_(self, **changes: Any) -> "CaseSpec":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "concentration": self.concentration,
            "thickness_m": self.thickness_m,
            "speed_m_s": self.speed_m_s,
            "seed": self.seed,
            "mean_floe_radius_m": self.mean_floe_radius_m,
            "domain_length_m": self.domain_length_m,
            "domain_width_m": self.domain_width_m,
            "time_step_s": self.time_step_s,
            "law": self.law,
            "hull_form": self.hull_form,
            "friction_coefficient": self.friction_coefficient,
            "record_every": self.record_every,
            "steady_fraction": self.steady_fraction,
            "compressive_strength_pa": self.compressive_strength_pa,
            "null_field": self.null_field,
        }


#: Campaign standard case, exactly as pinned by the mission statement:
#: V = 3 m/s, h = 1.0 m, C_ice = 0.6.
STANDARD_CASE = CaseSpec()

#: Named variants used by the validation campaign.
CASE_SPECS: dict[str, CaseSpec] = {
    "standard": STANDARD_CASE,
    "standard_seed_11": STANDARD_CASE.with_(name="standard_seed_11", seed=11),
    "standard_seed_23": STANDARD_CASE.with_(name="standard_seed_23", seed=23),
    "penalty_baseline": STANDARD_CASE.with_(name="penalty_baseline", law="penalty"),
    "block_hull_baseline": STANDARD_CASE.with_(name="block_hull_baseline", hull_form="block"),
    "no_ice": STANDARD_CASE.with_(name="no_ice", concentration=0.0,
                                 null_field=True),
    "dt_fine": STANDARD_CASE.with_(name="dt_fine", time_step_s=2.0e-3),
    "sparse_ice": STANDARD_CASE.with_(name="sparse_ice", concentration=0.1),
    "floe_small": STANDARD_CASE.with_(name="floe_small", mean_floe_radius_m=3.0),
    "floe_large": STANDARD_CASE.with_(name="floe_large", mean_floe_radius_m=8.0),
}
for _v in (2.0, 2.5, 3.5, 4.0, 4.5, 5.0):
    CASE_SPECS[f"speed_{_v:g}"] = STANDARD_CASE.with_(
        name=f"speed_{_v:g}", speed_m_s=_v)
for _h in (0.5, 0.75, 1.25, 1.5):
    CASE_SPECS[f"thick_{_h:g}"] = STANDARD_CASE.with_(
        name=f"thick_{_h:g}", thickness_m=_h)


def case_spec(name: str) -> CaseSpec:
    try:
        return CASE_SPECS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown case {name!r}; known: {sorted(CASE_SPECS)}") from exc