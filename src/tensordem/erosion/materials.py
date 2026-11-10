"""Material library and parameter provenance for solid-particle erosion.

Solid-particle erosion (sand erosion, slurry erosion, ice-mélange abrasion) is a
*target/particle-pair* problem: every published model is calibrated for a class
of target (ductile metal, brittle ceramic) and for one class of erodent.  The
constants are therefore stored together with the provenance string that says how
they were obtained, so that a downstream user can decide whether to trust an
absolute rate prediction or only a scaling law.

Conventions
-----------
* SI units everywhere unless a field name says otherwise (``*_gpa``, ``*_um``).
* ``hardness_hv_gpa`` is the Vickers hardness in GPa (HV x 9.80665e-3).
* ``flow_stress_pa`` follows the Meyer-hardness convention ``p = Hv / 2``
  (Tabor: the representatitive flow stress of an indented metal is about
  one third to one half of the measured hardness).  Finnie's classical formula
  uses exactly this quantity ``p`` in its denominator.
* ``density_g_cm3`` is provided because Oka's erosion unit (mm^3/g) is built on
  gram-based mass of erodent.

Nothing in this module is a DEM contact law: it is a *constitutive sink* that a
DEM/CFD particle stream consumes to remove wall mass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

HV_TO_GPA = 9.80665e-3
"""HV (kgf/mm^2) -> GPa."""

VALID_CLASSES = ("ductile", "brittle")


@dataclass(frozen=True)
class Material:
    """Target-wall material properties used by the erosion models."""

    name: str
    material_class: str
    hardness_hv_gpa: float
    density_kg_m3: float
    source: str

    def __post_init__(self) -> None:
        if self.material_class not in VALID_CLASSES:
            raise ValueError(
                f"material_class must be one of {VALID_CLASSES}, got {self.material_class!r}"
            )
        if not self.hardness_hv_gpa > 0.0:
            raise ValueError("hardness_hv_gpa must be positive")
        if not self.density_kg_m3 > 0.0:
            raise ValueError("density_kg_m3 must be positive")

    @property
    def flow_stress_pa(self) -> float:
        """Representative flow stress ``p = Hv/2`` in Pa (Finnie's ``p``)."""
        return 0.5 * self.hardness_hv_gpa * 1e9

    @property
    def density_g_cm3(self) -> float:
        return self.density_kg_m3 / 1000.0

    @property
    def is_ductile(self) -> bool:
        return self.material_class == "ductile"


MATERIALS: Mapping[str, Material] = {
    # --- ductile metals ---------------------------------------------------
    "carbon_steel": Material(
        name="carbon_steel",
        material_class="ductile",
        hardness_hv_gpa=2.00,  # ~204 HV, normalized/low-alloy pipe steel
        density_kg_m3=7850.0,
        source="typical normalized carbon steel (204 HV, 7850 kg/m3); "
        "hardness from handbook values, class from erosive behaviour",
    ),
    "stainless_steel_316": Material(
        name="stainless_steel_316",
        material_class="ductile",
        hardness_hv_gpa=1.80,
        density_kg_m3=8000.0,
        source="AISI 316 austenitic stainless (185 HV, 8000 kg/m3), handbook",
    ),
    "aluminium_6061": Material(
        name="aluminium_6061",
        material_class="ductile",
        hardness_hv_gpa=1.07,  # ~107 HV (T6)
        density_kg_m3=2700.0,
        source="AA6061-T6 (107 HV, 2700 kg/m3), handbook",
    ),
    "copper": Material(
        name="copper",
        material_class="ductile",
        hardness_hv_gpa=0.87,
        density_kg_m3=8960.0,
        source="annealed copper (89 HV, 8960 kg/m3), handbook",
    ),
    # --- brittle ----------------------------------------------------------
    "alumina_ceramic": Material(
        name="alumina_ceramic",
        material_class="brittle",
        hardness_hv_gpa=15.0,  # ~1530 HV
        density_kg_m3=3900.0,
        source="sintered Al2O3 (1530 HV, 3900 kg/m3), handbook",
    ),
    "soda_lime_glass": Material(
        name="soda_lime_glass",
        material_class="brittle",
        hardness_hv_gpa=5.5,  # ~560 HV
        density_kg_m3=2500.0,
        source="soda-lime glass (560 HV, 2500 kg/m3), handbook",
    ),
}


@dataclass(frozen=True)
class DNVDuctileCoefficients:
    """DNV-RP-O501 ductile-model constants for one wall-material/sand pair.

    The standard tabulates ``K`` and the velocity exponent ``n`` per material
    and per erodent; only the carbon-steel/sand pair is quoted here from open
    literature citations of the standard.  Any other pair **must** be supplied
    by the analyst from the standard's table -- the values marked ``assumed``
    below exist so that the API can be exercised in tests, not to predict a
    field erosion rate.
    """

    k: float
    n: float
    provenance: str


DNV_DUCTILE_COEFFICIENTS: Mapping[str, DNVDuctileCoefficients] = {
    "carbon_steel": DNVDuctileCoefficients(
        k=2.0e-9,
        n=2.6,
        provenance="documented: DNV-RP-O501 ductile model, steel/sand pair "
        "(K=2.0e-9, n=2.6) as quoted in the open literature; the dimensionless "
        "K multiplies (v/v_ref)^n * F(alpha) to give kg eroded per kg sand",
    ),
    "aluminium_6061": DNVDuctileCoefficients(
        k=1.3e-9,
        n=2.4,
        provenance="ASSUMED (not read from the standard): placeholder for the "
        "aluminium/sand pair; replace with the tabulated K,n before field use",
    ),
}

OKA_CONSTANTS: Mapping[str, float] = {
    "K": 65.0,
    "k1": 0.19,
    "k2": 2.2,
    "k3": 0.19,
    "v_ref_m_s": 1.0,
    "d_ref_um": 1.0,
}
"""Oka et al. (2005) global constants.

``k1``/``k2``/``k3`` are the hardness, velocity and particle-diameter exponents
and are *material independent*; the material enters through ``Hv`` in the angle
function.  ``K`` is the published leading constant.  Because the unit
convention of ``K`` (mm^3 of removed material per gram of erodent, with Hv in
GPa, v in m/s and d in um) is not reproducible from the open literature alone,
this library uses Oka **only for its scaling law and its angle function**, never
for an absolute rate: see :func:`tensordem.erosion.models.oka_relative_erosion`.
"""

BRITTLE_CONSTANTS: Mapping[str, float] = {
    "n_velocity": 2.4,
    "n_angle": 2.4,
}
"""Brittle (lateral-crack) scaling: ``E ~ v^2.4 * sin^2.4(alpha)``.

Evans, Gulden & Rosenblatt (1978) and Wiederhorn & Lawn (1979) give a lateral
crack volume that grows faster than the impact kinetic energy; the commonly
quoted engineering exponent for the brittle regime is 2.4 (bounded by 2.0 for a
pure energy argument and ~3.0 for strongly quasi-static Hertzian cracking).
The same exponent appears on ``sin(alpha)`` because only the normal velocity
component drives crack growth, which is what puts the brittle maximum at
normal incidence.
"""

LITERATURE_VELOCITY_EXPONENTS: Mapping[str, tuple[float, str]] = {
    "carbon_steel": (
        2.6,
        "DNV-RP-O501 ductile model default velocity exponent for steel/sand",
    ),
    "aluminium_6061": (
        2.2,
        "Oka et al. (2005) fitted velocity exponent k2 = 2.2 (material independent)",
    ),
    "copper": (
        2.3,
        "ductile-metal measurement band 2.0-2.5 (Hutchings 1981); midpoint used",
    ),
    "alumina_ceramic": (
        2.4,
        "brittle-regime exponent, Evans et al. (1978)",
    ),
}
"""Literature velocity exponents used by acceptance criterion C2."""


LITERATURE_PEAK_ANGLES_DEG: Mapping[str, tuple[float, str]] = {
    "ductile": (
        20.0,
        "ductile metals: erosion maximum at 15-30 deg (Finnie 1960; Hutchings 1981); "
        "nominal reference 20 deg",
    ),
    "brittle": (
        90.0,
        "brittle solids: maximum at normal incidence (alpha_max = 90 deg)",
    ),
}
"""Literature peak-impact-angle references used by acceptance criterion C1."""


def get_material(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}") from exc


def dnv_coefficients(name: str) -> DNVDuctileCoefficients:
    try:
        return DNV_DUCTILE_COEFFICIENTS[name]
    except KeyError as exc:
        raise KeyError(
            f"no DNV-RP-O501 ductile coefficients for {name!r}; "
            f"known: {sorted(DNV_DUCTILE_COEFFICIENTS)}"
        ) from exc


__all__ = [
    "BRITTLE_CONSTANTS",
    "DNV_DUCTILE_COEFFICIENTS",
    "DNVDuctileCoefficients",
    "HV_TO_GPA",
    "LITERATURE_PEAK_ANGLES_DEG",
    "LITERATURE_VELOCITY_EXPONENTS",
    "MATERIALS",
    "Material",
    "OKA_CONSTANTS",
    "dnv_coefficients",
    "get_material",
]