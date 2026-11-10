"""Lindqvist three-component reference reduced to the DEM model basis.

The DEM model here is a **unit-width model**: the wedge bow is extruded across a
finite-transverse sheet and the reaction is reducible to newtons per metre of
contact width.  The Lindqvist-style baselines of
:mod:`tensordem.threefield.resistance` are total forces built from hull
particulars (``R_crush = 0.5 sigma_c h^2 (tan a + mu / cos a)`` carries no beam,
``R_bend`` and ``R_sub`` do).  To compare apples with apples every component is
reduced to the same per-metre basis by dividing the total force by the hull beam
``B`` and the DEM reaction by the modelled width, which is the reduction already
used by ``docs/LEVEL_ICE_RESISTANCE_DEM.md``.

Two things are deliberately reported separately:

* the **three mechanisms** (crushing / bending / submersion) that both sides
  share, and
* the **speed-dependent added drag** ``0.5 rho_w B h V^2`` that Lindqvist adds on
  top and that the DEM does not represent (the DEM has a linear numerical
  ``drag`` term, not a hydrodynamic pressure drag of the displaced sheet).  It is
  excluded from the three-component comparison and reported for transparency.

Coefficient provenance is inherited unchanged from ``docs/threefield/FORMULAS.md``:
the exponents and the ``0.5``/``0.25`` structural coefficients are derived
there, the ``0.9``/``0.05`` submersion conventions are pinned implementation
choices, and agreement with the published Lindqvist/Riska numbers is WITHHELD
(the papers are not readable offline).  Nothing in this module upgrades the
literature grade of those coefficients: agreement between a DEM built on the
same ``sigma_c``/``sigma_f``/``mu`` and this formula is a **consistency** check of
the mapping, not an independent validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..threefield.resistance import (
    Environment,
    HullGeometry,
    IceProperties,
    lindqvist_resistance,
    relative_error_pct,
)

MECHANISMS: tuple[str, str, str] = ("crushing", "bending", "submersion")


@dataclass(frozen=True)
class LindqvistReference:
    """Lindqvist components for one geometry, in N and in N/m."""

    beam_m: float
    length_m: float
    components_n: dict[str, float]
    speed_dependent_n: float

    @property
    def total_n(self) -> float:
        return sum(self.components_n.values())

    @property
    def components_per_width_n_m(self) -> dict[str, float]:
        return {name: value / self.beam_m for name, value in self.components_n.items()}

    @property
    def total_per_width_n_m(self) -> float:
        return self.total_n / self.beam_m

    @property
    def shares(self) -> dict[str, float]:
        total = self.total_n
        if total <= 0.0:
            return {name: 0.0 for name in MECHANISMS}
        return {name: value / total for name, value in self.components_n.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "beam_m": self.beam_m,
            "length_m": self.length_m,
            "components_N": dict(self.components_n),
            "components_per_width_N_m": self.components_per_width_n_m,
            "total_N": self.total_n,
            "total_per_width_N_m": self.total_per_width_n_m,
            "speed_dependent_N": self.speed_dependent_n,
            "shares": self.shares,
        }


def build_reference(
    *,
    thickness_m: float,
    beam_m: float,
    length_m: float,
    entry_angle_deg: float,
    speed_m_s: float,
    friction_coefficient: float,
    compressive_strength_pa: float = 1.5e6,
    flexural_strength_pa: float = 0.5e6,
    density_kg_m3: float = 917.0,
    water_density_kg_m3: float = 1025.0,
    gravity_m_s2: float = 9.80665,
) -> LindqvistReference:
    """Build the Lindqvist components for the DEM model geometry."""
    ice = IceProperties(
        thickness_m=thickness_m,
        compressive_strength_pa=compressive_strength_pa,
        flexural_strength_pa=flexural_strength_pa,
        density_kg_m3=density_kg_m3,
    )
    hull = HullGeometry(
        beam_m=beam_m,
        length_m=length_m,
        waterline_entry_angle_deg=entry_angle_deg,
        friction_coefficient=friction_coefficient,
    )
    environment = Environment(
        water_density_kg_m3=water_density_kg_m3, gravity_m_s2=gravity_m_s2
    )
    result = lindqvist_resistance(ice, hull, speed_m_s, environment)
    return LindqvistReference(
        beam_m=beam_m,
        length_m=length_m,
        components_n={
            "crushing": result["crushing_N"],
            "bending": result["bending_N"],
            "submersion": result["submersion_N"],
        },
        speed_dependent_n=result["speed_dependent_N"],
    )


def compare_components(
    dem_components_n_m: Mapping[str, float],
    dem_total_n_m: float,
    reference: LindqvistReference,
    *,
    tolerance_pct: float = 15.0,
    share_tolerance_pct: float = 15.0,
) -> dict[str, Any]:
    """Component-by-component comparison of the DEM split with Lindqvist.

    ``dem_components_n_m`` are per-unit-width components.  Relative errors are
    computed against the Lindqvist value of the same component; the total error
    is computed against ``R_crush + R_bend + R_sub`` (the added drag is reported
    separately, see the module docstring).  Shares are compared separately
    because a mechanism share is scale free and therefore survives the absolute
    scale mismatch of a model-scale DEM.
    """
    rows: list[dict[str, Any]] = []
    for name in MECHANISMS:
        dem_value = float(dem_components_n_m[name])
        ref_value = float(reference.components_per_width_n_m[name])
        rows.append(
            {
                "mechanism": name,
                "dem_N_m": dem_value,
                "lindqvist_N_m": ref_value,
                "error_pct": relative_error_pct(dem_value, ref_value) if ref_value != 0.0 else None,
                "signed_error_pct": (dem_value - ref_value) / ref_value * 100.0
                if ref_value != 0.0 else None,
                "dem_share": None,  # filled below
                "lindqvist_share": reference.shares[name],
            }
        )
    dem_total = float(dem_total_n_m)
    for row in rows:
        row["dem_share"] = row["dem_N_m"] / dem_total if dem_total != 0.0 else 0.0
        if row["dem_share"] is not None and row["lindqvist_share"]:
            row["share_error_pct"] = relative_error_pct(row["dem_share"], row["lindqvist_share"])
        else:
            row["share_error_pct"] = None
    reference_total = reference.total_per_width_n_m
    total_error = relative_error_pct(dem_total, reference_total)
    total_signed = (dem_total - reference_total) / reference_total * 100.0 if reference_total else None
    share_passed = all(
        row["share_error_pct"] is not None and abs(row["share_error_pct"]) <= share_tolerance_pct
        for row in rows
    )
    return {
        "rows": rows,
        "dem_total_N_m": dem_total,
        "lindqvist_total_N_m": reference.total_per_width_n_m,
        "total_error_pct": total_error,
        "total_signed_error_pct": total_signed,
        "tolerance_pct": tolerance_pct,
        "share_tolerance_pct": share_tolerance_pct,
        "total_passed": bool(abs(total_error) <= tolerance_pct),
        "shares_passed": bool(share_passed),
        "reference": reference.as_dict(),
    }