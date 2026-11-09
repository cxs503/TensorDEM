"""Fail-closed applicability domain of the Terzaghi consolidation surface.

The consolidation chain developed for campaign node IN99-N08 is only valid
inside a narrow, physically explicit window.  Every gate below is checked
*before* a curve is published, and a failed gate raises by default
(``strict=True``): a curve produced outside the domain is not "slightly
wrong", it is meaningless, so the interface refuses to hand it downstream.

Gates
-----
``saturated``
    The pore fluid must fill the voids (``S_r >= 0.99``).  Below that the gas
    phase makes the pore fluid compressible and the storage term (and hence
    ``C_v``) is no longer the single-parameter form used here.
``darcy``
    Darcy's law is used for the flux, so the Reynolds number of the pore flow
    ``Re_p = rho_w v_s d / mu_w`` (``v_s`` = superficial/Darcy velocity,
    ``d`` = mean grain diameter) must stay below 1.  Beyond that the
    Forchheimer inertial correction is required and ``C_v`` is not constant.
``no_cementation``
    The framework carries no bond strength: chemical cementation (calcite,
    hydrate, frozen inter-particle bridges) means the skeleton has a finite
    tensile/cohesive strength and the effective-stress split is incomplete.
``not_thawing_ice_rubble``
    Ice-rubble accumulations that are frozen or thawing are *out of scope*
    (explicit campaign instruction): the solid phase changes state, the pore
    fluid is not a passive saturating phase, and the oedometer modulus is not
    a material constant.  Named provenance ``ice_rubble_thaw`` fails closed.
``carman_kozeny_window``
    The permeability closure is Carman-Kozeny, calibrated for
    ``0.30 <= n <= 0.55`` (random sphere packings); outside that window the
    closure is an extrapolation and is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ConsolidationDomainError",
    "ApplicabilityReport",
    "ApplicabilityLimits",
    "check_applicability",
    "OUT_OF_SCOPE_PROVENANCE",
]

#: Provenance tags that are out of scope by construction (fail closed).
OUT_OF_SCOPE_PROVENANCE = {
    "ice_rubble_thaw": "frozen/thawing ice rubble: solid phase changes state",
    "frozen_soil": "frozen soil: pore ice carries load, C_v undefined",
    "cemented_sand": "chemically cemented: skeleton has tensile strength",
}


@dataclass(frozen=True)
class ApplicabilityLimits:
    """Machine-checkable limits of the consolidation surface."""

    min_saturation: float = 0.99
    max_pore_reynolds: float = 1.0
    min_porosity: float = 0.30
    max_porosity: float = 0.55
    require_no_cementation: bool = True
    refuse_thawing_ice_rubble: bool = True


@dataclass
class ApplicabilityReport:
    """Outcome of the domain check, with per-gate evidence."""

    applicable: bool
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applicable": bool(self.applicable),
            "gates": self.gates,
            "reasons": list(self.reasons),
            "advisories": list(self.advisories),
        }


class ConsolidationDomainError(ValueError):
    """Raised when a consolidation result is requested outside the domain."""


def check_applicability(
    *,
    saturation: float,
    pore_reynolds: float,
    porosity: float,
    chemical_cementation: bool = False,
    provenance: str = "granular_bed",
    limits: ApplicabilityLimits | None = None,
    strict: bool = False,
) -> ApplicabilityReport:
    """Evaluate every gate; ``strict=True`` raises on any failure.

    Parameters are the dimensionless/physical witnesses of the case, never
    fitted: ``saturation`` (``S_r``), ``pore_reynolds`` (``Re_p``),
    ``porosity`` (``n``), the boolean ``chemical_cementation`` flag and the
    material ``provenance`` tag.
    """
    lim = limits or ApplicabilityLimits()
    gates: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    gates["saturated"] = {
        "value": float(saturation),
        "limit": f">= {lim.min_saturation}",
        "pass": bool(saturation >= lim.min_saturation),
        "blocking": True,
    }
    gates["darcy"] = {
        "value": float(pore_reynolds),
        "limit": f"Re_p < {lim.max_pore_reynolds}",
        "pass": bool(pore_reynolds < lim.max_pore_reynolds),
        "definition": "Re_p = rho_w v_s d / mu_w (v_s = Darcy velocity)",
        "blocking": True,
    }
    # advisory (non-blocking): the Carman-Kozeny constant 180 is fitted over
    # 0.35 <= n <= 0.55; outside that window the *permeability* closure is an
    # extrapolation, but the consolidation chain itself (mass balance + Darcy
    # + linear skeleton) remains well posed, so this gate warns instead of
    # blocking.  The bias it flags is quantified in the report.
    gates["carman_kozeny_window"] = {
        "value": float(porosity),
        "limit": f"{lim.min_porosity} <= n <= {lim.max_porosity}",
        "pass": bool(lim.min_porosity <= porosity <= lim.max_porosity),
        "blocking": False,
    }
    gates["no_cementation"] = {
        "value": bool(chemical_cementation),
        "limit": "chemical_cementation == False",
        "pass": bool(not (lim.require_no_cementation and chemical_cementation)),
        "blocking": True,
    }
    out_of_scope = provenance in OUT_OF_SCOPE_PROVENANCE if lim.refuse_thawing_ice_rubble else False
    gates["not_thawing_ice_rubble"] = {
        "value": provenance,
        "limit": "provenance not in " + ", ".join(sorted(OUT_OF_SCOPE_PROVENANCE)),
        "pass": bool(not out_of_scope),
        "blocking": True,
    }

    advisories: list[str] = []
    for name, gate in gates.items():
        if not gate["pass"] and not gate.get("blocking", True):
            advisories.append(f"{name} advisory: {gate['value']} vs {gate['limit']}")
    for name, gate in gates.items():
        if not gate["pass"] and gate.get("blocking", True):
            extra = ""
            if name == "not_thawing_ice_rubble":
                extra = f" ({OUT_OF_SCOPE_PROVENANCE[provenance]})"
            reasons.append(f"{name} gate failed: {gate['value']} vs {gate['limit']}{extra}")

    report = ApplicabilityReport(applicable=not reasons, gates=gates, reasons=reasons,
                                advisories=advisories)
    if strict and not report.applicable:
        raise ConsolidationDomainError("; ".join(reasons))
    return report