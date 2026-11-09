"""Freeze-bond correction to the Lindqvist-family level-ice resistance anchor.

The level-ice anchor of :mod:`tensordem.level_ice.formula`
(``lindqvist_components``) decomposes the resistance into crushing, bending and
submersion terms with a machine-checkable coefficient ledger
(``ledger_sha256``).  It is a *continuum, unbonded* description of level ice: it
has no notion of refrozen rubble cohesion, so it cannot say how much a refrozen
ice field that arches between two banks adds (or removes) from the resistance.

This module produces a **bounded difference interval** instead of a point
prediction:

* the DEM channel campaign gives the *ratio* ``r(kappa) = F_platen(kappa) /
  F_platen(0)``, i.e. how much more push force a refrozen field carries compared
  with the same frictional rubble without freeze bonds (both at the same
  displacement, same geometry, same rate -- the ratio is a DEM-only quantity);
* two envelopes are reported, because the mapping from the 2D channel ratio to a
  3D empirical resistance is *not* derived anywhere: (i) the crushing term only is
  scaled by ``r`` (the cohesion acts on the crushing/contact term), which gives the
  lower bound, and (ii) all three components are scaled by ``r``, which gives the
  upper bound.  The interval between them is the honest width of what the current
  evidence supports.

Nothing here claims that the interval is validated: it is a *relative* DEM
measurement attached to a *coefficient-graded* empirical anchor, and the report
labels it accordingly (see ``docs/freeze_bond_arching.md``, "Relationship to the
Lindqvist family").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..level_ice.formula import (
    ICEBREAKER,
    IceState,
    HullForm,
    lindqvist_components,
    ledger_sha256,
)

__all__ = ["ResistanceBand", "resistance_band", "band_over_speeds"]


@dataclass
class ResistanceBand:
    """Bounded resistance interval for a refrozen (freeze-bonded) ice field."""

    speed_m_s: float
    crushing_N: float
    bending_N: float
    submersion_N: float
    unbonded_total_N: float
    ratio: float
    lower_total_N: float
    upper_total_N: float
    kappa: float
    ledger_sha256: str
    assumptions: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "speed_m_s": self.speed_m_s,
            "crushing_N": self.crushing_N,
            "bending_N": self.bending_N,
            "submersion_N": self.submersion_N,
            "unbonded_total_N": self.unbonded_total_N,
            "ratio_dem_bonded_over_unbonded": self.ratio,
            "lower_total_N": self.lower_total_N,
            "upper_total_N": self.upper_total_N,
            "delta_lower_N": self.lower_total_N - self.unbonded_total_N,
            "delta_upper_N": self.upper_total_N - self.unbonded_total_N,
            "kappa": self.kappa,
            "ledger_sha256": self.ledger_sha256,
            "assumptions": self.assumptions,
        }


def resistance_band(
    kappa: float,
    ratio: float,
    *,
    ice: IceState | None = None,
    hull: HullForm = ICEBREAKER,
    speed_m_s: float = 0.0,
) -> ResistanceBand:
    """Attach a DEM bonded/unbonded ratio to the Lindqvist anchor.

    ``ratio = F_platen(kappa) / F_platen(kappa=0)`` from the same-geometry DEM
    channel campaign.  ``ratio < 1`` is allowed and then *lowers* the lower bound:
    a refrozen field that crushes in a more brittle way can transmit less axial
    load than frictionally-jammed rubble.
    """
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be finite and positive")
    ice_state = ice or IceState()
    anchor = lindqvist_components(ice_state, hull, speed_m_s)
    r_c = float(anchor["crushing_N"])
    r_b = float(anchor["bending_N"])
    r_s = float(anchor["submersion_N"])
    unbonded = r_c + r_b + r_s
    lower = r_c * ratio + r_b + r_s
    upper = unbonded * ratio
    return ResistanceBand(
        speed_m_s=float(speed_m_s), crushing_N=r_c, bending_N=r_b,
        submersion_N=r_s, unbonded_total_N=unbonded, ratio=float(ratio),
        lower_total_N=float(min(lower, upper)), upper_total_N=float(max(lower, upper)),
        kappa=float(kappa), ledger_sha256=str(anchor["ledger_sha256"]),
        assumptions={
            "ratio_source": "2-D quasi-static DEM channel push, same geometry and "
                            "displacement for both arms",
            "lower_bound": "only the crushing term scaled by the DEM ratio",
            "upper_bound": "all three Lindqvist terms scaled by the DEM ratio",
            "not_derived": "the 2-D -> 3-D mapping of the ratio is not established; "
                           "the interval is a bound, not a prediction",
            "ice_state": ice_state.as_dict(),
            "hull": hull.as_dict(),
        },
    )


def band_over_speeds(
    kappa: float,
    ratio: float,
    *,
    speeds_m_s: Iterable[float] = (0.0, 0.5, 1.0, 1.5),
    ice: IceState | None = None,
    hull: HullForm = ICEBREAKER,
) -> dict[str, Any]:
    """Evaluate the band across speeds (added drag grows with speed)."""
    bands = [resistance_band(kappa, ratio, ice=ice, hull=hull, speed_m_s=v)
             for v in speeds_m_s]
    return {
        "kappa": float(kappa),
        "ratio": float(ratio),
        "ledger_sha256": ledger_sha256(),
        "bands": [b.as_dict() for b in bands],
        "remarks": "interval width is the honest uncertainty of mapping a 2-D DEM "
                   "ratio onto the 3-D Lindqvist family",
    }