"""Ice load on vertical structures: dispatch between crushing and rubble pile-up.

This is the public entry point other modules / the platform use to obtain an ice
load on a fixed vertical structure.  It routes between two DEM engines:

* :mod:`tensordem.ice_crushing_vertical_structure` -- crushing-controlled load
  without an accumulation model (design-basis ISO 19906 envelope);
* :mod:`tensordem.ice_rubble_pileup_vertical` -- vertical-section pile-up DEM
  that resolves the rubble pile, its porosity and the load it transmits to the
  structure (this is the campaign p227w2-N04 capability).

Loads are returned per unit width ``[N/m]`` together with the ISO 19906
reference for the same configuration so a caller can form a design ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ice_rubble_pileup_vertical import (
    PileupVerticalConfig,
    iso_19906_reference,
    run_vertical_pileup_case,
)

__all__ = ["StructureLoadResult", "vertical_structure_load"]


@dataclass
class StructureLoadResult:
    F_max: float            # N/m, per unit width
    F_steady: float         # N/m
    F_ISO: float            # N
    F_ISO_per_width: float  # N/m
    ratio_max: float
    porosity: float
    pile_height: float
    pile_angle_deg: float
    engine: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def vertical_structure_load(sigma_c: float, h: float, D: float,
                            drift_speed: float = 0.5, mu: float = 0.45,
                            shape: str = "cylinder", seed: int = 11,
                            model: str = "pileup", **cfg_kw) -> StructureLoadResult:
    """Ice load on a vertical structure [N/m] for the given ice-field conditions.

    Parameters
    ----------
    sigma_c, h, D, drift_speed, mu, shape : ice-field and structure conditions
    model : ``"pileup"`` (vertical-section rubble DEM, default) or
        ``"iso"`` (ISO 19906 A.8.2.6 reference only, no DEM).
    """
    cfg = PileupVerticalConfig(sigma_c=sigma_c, h=h, D=D, drift_speed=drift_speed,
                               mu=mu, shape=shape, seed=seed, **cfg_kw)
    ref = iso_19906_reference(cfg)
    if model == "iso":
        return StructureLoadResult(
            F_max=float("nan"), F_steady=float("nan"),
            F_ISO=ref["F_ISO"], F_ISO_per_width=ref["F_ISO_per_width"],
            ratio_max=float("nan"), porosity=float("nan"),
            pile_height=float("nan"), pile_angle_deg=float("nan"), engine="iso")
    if model != "pileup":
        raise ValueError(f"unknown model {model!r}")
    res = run_vertical_pileup_case(cfg)
    return StructureLoadResult(
        F_max=res.F_max, F_steady=res.F_steady, F_ISO=res.F_ISO,
        F_ISO_per_width=res.F_ISO_per_width, ratio_max=res.ratio_max,
        porosity=res.porosity, pile_height=res.pile_height,
        pile_angle_deg=res.pile_angle_deg, engine="pileup")
