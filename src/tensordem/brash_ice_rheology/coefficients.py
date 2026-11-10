"""Graded coefficient ledger for the brash-ice channel resistance model.

Policy
------
Every constant that enters the model carries a grade and a one-line
justification, and the set is hashed (SHA-256) into the results so a report can
name exactly which coefficient set produced which number.  This mirrors
:mod:`tensordem.level_ice.formula` and
:mod:`tensordem.ice_resistance_scaling` (``BROKEN_ICE_LEDGER``).

Grades
------
``LITERATURE``  published value reproduced from a source readable in the
                repository (the Jop et al. 2006 mu(I) constants already pinned
                in :mod:`tensordem.granular_column.scaling`).
``DERIVED``     obtained from a stated derivation in this repository.
``INHERITED``   taken over from an existing graded in-repository ledger.
``CONVENTION``  an order-one modelling choice; declared, not claimed to be a
                published number.
``CALIBRATED``  the single free closure constant, fixed against the reference
                anchor at the standard case (see
                :mod:`tensordem.brash_ice_rheology.calibration`).

``literature_agreement`` is WITHHELD: Suyuthi et al. (2013) and the FSICR
commentary are not readable in this offline sandbox, so no claim is made that
any coefficient reproduces a published brash-channel resistance number.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

__all__ = [
    "COEFFICIENT_LEDGER",
    "REFERENCE_POROSITY",
    "CONFINEMENT_COEFFICIENT",
    "ledger_sha256",
    "coefficient_table",
    "check_coefficient",
]

#: Solid (void-free) ice density, kg/m^3.
ICE_SOLID_DENSITY = 917.0
#: Sea-water density, kg/m^3.
WATER_DENSITY = 1025.0
#: Standard gravity, m/s^2.
GRAVITY = 9.80665

#: Porosity at which every porosity-normalised closure is defined (=1).
REFERENCE_POROSITY = 0.3

#: The single calibrated closure constant: the share of the incoming brash
#: momentum flux that is converted into a *sustained* confinement pressure in
#: the hull shear band.  Fixed in
#: :func:`tensordem.brash_ice_rheology.calibration.calibrate_family`.
CONFINEMENT_COEFFICIENT = 0.640713

#: name -> (value, grade, note)
COEFFICIENT_LEDGER: Dict[str, Tuple[float, str, str]] = {
    "solid_ice_density_kg_m3": (
        ICE_SOLID_DENSITY, "CONVENTION",
        "Sea-ice solid density 917 kg/m^3 (same value as "
        "tensordem.ice_resistance_scaling.ICE_DENSITY)."),
    "water_density_kg_m3": (
        WATER_DENSITY, "CONVENTION",
        "Sea-water density 1025 kg/m^3 (same value as "
        "tensordem.ice_resistance_scaling.WATER_DENSITY)."),
    "gravity_m_s2": (
        GRAVITY, "CONVENTION",
        "Standard gravity 9.80665 m/s^2."),
    "mu_s": (
        0.36, "LITERATURE",
        "Jop, Forterre & Pouliquen (2006) static friction of the mu(I) law; "
        "the same value is pinned in tensordem.granular_column.scaling."
        "JOP_MU_S and used by tensordem.mu_i_rheology."),
    "mu_2": (
        0.84, "LITERATURE",
        "Jop et al. (2006) dynamic friction coefficient "
        "(tensordem.granular_column.scaling.JOP_MU_2)."),
    "i_0": (
        0.30, "LITERATURE",
        "Jop et al. (2006) crossover inertial number "
        "(tensordem.granular_column.scaling.JOP_I_0)."),
    "shear_band_coefficient": (
        2.0, "DERIVED",
        "Shear-band thickness delta = c*<d> at the hull.  A dense granular "
        "shear band localises in a layer a few grain diameters thick; "
        "c = 2 puts the band at the lower end of the 2-5 d range reported for "
        "plane shear of dense packings, i.e. the conservative (thin-band, "
        "high-rate) choice."),
    "confinement_coefficient": (
        CONFINEMENT_COEFFICIENT, "CALIBRATED",
        "p_dyn = 0.5*C_c*rho_bulk*v^2: the share of the brash momentum flux "
        "that is trapped as a sustained confinement pressure in the hull "
        "shear band.  The only free constant of the model; fixed by the "
        "minimax protocol of calibration.calibrate_family over the standard "
        "beam series B = 15/25/35 m (H=1 m, phi=0.3, v=3 m/s), i.e. it "
        "balances the model's and the anchor's B-scaling."),
    "displaced_ice_volume_factor": (
        1.0, "DERIVED",
        "F_buoy = rho_w g V_displaced with V_displaced = 1.0*(1-phi)*B*h_sub, "
        "the *solid* ice volume swept per metre of advance.  Only the ice "
        "displaces water - the voids are already full of it - so the brash "
        "departure from the level-ice Lindqvist convention "
        "(0.9*B*H^2, tensordem.ice_resistance_scaling.BROKEN_ICE_LEDGER) is "
        "the factor (1-phi)*h_sub/H, which also carries the correct limits "
        "phi->1 => V->0 and h_sub->0 => V->0."),
    "keel_drag_coefficient": (
        2.2, "CONVENTION",
        "C_D of the submerged brash mass under the hull.  Bluff, porous, "
        "interlocked ice rubble sheds a cavity and carries a large wake, so "
        "C_D is well above the smooth-body 0.1-0.5 and sits in the 1.5-3 band "
        "used for blunt ice masses; 2.2 is inside that band.  Declared as a "
        "convention - it is not calibrated here."),
    "mobilization_speed_m_s": (
        0.05, "CONVENTION",
        "Quasi-static release speed of the processing closure "
        "f = v/(v+v_0).  Below v_0 the brash can escape laterally around the "
        "hull instead of being sheared/depressed, so the processing "
        "resistance vanishes with v.  At the standard case (v = 3 m/s) "
        "f = 0.984, i.e. the closure moves no reported number by more than "
        "1.7%; it only enforces the structural limit R(v=0) = 0."),
    "literature_agreement": (
        float("nan"), "WITHHELD",
        "Suyuthi et al. (2013) and the FSICR commentary are not readable in "
        "this offline sandbox; the reference anchor used for the comparison is "
        "the in-repository hashed broken-ice/brash anchor "
        "(tensordem.ice_resistance_scaling.broken_ice_anchor).  No claim is "
        "made that this coefficient set reproduces a published "
        "brash-channel number."),
}


def check_coefficient(name: str) -> Tuple[float, str, str]:
    """Return ``(value, grade, note)`` for ``name``."""
    try:
        return COEFFICIENT_LEDGER[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown coefficient {name!r}; "
                       f"known: {sorted(COEFFICIENT_LEDGER)}") from exc


def coefficient_table() -> Dict[str, Dict[str, object]]:
    """JSON-ready table of the ledger."""
    return {name: {"value": value, "grade": grade, "note": note}
            for name, (value, grade, note) in sorted(COEFFICIENT_LEDGER.items())}


def ledger_sha256() -> str:
    """SHA-256 over the sorted ledger: provenance stamp for every result."""
    payload = json.dumps({name: [value, grade, note]
                          for name, (value, grade, note)
                          in sorted(COEFFICIENT_LEDGER.items())},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
