"""Freeze-bond strength aging of an ice-ridge keel, anchored on Ettema & Schaefer (1986).

Physical background
-------------------
A newly formed ridge keel is rubble: the ice blocks touch but are not welded.
At sub-freezing pore-water temperature the contacts sinter and freeze, so the
rubble acquires cohesion with age.  This module answers two quantitative
questions for an ageing keel:

1. how strong is a freeze bond ``t`` after it started to grow, ``sigma_b(t)``;
2. how large is the keel resistance built on those bonds, ``R_keel(t)``.

Measured anchor (the reason this module exists)
-----------------------------------------------
R. Ettema & J.A. Schaefer, *Experiments on freeze-bonding between ice blocks in
floating ice rubble*, Journal of Glaciology **32**(112), 1986, pp. 397-403,
doi:10.3189/S0022143000012107.  Their 10-replicate shear tests (ice block on an
ice slab, water/air at ~0 degC, normal pressure on the contact up to 4 kPa)
give the quasi-static strength of a freeze bond:

* Eq. (5), p. 400:  ``tau = F_s / A = k * sigma`` for ``sigma <= 3.5 kPa`` with
  ``k = 2.01`` (distilled water), ``1.95`` (tap water), ``0.48`` (air);
* Eq. (6), p. 401:  ``tau = 2.01 * (1 - 0.06 C) * sigma`` for ``C < 12.5 %``
  (``C`` = NaCl concentration, percent by weight);
* Fig. 7 / p. 400: in air the bond reaches its asymptote ``450.4 Pa`` within
  3 min of contact; in distilled/tap/river water the strength still grows
  *linearly* with contact period up to 3 min (no saturation in the test window);
* p. 401 / Fig. 11: in 3 % saline water the bond saturates at ``~650 Pa``; in
  >= 12.5 % saline water it stays at ``~200 Pa`` and does not grow with time;
* p. 401: bonds formed in water are "about four times stronger" than bonds
  formed in air at the same normal pressure.

What is model and what is data
------------------------------
* ``sigma_b(t) = sigma_b,inf * (1 - exp(-t / tau_g))`` -- saturating growth.
  The form is the standard sintering saturation form (Timco & Frederking 1990;
  Mellor 1983) and reproduces every qualitative observation of Ettema &
  Schaefer (zero intercept, fast asymptote in air and in 3 % brine, linear
  growth in fresh water over the 3 min window, time-independent plateau above
  12.5 % brine).
* ``sigma_b,inf = k(S) * sigma_n * phi(dT)`` -- the quasi-static asymptote is
  Ettema & Schaefer's own pressure law, extrapolated from their 3 min window to
  full saturation and generalised to colder pore water by ``phi(dT)``.
* ``tau_g`` -- Arrhenius growth time.  The apparent activation energy is a
  *declared* parameter (:data:`DEFAULT_APPARENT_ACTIVATION_ENERGY`), not a
  measured one: it is chosen so that a bond at -10 degC saturates in hours,
  which is the consolidation time scale observed for ridge keels.

Evidence grading
----------------
* Ettema & Schaefer pressure/salinity laws and asymptote magnitudes: REAL
  (experiment, imported datum by datum in :data:`REFERENCE_DATA`);
* saturating time law form: REAL (sintering kinetics) and form-verified against
  the E&S time observations in :func:`validate_against_ettema_schaefer`;
* apparent activation energy and the ``phi(dT)`` hardening: PARAMETRIC_MODEL
  (declared; no -10 degC laboratory series in this repository, WITHHELD);
* tensile/shear split ``kappa = 0.5``: PARAMETRIC_MODEL (E&S measured shear
  strength only);
* bond-free rubble friction and the micro->macro cohesion transfer of the DEM
  cell: measured by ``tensordem.ridge_dem.calibrate_rubble_strength`` (REAL for
  this micro-parameter set; see ``scripts/run_ridge_freeze_bond_aging.py``).

Limitations
-----------
* Uniform pore-water temperature and salinity; no thermal gradient, no brine
  drainage, no melt/creep damage (growth only).
* The ``sigma_n`` of the E&S saline figures is ambiguous in the source: the text
  of p. 401 states 450 Pa while the Fig. 11 caption states 3.8 kPa.  Both
  readings are reported by the validation; the text reading is preferred because
  the 3.8 kPa reading would contradict the authors' own Eq. (6) by an order of
  magnitude.
* All quantities SI (m, kg, s, K, Pa; salinity in percent by weight).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .ridge_dem import RidgeMaterial

__all__ = [
    "FREEZING_POINT_K",
    "SEAWATER_FREEZING_DEPRESSION_K_PER_PPT",
    "DEFAULT_APPARENT_ACTIVATION_ENERGY",
    "REFERENCE_DATA",
    "EttemaSchaeferDatum",
    "SaturationFreezeBondModel",
    "AgingState",
    "KeelAgingResistance",
    "validate_against_ettema_schaefer",
    "aged_material",
    "cohesion_transfer_coefficient",
]

#: Freezing point of fresh water at 1 atm, K.
FREEZING_POINT_K = 273.15
#: Linear freezing-point depression of sea water, K per ppt (Assur's approximation).
SEAWATER_FREEZING_DEPRESSION_K_PER_PPT = 0.054
#: Declared apparent activation energy of freeze-bond growth, J/mol.
DEFAULT_APPARENT_ACTIVATION_ENERGY = 2.5e5
#: Universal gas constant, J/(mol K).
GAS_CONSTANT = 8.314462618


# ---------------------------------------------------------------------------
# the imported experimental data set
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EttemaSchaeferDatum:
    """One digitised datum of Ettema & Schaefer (1986).

    ``kind`` is one of

    ``"coefficient"``   shear strength per unit normal pressure, ``k`` in
                        ``tau = k * sigma`` (dimensionless);
    ``"plateau"``       measured saturated shear strength [Pa] at
                        ``normal_pressure_pa``;
    ``"ratio"``         dimensionless ratio of two measured strengths;
    ``"time_form"``     qualitative statement about the time dependence
                        (``value`` is the contact period [s] at which the
                        observation was made).
    """

    key: str
    kind: str
    medium: str
    salinity_percent: float
    value: float
    normal_pressure_pa: float | None = None
    source: str = ""
    calibration: bool = False
    note: str = ""


#: Ettema & Schaefer (1986) data used by this module.  ``calibration=True``
#: entries fix the two strength coefficients of the model; the rest are
#: *holdout* data that the calibrated model must predict.
REFERENCE_DATA: tuple[EttemaSchaeferDatum, ...] = (
    EttemaSchaeferDatum(
        key="distilled_k", kind="coefficient", medium="distilled water",
        salinity_percent=0.0, value=2.01, source="Eq. (5), p. 400",
        calibration=True, note="fixes water_coefficient"),
    EttemaSchaeferDatum(
        key="tap_k", kind="coefficient", medium="tap water",
        salinity_percent=0.0, value=1.95, source="Eq. (5), p. 400",
        note="holdout: fresh tap water is predicted with S = 0"),
    EttemaSchaeferDatum(
        key="air_k", kind="coefficient", medium="air (dry contact)",
        salinity_percent=0.0, value=0.48, source="Eq. (5), p. 400",
        calibration=True, note="fixes dry_coefficient"),
    EttemaSchaeferDatum(
        key="saline_3pct_plateau", kind="plateau", medium="3 % NaCl solution",
        salinity_percent=3.0, value=650.0, normal_pressure_pa=450.0,
        source="p. 401 text and Fig. 11",
        note="normal pressure 450 Pa per text p. 401; Fig. 11 caption says 3.8 kPa"),
    EttemaSchaeferDatum(
        key="saline_12p5_plateau", kind="plateau", medium="12.5 % NaCl solution",
        salinity_percent=12.5, value=200.0, normal_pressure_pa=450.0,
        source="p. 401 text and Fig. 12"),
    EttemaSchaeferDatum(
        key="saline_25_plateau", kind="plateau", medium="25 % NaCl solution",
        salinity_percent=25.0, value=200.0, normal_pressure_pa=450.0,
        source="p. 401 text and Fig. 12"),
    EttemaSchaeferDatum(
        key="water_over_air_ratio", kind="ratio", medium="water vs air",
        salinity_percent=0.0, value=4.0,
        source="p. 401 'about four times stronger'",
        note="model ratio k_water / k_air = 2.01 / 0.48 = 4.19"),
    EttemaSchaeferDatum(
        key="air_asymptote_time", kind="time_form", medium="air (dry contact)",
        salinity_percent=0.0, value=180.0,
        source="Fig. 7 and p. 400 (asymptote 450.4 Pa within 3 min)",
        note="bond must be saturated to >= 90 % at 180 s"),
    EttemaSchaeferDatum(
        key="water_linear_over_3min", kind="time_form",
        medium="distilled / tap / river water", salinity_percent=0.0, value=180.0,
        source="Fig. 7 and p. 400 (linear growth up to 3 min)",
        note="bond must still be in the near-linear regime at 180 s (< 50 % of asymptote)"),
    EttemaSchaeferDatum(
        key="saline_12p5_no_time_growth", kind="time_form",
        medium="12.5 / 25 % NaCl solution", salinity_percent=12.5, value=180.0,
        source="p. 401 (strength did not increase with time)",
        note="bond must be within 10 % of its asymptote already at 180 s"),
)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SaturationFreezeBondModel:
    """Saturating freeze-bond strength law anchored on Ettema & Schaefer (1986).

    ``sigma_b(t) = k(S) * sigma_n * phi(dT) * (1 - exp(-t / tau_g))``

    with ``k(S) = k_w (1 - beta min(S, S_cap))`` for bonds grown in water,
    ``k(S) = k_dry`` for dry contacts, ``phi(dT) = min(1 + (dT/dT_*)**n, cap)``
    and ``tau_g = tau_ref exp(-S / S_tau) exp(Q/R (1/T - 1/T_ref))``.
    """

    water_coefficient: float = 2.01
    dry_coefficient: float = 0.48
    salinity_slope_per_percent: float = 0.06
    salinity_cap_percent: float = 12.5
    shear_to_tension_ratio: float = 0.5
    wet_growth_time_s: float = 300.0
    dry_growth_time_s: float = 60.0
    salinity_growth_decay_per_percent: float = 3.0
    apparent_activation_energy_J_mol: float = DEFAULT_APPARENT_ACTIVATION_ENERGY
    reference_temperature_K: float = FREEZING_POINT_K
    hardening_scale_K: float = 10.0
    hardening_exponent: float = 0.5
    hardening_cap: float = 4.0
    consolidation_time_s: float = 86400.0

    def __post_init__(self) -> None:
        for name in ("water_coefficient", "dry_coefficient", "salinity_cap_percent",
                     "shear_to_tension_ratio", "wet_growth_time_s", "dry_growth_time_s",
                     "salinity_growth_decay_per_percent",
                     "apparent_activation_energy_J_mol", "reference_temperature_K",
                     "hardening_scale_K", "hardening_exponent", "hardening_cap",
                     "consolidation_time_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self.salinity_slope_per_percent < 0.0:
            raise ValueError("salinity_slope_per_percent must be >= 0")
        if self.salinity_slope_per_percent * self.salinity_cap_percent >= 1.0:
            raise ValueError("salinity coefficient would reach zero before the cap")
        if not 0.0 < self.shear_to_tension_ratio <= 1.0:
            raise ValueError("shear_to_tension_ratio must be in (0, 1]")

    # ------------------------------------------------------------- helpers
    def freezing_temperature_K(self, salinity_percent: float = 0.0) -> float:
        """Freezing point of the pore water [K] (linear Assur approximation)."""
        s = float(salinity_percent)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("salinity_percent must be finite and >= 0")
        return FREEZING_POINT_K - SEAWATER_FREEZING_DEPRESSION_K_PER_PPT * s

    def temperature_drop_K(self, temperature_K: float, salinity_percent: float = 0.0) -> float:
        """Sub-freezing temperature drop ``dT = T_f(S) - T`` [K], clipped at 0."""
        if not math.isfinite(temperature_K):
            raise ValueError("temperature_K must be finite")
        return max(self.freezing_temperature_K(salinity_percent) - float(temperature_K), 0.0)

    def salinity_coefficient(self, salinity_percent: float = 0.0, *, dry: bool = False) -> float:
        """``k(S)``: shear strength per unit normal pressure (dimensionless)."""
        if dry:
            return self.dry_coefficient
        s = float(salinity_percent)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("salinity_percent must be finite and >= 0")
        s_eff = min(s, self.salinity_cap_percent)
        return self.water_coefficient * (1.0 - self.salinity_slope_per_percent * s_eff)

    def temperature_factor(self, delta_T_K: float) -> float:
        """``phi(dT)``: hardening of the saturated bond in colder pore water."""
        dT = float(delta_T_K)
        if not math.isfinite(dT) or dT < 0.0:
            raise ValueError("delta_T_K must be finite and >= 0")
        return min(1.0 + (dT / self.hardening_scale_K) ** self.hardening_exponent,
                   self.hardening_cap)

    def asymptotic_shear_strength(self, normal_pressure_pa: float, *,
                                  salinity_percent: float = 0.0,
                                  temperature_K: float = FREEZING_POINT_K,
                                  dry: bool = False) -> float:
        """``sigma_b,inf`` in shear [Pa]: the fully saturated freeze bond."""
        if not math.isfinite(normal_pressure_pa) or normal_pressure_pa < 0.0:
            raise ValueError("normal_pressure_pa must be finite and >= 0")
        dT = self.temperature_drop_K(temperature_K, salinity_percent)
        return (self.salinity_coefficient(salinity_percent, dry=dry)
                * float(normal_pressure_pa) * self.temperature_factor(dT))

    def asymptotic_tensile_strength(self, normal_pressure_pa: float, *,
                                    salinity_percent: float = 0.0,
                                    temperature_K: float = FREEZING_POINT_K,
                                    dry: bool = False) -> float:
        """``sigma_b,inf`` in tension [Pa] = ``kappa`` times the shear value."""
        return self.shear_to_tension_ratio * self.asymptotic_shear_strength(
            normal_pressure_pa, salinity_percent=salinity_percent,
            temperature_K=temperature_K, dry=dry)

    def growth_time_s(self, *, salinity_percent: float = 0.0,
                      temperature_K: float = FREEZING_POINT_K,
                      dry: bool = False) -> float:
        """``tau_g``: e-folding growth time of the bond [s].

        Arrhenius in temperature (colder pore water -> slower sintering) and
        exponentially shorter in brine (the brine film freezes in seconds, which
        is what Ettema & Schaefer saw at >= 12.5 % NaCl).
        """
        temperature = float(temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        s = float(salinity_percent)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("salinity_percent must be finite and >= 0")
        tau_ref = self.dry_growth_time_s if dry else self.wet_growth_time_s
        arrhenius = math.exp(self.apparent_activation_energy_J_mol / GAS_CONSTANT
                             * (1.0 / temperature - 1.0 / self.reference_temperature_K))
        salinity_decay = 1.0 if dry else math.exp(
            -s / self.salinity_growth_decay_per_percent)
        return tau_ref * arrhenius * salinity_decay

    # ------------------------------------------------------------ the law
    def _check_age(self, age_s) -> np.ndarray:
        arr = np.asarray(age_s, dtype=np.float64)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError("age_s must be finite and >= 0")
        return arr

    def _saturation(self, age_s, tau: float) -> np.ndarray:
        return 1.0 - np.exp(-self._check_age(age_s) / float(tau))

    def shear_strength(self, age_s, normal_pressure_pa: float, *,
                       salinity_percent: float = 0.0,
                       temperature_K: float = FREEZING_POINT_K,
                       dry: bool = False):
        """``sigma_b(t)`` in shear [Pa] -- the quantity E&S measured."""
        sigma_inf = self.asymptotic_shear_strength(
            normal_pressure_pa, salinity_percent=salinity_percent,
            temperature_K=temperature_K, dry=dry)
        tau = self.growth_time_s(salinity_percent=salinity_percent,
                                 temperature_K=temperature_K, dry=dry)
        return sigma_inf * self._saturation(age_s, tau)

    def tensile_strength(self, age_s, normal_pressure_pa: float, *,
                         salinity_percent: float = 0.0,
                         temperature_K: float = FREEZING_POINT_K,
                         dry: bool = False):
        """``sigma_b(t)`` in tension [Pa] = ``kappa`` times the shear strength."""
        return self.shear_to_tension_ratio * self.shear_strength(
            age_s, normal_pressure_pa, salinity_percent=salinity_percent,
            temperature_K=temperature_K, dry=dry)

    def tensile_rate(self, age_s, normal_pressure_pa: float, *,
                     salinity_percent: float = 0.0,
                     temperature_K: float = FREEZING_POINT_K,
                     dry: bool = False):
        """``d sigma_b / dt`` in tension [Pa/s]."""
        sigma_inf = self.asymptotic_tensile_strength(
            normal_pressure_pa, salinity_percent=salinity_percent,
            temperature_K=temperature_K, dry=dry)
        tau = self.growth_time_s(salinity_percent=salinity_percent,
                                 temperature_K=temperature_K, dry=dry)
        return sigma_inf / tau * np.exp(-self._check_age(age_s) / tau)

    def bonded_fraction(self, age_s):
        """Fraction of rubbing contacts that carry a freeze bond [-]."""
        return self._saturation(age_s, self.consolidation_time_s)


# ---------------------------------------------------------------------------
# validation against the imported experiment
# ---------------------------------------------------------------------------
def _time_form_saturation(model: SaturationFreezeBondModel, datum: EttemaSchaeferDatum) -> float:
    """Fraction of the asymptote reached at ``datum.value`` seconds (1 kPa ref.)."""
    reference_pressure = 1000.0
    dry = "air" in datum.medium
    sigma_inf = model.asymptotic_shear_strength(
        reference_pressure, salinity_percent=datum.salinity_percent,
        temperature_K=model.reference_temperature_K, dry=dry)
    sigma = float(model.shear_strength(datum.value, reference_pressure,
                                       salinity_percent=datum.salinity_percent,
                                       temperature_K=model.reference_temperature_K,
                                       dry=dry))
    return sigma / sigma_inf


def validate_against_ettema_schaefer(model: SaturationFreezeBondModel | None = None, *,
                                     tolerance: float = 0.20,
                                     normal_pressure_override: float | None = None,
                                     data: Sequence[EttemaSchaeferDatum] = REFERENCE_DATA
                                     ) -> dict:
    """Score ``model`` against Ettema & Schaefer (1986) without re-tuning.

    Only ``distilled_k`` and ``air_k`` are calibration anchors (they fix
    ``water_coefficient`` and ``dry_coefficient``); every other entry is a
    holdout.  ``normal_pressure_override`` replaces the contact pressure of the
    saline-plateau data, which exposes the p. 401 text / Fig. 11 caption
    ambiguity; ``None`` keeps the text value of 450 Pa.
    """
    model = SaturationFreezeBondModel() if model is None else model
    rows: list[dict] = []
    for datum in data:
        sigma_n = datum.normal_pressure_pa
        if normal_pressure_override is not None and datum.kind == "plateau":
            sigma_n = float(normal_pressure_override)
        if datum.kind == "coefficient":
            predicted = model.salinity_coefficient(datum.salinity_percent,
                                                   dry="air" in datum.medium)
            measured = datum.value
            comparison = "predicts k = sigma_b / sigma_n"
            ok = abs(predicted - measured) / abs(measured) <= tolerance
        elif datum.kind == "plateau":
            predicted = model.asymptotic_shear_strength(
                sigma_n, salinity_percent=datum.salinity_percent,
                temperature_K=model.reference_temperature_K)
            measured = datum.value
            comparison = f"predicts sigma_b,inf at sigma_n = {sigma_n:g} Pa"
            ok = abs(predicted - measured) / abs(measured) <= tolerance
        elif datum.kind == "ratio":
            predicted = model.water_coefficient / model.dry_coefficient
            measured = datum.value
            comparison = "predicts k_water / k_air"
            ok = abs(predicted - measured) / abs(measured) <= tolerance
        else:  # time_form: reported as a one-sided bound
            saturation = _time_form_saturation(model, datum)
            if datum.key == "water_linear_over_3min":
                measured, predicted = 0.50, saturation
                comparison = "saturation at 180 s must be <= 0.50 (still linear)"
                ok = saturation <= measured
            else:
                measured, predicted = 0.90, saturation
                comparison = "saturation at 180 s must be >= 0.90 (asymptote reached)"
                ok = saturation >= measured
        rel = abs(predicted - measured) / abs(measured) if measured else math.inf
        rows.append({
            "key": datum.key,
            "kind": datum.kind,
            "medium": datum.medium,
            "salinity_percent": datum.salinity_percent,
            "normal_pressure_pa": sigma_n,
            "measured": float(measured),
            "predicted": float(predicted),
            "rel_error": float(rel),
            "tolerance": tolerance,
            "pass": bool(ok),
            "role": "calibration" if datum.calibration else "holdout",
            "source": datum.source,
            "note": datum.note,
            "comparison": comparison,
        })
    holdout = [r for r in rows if r["role"] == "holdout"]
    return {
        "rows": rows,
        "n_data": len(rows),
        "n_calibration": sum(1 for r in rows if r["role"] == "calibration"),
        "n_holdout": len(holdout),
        "max_rel_error": max(r["rel_error"] for r in rows),
        "max_holdout_rel_error": max((r["rel_error"] for r in holdout), default=math.nan),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "all_pass": all(r["pass"] for r in rows),
        "tolerance": tolerance,
        "normal_pressure_override_pa": normal_pressure_override,
        "source": ("Ettema & Schaefer (1986), J. Glaciology 32(112):397-403, "
                   "doi:10.3189/S0022143000012107"),
    }


# ---------------------------------------------------------------------------
# keel resistance built on the bond strength
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgingState:
    """Everything the aging model gives a DEM material at one age."""

    age_s: float
    sigma_b_shear_Pa: float
    sigma_b_tension_Pa: float
    bonded_fraction: float
    growth_time_s: float
    normal_pressure_Pa: float
    salinity_percent: float
    temperature_K: float

    def as_dict(self) -> dict:
        return {
            "age_s": self.age_s,
            "age_hours": self.age_s / 3600.0,
            "sigma_b_shear_Pa": self.sigma_b_shear_Pa,
            "sigma_b_tension_Pa": self.sigma_b_tension_Pa,
            "bonded_fraction": self.bonded_fraction,
            "growth_time_s": self.growth_time_s,
            "normal_pressure_Pa": self.normal_pressure_Pa,
            "salinity_percent": self.salinity_percent,
            "temperature_K": self.temperature_K,
        }


def aging_state(model: SaturationFreezeBondModel, age_s: float, *,
                normal_pressure_Pa: float,
                salinity_percent: float = 0.0,
                temperature_K: float = FREEZING_POINT_K,
                dry: bool = False) -> AgingState:
    """Evaluate the aging model at one age (scalar in, :class:`AgingState` out)."""
    age = float(age_s)
    if not math.isfinite(age) or age < 0.0:
        raise ValueError("age_s must be finite and >= 0")
    return AgingState(
        age_s=age,
        sigma_b_shear_Pa=float(model.shear_strength(
            age, normal_pressure_Pa, salinity_percent=salinity_percent,
            temperature_K=temperature_K, dry=dry)),
        sigma_b_tension_Pa=float(model.tensile_strength(
            age, normal_pressure_Pa, salinity_percent=salinity_percent,
            temperature_K=temperature_K, dry=dry)),
        bonded_fraction=float(model.bonded_fraction(age)),
        growth_time_s=float(model.growth_time_s(
            salinity_percent=salinity_percent, temperature_K=temperature_K, dry=dry)),
        normal_pressure_Pa=float(normal_pressure_Pa),
        salinity_percent=float(salinity_percent),
        temperature_K=float(temperature_K),
    )


def aged_material(material: RidgeMaterial, model: SaturationFreezeBondModel, age_s: float, *,
                  normal_pressure_Pa: float,
                  salinity_percent: float = 0.0,
                  temperature_K: float = FREEZING_POINT_K,
                  dry: bool = False,
                  floor_strength_Pa: float = 1.0) -> tuple[RidgeMaterial, AgingState]:
    """Return the :class:`~tensordem.ridge_dem.RidgeMaterial` of an aged keel.

    The bond tensile/shear strengths are replaced by the aging law and the
    bonded-contact fraction by ``bonded_fraction(age)`` (a freshly formed keel
    has no bonds at all, a fully consolidated one bonds every contact).  A floor
    keeps the DEM ductility guard satisfied at ``t -> 0``, where the physical
    strength is zero; the *bond-free* case is expressed by the returned bonded
    fraction, not by a zero-strength bond.
    """
    if not isinstance(material, RidgeMaterial):
        raise TypeError("material must be a RidgeMaterial")
    state = aging_state(model, age_s, normal_pressure_Pa=normal_pressure_Pa,
                        salinity_percent=salinity_percent,
                        temperature_K=temperature_K, dry=dry)
    tensile = max(state.sigma_b_tension_Pa, floor_strength_Pa)
    shear = max(state.sigma_b_shear_Pa, floor_strength_Pa)
    if tensile / material.bond_stiffness > 0.05:
        raise ValueError(
            "aged bond too ductile for the explicit DEM: "
            f"sigma_b/k = {tensile / material.bond_stiffness:.3g} > 0.05; "
            "raise bond_stiffness for the aging study")
    aged = replace(material, bond_tensile_strength=tensile, bond_shear_strength=shear,
                   freeze_bond_fraction=min(max(state.bonded_fraction, 0.0), 1.0))
    return aged, state


@dataclass(frozen=True)
class KeelAgingResistance:
    """Lindqvist-type keel resistance with an ageing cohesive term [N/m].

    ``R_keel(t) = mu * W' + chi * sigma_b,shear(t) * A_c``

    * ``W'`` is the buoyant weight of the keel per metre of ridge width,
      ``(rho_w - rho_i) g H^2 / 2`` -- it does not age;
    * ``mu = tan(phi_0)`` is the *bond-free* rubble friction coefficient,
      measured by the DEM biaxial cell at ``t -> 0``;
    * ``chi`` is the micro->macro cohesion transfer coefficient of the DEM cell,
      measured at the saturated age; ``chi * sigma_b`` is therefore the keel
      macro cohesion at that age, scaled by the bonded fraction for younger
      keels;
    * ``A_c = contact_factor * H`` is the contact area per metre of width that
      has to be sheared for the keel to fail.

    At ``t = 0`` both factors vanish (no bonds, no cohesion) and the resistance
    degenerates to ``mu * W'``: the unbonded-keel resistance.
    """

    keel_depth_m: float = 10.0
    width_m: float = 1.0
    ice_density: float = 917.0
    water_density: float = 1025.0
    gravity: float = 9.81
    friction_coefficient: float = 0.45
    cohesion_transfer: float = 1.0
    contact_factor: float = 1.0
    porosity: float = 0.30
    salinity_percent: float = 0.0
    temperature_K: float = 263.15
    dry: bool = False

    def __post_init__(self) -> None:
        for name in ("keel_depth_m", "width_m", "ice_density", "water_density",
                     "gravity", "friction_coefficient", "cohesion_transfer",
                     "contact_factor", "porosity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.keel_depth_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("keel_depth_m and width_m must be positive")
        if self.ice_density > self.water_density:
            raise ValueError("the keel must be buoyant: ice_density <= water_density")
        if not 0.0 <= self.porosity < 1.0:
            raise ValueError("porosity must be in [0, 1)")

    @property
    def normal_pressure_Pa(self) -> float:
        """Hydrostatic contact pressure at mid-depth of the keel [Pa]."""
        return (self.water_density - self.ice_density) * self.gravity * 0.5 * self.keel_depth_m

    @property
    def buoyant_weight_per_m(self) -> float:
        """Buoyant weight of the keel per metre of ridge width [N/m]."""
        return ((self.water_density - self.ice_density) * self.gravity
                * (1.0 - self.porosity) * 0.5 * self.keel_depth_m ** 2 * self.width_m)

    @property
    def contact_area_m2(self) -> float:
        """Areal resistance that has to be sheared [m^2 per metre of width]."""
        return self.contact_factor * self.keel_depth_m * self.width_m

    def series(self, model: SaturationFreezeBondModel, ages_s) -> dict:
        """Resistance history ``R_keel(t)`` with its components."""
        ages = np.asarray(ages_s, dtype=np.float64)
        if np.any(~np.isfinite(ages)) or np.any(ages < 0.0):
            raise ValueError("ages_s must be finite and >= 0")
        sigma = np.asarray(model.shear_strength(
            ages, self.normal_pressure_Pa, salinity_percent=self.salinity_percent,
            temperature_K=self.temperature_K, dry=self.dry), dtype=np.float64)
        bonded = np.asarray(model.bonded_fraction(ages), dtype=np.float64)
        cohesion = self.cohesion_transfer * sigma * bonded
        friction = self.friction_coefficient * self.buoyant_weight_per_m
        cohesive_resistance = cohesion * self.contact_area_m2
        total = friction + cohesive_resistance
        return {
            "ages_s": ages,
            "ages_hours": ages / 3600.0,
            "sigma_b_shear_Pa": sigma,
            "bonded_fraction": bonded,
            "cohesion_Pa": cohesion,
            "rubble_friction_N_per_m": np.full_like(ages, friction),
            "cohesive_N_per_m": cohesive_resistance,
            "total_N_per_m": total,
            "normal_pressure_Pa": self.normal_pressure_Pa,
            "buoyant_weight_N_per_m": self.buoyant_weight_per_m,
            "contact_area_m2": self.contact_area_m2,
        }

    def degenerate_resistance_N_per_m(self) -> float:
        """``R_keel(0)``: the unbonded (bond-free) keel resistance [N/m]."""
        return self.friction_coefficient * self.buoyant_weight_per_m

    def summary(self, model: SaturationFreezeBondModel, ages_s) -> dict:
        """Monotonicity / degeneration diagnostics used by the acceptance checks."""
        series = self.series(model, ages_s)
        total = series["total_N_per_m"]
        cohesion = series["cohesive_N_per_m"]
        diffs = np.diff(total)
        return {
            "R_at_0_N_per_m": float(total[0]),
            "R_degenerate_N_per_m": float(self.degenerate_resistance_N_per_m()),
            "t0_degeneration_rel_error": float(
                abs(total[0] - self.degenerate_resistance_N_per_m())
                / max(self.degenerate_resistance_N_per_m(), 1e-12)),
            "R_final_N_per_m": float(total[-1]),
            "R_growth_factor": float(total[-1] / max(total[0], 1e-12)),
            "monotone_increasing": bool(np.all(diffs >= -1e-9)),
            "cohesive_dominates_at_infinity": bool(cohesion[-1] >= series["rubble_friction_N_per_m"][0]),
            "sigma_b_at_0_Pa": float(series["sigma_b_shear_Pa"][0]),
            "sigma_b_final_Pa": float(series["sigma_b_shear_Pa"][-1]),
            "normal_pressure_Pa": series["normal_pressure_Pa"],
        }


def cohesion_transfer_coefficient(dem_cohesion_Pa: float,
                                 sigma_b_shear_Pa: float) -> float:
    """``chi = c_DEM / sigma_b``: micro->macro cohesion transfer of the DEM cell.

    The biaxial cell measures one macro cohesion ``c`` for a known bond strength
    ``sigma_b``; the resulting ``chi`` is the factor that converts a bond
    strength into a keel-scale cohesion, and it is *measured*, not assumed.
    """
    for name, value in (("dem_cohesion_Pa", dem_cohesion_Pa),
                        ("sigma_b_shear_Pa", sigma_b_shear_Pa)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(dem_cohesion_Pa) / float(sigma_b_shear_Pa)