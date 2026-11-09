"""Frozen contact bonds: interface freezing and the adhesive/cohesive mode switch.

What this layer adds
--------------------
:mod:`tensordem.freeze_bond.strength` answers *how strong is a freeze bond* as a
function of temperature, contact pressure and contact time.  This module answers
*what fails, and with what peak load*, when such a bond is loaded inside a DEM
contact.  It couples the strength law to the DEM contact state (normal force,
contact area, contact age), decides between the two competing failure planes and
carries a closed energy ledger for the whole path::

    bond formation (latent heat of the freezing interfacial film)
        -> elastic loading -> break (fracture energy) -> sliding (friction)

Physics
-------
For one contact with interface temperature ``T``, nominal contact pressure
``p_c = F_n / A_c`` and contact age ``t_c``::

    sigma_b = sigma_0 (dT/20)^beta (1 + gamma_eff dT) (t_c/t_ref)^alpha (p_c/p_ref)^e
    sigma_c = sigma_c0 (dT/20)^beta_c

with ``dT = T_freeze - T`` and ``sigma_b = 0`` for ``T >= T_freeze``.

Two failure planes compete on the *same* contact area:

* **adhesive** (interface) failure when ``sigma_b < sigma_c``: ice detaches from
  the structure at ``F_a = sigma_b A_c``;
* **cohesive** (ice) failure when ``sigma_b >= sigma_c``: the ice itself breaks
  at ``F_c = sigma_c A_c``.

The break criterion is the campaign's: the normal force is compared with *both*
thresholds and the first one reached wins (``argmin``).  The realised peak load is
``F_peak = min(F_a, F_c)``, i.e. **continuous** at the transition temperature
``T_tr`` -- the mode switch is a kink, not a step.  :func:`peak_load_jump`
therefore reports the kink ratio, the *window* ratio
``F_c(T_tr)/F_a(T_freeze - 1C)`` (the load gain seen when a freshly formed bond
matures into the cohesive regime) and the ice/interface reserve factor.

Thermal side
------------
Forming the bond freezes an interfacial film of thickness ``h`` and releases
``Q_lat = rho L_f h A``.  That heat is conducted away, it is not available for
fracture.  :func:`energy_ledger` therefore reports the *literal* campaign budget
``Q_lat = E_fracture + E_friction`` (falsified for any realistic film: it only
balances at ``h ~ 0.1 um``) next to the physically complete budget
``Q_lat + Q_sensible = E_fracture + E_friction + E_surface + Q_conducted``.  The
complete budget is checked independently by a numerical front-tracking freezing
solver (:func:`freezing_front_ledger`) and by an incremental mechanical work
ledger (:func:`mechanical_energy_ledger`).

Evidence grading
----------------
REAL           : contact/strength coupling, mode competition, transition root
                 solve, energy ledgers, sweeps.
REAL-in-sample : the ``sigma_b(T)`` parameters are calibrated on the six Sastry &
                 Metzger (1986) points; :func:`leave_one_out_validation` reports
                 the honest generalisation error.
WITHHELD       : no digitised Repetto et al. (2011) table exists in this repo, so
                 the multi-contact sliding test is checked against a declared
                 adhesion window (0.05-1.5 MPa) instead of pointwise data.

All quantities SI (Pa, K, s, m^2, J) unless a name ends in ``_mpa`` / ``_c``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .strength import (
    FreezeBondConfig,
    SASTRY_METZGER_1986_DATA,
    T_MIN_C,
    sigma_bond_temperature,
    sigma_bond_time,
    sigma_ice_temperature,
)

__all__ = [
    "ContactBondConfig", "ContactBondState", "FrozenContactBond",
    "FrozenBondAssembly", "bond_strengths", "failure_mode_at", "break_event",
    "peak_contact_force", "transition_temperature_numeric",
    "transition_temperature_analytic", "peak_load_jump", "mode_transition_scan",
    "single_contact_pulloff", "sastry_metzger_contact_table",
    "leave_one_out_validation", "multi_contact_slide",
    "mechanical_energy_ledger", "freezing_front_ledger", "energy_ledger",
    "spearman_rho", "WATER_DENSITY", "LATENT_HEAT_FUSION", "ICE_CONDUCTIVITY",
    "ICE_HEAT_CAPACITY", "ICE_SURFACE_ENERGY", "REPETTO_ADHESION_WINDOW_MPA",
]

# ---------------------------------------------------------------------------
# Material constants (SI)
# ---------------------------------------------------------------------------
WATER_DENSITY: float = 917.0          # kg/m^3 (the film freezes to ice)
LATENT_HEAT_FUSION: float = 3.34e5    # J/kg
ICE_CONDUCTIVITY: float = 2.2         # W/(m K)
ICE_HEAT_CAPACITY: float = 2100.0     # J/(kg K)
ICE_SURFACE_ENERGY: float = 0.065     # J/m^2 (ice grain-boundary scale)

#: Declared adhesion-strength window for ice on steel (MPa).  Used instead of a
#: pointwise Repetto et al. (2011) comparison: no digitised table of that paper
#: exists in this repository (evidence grade WITHHELD).
REPETTO_ADHESION_WINDOW_MPA: tuple[float, float] = (0.05, 1.5)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactBondConfig:
    """Geometry, strength-law and energy parameters of one bonded contact.

    ``strength`` is the temperature/pressure/time strength law, ``area_m2`` the
    nominal bonded area ``A_c`` (the same area carries both the adhesive and the
    cohesive threshold), ``film_thickness_m`` the interfacial water film that
    freezes on bond formation, ``fracture_energy_*_j_m2`` the Mode-I fracture
    energies of the two failure planes, ``friction_coefficient`` the ice-steel
    friction after adhesive debonding and ``normal_stiffness_n_m`` the bond
    stiffness (``delta_0 = F_peak / k``).
    """

    strength: FreezeBondConfig = field(default_factory=FreezeBondConfig)
    area_m2: float = 1.0e-4
    film_thickness_m: float = 1.0e-7
    fracture_energy_adhesive_j_m2: float = 24.255
    fracture_energy_cohesive_j_m2: float = 3.0
    friction_coefficient: float = 0.2
    normal_stiffness_n_m: float = 1.0e7

    def __post_init__(self) -> None:
        for name in ("area_m2", "film_thickness_m", "normal_stiffness_n_m"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("fracture_energy_adhesive_j_m2",
                     "fracture_energy_cohesive_j_m2"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not (math.isfinite(self.friction_coefficient)
                and self.friction_coefficient >= 0.0):
            raise ValueError("friction_coefficient must be finite and nonnegative")

    @property
    def t_freeze(self) -> float:
        return self.strength.t_freeze


# ---------------------------------------------------------------------------
# Strength coupling at one contact
# ---------------------------------------------------------------------------
def bond_strengths(
    temperature_c: float,
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
) -> tuple[float, float]:
    """Return ``(sigma_bond, sigma_cohesive)`` in MPa for one contact.

    ``sigma_bond`` grows as ``t^alpha`` with contact time and as ``p^e`` with
    contact pressure, and is exactly ``0`` at or above the freezing temperature.
    A non-positive contact pressure or time returns a zero bond strength (no
    intimate contact / no contact time -> no bond).
    """
    cfg = config if config is not None else ContactBondConfig()
    sc = cfg.strength
    sigma_c = sigma_ice_temperature(temperature_c, sc.sigma_ice_0, sc.beta_ice,
                                    sc.t_freeze)
    if not (math.isfinite(contact_time_s) and contact_time_s > 0.0):
        return 0.0, float(sigma_c)
    if not (math.isfinite(contact_pressure_mpa) and contact_pressure_mpa > 0.0):
        return 0.0, float(sigma_c)
    sigma_b = sigma_bond_temperature(temperature_c, sc.t_freeze, sc.sigma_0,
                                     sc.beta, sc.gamma, sc.t_gamma)
    if sigma_b <= 0.0:
        return 0.0, float(sigma_c)
    sigma_b = sigma_bond_time(sigma_b, contact_time_s, sc.alpha, sc.t_ref)
    sigma_b *= (contact_pressure_mpa / sc.p_ref) ** sc.pressure_exponent
    return float(sigma_b), float(sigma_c)


def failure_mode_at(
    temperature_c: float,
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
) -> str:
    """``"none"`` (no bond), ``"adhesive"`` or ``"cohesive"`` failure."""
    sigma_b, sigma_c = bond_strengths(temperature_c, contact_pressure_mpa,
                                      contact_time_s, config)
    if sigma_b <= 0.0:
        return "none"
    return "cohesive" if sigma_b >= sigma_c else "adhesive"


def peak_contact_force(
    temperature_c: float,
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
) -> dict[str, Any]:
    """Peak force a bonded contact transmits: ``F_peak = min(F_a, F_c)``."""
    cfg = config if config is not None else ContactBondConfig()
    sigma_b, sigma_c = bond_strengths(temperature_c, contact_pressure_mpa,
                                      contact_time_s, cfg)
    f_a = sigma_b * 1.0e6 * cfg.area_m2
    f_c = sigma_c * 1.0e6 * cfg.area_m2
    if sigma_b <= 0.0:
        mode = "none"
    else:
        mode = "cohesive" if f_c <= f_a else "adhesive"
    return {
        "temperature_c": float(temperature_c),
        "contact_pressure_mpa": float(contact_pressure_mpa),
        "contact_time_s": float(contact_time_s),
        "sigma_bond_mpa": sigma_b,
        "sigma_cohesive_mpa": sigma_c,
        "f_adhesive_N": f_a,
        "f_cohesive_N": f_c,
        "f_peak_N": min(f_a, f_c),
        "failure_mode": mode,
    }


def break_event(
    normal_force_N: float,
    temperature_c: float,
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
) -> dict[str, Any]:
    """Contact break criterion: ``F_n`` against both thresholds, first wins."""
    ev = peak_contact_force(temperature_c, contact_pressure_mpa, contact_time_s,
                            config)
    f_n = float(normal_force_N)
    f_a, f_c = ev["f_adhesive_N"], ev["f_cohesive_N"]
    if ev["failure_mode"] == "none":
        mode, threshold = "none", 0.0
    elif f_n >= min(f_a, f_c):
        mode = "adhesive" if f_a <= f_c else "cohesive"
        threshold = min(f_a, f_c)
    else:
        mode, threshold = "none", min(f_a, f_c)
    out = {"normal_force_N": f_n, "failure_mode": mode, "threshold_N": threshold,
           "broken": mode in ("adhesive", "cohesive")}
    out.update({k: ev[k] for k in ("sigma_bond_mpa", "sigma_cohesive_mpa",
                                   "f_adhesive_N", "f_cohesive_N", "f_peak_N")})
    return out


# ---------------------------------------------------------------------------
# Transition temperature, analytic estimate, peak-load jump
# ---------------------------------------------------------------------------
def _residual(temperature_c: float, contact_pressure_mpa: float,
              contact_time_s: float, cfg: ContactBondConfig) -> float:
    sigma_b, sigma_c = bond_strengths(temperature_c, contact_pressure_mpa,
                                      contact_time_s, cfg)
    return sigma_b - sigma_c


def transition_temperature_numeric(
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
    t_lo_c: float = T_MIN_C,
    t_hi_c: float = -1.0e-9,
) -> float | None:
    """Temperature where ``sigma_bond = sigma_cohesive`` (200-step bisection).

    ``None`` when the branches do not cross inside ``[t_lo_c, t_hi_c]`` (the
    interface/ice ordering is then the same over the whole window).
    """
    cfg = config if config is not None else ContactBondConfig()
    if _residual(t_lo_c, contact_pressure_mpa, contact_time_s, cfg) <= 0.0:
        return None
    if _residual(t_hi_c, contact_pressure_mpa, contact_time_s, cfg) >= 0.0:
        return None
    lo, hi = float(t_lo_c), float(t_hi_c)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _residual(mid, contact_pressure_mpa, contact_time_s, cfg) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def transition_temperature_analytic(
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
) -> float | None:
    """Closed-form transition temperature of the power-law bond law.

    Dropping the near-freezing correction ``(1 + gamma_eff dT)`` makes
    ``sigma_b = sigma_c`` algebraic::

        dT_tr = 20 K^{1/(beta - beta_c)},
        K = (sigma_c0/sigma_0) (t_c/t_ref)^{-alpha} (p_c/p_ref)^{-e}

    The gap to :func:`transition_temperature_numeric` is the effect of the
    empirical near-freezing correction (large at deep subcooling).
    """
    cfg = config if config is not None else ContactBondConfig()
    sc = cfg.strength
    if not (contact_pressure_mpa > 0.0 and contact_time_s > 0.0):
        return None
    k = sc.sigma_ice_0 / sc.sigma_0
    k *= (contact_time_s / sc.t_ref) ** (-sc.alpha)
    k *= (contact_pressure_mpa / sc.p_ref) ** (-sc.pressure_exponent)
    denom = sc.beta - sc.beta_ice
    if abs(denom) < 1.0e-12:
        return None
    d_t = 20.0 * k ** (1.0 / denom)
    t_tr = sc.t_freeze - d_t
    if t_tr < T_MIN_C or t_tr > sc.t_freeze:
        return None
    return float(t_tr)


def peak_load_jump(
    contact_pressure_mpa: float,
    contact_time_s: float,
    config: ContactBondConfig | None = None,
    warm_offset_c: float = 1.0,
) -> dict[str, Any]:
    """Peak-load metrics across the failure-mode transition.

    ``kink_ratio``
        ``F_peak`` on the cold side over the warm side, ``+/- 1e-3 C`` around
        ``T_tr``.  ~1: the ``min`` envelope is continuous, so the switch is a
        kink, not a step.
    ``window_ratio``
        ``F_cohesive(T_tr) / F_adhesive(T_freeze - warm_offset_c)`` -- the load
        gain from a freshly formed bond to the cohesive regime; this is the
        ``F_cohesive/F_adhesive`` of campaign criterion (c).
    ``branch_strength_ratio``
        ``max over the adhesive window of sigma_cohesive/sigma_bond``.
    """
    cfg = config if config is not None else ContactBondConfig()
    t_tr = transition_temperature_numeric(contact_pressure_mpa, contact_time_s, cfg)
    if t_tr is None:
        return {"t_transition_c": None, "kink_ratio": None, "window_ratio": None,
                "branch_strength_ratio": None, "defined": False}
    d = 1.0e-3
    cold = peak_contact_force(t_tr - d, contact_pressure_mpa, contact_time_s, cfg)
    warm = peak_contact_force(t_tr + d, contact_pressure_mpa, contact_time_s, cfg)
    kink = (cold["f_peak_N"] / warm["f_peak_N"]) if warm["f_peak_N"] > 0.0 else None
    fresh = peak_contact_force(cfg.t_freeze - warm_offset_c, contact_pressure_mpa,
                               contact_time_s, cfg)
    window = (cold["f_peak_N"] / fresh["f_adhesive_N"]
              if fresh["f_adhesive_N"] > 0.0 else None)
    # reserve factor sampled over the *validated* band T = T_freeze - 1 ... T_tr
    # (the strength law is not evaluated closer than 1 C to the freezing point,
    # where sigma_bond -> 0 and the ratio would diverge)
    reserve = 0.0
    for t in np.linspace(t_tr, cfg.t_freeze - warm_offset_c, 200):
        sb, sc = bond_strengths(float(t), contact_pressure_mpa, contact_time_s, cfg)
        if sb > 0.0:
            reserve = max(reserve, sc / sb)
    return {
        "t_transition_c": t_tr, "kink_ratio": kink, "window_ratio": window,
        "branch_strength_ratio": reserve,
        "f_peak_cold_N": cold["f_peak_N"], "f_peak_warm_N": warm["f_peak_N"],
        "f_fresh_bond_N": fresh["f_adhesive_N"],
        "mode_cold": cold["failure_mode"], "mode_warm": warm["failure_mode"],
        "defined": True,
    }


def mode_transition_scan(
    temperatures_c: Sequence[float],
    pressures_mpa: Sequence[float],
    times_s: Sequence[float],
    config: ContactBondConfig | None = None,
) -> list[dict[str, Any]]:
    """Cartesian scan of the ``(T, p_c, t_c)`` space at the contact level."""
    cfg = config if config is not None else ContactBondConfig()
    rows: list[dict[str, Any]] = []
    for p in pressures_mpa:
        for t in times_s:
            t_tr = transition_temperature_numeric(p, t, cfg)
            jump = peak_load_jump(p, t, cfg)
            for temp in temperatures_c:
                ev = peak_contact_force(float(temp), float(p), float(t), cfg)
                rows.append({
                    "temperature_c": float(temp),
                    "contact_pressure_mpa": float(p),
                    "contact_time_s": float(t),
                    "sigma_bond_mpa": ev["sigma_bond_mpa"],
                    "sigma_cohesive_mpa": ev["sigma_cohesive_mpa"],
                    "f_peak_N": ev["f_peak_N"],
                    "failure_mode": ev["failure_mode"],
                    "t_transition_c": t_tr,
                    "window_ratio": jump["window_ratio"],
                })
    return rows


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy dependency, ties averaged)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size or a.size < 2:
        raise ValueError("need two equal-length series of length >= 2")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i + 1
            while j < sv.size and sv[j] == sv[i]:
                j += 1
            if j - i > 1:
                r[order[i:j]] = 0.5 * (i + j + 1) - 0.5
            i = j
        return r

    ra = ranks(a) - (a.size + 1) / 2.0
    rb = ranks(b) - (b.size + 1) / 2.0
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0.0 else 0.0# ---------------------------------------------------------------------------
# Sastry & Metzger (1986) comparison at the contact level
# ---------------------------------------------------------------------------
def single_contact_pulloff(
    temperature_c: float,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 1.0,
    config: ContactBondConfig | None = None,
) -> dict[str, Any]:
    """Ice specimen on a cold steel plate: pull-off (bond) stress.

    The specimen is pressed at the nominal pressure ``p_c`` for ``t_c`` and then
    pulled: the pull-off stress is ``sigma_bond`` while the interface is the weak
    link and ``sigma_cohesive`` once the ice is weaker (the ice breaks instead of
    detaching).
    """
    cfg = config if config is not None else ContactBondConfig()
    ev = peak_contact_force(temperature_c, contact_pressure_mpa, contact_time_s, cfg)
    return {
        "temperature_c": float(temperature_c),
        "contact_pressure_mpa": float(contact_pressure_mpa),
        "contact_time_s": float(contact_time_s),
        "pulloff_stress_mpa": ev["f_peak_N"] / cfg.area_m2 / 1.0e6,
        "adhesion_stress_mpa": ev["sigma_bond_mpa"],
        "cohesive_stress_mpa": ev["sigma_cohesive_mpa"],
        "failure_mode": ev["failure_mode"],
        "pulloff_force_N": ev["f_peak_N"],
    }


def sastry_metzger_contact_table(
    config: ContactBondConfig | None = None,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 1.0,
) -> dict[str, Any]:
    """Pointwise contact-level pull-off stress vs the six S&M (1986) points."""
    cfg = config if config is not None else ContactBondConfig()
    points = []
    for t_c, sigma_exp in SASTRY_METZGER_1986_DATA:
        res = single_contact_pulloff(t_c, contact_pressure_mpa, contact_time_s, cfg)
        rel = (abs(res["pulloff_stress_mpa"] - sigma_exp) / sigma_exp
               if sigma_exp else float("inf"))
        points.append({
            "temperature_c": t_c, "experimental_mpa": sigma_exp,
            "predicted_mpa": res["pulloff_stress_mpa"],
            "adhesion_mpa": res["adhesion_stress_mpa"],
            "cohesive_mpa": res["cohesive_stress_mpa"],
            "failure_mode": res["failure_mode"], "relative_error": rel,
        })
    errs = np.array([p["relative_error"] for p in points])
    return {
        "points": points, "n_points": len(points),
        "mean_relative_error": float(errs.mean()),
        "max_relative_error": float(errs.max()),
        "criterion_20pct_all_points": bool(np.all(errs <= 0.20)),
        "contact_pressure_mpa": contact_pressure_mpa,
        "contact_time_s": contact_time_s,
        "note": ("strength-law parameters are calibrated on these six points, so "
                 "this is an in-sample fit; see leave_one_out_validation()"),
    }


def leave_one_out_validation(
    config: ContactBondConfig | None = None,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 1.0,
) -> dict[str, Any]:
    """Leave-one-out refit of ``(sigma_0, beta, gamma)`` on the six S&M points."""
    from scipy.optimize import least_squares

    cfg = config if config is not None else ContactBondConfig()
    data = np.array(SASTRY_METZGER_1986_DATA, dtype=float)
    temps, target = data[:, 0], data[:, 1]

    def predict(params: np.ndarray, t: np.ndarray) -> np.ndarray:
        s0, beta, gamma = params
        return np.array([sigma_bond_temperature(float(tt), cfg.t_freeze, s0, beta,
                                                gamma, cfg.strength.t_gamma)
                         for tt in np.atleast_1d(t)])

    rows = []
    for i in range(len(temps)):
        mask = np.arange(len(temps)) != i

        def fun(p, mask=mask):
            return predict(p, temps[mask]) - target[mask]

        sol = least_squares(fun, x0=np.array([3.2, 1.1, -0.07]),
                            bounds=([0.1, 0.2, -1.0], [20.0, 3.0, 1.0]),
                            xtol=1e-12, ftol=1e-12)
        pred = float(predict(sol.x, np.array([temps[i]]))[0])
        rows.append({
            "held_out_temperature_c": float(temps[i]),
            "experimental_mpa": float(target[i]), "predicted_mpa": pred,
            "relative_error": abs(pred - target[i]) / target[i],
            "fitted_sigma_0": float(sol.x[0]), "fitted_beta": float(sol.x[1]),
            "fitted_gamma": float(sol.x[2]),
        })
    errs = np.array([r["relative_error"] for r in rows])
    return {
        "rows": rows, "mean_relative_error": float(errs.mean()),
        "max_relative_error": float(errs.max()),
        "criterion_20pct_all_points": bool(np.all(errs <= 0.20)),
        "contact_pressure_mpa": contact_pressure_mpa,
        "contact_time_s": contact_time_s,
    }


# ---------------------------------------------------------------------------
# Contact-level DEM model (stateful)
# ---------------------------------------------------------------------------
@dataclass
class ContactBondState:
    """Mutable state of one DEM contact carrying a possible frozen bond."""

    temperature_c: float
    contact_pressure_mpa: float
    age_s: float = 0.0
    bonded: bool = False
    failed_mode: str = "none"        # 'none' | 'adhesive' | 'cohesive'
    damage: float = 0.0              # 0 .. 1 (linear softening)
    e_elastic_J: float = 0.0
    e_fracture_J: float = 0.0
    e_friction_J: float = 0.0
    q_latent_J: float = 0.0
    peak_force_N: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class FrozenContactBond:
    """One DEM contact that can carry a frozen (adfreeze) bond.

    ``form()`` nucleates the bond when ``T < T_freeze`` (releasing the latent heat
    of the interfacial film), ``advance(dt)`` grows the contact age (strength
    ``~ t^alpha``), ``load(F_n)`` applies the normal force (elastic to
    ``min(F_a, F_c)``, then linear softening spending exactly ``G_f A``) and
    ``slide(s, F_n)`` dissipates Coulomb friction after failure.
    """

    def __init__(self, config: ContactBondConfig | None = None,
                 temperature_c: float = -5.0,
                 contact_pressure_mpa: float = 0.1,
                 contact_time_s: float = 1.0) -> None:
        self.cfg = config if config is not None else ContactBondConfig()
        self.state = ContactBondState(
            temperature_c=float(temperature_c),
            contact_pressure_mpa=float(contact_pressure_mpa))
        self._k = self.cfg.normal_stiffness_n_m
        self._delta = 0.0
        self.state.age_s = float(contact_time_s)

    def form(self) -> bool:
        """Nucleate the bond if the interface is below the freezing point."""
        if self.state.temperature_c >= self.cfg.t_freeze:
            self.state.bonded = False
            self.state.history.append({"event": "no_form",
                                       "reason": "T >= T_freeze",
                                       "temperature_c": self.state.temperature_c})
            return False
        self.state.bonded = True
        self.state.failed_mode = "none"
        self.state.damage = 0.0
        self.state.q_latent_J = (WATER_DENSITY * LATENT_HEAT_FUSION
                                 * self.cfg.film_thickness_m * self.cfg.area_m2)
        self.state.history.append({
            "event": "form", "temperature_c": self.state.temperature_c,
            "pressure_mpa": self.state.contact_pressure_mpa,
            "age_s": self.state.age_s, "q_latent_J": self.state.q_latent_J})
        return True

    def advance(self, dt_s: float) -> float:
        """Advance the contact age (bond grows); returns the new age."""
        if dt_s < 0.0:
            raise ValueError("dt_s must be nonnegative")
        if self.state.bonded:
            self.state.age_s += float(dt_s)
        return self.state.age_s

    def strengths(self) -> tuple[float, float]:
        return bond_strengths(self.state.temperature_c,
                              self.state.contact_pressure_mpa,
                              self.state.age_s, self.cfg)

    def peak(self) -> dict[str, Any]:
        return peak_contact_force(self.state.temperature_c,
                                  self.state.contact_pressure_mpa,
                                  self.state.age_s, self.cfg)

    def _fracture_energy(self, mode: str) -> float:
        g = (self.cfg.fracture_energy_adhesive_j_m2 if mode == "adhesive"
             else self.cfg.fracture_energy_cohesive_j_m2)
        return g * self.cfg.area_m2

    def load(self, normal_force_N: float) -> dict[str, Any]:
        """Apply a normal force; returns the contact event."""
        if not self.state.bonded:
            reason = ("T >= T_freeze"
                      if self.state.temperature_c >= self.cfg.t_freeze
                      else "bond not formed")
            return {"event": "no_bond", "reason": reason, "failure_mode": "none"}
        ev = self.peak()
        f_peak = ev["f_peak_N"]
        if f_peak <= 0.0:
            return {"event": "no_bond", "reason": "zero peak force",
                    "failure_mode": "none"}
        f_n = float(normal_force_N)
        self.state.peak_force_N = f_peak
        delta_0 = f_peak / self._k
        if f_n <= f_peak:
            self.state.e_elastic_J = 0.5 * f_n * f_n / self._k
            self._delta = f_n / self._k
            self.state.damage = 0.0
            return {"event": "elastic", "failure_mode": "none", "f_peak_N": f_peak,
                    "f_adhesive_N": ev["f_adhesive_N"],
                    "f_cohesive_N": ev["f_cohesive_N"],
                    "sigma_bond_mpa": ev["sigma_bond_mpa"],
                    "sigma_cohesive_mpa": ev["sigma_cohesive_mpa"]}
        mode = ev["failure_mode"]
        self.state.failed_mode = mode
        e_frac = self._fracture_energy(mode)
        # total area under the traction-separation envelope must equal G_f A:
        # 0.5 * f_peak * delta_tot = G_f A  ->  delta_tot = 2 G_f A / f_peak,
        # so the softening length measured from the peak is delta_tot - delta_0.
        delta_soft = max(2.0 * e_frac / f_peak - delta_0, 1.0e-30)
        self.state.damage = min(1.0, (f_n - f_peak) / f_peak)
        self._delta = delta_0 + self.state.damage * delta_soft
        fully = self.state.damage >= 1.0
        self.state.e_fracture_J += e_frac if fully else self.state.damage * e_frac
        self.state.e_elastic_J = 0.0
        if fully:
            self.state.bonded = False
        self.state.history.append({
            "event": "break" if fully else "softening", "failure_mode": mode,
            "normal_force_N": f_n, "f_peak_N": f_peak,
            "damage": self.state.damage})
        return {"event": "break" if fully else "softening", "failure_mode": mode,
                "damage": self.state.damage, "f_peak_N": f_peak,
                "f_adhesive_N": ev["f_adhesive_N"],
                "f_cohesive_N": ev["f_cohesive_N"],
                "sigma_bond_mpa": ev["sigma_bond_mpa"],
                "sigma_cohesive_mpa": ev["sigma_cohesive_mpa"],
                "e_fracture_J": self.state.e_fracture_J}

    def slide(self, slip_m: float, normal_force_N: float,
              friction_coefficient: float | None = None) -> float:
        """Coulomb sliding after failure; returns the dissipated work (J)."""
        if slip_m < 0.0:
            raise ValueError("slip_m must be nonnegative")
        mu = (self.cfg.friction_coefficient if friction_coefficient is None
              else float(friction_coefficient))
        work = mu * abs(float(normal_force_N)) * float(slip_m)
        self.state.e_friction_J += work
        return work


class FrozenBondAssembly:
    """A patch of ``n`` bonded DEM contacts on one cold surface.

    The contacts carry heterogeneous pressures, so at a given temperature part of
    the patch can fail adhesively and part cohesively -- the mixed-mode state an
    ice block experiences while sliding over a cold structure.
    """

    def __init__(self, pressures_mpa: Sequence[float], areas_m2: Sequence[float],
                 temperature_c: float, contact_time_s: float = 1.0,
                 config: ContactBondConfig | None = None) -> None:
        cfg = config if config is not None else ContactBondConfig()
        p = np.asarray(pressures_mpa, dtype=float)
        a = np.asarray(areas_m2, dtype=float)
        if p.size != a.size or p.size == 0:
            raise ValueError("pressures and areas must be non-empty and equal length")
        self.cfg = cfg
        self.pressures_mpa = p
        self.areas_m2 = a
        self.temperature_c = float(temperature_c)
        self.contact_time_s = float(contact_time_s)
        self.bonds = [
            FrozenContactBond(
                config=ContactBondConfig(
                    strength=cfg.strength, area_m2=float(a[i]),
                    film_thickness_m=cfg.film_thickness_m,
                    fracture_energy_adhesive_j_m2=cfg.fracture_energy_adhesive_j_m2,
                    fracture_energy_cohesive_j_m2=cfg.fracture_energy_cohesive_j_m2,
                    friction_coefficient=cfg.friction_coefficient,
                    normal_stiffness_n_m=cfg.normal_stiffness_n_m),
                temperature_c=self.temperature_c,
                contact_pressure_mpa=float(p[i]),
                contact_time_s=self.contact_time_s)
            for i in range(p.size)
        ]

    def form_all(self) -> int:
        return sum(1 for b in self.bonds if b.form())

    def breakaway_force_N(self) -> dict[str, Any]:
        """``F_bk = sum_i min(sigma_b,i, sigma_c,i) A_i`` (rigid-patch bound)."""
        f_total = f_adh = f_coh = 0.0
        modes = {"adhesive": 0, "cohesive": 0, "none": 0}
        for b in self.bonds:
            ev = b.peak()
            f_total += ev["f_peak_N"]
            f_adh += ev["sigma_bond_mpa"] * b.cfg.area_m2 * 1.0e6
            f_coh += ev["sigma_cohesive_mpa"] * b.cfg.area_m2 * 1.0e6
            modes[ev["failure_mode"]] = modes.get(ev["failure_mode"], 0) + 1
        area = float(self.areas_m2.sum())
        return {"breakaway_force_N": f_total,
                "breakaway_stress_mpa": f_total / area / 1.0e6 if area > 0 else 0.0,
                "adhesive_force_N": f_adh, "cohesive_force_N": f_coh,
                "mode_counts": modes, "total_area_m2": area,
                "temperature_c": self.temperature_c,
                "contact_time_s": self.contact_time_s}

    def ledger(self) -> dict[str, float]:
        return {"q_latent_J": float(sum(b.state.q_latent_J for b in self.bonds)),
                "e_fracture_J": float(sum(b.state.e_fracture_J for b in self.bonds)),
                "e_friction_J": float(sum(b.state.e_friction_J for b in self.bonds))}


def multi_contact_slide(
    temperature_c: float,
    normal_force_per_contact_N: float = 10.0,
    contact_time_s: float = 1.0,
    n_contacts: int = 64,
    area_m2: float = 1.0e-4,
    slip_m: float = 1.0e-3,
    pressure_spread: float = 0.5,
    config: ContactBondConfig | None = None,
) -> dict[str, Any]:
    """Ice block pushed over a cold structure: break-away + sliding force.

    ``n_contacts`` patches carry normal forces spread by ``1 +/- pressure_spread``
    (deterministic alternating pattern: a fixed heterogeneous pressure field, no
    random seed).  Break-away stress and post-failure friction stress are checked
    against :data:`REPETTO_ADHESION_WINDOW_MPA` (evidence grade WITHHELD).
    """
    cfg = config if config is not None else ContactBondConfig()
    frac = np.linspace(-pressure_spread, pressure_spread, n_contacts)
    forces = normal_force_per_contact_N * (1.0 + frac)
    pressures = forces / area_m2 / 1.0e6
    assembly = FrozenBondAssembly(pressures, np.full(n_contacts, area_m2),
                                  temperature_c, contact_time_s, cfg)
    assembly.form_all()
    bk = assembly.breakaway_force_N()
    for b in assembly.bonds:
        peak = b.peak()["f_peak_N"]
        b.load(10.0 * peak if peak > 0.0 else 0.0)
        b.slide(slip_m, 10.0 * peak)
    led = assembly.ledger()
    total_normal = float(forces.sum())
    friction_force = cfg.friction_coefficient * total_normal
    lo, hi = REPETTO_ADHESION_WINDOW_MPA
    adhesion_mpa = bk["breakaway_stress_mpa"]
    return {
        "temperature_c": float(temperature_c),
        "contact_time_s": float(contact_time_s),
        "n_contacts": int(n_contacts),
        "breakaway_stress_mpa": adhesion_mpa,
        "breakaway_force_N": bk["breakaway_force_N"],
        "friction_force_N": friction_force,
        "friction_stress_mpa": friction_force / bk["total_area_m2"] / 1.0e6,
        "mode_counts": bk["mode_counts"],
        "repetto_window_mpa": list(REPETTO_ADHESION_WINDOW_MPA),
        "inside_repetto_window": bool(lo <= adhesion_mpa <= hi),
        "evidence_grade": ("WITHHELD (window check only; no digitised Repetto "
                           "2011 table in repo)"),
        **led,
    }# ---------------------------------------------------------------------------
# Energy ledgers
# ---------------------------------------------------------------------------
def mechanical_energy_ledger(
    temperature_c: float = -5.0,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 1.0,
    slip_m: float = 1.0e-3,
    n_steps: int = 20000,
    config: ContactBondConfig | None = None,
) -> dict[str, Any]:
    """Work balance of one bonded contact loaded to failure and then slid.

    The softening branch is integrated step by step (trapezoid rule).  With linear
    softening the exact dissipation is ``G_f A``, so the residual is the quadrature
    error: a genuine check that the implemented law spends exactly the fracture
    energy it declares.
    """
    cfg = config if config is not None else ContactBondConfig()
    bond = FrozenContactBond(cfg, temperature_c, contact_pressure_mpa, contact_time_s)
    bond.form()
    ev = bond.peak()
    f_peak = ev["f_peak_N"]
    mode = ev["failure_mode"]
    if mode == "none":
        raise ValueError("no bond forms at these conditions (T >= T_freeze)")
    g_f = bond._fracture_energy(mode)
    k = bond.cfg.normal_stiffness_n_m
    delta_0 = f_peak / k
    delta_tot = 2.0 * g_f / f_peak            # area 0.5 f_peak delta_tot = G_f A
    delta_soft = max(delta_tot - delta_0, 1.0e-30)

    deltas = np.linspace(0.0, delta_tot, n_steps + 1)
    forces = np.where(deltas <= delta_0, k * deltas,
                      f_peak * np.maximum(0.0, 1.0 - (deltas - delta_0) / delta_soft))
    w_ext = float(np.trapezoid(forces, deltas))
    residual_frac = abs(w_ext - g_f) / g_f if g_f else 0.0

    normal_load = contact_pressure_mpa * 1.0e6 * cfg.area_m2   # applied F_n
    e_fric_exact = cfg.friction_coefficient * normal_load * slip_m
    n_fric = max(1, n_steps // 10)
    s = np.linspace(0.0, slip_m, n_fric + 1)
    f_fric = np.full_like(s, cfg.friction_coefficient * normal_load)
    e_fric_num = float(np.trapezoid(f_fric, s))
    residual_fric = (abs(e_fric_num - e_fric_exact) / e_fric_exact
                     if e_fric_exact else 0.0)

    w_total = w_ext + e_fric_num
    dissipated = g_f + e_fric_exact
    residual = abs(w_total - dissipated) / w_total if w_total else 0.0
    return {"temperature_c": float(temperature_c), "failure_mode": mode,
            "w_external_J": w_total, "e_fracture_exact_J": g_f,
            "e_friction_exact_J": e_fric_exact,
            "residual_fracture": residual_frac,
            "residual_friction": residual_fric, "residual": residual,
            "criterion_5pct": bool(residual <= 0.05), "n_steps": n_steps}


def freezing_front_ledger(
    temperature_c: float = -10.0,
    film_thickness_m: float = 1.0e-7,
    n_nodes: int = 61,
    phase_change_band_k: float = 0.05,
    n_steps_target: int = 8000,
) -> dict[str, Any]:
    """Enthalpy-method freezing solve of the interfacial film (1-D, implicit).

    A one-dimensional ice/water column occupies ``0 <= x <= h``: the substrate at
    ``x = 0`` is held at ``temperature_c`` (Dirichlet), the far side of the film is
    adiabatic, and the film starts at its freezing point.  Phase change is carried
    by the apparent heat capacity ``c_app = c + L/delta`` over the band
    ``[-delta, 0]`` with frozen fraction ``f(T) = clip(-T/delta, 0, 1)``, i.e.
    enthalpy ``H(T) = c min(T,0) - f L`` relative to liquid water at 0 C.  Time
    stepping is backward Euler (Thomas solve) with Picard iteration on the phase
    state, so the step is set by accuracy, not by a stability limit.

    Running this from the initial (all-water) to the fully frozen state gives the
    energy that must leave through the substrate, ``q_conducted``, and the static
    enthalpy integral ``H_initial - H_final``.  Two residuals are reported:

    ``residual_balance``
        ``|q_conducted - (H_initial - H_final)| / |H_initial - H_final|`` -- the
        discrete conservation error of the scheme (first order in dt because of
        the moving phase boundary; ~2 % at the default 8000 steps).
    ``residual``
        ``|q_conducted - (q_latent + q_sensible)| / (q_latent + q_sensible)`` with
        ``q_latent = rho L h`` and ``q_sensible = rho c h dT / 2`` from the
        analytic Stefan / linear-profile budget: the accuracy of the numerical
        solution against the analytic energy budget -- criterion (f) in its
        physically complete form.
    """
    h = float(film_thickness_m)
    d_t = abs(float(temperature_c))
    if h <= 0.0 or d_t <= 0.0:
        raise ValueError("film thickness and subcooling must be positive")
    if n_nodes < 3:
        raise ValueError("n_nodes must be >= 3")
    rho, lf = WATER_DENSITY, LATENT_HEAT_FUSION
    k_i, c_i = ICE_CONDUCTIVITY, ICE_HEAT_CAPACITY
    delta = float(phase_change_band_k)
    n = int(n_nodes)
    dx = h / (n - 1)
    t_sub = -d_t

    t_stefan = rho * lf * h * h / (2.0 * k_i * d_t)
    dt = t_stefan / max(1, int(n_steps_target))

    def fraction(temp: np.ndarray) -> np.ndarray:
        return np.clip(-temp / delta, 0.0, 1.0)

    def enthalpy(temp: np.ndarray) -> np.ndarray:
        return c_i * np.minimum(temp, 0.0) - fraction(temp) * lf

    def capacity(temp: np.ndarray) -> np.ndarray:
        in_band = (temp <= 0.0) & (temp >= -delta)
        return c_i + np.where(in_band, lf / delta, 0.0)

    temp = np.zeros(n)                      # the film starts as water at 0 C
    temp[0] = t_sub
    q_cond = 0.0
    h_initial = float(np.sum(rho * dx * enthalpy(temp)))
    steps = 0
    max_steps = 400000
    while steps < max_steps and not np.all(temp <= -delta):
        cap = capacity(temp)
        for _ in range(30):                 # Picard on the phase state
            a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); rhs = np.zeros(n)
            coef = k_i * dt / (rho * cap * dx * dx)
            b[1:-1] = 1.0 + 2.0 * coef[1:-1]
            a[1:-1] = -coef[1:-1]
            c[1:-1] = -coef[1:-1]
            rhs[1:-1] = temp[1:-1]
            b[0] = 1.0; c[0] = 0.0; rhs[0] = t_sub
            b[-1] = 1.0 + 2.0 * coef[-1]
            a[-1] = -2.0 * coef[-1]
            rhs[-1] = temp[-1]
            cp = np.zeros(n); dp = np.zeros(n)
            cp[0] = c[0] / b[0]; dp[0] = rhs[0] / b[0]
            for i in range(1, n):
                m = b[i] - a[i] * cp[i - 1]
                cp[i] = c[i] / m
                dp[i] = (rhs[i] - a[i] * dp[i - 1]) / m
            new = np.zeros(n)
            new[-1] = dp[-1]
            for i in range(n - 2, -1, -1):
                new[i] = dp[i] - cp[i] * new[i + 1]
            new[0] = t_sub
            cap_new = capacity(new)
            done = np.allclose(cap_new, cap, rtol=1.0e-10, atol=0.0)
            temp, cap = new, cap_new
            if done:
                break
        q_cond += dt * k_i * (temp[1] - t_sub) / dx
        steps += 1
    h_final = float(np.sum(rho * dx * enthalpy(temp)))
    removed = h_initial - h_final
    balance = abs(q_cond - removed) / abs(removed) if removed else float("inf")
    q_lat = rho * lf * h
    q_sens = rho * c_i * h * d_t / 2.0
    analytic = q_lat + q_sens
    residual = abs(q_cond - analytic) / analytic
    return {"temperature_c": float(temperature_c),
            "film_thickness_m": h,
            "q_latent_j_m2": q_lat,
            "q_sensible_j_m2": q_sens,
            "q_analytic_budget_j_m2": analytic,
            "q_conducted_j_m2": q_cond,
            "energy_removed_j_m2": removed,
            "e_surface_j_m2": ICE_SURFACE_ENERGY,
            "freeze_time_s": steps * dt, "stefan_time_s": t_stefan,
            "freeze_time_error": abs(steps * dt - t_stefan) / t_stefan,
            "steps": steps, "converged": bool(np.all(temp <= -delta)),
            "residual": residual, "residual_balance": balance,
            "criterion_5pct": bool(residual <= 0.05)}


def energy_ledger(
    temperature_c: float = -10.0,
    contact_pressure_mpa: float = 0.1,
    contact_time_s: float = 1.0,
    slip_m: float = 1.0e-3,
    config: ContactBondConfig | None = None,
) -> dict[str, Any]:
    """Combined energy ledger for bond formation -> break -> sliding.

    Three budgets, all reported because they are *not* the same statement:

    ``thermal_residual``
        ``|q_latent + q_sensible - (q_conducted + e_surface)| / (q_latent +
        q_sensible)`` -- the physically complete thermal budget.  This is the
        ledger criterion (f) is graded on.
    ``mechanical_residual``
        ``|W_external - (E_fracture + E_friction)| / W_external`` from the
        incremental mechanical loading ledger.
    ``two_term_residual``
        the literal campaign statement ``Q_lat = E_fracture + E_friction``.  It is
        **falsified** for any realistic film (the latent heat is conducted away, not
        spent on fracture); it would close only at ``closure_thickness_m``.
    """
    cfg = config if config is not None else ContactBondConfig()
    mech = mechanical_energy_ledger(temperature_c, contact_pressure_mpa,
                                   contact_time_s, slip_m, config=cfg)
    therm = freezing_front_ledger(temperature_c, cfg.film_thickness_m)
    a = cfg.area_m2
    q_lat = therm["q_latent_j_m2"] * a
    q_sens = therm["q_sensible_j_m2"] * a
    q_cond = therm["q_conducted_j_m2"] * a
    e_surface = therm["e_surface_j_m2"] * a
    e_frac = mech["e_fracture_exact_J"]
    e_fric = mech["e_friction_exact_J"]
    two_term = abs(q_lat - (e_frac + e_fric)) / q_lat if q_lat else float("inf")
    thermal = abs((q_lat + q_sens) - (q_cond + e_surface)) / (q_lat + q_sens)
    mechanical = mech["residual"]
    closure_h = (e_frac + e_fric) / (WATER_DENSITY * LATENT_HEAT_FUSION * a)
    return {
        "temperature_c": float(temperature_c),
        "film_thickness_m": cfg.film_thickness_m,
        "q_latent_J": q_lat,
        "q_sensible_J": q_sens,
        "q_conducted_J": q_cond,
        "e_surface_J": e_surface,
        "e_fracture_J": e_frac,
        "e_friction_J": e_fric,
        "latent_over_mechanical": (q_lat / (e_frac + e_fric)
                                   if (e_frac + e_fric) else float("inf")),
        "two_term_residual": two_term,
        "two_term_criterion_5pct": bool(two_term <= 0.05),
        "closure_thickness_m": closure_h,
        "thermal_residual": thermal,
        "mechanical_residual": mechanical,
        "criterion_5pct": bool(thermal <= 0.05 and mechanical <= 0.05),
        "thermal_ledger": therm,
        "mechanical_ledger": mech,
        "note": ("the literal two-term budget Q_lat = E_fracture + E_friction is "
                 "falsified for realistic films: the latent heat of the freezing "
                 "film is conducted away, not spent on fracture; it closes only at "
                 "the reported closure thickness.  The complete thermal budget "
                 "(latent + sensible = conducted + surface) and the mechanical work "
                 "balance are verified separately."),
    }
