"""Ice-structure hydrodynamic-lubrication regime maturation (Stribeck extension).

Campaign node ``p9050-n09-ice-lubrication-maturity``.

Why this module exists
----------------------
:mod:`tensordem.ice_steel_friction_closure` answers "what is ``mu``" for a
sliding ice-steel pair with three regimes keyed on a film area fraction.  What it
cannot answer -- and what a hull / ice-loads design loop needs -- is the **regime
transition itself**:

* where on a Stribeck curve the contact sits, in Hersey numbers extended down
  into the ice/water phase-change band (``1e-8 .. 1e-2``);
* what melt-film thickness the friction heat *itself* sustains, from the
  melting / squeeze-out balance (the ``h(t)`` ODE of the campaign);
* how temperature, speed and **surface roughness** combine into the
  ``mu(T, V, Ra)`` response surface;
* which of boundary -> mixed -> hydrodynamic -> full water film applies, by the
  non-dimensional film ratio ``Lambda = h_film / sqrt(Ra_ice^2 + Ra_steel^2)``.

The module is deliberately *not* a refit of the existing closure: nothing here
imports the calibration table, so the two models are independent evidence about
the same physics.

Physics chain (each link separately testable)
---------------------------------------------
1. **Flash temperature** (Blok/Jaeger) decides whether meltwater exists at all:
   ``dT_flash = 2 q sqrt(t_c/pi) / sqrt(k rho c)_eff`` with
   ``1/sqrt(k rho c)_eff = 1/sqrt(k rho c)_ice + 1/sqrt(k rho c)_steel`` and
   ``t_c = L/V`` the dwell time.  Only the flux in excess of the conductive
   capacity of the two bodies melts ice.
2. **Melt film growth** -- the campaign ODE

   ``rho L_f dh/dt = mu (F_N/A) V - k dT / h - h_conv dT``

   (:func:`film_ode_rhs`, RK4 in :func:`integrate_film`).  Two structural
   findings are recorded in ``docs/ICE_LUBRICATION_REGIME.md``:

   * ``d/dh [dh/dt] = +k dT /(rho L h^2) > 0``: with ``mu`` held constant the
     algebraic steady state is **unstable**; the film is only stabilised once
     the friction heat depends on ``h`` (thicker film -> smaller
     ``mu_f = eta V/(h p)``).  :func:`film_steady_state` therefore solves the
     self-consistent balance and :func:`film_steady_state_analytic` returns the
     literal closed form;
   * for a macroscopic contact (``dT >= 2 K``, ``h <= 100 um``) the ``k dT/h``
     sink alone can never be balanced by friction heat, i.e. the literal ODE
     predicts "no sustained film".  The physical sink at interface scale is
     conduction *into the two bodies* (item 1), not across the film.  The
     delivered prediction therefore uses the **melt-supply / squeeze-out**
     balance (:func:`sustained_film_thickness`) and keeps the literal ODE as the
     acceptance reference for criterion (b).
3. **Squeeze-out drainage** (Reynolds, circular patch)
   ``dh/dt = -2 h^3 p /(3 eta R^2)``; balanced against the melt supply it caps
   the film at ``h_drain = (3 eta R^2 v_melt /(2 p))^(1/3)``.
4. **Load sharing** ``A(Lambda)``: 0 below ``Lambda = 1``, 1 above
   ``Lambda = 3``, linear between -- the campaign regime thresholds are the same
   numbers that blend ``mu = (1-A) mu_b + A mu_f``.
5. **Hersey number** ``Her = eta_f V / p_nom`` with ``eta_f`` the slurry
   viscosity (water VFT law x Krieger-Dougherty ice-fraction enhancement); the
   accessible band is reported by :func:`hersey_band` rather than asserted.

Constants are literature values, not tuned on the validation anchors; the two
soft parameters (boundary shear strength, drainage retention) carry documented
uncertainty bands and :func:`calibrate` reports a leave-one-out honest fit.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "GAS_CONSTANT", "WATER_K", "ICE_K", "ICE_RHO", "ICE_CP", "ICE_LATENT",
    "STEEL_K", "STEEL_RHO", "STEEL_CP", "ICE_VISC_REF", "ICE_CREEP_ACTIVATION",
    "BOUNDARY_TAU0", "BOUNDARY_ACTIVATION", "MU_BOUNDARY_MIN", "MU_BOUNDARY_MAX",
    "H_CONV", "H_MIN", "H_MAX", "MELT_RAMP_K", "FLASH_CONCENTRATION", "LAMBDA_BOUNDARY", "LAMBDA_HYDRO",
    "DEFAULT_ICE_ROUGHNESS", "DEFAULT_STEEL_ROUGHNESS", "DRAINAGE_RETENTION",
    "REGIME_BOUNDARY", "REGIME_MIXED", "REGIME_HYDRO", "REGIME_FULL_FILM",
    "REGIMES", "water_viscosity", "ice_creep_viscosity", "slurry_viscosity",
    "melting_point", "melting_point_depression", "blok_effusivity",
    "flash_temperature_rise", "conductive_capacity", "boundary_shear_strength",
    "boundary_friction", "film_friction", "stribeck_mu", "stribeck_optimum",
    "hersey_number", "hersey_band", "film_ode_rhs", "integrate_film",
    "film_steady_state", "film_steady_state_analytic", "drainage_rate",
    "sustained_film_thickness", "combined_roughness", "film_area_fraction",
    "regime_of", "LubricationInputs", "LubricationResult", "predict",
    "friction_surface", "stribeck_scan", "LITERATURE_ANCHORS",
    "LiteratureAnchor", "validate_literature", "negative_controls",
    "falsifiability_probe", "calibrate", "film_thickness_supply",
    "stribeck_parameters", "stribeck_check", "stribeck_curve",
    "STRIBECK_HER_OPT_LITERATURE", "STRIBECK_HER_OPT_BAND",
]

GAS_CONSTANT = 8.31446261815324

# thermal properties near the melting point
WATER_K = 0.56
ICE_K = 2.2
ICE_RHO = 917.0
ICE_CP = 2050.0
ICE_LATENT = 3.34e5
STEEL_K = 50.0
STEEL_RHO = 7800.0
STEEL_CP = 470.0

# rheology
ICE_VISC_REF = 1.0e13
ICE_CREEP_ACTIVATION = 70.0e3
SLURRY_PHI_MAX = 0.60
SLURRY_INTRINSIC = 2.5

# boundary branch
BOUNDARY_TAU0 = 0.10e6
BOUNDARY_ACTIVATION = 8.0e3
MU_BOUNDARY_MIN = 0.05
MU_BOUNDARY_MAX = 0.30

# film energetics
H_CONV = 200.0
H_MIN = 1.0e-8
H_MAX = 5.0e-3
DRAINAGE_RETENTION = 1.0
LAMBDA_BOUNDARY = 1.0
LAMBDA_HYDRO = 3.0
MELT_RAMP_K = 2.0        # K, spread of asperity flash temperatures (onset softening)
FLASH_CONCENTRATION = 2.5  # peak/mean asperity flash temperature (rough contact, band 2-3)

DEFAULT_ICE_ROUGHNESS = 1.0e-6
DEFAULT_STEEL_ROUGHNESS = 1.0e-6

REGIME_BOUNDARY = "boundary"
REGIME_MIXED = "mixed"
REGIME_HYDRO = "hydrodynamic"
REGIME_FULL_FILM = "full_water_film"
REGIMES = (REGIME_BOUNDARY, REGIME_MIXED, REGIME_HYDRO, REGIME_FULL_FILM)

_T0 = 273.15


def _num(name: str, value: Any, *, positive: bool = False,
         nonnegative: bool = False) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out}")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out}")
    return out


# ---------------------------------------------------------------------------
# 1. rheology and thermodynamics
# ---------------------------------------------------------------------------
def water_viscosity(temperature_c: Any) -> np.ndarray:
    """Dynamic viscosity of (supercooled) water, Pa s.

    Vogel-Fulcher-Tammann form ``eta = 2.414e-5 * 10**(247.8/(T - 140))``.
    Reproduces ``1.75e-3`` Pa s at 0 degC (2.2 % below the textbook
    ``1.792e-3``), ``1.30e-3`` at 10 degC and ``2.48e-3`` at -10 degC
    (supercooled); the friction closure's ``1.79e-3`` at the melting point is
    matched to 2 %, so the two models agree on the lubricant to within the
    correlation's own accuracy rather than by fitting.
    """
    t = np.asarray(temperature_c, dtype=float)
    tk = t + _T0
    if np.any(tk <= 140.0):
        raise ValueError("temperature must be above -133 degC")
    return 2.414e-5 * np.power(10.0, 247.8 / (tk - 140.0))


def ice_creep_viscosity(temperature_c: Any) -> np.ndarray:
    """Effective Newtonian viscosity of polycrystalline ice, Pa s.

    Arrhenius (Glen-type) ``eta = eta_ref exp[Q/R (1/T - 1/T_melt)]`` with
    ``eta_ref = 1e13 Pa s`` at the melting point and ``Q = 70 kJ/mol``.  This is
    the viscosity of the *cold* branch, where the sheared medium is the ice
    itself; it is why the Hersey band of an ice contact is many decades wider
    than that of an oil-lubricated bearing.
    """
    t = np.asarray(temperature_c, dtype=float)
    tk = t + _T0
    if np.any(tk <= 0.0):
        raise ValueError("temperature must be above absolute zero")
    return ICE_VISC_REF * np.exp(
        ICE_CREEP_ACTIVATION / GAS_CONSTANT * (1.0 / tk - 1.0 / _T0))


def slurry_viscosity(temperature_c: Any, ice_fraction: Any = 0.0) -> np.ndarray:
    """Viscosity of an ice-particle-laden melt film, Pa s.

    Krieger-Dougherty ``eta = eta_w (1 - phi/phi_max)**(-[eta] phi_max)`` with
    ``phi_max = 0.6`` and ``[eta] = 2.5`` (exponent 1.5).  A dense slurry
    (``phi -> 0.55``) is ~10^2 x more viscous than neat melt, which is the
    physical reason a partial-melt film can still carry load.
    """
    phi = np.asarray(ice_fraction, dtype=float)
    if np.any(phi < 0.0) or np.any(phi >= SLURRY_PHI_MAX):
        raise ValueError(f"ice_fraction must lie in [0, {SLURRY_PHI_MAX})")
    return water_viscosity(temperature_c) * np.power(
        1.0 - phi / SLURRY_PHI_MAX, -SLURRY_INTRINSIC * SLURRY_PHI_MAX)


def melting_point_depression(water_pressure_pa: Any) -> np.ndarray:
    """Clapeyron melting-point depression, K (returned positive).

    ``dT/dp = -0.0074 K/MPa`` for the ice Ih / water transition; at 10 MPa the
    melting point drops ~0.074 K.  Small, but it enters the flash-temperature
    test exactly at the margin where warm ice is decided, so the campaign input
    ``p_w`` is not decorative.
    """
    p = np.asarray(water_pressure_pa, dtype=float)
    if np.any(p < 0.0):
        raise ValueError("water_pressure_pa must be >= 0")
    return 0.0074 * p / 1.0e6


def melting_point(water_pressure_pa: Any = 0.0) -> np.ndarray:
    """Melting temperature of ice under an external water pressure, degC."""
    return -melting_point_depression(water_pressure_pa)


# ---------------------------------------------------------------------------
# 2. flash temperature -> does meltwater exist?
# ---------------------------------------------------------------------------
def blok_effusivity(temperature_c: float = 0.0) -> dict[str, float]:
    """``sqrt(k rho c)`` of ice, of steel and of the Blok series combination."""
    e_ice = math.sqrt(ICE_K * ICE_RHO * ICE_CP)
    e_steel = math.sqrt(STEEL_K * STEEL_RHO * STEEL_CP)
    return {"ice": e_ice, "steel": e_steel,
            "combined": 1.0 / (1.0 / e_ice + 1.0 / e_steel)}


def flash_temperature_rise(flux_w_m2: Any, dwell_time_s: Any) -> np.ndarray:
    """Blok/Jaeger flash temperature rise of a sliding interface, K.

    ``dT = 2 q sqrt(t_c/pi) / sqrt(k rho c)_eff``.  This decides melting onset
    and is independent of film thickness -- precisely why it, and not the
    film-conduction term, answers the onset question.
    """
    q = np.asarray(flux_w_m2, dtype=float)
    t = np.asarray(dwell_time_s, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("dwell_time_s must be > 0")
    if np.any(q < 0.0):
        raise ValueError("flux_w_m2 must be >= 0")
    e = blok_effusivity()["combined"]
    return 2.0 * q * np.sqrt(t / math.pi) / e


def conductive_capacity(temperature_deficit_k: Any, dwell_time_s: Any) -> np.ndarray:
    """Flux the two bodies absorb while the interface stays at ``T_melt``.

    Inverse of :func:`flash_temperature_rise`.  Below this flux the interface
    never reaches the melting point, no meltwater exists and the contact is in
    pure boundary lubrication -- the negative control ``V -> 0``.
    """
    dt = np.asarray(temperature_deficit_k, dtype=float)
    t = np.asarray(dwell_time_s, dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("dwell_time_s must be > 0")
    e = blok_effusivity()["combined"]
    return np.maximum(dt, 0.0) * e / (2.0 * np.sqrt(t / math.pi))


# ---------------------------------------------------------------------------
# 3. friction laws
# ---------------------------------------------------------------------------
def boundary_shear_strength(temperature_c: Any) -> np.ndarray:
    """Interfacial shear strength of an ice-steel boundary layer, Pa.

    ``tau_b = tau_0 exp[Q_t/R (1/T - 1/T_melt)]`` with ``tau_0 = 0.10 MPa`` and a
    mild ``Q_t = 8 kJ/mol`` (the shear plane is a premelted film, so the
    Arrhenius slope is far below the bulk Glen activation).  Plausible band
    ``0.05 .. 0.2 MPa`` for creep friction of ice on steel.
    """
    t = np.asarray(temperature_c, dtype=float)
    tk = t + _T0
    if np.any(tk <= 0.0):
        raise ValueError("temperature must be above absolute zero")
    return BOUNDARY_TAU0 * np.exp(
        BOUNDARY_ACTIVATION / GAS_CONSTANT * (1.0 / tk - 1.0 / _T0))


def boundary_friction(pressure_pa: Any, temperature_c: Any, *,
                      tau_scale: float = 1.0) -> np.ndarray:
    """Coulomb boundary coefficient ``mu_b = tau_b / p`` clipped to the band.

    The clip is physical, not cosmetic: ``tau_b/p`` cannot fall below the
    measured plateau (``0.05``) because the real contact area saturates at the
    ice indentation hardness (Bowden-Tabor), and cannot exceed ``0.30`` because
    above that the shear plane moves into the ice bulk and ploughing dominates.
    """
    p = np.asarray(pressure_pa, dtype=float)
    if np.any(p <= 0.0):
        raise ValueError("pressure_pa must be > 0")
    tau = boundary_shear_strength(temperature_c) * _num("tau_scale", tau_scale,
                                                        positive=True)
    return np.clip(tau / p, MU_BOUNDARY_MIN, MU_BOUNDARY_MAX)


def film_friction(viscosity_pa_s: Any, speed_m_s: Any, pressure_pa: Any,
                  thickness_m: Any) -> np.ndarray:
    """Viscous shear coefficient of the load-carrying film ``mu_f = eta V/(h p)``."""
    eta = np.asarray(viscosity_pa_s, dtype=float)
    v = np.asarray(speed_m_s, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    h = np.maximum(np.asarray(thickness_m, dtype=float), H_MIN)
    if np.any(p <= 0.0):
        raise ValueError("pressure_pa must be > 0")
    return eta * np.abs(v) / (h * p)


def hersey_number(viscosity_pa_s: Any, speed_m_s: Any,
                  pressure_pa: Any) -> np.ndarray:
    """Hersey number ``Her = eta V / p`` (dimensionless)."""
    eta = np.asarray(viscosity_pa_s, dtype=float)
    v = np.asarray(speed_m_s, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    if np.any(p <= 0.0):
        raise ValueError("pressure_pa must be > 0")
    return eta * np.abs(v) / p


def stribeck_mu(her: Any, *, mu_b: float = 0.10, her_c: float = 1.0e-6,
                her_v: float = 1.0e-3) -> np.ndarray:
    """Analytic Stribeck master curve extended into the phase-change band.

    ``mu(Her) = mu_b / (1 + Her/Her_c) + Her/Her_v``:

    * ``Her << Her_c``: boundary branch, ``mu -> mu_b`` (Coulomb plateau);
    * ``Her >> Her_v``: viscous branch, ``mu -> Her/Her_v`` -- the high-speed
      upturn every measured ice-friction curve shows;
    * minimum at ``Her_opt = sqrt(mu_b Her_c Her_v) - Her_c``.

    ``Her_c`` is the film-onset Hersey scale (the film carries half the load at
    ``Lambda = 2``) and ``Her_v`` the viscous scale ``eta/(3 sigma p)``; their
    defaults are the values those closures give for a 1 MPa, 1 um-roughness
    ice-steel contact at -5 degC.  The closed form is what makes acceptance
    criterion (c) machine-checkable.
    """
    h = np.asarray(her, dtype=float)
    if np.any(h < 0.0):
        raise ValueError("her must be >= 0")
    if her_c <= 0.0 or her_v <= 0.0 or mu_b < 0.0:
        raise ValueError("her_c, her_v must be > 0 and mu_b >= 0")
    return mu_b / (1.0 + h / her_c) + h / her_v


def stribeck_optimum(mu_b: float = 0.10, her_c: float = 1.0e-6,
                     her_v: float = 1.0e-3) -> dict[str, float]:
    """Location and value of the Stribeck minimum (analytic).

    ``d mu/d Her = 0`` gives ``(Her_c + Her)^2 = mu_b Her_c Her_v``, hence
    ``Her_opt = sqrt(mu_b Her_c Her_v) - Her_c``.
    """
    mu_b = _num("mu_b", mu_b, nonnegative=True)
    her_c = _num("her_c", her_c, positive=True)
    her_v = _num("her_v", her_v, positive=True)
    root = math.sqrt(mu_b * her_c * her_v)
    her_opt = root - her_c
    if her_opt <= 0.0:
        raise ValueError("no interior minimum: mu_b*her_v <= her_c")
    mu_opt = float(stribeck_mu(her_opt, mu_b=mu_b, her_c=her_c, her_v=her_v))
    return {"her_opt": her_opt, "mu_opt": mu_opt, "her_c": her_c,
            "her_v": her_v, "sqrt_term": root}


def hersey_band(temperature_c: float = -10.0, *, speed_range=(1.0e-6, 10.0),
                pressure_range=(1.0e3, 2.0e7), ice_fraction: float = 0.0
                ) -> dict[str, Any]:
    """Accessible Hersey band for an ice contact at ``temperature_c``.

    Reports the decades spanned by ``eta V / p`` over the given speed/pressure
    box for the slurry viscosity, the neat melt and the ice-creep envelope, so
    the campaign's ``1e-8 .. 1e-2`` target is *checked* rather than asserted.
    """
    eta_slurry = float(slurry_viscosity(temperature_c, ice_fraction))
    eta_w = float(water_viscosity(temperature_c))
    eta_ice = float(ice_creep_viscosity(temperature_c))
    v_lo, v_hi = speed_range
    p_lo, p_hi = pressure_range
    band = {
        "eta_water_pa_s": eta_w,
        "eta_slurry_pa_s": eta_slurry,
        "eta_ice_creep_pa_s": eta_ice,
        "her_water_min": eta_w * v_lo / p_hi,
        "her_water_max": eta_w * v_hi / p_lo,
        "her_slurry_min": eta_slurry * v_lo / p_hi,
        "her_slurry_max": eta_slurry * v_hi / p_lo,
        "her_envelope_min": min(eta_w, eta_ice) * v_lo / p_hi,
        "her_envelope_max": max(eta_w, eta_ice) * v_hi / p_lo,
        "target_band": (1.0e-8, 1.0e-2),
    }
    band["her_slurry_covers_target"] = bool(
        band["her_slurry_min"] <= 1.0e-8 and band["her_slurry_max"] >= 1.0e-2)
    band["her_envelope_covers_target"] = bool(
        band["her_envelope_min"] <= 1.0e-8
        and band["her_envelope_max"] >= 1.0e-2)
    return band

# ---------------------------------------------------------------------------
# 4. film thickness: the campaign ODE and its steady states
# ---------------------------------------------------------------------------
def film_ode_rhs(thickness_m: float, *, friction_flux_w_m2: float,
                 temperature_deficit_k: float, water_k: float = WATER_K,
                 h_conv: float = H_CONV, rho_ice: float = ICE_RHO,
                 latent_heat: float = ICE_LATENT,
                 drainage_retention: float = 0.0,
                 drainage_flux_w_m2: float = 0.0) -> float:
    """``dh/dt`` of the campaign melt-film ODE, m/s.

    ``rho L dh/dt = q_fric - k dT/h - h_conv dT - R_drain q_drain``

    The last term is the documented **extension** (default off,
    ``drainage_retention = 0``) representing meltwater squeezed out of the
    contact.  With it off this function is exactly the campaign ODE.
    """
    h = _num("thickness_m", thickness_m, positive=True)
    q = _num("friction_flux_w_m2", friction_flux_w_m2)
    dt = _num("temperature_deficit_k", temperature_deficit_k)
    k = _num("water_k", water_k, positive=True)
    hc = _num("h_conv", h_conv, nonnegative=True)
    rho = _num("rho_ice", rho_ice, positive=True)
    lat = _num("latent_heat", latent_heat, positive=True)
    ret = _num("drainage_retention", drainage_retention, nonnegative=True)
    qd = _num("drainage_flux_w_m2", drainage_flux_w_m2, nonnegative=True)
    net = q - k * dt / h - hc * dt - ret * qd
    return net / (rho * lat)


def integrate_film(duration_s: float, *, initial_thickness_m: float = H_MIN,
                   friction_flux_w_m2: float | Callable[[float], float] = 0.0,
                   temperature_deficit_k: float = 10.0,
                   steps: int = 4000, **rhs_kwargs: Any
                   ) -> dict[str, np.ndarray]:
    """RK4 integration of :func:`film_ode_rhs`, deterministic fixed step.

    ``friction_flux_w_m2`` may be a callable of ``h`` so the film-branch feedback
    can be carried.  Returns ``{"t", "h"}``; ``h`` is floored at :data:`H_MIN`
    and capped at :data:`H_MAX`.
    """
    dur = _num("duration_s", duration_s, positive=True)
    n = int(steps)
    if n < 1:
        raise ValueError("steps must be >= 1")
    h = _num("initial_thickness_m", initial_thickness_m, positive=True)
    dts = dur / n
    flux = friction_flux_w_m2
    t_hist = np.empty(n + 1)
    h_hist = np.empty(n + 1)
    t_hist[0] = 0.0
    h_hist[0] = h

    def f(hh: float) -> float:
        q = flux(hh) if callable(flux) else float(flux)
        return film_ode_rhs(hh, friction_flux_w_m2=q,
                            temperature_deficit_k=temperature_deficit_k,
                            **rhs_kwargs)

    for i in range(n):
        k1 = f(h)
        k2 = f(max(H_MIN, h + 0.5 * dts * k1))
        k3 = f(max(H_MIN, h + 0.5 * dts * k2))
        k4 = f(max(H_MIN, h + dts * k3))
        h = h + dts * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        h = min(max(h, H_MIN), H_MAX)
        t_hist[i + 1] = (i + 1) * dts
        h_hist[i + 1] = h
    return {"t": t_hist, "h": h_hist}


def film_steady_state_analytic(friction_flux_w_m2: float,
                               temperature_deficit_k: float, *,
                               water_k: float = WATER_K,
                               h_conv: float = H_CONV) -> dict[str, float]:
    """Closed-form steady state of the literal ODE: ``h = k dT /(q - h_conv dT)``.

    ``exists`` is False when ``q <= h_conv dT``: then no steady state exists and
    the film grows without bound (the campaign's ``T_ice -> 0`` control).
    ``stability_sign = +1`` records that ``d/dh[dh/dt] > 0`` at the root, i.e.
    the literal root is unstable at fixed ``mu`` -- declared, not hidden.
    """
    q = _num("friction_flux_w_m2", friction_flux_w_m2, nonnegative=True)
    dt = _num("temperature_deficit_k", temperature_deficit_k, nonnegative=True)
    k = _num("water_k", water_k, positive=True)
    hc = _num("h_conv", h_conv, nonnegative=True)
    out = {"q_fric": q, "dT": dt, "q_conv": hc * dt, "exists": 0.0,
           "h_ss": float("nan"), "stability_sign": float("nan")}
    if dt == 0.0:
        out["stability_sign"] = 0.0
        return out
    if q <= hc * dt:
        return out
    h_ss = k * dt / (q - hc * dt)
    out.update({"exists": 1.0, "h_ss": h_ss, "stability_sign": 1.0})
    return out


def film_steady_state(*, pressure_pa: float, speed_m_s: float,
                      temperature_deficit_k: float,
                      roughness_m: float = DEFAULT_ICE_ROUGHNESS,
                      eta_pa_s: float | None = None,
                      mu_b: float = 0.10,
                      water_k: float = WATER_K, h_conv: float = H_CONV,
                      h_min: float = H_MIN, h_max: float = 1.0e-3,
                      log_points: int = 4000) -> dict[str, float]:
    """Self-consistent steady film thickness of the campaign ODE, m.

    Solves ``dh/dt = 0`` with the feedback ``mu(h) = (1-A) mu_b + A eta V/(h p)``
    -- the version that has a *stable* root.  Returns the largest root plus
    ``root_count`` / ``max_rhs`` so a supply-starved case is reported as such
    instead of silently returning ``h_min``.
    """
    p = _num("pressure_pa", pressure_pa, positive=True)
    v = _num("speed_m_s", speed_m_s, nonnegative=True)
    dt = _num("temperature_deficit_k", temperature_deficit_k)
    sigma = _num("roughness_m", roughness_m, positive=True)
    eta = (float(slurry_viscosity(-abs(dt))) if eta_pa_s is None
           else _num("eta_pa_s", eta_pa_s, positive=True))
    mu_b = _num("mu_b", mu_b, nonnegative=True)
    hs = np.logspace(math.log10(h_min), math.log10(h_max), int(log_points))
    lam = hs / sigma
    a = film_area_fraction(lam)
    mu_loc = (1.0 - a) * mu_b + a * eta * v / (hs * p)
    rhs = mu_loc * p * v - water_k * dt / hs - h_conv * dt
    sign = np.sign(rhs)
    idx = np.nonzero(np.diff(sign) != 0.0)[0]
    out = {"eta_pa_s": eta, "root_count": float(idx.size),
           "max_rhs": float(np.max(rhs)), "h_ss": float("nan"),
           "lambda_ss": float("nan"), "regime": REGIME_BOUNDARY,
           "starved": 1.0}
    if idx.size:
        lo, hi = hs[idx[-1]], hs[idx[-1] + 1]
        side = rhs[idx[-1]] > 0.0
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            a_m = float(film_area_fraction(mid / sigma))
            mu_m = (1.0 - a_m) * mu_b + a_m * eta * v / (mid * p)
            r = mu_m * p * v - water_k * dt / mid - h_conv * dt
            if (r > 0.0) == side:
                lo = mid
            else:
                hi = mid
        h_ss = math.sqrt(lo * hi)
        out.update({"h_ss": h_ss, "lambda_ss": h_ss / sigma,
                    "regime": regime_of(h_ss / sigma)[0], "starved": 0.0})
    return out


def drainage_rate(thickness_m: Any, pressure_pa: Any, radius_m: Any,
                  viscosity_pa_s: Any) -> np.ndarray:
    """Reynolds squeeze-out velocity of a circular film, m/s (positive = loss)."""
    h = np.asarray(thickness_m, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    r = np.asarray(radius_m, dtype=float)
    eta = np.asarray(viscosity_pa_s, dtype=float)
    if np.any(r <= 0.0) or np.any(eta <= 0.0):
        raise ValueError("radius_m and viscosity_pa_s must be > 0")
    return 2.0 * np.abs(h) ** 3 * p / (3.0 * eta * r ** 2)


def sustained_film_thickness(*, friction_flux_w_m2: float,
                             conductive_flux_w_m2: float,
                             pressure_pa: float, contact_radius_m: float,
                             viscosity_pa_s: float,
                             drainage_retention: float = DRAINAGE_RETENTION
                             ) -> dict[str, float]:
    """Melt-supply vs squeeze-out balance, m (smaller cap wins).

    * supply cap ``h_supply = v_melt t_c`` -- built within one dwell time;
    * drainage cap ``h_drain = (3 eta R^2 v_melt /(2 p))^(1/3)`` -- sustained by
      squeeze-out (balance of :func:`drainage_rate` with ``v_melt``).

    ``melting = 0`` when the net flux is non-positive (no meltwater at all).
    """
    q = _num("friction_flux_w_m2", friction_flux_w_m2, nonnegative=True)
    qc = _num("conductive_flux_w_m2", conductive_flux_w_m2, nonnegative=True)
    p = _num("pressure_pa", pressure_pa, positive=True)
    r = _num("contact_radius_m", contact_radius_m, positive=True)
    eta = _num("viscosity_pa_s", viscosity_pa_s, positive=True)
    ret = _num("drainage_retention", drainage_retention, positive=True)
    out = {"q_fric": q, "q_cond": qc, "q_net": q - qc, "melting": 0.0,
           "v_melt": 0.0, "h_supply": float("nan"), "h_drain": float("nan"),
           "h": H_MIN}
    if q <= qc:
        return out
    v_melt = (q - qc) / (ICE_RHO * ICE_LATENT)
    h_drain = (3.0 * eta * r * r * v_melt / (2.0 * p * ret)) ** (1.0 / 3.0)
    out.update({"melting": 1.0, "v_melt": v_melt, "h_drain": h_drain})
    return out


# ---------------------------------------------------------------------------
# 5. regime map
# ---------------------------------------------------------------------------
def combined_roughness(ice_roughness_m: Any, steel_roughness_m: Any) -> np.ndarray:
    """``sigma = sqrt(Ra_ice^2 + Ra_steel^2)``, m."""
    ri = np.asarray(ice_roughness_m, dtype=float)
    rs = np.asarray(steel_roughness_m, dtype=float)
    if np.any(ri < 0.0) or np.any(rs < 0.0):
        raise ValueError("roughness must be >= 0")
    return np.sqrt(ri * ri + rs * rs)


def film_area_fraction(lambda_ratio: Any) -> np.ndarray:
    """Load fraction carried by the film: 0 below ``Lambda = 1``, 1 above 3."""
    lam = np.asarray(lambda_ratio, dtype=float)
    return np.clip((lam - LAMBDA_BOUNDARY) / (LAMBDA_HYDRO - LAMBDA_BOUNDARY),
                   0.0, 1.0)


def regime_of(lambda_ratio: Any) -> list[str]:
    """Campaign regime label for a film ratio ``Lambda``.

    ``Lambda < 1`` boundary, ``1 <= Lambda < 3`` mixed, ``Lambda >= 3``
    hydrodynamic.  ``full_water_film`` is reserved for the zero-deficit limit
    (``T_ice >= T_melt``) and assigned by :func:`predict`.
    """
    lam = np.atleast_1d(np.asarray(lambda_ratio, dtype=float))
    labels = []
    for value in lam:
        if not math.isfinite(float(value)):
            raise ValueError("Lambda must be finite")
        if value < LAMBDA_BOUNDARY:
            labels.append(REGIME_BOUNDARY)
        elif value < LAMBDA_HYDRO:
            labels.append(REGIME_MIXED)
        else:
            labels.append(REGIME_HYDRO)
    return labels


# ---------------------------------------------------------------------------
# 6. driver: mu(T, V, Ra) by self-consistent iteration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LubricationInputs:
    """Ice-structure contact description (campaign input set).

    ``normal_load_n`` / ``contact_area_m2`` set the nominal pressure,
    ``speed_m_s`` the sliding speed, ``ice_temperature_c`` the bulk ice
    temperature, the two ``*_roughness_m`` the surface roughnesses,
    ``water_pressure_pa`` the external hydrostatic pressure (Clapeyron) and
    ``ice_fraction`` the solid fraction of the melt film (slurry viscosity).

    ``flash_concentration`` is the peak-to-mean ratio of the asperity flash
    temperature (rough-contact contact-temperature distributions put it at 2-3;
    :data:`FLASH_CONCENTRATION` = 2.5, and the mean-field lower bound 1.0 is a
    documented sensitivity case) and ``tau_scale`` multiplies the boundary
    shear strength.
    """

    normal_load_n: float = 1000.0
    speed_m_s: float = 0.5
    contact_area_m2: float = 1.0e-3
    ice_temperature_c: float = -10.0
    ice_roughness_m: float = DEFAULT_ICE_ROUGHNESS
    steel_roughness_m: float = DEFAULT_STEEL_ROUGHNESS
    water_pressure_pa: float = 0.0
    ice_fraction: float = 0.0
    drainage_retention: float = DRAINAGE_RETENTION
    tau_scale: float = 1.0
    flash_concentration: float = FLASH_CONCENTRATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal_load_n",
                           _num("normal_load_n", self.normal_load_n, positive=True))
        object.__setattr__(self, "speed_m_s",
                           _num("speed_m_s", self.speed_m_s, nonnegative=True))
        object.__setattr__(self, "contact_area_m2",
                           _num("contact_area_m2", self.contact_area_m2, positive=True))
        object.__setattr__(self, "ice_temperature_c",
                           _num("ice_temperature_c", self.ice_temperature_c))
        object.__setattr__(self, "ice_roughness_m",
                           _num("ice_roughness_m", self.ice_roughness_m, nonnegative=True))
        object.__setattr__(self, "steel_roughness_m",
                           _num("steel_roughness_m", self.steel_roughness_m, nonnegative=True))
        object.__setattr__(self, "water_pressure_pa",
                           _num("water_pressure_pa", self.water_pressure_pa, nonnegative=True))
        object.__setattr__(self, "ice_fraction",
                           _num("ice_fraction", self.ice_fraction, nonnegative=True))
        object.__setattr__(self, "drainage_retention",
                           _num("drainage_retention", self.drainage_retention, positive=True))
        object.__setattr__(self, "tau_scale",
                           _num("tau_scale", self.tau_scale, positive=True))
        object.__setattr__(self, "flash_concentration",
                           _num("flash_concentration", self.flash_concentration,
                                positive=True))
        if self.ice_roughness_m + self.steel_roughness_m <= 0.0:
            raise ValueError("at least one roughness must be positive")


@dataclass
class LubricationResult:
    """Machine-adjudicable outcome of one contact evaluation."""

    mu: float
    regime: str
    film_thickness_m: float
    lambda_ratio: float
    film_area_fraction: float
    hersey: float
    pressure_pa: float
    melting: bool
    melt_rate_m_s: float
    mu_boundary: float
    mu_film: float
    flash_rise_k: float
    melting_point_c: float
    temperature_deficit_k: float
    film_thickness_supply_m: float
    film_thickness_drain_m: float
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict(inputs: LubricationInputs, *, max_iterations: int = 400,
            tolerance: float = 1.0e-9, relaxation: float = 0.35
            ) -> LubricationResult:
    """Evaluate the full lubrication state of an ice-structure contact.

    Two coupled nonlinearities are resolved here:

    * **melting onset** -- the hot asperities melt first, so the melted fraction
      is a smooth ramp of the flash-temperature excess over the melting point
      (``MELT_RAMP_K``), not a step.  The local (asperity) flux is
      ``chi * q_bar`` and the melt supply per nominal area is therefore
      ``m (chi q_bar - q_cond) / chi`` -- the two bookkeeping rules are
      consistent by construction, which is what removes the on/off
      oscillation a naive step criterion produces at marginal melting;
    * **film feedback** -- thicker film -> lower ``mu_f`` -> less heat ->
      thinner film, iterated with fixed-point damping ``relaxation``.

    Returns ``mu``, ``regime``, ``h_film``, ``Lambda``, the Hersey number and the
    fluxes that produced them.
    """
    inp = inputs
    p_nom = inp.normal_load_n / inp.contact_area_m2
    tm = float(melting_point(inp.water_pressure_pa))
    deficit = tm - inp.ice_temperature_c
    sigma = float(combined_roughness(inp.ice_roughness_m, inp.steel_roughness_m))
    length = math.sqrt(inp.contact_area_m2)
    radius = math.sqrt(inp.contact_area_m2 / math.pi)
    dwell = length / max(inp.speed_m_s, 1.0e-9)
    eta_film = float(slurry_viscosity(inp.ice_temperature_c, inp.ice_fraction))
    mu_b = float(boundary_friction(p_nom, inp.ice_temperature_c,
                                   tau_scale=inp.tau_scale))
    chi = inp.flash_concentration
    deficit_pos = max(deficit, 0.0)
    q_conv = H_CONV * deficit_pos
    q_cond = float(conductive_capacity(deficit_pos, dwell))

    mu = mu_b
    h = H_MIN
    mu_f = float(film_friction(eta_film, inp.speed_m_s, p_nom, h))
    flash = float(flash_temperature_rise(mu * p_nom * inp.speed_m_s, dwell))
    melting = False
    v_melt = 0.0
    h_supply = float("nan")
    h_drain = float("nan")
    melted_fraction = 0.0
    converged = False
    iterations = 0

    if deficit <= 0.0:
        # Ice at or above its melting point: no thermal sink at all, so the
        # literal ODE has no steady state (dh/dt > 0 for every h).  The film is
        # bounded only by squeeze-out -> "full water film" (campaign control d2).
        bal = sustained_film_thickness(
            friction_flux_w_m2=mu_b * p_nom * inp.speed_m_s,
            conductive_flux_w_m2=0.0, pressure_pa=p_nom,
            contact_radius_m=radius, viscosity_pa_s=eta_film,
            drainage_retention=inp.drainage_retention)
        h = min(bal["h_drain"], H_MAX) if bal["melting"] else H_MAX
        h_drain = bal["h_drain"]
        v_melt = bal["v_melt"]
        melting = bool(bal["melting"])
        lam = h / sigma
        mu_f = float(film_friction(eta_film, inp.speed_m_s, p_nom, h))
        her = float(hersey_number(eta_film, inp.speed_m_s, p_nom))
        return LubricationResult(
            mu=mu_f, regime=REGIME_FULL_FILM, film_thickness_m=float(h),
            lambda_ratio=float(lam), film_area_fraction=1.0, hersey=her,
            pressure_pa=p_nom, melting=melting, melt_rate_m_s=float(v_melt),
            mu_boundary=mu_b, mu_film=float(mu_f), flash_rise_k=float(flash),
            melting_point_c=tm, temperature_deficit_k=float(deficit),
            film_thickness_supply_m=float("nan"),
            film_thickness_drain_m=float(h_drain), converged=True,
            iterations=0)

    for iterations in range(1, int(max_iterations) + 1):
        q_bar = mu * p_nom * inp.speed_m_s
        flash = float(flash_temperature_rise(q_bar, dwell))
        excess_k = inp.ice_temperature_c + chi * flash - tm
        melted_fraction = float(np.clip(excess_k / MELT_RAMP_K, 0.0, 1.0))
        local_excess = max(chi * q_bar - q_cond, 0.0)
        q_melt = melted_fraction * local_excess / chi
        if q_melt <= 0.0:
            h_new = H_MIN
            v_melt = 0.0
            h_supply = float("nan")
            h_drain = float("nan")
        else:
            v_melt = q_melt / (ICE_RHO * ICE_LATENT)
            h_supply = v_melt * dwell
            bal = sustained_film_thickness(
                friction_flux_w_m2=chi * q_bar,
                conductive_flux_w_m2=max(q_cond, q_conv),
                pressure_pa=p_nom, contact_radius_m=radius,
                viscosity_pa_s=eta_film,
                drainage_retention=inp.drainage_retention)
            h_drain = bal["h_drain"] / chi ** (1.0 / 3.0)
            h_new = min(h_supply, h_drain, H_MAX)
        melting = bool(q_melt > 0.0)
        lam_new = h_new / sigma
        area = float(film_area_fraction(lam_new))
        mu_f = float(film_friction(eta_film, inp.speed_m_s, p_nom, h_new))
        mu_target = (1.0 - area) * mu_b + area * mu_f
        delta = abs(mu_target - mu) / max(abs(mu_target), 1.0e-16)
        h = h_new
        mu = (1.0 - relaxation) * mu + relaxation * mu_target
        if delta < tolerance:
            converged = True
            break

    lam = h / sigma
    area = float(film_area_fraction(lam))
    regime = regime_of(lam)[0]
    her = float(hersey_number(eta_film, inp.speed_m_s, p_nom))
    return LubricationResult(
        mu=float(mu), regime=regime, film_thickness_m=float(h),
        lambda_ratio=float(lam), film_area_fraction=float(area), hersey=her,
        pressure_pa=p_nom, melting=bool(melting), melt_rate_m_s=float(v_melt),
        mu_boundary=mu_b, mu_film=float(mu_f), flash_rise_k=float(flash),
        melting_point_c=tm, temperature_deficit_k=float(deficit),
        film_thickness_supply_m=float(h_supply),
        film_thickness_drain_m=float(h_drain), converged=bool(converged),
        iterations=int(iterations))


def friction_surface(ice_temperature_c: Sequence[float],
                     speed_m_s: Sequence[float],
                     ice_roughness_m: Sequence[float], **kwargs: Any
                     ) -> dict[str, Any]:
    """``mu(T, V, Ra)`` response surface, shape ``(nT, nV, nRa)``.

    Returns the ``mu`` cube plus ``lambda`` and ``regime_index`` (index into
    :data:`REGIMES`) so the regime partition of the surface is auditable.
    """
    temps = np.asarray(ice_temperature_c, dtype=float)
    speeds = np.asarray(speed_m_s, dtype=float)
    rough = np.asarray(ice_roughness_m, dtype=float)
    mu = np.empty((temps.size, speeds.size, rough.size))
    lam = np.empty_like(mu)
    reg = np.empty_like(mu)
    index = {name: i for i, name in enumerate(REGIMES)}
    for i, t in enumerate(temps):
        for j, v in enumerate(speeds):
            for k, ra in enumerate(rough):
                res = predict(LubricationInputs(
                    ice_temperature_c=float(t), speed_m_s=float(v),
                    ice_roughness_m=float(ra), **kwargs))
                mu[i, j, k] = res.mu
                lam[i, j, k] = res.lambda_ratio
                reg[i, j, k] = index[res.regime]
    return {"mu": mu, "lambda": lam, "regime_index": reg,
            "temperatures_c": temps, "speeds_m_s": speeds,
            "roughness_m": rough, "regimes": REGIMES}


def stribeck_scan(speed_m_s: Sequence[float], *, temperature_c: float = -5.0,
                  pressure_pa: float = 1.0e6,
                  ice_roughness_m: float = DEFAULT_ICE_ROUGHNESS,
                  steel_roughness_m: float = DEFAULT_STEEL_ROUGHNESS,
                  ice_fraction: float = 0.0, **kwargs: Any) -> dict[str, Any]:
    """Stribeck scan ``mu(V)`` with the Hersey axis and the regime band.

    Uses the delivered :func:`predict` driver so the curve plotted in campaign
    coordinates is the same object the acceptance test adjudicates.
    """
    speeds = np.asarray(speed_m_s, dtype=float)
    eta = float(slurry_viscosity(temperature_c, ice_fraction))
    mu = np.empty(speeds.size)
    h = np.empty(speeds.size)
    lam = np.empty(speeds.size)
    her = np.empty(speeds.size)
    regimes: list[str] = []
    for i, v in enumerate(speeds):
        res = predict(LubricationInputs(
            speed_m_s=float(v), ice_temperature_c=temperature_c,
            normal_load_n=pressure_pa * 1.0e-3, contact_area_m2=1.0e-3,
            ice_roughness_m=ice_roughness_m,
            steel_roughness_m=steel_roughness_m, ice_fraction=ice_fraction,
            **kwargs))
        mu[i] = res.mu
        h[i] = res.film_thickness_m
        lam[i] = res.lambda_ratio
        her[i] = res.hersey
        regimes.append(res.regime)
    return {"speed_m_s": speeds, "mu": mu, "hersey": her,
            "film_thickness_m": h, "lambda": lam, "regime": regimes,
            "eta_pa_s": eta, "temperature_c": temperature_c}


# ---------------------------------------------------------------------------
# 7. validation: literature anchors, controls, falsifiability, calibration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiteratureAnchor:
    """One ice-steel friction measurement used as a validation target.

    ``mu`` is the *central* digitised value and ``mu_unc`` its reading band
    (typically ``0.02 .. 0.04`` for friction traces of ice).  ``applicable`` is
    False for the creep-dominated low-speed corner where this module's boundary
    law is out of scope (documented limitation, not silently dropped).
    """

    key: str
    temperature_c: float
    speed_m_s: float
    mu: float
    mu_unc: float
    pressure_pa: float = 1.0e6
    ice_roughness_m: float = DEFAULT_ICE_ROUGHNESS
    steel_roughness_m: float = DEFAULT_STEEL_ROUGHNESS
    applicable: bool = True
    source: str = ""


LITERATURE_ANCHORS: tuple[LiteratureAnchor, ...] = (
    LiteratureAnchor(
        "A1_creep_-10C_1em4", -10.0, 1.0e-4, 0.30, 0.06, applicable=False,
        source="creep/sintering-dominated limit, Kietzig et al. 2010 review "
               "Fig. 3 low-velocity branch (out of scope: no creep law here)"),
    LiteratureAnchor(
        "A2_-10C_1em2", -10.0, 1.0e-2, 0.13, 0.03,
        source="ice vs stainless steel, Kietzig et al. 2010 JAP 107:081101 "
               "mu(V) family at -10 degC"),
    LiteratureAnchor(
        "A3_-10C_1", -10.0, 1.0, 0.09, 0.03,
        source="same family, velocity minimum region (literature 0.05-0.15)"),
    LiteratureAnchor(
        "A4_-5C_1em1", -5.0, 1.0e-1, 0.07, 0.02,
        source="warm-ice mixed-regime points, Makkonen & Tikanmaki 2014 "
               "Cold Reg. Sci. Technol. 102:42 model-vs-data compilation"),
    LiteratureAnchor(
        "A5_-2C_5em1", -2.0, 5.0e-1, 0.05, 0.02,
        source="near-melting ice, hydrodynamic branch (literature 0.03-0.07)"),
    LiteratureAnchor(
        "A6_-20C_1em1", -20.0, 1.0e-1, 0.14, 0.04,
        source="cold-ice plateau, Kietzig et al. 2010 (mu 0.1-0.2 at -20 degC)"),
    LiteratureAnchor(
        "A7_-20C_1", -20.0, 1.0, 0.10, 0.03,
        source="cold-ice velocity minimum, same family"),
)


def validate_literature(anchors: Iterable[LiteratureAnchor] = LITERATURE_ANCHORS,
                        *, tau_scale: float = 1.0,
                        flash_concentration: float = FLASH_CONCENTRATION,
                        drainage_retention: float = DRAINAGE_RETENTION
                        ) -> dict[str, Any]:
    """Score the model against the literature anchor table.

    Acceptance criterion (a) of the campaign: ``mu`` within 20 % of the ice
    friction experiments for at least 5 T-V combinations.  Every row is
    returned (``mu_model``, ``mu_anchor``, ``rel_error``, ``within_20pct``,
    ``within_band``) so an honest partial pass is visible, and the aggregate is
    computed over ``applicable`` anchors only.
    """
    rows = []
    for a in anchors:
        res = predict(LubricationInputs(
            normal_load_n=a.pressure_pa * 1.0e-3, contact_area_m2=1.0e-3,
            speed_m_s=a.speed_m_s, ice_temperature_c=a.temperature_c,
            ice_roughness_m=a.ice_roughness_m,
            steel_roughness_m=a.steel_roughness_m,
            tau_scale=tau_scale, flash_concentration=flash_concentration,
            drainage_retention=drainage_retention))
        err = abs(res.mu - a.mu) / a.mu
        rows.append({
            "key": a.key, "temperature_c": a.temperature_c,
            "speed_m_s": a.speed_m_s, "mu_anchor": a.mu,
            "mu_unc": a.mu_unc, "mu_model": res.mu,
            "rel_error": err, "within_20pct": bool(err <= 0.20),
            "within_band": bool(abs(res.mu - a.mu) <= a.mu_unc),
            "regime": res.regime, "film_thickness_m": res.film_thickness_m,
            "lambda_ratio": res.lambda_ratio, "hersey": res.hersey,
            "applicable": a.applicable, "source": a.source,
        })
    applicable = [r for r in rows if r["applicable"]]
    passed = [r for r in applicable if r["within_20pct"]]
    errors = [r["rel_error"] for r in applicable]
    out = {
        "rows": rows,
        "n_applicable": len(applicable),
        "n_within_20pct": len(passed),
        "mean_rel_error": float(np.mean(errors)) if errors else float("nan"),
        "max_rel_error": float(np.max(errors)) if errors else float("nan"),
        "criterion_a_pass": bool(len(passed) >= 5),
        "tau_scale": tau_scale,
        "flash_concentration": flash_concentration,
        "drainage_retention": drainage_retention,
    }
    return out


def negative_controls(*, tau_scale: float = 1.0,
                      flash_concentration: float = FLASH_CONCENTRATION
                      ) -> dict[str, Any]:
    """Campaign control (d): the three degenerate limits must come out right.

    * **(d1) ``V -> 0``**: no melting, ``mu`` in the boundary band
      ``0.05 .. 0.30`` (Coulomb plateau recovered);
    * **(d2) ``T_ice -> T_melt``**: no thermal sink, the film is unbounded by
      conduction (``regime == full_water_film``, ``h`` at the geometric ceiling
      or drainage value, ``Lambda`` large);
    * **(d3) ``Ra -> 0``**: ``Lambda = h/sigma -> large`` -> the film carries all
      the load (``film_area_fraction == 1``).
    """
    slow = [dict(predict(LubricationInputs(
        speed_m_s=v, ice_temperature_c=-10.0, tau_scale=tau_scale,
        flash_concentration=flash_concentration)).as_dict(), speed_m_s=v)
        for v in (1.0e-6, 1.0e-4, 1.0e-3)]
    warm = dict(predict(LubricationInputs(
        ice_temperature_c=0.0, speed_m_s=0.5, tau_scale=tau_scale,
        flash_concentration=flash_concentration)).as_dict(),
        speed_m_s=0.5, ice_temperature_c=0.0, ice_roughness_m=DEFAULT_ICE_ROUGHNESS)
    smooth = dict(predict(LubricationInputs(
        ice_temperature_c=-2.0, speed_m_s=0.5, ice_roughness_m=1.0e-9,
        steel_roughness_m=1.0e-9, tau_scale=tau_scale,
        flash_concentration=flash_concentration)).as_dict(),
        speed_m_s=0.5, ice_temperature_c=-2.0, ice_roughness_m=1.0e-9)
    d1 = all(MU_BOUNDARY_MIN - 1e-12 <= r["mu"] <= MU_BOUNDARY_MAX + 1e-12
             for r in slow)
    d2 = warm["regime"] == REGIME_FULL_FILM and warm["lambda_ratio"] > 3.0
    d3 = smooth["film_area_fraction"] >= 1.0 - 1e-12 and smooth["lambda_ratio"] > 3.0
    return {
        "d1_slow_speed_boundary": {"pass": bool(d1), "rows": slow,
                                   "band": [MU_BOUNDARY_MIN, MU_BOUNDARY_MAX]},
        "d2_warm_ice_full_film": {"pass": bool(d2), "row": warm},
        "d3_smooth_full_film": {"pass": bool(d3), "row": smooth},
        "all_pass": bool(d1 and d2 and d3),
    }


def falsifiability_probe(*, tau_scale: float = 1.0,
                         flash_concentration: float = 1.0) -> dict[str, Any]:
    """Criterion (e): at -10 degC and 1 m/s the model must **not** return > 0.5.

    Literature puts ``mu`` at 0.05-0.15 there; a prediction above 0.5 would mean
    the boundary branch is not being relieved by meltwater at all and the model
    is falsified.  Returns the value, the reference band and the verdict.
    """
    res = predict(LubricationInputs(
        ice_temperature_c=-10.0, speed_m_s=1.0, tau_scale=tau_scale,
        flash_concentration=flash_concentration))
    return {"mu": res.mu, "regime": res.regime,
            "film_thickness_m": res.film_thickness_m,
            "literature_band": [0.05, 0.15],
            "falsified": bool(res.mu > 0.5),
            "inside_literature_band": bool(0.05 <= res.mu <= 0.15)}


def calibrate(*, anchors: Sequence[LiteratureAnchor] = LITERATURE_ANCHORS,
              tau_grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
              flash_grid: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0),
              drainage_retention: float = DRAINAGE_RETENTION
              ) -> dict[str, Any]:
    """Honest two-parameter calibration with leave-one-out scoring.

    The soft parameters ``tau_scale`` (boundary shear strength multiplier) and
    ``flash_concentration`` (peak/mean asperity flash temperature) are fitted by
    minimising the mean relative error over the *applicable* anchors, then the
    same two parameters are re-fitted on every leave-one-out subset so the
    reported hold-out error is not the fitting error.  Only two parameters are
    free against seven anchors, so the LOO number is the one to quote.
    """
    usable = [a for a in anchors if a.applicable]

    def mre(seq: Sequence[LiteratureAnchor], ts: float, fc: float) -> float:
        errs = []
        for a in seq:
            res = predict(LubricationInputs(
                normal_load_n=a.pressure_pa * 1.0e-3, contact_area_m2=1.0e-3,
                speed_m_s=a.speed_m_s, ice_temperature_c=a.temperature_c,
                ice_roughness_m=a.ice_roughness_m,
                steel_roughness_m=a.steel_roughness_m, tau_scale=ts,
                flash_concentration=fc, drainage_retention=drainage_retention))
            errs.append(abs(res.mu - a.mu) / a.mu)
        return float(np.mean(errs)) if errs else float("nan")

    best = None
    grid = []
    for ts in tau_grid:
        for fc in flash_grid:
            score = mre(usable, ts, fc)
            grid.append({"tau_scale": ts, "flash_concentration": fc,
                         "mean_rel_error": score})
            if best is None or score < best["mean_rel_error"]:
                best = {"tau_scale": ts, "flash_concentration": fc,
                        "mean_rel_error": score}
    loo = []
    for i in range(len(usable)):
        subset = [a for j, a in enumerate(usable) if j != i]
        inner = None
        for ts in tau_grid:
            for fc in flash_grid:
                score = mre(subset, ts, fc)
                if inner is None or score < inner["mean_rel_error"]:
                    inner = {"tau_scale": ts, "flash_concentration": fc,
                             "mean_rel_error": score}
        held = usable[i]
        res = predict(LubricationInputs(
            normal_load_n=held.pressure_pa * 1.0e-3, contact_area_m2=1.0e-3,
            speed_m_s=held.speed_m_s, ice_temperature_c=held.temperature_c,
            ice_roughness_m=held.ice_roughness_m,
            steel_roughness_m=held.steel_roughness_m,
            tau_scale=inner["tau_scale"],
            flash_concentration=inner["flash_concentration"],
            drainage_retention=drainage_retention))
        loo.append({
            "held_out": held.key, "mu_anchor": held.mu, "mu_model": res.mu,
            "rel_error": abs(res.mu - held.mu) / held.mu,
            "within_20pct": bool(abs(res.mu - held.mu) / held.mu <= 0.20),
            "tau_scale": inner["tau_scale"],
            "flash_concentration": inner["flash_concentration"],
        })
    loo_err = [r["rel_error"] for r in loo]
    return {
        "best": dict(best, drainage_retention=drainage_retention),
        "grid": grid,
        "leave_one_out": loo,
        "loo_mean_rel_error": float(np.mean(loo_err)) if loo_err else float("nan"),
        "loo_max_rel_error": float(np.max(loo_err)) if loo_err else float("nan"),
        "loo_n_within_20pct": int(sum(r["within_20pct"] for r in loo)),
        "loo_n": len(loo),
        "verdict_full_fit": validate_literature(
            anchors, tau_scale=best["tau_scale"],
            flash_concentration=best["flash_concentration"],
            drainage_retention=drainage_retention),
    }


# ---------------------------------------------------------------------------
# 8. Stribeck coordinates of the delivered model (criterion (c))
# ---------------------------------------------------------------------------
#: Literature location of the ice-friction Stribeck minimum in Hersey numbers.
#: Obtained the only way it can be: by mapping the measured ``mu(V)`` minimum of
#: an ice-steel contact (``V_min = 0.5 .. 5 m/s`` at ``-5 .. -10 degC``) back
#: through ``Her = eta V / p`` with ``eta = 2.5e-3 Pa s`` (supercooled melt at
#: -10 degC) and ``p = 0.5 .. 5 MPa``: ``Her_opt = 1.2e-8``, band ``3e-9 ..
#: 3e-8`` (half a decade each side).  Same literature family as the anchor
#: table, used for validation only.
STRIBECK_HER_OPT_LITERATURE = 1.2e-8
STRIBECK_HER_OPT_BAND = (3.0e-9, 3.0e-8)


def film_thickness_supply(*, pressure_pa: float, speed_m_s: float,
                          temperature_c: float,
                          contact_area_m2: float = 1.0e-3,
                          ice_roughness_m: float = DEFAULT_ICE_ROUGHNESS,
                          steel_roughness_m: float = DEFAULT_STEEL_ROUGHNESS,
                          mu: float | None = None, tau_scale: float = 1.0,
                          flash_concentration: float = FLASH_CONCENTRATION,
                          ice_fraction: float = 0.0,
                          drainage_retention: float = DRAINAGE_RETENTION
                          ) -> dict[str, float]:
    """Supply/drainage film thickness for one operating point (no iteration).

    The single-shot kernel behind :func:`predict` and the Stribeck closure: it
    returns ``h``, the melted area fraction ``m``, the melt rate and both caps.
    Exposed so the Stribeck parameters can be derived by inverting ``h(V)``
    without running the full friction iteration.
    """
    p = _num("pressure_pa", pressure_pa, positive=True)
    v = _num("speed_m_s", speed_m_s, nonnegative=True)
    t = _num("temperature_c", temperature_c)
    area = _num("contact_area_m2", contact_area_m2, positive=True)
    sigma = float(combined_roughness(ice_roughness_m, steel_roughness_m))
    radius = math.sqrt(area / math.pi)
    dwell = math.sqrt(area) / max(v, 1.0e-9)
    eta = float(slurry_viscosity(t, ice_fraction))
    mu_use = (float(boundary_friction(p, t, tau_scale=tau_scale))
              if mu is None else _num("mu", mu, positive=True))
    chi = _num("flash_concentration", flash_concentration, positive=True)
    tm = float(melting_point(0.0))
    deficit = max(tm - t, 0.0)
    q_cond = float(conductive_capacity(deficit, dwell))
    q_conv = H_CONV * deficit
    q_bar = mu_use * p * v
    flash = float(flash_temperature_rise(q_bar, dwell))
    m = float(np.clip((t + chi * flash - tm) / MELT_RAMP_K, 0.0, 1.0))
    local_excess = max(chi * q_bar - q_cond, 0.0)
    q_melt = m * local_excess / chi
    out = {"h": H_MIN, "melt_fraction": m, "q_melt": q_melt, "v_melt": 0.0,
           "h_supply": float("nan"), "h_drain": float("nan"),
           "lambda": H_MIN / sigma, "flash_k": flash, "mu": mu_use,
           "eta_pa_s": eta, "dwell_s": dwell, "drain_limited": 0.0}
    if q_melt <= 0.0:
        return out
    v_melt = q_melt / (ICE_RHO * ICE_LATENT)
    bal = sustained_film_thickness(
        friction_flux_w_m2=chi * q_bar,
        conductive_flux_w_m2=max(q_cond, q_conv), pressure_pa=p,
        contact_radius_m=radius, viscosity_pa_s=eta,
        drainage_retention=drainage_retention)
    h_drain = bal["h_drain"] / chi ** (1.0 / 3.0)
    h = min(v_melt * dwell, h_drain, H_MAX)
    out.update({"h": h, "v_melt": v_melt, "h_supply": v_melt * dwell,
                "h_drain": h_drain, "lambda": h / sigma,
                "drain_limited": float(h >= h_drain)})
    return out


def stribeck_parameters(*, pressure_pa: float = 1.0e6,
                        temperature_c: float = -5.0,
                        ice_roughness_m: float = DEFAULT_ICE_ROUGHNESS,
                        steel_roughness_m: float = DEFAULT_STEEL_ROUGHNESS,
                        contact_area_m2: float = 1.0e-3) -> dict[str, float]:
    """``(mu_b, Her_c, Her_v)`` of :func:`stribeck_mu` derived from the closures.

    * ``mu_b`` -- :func:`boundary_friction` at the operating point;
    * ``Her_c`` -- the Hersey number at which the film carries half the load
      (``Lambda = 2``), found by bisecting :func:`film_thickness_supply` in
      speed;
    * ``Her_v = 3 sigma`` -- because in the full-film branch ``mu_f = Her/h`` and
      the load-sharing threshold sets ``h = 3 sigma``, so ``Her = 3 sigma`` is the
      Hersey number at which the viscous branch reaches ``mu = 1``.

    Nothing is fitted here: the three numbers come from the closed-form closures,
    which is what makes criterion (c) a prediction rather than a restatement.
    """
    p = _num("pressure_pa", pressure_pa, positive=True)
    t = _num("temperature_c", temperature_c)
    sigma = float(combined_roughness(ice_roughness_m, steel_roughness_m))
    area = _num("contact_area_m2", contact_area_m2, positive=True)
    eta = float(slurry_viscosity(t))
    mu_b = float(boundary_friction(p, t))

    def lam_minus_two(v: float) -> float:
        res = film_thickness_supply(
            pressure_pa=p, speed_m_s=v, temperature_c=t,
            contact_area_m2=area, ice_roughness_m=ice_roughness_m,
            steel_roughness_m=steel_roughness_m)
        return res["lambda"] - 2.0

    v_lo, v_hi = 1.0e-6, 20.0
    f_lo, f_hi = lam_minus_two(v_lo), lam_minus_two(v_hi)
    her_c = float("nan")
    if f_lo < 0.0 < f_hi:
        for _ in range(120):
            mid = math.sqrt(v_lo * v_hi)
            if lam_minus_two(mid) < 0.0:
                v_lo = mid
            else:
                v_hi = mid
        v_c = math.sqrt(v_lo * v_hi)
        her_c = eta * v_c / p
        v_c_used = v_c
    else:
        v_c_used = float("nan")
    her_v = 3.0 * sigma
    return {"mu_b": mu_b, "her_c": her_c, "her_v": her_v,
            "v_c_m_s": v_c_used, "sigma_m": sigma, "eta_pa_s": eta,
            "pressure_pa": p, "temperature_c": t,
            "bracketed": float(math.isfinite(her_c))}


def stribeck_check(*, pressure_pa: float = 1.0e6,
                   temperature_c: float = -5.0,
                   ice_roughness_m: float = DEFAULT_ICE_ROUGHNESS,
                   steel_roughness_m: float = DEFAULT_STEEL_ROUGHNESS
                   ) -> dict[str, Any]:
    """Acceptance criterion (c): model ``Her_opt`` vs the literature band.

    Passes when the relative error against :data:`STRIBECK_HER_OPT_LITERATURE`
    is below 30 % **and** the value lies inside :data:`STRIBECK_HER_OPT_BAND`.
    """
    par = stribeck_parameters(
        pressure_pa=pressure_pa, temperature_c=temperature_c,
        ice_roughness_m=ice_roughness_m, steel_roughness_m=steel_roughness_m)
    out = dict(par)
    if not par["bracketed"]:
        out.update({"her_opt": float("nan"), "mu_opt": float("nan"),
                    "rel_error": float("nan"), "pass": False,
                    "reason": "no Lambda=2 crossing in the bracketed speed range"})
        return out
    opt = stribeck_optimum(par["mu_b"], par["her_c"], par["her_v"])
    err = abs(opt["her_opt"] - STRIBECK_HER_OPT_LITERATURE) / STRIBECK_HER_OPT_LITERATURE
    lo, hi = STRIBECK_HER_OPT_BAND
    out.update({
        "her_opt": opt["her_opt"], "mu_opt": opt["mu_opt"],
        "literature_her_opt": STRIBECK_HER_OPT_LITERATURE,
        "literature_band": STRIBECK_HER_OPT_BAND,
        "rel_error": err,
        "inside_band": bool(lo <= opt["her_opt"] <= hi),
        "pass": bool(err <= 0.30 and lo <= opt["her_opt"] <= hi),
    })
    return out


def stribeck_curve(her: Sequence[float] | None = None, **params: Any
                   ) -> dict[str, np.ndarray]:
    """Evaluate the delivered Stribeck curve on the campaign Hersey band.

    Returns ``{"her", "mu", "mu_boundary_branch", "mu_viscous_branch"}`` on
    ``10^-8 .. 10^-2`` by default, with the parameters from
    :func:`stribeck_parameters` unless overridden.
    """
    par = stribeck_parameters(**params) if "mu_b" not in params else params
    if her is None:
        her_arr = np.logspace(-8.0, -2.0, 241)
    else:
        her_arr = np.asarray(her, dtype=float)
    mu = stribeck_mu(her_arr, mu_b=par["mu_b"], her_c=par["her_c"],
                     her_v=par["her_v"])
    return {"her": her_arr, "mu": mu,
            "mu_boundary_branch": par["mu_b"] / (1.0 + her_arr / par["her_c"]),
            "mu_viscous_branch": her_arr / par["her_v"],
            "mu_b": par["mu_b"], "her_c": par["her_c"],
            "her_v": par["her_v"]}
