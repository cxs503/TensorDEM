"""Moist-particle convective drying closures: Ranz-Marshall dual analogy.

Physics implemented
-------------------
1. **Convective transfer coefficients** for a sphere in a gas stream:
   Ranz & Marshall (1952)

       Nu = 2 + 0.6 Re^0.5 Pr^(1/3)
       Sh = 2 + 0.6 Re^0.5 Sc^(1/3)

   giving  h = Nu k_f/d_p  and  h_m = Sh D_AB/d_p, with the film properties
   evaluated at T_f = (T_p + T_g)/2 and the mixture humidity.

2. **Evaporation mass flux** from the particle surface

       mdot = h_m A_p [ a_w(X) rho_v,sat(T_p) - rho_v,g ]

   with rho_v,sat the saturated vapour density at the particle temperature and
   rho_v,g the free-stream vapour density.  ``a_w`` is the water activity:
   a_w = 1 in the constant-rate period, a_w = exp[-kappa (X - X_c)] in the
   falling-rate period (exponential sorption isotherm, X on a dry basis).

3. **Particle energy equation** with the latent-heat sink and the enthalpy of
   the leaving liquid (the CP_L terms cancel, leaving only the latent term):

       (m_s c_ps + m_w c_pl) dT_p/dt = h A_p (T_g - T_p) - mdot h_fg(T_p)

4. **Wet-bulb pinning**: the constant-rate fixed point solves

       h (T_g - T*) = h_m h_fg(T*) [rho_v,sat(T*) - rho_v,g]

   ("psychrometric closed form with Lewis factor").  When the diffusivity is
   overridden to D_AB = alpha (Le = 1), this root coincides with the classical
   chart wet bulb of :func:`tensordem.rotary_dryer.psychrometrics.wet_bulb_temperature`
   to <0.05 K -- the analogy hook is what makes the degeneration testable.

5. **Thin-layer kinetics**: Lewis (single exponential) and Page models

       Lewis: X(t) = X_e + (X_0 - X_e) exp(-k t)
       Page : X(t) = X_e + (X_0 - X_e) exp[-(k t)^n]

   fitted to the simulated falling-rate curve, with R^2 / RMSE reported.

References
----------
Ranz, W.E., Marshall, W.R., 1952. Evaporation from drops. Chem. Eng. Prog. 48.
Whitaker, S., 1972. Forced convection heat transfer correlations for flow in
    pipes, past flat plates, single cylinders, single spheres ... AIChE J. 18.
Mujumdar, A.S., 2014. Handbook of Industrial Drying, 4th ed., CRC Press.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .psychrometrics import (
    CP_L,
    CP_V,
    P_STD,
    R_V,
    air_conductivity,
    air_viscosity,
    humid_heat_capacity,
    humidity_ratio_from_vapour_pressure,
    latent_heat,
    moist_air_cp,
    moist_air_density,
    saturation_pressure,
    temperature_from_enthalpy,
    vapour_density,
    vapour_pressure_from_humidity_ratio,
    water_vapour_diffusivity,
    wet_bulb_temperature,
)

__all__ = [
    "nusselt_ranz_marshall",
    "sherwood_ranz_marshall",
    "nusselt_whitaker",
    "sherwood_whitaker",
    "driving_force",
    "film_properties",
    "transfer_coefficients",
    "sorption_activity",
    "evaporation_rate",
    "wet_bulb_residual_closures",
    "wet_bulb_from_closures",
    "ParticleDryingState",
    "particle_derivatives",
    "simulate_single_particle",
    "page_model",
    "lewis_model",
    "fit_thin_layer",
    "specific_surface_area",
    "DRY_AIR_CHAIN",
]


# ---------------------------------------------------------------------------
# dimensionless closures
# ---------------------------------------------------------------------------
def nusselt_ranz_marshall(Re: float, Pr: float) -> float:
    """Nu = 2 + 0.6 Re^0.5 Pr^(1/3) (Ranz & Marshall 1952)."""
    if Re < 0.0:
        raise ValueError("Re must be >= 0")
    if Pr <= 0.0:
        raise ValueError("Pr must be > 0")
    return 2.0 + 0.6 * math.sqrt(Re) * Pr ** (1.0 / 3.0)


def sherwood_ranz_marshall(Re: float, Sc: float) -> float:
    """Sh = 2 + 0.6 Re^0.5 Sc^(1/3) (Ranz & Marshall 1952).

    The Re -> 0 limit is Sh = 2, the exact solution of steady diffusion from a
    sphere into a stagnant medium (Maxwell / Langmuir).
    """
    if Re < 0.0:
        raise ValueError("Re must be >= 0")
    if Sc <= 0.0:
        raise ValueError("Sc must be > 0")
    return 2.0 + 0.6 * math.sqrt(Re) * Sc ** (1.0 / 3.0)


def nusselt_whitaker(Re: float, Pr: float) -> float:
    """Whitaker (1972) sphere correlation, used as an independent cross-check.

    Nu = 2 + (0.4 Re^0.5 + 0.06 Re^(2/3)) Pr^0.4, 0.71 < Pr < 380.
    """
    if Re < 0.0:
        raise ValueError("Re must be >= 0")
    return 2.0 + (0.4 * math.sqrt(Re) + 0.06 * Re ** (2.0 / 3.0)) * Pr ** 0.4


def sherwood_whitaker(Re: float, Sc: float) -> float:
    """Whitaker (1972) mass-transfer analogue Sh = 2 + (0.4 Re^.5 + 0.06 Re^(2/3)) Sc^0.4."""
    if Re < 0.0:
        raise ValueError("Re must be >= 0")
    return 2.0 + (0.4 * math.sqrt(Re) + 0.06 * Re ** (2.0 / 3.0)) * Sc ** 0.4


# ---------------------------------------------------------------------------
# film properties and transfer coefficients
# ---------------------------------------------------------------------------
def film_properties(T_g: float, T_p: float, Y: float, p: float = P_STD) -> Dict[str, float]:
    """Gas properties at the film temperature T_f = (T_g + T_p)/2."""
    T_f = 0.5 * (T_g + T_p)
    mu = air_viscosity(T_f)
    k = air_conductivity(T_f)
    D = water_vapour_diffusivity(T_f, p)
    rho = moist_air_density(T_f, Y, p)
    cp = moist_air_cp(Y)
    return {
        "T_f": T_f,
        "mu": mu,
        "k": k,
        "D_ab": D,
        "rho": rho,
        "cp": cp,
        "alpha": k / (rho * cp),
        "nu": mu / rho,
        "Pr": mu * cp / k,
        "Sc": mu / (rho * D),
    }


def transfer_coefficients(
    T_g: float,
    T_p: float,
    Y: float,
    u_rel: float,
    d_p: float,
    p: float = P_STD,
    d_ab_override: Optional[object] = None,
) -> Dict[str, float]:
    """Heat and mass transfer coefficients for a sphere at relative speed u_rel.

    Parameters
    ----------
    T_g, T_p : float
        Gas and particle temperature (K); the film state is their mean.
    Y : float
        Free-stream humidity ratio (kg/kg_da) used for the film mixture.
    u_rel : float
        Relative gas-particle speed (m/s).
    d_p : float
        Particle diameter (m).
    p : float
        Absolute pressure (Pa).
    d_ab_override : float or "alpha", optional
        Replace D_AB by this value.  The string ``"alpha"`` uses the local
        thermal diffusivity alpha = k/(rho cp) at the *film* temperature, which
        forces Sc = Pr (Le = 1) and turns the dual closure into the classical
        Lewis relation (this is the verification hook for the wet-bulb limit).

    Returns
    -------
    dict with Re, Pr, Sc, Nu, Sh, h [W/(m^2 K)], h_m [m/s], and film props.
    """
    if d_p <= 0.0:
        raise ValueError("d_p must be > 0")
    props = film_properties(T_g, T_p, Y, p)
    if d_ab_override is None:
        D = props["D_ab"]
    elif isinstance(d_ab_override, str) and d_ab_override == "alpha":
        D = props["alpha"]
    else:
        D = float(d_ab_override)
    Sc = props["mu"] / (props["rho"] * D)
    Re = props["rho"] * abs(u_rel) * d_p / props["mu"]
    Pr = props["Pr"]
    Nu = nusselt_ranz_marshall(Re, Pr)
    Sh = sherwood_ranz_marshall(Re, Sc)
    h = Nu * props["k"] / d_p
    h_m = Sh * D / d_p
    props.update(
        {
            "Re": Re,
            "Pr": Pr,
            "Sc": Sc,
            "Nu": Nu,
            "Sh": Sh,
            "h": h,
            "h_m": h_m,
            "D_ab_used": D,
            "lewis_factor": h_m / (h / (props["rho"] * props["cp"])),
        }
    )
    return props


# ---------------------------------------------------------------------------
# sorption activity and mass flux
# ---------------------------------------------------------------------------
def sorption_activity(X: float, X_e: float, X_c: float, kappa: float) -> float:
    """Water activity a_w in X (dry-basis moisture).

    * X <= X_e          -> 0 (equilibrium moisture, no further drying)
    * X >= X_c          -> 1 (free water, constant-rate period)
    * otherwise         -> normalised exponential isotherm

          a_w = [exp(-kappa (X_c - X)) - exp(-kappa (X_c - X_e))]
                / [1 - exp(-kappa (X_c - X_e))]

      which is continuous and monotone in X with a_w(X_e) = 0 and
      a_w(X_c) = 1 (exponential sorption isotherm anchored on the
      equilibrium moisture; penetration-theory shape, see Mujumdar 2014).
    """
    if X <= X_e:
        return 0.0
    if X >= X_c:
        return 1.0
    a_e = math.exp(-kappa * (X_c - X_e))
    val = (math.exp(-kappa * (X_c - X)) - a_e) / (1.0 - a_e)
    return min(max(val, 0.0), 1.0)


def driving_force(
    T_p: float,
    T_g: float,
    Y_g: float,
    a_w: float,
    tr: Dict[str, float],
    p: float = P_STD,
    form: str = "vapour_density",
) -> Tuple[float, float, float]:
    """Mass-transfer driving force (kg/m^3 equivalent) and its two terms.

    form = "vapour_density" (default, mission spec):
        rho_v,sat(T_p) a_w - rho_v,g          [kg/m^3]
    form = "humidity_ratio":
        rho_da (Y_sat(T_p) a_w - Y_g)         [kg/m^3], rho_da = rho_f/(1+Y_g)

    The two forms are equivalent for an isothermal gas; for T_p != T_g the
    vapour-density form carries the 1/T Jacobian of each term, which is why the
    humidity-ratio form is the one that reproduces the classical psychrometric
    wet bulb exactly in the Lewis = 1 limit (see the module docstring).
    """
    p_vs = saturation_pressure(T_p) * a_w
    rho_vs = vapour_density(p_vs, T_p)
    # free-stream vapour density is evaluated at the free-stream temperature
    rho_vg = vapour_density(vapour_pressure_from_humidity_ratio(Y_g, p), T_g)
    if form == "vapour_density":
        delta = rho_vs - rho_vg
    elif form == "humidity_ratio":
        if a_w <= 0.0:
            Y_s = 0.0
        elif p_vs >= p:
            Y_s = float("inf")
        else:
            Y_s = humidity_ratio_from_vapour_pressure(p_vs, p)
        rho_da = tr["rho"] / (1.0 + Y_g)
        delta = rho_da * (Y_s - Y_g)
        rho_vs = rho_da * Y_s
        rho_vg = rho_da * Y_g
    else:
        raise ValueError(f"unknown driving-force form: {form}")
    return delta, rho_vs, rho_vg


def evaporation_rate(
    T_p: float,
    X: float,
    T_g: float,
    Y_g: float,
    u_rel: float,
    d_p: float,
    area: float,
    X_e: float,
    X_c: float,
    kappa: float,
    p: float = P_STD,
    d_ab_override: Optional[object] = None,
    driving_force_form: str = "vapour_density",
) -> Tuple[float, Dict[str, float]]:
    """Evaporation mass flow from one particle (kg/s), clipped at condensation.

    mdot = h_m A_p [a_w rho_v,sat(T_p) - rho_v,g]   (default form)

    The driving force can also be supplied as rho_f (Y_sat a_w - Y_g); see
    :func:`driving_force`.
    """
    tr = transfer_coefficients(T_g, T_p, Y_g, u_rel, d_p, p, d_ab_override)
    a_w = sorption_activity(X, X_e, X_c, kappa)
    delta, rho_vs, rho_vg = driving_force(T_p, T_g, Y_g, a_w, tr, p, driving_force_form)
    mdot = tr["h_m"] * area * delta
    tr["a_w"] = a_w
    tr["rho_v_sat_eff"] = rho_vs
    tr["rho_v_g"] = rho_vg
    tr["driving_force"] = delta
    return max(mdot, 0.0), tr


# ---------------------------------------------------------------------------
# wet bulb with the implemented dual closure (Lewis-factor consistent)
# ---------------------------------------------------------------------------
def wet_bulb_residual_closures(
    T_s: float,
    T_g: float,
    Y_g: float,
    u_rel: float,
    d_p: float,
    p: float = P_STD,
    d_ab_override: Optional[object] = None,
    driving_force_form: str = "vapour_density",
) -> float:
    """Residual h (T_g - T_s) - h_m h_fg(T_s) * driving force (W/m^2)."""
    tr = transfer_coefficients(T_g, T_s, Y_g, u_rel, d_p, p, d_ab_override)
    delta, _, _ = driving_force(T_s, T_g, Y_g, 1.0, tr, p, driving_force_form)
    return tr["h"] * (T_g - T_s) - tr["h_m"] * latent_heat(T_s) * delta


def wet_bulb_from_closures(
    T_g: float,
    Y_g: float,
    u_rel: float,
    d_p: float,
    p: float = P_STD,
    d_ab_override: Optional[object] = None,
    driving_force_form: str = "vapour_density",
    tol: float = 1e-10,
) -> float:
    """Pseudo wet-bulb temperature: fixed point of the dual (Nu, Sh) closure.

    Solved by bisection on [max(248 K, T_g - 150 K), T_g]; the residual is positive far
    below T_g (large sensible heat supply, negligible rho_v,sat) and negative
    at T_g, so the bracket is guaranteed.  With ``d_ab_override = alpha`` the
    root equals the classical psychrometric wet bulb (Lewis factor = 1).
    """
    lo = max(248.0, T_g - 150.0)
    hi = T_g
    if wet_bulb_residual_closures(lo, T_g, Y_g, u_rel, d_p, p, d_ab_override, driving_force_form) <= 0.0:
        raise ValueError("wet_bulb_from_closures: residual at cold bracket <= 0")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if wet_bulb_residual_closures(mid, T_g, Y_g, u_rel, d_p, p, d_ab_override, driving_force_form) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# single-particle drying (lumped capacitance)
# ---------------------------------------------------------------------------
def specific_surface_area(d_p: float, rho_p: float) -> float:
    """Specific surface area of a monodisperse sphere packing (m^2 per kg dry solid)."""
    vol = math.pi * d_p ** 3 / 6.0
    m_s = rho_p * vol
    return math.pi * d_p ** 2 / m_s


@dataclass
class ParticleDryingState:
    """Lumped particle state (per particle)."""

    T_p: float
    X: float  # dry-basis moisture

    @property
    def wet_basis(self) -> float:
        return self.X / (1.0 + self.X)


def particle_derivatives(
    state: ParticleDryingState,
    T_g: float,
    Y_g: float,
    u_rel: float,
    d_p: float,
    rho_p: float,
    cp_s: float,
    X_e: float,
    X_c: float,
    kappa: float,
    p: float = P_STD,
    q_wall: float = 0.0,
    d_ab_override: Optional[object] = None,
    driving_force_form: str = "vapour_density",
) -> Tuple[float, float, Dict[str, float]]:
    """(dT_p/dt, dX/dt) for one particle, plus the closure diagnostics."""
    vol = math.pi * d_p ** 3 / 6.0
    m_s = rho_p * vol
    area = math.pi * d_p ** 2
    m_w = state.X * m_s
    mdot, tr = evaporation_rate(
        state.T_p, state.X, T_g, Y_g, u_rel, d_p, area, X_e, X_c, kappa, p,
        d_ab_override, driving_force_form,
    )
    q_conv = tr["h"] * area * (T_g - state.T_p)
    c_eff = m_s * cp_s + m_w * CP_L
    dT = (q_conv + q_wall - mdot * latent_heat(state.T_p)) / c_eff
    dX = -mdot / m_s
    if state.X <= X_e and dX < 0.0:
        dX = 0.0
    tr.update({"mdot": mdot, "q_conv": q_conv, "m_s": m_s, "m_w": m_w, "area": area})
    return dT, dX, tr


def simulate_single_particle(
    T_g: float,
    Y_g: float,
    u_rel: float,
    d_p: float,
    rho_p: float,
    cp_s: float,
    T_p0: float,
    X0: float,
    X_e: float,
    X_c: float,
    kappa: float,
    t_end: float,
    dt: float = 0.01,
    p: float = P_STD,
    d_ab_override: Optional[object] = None,
    driving_force_form: str = "vapour_density",
) -> Dict[str, np.ndarray]:
    """Explicit-Euler drying of one particle in a fixed gas state.

    Returns arrays ``t, T_p, X, mdot, h, h_m, Re, Sh, Nu, a_w``.
    """
    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    T_p = np.empty(n)
    X = np.empty(n)
    mdot = np.empty(n)
    h = np.empty(n)
    h_m = np.empty(n)
    Re = np.empty(n)
    Sh = np.empty(n)
    Nu = np.empty(n)
    a_w = np.empty(n)
    T_p[0] = T_p0
    X[0] = X0
    for i in range(n):
        dT, dX, tr = particle_derivatives(
            ParticleDryingState(T_p[i], X[i]), T_g, Y_g, u_rel, d_p, rho_p, cp_s,
            X_e, X_c, kappa, p, 0.0, d_ab_override, driving_force_form,
        )
        mdot[i] = tr["mdot"]
        h[i] = tr["h"]
        h_m[i] = tr["h_m"]
        Re[i] = tr["Re"]
        Sh[i] = tr["Sh"]
        Nu[i] = tr["Nu"]
        a_w[i] = tr["a_w"]
        if i + 1 < n:
            T_p[i + 1] = T_p[i] + dT * dt
            X[i + 1] = max(X[i] + dX * dt, X_e)
    return {
        "t": t,
        "T_p": T_p,
        "X": X,
        "mdot": mdot,
        "h": h,
        "h_m": h_m,
        "Re": Re,
        "Sh": Sh,
        "Nu": Nu,
        "a_w": a_w,
    }


# ---------------------------------------------------------------------------
# thin-layer kinetics: Lewis and Page, and their fit
# ---------------------------------------------------------------------------
def lewis_model(t, X_e: float, X_0: float, k: float):
    """Lewis (single exponential) thin-layer model."""
    t = np.asarray(t, dtype=float)
    return X_e + (X_0 - X_e) * np.exp(-k * t)


def page_model(t, X_e: float, X_0: float, k: float, n: float):
    """Page thin-layer model, X = X_e + (X_0 - X_e) exp[-(k t)^n]."""
    t = np.asarray(t, dtype=float)
    return X_e + (X_0 - X_e) * np.exp(-np.power(np.maximum(k * t, 0.0), n))


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def fit_thin_layer(t, X) -> Dict[str, Dict[str, float]]:
    """Least-squares fits of Lewis and Page models to a moisture curve X(t).

    Returns {'page': {X_e, X_0, k, n, r2, rmse, dof}, 'lewis': {...}}.
    The Page fit keeps all four parameters free (X_e, X_0, k, n) so that R^2 is
    a genuine goodness-of-fit measure of the *simulated* curve, not a
    self-fulfilling identity of the model used to generate it.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    out: Dict[str, Dict[str, float]] = {}

    p_page, _ = curve_fit(
        page_model, t, X, p0=[X.min(), X.max(), 1.0 / max(t[-1], 1e-9), 1.0],
        bounds=([0.0, X.max(), 1e-9, 0.2], [X.max() + 1e-6, 5.0 * X.max(), 10.0, 5.0]),
        maxfev=200000,
    )
    y_hat = page_model(t, *p_page)
    out["page"] = {
        "X_e": float(p_page[0]),
        "X_0": float(p_page[1]),
        "k": float(abs(p_page[2])),
        "n": float(p_page[3]),
        "r2": _r_squared(X, y_hat),
        "rmse": float(np.sqrt(np.mean((X - y_hat) ** 2))),
        "n_points": int(t.size),
    }

    p_lew, _ = curve_fit(
        lewis_model, t, X, p0=[X.min(), X.max(), 1.0 / max(t[-1], 1e-9)],
        bounds=([0.0, X.max(), 1e-9], [X.max() + 1e-6, 5.0 * X.max(), 10.0]),
        maxfev=200000,
    )
    y_hat_l = lewis_model(t, *p_lew)
    out["lewis"] = {
        "X_e": float(p_lew[0]),
        "X_0": float(p_lew[1]),
        "k": float(abs(p_lew[2])),
        "n": 1.0,
        "r2": _r_squared(X, y_hat_l),
        "rmse": float(np.sqrt(np.mean((X - y_hat_l) ** 2))),
        "n_points": int(t.size),
    }
    return out


#: Key used to record the vanishing-moisture degenerate control
DRY_AIR_CHAIN = "dry_limit"