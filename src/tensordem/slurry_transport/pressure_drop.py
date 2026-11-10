"""Horizontal-pipe pressure gradient of a settling slurry.

Chain of closures
-----------------
1. carrier friction factor ``f``: laminar ``64/Re`` below Re = 2000 and the
   implicit Colebrook-White equation above Re = 4000, with two independent
   explicit approximations (Swamee-Jain, Haaland) used as cross-checks;
2. water gradient ``i_w = f V^2 / (2 g D)`` [m water / m pipe];
3. solids excess gradient.  Two closures are provided:

   ``excess_gradient_suspension`` (derived, no fitted constant)
       Energy balance for a fully suspended load: supporting the submerged
       weight of the solids costs the power ``Cv (rho_s - rho_f) g w_s`` per
       unit mixture volume, delivered by the drag work ``rho_f g i_e V``::

           i_e = Cv (Ss - 1) (w_s / V) * kappa

       ``kappa = 1`` is the ideal-suspension limit; bed-load (saltating) cases
       are known to exceed it, which is why the DEM section measures kappa.

   ``excess_gradient_durand_form``
       Durand's power-law reduction ``Phi = i_e / (i_w Cv) = K Fr^(-m)`` with
       ``Fr = V / sqrt(g D (Ss-1))``.  ``K`` and ``m`` are *fitted in-tree* to
       the derived closure over the design envelope (see
       :func:`fit_durand_form`), so no literature constant is invented; the fit
       quality (R^2) is reported and gates the comparison.

4. mixture gradient ``i_m = i_w(mu_m) + i_e`` with the Einstein viscosity
   ``mu_m = mu_f (1 + 2.5 Cv)`` for the carrier in the mixture.
"""

from __future__ import annotations

import math

import numpy as np

from .closures import G, MU_W, RHO_W

__all__ = [
    "friction_factor_laminar",
    "colebrook_friction",
    "swamee_jain_friction",
    "haaland_friction",
    "friction_factor",
    "moody_reference_points",
    "darcy_head_loss",
    "water_gradient",
    "einstein_viscosity",
    "excess_gradient_suspension",
    "mixture_gradient",
    "fit_durand_form",
    "excess_gradient_durand_form",
    "excess_gradient_durand",
    "DURAND_EXCESS",
    "pressure_drop",
]

RE_LAMINAR_LIMIT = 2000.0
RE_TURBULENT_LIMIT = 4000.0

#: Durand excess-gradient correlation ``Phi = i_e/(i_w Cv) = K Fr^-m`` with
#: ``Fr = V/sqrt(g D (Ss-1))``.  Shipped as data with an explicit provenance and
#: an overridable constant; the literature quotes the constant over a band, which
#: is reported as a sensitivity rather than collapsed into one number.
DURAND_EXCESS = {
    "K": 85.0,
    "m": 1.5,
    "K_band": [85.0, 150.0],
    "provenance": (
        "classic Durand (1953) / Durand-Condolios (1952) excess-head-loss "
        "constant, quoted in the dredging literature over the band 85-150; "
        "shipped as data (overridable), the band is reported as a sensitivity"
    ),
}

#: three representative Moody-chart points (Re, relative roughness)
MOODY_POINTS = [
    {"Re": 1.0e4, "eps_D": 1.0e-4, "label": "smooth pipe, moderate Re"},
    {"Re": 1.0e5, "eps_D": 1.0e-3, "label": "steel pipe, fully rough onset"},
    {"Re": 1.0e6, "eps_D": 5.0e-4, "label": "large pipe, fully rough"},
]


def friction_factor_laminar(Re):
    """Darcy friction factor in the laminar limit, ``64/Re`` (exact)."""
    return 64.0 / float(Re)


def colebrook_friction(Re, eps_D, tol=1e-14, max_iter=200):
    """Solve Colebrook-White implicitly.

    ``1/sqrt(f) = -2 log10(eps/(3.7 D) + 2.51/(Re sqrt(f)))``

    Solved as a fixed point in ``x = 1/sqrt(f)``:
    ``x_{k+1} = -2 log10(eps/3.7/D + 2.51 x_k / Re)`` with Aitken-free
    geometric averaging (``x = 0.5 (x + x_new)``), which converges in < 20
    iterations for all Re > 2000; the residual of the Colebrook equation is
    returned in the dict so the gate is checkable.
    """
    Re = float(Re)
    eps_D = float(eps_D)
    if Re <= 0.0:
        raise ValueError("Re must be positive")
    x = 1.0 / math.sqrt(0.02)  # f ~ 0.02 seed
    log10 = math.log10
    for _ in range(max_iter):
        x_new = -2.0 * log10(eps_D / 3.7 + 2.51 * x / Re)
        x = 0.5 * (x + x_new)
        if abs(x_new - x) < tol:
            break
    f = 1.0 / x ** 2
    residual = x + 2.0 * log10(eps_D / 3.7 + 2.51 * x / Re)
    return {"f": f, "x": x, "residual": abs(residual), "iterations": _ + 1,
            "Re": Re, "eps_D": eps_D}


def swamee_jain_friction(Re, eps_D):
    """Swamee-Jain (1976) explicit approximation of Colebrook (+/-1%)."""
    Re = float(Re)
    eps_D = float(eps_D)
    return 0.25 / math.log10(eps_D / 3.7 + 5.74 / Re ** 0.9) ** 2


def haaland_friction(Re, eps_D):
    """Haaland (1983) explicit approximation of Colebrook (+/-1.5%)."""
    Re = float(Re)
    eps_D = float(eps_D)
    return (-1.8 * math.log10((eps_D / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def friction_factor(Re, eps_D):
    """Piecewise Darcy factor: 64/Re (Re<2000), Colebrook (Re>4000), blend.

    The 2000 < Re < 4000 window is linearly blended in ``1/Re`` (documented
    transition model, no physical claim).
    """
    Re = float(Re)
    if Re < RE_LAMINAR_LIMIT:
        return friction_factor_laminar(Re)
    if Re > RE_TURBULENT_LIMIT:
        return colebrook_friction(Re, eps_D)["f"]
    w = (Re - RE_LAMINAR_LIMIT) / (RE_TURBULENT_LIMIT - RE_LAMINAR_LIMIT)
    return (1.0 - w) * friction_factor_laminar(RE_LAMINAR_LIMIT) \
        + w * colebrook_friction(RE_TURBULENT_LIMIT, eps_D)["f"]


def moody_reference_points(points=None):
    """Compare the Colebrook solve with two independent explicit approximations.

    The Moody chart *is* a plot of Colebrook-White, so an independent
    "chart-reading" comparison is replaced by two published explicit
    approximations (Swamee-Jain, Haaland), each documented to +/-1% / +/-1.5%
    of Colebrook; agreement of the implicit solve with both at <1% therefore
    demonstrates the implicit solve and not a shared bug.
    """
    pts = points if points is not None else MOODY_POINTS
    rows = []
    for p in pts:
        col = colebrook_friction(p["Re"], p["eps_D"])
        sj = swamee_jain_friction(p["Re"], p["eps_D"])
        ha = haaland_friction(p["Re"], p["eps_D"])
        rows.append({
            "Re": p["Re"], "eps_D": p["eps_D"], "label": p.get("label", ""),
            "f_colebrook": col["f"], "residual": col["residual"],
            "f_swamee_jain": sj, "f_haaland": ha,
            "err_swamee_jain": abs(sj - col["f"]) / col["f"],
            "err_haaland": abs(ha - col["f"]) / col["f"],
        })
    return rows


def darcy_head_loss(f, L, D, V, g=G):
    """Darcy-Weisbach head loss [m]."""
    return float(f) * float(L) / float(D) * float(V) ** 2 / (2.0 * float(g))


def water_gradient(V, D, rho=RHO_W, mu=MU_W, eps_D=1e-3, g=G):
    """Carrier hydraulic gradient ``i_w = f V^2 / (2 g D)`` [-], plus Re and f."""
    Re = float(rho) * float(V) * float(D) / float(mu)
    f = friction_factor(Re, eps_D)
    iw = f * float(V) ** 2 / (2.0 * float(g) * float(D))
    return {"i_w": iw, "Re": Re, "f": f}


def einstein_viscosity(cv, mu=MU_W):
    """Einstein effective viscosity of the mixture (dilute limit)."""
    return float(mu) * (1.0 + 2.5 * float(cv))


def excess_gradient_suspension(V, cv, Ss, w_s, kappa=1.0):
    """Derived suspension-energy excess gradient ``Cv (Ss-1) (w_s/V) kappa``."""
    return float(cv) * (float(Ss) - 1.0) * (float(w_s) / float(V)) * float(kappa)


def mixture_gradient(V, D, cv, Ss, w_s, rho=RHO_W, mu=MU_W, eps_D=1e-3,
                     kappa=1.0, g=G):
    """Total mixture hydraulic gradient with the carrier in the mixture.

    Returns dict with ``i_m``, ``i_w``, ``i_e``, ``Re_m``, ``f``.
    """
    mu_m = einstein_viscosity(cv, mu)
    w = water_gradient(V, D, rho=rho, mu=mu_m, eps_D=eps_D, g=g)
    ie = excess_gradient_suspension(V, cv, Ss, w_s, kappa=kappa)
    return {"i_m": w["i_w"] + ie, "i_w": w["i_w"], "i_e": ie,
            "Re_m": w["Re"], "f": w["f"], "mu_m": mu_m}


def fit_durand_form(V_list, D, cv_list, Ss, d50, rho=RHO_W, mu=MU_W,
                    eps_D=1e-3, kappa=1.0, g=G):
    """Fit ``Phi = i_e/(i_w Cv) = K Fr^(-m)`` to the derived closure at fixed D.

    The Durand family represents the excess gradient as a power of the densimetric
    Froude number ``Fr = V/sqrt(g D (Ss-1))``.  Since the derived suspension
    closure gives ``i_e/(i_w Cv) = (Ss-1) w_s /(V i_w)`` and ``i_w ~ V^2``, the
    reduction is ``Phi ~ V^-3`` at fixed diameter, i.e. ``m = 3`` (the fitted
    value is reported, not imposed).  ``K`` is *diameter dependent* in this
    reduction, which is stated as a limitation: the classic Durand chart absorbs
    that dependence into the F_L chart, not into a single constant.

    Returns ``K``, ``m``, ``R^2``, ``n_points`` and the max relative residual of
    the log-log fit.
    """
    from .closures import terminal_velocity_numeric

    ws = terminal_velocity_numeric(d50, Ss * rho, rho, mu)["w"]
    Fr, Phi = [], []
    for cv in cv_list:
        for V in V_list:
            mg = mixture_gradient(V, D, cv, Ss, ws, rho=rho, mu=mu,
                                  eps_D=eps_D, kappa=kappa, g=g)
            Fr.append(V / math.sqrt(g * D * (Ss - 1.0)))
            Phi.append(mg["i_e"] / (mg["i_w"] * cv))
    Fr = np.asarray(Fr)
    Phi = np.asarray(Phi)
    A = np.vstack([np.ones_like(Fr), -np.log(Fr)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(Phi), rcond=None)
    log_fit = A @ coef
    ss_res = float(np.sum((np.log(Phi) - log_fit) ** 2))
    ss_tot = float(np.sum((np.log(Phi) - np.mean(np.log(Phi))) ** 2))
    return {"K": float(math.exp(coef[0])), "m": float(coef[1]),
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "n_points": int(Fr.size), "D": float(D), "d50": float(d50),
            "w_s": float(ws),
            "max_rel_residual": float(np.max(np.abs(np.exp(log_fit - np.log(Phi)) - 1.0)))}


def excess_gradient_durand_form(V, D, cv, Ss, K, m, i_w, g=G):
    """Durand-form excess gradient ``i_e = K Fr^(-m) Cv i_w``."""
    Fr = float(V) / math.sqrt(float(g) * float(D) * (float(Ss) - 1.0))
    return float(K) * Fr ** (-float(m)) * float(cv) * float(i_w)


def excess_gradient_durand(V, D, cv, Ss, i_w=None, K=None, m=None,
                           rho=RHO_W, mu=MU_W, eps_D=1e-3, g=G):
    """Classic Durand excess gradient (shipped constant, see DURAND_EXCESS)."""
    K = DURAND_EXCESS["K"] if K is None else float(K)
    m = DURAND_EXCESS["m"] if m is None else float(m)
    if i_w is None:
        i_w = water_gradient(V, D, rho=rho, mu=mu, eps_D=eps_D, g=g)["i_w"]
    return excess_gradient_durand_form(V, D, cv, Ss, K, m, i_w, g=g)


def pressure_drop(gradient, L, rho=RHO_W, g=G):
    """Convert a hydraulic gradient [-] into a pressure drop [Pa]."""
    return float(gradient) * float(rho) * float(g) * float(L)