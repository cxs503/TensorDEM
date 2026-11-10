"""Pump - pipeline matching for dredge slurry transport.

Components
----------
* pipe characteristic ``H_pipe(Q) = H_st + i_m(Q) L + zeta_minor V^2/(2g)``
  with ``i_m`` from :mod:`tensordem.slurry_transport.pressure_drop`;
* dredge-pump head curve, affinity-scaled with the shaft speed
  ``H(Q; n) = (n/n0)^2 H0 - a Q - b Q^2``;
* operating point = intersection ``H_pipe(Q*) = H_pump(Q*)`` solved with Brent's
  method on a bracketed interval, residual reported (gate: < 1e-6 m);
* shaft power ``P = rho_m g Q H / eta_p`` and the power identity
  ``P_hyd = rho_m g Q H = eta_p P_shaft`` (gate: 0.1%);
* NPSH margin ``NPSH_a - NPSH_r >= 0.5 m``;
* soil production ``Q_soil = Q Cv / (1 - n_void)`` (bank m3/h) and solids
  tonnage.

All functions are pure; the operating point is a root of a monotone function,
so no external solver state is involved.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .closures import G, MU_W, RHO_W
from .pressure_drop import mixture_gradient

__all__ = [
    "PipeSpec",
    "PumpSpec",
    "SlurrySpec",
    "system_head",
    "pump_head",
    "pump_efficiency",
    "operating_point",
    "npsh_available",
    "npsh_required",
    "npsh_margin",
    "shaft_power",
    "hydraulic_power",
    "production_rate",
]


class PipeSpec:
    """Horizontal discharge line (optionally with a static lift)."""

    def __init__(self, D, L, eps_D=1e-3, zeta_minor=5.0, H_static=0.0,
                 L_suction=10.0, h_suction=2.0):
        self.D = float(D)
        self.L = float(L)
        self.eps_D = float(eps_D)
        self.zeta_minor = float(zeta_minor)
        self.H_static = float(H_static)
        self.L_suction = float(L_suction)
        self.h_suction = float(h_suction)

    @property
    def area(self):
        return math.pi / 4.0 * self.D ** 2

    def velocity(self, Q):
        return float(Q) / self.area

    def __repr__(self):
        return (f"PipeSpec(D={self.D}, L={self.L}, eps_D={self.eps_D}, "
                f"zeta_minor={self.zeta_minor}, H_static={self.H_static})")


class PumpSpec:
    """Centrifugal dredge pump, quadratic head curve, efficiency parabola."""

    def __init__(self, H0, a, b, eta_p=0.72, n0=1.0, Q_opt=None, eta_max=None,
                 npsh_r0=4.0, npsh_c=0.0):
        self.H0 = float(H0)
        self.a = float(a)
        self.b = float(b)
        self.eta_p = float(eta_p)
        self.n0 = float(n0)
        self.Q_opt = float(Q_opt) if Q_opt is not None else None
        self.eta_max = float(eta_max) if eta_max is not None else self.eta_p
        self.npsh_r0 = float(npsh_r0)
        self.npsh_c = float(npsh_c)

    def head(self, Q, n=None):
        return pump_head(Q, self, n=n)

    def __repr__(self):
        return (f"PumpSpec(H0={self.H0}, a={self.a}, b={self.b}, "
                f"eta_p={self.eta_p}, npsh_r0={self.npsh_r0})")


class SlurrySpec:
    """Solids specification of the transported mixture."""

    def __init__(self, Ss, cv, d50, rho_f=RHO_W, mu=MU_W, w_s=None,
                 n_void=0.35, kappa=1.0):
        self.Ss = float(Ss)
        self.cv = float(cv)
        self.d50 = float(d50)
        self.rho_f = float(rho_f)
        self.mu = float(mu)
        self.n_void = float(n_void)
        self.kappa = float(kappa)
        if w_s is None:
            from .closures import terminal_velocity_numeric
            w_s = terminal_velocity_numeric(d50, self.Ss * self.rho_f,
                                            self.rho_f, self.mu)["w"]
        self.w_s = float(w_s)

    @property
    def rho_s(self):
        return self.Ss * self.rho_f

    @property
    def rho_m(self):
        return self.rho_f * (1.0 + self.cv * (self.Ss - 1.0))

    def __repr__(self):
        return (f"SlurrySpec(Ss={self.Ss}, cv={self.cv}, d50={self.d50}, "
                f"w_s={self.w_s:.5f})")


def pump_head(Q, pump, n=None):
    """Pump head curve [m], affinity-scaled with speed ratio ``n/n0``."""
    speed = (pump.n0 if n is None else float(n)) / pump.n0
    return speed ** 2 * pump.H0 - pump.a * float(Q) - pump.b * float(Q) ** 2


def pump_efficiency(Q, pump):
    """Parabolic efficiency curve; flat at ``eta_max`` if ``Q_opt`` is unset."""
    if pump.Q_opt is None or pump.Q_opt <= 0.0:
        return pump.eta_max
    x = (float(Q) - pump.Q_opt) / (0.6 * pump.Q_opt)
    eta = pump.eta_max * (1.0 - x * x)
    return float(max(eta, 0.05))


def system_head(Q, pipe, slurry, kappa=None, g=G):
    """Pipeline head requirement [m] at flow rate ``Q`` [m3/s]."""
    kappa = slurry.kappa if kappa is None else float(kappa)
    V = pipe.velocity(Q)
    if V <= 0.0:
        return float("inf")
    mg = mixture_gradient(V, pipe.D, slurry.cv, slurry.Ss, slurry.w_s,
                          rho=slurry.rho_f, mu=slurry.mu, eps_D=pipe.eps_D,
                          kappa=kappa, g=g)
    H = pipe.H_static + mg["i_m"] * pipe.L \
        + pipe.zeta_minor * V * V / (2.0 * g)
    return H


def operating_point(pipe, pump, slurry, n=None, Q_bracket=None,
                    kappa=None, g=G, branch="stable", n_scan=600,
                    Q_max=None):
    """Intersection of pipe demand and pump head (the stable branch).

    The demand curve ``H_pipe(Q)`` is U-shaped (the solids term
    ``Cv (Ss-1) w_s / V`` diverges as ``Q -> 0``) and the pump curve is
    decreasing, so the pair has **two** intersections: a low-flow one on the
    rising branch and the physically selected one on the falling branch.  The
    solver scans the flow range up to the pump shut-off flow, collects the sign
    changes of ``H_pipe - H_pump`` and keeps the highest-flow crossing
    (``branch="stable"``; ``"first"`` returns the low-flow root for the
    stability analysis in the report).

    Returns a dict with ``Q``, ``H``, ``residual`` (``|H_pipe - H_pump|`` [m]),
    velocity, gradient breakdown, shaft power, NPSH margin and production.
    """
    H0 = pump.H0 if n is None else pump.H0 * (float(n) / pump.n0) ** 2
    # pump shut-off flow: H_pump(Q) = 0
    disc = pump.a ** 2 + 4.0 * pump.b * max(H0, 1e-12)
    Q_off = (-pump.a + math.sqrt(disc)) / (2.0 * pump.b) if pump.b > 0 \
        else max(1e-6, 10.0)
    if Q_max is not None:
        Q_off = min(Q_off, float(Q_max))

    def res(Q):
        return system_head(Q, pipe, slurry, kappa=kappa, g=g) \
            - pump_head(Q, pump, n=n)

    if Q_bracket is not None:
        Q_lo, Q_hi = float(Q_bracket[0]), float(Q_bracket[1])
        f_lo, f_hi = res(Q_lo), res(Q_hi)
        if f_lo * f_hi > 0.0:
            raise ValueError(f"bracket does not straddle a root "
                             f"(f_lo={f_lo:.4g}, f_hi={f_hi:.4g})")
    else:
        Qs = np.geomspace(1e-5, Q_off * 0.999, n_scan)
        fs = np.array([res(q) for q in Qs])
        good = np.isfinite(fs)
        crossings = []
        for k in range(len(Qs) - 1):
            if not (good[k] and good[k + 1]):
                continue
            if fs[k] == 0.0:
                crossings.append((Qs[k], Qs[k]))
            elif fs[k] * fs[k + 1] < 0.0:
                crossings.append((Qs[k], Qs[k + 1]))
        if not crossings:
            raise ValueError("no operating point: pump head never meets the "
                             f"pipeline demand below shut-off Q={Q_off:.4g} m3/s")
        Q_lo, Q_hi = crossings[-1] if branch == "stable" else crossings[0]
    Q = brentq(res, Q_lo, Q_hi, xtol=1e-15, rtol=1e-15, maxiter=300)

    H_pipe = system_head(Q, pipe, slurry, kappa=kappa, g=g)
    H_pump = pump_head(Q, pump, n=n)
    V = pipe.velocity(Q)
    mg = mixture_gradient(V, pipe.D, slurry.cv, slurry.Ss, slurry.w_s,
                          rho=slurry.rho_f, mu=slurry.mu, eps_D=pipe.eps_D,
                          kappa=slurry.kappa if kappa is None else kappa, g=g)
    eta = pump_efficiency(Q, pump)
    P_shaft = shaft_power(Q, H_pipe, slurry.rho_m, eta, g=g)
    from .critical_velocity import durand_deposition_velocity
    vc = float(durand_deposition_velocity(pipe.D, slurry.Ss, slurry.cv,
                                          slurry.d50, g=g))
    return {
        "Q": float(Q), "H": float(H_pipe), "H_pipe": float(H_pipe),
        "H_pump": float(H_pump), "residual": abs(H_pipe - H_pump),
        "V": float(V), "i_m": mg["i_m"], "i_w": mg["i_w"], "i_e": mg["i_e"],
        "f": mg["f"], "Re": mg["Re_m"], "eta_p": eta,
        "P_shaft": P_shaft,
        "P_hydraulic": hydraulic_power(Q, H_pipe, slurry.rho_m, g=g),
        "npsh_a": npsh_available(pipe, pump, slurry, Q, g=g),
        "npsh_r": npsh_required(Q, pump),
        "npsh_margin": npsh_margin(pipe, pump, slurry, Q, g=g),
        "production": production_rate(Q, slurry),
        "V_c": vc, "V_over_Vc": float(V / vc),
        "Q_shutoff": float(Q_off), "branch": branch,
        "n_crossings": int(len(crossings) if Q_bracket is None else 1),
    }


def shaft_power(Q, H, rho_m, eta_p, g=G):
    """Shaft power [W] = rho_m g Q H / eta_p."""
    return float(rho_m) * float(g) * float(Q) * float(H) / float(eta_p)


def hydraulic_power(Q, H, rho_m, g=G):
    """Hydraulic (water) power [W] = rho_m g Q H."""
    return float(rho_m) * float(g) * float(Q) * float(H)


def npsh_available(pipe, pump, slurry, Q, p_atm=101325.0, p_vap=2339.0, g=G):
    """NPSH_a [m] = (p_atm - p_vap)/(rho_m g) + h_s - h_f,suction."""
    from .pressure_drop import water_gradient
    V = pipe.velocity(Q)
    hf = water_gradient(V, pipe.D, rho=slurry.rho_f, mu=slurry.mu,
                        eps_D=pipe.eps_D, g=g)["i_w"] * pipe.L_suction
    return (float(p_atm) - float(p_vap)) / (slurry.rho_m * g) \
        + pipe.h_suction - hf


def npsh_required(Q, pump):
    """Pump NPSH_r [m] = npsh_r0 + npsh_c Q^2."""
    return pump.npsh_r0 + pump.npsh_c * float(Q) ** 2


def npsh_margin(pipe, pump, slurry, Q, g=G):
    return npsh_available(pipe, pump, slurry, Q, g=g) - npsh_required(Q, pump)


def production_rate(Q, slurry):
    """Production [m3/h]: solids volume, in-situ (bank) soil and tonnage."""
    Q_solid = float(Q) * slurry.cv
    Q_bank = Q_solid / (1.0 - slurry.n_void)
    return {
        "Q_mixture_m3h": 3600.0 * float(Q),
        "Q_solid_m3h": 3600.0 * Q_solid,
        "Q_bank_m3h": 3600.0 * Q_bank,
        "solids_tph": 3600.0 * Q_solid * slurry.rho_s / 1000.0,
    }