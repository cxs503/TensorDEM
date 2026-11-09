"""Inter-phase drag closures for dense particle--fluid (packed / fluidised) beds.

Scope
-----
This module owns the *constitutive* side of an unresolved (point-particle)
DEM--CFD coupling: the force an interstitial fluid exerts on a particle as a
function of the slip velocity ``u = u_fluid - u_particle``, the particle
diameter ``d`` and the *local* voidage ``eps``.  Three closures are kept
explicitly separate so that a study can state which one it used:

``ergun``
    Ergun (1952) packed-bed relation written as a per-particle force::

        F = V_p u [ 150 mu (1-eps) / (eps^3 d^2) + 1.75 rho_f |u| / (eps^3 d) ]

    With uniform ``eps`` this reproduces the Ergun pressure gradient exactly
    through ``sum(F)/V_bed = dp/dz`` (asserted in the test-suite).

``wen_yu``
    Wen & Yu (1966) drag corrected for the neighbours, in the form used by
    Gidaspow (1994)::

        F = 0.5 rho_f C_D(eps Re) |u| u (pi d^2/4) eps^(-1.65),
        C_D = 24/Re (1 + 0.15 Re^0.687),  Re = eps rho_f |u| d / mu

    Stokes limit: ``F = 3 pi mu d u eps^(-2.65)`` (checked in the tests).

``gidaspow``
    Gidaspow (1994) blend: Ergun branch for ``eps < 0.8``, Wen--Yu otherwise.
    The branches are *not* exactly continuous at ``eps = 0.8``; the jump is
    measured and reported rather than smoothed away.

Hindered-settling yardsticks
----------------------------
:func:`rz_exponent` returns the Richardson & Zaki (1954) index ``n`` from the
commonly tabulated piecewise correlation; :func:`rz_slip_velocity` and
:func:`rz_superficial_velocity` give the corresponding laws.  These are
correlations, never code paths of the closure: they are what the simulated
settling index is compared against.

Sign conventions
----------------
Forces are the force **on the particle** in newtons.  ``u_slip = u_fluid -
u_particle``, so drag accelerates the particle along ``u_slip``.
Pressure-gradient (Froude--Krylov) forces are ``F = -V_p grad p``; in a
hydrostatic field this *is* the buoyancy force, so a caller must never add a
separate Archimedes term (see :func:`hydrostatic_pressure_gradient` and the
static-bed residual check).

All quantities SI (m, kg, s, Pa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GRAVITY",
    "GIDASPOW_SWITCH",
    "BRANCHES",
    "ErgunDrop",
    "sphere_volume",
    "sphere_projected_area",
    "ergun_pressure_gradient",
    "ergun_beta",
    "wen_yu_cd",
    "drag_coefficient",
    "drag_force",
    "gidaspow_branch_jump",
    "hydrostatic_pressure_gradient",
    "pressure_gradient_force",
    "buoyant_weight",
    "terminal_velocity_sn",
    "terminal_velocity_stokes",
    "hindered_slip_velocity",
    "rz_exponent",
    "rz_slip_velocity",
    "rz_superficial_velocity",
]

GRAVITY = 9.80665
GIDASPOW_SWITCH = 0.8
BRANCHES = ("gidaspow", "ergun", "wen_yu")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def sphere_volume(diameter: float) -> float:
    return math.pi * diameter**3 / 6.0


def sphere_projected_area(diameter: float) -> float:
    return math.pi * diameter**2 / 4.0


# ---------------------------------------------------------------------------
# Ergun reference relation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ErgunDrop:
    """Ergun pressure gradient split into viscous and inertial contributions."""

    total: float
    viscous: float
    inertial: float

    @property
    def inertial_fraction(self) -> float:
        return 0.0 if self.total == 0.0 else self.inertial / self.total

    @property
    def regime(self) -> str:
        frac = self.inertial_fraction
        if frac < 0.10:
            return "laminar"
        if frac < 0.90:
            return "transition"
        return "inertial"


def ergun_pressure_gradient(
    u_superficial: float,
    *,
    diameter: float,
    porosity: float,
    rho_fluid: float,
    mu: float,
) -> ErgunDrop:
    """Ergun (1952) pressure gradient ``-dp/dz`` in Pa/m (magnitude)."""
    eps = float(porosity)
    if not 0.0 < eps < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {eps}")
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    u = abs(float(u_superficial))
    visc = 150.0 * mu * (1.0 - eps) ** 2 / (eps**3 * diameter**2) * u
    inert = 1.75 * rho_fluid * (1.0 - eps) / (eps**3 * diameter) * u * u
    return ErgunDrop(total=visc + inert, viscous=visc, inertial=inert)


def ergun_beta(
    u_superficial: float,
    *,
    diameter: float,
    porosity: float,
    rho_fluid: float,
    mu: float,
) -> float:
    """Ergun inter-phase coefficient ``beta = (dp/dz)/u`` in kg/(m^3 s).

    Normalised so that ``beta * u_superficial`` is the pressure gradient; the
    equivalent per-particle force is ``F = V_p beta u / (1 - eps)``.
    """
    drop = ergun_pressure_gradient(
        u_superficial, diameter=diameter, porosity=porosity, rho_fluid=rho_fluid, mu=mu
    )
    if u_superficial == 0.0:
        raise ValueError("beta is singular at zero velocity")
    return drop.total / abs(u_superficial)


# ---------------------------------------------------------------------------
# drag closures
# ---------------------------------------------------------------------------
def wen_yu_cd(re: np.ndarray | float) -> np.ndarray:
    """Schiller--Naumann / Wen--Yu coefficient ``24/Re (1 + 0.15 Re^0.687)``."""
    re = np.asarray(re, dtype=float)
    if np.any(re <= 0.0):
        raise ValueError("Reynolds number must be positive")
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def drag_coefficient(
    speed: np.ndarray | float,
    *,
    diameter: float,
    porosity: np.ndarray | float,
    rho_fluid: float,
    mu: float,
    branch: str = "gidaspow",
) -> np.ndarray:
    """Linear drag coefficient ``c`` in ``F_vec = c * u_vec`` [kg/s].

    ``speed`` is ``|u_vec|``; broadcasting over ``speed``/``porosity`` lets one
    call handle a whole assembly with per-particle voidage.
    """
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}, expected one of {BRANCHES}")
    speed = np.asarray(speed, dtype=float)
    eps = np.asarray(porosity, dtype=float)
    if np.any((eps <= 0.0) | (eps > 1.0)):
        raise ValueError("porosity must lie in (0, 1]")

    vp = sphere_volume(diameter)
    area = sphere_projected_area(diameter)

    ergun = vp * (
        150.0 * mu * (1.0 - eps) / (eps**3 * diameter**2)
        + 1.75 * rho_fluid * speed / (eps**3 * diameter)
    )

    safe_re = np.where(speed > 0.0, eps * rho_fluid * np.maximum(speed, 1e-300) * diameter / mu, 1e-30)
    cd = wen_yu_cd(np.maximum(safe_re, 1e-30))
    wen_yu = 0.5 * rho_fluid * cd * speed * area * eps**-1.65
    stokes = 3.0 * math.pi * mu * diameter * eps**-2.65
    wen_yu = np.where(speed > 0.0, wen_yu, stokes)

    if branch == "ergun":
        return ergun
    if branch == "wen_yu":
        return wen_yu
    return np.where(eps < GIDASPOW_SWITCH, ergun, wen_yu)


def drag_force(
    u_slip: np.ndarray,
    *,
    diameter: float,
    porosity: np.ndarray | float,
    rho_fluid: float,
    mu: float,
    branch: str = "gidaspow",
) -> np.ndarray:
    """Drag force on particles for a slip velocity field [N].

    ``u_slip`` is ``(N, 3)`` (or ``(3,)`` for one particle); ``porosity``
    broadcasts to ``(N,)``.
    """
    u = np.asarray(u_slip, dtype=float)
    speed = np.linalg.norm(u, axis=-1)
    coeff = drag_coefficient(
        speed,
        diameter=diameter,
        porosity=porosity,
        rho_fluid=rho_fluid,
        mu=mu,
        branch=branch,
    )
    coeff = np.asarray(coeff, dtype=float)
    if u.ndim > 1:
        return coeff[:, np.newaxis] * u
    return coeff * u


def gidaspow_branch_jump(
    speed: float,
    *,
    diameter: float,
    rho_fluid: float,
    mu: float,
    eps: float = GIDASPOW_SWITCH,
) -> dict[str, float]:
    """Mismatch of the two Gidaspow branches at the switch voidage."""
    ergun = float(
        drag_coefficient(speed, diameter=diameter, porosity=eps, rho_fluid=rho_fluid, mu=mu, branch="ergun")
    )
    wen = float(
        drag_coefficient(speed, diameter=diameter, porosity=eps, rho_fluid=rho_fluid, mu=mu, branch="wen_yu")
    )
    return {
        "eps": eps,
        "speed": speed,
        "coefficient_ergun": ergun,
        "coefficient_wen_yu": wen,
        "relative_jump": (wen - ergun) / ergun,
    }


# ---------------------------------------------------------------------------
# pressure gradient / buoyancy
# ---------------------------------------------------------------------------
def hydrostatic_pressure_gradient(rho_fluid: float, gravity: float = GRAVITY) -> np.ndarray:
    """``grad p`` of a fluid at rest, ``(0, 0, -rho_f g)`` with +z up [Pa/m]."""
    return np.array([0.0, 0.0, -rho_fluid * gravity], dtype=float)


def pressure_gradient_force(volumes: np.ndarray, grad_p: np.ndarray) -> np.ndarray:
    """``F = -V_p grad p`` per particle [N]; this is the buoyancy force."""
    vol = np.asarray(volumes, dtype=float)
    gp = np.asarray(grad_p, dtype=float)
    return -vol[:, np.newaxis] * gp


def buoyant_weight(
    diameter: float, *, rho_particle: float, rho_fluid: float, gravity: float = GRAVITY
) -> float:
    """Effective submerged weight ``V_p (rho_p - rho_f) g`` [N]."""
    return sphere_volume(diameter) * (rho_particle - rho_fluid) * gravity


# ---------------------------------------------------------------------------
# terminal velocity and hindered settling
# ---------------------------------------------------------------------------
def _bisect(f, lo: float, hi: float, tol: float = 1e-14, max_iter: int = 300) -> float:
    f_lo = f(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo <= tol * max(abs(hi), 1e-30):
            break
    return 0.5 * (lo + hi)


def terminal_velocity_sn(
    diameter: float,
    *,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
    tol: float = 1e-13,
) -> dict[str, float]:
    """Unbounded terminal velocity from the Schiller--Naumann balance."""
    weight = buoyant_weight(diameter, rho_particle=rho_particle, rho_fluid=rho_fluid, gravity=gravity)
    if weight <= 0.0:
        raise ValueError("particle is not heavier than the fluid: it will not settle")

    def residual(u: float) -> float:
        re = rho_fluid * u * diameter / mu
        cd = float(wen_yu_cd(re))
        return 0.5 * rho_fluid * cd * u * u * sphere_projected_area(diameter) - weight

    u_stokes = weight / (3.0 * math.pi * mu * diameter)
    hi = max(10.0 * u_stokes, 1e-12)
    for _ in range(200):
        if residual(hi) > 0.0:
            break
        hi *= 2.0
    u_t = _bisect(residual, 1e-16, hi, tol=tol)
    return {
        "u_terminal": u_t,
        "re_terminal": rho_fluid * u_t * diameter / mu,
        "buoyant_weight": weight,
        "u_stokes": u_stokes,
    }


def terminal_velocity_stokes(
    diameter: float, *, rho_particle: float, rho_fluid: float, mu: float, gravity: float = GRAVITY
) -> float:
    """Stokes terminal velocity ``(rho_p - rho_f) g d^2 / (18 mu)``."""
    return (rho_particle - rho_fluid) * gravity * diameter**2 / (18.0 * mu)


def hindered_slip_velocity(
    porosity: np.ndarray | float,
    *,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
    branch: str = "gidaspow",
) -> np.ndarray:
    """Steady slip velocity in a bed of local voidage ``porosity``.

    Solves ``F_drag(u, eps) = V_p (rho_p - rho_f) g``.  This is the algebraic
    route to the hindered-settling exponent; the DEM route is
    :func:`tensordem.bedcoupling.bed.simulate_batch_settling`.
    """
    eps = np.atleast_1d(np.asarray(porosity, dtype=float))
    weight = buoyant_weight(diameter, rho_particle=rho_particle, rho_fluid=rho_fluid, gravity=gravity)
    out = np.empty_like(eps)
    for idx, e in np.ndenumerate(eps):
        def residual(u: float, e: float = float(e)) -> float:
            c = float(
                drag_coefficient(u, diameter=diameter, porosity=e, rho_fluid=rho_fluid, mu=mu, branch=branch)
            )
            return c * u - weight

        u_stokes = weight / (3.0 * math.pi * mu * diameter) * max(float(e), 1e-3) ** -2.65
        hi = max(10.0 * u_stokes, 1e-12)
        for _ in range(200):
            if residual(hi) > 0.0:
                break
            hi *= 2.0
        out[idx] = _bisect(residual, 1e-18, hi)
    return out


def rz_exponent(re_terminal: np.ndarray | float) -> np.ndarray:
    """Richardson & Zaki (1954) index ``n`` from the tabulated piecewise form.

    ``n = 4.65``                       for ``Re < 0.2``
    ``n = 4.4 Re^-0.03``               for ``0.2 <= Re < 1``
    ``n = 4.4 Re^-0.1``                for ``1 <= Re < 500``
    ``n = 2.4``                        for ``Re >= 500``

    Literature tabulations differ by ~0.1 at low ``Re``; the sensitivity of the
    acceptance margin to that spread is reported in the node study.
    """
    re = np.asarray(re_terminal, dtype=float)
    if np.any(re <= 0.0):
        raise ValueError("Re must be positive")
    out = np.where(
        re < 0.2,
        4.65,
        np.where(re < 1.0, 4.4 * re**-0.03, np.where(re < 500.0, 4.4 * re**-0.1, 2.4)),
    )
    return out


def rz_slip_velocity(porosity: np.ndarray | float, u_terminal: float, n: np.ndarray | float) -> np.ndarray:
    """Hindered slip velocity ``u_t eps^(n-1)`` (Richardson--Zaki)."""
    eps = np.asarray(porosity, dtype=float)
    return u_terminal * eps ** (np.asarray(n, dtype=float) - 1.0)


def rz_superficial_velocity(porosity: np.ndarray | float, u_terminal: float, n: np.ndarray | float) -> np.ndarray:
    """Richardson--Zaki superficial velocity ``u_t eps^n``."""
    eps = np.asarray(porosity, dtype=float)
    return u_terminal * eps ** np.asarray(n, dtype=float)
