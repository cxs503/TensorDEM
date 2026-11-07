"""Analytic drag anchors used by the LBM--DEM closure calibration.

Every number the calibration produces is compared against one of these closed
forms, so they are kept in one auditable place with their validity ranges.

* ``stokes_drag``            -- ``F = 3 pi mu d U`` (Stokes 1851), exact for
  ``Re -> 0``, unbounded fluid.
* ``schiller_naumann_cd``    -- ``C_D = 24/Re (1 + 0.15 Re^0.687)``, standard
  correlation for a rigid sphere, ``Re < 800`` (Schiller & Naumann 1935).
* ``array_correction``       -- dilute simple-cubic-array correction
  ``F/F_inf = 1 + 1.7601 phi^(1/3) + phi`` (Hasimoto 1959; Sangani & Acrivos
  1982).  Used to remove the *known* finite-domain effect of a periodic box, so
  that an unbounded-flow drag can be compared with the anchors above.
* ``tandem_pair_ratio``      -- two equal spheres translating along their line
  of centres: the Faxen/method-of-reflections result
  ``F_pair / (2 F_single) = 1 - (3/2) xi + O(xi^3)``, ``xi = a/r``.  This is the
  leading term of the Stimson--Jeffery (1926) exact series that family of
  two-sphere benchmarks tabulates; the ``O(xi^3)`` truncation is reported
  explicitly (0.2 % at ``xi = 1/8``).
"""

from __future__ import annotations

import math

__all__ = [
    "stokes_drag",
    "schiller_naumann_cd",
    "drag_from_cd",
    "array_correction",
    "tandem_pair_ratio",
    "terminal_velocity_stokes",
    "terminal_velocity_schiller_naumann",
    "reynolds",
]


def reynolds(diameter: float, velocity: float, nu: float) -> float:
    return abs(velocity) * diameter / nu


def stokes_drag(mu: float, diameter: float, velocity: float) -> float:
    """``F = 3 pi mu d U`` (unbounded, ``Re -> 0``)."""
    return 3.0 * math.pi * mu * diameter * velocity


def schiller_naumann_cd(re: float) -> float:
    """Schiller--Naumann drag coefficient (valid ``Re`` up to a few hundred)."""
    if re <= 0.0:
        raise ValueError("Re must be positive")
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def drag_from_cd(cd: float, rho: float, velocity: float, diameter: float) -> float:
    """``F = 1/2 rho U^2 (pi d^2 / 4) C_D``."""
    return 0.5 * rho * velocity * velocity * math.pi * diameter**2 / 4.0 * cd


def array_correction(solid_fraction: float) -> float:
    """Dilute periodic-array drag enhancement for a simple cubic array."""
    if solid_fraction <= 0.0:
        return 1.0
    phi13 = solid_fraction ** (1.0 / 3.0)
    return 1.0 + 1.7601 * phi13 + solid_fraction


def tandem_pair_ratio(xi: float) -> float:
    """Leading-order tandem pair drag ratio ``1 - 3 xi / 2`` (``xi = a/r``)."""
    if not 0.0 < xi < 1.0:
        raise ValueError("xi = a/r must lie in (0, 1)")
    return 1.0 - 1.5 * xi


def tandem_pair_ratio_truncation(xi: float) -> float:
    """Magnitude of the neglected ``O(xi^3)`` term of the reflection series."""
    return xi**3


def terminal_velocity_stokes(diameter: float, rho_particle: float, rho_fluid: float,
                             mu: float, gravity: float = 9.81e-3) -> float:
    """``U_t = (rho_p - rho_f) g d^2 / (18 mu)``."""
    return (rho_particle - rho_fluid) * gravity * diameter**2 / (18.0 * mu)


def terminal_velocity_schiller_naumann(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = 9.81e-3,
) -> float:
    """Solve ``F_drag(U) = (rho_p - rho_f) g V`` with the Schiller--Naumann law.

    Bisection on ``U``; the residual is monotone in ``U`` so the bracket always
    contracts.  Returns ``U`` in the same units as ``diameter`` per second.
    """
    volume = math.pi * diameter**3 / 6.0
    weight = (rho_particle - rho_fluid) * gravity * volume
    lo, hi = 1e-14, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        re = reynolds(diameter, mid, mu / rho_fluid)
        force = drag_from_cd(schiller_naumann_cd(re), rho_fluid, mid, diameter)
        if force < weight:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)