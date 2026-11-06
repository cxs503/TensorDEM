"""A floating ice sheet as an infinite thin plate on a Winkler foundation.

Why this module exists
----------------------
``tensordem.ridge_geotech`` carries the **one-dimensional** half of the
elastic-foundation family -- the Hetenyi (1946) infinite beam and its finite
element counterpart -- verified against the closed form in
``docs/ridge_winkler_subgrade_modulus_verification.md``.  The **two-dimensional**
member is missing here, and it is the one ice engineering uses: a continuous ice
sheet floating on water is a thin elastic plate on a Winkler foundation of
modulus ``k = rho_w g``, and the loads that matter for a ship in ice (a resting or
grounded hull patch, a skeg/keel line load, an ice-road vehicle, a hovercraft
skirt) are local patches whose peak bending stress is the classical Westergaard
(1926) / Kerr (1979) quantity that ISO 19906 local-ice-pressure practice is built
around.  Without this module the package could not answer "what can this ice sheet
carry", and had no analytic plate target for its ice-sheet DEM specimens.

Sign and unit conventions
-------------------------
SI throughout.  ``w`` is the **downward** deflection (positive), the load ``P`` is
downward positive, moments are **sagging positive**, so the bottom-surface tensile
stress under a downward patch is ``6 M / h**2``.  Unlike the beam functions in
:mod:`tensordem.ridge_geotech`, ``k`` here is a foundation modulus **per unit
area** (N/m^3) -- the only convention change.

Two independent numerical routes
--------------------------------
* **Point load**: the exact Kelvin-function solution
  ``w = -(P l^2 / (2 pi D)) kei(r/l)``.  ``scipy.special.kelvin`` supplies
  ``ker``, ``kei`` and their first derivatives, and the Kelvin ODE
  ``x^2 g'' + x g' - i x^2 g = 0`` (Abramowitz & Stegun 9.9) gives
  ``kei'' = ker - kei'/x``, so no finite differencing is used anywhere.
* **Patch load**: the Hankel-transform representation of the same operator,

      w(r) = P / (pi a k l) * int_0^inf J1(a s / l) J0(r s / l) / (1 + s**4) ds,

  which shares no code with the Kelvin path, so patch and point results can be
  cross-checked against each other without circularity.

The two routes meet on the exact identity (used by
:func:`patch_centre_moment`)

    int_0^inf s**2 J1(eps s) / (1 + s**4) ds = kei'(eps),   eps = a / l,

and that closed form reproduces the published Westergaard constant
``0.6159`` asymptotically (see ``docs/ice_sheet_plate_on_foundation.md``).

Point versus patch moments
--------------------------
``M_r`` and ``M_t`` of a **point** load diverge logarithmically at the origin, so
:func:`point_load_moments` requires ``r > 0``.  A finite patch regularises the
divergence; :func:`patch_centre_moment` gives that value in closed form, which is
the quantity ice bearing-capacity rules are written in.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import j0, j1, kelvin

__all__ = [
    "SEA_WATER_DENSITY",
    "GRAVITY",
    "WESTERGAARD_CONSTANT",
    "plate_rigidity",
    "foundation_modulus",
    "characteristic_length",
    "kelvin_kei",
    "point_load_deflection",
    "point_load_moments",
    "patch_load_deflection",
    "patch_load_moments",
    "patch_centre_moment",
    "westergaard_max_stress",
    "ice_sheet_bearing_capacity",
]

#: Reference seawater density (kg/m^3) and gravity (m/s^2) for a floating sheet.
SEA_WATER_DENSITY = 1025.0
GRAVITY = 9.81

#: Published constant of the Westergaard (1926)/Kerr (1979) asymptote
#: ``sigma_max = 3 (1+nu) P / (2 pi h^2) * (ln(l/a) + WESTERGAARD_CONSTANT)``.
WESTERGAARD_CONSTANT = 0.6159


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def plate_rigidity(young_modulus: float, thickness: float, poisson_ratio: float) -> float:
    """Flexural rigidity ``D = E h^3 / (12 (1 - nu^2))`` of a thin plate (N m)."""
    young = _finite("young_modulus", young_modulus, positive=True)
    height = _finite("thickness", thickness, positive=True)
    nu = _finite("poisson_ratio", poisson_ratio)
    if not -1.0 < nu < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    return young * height**3 / (12.0 * (1.0 - nu**2))


def foundation_modulus(
    water_density: float = SEA_WATER_DENSITY,
    gravity: float = GRAVITY,
) -> float:
    """Winkler modulus of a floating ice sheet, ``k = rho_w g`` (N/m^3).

    Displacing water by ``w`` produces a restoring pressure ``rho_w g w``: a linear
    buoyancy foundation, no free-surface nonlinearity and no added mass (static
    model).
    """
    rho = _finite("water_density", water_density, positive=True)
    g = _finite("gravity", gravity, positive=True)
    return rho * g


def characteristic_length(rigidity: float, modulus: float) -> float:
    """Hetenyi characteristic length ``l = (D / k)**(1/4)`` (m)."""
    d = _finite("rigidity", rigidity, positive=True)
    k = _finite("modulus", modulus, positive=True)
    return (d / k) ** 0.25


def kelvin_kei(x):
    """``kei`` and its first two derivatives (numpy broadcasting, ``x >= 0``).

    The first derivative comes from ``scipy.special.kelvin``; the second uses the
    Kelvin ODE identity ``kei'' = ker - kei'/x`` (at ``x = 0`` it is ``-inf``,
    which is the logarithmic singularity of the point-load moment).
    """
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0.0):
        raise ValueError("kei is evaluated for x >= 0 only")
    values = kelvin(x_arr)
    ker, kei = values[1].real, values[1].imag
    kei_prime = values[3].imag
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(x_arr > 0.0, x_arr, 1.0)
        kei_second = np.where(x_arr > 0.0, ker - kei_prime / safe, -np.inf)
    return kei, kei_prime, kei_second


def point_load_deflection(
    r,
    *,
    load: float,
    rigidity: float,
    modulus: float,
    poisson_ratio: float = 0.3,
) -> np.ndarray:
    """Exact downward deflection of an infinite plate on a Winkler foundation.

    ``w(r) = -(P l^2 / (2 pi D)) kei(r/l)``; the centre value reduces to the
    classical ``w(0) = P / (8 sqrt(k D)) = P l^2 / (8 D)``.
    """
    p = _finite("load", load)
    d = _finite("rigidity", rigidity, positive=True)
    k = _finite("modulus", modulus, positive=True)
    _finite("poisson_ratio", poisson_ratio)
    ell = characteristic_length(d, k)
    kei, _, _ = kelvin_kei(np.atleast_1d(np.asarray(r, dtype=float)) / ell)
    return -(p * ell**2 / (2.0 * math.pi * d)) * np.atleast_1d(kei)


def point_load_moments(
    r,
    *,
    load: float,
    rigidity: float,
    modulus: float,
    poisson_ratio: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial and circumferential moments ``(M_r, M_t)`` of a point load (N m).

    Both are logarithmically singular at the origin, so ``r`` must be strictly
    positive; use :func:`patch_load_moments` or :func:`patch_centre_moment` for a
    finite loaded area.  Sagging positive.
    """
    p = _finite("load", load)
    d = _finite("rigidity", rigidity, positive=True)
    k = _finite("modulus", modulus, positive=True)
    nu = _finite("poisson_ratio", poisson_ratio)
    rho = np.atleast_1d(np.asarray(r, dtype=float)) / characteristic_length(d, k)
    if np.any(rho <= 0.0):
        raise ValueError("point_load_moments requires r > 0 (logarithmic singularity at r = 0)")
    _, kei_prime, kei_second = kelvin_kei(rho)
    prefactor = p / (2.0 * math.pi)
    return (
        np.atleast_1d(prefactor * (kei_second + nu * kei_prime / rho)),
        np.atleast_1d(prefactor * (kei_prime / rho + nu * kei_second)),
    )


# ------------------------------------------------------- patch: Hankel route
def _hankel_integral(eps: float, rho: float, kind: str) -> float:
    """``int_0^inf J1(eps s) K(rho s) s**n / (1 + s**4) ds`` (``n = 0,1,2``).

    ``kind`` picks the differentiated Bessel kernel ``K`` and the power:

    ======== ========================== ===
    kind     K(rho s)                   n
    ======== ========================== ===
    ``w``    ``J0(rho s)``              0
    ``dr``   ``-J1(rho s)``             1
    ``drr``  ``J0''(rho s)``            2
    ======== ========================== ===

    The integration variable is ``u = sqrt(s)``, which turns the far tail into
    ``u**-6`` or steeper and keeps the ``a -> 0`` limit (a point-like load, whose
    tail is only ``s**-3``) convergent at fixed quadrature cost.
    """
    if kind not in ("w", "dr", "drr"):
        raise ValueError(f"unknown kind {kind!r}")
    exponent = {"w": 0, "dr": 1, "drr": 2}[kind]

    def integrand(u: float) -> float:
        s = u * u
        if kind == "w":
            kernel = j0(rho * s)
        elif kind == "dr":
            kernel = -j1(rho * s)
        else:  # drr
            if rho == 0.0:
                kernel = -0.5
            else:
                kernel = j1(rho * s) / (rho * s) - j0(rho * s)
        return j1(eps * s) * kernel * s**exponent / (1.0 + s**4) * 2.0 * u

    cap = max(150.0, 8.0 / math.sqrt(eps))
    value, _ = quad(integrand, 0.0, cap, limit=2000, epsabs=1e-15, epsrel=1e-13)
    return value


def patch_load_deflection(
    r,
    *,
    load: float,
    radius: float,
    rigidity: float,
    modulus: float,
    poisson_ratio: float = 0.3,
) -> np.ndarray:
    """Downward deflection of a uniform circular patch load of radius ``radius``.

    Exact Hankel-transform solution of ``D lap^2 w + k w = q`` for a uniform
    pressure on a disc, ``load`` being the total force (N) so the pressure is
    ``P / (pi a^2)``.  ``radius -> 0`` reproduces :func:`point_load_deflection`.
    """
    p = _finite("load", load)
    a = _finite("radius", radius, positive=True)
    d = _finite("rigidity", rigidity, positive=True)
    k = _finite("modulus", modulus, positive=True)
    _finite("poisson_ratio", poisson_ratio)
    ell = characteristic_length(d, k)
    prefactor = p / (math.pi * a * k * ell)
    return np.array([prefactor * _hankel_integral(a / ell, float(rr) / ell, "w") for rr in np.atleast_1d(r)])


def patch_load_moments(
    r,
    *,
    load: float,
    radius: float,
    rigidity: float,
    modulus: float,
    poisson_ratio: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """``(M_r, M_t)`` of a uniform circular patch load; finite everywhere."""
    p = _finite("load", load)
    a = _finite("radius", radius, positive=True)
    d = _finite("rigidity", rigidity, positive=True)
    k = _finite("modulus", modulus, positive=True)
    nu = _finite("poisson_ratio", poisson_ratio)
    ell = characteristic_length(d, k)
    eps = a / ell
    prefactor = p / (math.pi * a * k * ell**3)
    radial, hoop = [], []
    for rr in np.atleast_1d(r):
        rho = float(rr) / ell
        drr = _hankel_integral(eps, rho, "drr")
        # w'/r has the finite limit w''(0) as rho -> 0 (the drr kernel at rho = 0).
        dr_over_r = _hankel_integral(eps, rho, "dr") / rho if rho > 0.0 else drr
        radial.append(-d * prefactor * (drr + nu * dr_over_r))
        hoop.append(-d * prefactor * (dr_over_r + nu * drr))
    return np.array(radial), np.array(hoop)


def patch_centre_moment(
    *,
    load: float,
    radius: float,
    rigidity: float,
    modulus: float,
    poisson_ratio: float = 0.3,
) -> float:
    """Exact sagging moment at the centre of a uniform circular patch (N m).

    Superposing the exact point-load field over a ring of radius ``s`` and
    averaging the moment tensor over the ring gives ``(M_rr + M_tt)/2 = 
    (1+nu) (kei'' + kei'/rho) / (4 pi)`` per unit load, and the radial integral
    collapses to a boundary term,

        M(0) = P (1+nu) l / (2 pi a) * kei'(a/l),

    which is the same value the Hankel route reaches through the identity
    ``int_0^inf s**2 J1(eps s)/(1+s**4) ds = kei'(eps)``.  Equibiaxial at the
    centre, so ``M_r(0) = M_t(0)``.
    """
    p = _finite("load", load)
    a = _finite("radius", radius, positive=True)
    d = _finite("rigidity", rigidity, positive=True)
    k = _finite("modulus", modulus, positive=True)
    nu = _finite("poisson_ratio", poisson_ratio)
    ell = characteristic_length(d, k)
    _, kei_prime, _ = kelvin_kei(np.array([a / ell]))
    return float(p * (1.0 + nu) * ell / (2.0 * math.pi * a) * kei_prime[0])


def westergaard_max_stress(
    load: float,
    *,
    radius: float,
    thickness: float,
    young_modulus: float,
    poisson_ratio: float,
    modulus: float,
) -> float:
    """Westergaard (1926)/Kerr (1979) peak bottom tensile stress (Pa).

    ``sigma = 3 (1+nu) P / (2 pi h^2) * (ln(l/a) + 0.6159)`` for a patch radius
    ``a`` small against ``l``.  It is an asymptote, so it departs from
    :func:`patch_centre_moment` only as ``O((a/l)**2 ln(a/l))``.
    """
    p = _finite("load", load)
    a = _finite("radius", radius, positive=True)
    h = _finite("thickness", thickness, positive=True)
    nu = _finite("poisson_ratio", poisson_ratio)
    d = plate_rigidity(young_modulus, thickness, poisson_ratio)
    ell = characteristic_length(d, modulus)
    if a >= ell:
        raise ValueError("the Westergaard asymptote needs radius << characteristic_length")
    return 3.0 * (1.0 + nu) * p / (2.0 * math.pi * h**2) * (math.log(ell / a) + WESTERGAARD_CONSTANT)


def ice_sheet_bearing_capacity(
    *,
    thickness: float,
    young_modulus: float = 5.0e9,
    poisson_ratio: float = 0.3,
    load_radius: float = 0.15,
    water_density: float = SEA_WATER_DENSITY,
    gravity: float = GRAVITY,
    allowable_stress: float | None = None,
    flexural_strength: float = 0.5e6,
) -> dict[str, Any]:
    """Bearing capacity of a floating ice sheet under a circular patch load.

    The exact centre moment sets the stress, and the published asymptote is
    reported alongside so a user can see whether the asymptote applies (it needs
    ``load_radius << characteristic_length``).  The allowable load follows from
    ``allowable_stress``, defaulting to ``flexural_strength`` -- the bending
    strength of the ice, 0.5 MPa by default (typical first-year sea ice, ISO 19906
    order of magnitude), *not* a fraction of ``young_modulus``: ice is brittle and
    fails in bending long before it yields.
    """
    young = _finite("young_modulus", young_modulus, positive=True)
    d = plate_rigidity(young, thickness, poisson_ratio)
    k = foundation_modulus(water_density, gravity)
    ell = characteristic_length(d, k)
    a = _finite("load_radius", load_radius, positive=True)
    limit = _finite("flexural_strength", flexural_strength, positive=True) if allowable_stress is None else _finite(
        "allowable_stress", allowable_stress, positive=True)
    centre = abs(patch_centre_moment(
        load=1.0, radius=a, rigidity=d, modulus=k, poisson_ratio=poisson_ratio))
    stress_per_load = 6.0 * centre / thickness**2
    asymptote = (
        westergaard_max_stress(1.0, radius=a, thickness=thickness, young_modulus=young,
                               poisson_ratio=poisson_ratio, modulus=k)
        if a < ell else None)
    return {
        "rigidity_Nm": d,
        "foundation_modulus_N_per_m3": k,
        "characteristic_length_m": ell,
        "load_radius_over_l": a / ell,
        "thickness_over_l": thickness / ell,
        "peak_stress_per_newton_Pa": stress_per_load,
        "westergaard_stress_per_newton_Pa": asymptote,
        "westergaard_relative_error": (
            (asymptote - stress_per_load) / stress_per_load if asymptote else None),
        "allowable_stress_Pa": limit,
        "allowable_load_N": limit / stress_per_load,
        "centre_deflection_per_newton_m": float(patch_load_deflection(
            0.0, load=1.0, radius=a, rigidity=d, modulus=k, poisson_ratio=poisson_ratio)[0]),
    }