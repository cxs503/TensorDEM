"""Effective-medium predictions for a confined Hertz-Mindlin sphere packing.

Digby (1981, *J. Appl. Mech.* 48:803) derived the effective isotropic moduli of
a random packing of identical elastic spheres in Hertzian contact.  The result
quoted by Digby and reproduced in the rock-physics literature (Mavko, Mukerji &
Dvorkin, *The Rock Physics Handbook*, "Hertz-Mindlin" model) is

    K_HM = [ Z^2 (1-phi)^2 G^2 P / (18 pi^2 (1-nu)^2) ]^(1/3)
    G_HM = (5-4nu)/(5(2-nu)) * [ 3 Z^2 (1-phi)^2 G^2 P / (2 pi^2 (1-nu)^2) ]^(1/3)

with ``Z`` the coordination number, ``phi`` the porosity, ``G``/``nu`` the grain
shear modulus and Poisson ratio, ``P`` the effective confining pressure.

The same result follows from the affine (Voigt-like) average over contacts of
tangent stiffness ``k_n``, ``k_t`` (see :func:`affine_contact_medium`), which is
what makes the ``P^(1/6)`` wave-speed scaling a *test* rather than an
assumption: ``K, G ~ P^(1/3)`` and ``v = sqrt(modulus/density) ~ P^(1/6)``.

Everything here is closed-form, unit-consistent (SI) and free of tunables.
"""
from __future__ import annotations

import math
from typing import Any


def _hm_prefactor(z: float, porosity: float, g_grain: float, nu: float,
                  pressure: float) -> float:
    if z <= 0 or g_grain <= 0 or pressure <= 0:
        raise ValueError("Z, grain modulus and pressure must be positive")
    if not 0.0 <= nu < 0.5:
        raise ValueError("grain Poisson ratio must be in [0, 0.5)")
    return (z * z * (1.0 - porosity) ** 2 * g_grain**2 * pressure
            / (1.0 - nu) ** 2)


def digby_hertz_mindlin(*, coordination_number: float, porosity: float,
                        grain_shear_modulus: float, grain_poisson_ratio: float,
                        pressure: float, bulk_density: float) -> dict[str, Any]:
    """Digby (1981) / Hertz-Mindlin effective moduli and wave speeds.

    Args:
        coordination_number:   measured ``Z`` of the load-bearing network.
        porosity:              ``phi`` of the current packing.
        grain_shear_modulus:   ``G`` of the grain material, Pa.
        grain_poisson_ratio:   ``nu`` of the grain material.
        pressure:              effective confining pressure, Pa.
        bulk_density:          packing bulk density ``rho``, kg/m^3.

    Returns:
        dict with ``K_Pa``, ``G_Pa``, ``v_p_m_s``, ``v_s_m_s``, ``poisson``,
        ``z``, ``porosity``, ``pressure_Pa``.
    """
    z = float(coordination_number)
    phi = float(porosity)
    g = float(grain_shear_modulus)
    nu = float(grain_poisson_ratio)
    p = float(pressure)
    rho = float(bulk_density)
    pre = _hm_prefactor(z, phi, g, nu, p)
    k = (pre / (18.0 * math.pi**2)) ** (1.0 / 3.0)
    shear_factor = (5.0 - 4.0 * nu) / (5.0 * (2.0 - nu))
    g_eff = shear_factor * (3.0 * pre / (2.0 * math.pi**2)) ** (1.0 / 3.0)
    return {
        "K_Pa": k,
        "G_Pa": g_eff,
        "v_p_m_s": math.sqrt((k + 4.0 * g_eff / 3.0) / rho),
        "v_s_m_s": math.sqrt(g_eff / rho),
        "poisson": poisson_from_velocities(
            math.sqrt((k + 4.0 * g_eff / 3.0) / rho), math.sqrt(g_eff / rho)),
        "z": z,
        "porosity": phi,
        "pressure_Pa": p,
        "bulk_density_kg_m3": rho,
        "k_over_g": k / g_eff,
    }


def affine_contact_medium(*, coordination_number: float, porosity: float,
                          mean_radius: float, pressure: float,
                          e_star: float, g_star: float,
                          bulk_density: float) -> dict[str, Any]:
    """Affine average of the *measured* contact stiffness (independent check).

    With ``n_p = (1-phi)/((4/3) pi R^3)`` particles per unit volume and ``Z``
    contacts per particle the affine (Voigt) average of the contact network is

        K = 2 R^2 n_p Z k_n / 9,     G = R^2 n_p Z (2 k_n + 3 k_t) / 15,

    evaluated at the mean contact force
    ``F = 3 P / (R n_p Z)`` so that ``k_n = 2 E* a``, ``k_t = 8 G* a``,
    ``a = (3 R F / (8 E*))^(1/3)``.  Substituting the Hertz force law recovers
    the Digby/Hertz-Mindlin closed form (same ``Z^2 (1-phi)^2 G^2 P`` scaling),
    so the two functions must agree to the last digit of the closed form; the
    difference is a regression check on the algebra, not extra physics.
    """
    z = float(coordination_number)
    phi = float(porosity)
    r = float(mean_radius)
    p = float(pressure)
    rho = float(bulk_density)
    n_p = (1.0 - phi) / ((4.0 / 3.0) * math.pi * r**3)
    if z <= 0 or n_p <= 0:
        raise ValueError("coordination number and number density must be positive")
    f_mean = 3.0 * p / (r * n_p * z)
    a = (3.0 * r * f_mean / (8.0 * e_star)) ** (1.0 / 3.0)
    k_n = 2.0 * e_star * a
    k_t = 8.0 * g_star * a
    k = 2.0 * r**2 * n_p * z * k_n / 9.0
    g_eff = r**2 * n_p * z * (2.0 * k_n + 3.0 * k_t) / 15.0
    return {
        "K_Pa": k,
        "G_Pa": g_eff,
        "v_p_m_s": math.sqrt((k + 4.0 * g_eff / 3.0) / rho),
        "v_s_m_s": math.sqrt(g_eff / rho),
        "poisson": poisson_from_velocities(
            math.sqrt((k + 4.0 * g_eff / 3.0) / rho), math.sqrt(g_eff / rho)),
        "contact_radius_m": a,
        "k_n_N_m": k_n,
        "k_t_N_m": k_t,
        "mean_normal_force_N": f_mean,
        "poisson_from_stiffness": (k_n - k_t) / (4.0 * k_n + k_t),
    }


def poisson_from_velocities(v_p: float, v_s: float) -> float:
    """Isotropic Poisson ratio from the P/S wave-speed ratio."""
    if v_p <= 0 or v_s <= 0:
        raise ValueError("wave speeds must be positive")
    r2 = (v_p / v_s) ** 2
    if r2 <= 1.0:
        raise ValueError("v_p must exceed v_s")
    return (r2 - 2.0) / (2.0 * (r2 - 1.0))


def moduli_from_velocities(bulk_density: float, v_p: float, v_s: float) -> dict[str, float]:
    """Isotropic ``(K, G)`` from bulk density and P/S wave speeds."""
    rho = float(bulk_density)
    g = rho * v_s**2
    k = rho * v_p**2 - 4.0 * g / 3.0
    return {"K_Pa": k, "G_Pa": g, "poisson": poisson_from_velocities(v_p, v_s)}


def loglog_slope(x: list[float] | tuple[float, ...],
                 y: list[float] | tuple[float, ...]) -> float:
    """Least-squares slope of ``log(y)`` against ``log(x)``."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two matched points")
    if any(v <= 0 for v in list(x) + list(y)):
        raise ValueError("log-log regression needs strictly positive data")
    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    if sxx == 0.0:
        raise ValueError("degenerate x values")
    return sxy / sxx