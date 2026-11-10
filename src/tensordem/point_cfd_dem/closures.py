"""Analytic closures for *point-particle* (CFD-DEM) two-way coupling.

This module is deliberately closure-only (no state, no I/O) so that every
coefficient appearing in a simulation can be checked against its textbook
definition in a unit test.

Conventions
-----------
* ``mu`` is the *dynamic* viscosity (Pa s), ``rho_f``/``rho_p`` the fluid and
  particle densities, ``d`` the particle diameter.
* Forces are returned as vectors; magnitudes are positive for a relative
  velocity that opposes the particle motion.
* The gravity vector ``g`` points **downwards** (``g = (0, 0, -9.81)`` in SI),
  i.e. ``F_net_weight = (rho_p - rho_f) V_p g`` gives the sinking force on a
  heavy particle.

References
----------
Stokes (1851); Schiller & Naumann (1935); Richardson & Zaki (1954) ::

    u_s / u_t = eps**n ,   n = 4.65                      (Re_t < 0.2)
                          n = 4.35 Re_t**-0.03           (0.2 < Re_t < 1)
                          n = 4.45 Re_t**-0.1            (1 < Re_t < 500)
                          n = 2.39                       (Re_t > 500)

with ``eps = 1 - phi`` the voidage and ``u_s`` the settling velocity measured in
the *container* frame (zero net volumetric flux).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "sphere_volume",
    "sphere_mass",
    "particle_reynolds",
    "stokes_drag_coefficient",
    "stokes_drag_force",
    "schiller_naumann_correction",
    "drag_correction",
    "drag_force",
    "terminal_velocity_stokes",
    "richardson_zaki_exponent",
    "richardson_zaki_hindrance",
    "relaxation_time_stokes",
    "added_mass_coefficient",
    "added_mass_force",
]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def sphere_volume(diameter):
    """Volume of a sphere, ``pi d^3 / 6``."""
    return math.pi * float(diameter) ** 3 / 6.0


def sphere_mass(diameter, density):
    """Mass of a solid sphere."""
    return sphere_volume(diameter) * float(density)


# --------------------------------------------------------------------------
# drag
# --------------------------------------------------------------------------
def particle_reynolds(rho_f, mu, diameter, u_rel):
    """Particle Reynolds number ``Re = rho_f |u_rel| d / mu``.

    ``u_rel`` may be a scalar speed or a vector; vectors are reduced to their
    magnitude.
    """
    arr = np.asarray(u_rel, dtype=float)
    if arr.ndim == 0:
        speed = abs(float(arr))
    else:
        speed = np.linalg.norm(arr, axis=-1)
    return rho_f * speed * np.asarray(diameter, dtype=float) / float(mu)


def stokes_drag_coefficient(mu, diameter):
    """Stokes drag coefficient ``3 pi mu d`` (force = coeff * |u_rel|).

    ``diameter`` may be a scalar or a per-particle array.
    """
    return 3.0 * math.pi * float(mu) * np.asarray(diameter, dtype=float)


def stokes_drag_force(mu, diameter, u_rel):
    """Vector Stokes drag ``3 pi mu d (u_f - u_p)``."""
    coeff = stokes_drag_coefficient(mu, diameter)
    u_rel = np.asarray(u_rel, dtype=float)
    if np.ndim(coeff) == 0:
        return coeff * u_rel
    return np.asarray(coeff)[..., None] * u_rel


def schiller_naumann_correction(Re):
    """Schiller-Naumann drag ratio ``C_D Re / 24 = 1 + 0.15 Re**0.687``.

    This is the multiplicative correction to the Stokes drag; it tends to 1 as
    ``Re -> 0`` so the Stokes limit is recovered exactly.
    """
    Re = np.asarray(Re, dtype=float)
    return 1.0 + 0.15 * np.power(np.maximum(Re, 0.0), 0.687)


def drag_correction(Re, scheme="schiller_naumann"):
    """Multiplicative correction with respect to the Stokes drag.

    ``scheme``
        ``"stokes"``           pure Stokes (Re -> 0 limit, no correction)
        ``"schiller_naumann"`` standard finite-Re correction
    """
    if scheme == "stokes":
        return np.ones_like(np.asarray(Re, dtype=float))
    if scheme == "schiller_naumann":
        return schiller_naumann_correction(Re)
    raise ValueError(f"unknown drag scheme {scheme!r}")


def drag_force(mu, diameter, u_rel, rho_f=1.0, hindrance=None,
               scheme="schiller_naumann"):
    """Full point-particle drag on a sphere.

    ``F = 3 pi mu d (u_f - u_p) * f(Re) * H(eps)``

    ``hindrance`` is an optional multiplicative factor (>= 1) that models the
    reduction of the mobility in a suspension, e.g.
    :func:`richardson_zaki_hindrance`.
    """
    u_rel = np.asarray(u_rel, dtype=float)
    Re = particle_reynolds(rho_f, mu, diameter, u_rel)
    coeff = stokes_drag_coefficient(mu, diameter) * drag_correction(Re, scheme)
    if hindrance is not None:
        coeff = coeff * np.asarray(hindrance, dtype=float)
    if u_rel.ndim == 1:
        return coeff * u_rel
    # broadcast a per-particle scalar coefficient onto the last axis
    return coeff[..., None] * u_rel


def terminal_velocity_stokes(rho_p, rho_f, g, diameter, mu):
    """Analytic Stokes terminal velocity ``v_t = (rho_p - rho_f) g d^2 / (18 mu)``.

    ``g`` is the *magnitude* of the gravitational acceleration.
    """
    return (float(rho_p) - float(rho_f)) * float(g) * float(diameter) ** 2 / (
        18.0 * float(mu)
    )


def relaxation_time_stokes(rho_p, diameter, mu):
    """Particle relaxation time ``tau_p = rho_p d^2 / (18 mu)``."""
    return float(rho_p) * float(diameter) ** 2 / (18.0 * float(mu))


# --------------------------------------------------------------------------
# hindered settling (Richardson & Zaki 1954)
# --------------------------------------------------------------------------
def richardson_zaki_exponent(Re_t):
    """Richardson-Zaki exponent ``n`` as a function of the single-particle Re."""
    Re_t = np.asarray(Re_t, dtype=float)
    n = np.where(
        Re_t < 0.2,
        4.65,
        np.where(
            Re_t < 1.0,
            4.35 * np.power(np.maximum(Re_t, 1e-12), -0.03),
            np.where(Re_t < 500.0,
                     4.45 * np.power(np.maximum(Re_t, 1e-12), -0.1),
                     2.39),
        ),
    )
    if n.ndim == 0:
        return float(n)
    return n


def richardson_zaki_hindrance(voidage, n):
    """Drag multiplier consistent with ``u_rel / u_t = eps**(n-1)``.

    Derivation: balance of the hindered drag against the buoyant weight gives
    ``u_rel = u_t / H``.  The Richardson-Zaki statement for the *relative*
    velocity is ``u_rel = u_t eps**(n-1)``, hence ``H = eps**(1-n)``.  The
    settling velocity in the container frame (zero net volumetric flux) is then
    ``u_s = eps u_rel = u_t eps**n``, i.e. the classical RZ formula.
    """
    eps = np.asarray(voidage, dtype=float)
    return np.power(np.maximum(eps, 1e-12), 1.0 - np.asarray(n, dtype=float))


# --------------------------------------------------------------------------
# added mass
# --------------------------------------------------------------------------
def added_mass_coefficient(shape="sphere_inviscid"):
    """Added mass coefficient ``C_A`` (force ``= -C_A rho_f V_p d(u_p-u_f)/dt``)."""
    table = {"sphere_inviscid": 0.5, "sphere_creeping_flow": 0.5,
             "sphere_potential": 0.5, "none": 0.0}
    try:
        return table[shape]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown added-mass shape {shape!r}")


def added_mass_force(rho_f, volume, acceleration_rel, coefficient=0.5):
    """Added-mass force ``-C_A rho_f V_p a_rel`` for an explicitly known ``a_rel``.

    Inside the coupled integrator the acceleration dependent part is treated
    *implicitly* (see :mod:`tensordem.point_cfd_dem.coupling`); this helper is
    the explicit form used for reporting and for analytic checks.
    """
    return -float(coefficient) * float(rho_f) * float(volume) * np.asarray(
        acceleration_rel, dtype=float
    )