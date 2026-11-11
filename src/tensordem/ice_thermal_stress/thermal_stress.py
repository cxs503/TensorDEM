"""Thermal stress in an ice block heated or cooled by a collision.

Step 3 of the node: ``sigma_thermal = E alpha_T dT / (1 - nu)``, superposed on
the mechanical contact stress.  The naive formula is only valid for a body
whose thermal strain is *fully restrained* and uniform.  A collision never
produces a uniform temperature field, so this module implements the three
classical cases and shows how they differ -- the difference is the whole point
of "superpose carefully":

1. **Fully restrained, uniform heating** :func:`restrained_uniform_stress`::

       sigma = - E alpha_T dT / (1 - nu)          (dT > 0 -> compression)

   The formula quoted in the node specification.

2. **Fully restrained, non-uniform heating** :func:`restrained_gradient_stress`

       sigma(z) = - E alpha_T (T(z) - <T>) / (1 - nu)

   with ``<T>`` the thickness average.  This field carries **no membrane
   resultant** (``int sigma dz = 0`` exactly), which is the invariant the tests
   check: a free-standing block cannot build a net axial force out of its own
   temperature field.

3. **Unrestrained (free) slab with a linear through-thickness gradient**
   :func:`free_slab_gradient_peak`::

       sigma_max = E alpha_T dT / (2 (1 - nu))

   i.e. **half** of case 1 -- the classical result for a plate that is free to
   bend.  Cases 1 and 3 together bound the collision problem: the contact
   patch is restrained by the surrounding cold ice (case 1 for the patch), the
   block as a whole is free (case 3).  The report quotes both.

Sign convention
---------------
Compression is **negative**, tension positive, as in the rest of the ice
mechanics chain.  Heating a restrained patch therefore makes its stress *more
compressive* (helps crushing), while rapid *cooling* of a surface makes it
tensile (helps thermal cracking).

Verification
------------
* :func:`stress_equilibrium_residual` -- ``int sigma dA`` of case 2, which must
  vanish; a non-zero value means the superposition is not statically admissible.
* :func:`restrained_vs_free_ratio` -- must be exactly 2.
* the benchmark compares the stress recovered from the finite-volume
  temperature field against the analytic field at the same nodes, and against
  the finite-element temperature field (:mod:`tensordem.ice_thermal_stress.
  conduction`), so the *field* is graded, not just the formula.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .properties import IceProperties, ice_properties

__all__ = [
    "restrained_uniform_stress",
    "restrained_gradient_stress",
    "free_slab_gradient_peak",
    "thermal_shock_critical_delta_t",
    "stress_equilibrium_residual",
    "restrained_vs_free_ratio",
    "field_stress_from_temperature",
    "STRAIN_RESTRAINT_BOUNDS",
]


#: ``("free", "restrained")`` -- the two limiting restraints of the collision
#: patch.  The truth is bracketed between them; the report quotes both.
STRAIN_RESTRAINT_BOUNDS = ("free", "restrained")


def _props(properties: IceProperties | float) -> IceProperties:
    return properties if isinstance(properties, IceProperties) else ice_properties(properties)


def restrained_uniform_stress(
    delta_t_k: Any,
    properties: IceProperties | float,
    *,
    poisson: float | None = None,
) -> Any:
    """``-E alpha_T dT / (1 - nu)``: full restraint, uniform temperature rise.

    Positive ``delta_t_k`` (heating) returns a **negative** (compressive)
    stress.  Accepts scalars and arrays.

    Examples
    --------
    >>> round(restrained_uniform_stress(20.0, -10.0) / 1e6, 2)
    -16.37
    """
    props = _props(properties)
    nu = props.poisson if poisson is None else float(poisson)
    out = -props.youngs_modulus_pa * props.alpha_linear_per_k * np.asarray(delta_t_k, float) / (
        1.0 - nu)
    return float(out) if np.ndim(out) == 0 else out


def restrained_gradient_stress(
    temperature_profile: Any,
    properties: IceProperties | float,
    *,
    dz: Any | None = None,
    poisson: float | None = None,
) -> np.ndarray:
    """``-E alpha_T (T(z) - <T>)/(1 - nu)``: full restraint, any ``T(z)``.

    ``<T>`` is the *area weighted* (thickness averaged) temperature, so the
    returned field is self-equilibrated by construction.  ``dz`` (cell widths)
    defaults to the uniform grid implied by ``len(T)``.
    """
    props = _props(properties)
    nu = props.poisson if poisson is None else float(poisson)
    t = np.asarray(temperature_profile, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("temperature_profile must be a 1-D array with >= 2 values")
    weights = np.ones_like(t) if dz is None else np.asarray(dz, dtype=float)
    if weights.shape != t.shape:
        raise ValueError("dz must have the same shape as the profile")
    mean = float(np.sum(t * weights) / np.sum(weights))
    return -props.youngs_modulus_pa * props.alpha_linear_per_k * (t - mean) / (1.0 - nu)


def free_slab_gradient_peak(
    delta_t_k: Any,
    properties: IceProperties | float,
    *,
    poisson: float | None = None,
) -> Any:
    """``E alpha_T dT / (2 (1 - nu))``: free bending slab, linear gradient.

    Half the restrained value (Boley & Weiner 1960, sec. 12.6).  Returns the
    magnitude as a positive number for a positive ``dT``.
    """
    props = _props(properties)
    nu = props.poisson if poisson is None else float(poisson)
    out = (props.youngs_modulus_pa * props.alpha_linear_per_k
           * np.asarray(delta_t_k, float) / (2.0 * (1.0 - nu)))
    return float(out) if np.ndim(out) == 0 else out


def thermal_shock_critical_delta_t(
    strength_pa: float,
    properties: IceProperties | float,
    *,
    poisson: float | None = None,
    restraint: str = "restrained",
) -> float:
    """Temperature jump at which thermal stress alone reaches the strength (K).

    ``dT_c = sigma_f (1 - nu) / (E alpha_T)`` for full restraint, twice that for
    a free slab.  This is the *falsifiable* number of the node: any claim that
    thermal cracking matters in a given collision must be checked against it.

    Examples
    --------
    >>> round(thermal_shock_critical_delta_t(25.0e6, -10.0), 1)
    30.5
    """
    props = _props(properties)
    nu = props.poisson if poisson is None else float(poisson)
    sigma = float(strength_pa)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"strength_pa must be > 0, got {strength_pa!r}")
    base = sigma * (1.0 - nu) / (props.youngs_modulus_pa * props.alpha_linear_per_k)
    if restraint == "restrained":
        return base
    if restraint == "free":
        return 2.0 * base
    raise ValueError(f"restraint must be 'restrained' or 'free', got {restraint!r}")


def stress_equilibrium_residual(
    sigma: Any, dz: Any | None = None
) -> float:
    """``|int sigma dz| / max|sigma|`` -- must vanish for a self-equilibrated field."""
    s = np.asarray(sigma, dtype=float)
    w = np.ones_like(s) if dz is None else np.asarray(dz, dtype=float)
    scale = float(np.max(np.abs(s)))
    if scale == 0.0:
        return 0.0
    return float(abs(np.sum(s * w)) / scale)


def restrained_vs_free_ratio() -> float:
    """Magnitude ratio of the restrained to the free-slab stress (=2)."""
    props = ice_properties(-10.0)
    a = float(restrained_uniform_stress(10.0, props))
    b = float(free_slab_gradient_peak(10.0, props))
    return abs(a) / abs(b)


def field_stress_from_temperature(
    temperature_profile: Any,
    properties: IceProperties | float,
    *,
    dz: Any | None = None,
    restraint: str = "restrained",
) -> np.ndarray:
    """Stress field for a computed ``T(z)``: restricted or free-bending branch.

    ``restraint = "restrained"``  -> :func:`restrained_gradient_stress`
    ``restraint = "free"``        -> the same shape scaled by 1/2, which is the
    bending stress of a free slab with the *same* gradient (thin-plate result).

    The *sign* of the returned field is kept physical: hot material in a
    restrained body is in compression (negative).
    """
    sigma = restrained_gradient_stress(temperature_profile, properties, dz=dz)
    t = np.asarray(temperature_profile, dtype=float)
    dev = t - (np.sum(t) / t.size)
    sigma = np.where(np.abs(dev) > 0.0, sigma, 0.0)  # kill round-off at the mean
    if restraint == "restrained":
        return sigma
    if restraint == "free":
        return 0.5 * sigma
    raise ValueError(f"restraint must be 'restrained' or 'free', got {restraint!r}")


#: module self-checks: the two classical bounds and the equilibrium invariant.
_p = ice_properties(-10.0)
assert abs(restrained_vs_free_ratio() - 2.0) < 1e-12
assert float(restrained_uniform_stress(0.0, _p)) == 0.0
_z = np.linspace(0.0, 0.01, 101)
_T = 5.0 * np.exp(-_z / 0.002)
_sig = restrained_gradient_stress(_T, _p)
assert stress_equilibrium_residual(_sig) < 1e-12
assert np.all(_sig[:5] < 0.0) and np.all(_sig[-5:] > 0.0)