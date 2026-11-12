"""Array-level 2-D Virial stress built on the certified 3-D instrument.

Why this module exists
----------------------
``granular3d.stress`` is the shared, independently verified per-contact /
volume-average stress instrument of this repository.  It is *dimension locked*
to ``(m, 3)`` vectors because it was written for the 3-D cases.  The 2-D plane
shear case (``elastic_limit_shear``, ``CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT``)
nevertheless carries its own private scalar sum ``S_xy = sum F_i,x (r_i-r_j)_y``
inside the force kernel.  A private sum cannot be cross-checked, and it hides a
*choice*: with a frictionless (central-force) contact the raw virial is
symmetric so ``sigma_xy == sigma_yx`` and the choice is invisible; with Coulomb
tangential forces acting, the raw virial is **not** symmetric and the two
components differ by the per-contact lever-arm torque

.. math:: \\sigma_{xy} - \\sigma_{yx} = \\frac{1}{A}\\sum_c l_x f_y - l_y f_x
                                      = \\frac{1}{A}\\sum_c |l|\\, f_t ,

which is exactly the non-central part the 3-D instrument documents as
``sigma - sigma.T``.  Reporting one of the two components as "the shear
stress" while adding a symmetric kinetic term on top of it is a bookkeeping
defect that no gate can see.

This module removes the private sum by *reusing* the certified per-contact
tensor: the 2-D vectors are zero-padded to 3-D, handed to
:func:`tensordem.granular3d.stress.contact_outer`, and the in-plane block is
returned.  Nothing is re-derived here, so the 2-D case and the 3-D cases cannot
drift apart.

Conventions (identical to ``granular3d.stress``)
-----------------------------------------------
``branch`` = ``l = x_j - x_i``; ``force_on_i`` = force on ``i`` from ``j``.
``sigma = (1/A) sum_c l (x) f``, tension positive / compression negative, so a
repulsive pair gives ``-tr(sigma)/2 > 0``.

Limitations (stated, not implied)
---------------------------------
* 2-D only *as a wrapper*: the underlying tensor is 3x3 with the ``z`` row and
  column identically zero.  ``pressure_2d`` therefore uses ``-tr/2``; the 3-D
  ``granular3d.stress.pressure`` (``-tr/3``) is **not** the 2-D pressure of the
  same state and must not be mixed with it.
* The kinetic (Reynolds) part stays separate, as in the 3-D instrument; use
  :func:`tensordem.granular3d.stress.kinetic_stress` for it (it is dimension
  agnostic).
"""

from __future__ import annotations

import numpy as np

from ..granular3d.stress import contact_outer

__all__ = [
    "contact_tensors",
    "virial",
    "pressure_2d",
    "shear_components",
]


def _vectors2(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (m, 2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _pad(v: np.ndarray) -> np.ndarray:
    """Embed ``(m, 2)`` vectors in 3-D by appending a zero ``z`` component."""
    out = np.zeros((v.shape[0], 3), dtype=float)
    out[:, :2] = v
    return out


def contact_tensors(branch, force_on_i) -> np.ndarray:
    """Per-contact tensors ``l (x) f`` of shape ``(m, 3, 3)`` (in-plane block)."""
    l = _vectors2(branch, "branch")
    f = _vectors2(force_on_i, "force_on_i")
    if l.shape != f.shape:
        raise ValueError(f"branch {l.shape} and force {f.shape} must match")
    return contact_outer(_pad(l), _pad(f))


def virial(branch, force_on_i, area: float) -> np.ndarray:
    """Volume (area) averaged Cauchy stress ``(1/A) sum_c l (x) f``, shape (3, 3)."""
    area = float(area)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("area must be finite and positive")
    return contact_tensors(branch, force_on_i).sum(axis=0) / area


def pressure_2d(sigma) -> float:
    """2-D scalar pressure ``p = -(sigma_xx + sigma_yy)/2`` (positive in compression)."""
    s = np.asarray(sigma, dtype=float)
    if s.shape not in ((3, 3), (2, 2)):
        raise ValueError("sigma must be 2x2 or 3x3")
    return float(-(s[0, 0] + s[1, 1]) / 2.0)


def shear_components(branch, force_on_i, area: float) -> dict:
    """Both off-diagonal components, the symmetric shear and their gap.

    Returns
    -------
    dict with
        ``sigma_xy``, ``sigma_yx`` : the two raw components,
        ``tau_symmetric``          : ``(sigma_xy + sigma_yx)/2`` -- the value a
                                     symmetric-stress closure is defined on,
        ``tau_antisymmetric``      : ``(sigma_xy - sigma_yx)/2`` -- the
                                     non-central (lever-arm) part in stress units,
        ``asymmetry_relative``     : ``|sigma_xy - sigma_yx| / |tau_symmetric|``
                                     (``nan`` when the symmetric part vanishes),
        ``n_contacts``             : number of contact rows used.
    """
    sigma = virial(branch, force_on_i, area)
    sxy = float(sigma[0, 1])
    syx = float(sigma[1, 0])
    tau_sym = 0.5 * (sxy + syx)
    tau_asym = 0.5 * (sxy - syx)
    denom = abs(tau_sym)
    rel = abs(sxy - syx) / denom if denom > 0.0 else float("nan")
    return {
        "sigma_xy": sxy,
        "sigma_yx": syx,
        "tau_symmetric": tau_sym,
        "tau_antisymmetric": tau_asym,
        "asymmetry_relative": rel,
        "n_contacts": int(np.asarray(branch, dtype=float).reshape(-1, 2).shape[0])
        if np.asarray(branch).size else 0,
    }