"""Rate-dependent (``mu(I)``) contact friction for the DEM contact kernel.

Motivation
----------
Rounds 1-4 of ``CASE-COLUMN-COLLAPSE`` showed that a *constant* contact
friction cannot reproduce the Lube et al. (2004) runout law over the whole
aspect-ratio range: the linear branch (``a < 1.7``) wants ``mu ~ 0.51`` while
the square-root branch (``a > 1.7``) wants ``mu ~ 0.80`` at the same rolling
resistance (``docs/case_column_collapse_round4_m16862.md``).  That demand is a
*rate* demand: a tall column is more inertial than a squat one, and dry
granular friction is known to grow with the inertial number

.. math:: I = \\dot\\gamma d / \\sqrt{P/\\rho}

(Jop, Forterre & Pouliquen 2006, *Nature* 441, 727-730,
doi:10.1038/nature04801; da Cruz, Emam, Prochnow, Roux & Chevoir 2005,
*Phys. Rev. E* 72, 011305, doi:10.1103/PhysRevE.72.011305).  The standard
closure is

.. math:: \\mu(I) = \\mu_s + \\frac{\\mu_2 - \\mu_s}{I_0/I + 1},

whose parameters are *material* constants measured in steady shear -- here
``mu_s``, ``mu_2`` and ``I_0`` are the Jop et al. (2006) values already
carried by :mod:`tensordem.granular_column.scaling` (``JOP_MU_S``,
``JOP_MU_2``, ``JOP_I_0``), so the column-collapse prediction below uses
**no parameter fitted to the runout data**.

Contact-local inertial number
-----------------------------
A contact does not know the macroscopic ``P`` and ``dot gamma``; the standard
DEM estimate uses the contact itself as the local stress/rate gauge:

* stress scale ``P ~ f_n / d^2`` (normal force over one grain cross-section),
* shear rate ``\\dot\\gamma ~ |v_t| / d`` (tangential slip velocity over one grain),

which gives

.. math:: I_c = \\frac{\\dot\\gamma d}{\\sqrt{P/\\rho}}
                = \\frac{|v_t|}{\\sqrt{f_n/(\\rho d^2)}} .

``v_t`` is the same contact-point slip velocity the Cundall-Strack spring sees,
so the closure needs no extra state.  ``I_c`` is a local proxy: it is *not*
the bulk inertial number of the flow, and the mapping between the two is a
modelling assumption (declared in the report, not hidden).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "InertialFriction",
    "JOP_2006",
    "contact_inertial_number",
    "mu_of_inertial_number",
]

#: Jop, Forterre & Pouliquen (2006) ``mu(I)`` parameters (glass beads, quoted
#: values ``mu_s = 0.36``, ``mu_2 = 0.84``, ``I_0 = 0.30``); the same numbers
#: are exported by :mod:`tensordem.granular_column.scaling`.
JOP_2006 = {"mu_s": 0.36, "mu_2": 0.84, "i_0": 0.30}


def mu_of_inertial_number(inertial_number, mu_s: float = JOP_2006["mu_s"],
                          mu_2: float = JOP_2006["mu_2"],
                          i_0: float = JOP_2006["i_0"]):
    """``mu(I) = mu_s + (mu_2 - mu_s) I / (i_0 + I)`` (elementwise).

    ``I <= 0`` returns ``mu_s`` (the quasi-static limit) and ``I -> inf``
    returns ``mu_2``; the result is strictly monotone in ``I`` and never leaves
    ``[mu_s, mu_2]``.
    """
    i_0 = float(i_0)
    if not math.isfinite(i_0) or i_0 <= 0.0:
        raise ValueError("i_0 must be finite and positive")
    mu_s = float(mu_s)
    mu_2 = float(mu_2)
    if not (math.isfinite(mu_s) and math.isfinite(mu_2)):
        raise ValueError("mu_s and mu_2 must be finite")
    if mu_s < 0.0 or mu_2 < mu_s:
        raise ValueError("need 0 <= mu_s <= mu_2")
    i = np.maximum(np.asarray(inertial_number, dtype=np.float64), 0.0)
    with np.errstate(invalid="ignore"):
        frac = i / (i_0 + i)
    # I = inf (zero confining stress) must give the mu_2 asymptote, not NaN
    frac = np.where(np.isinf(i), 1.0, frac)
    return mu_s + (mu_2 - mu_s) * frac


def contact_inertial_number(slip_speed, normal_force, diameter: float,
                            density: float):
    """Contact-local ``I_c = |v_t| / sqrt(f_n/(rho d^2))`` (array in, array out).

    A contact carrying no normal force (``f_n <= 0``) has no friction cap and is
    returned as ``I_c = 0`` instead of dividing by zero.
    """
    diameter = float(diameter)
    density = float(density)
    if not (math.isfinite(diameter) and diameter > 0.0):
        raise ValueError("diameter must be finite and positive")
    if not (math.isfinite(density) and density > 0.0):
        raise ValueError("density must be finite and positive")
    v = np.abs(np.asarray(slip_speed, dtype=np.float64))
    f_n = np.asarray(normal_force, dtype=np.float64)
    stress = np.maximum(f_n, 0.0) / (density * diameter * diameter)
    scale = np.sqrt(stress)
    safe = scale > 0.0
    out = np.zeros_like(v)
    out[safe] = v[safe] / scale[safe]
    return out


@dataclass(frozen=True)
class InertialFriction:
    """``mu(I)`` closure bound to the contact kernel (SI units).

    ``mu_s``   static (quasi-static) friction coefficient
    ``mu_2``   asymptotic dynamic friction coefficient (``I -> inf``)
    ``i_0``    crossover inertial number
    ``i_cap``  optional numerical upper bound on ``I_c`` (``None`` = no cap);
               it only guards against a near-zero normal force on a live
               contact and is reported in the case signature when used.
    """

    mu_s: float = JOP_2006["mu_s"]
    mu_2: float = JOP_2006["mu_2"]
    i_0: float = JOP_2006["i_0"]
    i_cap: float | None = None

    def __post_init__(self) -> None:
        for name in ("mu_s", "mu_2", "i_0"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if float(self.mu_s) < 0.0 or float(self.mu_2) < float(self.mu_s):
            raise ValueError("need 0 <= mu_s <= mu_2")
        if float(self.i_0) <= 0.0:
            raise ValueError("i_0 must be positive")
        if self.i_cap is not None:
            cap = float(self.i_cap)
            if not math.isfinite(cap) or cap <= 0.0:
                raise ValueError("i_cap must be finite and positive when given")

    def inertial_number(self, slip_speed, normal_force, diameter: float,
                        density: float):
        """Contact-local inertial number (see module docstring)."""
        i = contact_inertial_number(slip_speed, normal_force, diameter, density)
        if self.i_cap is not None:
            np.minimum(i, float(self.i_cap), out=i)
        return i

    def effective_mu(self, slip_speed, normal_force, diameter: float,
                     density: float):
        """``mu(I_c)`` for every live contact (array, same shape as inputs)."""
        i = self.inertial_number(slip_speed, normal_force, diameter, density)
        return mu_of_inertial_number(i, self.mu_s, self.mu_2, self.i_0)

    def signature(self) -> dict:
        return {"law": "mu(I) = mu_s + (mu_2 - mu_s) I/(i_0 + I)",
                "reference": ("Jop, Forterre & Pouliquen (2006) Nature 441:727-730, "
                              "doi:10.1038/nature04801"),
                "mu_s": float(self.mu_s), "mu_2": float(self.mu_2),
                "i_0": float(self.i_0),
                "i_cap": (None if self.i_cap is None else float(self.i_cap)),
                "inertial_number_proxy": "I_c = |v_t| / sqrt(f_n/(rho d^2))"}