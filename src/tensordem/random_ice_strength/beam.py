"""Unit-width bending beam strips of a floating ice plate as a strength bundle.

Why this module exists
----------------------
The deterministic breaking force of a floating ice sheet against a wide
structure is written per unit width of the ice/structure contact line, and the
package already carries that formula -- and its free coefficient -- in
:mod:`tensordem.ice_channel3d.slab`::

    p_b = 0.25 * sigma_f * h**2 / L_c   [N/m] ,
    L_c = (E h**3 / (12 (1 - nu**2) rho_w g))**(1/4)

where ``0.25`` is the Lindqvist bending efficiency that module already
inherited from :mod:`tensordem.level_ice.formula`.  **No new coefficient is
introduced here.**  This module's only job is to make ``sigma_f`` a *field*
rather than a number.

Two failure idealisations, and why both are reported
----------------------------------------------------
Discretise the contact line of width ``B`` into ``n = B/db`` unit-width strips
and give strip ``i`` its own strength ``sigma_i`` from the random field.  What
the structure feels depends on *how the line shares load*, and there are exactly
two clean limits:

``weakest_link`` (reference model -- continuous sheet, nominal-stress control)
    The sheet is continuous and bends as a plate: plane sections rotate
    together, so along the contact line the nominal bending stress is
    **uniform** in the width direction and proportional to the applied
    curvature.  The line therefore fails when the applied stress first reaches
    the *lowest* local strength anywhere along it, and at that instant every
    strip carries the same stress ``sigma_min = min_i sigma_i``::

        F = 0.25 h**2/L_c * B * sigma_min                [N]

    A brittle crack then runs across the width, so the load does not grow past
    that value -- this is the classical weakest-link idealisation behind the
    normalised ice pressure that decreases with contact area (the
    pressure-area effect).  Its statistics are those of the *minimum* of the
    strength field over the width, which is what couples the correlation length
    to the size effect.

``bundle`` (contrast model -- independent floes / broken ice)
    Each strip carries its own share and breaks at its own strength, and the
    total is the sum of the strip capacities::

        F = 0.25 h**2/L_c * sum_i sigma_i * db           [N]

    This is the equal-load-sharing bundle.  It is the right idealisation for a
    *discontinuous* cover (individual floes, brash ice) but it is **not** the
    continuous sheet: here ``F/B`` rises towards the mean as ``B`` grows (the
    width average concentrates), so its normalised 5 % quantile *increases*
    with width.  The two models therefore predict size effects of opposite
    sign, and the report shows both curves instead of silently quoting the one
    that matches expectation.

A structural property that matters for the acceptance test
----------------------------------------------------------
For a fixed realisation the minimum over a wider window can only be smaller:
``sigma_min(B') <= sigma_min(B)`` for ``B' >= B`` **pointwise**.  So when the
width sweep is run on nested windows of one realisation set (common random
numbers), every order statistic -- hence the ensemble mean and every quantile
of ``F/B`` -- is monotone in ``B`` for that idealisation *by construction*, not
only in expectation.  That is why the sweep in
:func:`tensordem.random_ice_strength.ensemble.size_effect_sweep` nests the
windows, and the report shows the non-nested sweep as the noise reference.

Excluded on purpose (listed as limitations, not smuggled in as coefficients)
--------------------------------------------------------------------------
Load redistribution after the first strip fails (which would raise the total
towards the bundle limit), residual post-failure strength, crushing at the
contact, and any velocity/dynamic effect.  Thickness and elastic modulus stay
deterministic: only the strength is a random field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..ice_channel3d.slab import (
    BENDING_EFFICIENCY,
    GRAVITY,
    ICE_DENSITY,
    POISSON_ICE,
    WATER_DENSITY,
    bending_capacity_per_width_n_per_m,
    characteristic_length_m,
)
from .distributions import WeibullMarginal

__all__ = [
    "FAILURE_MODELS",
    "IceSheetStrengthModel",
    "strip_force_per_width",
    "strip_forces",
    "breaking_force",
    "deterministic_breaking_force",
    "weakest_link_strength",
    "BENDING_EFFICIENCY",
]

#: ``weakest_link`` = continuous sheet, nominal-stress control (reference);
#: ``bundle`` = equal-load-sharing sum over independent strips (contrast).
FAILURE_MODELS = ("weakest_link", "bundle")


@dataclass(frozen=True)
class IceSheetStrengthModel:
    """Deterministic sheet geometry/material plus the random strength marginal."""

    marginal: WeibullMarginal
    thickness_m: float = 1.0
    elastic_modulus_pa: float = 5.0e9
    poisson_ratio: float = POISSON_ICE
    ice_density_kg_m3: float = ICE_DENSITY
    water_density_kg_m3: float = WATER_DENSITY
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        for name in ("thickness_m", "elastic_modulus_pa", "poisson_ratio",
                     "ice_density_kg_m3", "water_density_kg_m3", "gravity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not 0.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in (0, 0.5)")

    # -- geometry --------------------------------------------------------
    @property
    def char_length_m(self) -> float:
        """Plate characteristic length ``L_c`` (delegated to the graded slab module)."""
        return characteristic_length_m(self.thickness_m, self.elastic_modulus_pa,
                                       rho_water=self.water_density_kg_m3,
                                       poisson=self.poisson_ratio,
                                       gravity=self.gravity)

    @property
    def per_width_coefficient_m(self) -> float:
        """``p_b / sigma_f = 0.25 h**2 / L_c`` [m]: the deterministic transfer factor."""
        return BENDING_EFFICIENCY * self.thickness_m ** 2 / self.char_length_m

    # -- deterministic reference -----------------------------------------
    def deterministic_per_width_n_per_m(self) -> float:
        """``p_b`` evaluated at the *mean* strength -- the deterministic model."""
        return bending_capacity_per_width_n_per_m(self.marginal.mean, self.thickness_m,
                                                  self.char_length_m)

    def deterministic_force(self, width_m: float) -> float:
        if width_m <= 0.0:
            raise ValueError("width must be positive")
        return self.deterministic_per_width_n_per_m() * width_m

    # -- random fields ----------------------------------------------------
    def strip_forces(self, strengths, *, db_m: float = 1.0,
                     failure_model: str = "weakest_link") -> np.ndarray:
        """Total breaking force [N] on the contact line, one value per realisation.

        ``strengths`` is ``(n_real, n_strip)`` (or ``(n_strip,)``); the width is
        ``n_strip * db_m``.  ``failure_model`` selects the idealisation
        documented in the module docstring -- see :data:`FAILURE_MODELS`.
        """
        sigma = np.asarray(strengths, dtype=float)
        if sigma.ndim == 1:
            sigma = sigma[None, :]
        if sigma.ndim != 2:
            raise ValueError("strengths must be (n_real, n_strip) or (n_strip,)")
        if db_m <= 0.0:
            raise ValueError("db_m must be positive")
        if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
            raise ValueError("strengths must be finite and positive")
        n_strip = sigma.shape[1]
        width = n_strip * db_m
        if failure_model == "weakest_link":
            # nominal (plane-section) stress control: the whole line carries the
            # stress of its weakest strip at the instant of first fibre failure
            return self.per_width_coefficient_m * sigma.min(axis=1) * width
        if failure_model == "bundle":
            return self.per_width_coefficient_m * np.sum(sigma, axis=1) * db_m
        raise ValueError(f"unknown failure_model {failure_model!r}; "
                         f"expected one of {FAILURE_MODELS}")

    def per_width(self, force, *, db_m: float, n_strip: int):
        """``F / B`` for the same force ensemble [N/m]."""
        width = db_m * n_strip
        return np.asarray(force, dtype=float) / width

    def info(self) -> Dict[str, float]:
        return {
            "thickness_m": float(self.thickness_m),
            "elastic_modulus_pa": float(self.elastic_modulus_pa),
            "poisson_ratio": float(self.poisson_ratio),
            "ice_density_kg_m3": float(self.ice_density_kg_m3),
            "water_density_kg_m3": float(self.water_density_kg_m3),
            "char_length_m": float(self.char_length_m),
            "per_width_coefficient_m": float(self.per_width_coefficient_m),
            "bending_efficiency": float(BENDING_EFFICIENCY),
            "marginal_modulus": float(self.marginal.modulus),
            "marginal_scale_pa": float(self.marginal.scale),
            "marginal_mean_pa": float(self.marginal.mean),
            "marginal_cov": float(self.marginal.cov),
            "deterministic_per_width_n_per_m": float(self.deterministic_per_width_n_per_m()),
        }


def strip_force_per_width(sigma_f, *, thickness_m: float, char_length_m: float) -> np.ndarray:
    """Per-unit-width strip capacity ``0.25 sigma h**2 / L_c`` [N/m]."""
    sigma = np.asarray(sigma_f, dtype=float)
    return bending_capacity_per_width_n_per_m(sigma, thickness_m, char_length_m)


def weakest_link_strength(strengths) -> np.ndarray:
    """``min_i sigma_i`` over the strips of each realisation (the weakest-link stress)."""
    sigma = np.asarray(strengths, dtype=float)
    if sigma.ndim == 1:
        return sigma
    return sigma.min(axis=1)


def strip_forces(sheet: IceSheetStrengthModel, strengths, *, db_m: float = 1.0,
                 failure_model: str = "weakest_link") -> np.ndarray:
    """Convenience wrapper: :meth:`IceSheetStrengthModel.strip_forces`."""
    return sheet.strip_forces(strengths, db_m=db_m, failure_model=failure_model)


def breaking_force(sheet: IceSheetStrengthModel, strengths, *, db_m: float = 1.0,
                   failure_model: str = "weakest_link") -> np.ndarray:
    """Total breaking force [N] on a contact line, one value per realisation."""
    return sheet.strip_forces(strengths, db_m=db_m, failure_model=failure_model)


def deterministic_breaking_force(sheet: IceSheetStrengthModel, width_m: float) -> float:
    """Deterministic model value [N] at the mean strength (no randomness at all)."""
    return sheet.deterministic_force(width_m)