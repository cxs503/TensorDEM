"""Roundness -> effective friction closure, anchored on the m9375 sphere field.

Physics
-------
A shear cell of **circular** floes mobilises a stress ratio
``mu_sph = |tau| / p`` (the dimensionless effective friction of the field).  That
number is a measured DEM property of the disk field
(:mod:`tensordem.ice_ice_collision`, mission 9375) and is the ``R -> 1`` limit of
the attrition chain.

An **angular** floe adds a geometric interlocking term on top: a neighbour cannot
slide past a corner without riding over it, so part of the tangential work is
spent on the geometric barrier instead of on sliding.  The barrier is the radial
excess of the particle boundary over its inscribed circle, normalised by the
particle size -- exactly the quantity ``I = R_c / R_ins - 1`` of
:mod:`tensordem.floe_attrition.geometry`, which is ``0`` for a circle (no barrier)
and ``sec(pi/n) - 1`` for a sharp regular polygon.  Interlocking can only be
mobilised where floes are in sustained contact, so the term is weighted by the
DEM-measured coordination number of the field.

.. math::

    \\mu_{eff}(R, C) = \\underbrace{\\mu_{sph}(C)}_{\\text{DEM, m9375}}
                     + \\kappa_i \\, I(R) \\, \\frac{Z(C)}{Z_{ref}}

with ``kappa_i = 1`` an ``O(1)`` geometric coefficient (pre-registered; **not
fitted**), ``Z_ref = 4`` the 2D isostatic coordination and ``Z(C)`` the DEM
coordination of the same shear cell.  The limit is structurally exact: at
``R = 1`` the interlocking index is ``0.0`` so ``mu_eff`` returns the measured
sphere value bit for bit (criterion C4).

Evidence level / limitations
----------------------------
* ``mu_sph(C)`` and ``Z(C)``: REAL DEM numbers (shear cell of
  :mod:`tensordem.ice_ice_collision`, deterministic seed), file-backed in
  ``artifacts/floe_attrition/sphere_baseline.json``.
* ``I(R)``: REAL geometry (exact closed form, verified to ``1e-12``).
* ``kappa_i = 1``: **not calibrated against DEM at intermediate roundness**.  A
  shear cell of rounded polygons at fixed ``R`` is the DEM calibration that would
  replace it (roadmap); until then the closure is a *geometric model with an
  O(1) coefficient*, and a ``kappa_i`` sensitivity sweep is reported instead of a
  single number whenever the closure drives a headline quantity.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from .geometry import RoundedCube, RoundedPolygon, RoundedPolygon as _RP

__all__ = [
    "SphereBaseline",
    "load_sphere_baseline",
    "DEFAULT_BASELINE",
    "InterlockingFriction",
    "mu_effective_from_index",
    "mu_effective_polygon",
    "mu_effective_cube",
]

#: Fallback sphere baseline (used only when the DEM artifact is absent; the
#: effective friction then carries the ``source = "fallback"`` flag and the
#: evidence level drops from REAL to WITHHELD in the report).
DEFAULT_BASELINE: Dict[float, Tuple[float, float]] = {
    0.30: (1.15, 2.60),
    0.50: (0.75, 3.40),
    0.70: (0.62, 4.10),
}

Z_REF = 4.0  # 2D isostatic coordination number


@dataclass(frozen=True)
class SphereBaseline:
    """``mu_sph(C)`` and ``Z(C)`` of the m9375 disk shear cell (monotone tables)."""

    concentrations: Tuple[float, ...]
    mu_sph: Tuple[float, ...]
    coordination: Tuple[float, ...]
    source: str = "fallback"

    def __post_init__(self) -> None:
        c = np.asarray(self.concentrations, dtype=float)
        m = np.asarray(self.mu_sph, dtype=float)
        z = np.asarray(self.coordination, dtype=float)
        if c.ndim != 1 or c.size < 2 or not (np.diff(c) > 0).all():
            raise ValueError("concentrations must be strictly increasing")
        if m.shape != c.shape or z.shape != c.shape:
            raise ValueError("mu_sph and coordination must match concentrations")
        if not (np.isfinite(m).all() and np.isfinite(z).all()):
            raise ValueError("baseline values must be finite")
        if (m < 0).any() or (z < 0).any():
            raise ValueError("baseline values must be non-negative")
        object.__setattr__(self, "_mu", PchipInterpolator(c, m, extrapolate=False))
        object.__setattr__(self, "_z", PchipInterpolator(c, z, extrapolate=False))

    # -- interpolation with an explicit bracket guard ----------------------
    @property
    def bracket(self) -> Tuple[float, float]:
        return float(self.concentrations[0]), float(self.concentrations[-1])

    def _check(self, concentration: float) -> float:
        c = float(concentration)
        lo, hi = self.bracket
        if not (math.isfinite(c) and lo <= c <= hi):
            raise ValueError(
                f"concentration {c} outside the measured bracket [{lo}, {hi}]; "
                "the DEM baseline must not be extrapolated")
        return c

    def mu(self, concentration: float) -> float:
        return float(self._mu(self._check(concentration)))

    def z(self, concentration: float) -> float:
        return float(self._z(self._check(concentration)))

    def as_dict(self) -> dict:
        return {
            "concentrations": list(self.concentrations),
            "mu_sph": list(self.mu_sph),
            "coordination": list(self.coordination),
            "source": self.source,
            "z_ref": Z_REF,
        }


def load_sphere_baseline(path: Optional[str] = None) -> SphereBaseline:
    """Load the DEM sphere baseline artifact, falling back to ``DEFAULT_BASELINE``.

    The artifact is produced by ``scripts/floe_attrition_sphere_baseline.py``.
    Absence is not an error (the module must stay importable), but the ``source``
    field then reads ``"fallback"`` so reports can grade the evidence honestly.
    """
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        path = os.path.join(here, "artifacts", "floe_attrition", "sphere_baseline.json")
    try:
        with open(path) as fh:
            payload = json.load(fh)
        rows = payload["rows"]
        return SphereBaseline(
            concentrations=tuple(float(r["concentration"]) for r in rows),
            mu_sph=tuple(float(r["mu_sph"]) for r in rows),
            coordination=tuple(float(r["coordination"]) for r in rows),
            source=str(path),
        )
    except (OSError, KeyError, TypeError, ValueError):
        table = sorted(DEFAULT_BASELINE.items())
        return SphereBaseline(
            concentrations=tuple(k for k, _ in table),
            mu_sph=tuple(v[0] for _, v in table),
            coordination=tuple(v[1] for _, v in table),
            source="fallback",
        )


@dataclass(frozen=True)
class InterlockingFriction:
    """``mu_eff = mu_sph(C) + kappa_i I(R) Z(C)/Z_ref`` (see the module docstring)."""

    kappa_i: float = 1.0
    z_ref: float = Z_REF

    def __post_init__(self) -> None:
        if not (math.isfinite(self.kappa_i) and self.kappa_i >= 0.0):
            raise ValueError("kappa_i must be finite and non-negative")
        if not (math.isfinite(self.z_ref) and self.z_ref > 0.0):
            raise ValueError("z_ref must be finite and positive")

    # -- pieces ------------------------------------------------------------
    def concentration_weight(self, coordination: float) -> float:
        return float(coordination) / self.z_ref

    def interlocking_term(self, interlocking_index: float, coordination: float) -> float:
        index = float(interlocking_index)
        if not (math.isfinite(index) and index >= -1e-12):
            raise ValueError("interlocking index must be finite and non-negative")
        return self.kappa_i * index * self.concentration_weight(coordination)

    def mu(self, interlocking_index: float, concentration: float,
           baseline: SphereBaseline) -> float:
        """Effective friction of a field of shapes with ``I(R)`` at ``C``."""
        mu_sph = baseline.mu(concentration)
        z = baseline.z(concentration)
        return mu_sph + self.interlocking_term(interlocking_index, z)

    def mu_polygon(self, poly: RoundedPolygon, concentration: float,
                   baseline: SphereBaseline) -> float:
        return self.mu(poly.interlocking_index, concentration, baseline)

    def mu_cube(self, cube: RoundedCube, concentration: float,
                baseline: SphereBaseline) -> float:
        return self.mu(cube.interlocking_index, concentration, baseline)

    def breakdown(self, interlocking_index: float, concentration: float,
                  baseline: SphereBaseline) -> dict:
        mu_sph = baseline.mu(concentration)
        z = baseline.z(concentration)
        term = self.interlocking_term(interlocking_index, z)
        return {
            "concentration": float(concentration),
            "interlocking_index": float(interlocking_index),
            "mu_sphere": mu_sph,
            "coordination": z,
            "interlocking_term": term,
            "mu_effective": mu_sph + term,
            "sphere_share": (mu_sph / (mu_sph + term)) if (mu_sph + term) > 0 else 0.0,
        }


def mu_effective_from_index(interlocking_index: float, concentration: float,
                            baseline: Optional[SphereBaseline] = None,
                            model: Optional[InterlockingFriction] = None) -> float:
    baseline = baseline or load_sphere_baseline()
    model = model or InterlockingFriction()
    return model.mu(interlocking_index, concentration, baseline)


def mu_effective_polygon(poly: RoundedPolygon, concentration: float,
                         baseline: Optional[SphereBaseline] = None,
                         model: Optional[InterlockingFriction] = None) -> float:
    baseline = baseline or load_sphere_baseline()
    model = model or InterlockingFriction()
    return model.mu_polygon(poly, concentration, baseline)


def mu_effective_cube(cube: RoundedCube, concentration: float,
                      baseline: Optional[SphereBaseline] = None,
                      model: Optional[InterlockingFriction] = None) -> float:
    baseline = baseline or load_sphere_baseline()
    model = model or InterlockingFriction()
    return model.mu_cube(cube, concentration, baseline)