"""Archard-type corner attrition on the analytic roundness families.

Physics
-------
Repeated sliding of a floe corner against ice or a hull removes material at a
rate given by **Archard's wear law**

.. math::  \\dot V = k \\, W \\, \\dot s / H

(``k`` dimensionless wear coefficient, ``W`` normal load, ``s`` sliding distance,
``H`` hardness).  What makes this a *shape* process rather than a mass-loss
process is that the material is removed where the curvature is highest, i.e. at
the corners, so the removed volume maps one-to-one onto the fillet radius ``r``
of :class:`~tensordem.floe_attrition.geometry.RoundedPolygon` (or the corner
radius of :class:`~tensordem.floe_attrition.geometry.RoundedCube`).  The wear
state variable is therefore ``r``, and roundness follows from geometry alone.

Load model
----------
The contact is **pressure controlled** (displacement driven: the ice/steel
indentation pressure is the independent field variable of a transit), so the
normal load carried by one corner patch is ``W = p A_patch`` with the patch taken
from the *current* geometry:

* 2D: ``A_patch = b(r) = 2 r cot(alpha/2) + r (pi - alpha)`` [N/m per unit depth],
* 3D: ``A_patch = pi r^2`` (projected corner cap).

Because the patch grows as the corner blunts, the two families give two distinct
attrition laws, and both are analytic:

======================  ==========================  =============================
quantity                RoundedPolygon (2D)         RoundedCube (3D)
======================  ==========================  =============================
worn volume ``V_w(r)``  ``c2 r^2`` (exact)           ``3a(4-pi) r^2 + O(r^3)``
patch ``A_p``           ``chi r``                   ``pi r^2``
rate ``dr/dt``          constant (linear blunting)  ``prop. r`` (exponential)
======================  ==========================  =============================

Consequences (pre-registered in ``prereg/p472w2_n05_floe_attrition_prereg.json``):

* the time to reach a given roundness scales as ``tau ∝ 1 / p`` (exponent -1) in
  *both* families, because every term of the rate is linear in ``p``;
* ``R(t)`` is monotone, saturating at ``R = 1``, and blind to ``k``/``H`` through
  the product ``k/H`` only (identifiability limit, reported).

Conservation and controls
-------------------------
``integrate`` inverts ``worn_volume`` **exactly** at every step, so the ledger
``sum(Archard increments) == geometric material loss`` closes to machine
precision; a fully rounded shape (``r = r_max``) absorbs no further wear volume
and the surplus is carried separately as ``excess`` (mass loss), which keeps the
ledger honest instead of silently clipping.  ``sliding = 0`` is an exact
fixed point (negative control).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import brentq

from .geometry import RoundedCube, RoundedPolygon

__all__ = [
    "ArchardWear",
    "AttritionHistory",
    "integrate_polygon_attrition",
    "integrate_cube_attrition",
    "wear_volume_closed_form",
]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


@dataclass(frozen=True)
class ArchardWear:
    """Archard wear law ``V = k W s / H`` (SI: N, m, Pa -> m^3 or m^2)."""

    wear_coefficient: float = 5.0e-3
    hardness_pa: float = 3.0e7

    def __post_init__(self) -> None:
        object.__setattr__(self, "wear_coefficient",
                           _positive("wear_coefficient", self.wear_coefficient))
        object.__setattr__(self, "hardness_pa",
                           _positive("hardness_pa", self.hardness_pa))

    @property
    def k_over_h(self) -> float:
        """The only identifiable combination of ``k`` and ``H`` (1/Pa)."""
        return self.wear_coefficient / self.hardness_pa

    def worn_volume(self, load, sliding):
        """``V = k W s / H``; ``load`` in N (3D) or N/m (2D, per unit depth).

        Scalar in, scalar out; arrays are accepted elementwise (the closed-form
        reference curves evaluate the law on a sliding-distance grid).
        """
        load_a = np.asarray(load, dtype=float)
        sliding_a = np.asarray(sliding, dtype=float)
        if not (np.isfinite(load_a).all() and (load_a >= 0.0).all()):
            raise ValueError("load must be finite and non-negative")
        if not (np.isfinite(sliding_a).all() and (sliding_a >= 0.0).all()):
            raise ValueError("sliding must be finite and non-negative")
        out = self.k_over_h * load_a * sliding_a
        return float(out) if out.ndim == 0 else out


@dataclass
class AttritionHistory:
    """Time series of one attrition run plus its conservation ledger."""

    time: List[float] = field(default_factory=list)
    sliding: List[float] = field(default_factory=list)
    state: List[float] = field(default_factory=list)          # r [m]
    roundness: List[float] = field(default_factory=list)
    interlocking: List[float] = field(default_factory=list)
    patch: List[float] = field(default_factory=list)
    pressure: List[float] = field(default_factory=list)
    worn_archard: List[float] = field(default_factory=list)
    worn_geometric: List[float] = field(default_factory=list)
    excess: List[float] = field(default_factory=list)
    state_max: float = 0.0
    family: str = ""
    worn_at_start: float = 0.0

    # -- accessors ---------------------------------------------------------
    def arrays(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(getattr(self, k), dtype=float) for k in
                ("time", "sliding", "state", "roundness", "interlocking",
                 "patch", "pressure", "worn_archard", "worn_geometric", "excess")}

    @property
    def conservation_error(self) -> float:
        """Relative gap between the Archard ledger and the geometric loss."""
        arch = self.worn_archard[-1] if self.worn_archard else 0.0
        if arch <= 0.0:
            return 0.0
        geom = self.worn_geometric[-1] + self.excess[-1]
        return abs(geom - arch) / arch

    @property
    def saturated(self) -> bool:
        return bool(self.excess) and self.excess[-1] > 0.0

    def summary(self) -> dict:
        return {
            "family": self.family,
            "state_max": self.state_max,
            "roundness_initial": self.roundness[0] if self.roundness else None,
            "roundness_final": self.roundness[-1] if self.roundness else None,
            "sliding_total": self.sliding[-1] if self.sliding else 0.0,
            "worn_archard": self.worn_archard[-1] if self.worn_archard else 0.0,
            "worn_geometric": self.worn_geometric[-1] if self.worn_geometric else 0.0,
            "excess": self.excess[-1] if self.excess else 0.0,
            "conservation_error": self.conservation_error,
            "saturated": self.saturated,
        }


def _solve_state(worn_at: Callable[[float], float],
                 target: float, lo: float, hi: float) -> Optional[float]:
    """Solve ``worn(r) = target`` on ``[lo, hi]`` (monotone increasing)."""
    if target <= worn_at(lo):
        return lo
    if target >= worn_at(hi):
        return None
    return brentq(lambda r: worn_at(r) - target, lo, hi, xtol=1e-16, rtol=1e-15,
                  maxiter=200)


def integrate_polygon_attrition(poly: RoundedPolygon, wear: ArchardWear, *,
                                pressure: float, slip_per_step: float,
                                n_steps: int, patch_mode: str = "geometric") -> AttritionHistory:
    """Integrate corner attrition of a rounded polygon under a contact pressure.

    ``pressure`` is the contact pressure [Pa] on the corner patch, ``slip_per_step``
    the sliding distance [m] per step (one "pass"), ``n_steps`` the number of
    passes.  ``patch_mode = "geometric"`` uses the current patch width (pressure
    controlled); ``"initial"`` freezes the patch at its initial value, which is the
    classical constant-load Archard reference used for the C2 comparison.
    """
    pressure = float(pressure)
    slip_per_step = float(slip_per_step)
    if not (math.isfinite(pressure) and pressure >= 0.0):
        raise ValueError("pressure must be finite and non-negative")
    if not (math.isfinite(slip_per_step) and slip_per_step >= 0.0):
        raise ValueError("slip_per_step must be finite and non-negative")
    if int(n_steps) != n_steps or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if patch_mode not in ("geometric", "initial"):
        raise ValueError("patch_mode must be 'geometric' or 'initial'")

    r_max = poly.max_fillet_radius
    worn_at = lambda r: poly.with_fillet(r).worn_area  # noqa: E731
    r = poly.fillet_radius
    if patch_mode == "geometric" and r <= 0.0:
        raise ValueError(
            "a pressure-controlled contact needs a finite initial corner radius: "
            "the patch width of a mathematically sharp corner is zero and the "
            "Archard load would vanish (use fillet_radius > 0)")
    patch0 = poly.contact_patch_width
    worn0 = worn_at(r)
    history = AttritionHistory(family="rounded_polygon", state_max=r_max,
                               worn_at_start=worn0)
    _record(history, poly, 0.0, 0.0, r, pressure, 0.0)

    worn_total = 0.0
    excess = 0.0
    for step in range(1, int(n_steps) + 1):
        shape_now = poly.with_fillet(r)
        patch = shape_now.contact_patch_width if patch_mode == "geometric" else patch0
        increment = wear.worn_volume(pressure * patch, slip_per_step)
        worn_total += increment
        r_new = _solve_state(worn_at, worn0 + worn_total, r, r_max)
        if r_new is None:                      # fully rounded: surplus is mass loss
            r_new = r_max
            excess = worn_total - (worn_at(r_max) - worn0)
        r = r_new
        _record(history, poly, float(step), step * slip_per_step, r, pressure,
                worn_total, excess)
    history.worn_geometric = [max(0.0, w - worn0) for w in history.worn_geometric]
    history.excess[-1] = excess
    return history


def _record(history: AttritionHistory, poly: RoundedPolygon, t: float, sliding: float,
            r: float, pressure: float, worn: float, excess: float = 0.0) -> None:
    shape = poly.with_fillet(r)
    history.time.append(float(t))
    history.sliding.append(float(sliding))
    history.state.append(float(r))
    history.roundness.append(float(shape.wadell_roundness))
    history.interlocking.append(float(shape.interlocking_index))
    history.patch.append(float(shape.contact_patch_width))
    history.pressure.append(float(pressure))
    history.worn_archard.append(float(worn))
    history.worn_geometric.append(float(shape.worn_area))
    history.excess.append(float(excess))


def integrate_cube_attrition(cube: RoundedCube, wear: ArchardWear, *,
                             pressure: float, slip_per_step: float,
                             n_steps: int) -> AttritionHistory:
    """3D counterpart: corner-radius growth of a rounded cube ("rounded floe slab")."""
    pressure = float(pressure)
    slip_per_step = float(slip_per_step)
    if not (math.isfinite(pressure) and pressure >= 0.0):
        raise ValueError("pressure must be finite and non-negative")
    if not (math.isfinite(slip_per_step) and slip_per_step >= 0.0):
        raise ValueError("slip_per_step must be finite and non-negative")
    r_max = cube.max_radius
    if r_max <= 0.0:
        raise ValueError("cube must have a positive maximum radius")
    worn_at = lambda r: cube.with_radius(r).worn_volume  # noqa: E731
    r = max(cube.radius, 0.0)
    if r <= 0.0:
        raise ValueError(
            "a pressure-controlled contact needs a finite initial corner radius: "
            "the corner cap area of a sharp cube corner is zero")
    worn0 = worn_at(r)
    history = AttritionHistory(family="rounded_cube", state_max=r_max,
                               worn_at_start=worn0)
    _record_cube(history, cube, 0.0, 0.0, r, pressure, 0.0, 0.0)
    worn_total = 0.0
    excess = 0.0
    for step in range(1, int(n_steps) + 1):
        patch = cube.with_radius(r).contact_patch_area
        increment = wear.worn_volume(pressure * patch, slip_per_step)
        worn_total += increment
        r_new = _solve_state(worn_at, worn0 + worn_total, r, r_max)
        if r_new is None:
            r_new = r_max
            excess = worn_total - (worn_at(r_max) - worn0)
        r = r_new
        _record_cube(history, cube, float(step), step * slip_per_step, r, pressure,
                     worn_total, excess)
    history.worn_geometric = [max(0.0, w - worn0) for w in history.worn_geometric]
    return history


def _record_cube(history: AttritionHistory, cube: RoundedCube, t: float, sliding: float,
                 r: float, pressure: float, worn: float, excess: float) -> None:
    shape = cube.with_radius(r)
    history.time.append(float(t))
    history.sliding.append(float(sliding))
    history.state.append(float(r))
    history.roundness.append(float(shape.wadell_roundness))
    history.interlocking.append(float(shape.interlocking_index))
    history.patch.append(float(shape.contact_patch_area))
    history.pressure.append(float(pressure))
    history.worn_archard.append(float(worn))
    history.worn_geometric.append(float(shape.worn_volume))
    history.excess.append(float(excess))


def wear_volume_closed_form(poly: RoundedPolygon, state: float) -> float:
    """Exact worn area of the 2D family: ``n [cot(alpha/2) - (pi-alpha)/2] r^2``."""
    r = float(state)
    if r < 0.0:
        raise ValueError("state must be non-negative")
    c = 1.0 / math.tan(poly.corner_half_angle) - 0.5 * poly.arc_turn
    return poly.n_sides * c * r ** 2