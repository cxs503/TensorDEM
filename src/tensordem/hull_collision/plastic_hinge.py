"""Rigid-plastic hinge mechanics for ship side-shell collision energy absorption.

Why this module exists
----------------------
The ice-region collision verification surface (campaign node IN63-N01) needs the
*structural* half of the energy balance to be an explicit, independently
checkable quantity: the energy a ship side shell dissipates while it is dented by
an ice indenter.  This module states that model in closed form (rigid-plastic
hinge theory) together with a second, numerically integrated route for the same
quantity, so that the closed form can be falsified instead of trusted.

Model
-----
A side-shell panel of span ``span`` (transverse frame spacing) is loaded at
mid-span and collapses in a mechanism of plastic hinges.  For a
rigid-perfectly-plastic material the dissipation of one hinge line is

    E = M_p * theta

with ``M_p = sigma_y * Z`` the fully plastic moment and ``theta`` the total
*hinge rotation* (sum of the absolute rotations of all hinge lines).  Two
kinematic cases are provided:

* ``clamped``           three hinges, ``theta = 8*d/span``, ``P_u = 8*M_p/span``
* ``simply_supported``  one hinge,   ``theta = 4*d/span``, ``P_u = 4*M_p/span``

``theta`` and ``P_u`` are two *independent* derivations of the same mechanism
(kinematics vs virtual work), which is what the anchor check exploits.

Scope and limitations (do not over-claim)
-----------------------------------------
* Elastic-perfectly-plastic, no hardening, no fracture energy, no strain-rate
  sensitivity (ice collisions are strain-rate sensitive; the yield stress used
  here is the static value, so absorbed energy is *under*-estimated).
* The hinge is rigid-plastic: the elastic part of the hinge rotation is not
  subtracted, a bias of order ``M_p*theta_y/2`` (< 1e-3 for the design cases).
* Small rotations / small deflections relative to the span; the mechanism
  kinematics ``theta = 8*d/span`` is a small-angle relation
  (valid for ``d/span <= 0.3``, i.e. 7 % over-estimate of the rotation at the
  kinematic limit; see ``docs/HULL_COLLISION_ENERGY_SURFACE.md``).
* Membrane stretching of plating (the dominant sink for ``d/span > 0.1``) is
  *not* here; it lives in :mod:`tensordem.hull_collision.resistance`.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "plastic_section_modulus_rect",
    "plastic_moment",
    "section_plastic_modulus",
    "plastic_moment_rect_fiber",
    "hinge_rotations",
    "collapse_load",
    "hinge_energy",
    "three_point_bending_anchor",
    "SUPPORTED_MECHANISMS",
]

SUPPORTED_MECHANISMS = ("clamped", "simply_supported")


def _real(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def plastic_section_modulus_rect(width: float, thickness: float) -> float:
    """Plastic section modulus of a solid rectangle in bending: ``Z = b*t^2/4``.

    ``width`` is measured parallel to the bending axis, ``thickness`` is the
    depth of the section in the bending plane.  Units: m and m^3.
    """
    b = _real("width", width, positive=True)
    t = _real("thickness", thickness, positive=True)
    return b * t * t / 4.0


def plastic_moment(sigma_y: float, plastic_modulus: float) -> float:
    """Fully plastic moment ``M_p = sigma_y * Z`` [N*m]."""
    sy = _real("sigma_y", sigma_y, positive=True)
    z = _real("plastic_modulus", plastic_modulus, positive=True)
    return sy * z


def section_plastic_modulus(rectangles: Iterable[Sequence[float]]) -> float:
    """Plastic section modulus of a stack of rectangles.

    ``rectangles`` is an iterable of ``(y_lower, y_upper, width)`` triples in
    metres; ``y`` is measured from an arbitrary datum and the bending axis is
    horizontal.  The plastic neutral axis is found by bisection on the
    first moment of area (so the section may be non-symmetric, e.g. a stiffener
    with attached plating), then ``Z = sum_i w_i * int |y - y_pna| dy``.

    This is the *numerical* counterpart of :func:`plastic_section_modulus_rect`
    and is used to check the closed form with an independent discretisation.
    """
    rects = []
    for item in rectangles:
        try:
            y0, y1, width = (float(item[0]), float(item[1]), float(item[2]))
        except (TypeError, IndexError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("rectangles must be (y_lower, y_upper, width) triples") from exc
        if not all(math.isfinite(v) for v in (y0, y1, width)):
            raise ValueError("rectangle coordinates must be finite")
        if y1 <= y0 or width <= 0.0:
            raise ValueError("rectangles need y_upper > y_lower and width > 0")
        rects.append((y0, y1, width))
    if not rects:
        raise ValueError("at least one rectangle is required")

    total_area = sum((y1 - y0) * w for y0, y1, w in rects)
    lo = min(r[0] for r in rects)
    hi = max(r[1] for r in rects)

    def area_below(y: float) -> float:
        acc = 0.0
        for y0, y1, w in rects:
            if y <= y0:
                continue
            acc += (min(y, y1) - y0) * w
        return acc

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_below(mid) < 0.5 * total_area:
            lo = mid
        else:
            hi = mid
    y_pna = 0.5 * (lo + hi)

    modulus = 0.0
    for y0, y1, w in rects:
        # integral of |y - y_pna| over [y0, y1]
        def _F(y: float) -> float:
            d = y - y_pna
            return 0.5 * d * abs(d)
        modulus += w * (_F(y1) - _F(y0))
    return modulus


def plastic_moment_rect_fiber(
    sigma_y: float, width: float, thickness: float, *, n_fibers: int = 20001
) -> float:
    """Fully plastic moment of a rectangle from a fibre quadrature.

    ``M = int_A |sigma| * |y| dA = sigma_y * int_A |y| dA`` evaluated by midpoint
    rule on ``n_fibers`` horizontal strips (the two halves of a symmetric section
    carry yield stress of opposite sign but both contribute to the *plastic*
    moment with the same sign).  Independent of ``Z = b*t^2/4``.
    """
    sy = _real("sigma_y", sigma_y, positive=True)
    b = _real("width", width, positive=True)
    t = _real("thickness", thickness, positive=True)
    if isinstance(n_fibers, bool) or not isinstance(n_fibers, int) or n_fibers < 3:
        raise ValueError("n_fibers must be an integer >= 3")
    dy = t / n_fibers
    moment = 0.0
    for i in range(n_fibers):
        y = -0.5 * t + (i + 0.5) * dy
        moment += sy * abs(y) * b * dy
    return moment


def _check_mechanism(bc: str) -> str:
    if bc not in SUPPORTED_MECHANISMS:
        raise ValueError(f"unknown mechanism {bc!r}; expected one of {SUPPORTED_MECHANISMS}")
    return bc


def hinge_rotations(deflection: float, span: float, *, bc: str = "clamped") -> float:
    """Total (sum of absolute) plastic hinge rotation of the mechanism [rad].

    ``clamped``: hinges at both ends plus mid-span; each rigid arm rotates by
    ``2*d/span``, the end hinges see that rotation and the mid-span hinge sees
    twice it, hence ``theta = 2*(2d/span) + 2*(2d/span) = 8d/span``.
    ``simply_supported``: only the mid-span hinge, ``theta = 2*(2d/span) = 4d/span``.
    """
    _check_mechanism(bc)
    d = _real("deflection", deflection, nonnegative=True)
    s = _real("span", span, positive=True)
    ratio = d / s
    if ratio > 0.5:
        raise ValueError("small-rotation mechanism kinematics requires deflection <= 0.5*span")
    return (8.0 if bc == "clamped" else 4.0) * ratio


def collapse_load(plastic_moment_value: float, span: float, *, bc: str = "clamped") -> float:
    """Collapse load of the mechanism [N] (virtual-work / upper-bound route).

    Energy balance ``P_u * d = M_p * theta(d)`` gives ``P_u = 8*M_p/span`` for the
    clamped mechanism and ``4*M_p/span`` for the simply supported one - the
    classical plastic-collapse loads.
    """
    _check_mechanism(bc)
    mp = _real("plastic_moment", plastic_moment_value, positive=True)
    s = _real("span", span, positive=True)
    return (8.0 if bc == "clamped" else 4.0) * mp / s


def hinge_energy(plastic_moment_value: float, deflection: float, span: float, *, bc: str = "clamped") -> float:
    """``E = M_p * theta_max`` [J] for one hinge mechanism."""
    mp = _real("plastic_moment", plastic_moment_value, positive=True)
    return mp * hinge_rotations(deflection, span, bc=bc)


def three_point_bending_anchor(
    *,
    sigma_y: float,
    width: float,
    thickness: float,
    span: float,
    deflection: float,
    bc: str = "simply_supported",
    n_steps: int = 4096,
) -> dict:
    """Closed-form vs numerically integrated plastic hinge energy of a beam.

    Three independent routes to the same dissipation are compared:

    ``closed``   ``E = M_p^closed * theta_max`` with ``M_p^closed = sigma_y*b*t^2/4``
                 and ``theta_max = 4*d/span`` (simply supported) or ``8*d/span``
                 (clamped) from the kinematic mechanism.
    ``incremental``  rigid-plastic work accumulated along a monotone loading path,
                 ``E = sum_k M_p^fiber * (theta(d_k) - theta(d_{k-1}))``, with
                 ``M_p^fiber`` from a fibre quadrature over the section (i.e. no
                 ``b*t^2/4`` anywhere).
    ``virtual_work`` external work ``P_u * d`` with ``P_u`` from
                 :func:`collapse_load` (moment equilibrium of the mechanism).

    Because ``theta(d)`` is linear, routes ``closed`` and ``incremental`` differ
    only through the section integration, and ``virtual_work`` additionally
    tests the kinematics against equilibrium.  The reported relative errors are
    therefore a check of the section integral (<0.01 %) and of the mechanism
    kinematics (<1e-12), not of the rigid-plastic idealisation itself.
    """
    sy = _real("sigma_y", sigma_y, positive=True)
    b = _real("width", width, positive=True)
    t = _real("thickness", thickness, positive=True)
    s = _real("span", span, positive=True)
    d = _real("deflection", deflection, nonnegative=True)
    _check_mechanism(bc)
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("n_steps must be an integer >= 2")

    z_closed = plastic_section_modulus_rect(b, t)
    mp_closed = plastic_moment(sy, z_closed)
    mp_fiber = plastic_moment_rect_fiber(sy, b, t)
    z_fiber = mp_fiber / sy

    theta_max = hinge_rotations(d, s, bc=bc)
    energy_closed = mp_closed * theta_max

    energy_incremental = 0.0
    previous = 0.0
    for i in range(1, n_steps + 1):
        current = d * i / n_steps
        energy_incremental += mp_fiber * (hinge_rotations(current, s, bc=bc) - hinge_rotations(previous, s, bc=bc))
        previous = current

    p_u = collapse_load(mp_fiber, s, bc=bc)
    energy_virtual_work = p_u * d

    def _rel(a: float, b_: float) -> float:
        if b_ == 0.0:
            return 0.0 if a == 0.0 else math.inf
        return abs(a - b_) / b_

    return {
        "mechanism": bc,
        "sigma_y_MPa": sy / 1e6,
        "section_width_m": b,
        "section_thickness_m": t,
        "span_m": s,
        "deflection_m": d,
        "plastic_modulus_closed_m3": z_closed,
        "plastic_modulus_fiber_m3": z_fiber,
        "plastic_moment_closed_Nm": mp_closed,
        "plastic_moment_fiber_Nm": mp_fiber,
        "theta_max_rad": theta_max,
        "collapse_load_N": p_u,
        "energy_closed_J": energy_closed,
        "energy_incremental_J": energy_incremental,
        "energy_virtual_work_J": energy_virtual_work,
        "rel_error_section": _rel(z_fiber, z_closed),
        "rel_error_incremental_vs_closed": _rel(energy_incremental, energy_closed),
        "rel_error_virtual_work_vs_closed": _rel(energy_virtual_work, energy_closed),
        "max_rel_error": max(
            _rel(z_fiber, z_closed),
            _rel(energy_incremental, energy_closed),
            _rel(energy_virtual_work, energy_closed),
        ),
    }
