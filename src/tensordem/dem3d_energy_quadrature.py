"""Closed-form left-endpoint quadrature defect of the platen actuator ledger.

The historical platen work arm evaluates the actuator power at the *start* of
the step, ``W_legacy = P(t)*dt = -R(t) * (a(t+dt) - a(t))`` with ``R`` the
penalty reaction.  For a penalty platen the stored energy is quadratic in the
boundary coordinate ``a``,

    U(a) = 1/2 * k * sum_i max(0, z_i + r - a)^2      (top platen shown),

so with particles held fixed across the step the exact discrete actuator work
is the potential increment ``dU = U(a1) - U(a0)``.  Writing ``dz = a1 - a0``
and ``n`` for the number of particles whose overlap is positive at ``a1``:

    dU = k * sum_i delta_i0 * dz + 1/2 * k * n * dz^2
    W_legacy = k * sum_i delta_i0 * dz                (spring part of -R*dz)

hence the *left-endpoint defect* is exactly

    W_exact - W_legacy = 1/2 * k * n * dz^2            (>= 0, platen works on
                                                        the specimen)

For ``dz = v*dt`` the per-step defect is second order in ``dt``, so the
accumulated defect over a run of ``N`` steps is first order in ``dt``.  That is
the systematic bias measured in ``docs/dem3d_energy_quadrature_closure.md``.

This module is the closed form used by three consumers:

* ``EnergyAuditedIceDEM3D.step`` accumulates
  ``cumulative_platen_deficit_predicted_J`` next to the two ledger arms, so a
  production run carries its own analytic cross-check;
* ``tensordem.dem3d_triaxial`` reports predicted vs measured defect in its
  ``energy_ledger`` block;
* ``scripts/verify_dem3d_energy_quadrature_closure.py`` benchmarks the
  prediction against the engine and writes a machine-readable run artifact.

Only tensor arithmetic on particle positions and scalar boundary coordinates
is used: no geometry, no solver state, no side effects.
"""
from __future__ import annotations

import math

import torch


def _count_positive(overlap: torch.Tensor) -> int:
    return int((overlap > 0).sum().item())


def platen_defect_components(
    positions: torch.Tensor,
    *,
    radius: float,
    stiffness: float,
    top_z0: float | None = None,
    top_z1: float | None = None,
    bottom_z0: float | None = None,
    bottom_z1: float | None = None,
    lateral_pos0: torch.Tensor | None = None,
    lateral_pos1: torch.Tensor | None = None,
    lateral_stiffness: float | None = None,
) -> dict[str, float]:
    """Per-boundary left-endpoint defect ``1/2 k n dz^2`` for one step.

    ``positions`` must be the *pre-update* particle positions (the exact arm
    evaluates the boundary potential at fixed particle positions).  Overlap is
    counted after the boundary advance (``*_z1``/``lateral_pos1``), because a
    particle that is exactly entering contact contributes its full
    ``1/2 k dz^2`` to the potential increment.

    Returns a dict with ``top``, ``bottom``, ``lateral`` and ``total`` entries
    in joules.
    """
    out = {"top": 0.0, "bottom": 0.0, "lateral": 0.0}
    if top_z0 is not None and top_z1 is not None:
        dz = float(top_z1) - float(top_z0)
        overlap = (positions[:, 2] + radius - float(top_z1)).clamp_min(0.0)
        out["top"] = 0.5 * stiffness * dz * dz * _count_positive(overlap)
    if bottom_z0 is not None and bottom_z1 is not None:
        dz = float(bottom_z1) - float(bottom_z0)
        overlap = (float(bottom_z1) + radius - positions[:, 2]).clamp_min(0.0)
        out["bottom"] = 0.5 * stiffness * dz * dz * _count_positive(overlap)
    if lateral_pos0 is not None and lateral_pos1 is not None:
        k = stiffness if lateral_stiffness is None else float(lateral_stiffness)
        px, py = positions[:, 0], positions[:, 1]
        total = 0.0
        for idx in range(4):
            dz = float(lateral_pos1[idx]) - float(lateral_pos0[idx])
            if idx == 0:
                overlap = (float(lateral_pos1[0]) - px + radius).clamp_min(0.0)
            elif idx == 1:
                overlap = (px + radius - float(lateral_pos1[1])).clamp_min(0.0)
            elif idx == 2:
                overlap = (float(lateral_pos1[2]) - py + radius).clamp_min(0.0)
            else:
                overlap = (py + radius - float(lateral_pos1[3])).clamp_min(0.0)
            total += 0.5 * k * dz * dz * _count_positive(overlap)
        out["lateral"] = total
    out["total"] = out["top"] + out["bottom"] + out["lateral"]
    return out


def quadrature_order(levels: list[tuple[float, float]]) -> float:
    """Least-squares slope of ``log(defect)`` against ``log(dt)``.

    ``levels`` is a list of ``(dt, defect)`` pairs.  A second-order per-step
    defect gives slope ``2.0``; a first-order (left endpoint) rule gives
    ``1.0``.  Pairs with non-positive arguments are ignored.
    """
    pts = [(math.log(dt), math.log(d)) for dt, d in levels if dt > 0.0 and d > 0.0]
    if len(pts) < 2:
        raise ValueError("quadrature_order needs at least two positive (dt, defect) pairs")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0.0:
        raise ValueError("quadrature_order needs distinct dt values")
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return sxy / sxx


def relative_error(measured: float, reference: float) -> float:
    """``|measured - reference| / |reference|`` with a zero-reference guard."""
    if reference == 0.0:
        return 0.0 if measured == 0.0 else math.inf
    return abs(measured - reference) / abs(reference)