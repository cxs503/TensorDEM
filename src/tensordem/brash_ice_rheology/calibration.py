"""Calibration of the single free closure constant.

The model has exactly one constant that is not fixed by independent reasoning:
``C_c``, the share of the brash momentum flux ``0.5 rho_b v^2`` that is trapped
as a *sustained* confinement pressure in the hull shear band.  Physically it is
an order-one number; quantitatively it is fixed here, with a declared protocol:

``calibrate_confinement``
    bisection for the ``C_c`` that reproduces one target total (single-point
    anchoring at a designed calibration case).

``calibrate_family``
    minimax over the standard beam series: the ``C_c`` that equalises the
    modulus of the relative residual at the smallest and largest beam.  This is
    the protocol used for the ledger value, because the residual of criterion
    (b) is a *slope* mismatch between model (all components scale with beam) and
    anchor (crushing term beam independent) - a single-point anchoring would
    therefore hide the mismatch in one beam instead of spreading it.

Nothing else is fitted: the mu(I) constants are the published Jop values, the
shear-band thickness, displaced volume and drag coefficient carry DERIVED /
CONVENTION grades, and the speed exponent, the porosity sensitivity and the
three-component split are *predictions* of the calibrated model.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

__all__ = ["calibrate_confinement", "calibrate_family", "CALIBRATION_BRACKET"]

from .anchors import reference_anchor
from .coefficients import check_coefficient
from .material import STANDARD_BRASH, BrashIceState
from .model import (BrashHull, Closure, STANDARD_BEAMS_M, brash_resistance,
                    standard_hull)

#: Search bracket for ``C_c`` (order-one expected).
CALIBRATION_BRACKET = (0.0, 3.0)


def _total(state: BrashIceState, hull: BrashHull, speed: float, cc: float) -> float:
    return brash_resistance(state, hull, speed,
                            Closure.from_ledger().with_confinement(cc)).total_N


def calibrate_confinement(state: BrashIceState = STANDARD_BRASH,
                          hull: BrashHull = None,
                          speed: float = 3.0,
                          target_N: float = None,
                          bracket: Sequence[float] = CALIBRATION_BRACKET,
                          iterations: int = 120) -> Dict[str, object]:
    """Bisection for the ``C_c`` that reproduces ``target_N`` (monotone in C_c)."""
    hull = hull or standard_hull(25.0)
    if target_N is None:
        target_N = float(reference_anchor(state, hull, speed)["total_N"])
    lo, hi = float(bracket[0]), float(bracket[1])
    f_lo = _total(state, hull, speed, lo) - target_N
    f_hi = _total(state, hull, speed, hi) - target_N
    if f_lo * f_hi > 0.0:
        raise ValueError("target not bracketed: widen the C_c bracket")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _total(state, hull, speed, mid) - target_N > 0.0:
            hi = mid
        else:
            lo = mid
    cc = 0.5 * (lo + hi)
    total = _total(state, hull, speed, cc)
    return {"confinement_coefficient": cc, "target_N": target_N,
            "achieved_N": total, "relative_error": total / target_N - 1.0}


def calibrate_family(state: BrashIceState = STANDARD_BRASH,
                     beams: Sequence[float] = STANDARD_BEAMS_M,
                     speed: float = 3.0,
                     bracket: Sequence[float] = CALIBRATION_BRACKET,
                     iterations: int = 200) -> Dict[str, object]:
    """Minimax calibration of ``C_c`` over the standard beam series.

    Returns the calibrated constant, the per-beam residuals and the agreement
    with the ledger value (a reproducibility gate: the ledger constant *is* the
    output of this function).
    """
    beams = tuple(float(b) for b in beams)
    if len(beams) < 2:
        raise ValueError("family calibration needs at least two beams")
    hulls = [standard_hull(b) for b in beams]
    targets = [float(reference_anchor(state, h, speed)["total_N"]) for h in hulls]

    def residual(cc: float) -> list:
        return [_total(state, h, speed, cc) / t - 1.0 for h, t in zip(hulls, targets)]

    lo, hi = float(bracket[0]), float(bracket[1])
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        errs = residual(mid)
        # residual at the small beam is negative, at the large beam positive;
        # bisect so that |err_min| == |err_max| (minimax).
        if abs(errs[0]) > abs(errs[-1]):
            lo = mid
        else:
            hi = mid
    cc = 0.5 * (lo + hi)
    errs = residual(cc)
    ledger_value = check_coefficient("confinement_coefficient")[0]
    return {
        "confinement_coefficient": cc,
        "ledger_value": ledger_value,
        "matches_ledger": abs(cc - ledger_value) < 1.0e-5,
        "beams_m": list(beams),
        "targets_N": targets,
        "achieved_N": [_total(state, h, speed, cc) for h in hulls],
        "relative_errors": errs,
        "max_abs_relative_error": max(abs(e) for e in errs),
        "protocol": ("minimax (equal-modulus residual) over the standard beam "
                     "series at H=1 m, phi=0.3, v=3 m/s"),
    }
