"""Machine-checkable criteria (a)-(g) of the brash-channel campaign node.

Campaign node ``p307w1-n03-brash-ice-rheology`` states six physical criteria
plus a test-count criterion.  Each is implemented as a pure function returning
a verdict record ``{"value", "threshold"/"band", "verdict", "detail"}``; the
aggregate (:func:`evaluate_criteria`) is what the report and the artifact dump
carry.  Verdicts are never overwritten: a FAIL stays in the record.

(a) mu(I) constants: DEM steady-shear fit, and the literature confrontation.
(b) total resistance vs the reference anchor, B = 15 / 25 / 35 m, error < 25 %.
(c) component split: shear 40-60 %, buoyancy 20-30 %, keel drag 20-30 %.
(d) speed dependence R ~ v^n with n in [0.8, 1.2] over 1-5 m/s.
(e) porosity sensitivity: R(phi=0.4)/R(phi=0.2) - 1 in [-25 %, -15 %].
(f) negative controls: R(v = 0) = 0 exactly; R(1 mm/s) / R(3 m/s) < 1 %;
    R(phi -> 1) / R(0.3) < 1 %; R monotone increasing in v.
(g) pytest count (recorded from the actual run, not asserted here).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

__all__ = [
    "SPEED_EXPONENT_BAND",
    "SPEED_EXPONENT_RANGE_M_S",
    "POROSITY_BAND",
    "SHARE_BANDS",
    "B_TOLERANCE",
    "criterion_a",
    "criterion_b",
    "criterion_c",
    "criterion_d",
    "criterion_e",
    "criterion_f",
    "evaluate_criteria",
]

from .anchors import reference_anchor, scaling_diagnostic
from .coefficients import ledger_sha256
from .material import STANDARD_BRASH, BrashIceState
from .model import (Closure, STANDARD_BEAMS_M, brash_resistance,
                    porosity_sweep, power_law_exponent, speed_sweep,
                    standard_hull)

#: Criterion (b) relative-error tolerance.
B_TOLERANCE = 0.25
#: Criterion (c) share bands (fractions of the total).
SHARE_BANDS = {"shear": (0.40, 0.60), "buoyancy": (0.20, 0.30), "drag": (0.20, 0.30)}
#: Criterion (d): power-law range and band.
SPEED_EXPONENT_RANGE_M_S = (1.0, 5.0)
SPEED_EXPONENT_BAND = (0.8, 1.2)
#: Criterion (e): relative change of R between phi = 0.2 and phi = 0.4.
POROSITY_BAND = (-0.25, -0.15)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def criterion_a(mu_calibration: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """mu(I) constants: DEM fit quality + literature confrontation."""
    if mu_calibration is None:
        return {"criterion": "a", "verdict": "NO_DATA",
                "detail": "no DEM steady-shear calibration supplied "
                          "(run study.py or load the cached sweep)"}
    a1 = mu_calibration["a1_fit_quality"]
    a2 = mu_calibration["a2_literature"]
    ok = a1["verdict"] == "PASS" and a2["verdict"] == "PASS"
    return {"criterion": "a", "verdict": _verdict(ok),
            "fit_quality": a1, "literature": a2,
            "parameters": mu_calibration["parameters"],
            "literature_parameters": mu_calibration["literature"],
            "n_points": mu_calibration["n_points"]}


def criterion_b(state: BrashIceState = STANDARD_BRASH,
                speed: float = 3.0,
                beams: Sequence[float] = STANDARD_BEAMS_M,
                closure: Closure = None) -> Dict[str, object]:
    """Total resistance vs the reference anchor at three beams (< 25 %)."""
    closure = closure or Closure.from_ledger()
    rows: List[Dict[str, float]] = []
    for beam in beams:
        hull = standard_hull(beam)
        model = brash_resistance(state, hull, speed, closure)
        anchor = reference_anchor(state, hull, speed)
        rel = model.total_N / float(anchor["total_N"]) - 1.0
        rows.append({
            "beam_m": float(beam),
            "model_total_kN": model.total_kN,
            "anchor_total_kN": float(anchor["total_N"]) / 1.0e3,
            "relative_error": rel,
            "within_tolerance": abs(rel) < B_TOLERANCE,
        })
    diag = scaling_diagnostic(state, beams, speed, closure)
    ok = all(r["within_tolerance"] for r in rows)
    return {
        "criterion": "b", "verdict": _verdict(ok), "tolerance": B_TOLERANCE,
        "rows": rows, "max_abs_relative_error": max(abs(r["relative_error"]) for r in rows),
        "model_beam_exponent": diag["model_beam_exponent"],
        "anchor_beam_exponent": diag["anchor_beam_exponent"],
        "anchor_source": "tensordem.ice_resistance_scaling.broken_ice_anchor",
        "suyuthi_attribution": ("WITHHELD - Suyuthi et al. (2013) not readable "
                                "offline; in-repository brash anchor used"),
    }


def criterion_c(state: BrashIceState = STANDARD_BRASH, speed: float = 3.0,
                beams: Sequence[float] = STANDARD_BEAMS_M,
                closure: Closure = None) -> Dict[str, object]:
    """Component split must sit in the 40-60 / 20-30 / 20-30 % bands."""
    closure = closure or Closure.from_ledger()
    rows = []
    for beam in beams:
        shares = brash_resistance(state, standard_hull(beam), speed, closure).shares
        row = {"beam_m": float(beam), "shares": shares, "within_band": {}}
        for name, (lo, hi) in SHARE_BANDS.items():
            row["within_band"][name] = lo <= shares[name] <= hi
        row["pass"] = all(row["within_band"].values())
        rows.append(row)
    ok = all(r["pass"] for r in rows)
    return {"criterion": "c", "verdict": _verdict(ok), "bands": SHARE_BANDS,
            "rows": rows}


def criterion_d(state: BrashIceState = STANDARD_BRASH,
                beam: float = 25.0,
                speed_range: Sequence[float] = SPEED_EXPONENT_RANGE_M_S,
                closure: Closure = None) -> Dict[str, object]:
    """Speed exponent ``n`` of ``R ~ v^n`` over the transit range 1-5 m/s."""
    closure = closure or Closure.from_ledger()
    hull = standard_hull(beam)
    v_lo, v_hi = float(speed_range[0]), float(speed_range[1])
    speeds = [v_lo + (v_hi - v_lo) * i / 8.0 for i in range(9)]
    records = speed_sweep(state, hull, speeds, closure)
    totals = [r["total_N"] for r in records]
    n = power_law_exponent(speeds, totals)
    lo, hi = SPEED_EXPONENT_BAND
    return {"criterion": "d", "verdict": _verdict(lo <= n <= hi),
            "exponent": n, "band": SPEED_EXPONENT_BAND,
            "speed_range_m_s": [v_lo, v_hi],
            "speed_m_s": speeds, "total_kN": [t / 1.0e3 for t in totals]}


def criterion_e(state: BrashIceState = STANDARD_BRASH, beam: float = 25.0,
                speed: float = 3.0,
                closure: Closure = None) -> Dict[str, object]:
    """Porosity sensitivity: R(0.4) vs R(0.2) must drop by 15-25 %."""
    closure = closure or Closure.from_ledger()
    hull = standard_hull(beam)
    records = porosity_sweep(state, hull, speed, (0.2, 0.25, 0.3, 0.35, 0.4), closure)
    r_low = records[0]["total_N"]
    r_high = records[-1]["total_N"]
    change = r_high / r_low - 1.0
    lo, hi = POROSITY_BAND
    return {"criterion": "e", "verdict": _verdict(lo <= change <= hi),
            "relative_change": change, "band": POROSITY_BAND,
            "porosity": [r["state"]["porosity"] for r in records],
            "total_kN": [r["total_N"] / 1.0e3 for r in records],
            "components_kN": [{"shear": r["F_shear_N"] / 1.0e3,
                               "buoyancy": r["F_buoy_N"] / 1.0e3,
                               "drag": r["F_drag_N"] / 1.0e3} for r in records]}


def criterion_f(state: BrashIceState = STANDARD_BRASH, beam: float = 25.0,
                closure: Closure = None,
                zero_tolerance: float = 1.0e-3) -> Dict[str, object]:
    """Negative controls: ``R(0) = 0``, ``R(phi->1) -> 0``, monotone in v."""
    closure = closure or Closure.from_ledger()
    hull = standard_hull(beam)
    r_zero = brash_resistance(state, hull, 0.0, closure)
    r_ref = brash_resistance(state, hull, 3.0, closure)
    r_creep = brash_resistance(state, hull, 1.0e-3, closure)
    r_water = brash_resistance(state.with_porosity(0.999), hull, 3.0, closure)
    speeds = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    totals = [brash_resistance(state, hull, v, closure).total_N for v in speeds]
    monotone = all(b > a for a, b in zip(totals, totals[1:]))
    ok = (abs(r_zero.total_N) <= zero_tolerance * r_ref.total_N
          and r_creep.total_N / r_ref.total_N < 1.0e-2
          and r_water.total_N / r_ref.total_N < 1.0e-2
          and monotone)
    return {
        "criterion": "f", "verdict": _verdict(ok),
        "total_at_zero_speed_N": r_zero.total_N,
        "total_at_3m_s_kN": r_ref.total_kN,
        "creep_speed_m_s": 1.0e-3,
        "creep_speed_ratio": r_creep.total_N / r_ref.total_N,
        "pure_water_ratio": r_water.total_N / r_ref.total_N,
        "monotone_in_speed": monotone,
        "speed_m_s": speeds, "total_kN": [t / 1.0e3 for t in totals],
    }


def evaluate_criteria(mu_calibration: Optional[Dict[str, object]] = None,
                      state: BrashIceState = STANDARD_BRASH,
                      closure: Closure = None,
                      pytest_count: Optional[int] = None,
                      pytest_minimum: int = 18) -> Dict[str, object]:
    """Full (a)-(g) verdict table plus the aggregate verdict."""
    closure = closure or Closure.from_ledger()
    criteria = {
        "a": criterion_a(mu_calibration),
        "b": criterion_b(state, 3.0, STANDARD_BEAMS_M, closure),
        "c": criterion_c(state, 3.0, STANDARD_BEAMS_M, closure),
        "d": criterion_d(state, 25.0, SPEED_EXPONENT_RANGE_M_S, closure),
        "e": criterion_e(state, 25.0, 3.0, closure),
        "f": criterion_f(state, 25.0, closure),
    }
    g = {"criterion": "g", "verdict": "NO_DATA",
         "detail": "test count not supplied"}
    if pytest_count is not None:
        g = {"criterion": "g", "verdict": _verdict(pytest_count >= pytest_minimum),
             "pytest_count": int(pytest_count), "minimum": int(pytest_minimum)}
    criteria["g"] = g
    passed = [k for k, v in criteria.items() if v["verdict"] == "PASS"]
    failed = [k for k, v in criteria.items() if v["verdict"] == "FAIL"]
    unknown = [k for k, v in criteria.items()
               if v["verdict"] not in ("PASS", "FAIL")]
    return {
        "criteria": criteria,
        "passed": sorted(passed),
        "failed": sorted(failed),
        "not_evaluated": sorted(unknown),
        "n_passed": len(passed),
        "ledger_sha256": ledger_sha256(),
    }
