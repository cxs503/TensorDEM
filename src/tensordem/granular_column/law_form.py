"""Reference-law *form* sensitivity for ``CASE-COLUMN-COLLAPSE``.

Why this module exists
----------------------
The anchor ``CASE-COLUMN-COLLAPSE`` judges DEM runout against the empirical law
of Lube et al. (2004), which is **piecewise**::

    r_inf/r_i = 1 + 1.24 a          (a < 1.7)
    r_inf/r_i = 1 + 1.6  sqrt(a)    (a > 1.7)

That law is a *fit to experiments*, not a derived result, so three different
things are entangled whenever a DEM point is compared against it:

1. **The fit's own defect**: the two branches do not meet.  They cross at
   ``a* = (1.6/1.24)**2 = 1.6653``, while the card switches at ``a = 1.7``;
   at the card break the linear branch is ``+0.71 %`` above the square-root
   branch.  The slope jumps by ``~2.02x`` (``1.24`` vs ``0.614``).
2. **Discretisation error** of the DEM (grains per column radius, ``dt``).
3. **Model form**: whether a *single* smooth law would describe the DEM data
   as well as the two-piece law does.

Statements such as "the DEM error changes sign across the break, therefore both
branches are reproduced" mix all three.  This module quantifies (1) and (3) with
closed-form/least-squares arithmetic so that a data set (a list of
``aspect_measured`` / ``runout_ratio`` points) can arbitrate instead of prose.

Everything here is pure ``math`` (no numpy, no DEM), so it is unit-testable in
milliseconds.  Callers supply the points; nothing is hard-coded to a mission.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

#: ``r_inf/r_i = 1 + 1.24 a`` below the break (Lube et al. 2004).
LINEAR_PREFACTOR = 1.24
#: ``r_inf/r_i = 1 + 1.6 a^(1/2)`` above the break (Lube et al. 2004).
SQRT_PREFACTOR = 1.6
#: Break printed on the case card / in the paper's abstract as transcribed.
CARD_BREAK = 1.7

__all__ = [
    "LINEAR_PREFACTOR",
    "SQRT_PREFACTOR",
    "CARD_BREAK",
    "crossover_aspect",
    "law_linear",
    "law_sqrt",
    "law_piecewise",
    "law_single_power",
    "discontinuity",
    "fit_single_power",
    "error_table",
    "rms_by_form",
    "local_slope",
    "slope_ratio_across_break",
    "error_trend",
]


def _as_float(value: float, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def crossover_aspect(linear_prefactor: float = LINEAR_PREFACTOR,
                     sqrt_prefactor: float = SQRT_PREFACTOR) -> float:
    """Aspect ratio where the two branches of the 2004 law actually intersect.

    ``1 + 1.24 a = 1 + 1.6 a^(1/2)``  =>  ``a* = (1.6/1.24)**2``.
    """
    linear = _as_float(linear_prefactor, "linear_prefactor")
    root = _as_float(sqrt_prefactor, "sqrt_prefactor")
    if linear <= 0.0 or root <= 0.0:
        raise ValueError("prefactors must be positive")
    return (root / linear) ** 2


def law_linear(aspect: float, prefactor: float = LINEAR_PREFACTOR) -> float:
    """Linear (shallow) branch, valid for small ``a``; evaluable anywhere."""
    return 1.0 + _as_float(prefactor, "prefactor") * _as_float(aspect, "aspect")


def law_sqrt(aspect: float, prefactor: float = SQRT_PREFACTOR) -> float:
    """Square-root (inertial) branch, valid for large ``a``; evaluable anywhere."""
    a = _as_float(aspect, "aspect")
    if a < 0.0:
        raise ValueError(f"aspect must be non-negative, got {aspect!r}")
    return 1.0 + _as_float(prefactor, "prefactor") * math.sqrt(a)


def law_piecewise(aspect: float, break_at: float = CARD_BREAK) -> float:
    """The 2004 law exactly as the case card states it (discontinuous at the break)."""
    a = _as_float(aspect, "aspect")
    b = _as_float(break_at, "break_at")
    if b <= 0.0:
        raise ValueError(f"break_at must be positive, got {break_at!r}")
    return law_linear(a) if a < b else law_sqrt(a)


def law_single_power(aspect: float, prefactor: float, exponent: float,
                     offset: float = 1.0) -> float:
    """A *single* smooth alternative: ``offset + prefactor * a**exponent``."""
    a = _as_float(aspect, "aspect")
    c = _as_float(prefactor, "prefactor")
    n = _as_float(exponent, "exponent")
    o = _as_float(offset, "offset")
    if a < 0.0:
        raise ValueError(f"aspect must be non-negative, got {aspect!r}")
    if c <= 0.0:
        raise ValueError(f"prefactor must be positive, got {prefactor!r}")
    return o + c * a ** n


# ---------------------------------------------------------------------------
# defect of the reference law itself
# ---------------------------------------------------------------------------
def discontinuity(break_at: float = CARD_BREAK) -> dict:
    """Quantify the 2004 fit's own jump/slope break at ``break_at``.

    Returns a dict with the branch values at the break, their absolute and
    relative difference, the two branch slopes there, the slope ratio, the true
    crossover and how far the card's break sits from it.
    """
    b = _as_float(break_at, "break_at")
    if b <= 0.0:
        raise ValueError(f"break_at must be positive, got {break_at!r}")
    v_lin = law_linear(b)
    v_sqr = law_sqrt(b)
    slope_lin = LINEAR_PREFACTOR
    slope_sqr = 0.5 * SQRT_PREFACTOR / math.sqrt(b)
    a_star = crossover_aspect()
    return {
        "break_at": b,
        "crossover_aspect": a_star,
        "break_minus_crossover": b - a_star,
        "value_linear_branch": v_lin,
        "value_sqrt_branch": v_sqr,
        "absolute_jump": v_lin - v_sqr,
        "relative_jump": (v_lin - v_sqr) / v_sqr,
        "linear_slope_at_break": slope_lin,
        "sqrt_slope_at_break": slope_sqr,
        "slope_ratio_across_break": slope_lin / slope_sqr,
    }


# ---------------------------------------------------------------------------
# arbitrating forms against a measured data set
# ---------------------------------------------------------------------------
def fit_single_power(points: Sequence[Mapping[str, float]],
                     offset: float = 1.0,
                     aspect_key: str = "aspect_measured",
                     value_key: str = "runout_ratio") -> dict:
    """Least-squares fit of ``value = offset + c * a**n`` in log space.

    Only the ``value - offset > 0`` part is fittable; fewer than two usable
    points raises ``ValueError``.  Returns ``c``, ``n``, the RMS and the
    worst absolute relative residual of the fit *in the original units*.
    """
    xs: list[float] = []
    ys: list[float] = []
    for p in points:
        a = float(p[aspect_key])
        v = float(p[value_key])
        if not (math.isfinite(a) and math.isfinite(v)) or a <= 0.0:
            continue
        if v - offset <= 0.0:
            continue
        xs.append(math.log(a))
        ys.append(math.log(v - offset))
    if len(xs) < 2:
        raise ValueError("need at least two points with value > offset")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("aspect ratios are degenerate (all equal)")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    exponent = sxy / sxx
    prefactor = math.exp(my - exponent * mx)
    resid = []
    for p in points:
        a = float(p[aspect_key])
        v = float(p[value_key])
        if not (math.isfinite(a) and math.isfinite(v)) or a <= 0.0:
            continue
        if v - offset <= 0.0:
            continue
        pred = law_single_power(a, prefactor, exponent, offset)
        resid.append((pred - v) / v)
    rms = math.sqrt(sum(r * r for r in resid) / len(resid))
    return {
        "prefactor": prefactor,
        "exponent": exponent,
        "offset": offset,
        "n_points": n,
        "rms_relative": rms,
        "max_abs_relative": max(abs(r) for r in resid),
        "bias_relative": sum(resid) / len(resid),
    }


def error_table(points: Sequence[Mapping[str, float]],
                break_at: float = CARD_BREAK,
                aspect_key: str = "aspect_measured",
                value_key: str = "runout_ratio") -> list[dict]:
    """Per-point relative error of three reference forms against the data."""
    rows = []
    for p in points:
        a = float(p[aspect_key])
        v = float(p[value_key])
        if not (math.isfinite(a) and math.isfinite(v)) or v == 0.0:
            continue
        rows.append({
            "case_id": p.get("case_id", ""),
            "aspect_measured": a,
            "grains_per_radius": float(p.get("grains_per_radius", float("nan"))),
            "runout_ratio": v,
            "form_piecewise": law_piecewise(a, break_at),
            "form_linear_only": law_linear(a),
            "form_sqrt_only": law_sqrt(a),
            "err_piecewise": (v - law_piecewise(a, break_at)) / law_piecewise(a, break_at),
            "err_linear_only": (v - law_linear(a)) / law_linear(a),
            "err_sqrt_only": (v - law_sqrt(a)) / law_sqrt(a),
        })
    return rows


def rms_by_form(rows: Iterable[Mapping[str, float]]) -> dict:
    """Aggregate the ``error_table`` rows per candidate form."""
    keys = ("err_piecewise", "err_linear_only", "err_sqrt_only")
    rows = [r for r in rows if all(math.isfinite(float(r[k])) for k in keys)]
    if not rows:
        raise ValueError("no usable rows")
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        rms = math.sqrt(sum(v * v for v in vals) / len(vals))
        worst = max(vals, key=abs)
        worst_row = rows[vals.index(worst)]
        out[k[4:]] = {
            "n": len(vals),
            "rms_relative": rms,
            "bias_relative": sum(vals) / len(vals),
            "max_abs_relative": abs(worst),
            "worst_case": worst_row.get("case_id", ""),
            "n_within_0p03": sum(1 for v in vals if abs(v) < 0.03),
            "n_within_0p15": sum(1 for v in vals if abs(v) <= 0.15),
        }
    forms = [k[4:] for k in keys]
    best_rms = min(out[f]["rms_relative"] for f in forms)
    out["best_forms"] = [f for f in forms
                         if out[f]["rms_relative"] <= best_rms * (1 + 1e-9) + 1e-12]
    out["best_form"] = out["best_forms"][0]
    return out


def local_slope(a_lo: float, r_lo: float, a_hi: float, r_hi: float) -> float:
    """Two-point finite-difference slope; ``nan`` for a degenerate base."""
    a_lo = _as_float(a_lo, "a_lo")
    a_hi = _as_float(a_hi, "a_hi")
    if a_hi == a_lo:
        return float("nan")
    return (float(r_hi) - float(r_lo)) / (a_hi - a_lo)


def slope_ratio_across_break(points_lo: Sequence[Mapping[str, float]],
                             points_hi: Sequence[Mapping[str, float]],
                             aspect_key: str = "aspect_measured",
                             value_key: str = "runout_ratio") -> dict:
    """Compare the measured slope below/above the break against the law's ratio.

    ``points_lo`` / ``points_hi`` are the measured points on each side.  The
    slope of each side uses the extreme two points of the supplied side; with
    only two points that is the plain two-point difference.
    """
    def _extreme(pts):
        pts = sorted(pts, key=lambda p: float(p[aspect_key]))
        if len(pts) < 2:
            raise ValueError("need >= 2 points per side")
        return (float(pts[0][aspect_key]), float(pts[0][value_key]),
                float(pts[-1][aspect_key]), float(pts[-1][value_key]))

    lo = _extreme(points_lo)
    hi = _extreme(points_hi)
    s_lo = local_slope(*lo)
    s_hi = local_slope(*hi)
    law = discontinuity()
    ratio = s_lo / s_hi if (math.isfinite(s_hi) and s_hi != 0.0) else float("nan")
    return {
        "slope_below": s_lo,
        "slope_above": s_hi,
        "slope_ratio_measured": ratio,
        "law_slope_ratio": law["slope_ratio_across_break"],
        "a_below": [lo[0], lo[2]],
        "a_above": [hi[0], hi[2]],
    }


def error_trend(points: Sequence[Mapping[str, float]],
                aspect_key: str = "aspect_measured",
                err_key: str = "err_piecewise") -> dict:
    """Least-squares slope of the relative error vs aspect ratio.

    A systematic positive slope means the DEM runout grows faster in ``a`` than
    the reference law does; ``r2`` says how much of the spread that explains.
    """
    xs, ys = [], []
    for p in points:
        a = float(p[aspect_key])
        e = float(p[err_key])
        if math.isfinite(a) and math.isfinite(e):
            xs.append(a)
            ys.append(e)
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two usable points")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0.0 else float("nan")
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "n_points": n,
        "slope_per_aspect": slope,
        "intercept": intercept,
        "r2": r2,
        "span_aspect": [min(xs), max(xs)],
    }