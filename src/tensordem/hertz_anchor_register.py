"""W26 anchor register: Hertz impact duration / peak force, operator-layer checks.

Mission ``paper-exp-dem-hertz-impact-duration-1`` asks for a *registered*
literature anchor (Johnson 1985, Hertzian impact: contact duration ``t_c`` and
peak overlap / peak force) plus machine checks that do **not** depend on the
primary text being present in the worktree.

The primary source is not in this environment (no copy of Johnson, *Contact
Mechanics*, CUP 1985 anywhere on the filesystem), so the anchor itself is
registered ``WITHHELD`` -- never transcribed from memory.  What can be machine
checked without the book is the *operator layer* of the closed form, and that is
what this module computes, from first principles, with no hand-typed constants:

1. **Dimension closure by unit-rescaling covariance.**  Each closed form is a
   monomial in the physical parameters.  Evaluating it once in SI and once in a
   rescaled unit system (different mass/length/time units) must return the same
   physical answer after converting back.  The relative residual is machine
   epsilon iff the exponents are dimensionally consistent; a deliberately wrong
   exponent (``v**(-1/4)`` instead of ``v**(-1/5)``) makes the residual O(1).
   This is a *sensitivity-checked* closure test, not a tautology.

2. **Algebraic identity between the two textbook forms.**  The stiffness form
   ``t_c = C (5 m / (4 k_h))^(2/5) v^(-1/5)`` and the reduced-parameter form
   ``t_c = c (m^2 / (R E*^2))^(1/5) v^(-1/5)`` are the same function iff
   ``c = C (15/16)^(2/5)``; numerically ``c = 2.86805``, i.e. the ``2.87`` of the
   mission brief to three significant figures.  The residual is reported, and
   ``2.87`` is accepted only as a *rounded* statement with an explicit tolerance.

3. **Limits, including the ``v -> 0`` divergence form.**  ``t_c ~ v^(-1/5)``
   diverges, ``delta_max ~ v^(4/5)`` and ``F_max ~ v^(6/5)`` vanish.  The
   exponents are recovered by log-log regression against the analytic values and
   the *invariant* ``t_c v^(1/5)`` is checked to be speed independent.  The
   divergence is an asymptote of an ideal elastic model: the module also returns
   the speed at which ``t_c`` reaches a given physical time scale, which is what
   makes the divergence physically meaningful rather than a curiosity.

4. **Stiffness-ratio and damping applicability domain.**  A linear unilateral
   spring that matches the Hertz *duration* has
   ``k_eq / k_secant(delta_max) = (4/5) pi^2 / C^2 = 0.911678...``, a universal
   constant (verified below to 1e-12 over a condition sweep); consequently a
   linear spring pinned at the peak *secant* stiffness is ~4.5 % short in
   duration -- already outside the 3 % band -- and the damping-free closed form
   is only valid while the dimensionless Hunt-Crossley damping number
   ``alpha v0`` stays small.  Both domains are quantified by integrating the
   damped impact.

Every number in the artifact is produced by this module (or by the shipped
solver through :mod:`tensordem.hertz_impact_time`); nothing is transcribed.
All quantities SI.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable, Sequence

__all__ = [
    "PRIMARY_ANCHOR",
    "SIGNIFICANT_FIGURE_FORM_COEFF",
    "contact_time_integral",
    "prefactor",
    "hertz_stiffness",
    "peak_overlap",
    "peak_force",
    "contact_time",
    "reduced_form_coefficient",
    "contact_time_reduced",
    "quadrature_integral",
    "dimension_closure_table",
    "limit_table",
    "algebraic_identity_table",
    "stiffness_ratio_rows",
    "hunt_crossley_impact",
    "damping_domain_rows",
    "describe_anchor",
]

# ---------------------------------------------------------------------------
# The anchor record itself (honest status: the primary document is absent)
# ---------------------------------------------------------------------------

#: ``2.87`` of the mission brief, kept only to state the rounding tolerance.
SIGNIFICANT_FIGURE_FORM_COEFF = 2.87

PRIMARY_ANCHOR: dict[str, Any] = {
    "id": "JOHNSON-1985-CONTACT-MECHANICS-HERTZ-IMPACT",
    "citation": {
        "author": "K. L. Johnson",
        "title": "Contact Mechanics",
        "publisher": "Cambridge University Press",
        "year": 1985,
        "claim_scope": "elastic (Hertzian) impact: contact duration t_c and "
                       "peak overlap / peak force in terms of m_eff, R_eff, E*",
    },
    "status": "WITHHELD",
    "verbatim": None,
    "document_sha256": None,
    "quote_position": None,
    "withheld_reason": (
        "The primary document is not present in this environment: a whole "
        "filesystem search for the PDF/text turned up no copy, and the mission "
        "forbids transcribing numbers from memory or from secondary sources. "
        "The anchor is therefore registered WITHHELD, and only the operator "
        "layer (dimensional closure, algebraic identity, limits, applicability "
        "domain) is machine checked. Secondary in-tree citations of the same "
        "result are registered separately with their sha256 and line positions; "
        "they are code/docstrings, not the anchor."
    ),
    "secondary_in_tree_citations": [
        # filled by the generator with path/sha256/line/quote (code, not anchor)
    ],
}

# ---------------------------------------------------------------------------
# Closed forms, re-derived from their defining expressions
# ---------------------------------------------------------------------------

def contact_time_integral() -> float:
    """``I = int_0^1 du / sqrt(1 - u^(5/2))``.

    Substituting ``s = u^(5/2)`` turns it into ``(2/5) B(2/5, 1/2)``; evaluated
    through ``lgamma`` so the constant is computed, not typed.
    """
    return 0.4 * math.exp(math.lgamma(0.4) + math.lgamma(0.5) - math.lgamma(0.9))


def prefactor() -> float:
    """``C = 2 I`` -- the dimensionless prefactor of the Hertz contact time."""
    return 2.0 * contact_time_integral()


def quadrature_integral() -> float:
    """Independent quadrature of ``I`` (cross-check of the Beta form).

    The integrand ``(1 - u^(5/2))^(-1/2)`` has an integrable endpoint
    singularity at ``u = 1``, so the integral is evaluated in the singularity-free
    variable ``u = 1 - y^2``::

        I = int_0^1 2 y dy / sqrt(1 - (1 - y^2)^(5/2))

    which is smooth on the closed interval (the numerator removes the square-root
    zero).  Simpson with ``2e5`` intervals; independent of ``lgamma``.
    """
    n = 200_000
    h = 1.0 / n
    total = 0.0
    for i in range(n + 1):
        y = i * h
        if y == 0.0:
            # limit of the integrand at y -> 0: 2 y / sqrt(2.5 y^2) = 2/sqrt(2.5)
            f = 2.0 / math.sqrt(2.5)
        else:
            f = 2.0 * y / math.sqrt(1.0 - (1.0 - y * y) ** 2.5)
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += weight * f
    return total * h / 3.0


def hertz_stiffness(E_star: float, R_star: float) -> float:
    """``k_h = (4/3) E* sqrt(R*)``  [N m^(-3/2)], so that ``F = k_h delta^(3/2)``."""
    if not (E_star > 0.0 and R_star > 0.0):
        raise ValueError("E_star and R_star must be positive")
    return (4.0 / 3.0) * E_star * math.sqrt(R_star)


def peak_overlap(v0: float, m_eff: float, k_h: float) -> float:
    """``delta_max = (5 m v0^2 / (4 k_h))^(2/5)`` from energy conservation."""
    if not (v0 > 0.0 and m_eff > 0.0 and k_h > 0.0):
        raise ValueError("v0, m_eff, k_h must be positive")
    return (5.0 * m_eff * v0 * v0 / (4.0 * k_h)) ** 0.4


def peak_force(v0: float, m_eff: float, k_h: float) -> float:
    """``F_max = k_h delta_max^(3/2)`` (Hertz law at peak overlap)."""
    return k_h * peak_overlap(v0, m_eff, k_h) ** 1.5


def contact_time(v0: float, m_eff: float, k_h: float) -> float:
    """``t_c = C (5 m / (4 k_h))^(2/5) v0^(-1/5)`` (undamped elastic impact)."""
    if not (v0 > 0.0 and m_eff > 0.0 and k_h > 0.0):
        raise ValueError("v0, m_eff, k_h must be positive")
    return prefactor() * (5.0 * m_eff / (4.0 * k_h)) ** 0.4 * v0 ** -0.2


def reduced_form_coefficient() -> float:
    """``c = C (15/16)^(2/5)`` -- coefficient of the ``(m^2/(R E*^2))^(1/5)`` form.

    Derivation encoded here (no constant typed by hand): substituting
    ``k_h = (4/3) E* sqrt(R)`` into the stiffness form gives
    ``(5 m / (4 k_h))^(2/5) = (15/16)^(2/5) (m^2 / (R E*^2))^(1/5)``.
    """
    return prefactor() * (15.0 / 16.0) ** 0.4


def contact_time_reduced(v0: float, m_eff: float, R_eff: float, E_star: float) -> float:
    """``t_c = c (m^2 / (R E*^2))^(1/5) v0^(-1/5)`` with ``c`` above."""
    if not (v0 > 0.0 and m_eff > 0.0 and R_eff > 0.0 and E_star > 0.0):
        raise ValueError("all arguments must be positive")
    return reduced_form_coefficient() * (
        m_eff * m_eff / (R_eff * E_star * E_star)
    ) ** 0.2 * v0 ** -0.2


# ---------------------------------------------------------------------------
# 1. Dimension closure by unit-rescaling covariance
# ---------------------------------------------------------------------------

_Dim = tuple[Fraction, Fraction, Fraction]


def _dim(mass: int = 0, length: int = 0, time: int = 0) -> _Dim:
    return (Fraction(mass), Fraction(length), Fraction(time))


BASE_DIMS: dict[str, _Dim] = {
    "mass": _dim(1),
    "length": _dim(0, 1),
    "time": _dim(0, 0, 1),
    "velocity": _dim(0, 1, -1),
    "hertz_stiffness": _dim(1, Fraction(-1, 2), -2),   # N m^(-3/2)
    "youngs_modulus": _dim(1, -1, -2),                 # Pa
    "force": _dim(1, 1, -2),
}


def _unit_scale(dim: _Dim, units: dict[str, float]) -> float:
    """Size of the composed unit, in SI, for a quantity of dimension ``dim``."""
    a, b, c = units["mass"], units["length"], units["time"]
    return (float(a) ** float(dim[0])) * (float(b) ** float(dim[1])) * (
        float(c) ** float(dim[2])
    )


def _closure_residual(
    func: Callable[..., float],
    params: dict[str, tuple[float, _Dim]],
    out_dim: _Dim,
    units: dict[str, float],
) -> dict[str, Any]:
    """Evaluate ``func`` in SI and in a rescaled unit system; compare.

    ``func`` is called with the *same physical* arguments, but in the rescaled
    system every value is divided by the size of its unit.  The output is
    converted back by the output unit size.  A dimensionally consistent
    monomial returns the identical physical value.
    """
    si_args = [value for value, _ in params.values()]
    scaled_args = [
        value / _unit_scale(dim, units) for value, dim in params.values()
    ]
    value_si = func(*si_args)
    value_scaled = func(*scaled_args)
    back = value_scaled * _unit_scale(out_dim, units)
    return {
        "value_si": value_si,
        "value_scaled": value_scaled,
        "value_scaled_back_si": back,
        "residual": abs(back / value_si - 1.0),
    }


def dimension_closure_table(
    m_eff: float = 0.06,
    R_star: float = 0.025,
    E_star: float = 1.098901098901099e9,
    v0: float = 1.0,
) -> dict[str, Any]:
    """Dimension closure residuals for ``t_c``, ``delta_max``, ``F_max``.

    Two unit systems are used (a cgs/ms system and a deliberately mixed one) so
    that a closure that accidentally holds in one rescaling is not accepted.
    One negative control with a wrong velocity exponent is included; its residual
    must be large, which is what makes the passing rows meaningful.
    """
    k_h = hertz_stiffness(E_star, R_star)
    unit_systems = {
        "si": {"mass": 1.0, "length": 1.0, "time": 1.0},
        "cgs_ms": {"mass": 1e-3, "length": 1e-2, "time": 1e-3},
        "mixed": {"mass": 2.5, "length": 7e-3, "time": 3.1e-4},
    }
    quantities: dict[str, tuple[Callable[..., float], dict[str, tuple[float, _Dim]], _Dim]] = {
        "t_c": (
            lambda m, k, v: contact_time(v, m, k),
            {"m_eff": (m_eff, BASE_DIMS["mass"]),
             "k_h": (k_h, BASE_DIMS["hertz_stiffness"]),
             "v0": (v0, BASE_DIMS["velocity"])},
            BASE_DIMS["time"],
        ),
        "delta_max": (
            lambda m, k, v: peak_overlap(v, m, k),
            {"m_eff": (m_eff, BASE_DIMS["mass"]),
             "k_h": (k_h, BASE_DIMS["hertz_stiffness"]),
             "v0": (v0, BASE_DIMS["velocity"])},
            BASE_DIMS["length"],
        ),
        "F_max": (
            lambda m, k, v: peak_force(v, m, k),
            {"m_eff": (m_eff, BASE_DIMS["mass"]),
             "k_h": (k_h, BASE_DIMS["hertz_stiffness"]),
             "v0": (v0, BASE_DIMS["velocity"])},
            BASE_DIMS["force"],
        ),
        "t_c_reduced_form": (
            lambda m, r, e, v: contact_time_reduced(v, m, r, e),
            {"m_eff": (m_eff, BASE_DIMS["mass"]),
             "R_eff": (R_star, BASE_DIMS["length"]),
             "E_star": (E_star, BASE_DIMS["youngs_modulus"]),
             "v0": (v0, BASE_DIMS["velocity"])},
            BASE_DIMS["time"],
        ),
    }

    def _wrong_exponent(m: float, k: float, v: float) -> float:
        # velocity exponent -1/4 instead of -1/5: dimensionally identical,
        # physically wrong.  A closure test must not accept it.
        return prefactor() * (5.0 * m / (4.0 * k)) ** 0.4 * v ** -0.25

    rows: list[dict[str, Any]] = []
    for name, (func, params, out_dim) in quantities.items():
        for unit_name, units in unit_systems.items():
            res = _closure_residual(func, params, out_dim, units)
            rows.append({
                "quantity": name,
                "unit_system": unit_name,
                "unit_sizes_si": units,
                "output_dimension": [str(x) for x in out_dim],
                "value_si": res["value_si"],
                "value_in_scaled_units": res["value_scaled"],
                "value_scaled_back_to_si": res["value_scaled_back_si"],
                "relative_residual": res["residual"],
                "sensitivity_control": False,
            })
    for unit_name, units in unit_systems.items():
        res = _closure_residual(
            _wrong_exponent,
            {"m_eff": (m_eff, BASE_DIMS["mass"]),
             "k_h": (k_h, BASE_DIMS["hertz_stiffness"]),
             "v0": (v0, BASE_DIMS["velocity"])},
            BASE_DIMS["time"],
            units,
        )
        rows.append({
            "quantity": "t_c_wrong_velocity_exponent_-1/4",
            "unit_system": unit_name,
            "unit_sizes_si": units,
            "output_dimension": [str(x) for x in BASE_DIMS["time"]],
            "value_si": res["value_si"],
            "value_in_scaled_units": res["value_scaled"],
            "value_scaled_back_to_si": res["value_scaled_back_si"],
            "relative_residual": res["residual"],
            "sensitivity_control": True,
        })
    honest = [r["relative_residual"] for r in rows if not r["sensitivity_control"]]
    control = [r["relative_residual"] for r in rows if r["sensitivity_control"]]
    return {
        "rows": rows,
        "max_residual_honest": max(honest),
        "min_residual_control_nontrivial": min(
            r["relative_residual"] for r in rows
            if r["sensitivity_control"] and r["unit_system"] != "si"
        ),
        "min_residual_control": min(control),
        "closure_holds_at_1e-12": max(honest) <= 1e-12,
        # the identity rescaling cannot expose a wrong exponent; the sensitivity
        # control must separate on every *non-trivial* unit system
        "control_separates": min(
            r["relative_residual"] for r in rows
            if r["sensitivity_control"] and r["unit_system"] != "si"
        ) > 1e-6,
    }


# ---------------------------------------------------------------------------
# 2. Algebraic identity between the two textbook forms
# ---------------------------------------------------------------------------

def algebraic_identity_table(
    conditions: Sequence[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Check ``contact_time == contact_time_reduced`` and the mission's ``2.87``.

    Two statements are checked separately:

    * the *structural* identity (both forms agree to machine precision for every
      condition), and
    * the *numeric* coefficient: ``c = 2.86805...`` vs the ``2.87`` printed in the
      mission brief, accepted only as a 3-significant-figure rounding with an
      explicit tolerance (``<= 1e-3`` relative), never as an anchor value.
    """
    if conditions is None:
        conditions = [
            {"label": "ice_grain", "m": 0.06, "R": 0.025,
             "E": 1.098901098901099e9, "v": 1.0},
            {"label": "steel_ball", "m": 0.1, "R": 0.01, "E": 1.0e11, "v": 0.5},
            {"label": "fine_sand", "m": 1.0e-9, "R": 1.5e-4, "E": 5.0e10, "v": 3.0},
            {"label": "floe_edge", "m": 50.0, "R": 2.0, "E": 5.0e9, "v": 0.05},
        ]
    rows = []
    for cond in conditions:
        k_h = hertz_stiffness(cond["E"], cond["R"])
        t_a = contact_time(cond["v"], cond["m"], k_h)
        t_b = contact_time_reduced(cond["v"], cond["m"], cond["R"], cond["E"])
        rows.append({
            "label": cond["label"],
            "m_eff": cond["m"], "R_eff": cond["R"], "E_star": cond["E"],
            "v0": cond["v"], "k_h": k_h,
            "t_c_stiffness_form": t_a,
            "t_c_reduced_form": t_b,
            "relative_residual": abs(t_b / t_a - 1.0),
        })
    c = reduced_form_coefficient()
    rounded_residual = abs(c / SIGNIFICANT_FIGURE_FORM_COEFF - 1.0)
    return {
        "coefficient_value": c,
        "coefficient_stiffness_form": prefactor(),
        "coefficient_predicted_by_derivation": prefactor() * (15.0 / 16.0) ** 0.4,
        "mission_brief_value": SIGNIFICANT_FIGURE_FORM_COEFF,
        "rounded_value_relative_difference": rounded_residual,
        "rounded_value_consistent_at_3sf": rounded_residual <= 1e-3,
        "rows": rows,
        "max_structural_residual": max(r["relative_residual"] for r in rows),
    }


# ---------------------------------------------------------------------------
# 3. Limits, including the v -> 0 divergence form
# ---------------------------------------------------------------------------

def _loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = float(len(lx))
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx


def limit_table(
    m_eff: float = 0.06,
    k_h: float = 2.0e8,
    speeds: Sequence[float] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1),
    reference_time_s: float = 1.0,
) -> dict[str, Any]:
    """Speed sweep with exponents and the divergence asymptote.

    Reported per speed: ``t_c``, ``delta_max``, ``F_max`` and the three invariants
    ``t_c v0^(1/5)``, ``delta_max / v0^(4/5)``, ``F_max / v0^(6/5)``.  The
    divergence ``t_c -> inf`` as ``v0 -> 0`` is quantified in two ways: the
    log-log slope must be ``-1/5`` and ``t_c v0^(1/5)`` must be speed
    independent.  The speed at which ``t_c`` reaches ``reference_time_s`` is
    returned because that is where the asymptote stops being physical.
    """
    rows = []
    for v in speeds:
        t_c = contact_time(v, m_eff, k_h)
        d_max = peak_overlap(v, m_eff, k_h)
        f_max = peak_force(v, m_eff, k_h)
        rows.append({
            "v0": v,
            "t_c": t_c,
            "delta_max": d_max,
            "F_max": f_max,
            "t_c_times_v0_pow_1_5": t_c * v ** 0.2,
            "delta_max_over_v0_pow_4_5": d_max / v ** 0.8,
            "F_max_over_v0_pow_6_5": f_max / v ** 1.2,
        })
    exact = {"t_c": -0.2, "delta_max": 0.8, "F_max": 1.2}
    slopes = {
        key: _loglog_slope(speeds, [r[key] for r in rows])
        for key in ("t_c", "delta_max", "F_max")
    }
    invariants = {
        "t_c_times_v0_pow_1_5": [r["t_c_times_v0_pow_1_5"] for r in rows],
        "delta_max_over_v0_pow_4_5": [r["delta_max_over_v0_pow_4_5"] for r in rows],
        "F_max_over_v0_pow_6_5": [r["F_max_over_v0_pow_6_5"] for r in rows],
    }
    spread = {k: max(v) / min(v) - 1.0 for k, v in invariants.items()}
    A = contact_time(1.0, m_eff, k_h)          # t_c = A v^(-1/5)
    v_ref = (A / reference_time_s) ** 5.0
    return {
        "conditions": {"m_eff": m_eff, "k_h": k_h, "speeds": list(speeds)},
        "rows": rows,
        "exponents_measured": slopes,
        "exponents_exact": exact,
        "exponent_residuals": {k: abs(slopes[k] - exact[k]) for k in slopes},
        "invariant_spread": spread,
        "divergence": {
            "form": "t_c = A v0^(-1/5), A = C (5 m / (4 k_h))^(2/5) "
                    "[m^(2/5) s^(4/5)] => t_c -> +inf as v0 -> 0",
            "A_si": A,
            "speed_at_reference_time_m_s": v_ref,
            "reference_time_s": reference_time_s,
            "speed_ratio_for_10x_duration": 1e-5,
            "decade_note": "t_c grows by a factor 10 for every 5 decades of "
                           "speed decrease (v^(-1/5)); this is the quantified "
                           "divergence form of the v -> 0 limit",
            "reading": "the divergence is an ideal-elastic asymptote; the "
                       "closed form loses physical meaning once t_c exceeds the "
                       "external time scale (gravity, preload, or the next "
                       "collision), i.e. below the quoted v_ref",
        },
        "vanishing_limits": {
            "delta_max_as_v_to_0": "delta_max ~ v0^(4/5) -> 0 (spread check)",
            "F_max_as_v_to_0": "F_max ~ v0^(6/5) -> 0 (spread check)",
        },
    }


# ---------------------------------------------------------------------------
# 4. Stiffness ratio and damping applicability domain
# ---------------------------------------------------------------------------

def stiffness_ratio_rows(
    conditions: Sequence[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Equivalent-linear stiffness vs Hertz secant stiffness at peak overlap.

    ``k_eq`` is defined as the linear spring that reproduces the Hertz duration:
    ``pi sqrt(m / k_eq) = t_c``  =>  ``k_eq = pi^2 m / t_c^2``.  The ratio
    ``k_eq / (k_h sqrt(delta_max))`` is the dimensionless stiffness-selection
    number; the algebra of the two closed forms makes it the constant
    ``(4/5) pi^2 / C^2``, so any linearisation anchored at the *peak secant*
    stiffness (ratio 1) is stiff by ``1/0.9117 = 1.0968`` and its duration is
    short by ``1 - sqrt(0.9117) = 4.5 %`` -- outside the 3 % band.
    """
    if conditions is None:
        conditions = [
            {"label": "ice_grain_v0.1", "m": 0.06, "k": 2.0e8, "v": 0.1},
            {"label": "ice_grain_v1.0", "m": 0.06, "k": 2.0e8, "v": 1.0},
            {"label": "steel_ball_v0.5", "m": 0.1, "k": 2.3e10, "v": 0.5},
            {"label": "fine_sand_v3.0", "m": 1.0e-9, "k": 1.1e7, "v": 3.0},
            {"label": "floe_edge_v0.05", "m": 50.0, "k": 3.0e9, "v": 0.05},
        ]
    predicted = 0.8 * math.pi ** 2 / prefactor() ** 2
    rows = []
    for cond in conditions:
        t_c = contact_time(cond["v"], cond["m"], cond["k"])
        d_max = peak_overlap(cond["v"], cond["m"], cond["k"])
        k_eq = math.pi ** 2 * cond["m"] / t_c ** 2
        k_secant = cond["k"] * math.sqrt(d_max)
        ratio = k_eq / k_secant
        t_linear = math.pi * math.sqrt(cond["m"] / k_secant)
        rows.append({
            "label": cond["label"],
            "m_eff": cond["m"], "k_h": cond["k"], "v0": cond["v"],
            "t_c": t_c,
            "delta_max": d_max,
            "k_equivalent_linear": k_eq,
            "k_secant_at_peak": k_secant,
            "ratio_k_eq_over_k_secant": ratio,
            "ratio_residual_vs_constant": abs(ratio / predicted - 1.0),
            "t_c_linear_spring_secant": t_linear,
            "linear_duration_relative_error": t_linear / t_c - 1.0,
            "stiffness_scale_to_match_duration": 1.0 / ratio,
        })
    errors = [abs(r["linear_duration_relative_error"]) for r in rows]
    return {
        "ratio_constant_predicted": predicted,
        "ratio_constant_form": "(4/5) pi^2 / C^2",
        "max_ratio_residual": max(r["ratio_residual_vs_constant"] for r in rows),
        "rows": rows,
        "max_linear_duration_error": max(errors),
        "linear_secant_spring_within_3pct": max(errors) <= 0.03,
        "reading": "a linear spring set to the Hertz peak secant stiffness is "
                   "systematically short in duration (universal "
                   "1 - sqrt(k_eq/k_secant) = -4.5 %), which is already outside "
                   "the 3 % band: a linear surrogate must scale the secant "
                   "stiffness by 1/0.9117 = 1.0968, or the duration must come "
                   "from the Hertz closed form itself",
    }


def hunt_crossley_impact(
    v0: float,
    m_eff: float,
    k_h: float,
    alpha: float,
    *,
    damping_free_time: float | None = None,
    steps_per_contact: int = 4000,
) -> dict[str, Any]:
    """Integrate one Hunt-Crossley impact ``F = k_h d^(3/2) (1 + alpha d_dot)``.

    Returns the duration, the coefficient of restitution and the duration ratio
    against the undamped closed form.  ``alpha`` has dimension s/m; the
    dimensionless group is ``alpha v0``.
    """
    if not (v0 > 0.0 and m_eff > 0.0 and k_h > 0.0):
        raise ValueError("v0, m_eff, k_h must be positive")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    t_ref = damping_free_time or contact_time(v0, m_eff, k_h)
    dt = t_ref / steps_per_contact

    def accel(d: float, v: float) -> float:
        if d <= 0.0:
            return 0.0
        return -(k_h * d ** 1.5 * (1.0 + alpha * v)) / m_eff

    d, v = 0.0, v0
    t = 0.0
    peak = 0.0
    compressed = False
    max_steps = 40 * steps_per_contact
    for _ in range(max_steps):
        a1 = accel(d, v)
        d2, v2 = d + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = accel(d2, v2)
        d3, v3 = d + 0.5 * dt * v2, v + 0.5 * dt * a2
        a3 = accel(d3, v3)
        d4, v4 = d + dt * v3, v + dt * a3
        a4 = accel(d4, v4)
        d_new = d + dt / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v_new = v + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        t += dt
        if d_new > peak:
            peak = d_new
        if d_new > 0.0:
            compressed = True
        elif compressed:
            # linear interpolation of the exit instant inside this step
            frac = d_new / (d_new - d) if d_new != d else 0.0
            t_exit = t - dt + frac * dt
            v_exit = v + frac * (v_new - v)
            return {
                "v0": v0, "m_eff": m_eff, "k_h": k_h, "alpha": alpha,
                "alpha_v0": alpha * v0,
                "t_c": t_exit,
                "t_c_damping_free": t_ref,
                "duration_relative_change": t_exit / t_ref - 1.0,
                "delta_max": peak,
                "restitution": abs(v_exit) / v0,
                "steps_per_contact": steps_per_contact,
            }
        d, v = d_new, v_new
    raise RuntimeError("impact did not separate within the step budget")


def damping_domain_rows(
    *,
    v0: float = 1.0,
    m_eff: float = 0.06,
    k_h: float = 2.0e8,
    alpha_values: Sequence[float] = (0.0, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1),
) -> dict[str, Any]:
    """Sweep the dimensionless damping number ``alpha v0``.

    The closed form used as an anchor is undamped (``e = 1``).  The applicability
    boundary is the largest ``alpha v0`` for which the duration still agrees with
    the undamped form to ``<= 3 %``; the same rows carry the realised restitution
    so the law's energy loss is visible.
    """
    base = contact_time(v0, m_eff, k_h)
    rows = []
    for alpha in alpha_values:
        row = hunt_crossley_impact(v0, m_eff, k_h, alpha)
        row["duration_error_vs_undamped"] = row["duration_relative_change"]
        row["within_3pct_of_undamped"] = abs(row["duration_relative_change"]) <= 0.03
        rows.append(row)
    within = [r for r in rows if r["within_3pct_of_undamped"]]
    alpha_bound = max((r["alpha_v0"] for r in within), default=0.0)
    # Hunt-Crossley small-damping asymptote e = 1 - (2/3) alpha v0
    for r in rows:
        predicted = 1.0 - (2.0 / 3.0) * r["alpha_v0"]
        r["restitution_small_damping_asymptote"] = predicted
        r["restitution_asymptote_residual"] = abs(r["restitution"] / predicted - 1.0)
    return {
        "conditions": {"v0": v0, "m_eff": m_eff, "k_h": k_h,
                       "damping_free_time": base},
        "rows": rows,
        "alpha_v0_bound_for_3pct_duration": alpha_bound,
        "restitution_at_bound": next(
            (r["restitution"] for r in rows if r["alpha_v0"] == alpha_bound), 1.0),
        "reading": "the undamped closed form is admissible only while the "
                   "dimensionless damping number alpha*v0 stays below the quoted "
                   "bound; beyond it the anchor's duration is a lower bound and "
                   "the realised restitution must be used instead of e = 1",
    }


# ---------------------------------------------------------------------------
# 5. Anchor record assembly
# ---------------------------------------------------------------------------

def describe_anchor() -> dict[str, Any]:
    """The anchor record plus its operator-layer support, copy-safe."""
    import copy

    anchor = copy.deepcopy(PRIMARY_ANCHOR)
    anchor["operator_layer_support"] = {
        "dimension_closure": dimension_closure_table(),
        "algebraic_identity": algebraic_identity_table(),
        "limits": limit_table(),
        "stiffness_ratio": stiffness_ratio_rows(),
        "damping_domain": damping_domain_rows(),
    }
    return anchor