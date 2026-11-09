"""External-anchor verification harness for the ice-hull-propeller study.

The ``ice-hull-propeller-threefield`` research plan compares the DEM/LBM thermal
three-field results with the ISO 19906:2019 pressure-area practice.  That
comparison is only legitimate if the external anchors are (i) transcribed
verbatim from the standard, (ii) converted to the case calibre by a script that
can be recomputed, and (iii) screened by dimensional and internal-consistency
checks.  This module implements (ii) and (iii) plus the *claim gate* that makes
(i) impossible to skip silently.

Evidence grading (Tensor ecosystem convention):

``REAL``
    the dimensional algebra, the conversion chain, the recompute residuals and
    the claim gate -- all closed-form arithmetic over registered numbers.
``WITHHELD``
    the numerical anchors themselves.  The ISO 19906:2019 primary text is not in
    this worktree (paywalled standard; ``docs/ICE_CONTACT_PRESSURE_ROLLING_PROXY.md``
    already records that the pressure-area exponent "was not digitised from the
    standard here"), so every number below is an **unverified candidate** and
    :func:`require_anchor` refuses to hand it out for a claim.

Why this is not a formality: a pressure-area power law is fully determined by
three numbers (coefficient, reference area, exponent) and *any* plausible triple
reproduces a smooth monotone curve, so a wrong-but-plausible transcription is
invisible to numerics.  Only the verbatim quote + table/page locator can catch
it, which is why the gate keys on provenance rather than on plausibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DIM_LENGTH",
    "DIM_STRESS",
    "DIM_FORCE",
    "DIM_LINE_LOAD",
    "DIMENSIONLESS",
    "Dimension",
    "DimensionMismatch",
    "UnverifiedAnchorError",
    "AnchorCandidate",
    "require_anchor",
    "pressure_area_candidate",
    "global_force_locator_candidate",
    "coefficient_dimension",
    "pressure_area_coefficient",
    "pressure_from_area",
    "line_load_from_pressure",
    "global_force",
    "CaliberLevel",
    "ChainPoint",
    "convert_pressure_area",
    "dimensional_homogeneity_checks",
    "internal_consistency_checks",
    "recompute_residual",
]


# --------------------------------------------------------------------------- #
# dimensional algebra (exact rational exponents, no floating point)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dimension:
    """Physical dimension as exact exponents of (mass, length, time)."""

    mass: Fraction = Fraction(0)
    length: Fraction = Fraction(0)
    time: Fraction = Fraction(0)

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.mass + other.mass, self.length + other.length,
                         self.time + other.time)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.mass - other.mass, self.length - other.length,
                         self.time - other.time)

    def __pow__(self, exponent: Fraction | int) -> "Dimension":
        power = Fraction(exponent)
        return Dimension(self.mass * power, self.length * power,
                         self.time * power)

    def as_tuple(self) -> tuple[Fraction, Fraction, Fraction]:
        return (self.mass, self.length, self.time)

    def __str__(self) -> str:  # pragma: no cover - presentation helper
        return f"(M^{self.mass} L^{self.length} T^{self.time})"


DIMENSIONLESS = Dimension()
DIM_LENGTH = Dimension(Fraction(0), Fraction(1), Fraction(0))
DIM_STRESS = Dimension(Fraction(1), Fraction(-1), Fraction(-2))   # Pa   = M L^-1 T^-2
DIM_FORCE = Dimension(Fraction(1), Fraction(1), Fraction(-2))      # N    = M L T^-2
DIM_LINE_LOAD = Dimension(Fraction(1), Fraction(0), Fraction(-2))  # N/m  = M T^-2


class DimensionMismatch(ValueError):
    """Raised when an equation is not dimensionally homogeneous."""


class UnverifiedAnchorError(RuntimeError):
    """Raised when an unverified candidate anchor is used for a claim."""


# --------------------------------------------------------------------------- #
# anchors and the claim gate
# --------------------------------------------------------------------------- #
VERIFIED = "VERIFIED"
UNVERIFIED_CANDIDATE = "UNVERIFIED_CANDIDATE"
LOCATOR_ONLY = "LOCATOR_ONLY"


@dataclass(frozen=True)
class AnchorCandidate:
    """One external anchor, with its provenance and verification status.

    ``quote`` must be the literal sentence(s) of the source, ``locator`` the
    table/clause/page.  Both are ``None`` until the primary text has been read
    in this worktree: a remembered formula is not an anchor.
    """

    anchor_id: str
    quantity: str
    value: float | None
    unit: str
    dimension: Dimension
    provenance_type: str
    verification_status: str
    quote: str | None = None
    locator: str | None = None
    locator_hint: str | None = None
    source_note: str | None = None

    @property
    def usable_for_claim(self) -> bool:
        """True only for a verbatim, located transcription."""
        return (
            self.verification_status == VERIFIED
            and bool(self.quote)
            and bool(self.locator)
            and self.value is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "quantity": self.quantity,
            "candidate_value": self.value,
            "unit": self.unit,
            "dimension": {
                "mass": str(self.dimension.mass),
                "length": str(self.dimension.length),
                "time": str(self.dimension.time),
            },
            "provenance_type": self.provenance_type,
            "verification_status": self.verification_status,
            "quote": self.quote,
            "locator": self.locator,
            "locator_hint": self.locator_hint,
            "source_note": self.source_note,
            "usable_for_claim": self.usable_for_claim,
        }


def require_anchor(anchor: AnchorCandidate) -> float:
    """Return the anchor value, or refuse if it is not verbatim-verified.

    This is the only sanctioned way for downstream code (comparison, plotting,
    claim text) to read an external anchor.
    """
    if not anchor.usable_for_claim:
        raise UnverifiedAnchorError(
            f"anchor {anchor.anchor_id!r} is {anchor.verification_status} "
            f"(quote={'yes' if anchor.quote else 'missing'}, "
            f"locator={'yes' if anchor.locator else 'missing'}); "
            "an unverified registration may only be used as a cross-check "
            "reference, never in a claim"
        )
    assert anchor.value is not None  # guaranteed by usable_for_claim
    return float(anchor.value)


_IN_REPO_NOT_DIGITISED = (
    "docs/ICE_CONTACT_PRESSURE_ROLLING_PROXY.md:170 -- \"the pressure-area "
    "exponent was not digitised from the standard here\""
)


def pressure_area_candidate(*, exponent: float, c_ref_pa: float,
                            a_ref_m2: float, candidate_id: str,
                            locator_hint: str,
                            provenance_note: str) -> AnchorCandidate:
    """Register the pressure-area triple as an *unverified* candidate.

    Three numbers define the law ``p = C_A A^ex`` with the reference point
    ``p(A_R) = C_R``; they are registered together so the harness can be run and
    so the gap is visible, but they stay unusable for claims until transcribed.
    """
    for name, value in (("exponent", exponent), ("c_ref_pa", c_ref_pa),
                        ("a_ref_m2", a_ref_m2)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    return AnchorCandidate(
        anchor_id=candidate_id,
        quantity=(
            "pressure-area exponent ex, reference pressure C_R, reference area A_R "
            f"(ex={exponent!r}, C_R={c_ref_pa!r} Pa, A_R={a_ref_m2!r} m^2)"
        ),
        value=float(exponent),
        unit="1 (exponent); C_R in Pa; A_R in m^2",
        dimension=DIMENSIONLESS,
        provenance_type=provenance_note,
        verification_status=UNVERIFIED_CANDIDATE,
        quote=None,
        locator=None,
        locator_hint=locator_hint,
        source_note=_IN_REPO_NOT_DIGITISED,
    )


def global_force_locator_candidate(*, candidate_id: str,
                                   locator_hint: str) -> AnchorCandidate:
    """Register the design-global-ice-force anchor as locator-only (no value).

    No tabulated single "design global ice force" value is recalled, so nothing
    numeric is registered: the anchor's *existence and location* is itself the
    open question.  Registering a number here would be fabrication.
    """
    return AnchorCandidate(
        anchor_id=candidate_id,
        quantity="design global ice force (clause-level action format)",
        value=None,
        unit="N",
        dimension=DIM_FORCE,
        provenance_type="locator-hypothesis-only",
        verification_status=LOCATOR_ONLY,
        quote=None,
        locator=None,
        locator_hint=locator_hint,
        source_note=(
            "no value registered on purpose: recall-level candidates are not "
            "transcriptions; the clause/table that carries the design global "
            "force format has to be located in the primary text first"
        ),
    )


# --------------------------------------------------------------------------- #
# conversion chain: ISO pressure-area -> case calibre
# --------------------------------------------------------------------------- #
def coefficient_dimension(exponent: float) -> Dimension:
    """Dimension of ``C_A`` in ``p = C_A A^ex``: stress x length^(-2 ex)."""
    ex = Fraction(float(exponent)).limit_denominator(1000)
    return DIM_STRESS * (DIM_LENGTH ** (-2 * ex))


def pressure_area_coefficient(c_ref_pa: float, a_ref_m2: float,
                              exponent: float) -> float:
    """``C_A = C_R A_R^(-ex)`` so that ``p(A_R) = C_R`` exactly."""
    if c_ref_pa <= 0.0 or a_ref_m2 <= 0.0:
        raise ValueError("reference pressure and area must be positive")
    if not math.isfinite(exponent):
        raise ValueError("exponent must be finite")
    return c_ref_pa * a_ref_m2 ** (-float(exponent))


def pressure_from_area(c_a: float, area_m2: float, exponent: float) -> float:
    if area_m2 <= 0.0:
        raise ValueError("area must be positive")
    return c_a * area_m2 ** float(exponent)


def line_load_from_pressure(pressure_pa: float, thickness_m: float) -> float:
    """``q = p h`` in N/m -- the dimension the unit-width DEM bow reaction has."""
    if thickness_m <= 0.0:
        raise ValueError("thickness must be positive")
    return pressure_pa * thickness_m


def global_force(pressure_pa: float, area_m2: float) -> float:
    if area_m2 <= 0.0:
        raise ValueError("area must be positive")
    return pressure_pa * area_m2


@dataclass(frozen=True)
class CaliberLevel:
    """One calibre of the case: a contact width times an ice thickness."""

    label: str
    width_m: float
    thickness_m: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.thickness_m <= 0.0:
            raise ValueError("caliber width and thickness must be positive")

    @property
    def area_m2(self) -> float:
        return self.width_m * self.thickness_m


@dataclass(frozen=True)
class ChainPoint:
    """Converted pressure / line load / force at one calibre level."""

    label: str
    width_m: float
    thickness_m: float
    area_m2: float
    pressure_pa: float
    pressure_mpa: float
    line_load_n_per_m: float
    force_n: float
    area_ratio_to_reference: float
    outside_reference_area: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "width_m": self.width_m,
            "thickness_m": self.thickness_m,
            "area_m2": self.area_m2,
            "pressure_pa": self.pressure_pa,
            "pressure_mpa": self.pressure_mpa,
            "line_load_n_per_m": self.line_load_n_per_m,
            "force_n": self.force_n,
            "area_ratio_to_reference": self.area_ratio_to_reference,
            "outside_reference_area": self.outside_reference_area,
            "note": self.note,
        }


def convert_pressure_area(*, exponent: float, c_ref_pa: float, a_ref_m2: float,
                          levels: Sequence[CaliberLevel]) -> dict[str, Any]:
    """Run the full conversion chain and return values **and** residuals.

    Chain: ``(ex, C_R, A_R) -> C_A = C_R A_R^-ex -> p(A) = C_A A^ex ->
    q = p h -> F = p A``, with the three consistency residuals reported rather
    than asserted so the artifact carries the numbers, not a claim of success.
    """
    c_a = pressure_area_coefficient(c_ref_pa, a_ref_m2, exponent)
    points: list[ChainPoint] = []
    for level in levels:
        area = level.area_m2
        p = pressure_from_area(c_a, area, exponent)
        q = line_load_from_pressure(p, level.thickness_m)
        f = global_force(p, area)
        # residual 1: F must equal q * w (kN/m x m)
        res_force = abs(f - q * level.width_m) / max(abs(f), 1e-300)
        # residual 2: q must equal p h
        res_line = abs(q - p * level.thickness_m) / max(abs(q), 1e-300)
        if res_force > 1e-12 or res_line > 1e-12:
            raise AssertionError(
                f"conversion chain not self-consistent at {level.label!r}: "
                f"force residual {res_force:.3e}, line-load residual {res_line:.3e}")
        ratio = area / a_ref_m2
        points.append(ChainPoint(
            label=level.label,
            width_m=level.width_m,
            thickness_m=level.thickness_m,
            area_m2=area,
            pressure_pa=p,
            pressure_mpa=p / 1.0e6,
            line_load_n_per_m=q,
            force_n=f,
            area_ratio_to_reference=ratio,
            outside_reference_area=not (0.5 <= ratio <= 2.0),
            note=level.note,
        ))
    # residual 3: reference-point recovery p(A_R) == C_R
    p_ref = pressure_from_area(c_a, a_ref_m2, exponent)
    res_ref = abs(p_ref - c_ref_pa) / c_ref_pa
    if res_ref > 1e-12:
        raise AssertionError(f"reference point not recovered: {res_ref:.3e}")
    # residual 4: inversion A(p) round trip
    log_slope = _log_slope(c_a, points[0].area_m2, exponent)
    return {
        "exponent": float(exponent),
        "reference_pressure_pa": float(c_ref_pa),
        "reference_area_m2": float(a_ref_m2),
        "pressure_area_coefficient_pa": c_a,
        "coefficient_dimension": str(coefficient_dimension(exponent)),
        "points": [point.as_dict() for point in points],
        "residuals": {
            "reference_point_recovery_relative": res_ref,
            "inversion_log_slope_minus_exponent": log_slope - float(exponent),
        },
    }


def _log_slope(c_a: float, area_m2: float, exponent: float) -> float:
    """Numerical ``d ln p / d ln A`` -- must equal the exponent."""
    h = 1e-6
    p_hi = pressure_from_area(c_a, area_m2 * math.exp(h), exponent)
    p_lo = pressure_from_area(c_a, area_m2 * math.exp(-h), exponent)
    return (math.log(p_hi) - math.log(p_lo)) / (2.0 * h)


# --------------------------------------------------------------------------- #
# machine checks
# --------------------------------------------------------------------------- #
def _check(name: str, lhs: Dimension, rhs: Dimension,
           detail: str) -> dict[str, Any]:
    ok = lhs.as_tuple() == rhs.as_tuple()
    if not ok:
        raise DimensionMismatch(f"{name}: {lhs} != {rhs} ({detail})")
    return {
        "equation": name,
        "lhs_dimension": str(lhs),
        "rhs_dimension": str(rhs),
        "homogeneous": ok,
        "detail": detail,
    }


def dimensional_homogeneity_checks(exponent: float) -> list[dict[str, Any]]:
    """Every equation of the chain, checked by exact exponent arithmetic."""
    ex = Fraction(float(exponent)).limit_denominator(1000)
    return [
        _check("q = p h", DIM_STRESS * DIM_LENGTH, DIM_LINE_LOAD,
               "pressure x length must be a line load [N/m]"),
        _check("F = p A", DIM_STRESS * (DIM_LENGTH ** 2), DIM_FORCE,
               "pressure x area must be a force [N]"),
        _check("F = q w", DIM_LINE_LOAD * DIM_LENGTH, DIM_FORCE,
               "line load x width must be a force [N]"),
        _check("A = w h", DIM_LENGTH * DIM_LENGTH, DIM_LENGTH ** 2,
               "contact area from width and thickness [m^2]"),
        _check("p(A) = C_A A^ex",
               (DIM_STRESS / (DIM_LENGTH ** (2 * ex))) * (DIM_LENGTH ** (2 * ex)),
               DIM_STRESS,
               "C_A carries the compensating exponent so the product is a stress"),
        _check("A / A_R", (DIM_LENGTH ** 2) / (DIM_LENGTH ** 2), DIMENSIONLESS,
               "the pressure-area ratio must be dimensionless"),
    ]


def internal_consistency_checks(*, exponent: float, c_ref_pa: float,
                                a_ref_m2: float, levels: Sequence[CaliberLevel],
                                chain: Mapping[str, Any]) -> dict[str, Any]:
    """Consistency of the converted numbers with the law that produced them."""
    c_a = float(chain["pressure_area_coefficient_pa"])
    points = chain["points"]
    checks: list[dict[str, Any]] = []

    # 1. reference point recovery
    p_ref = pressure_from_area(c_a, a_ref_m2, exponent)
    checks.append({
        "check": "reference point p(A_R) == C_R",
        "relative_residual": abs(p_ref - c_ref_pa) / c_ref_pa,
        "tolerance": 1e-12,
        "ok": abs(p_ref - c_ref_pa) / c_ref_pa < 1e-12,
    })

    # 2. round trip A -> p -> A on every level
    worst_round = 0.0
    for level, point in zip(levels, points):
        a_back = (point["pressure_pa"] / c_a) ** (1.0 / float(exponent))
        worst_round = max(worst_round,
                          abs(a_back - level.area_m2) / level.area_m2)
    checks.append({
        "check": "area round trip A -> p -> A",
        "worst_relative_residual": worst_round,
        "tolerance": 1e-9,
        "ok": worst_round < 1e-9,
    })

    # 3. log-slope of the converted curve equals the exponent
    slope = _log_slope(c_a, levels[0].area_m2, exponent)
    checks.append({
        "check": "d ln p / d ln A == ex on the converted curve",
        "value": slope,
        "target": float(exponent),
        "absolute_residual": abs(slope - float(exponent)),
        "tolerance": 1e-9,
        "ok": abs(slope - float(exponent)) < 1e-9,
    })

    # 4. scale covariance: lengths x lambda -> p x lambda^(2 ex), q x lambda^(1+2ex)
    lam = 3.0
    p0 = points[0]["pressure_pa"]
    p_scaled = pressure_from_area(c_a, points[0]["area_m2"] * lam ** 2, exponent)
    expected_ratio = lam ** (2.0 * float(exponent))
    got_ratio = p_scaled / p0
    checks.append({
        "check": "scale covariance of p(A) under A -> lambda^2 A",
        "lambda": lam,
        "expected_ratio": expected_ratio,
        "measured_ratio": got_ratio,
        "relative_residual": abs(got_ratio - expected_ratio) / expected_ratio,
        "tolerance": 1e-12,
        "ok": abs(got_ratio - expected_ratio) / expected_ratio < 1e-12,
    })

    # 5. monotonicity: a size effect needs ex < 0 (smaller area -> higher pressure)
    lo, hi = sorted((points[0]["area_m2"], points[-1]["area_m2"]))
    if hi > lo:
        p_lo = pressure_from_area(c_a, lo, exponent)
        p_hi = pressure_from_area(c_a, hi, exponent)
        decreases = p_hi < p_lo
    else:
        decreases = None
    checks.append({
        "check": "pressure decreases with contact area (size effect)",
        "exponent_negative": float(exponent) < 0.0,
        "converted_curve_decreases": decreases,
        "ok": (float(exponent) < 0.0) and (decreases in (True, None)),
    })

    # 6. unit handling: MPa bookkeeping is exact
    worst_unit = max(abs(point["pressure_mpa"] * 1.0e6 - point["pressure_pa"])
                     / point["pressure_pa"] for point in points)
    checks.append({
        "check": "MPa -> Pa bookkeeping (x 1e6)",
        "worst_relative_residual": worst_unit,
        "tolerance": 1e-15,
        "ok": worst_unit < 1e-15,
    })

    return {
        "checks": checks,
        "all_ok": all(item["ok"] for item in checks),
    }


def recompute_residual(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the conversion chain from the artifact's own stored numbers.

    This is the "复算" criterion: read the JSON, rebuild ``p``, ``q`` and ``F``
    from ``exponent`` / ``C_R`` / ``A_R`` / geometry and compare with the values
    the artifact stores.  Returns the worst relative residual, which must be
    ``< 1e-9``.
    """
    chain = artifact["conversion_chain"]
    exponent = float(chain["exponent"])
    c_ref = float(chain["reference_pressure_pa"])
    a_ref = float(chain["reference_area_m2"])
    c_a = pressure_area_coefficient(c_ref, a_ref, exponent)
    stored_c_a = float(chain["pressure_area_coefficient_pa"])
    worst = abs(c_a - stored_c_a) / abs(stored_c_a)
    details = []
    for point in chain["points"]:
        p = pressure_from_area(c_a, float(point["area_m2"]), exponent)
        q = line_load_from_pressure(p, float(point["thickness_m"]))
        f = global_force(p, float(point["area_m2"]))
        residuals = {
            "pressure": abs(p - float(point["pressure_pa"])) / abs(p),
            "line_load": abs(q - float(point["line_load_n_per_m"])) / abs(q),
            "force": abs(f - float(point["force_n"])) / abs(f),
        }
        worst = max(worst, *residuals.values())
        details.append({"label": point["label"], "relative_residuals": residuals})
    # Independent algebraic route: recompute the pressure from the reference
    # point, p = C_R (A/A_R)^ex, which never touches C_A.  Any gluing error in
    # the coefficient or the area ratio shows up here as a first-order residual
    # instead of being reproduced bit-identically by the same expression.
    independent = []
    for point in chain["points"]:
        p_alt = c_ref * (float(point["area_m2"]) / a_ref) ** exponent
        residual = abs(p_alt - float(point["pressure_pa"])) / abs(p_alt)
        worst = max(worst, residual)
        independent.append({"label": point["label"],
                            "pressure_reference_form_pa": p_alt,
                            "relative_residual": residual})
    return {
        "worst_relative_residual": worst,
        "tolerance": 1e-9,
        "ok": worst < 1e-9,
        "per_point": details,
        "independent_reference_form": independent,
        "independent_form_note": (
            "p = C_R (A/A_R)^ex is the same law written without C_A; agreement "
            "to machine precision shows the coefficient is not mis-glued"
        ),
    }