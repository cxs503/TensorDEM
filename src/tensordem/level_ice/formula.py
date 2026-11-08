"""Level-ice resistance engineering anchors (Lindqvist 1989 / Riska-type).

Why this module exists
----------------------
:mod:`tensordem.threefield.resistance` already carries a Lindqvist/Riska
*family* baseline for the coupled three-field workflow.  That module is
explicit that its coefficient set is an implementation ledger
(``docs/threefield/FORMULAS.md``: ``DERIVED`` / ``CONVENTION`` / ``WITHHELD``)
and that it must not be read as a reproduction of the published papers.  The
level-ice DEM verification surface needs something with tighter bookkeeping:

* a **closed three-component decomposition** (crushing / bending / submersion)
  whose partial derivatives with respect to every physical input are stated,
  so a DEM component split can be compared against it term by term;
* a **machine-checkable provenance hash** over the coefficient ledger and the
  experimental reference table, so a reported resistance number can be traced
  to the exact constants that produced it;
* an explicit **provenance grade** per coefficient, carried in the returned
  payloads instead of being hidden in a docstring.

What is implemented
-------------------
``lindqvist_components`` returns

    R_c = 0.5 * sigma_c * h^2 * (tan(alpha) + mu / cos(alpha))      crushing
    R_b = 0.25 * sigma_f * B * h^2 / L                              bending
    R_s = rho_w * g * h^2 * B * (0.9 + 0.05 * mu * L / B)           submersion
    R_v = 0.5 * rho_w * B * h * V^2                                 added drag

with ``R = R_c + R_b + R_s + R_v``.  This is the classical three-component
Lindqvist structure: the crushing term is a *total* force localised on the
stem (its implicit contact width is the ice thickness ``h``, which is why
``h^2`` appears and the beam ``B`` does not), while bending and submersion
scale with the waterline breadth ``B``.

Provenance policy
-----------------
The primary papers are not readable from this offline sandbox, so no
coefficient value here is claimed to reproduce a published number.  Every
constant is graded:

``DERIVED``   follows from a stated physical assumption and is rederived in
              the docstring / by the test suite;
``CONVENTION``  a pinned order-one modelling choice;
``WITHHELD``  literature agreement cannot be established offline.

``reference_cases`` loads the MT Uikku level-ice table that ships with this
repository (`benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv`),
and ``fit_hull_to_published`` performs the only literature-facing check that
is possible offline: a three-parameter fit of the hull particulars
``(B, L, alpha)`` to the paper's own Lindqvist prediction column.  The fit
residual measures how much of the published column is explained by this
structure; it does **not** validate the coefficient values.

All quantities are SI: m, Pa, kg/m^3, m/s, N.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

WATER_DENSITY = 1025.0
ICE_DENSITY = 917.0
GRAVITY = 9.80665

#: Coefficient ledger.  Every entry is (value, grade, justification).  The
#: ledger is hashed verbatim into every payload (``ledger_sha256``).
COEFFICIENT_LEDGER: dict[str, tuple[float, str, str]] = {
    "crushing_efficiency": (
        0.5,
        "DERIVED",
        "Mean of a triangular contact-pressure profile over the ice edge: "
        "sigma_c at the lower surface, zero at the upper surface, so "
        "(1/h) int_0^h sigma_c (y/h) dy = sigma_c/2.",
    ),
    "bending_efficiency": (
        0.25,
        "CONVENTION",
        "Root bending moment sigma_f * B * h^2 / 6 divided by an effective "
        "lever arm L/1.5 (1.5x the pure cantilever factor 1/6).  Pinned "
        "order-one choice; it is the single largest structural convention in "
        "the Lindqvist bending term.",
    ),
    "submersion_buoyancy": (
        0.9,
        "CONVENTION",
        "Dimensionless channel-opening coefficient: the cleared channel "
        "floats a displaced ice volume fraction 0.9 of B*h^2.",
    ),
    "submersion_friction": (
        0.05,
        "CONVENTION",
        "Scales the friction of the displaced rubble sheet over the hull "
        "length as 0.05 * mu * L/B.",
    ),
    "drag_coefficient": (
        0.5,
        "DERIVED",
        "Dynamic-pressure drag 0.5*rho_w*C_d with C_d = 1 on the displaced "
        "sheet area B*h.",
    ),
    "literature_agreement": (
        float("nan"),
        "WITHHELD",
        "Lindqvist (1989) / Riska (1989) are not readable offline; no claim "
        "is made that this coefficient set reproduces the published "
        "predictions.  The MT Uikku column fit in fit_hull_to_published is a "
        "structure-level check only.",
    ),
}


def ledger_sha256() -> str:
    """SHA-256 over the graded coefficient ledger (stable JSON dump)."""
    payload = {
        name: [value if math.isfinite(value) else "nan", grade, text]
        for name, (value, grade, text) in sorted(COEFFICIENT_LEDGER.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
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


@dataclass(frozen=True)
class IceState:
    """Level-ice material state and thickness (SI)."""

    thickness_m: float = 1.0
    compressive_strength_pa: float = 1.5e6
    flexural_strength_pa: float = 0.5e6
    elastic_modulus_pa: float = 5.0e9
    density_kg_m3: float = ICE_DENSITY

    def __post_init__(self) -> None:
        _finite("thickness_m", self.thickness_m, positive=True)
        _finite("compressive_strength_pa", self.compressive_strength_pa, positive=True)
        _finite("flexural_strength_pa", self.flexural_strength_pa, positive=True)
        _finite("elastic_modulus_pa", self.elastic_modulus_pa, positive=True)
        _finite("density_kg_m3", self.density_kg_m3, positive=True)

    def as_dict(self) -> dict[str, float]:
        return {
            "thickness_m": self.thickness_m,
            "compressive_strength_pa": self.compressive_strength_pa,
            "flexural_strength_pa": self.flexural_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "density_kg_m3": self.density_kg_m3,
        }


@dataclass(frozen=True)
class HullForm:
    """Hull particulars entering the engineering anchors (SI / degrees)."""

    length_m: float = 120.0
    beam_m: float = 20.0
    draught_m: float = 7.0
    entry_angle_deg: float = 22.0
    stem_angle_deg: float = 30.0
    friction_coefficient: float = 0.05
    name: str = "hull"

    def __post_init__(self) -> None:
        _finite("length_m", self.length_m, positive=True)
        _finite("beam_m", self.beam_m, positive=True)
        _finite("draught_m", self.draught_m, positive=True)
        for name in ("entry_angle_deg", "stem_angle_deg"):
            value = _finite(name, getattr(self, name), positive=True)
            if value >= 89.0:
                raise ValueError(f"{name} must be < 89 degrees")
        _finite("friction_coefficient", self.friction_coefficient, nonnegative=True)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "length_m": self.length_m,
            "beam_m": self.beam_m,
            "draught_m": self.draught_m,
            "entry_angle_deg": self.entry_angle_deg,
            "stem_angle_deg": self.stem_angle_deg,
            "friction_coefficient": self.friction_coefficient,
        }


#: Declared hull presets used by the verification campaign.  ``icebreaker`` is
#: a generic polar icebreaker; ``transport`` is the icebreaking-tanker
#: equivalent of the MT Uikku reference set produced by the offline hull fit
#: (:func:`fit_hull_to_published`); see ``docs/LEVEL_ICE_RESISTANCE_DEM.md``.
ICEBREAKER = HullForm(
    length_m=120.0, beam_m=20.0, draught_m=7.0,
    entry_angle_deg=22.0, stem_angle_deg=30.0, friction_coefficient=0.05,
    name="icebreaker",
)
TRANSPORT = HullForm(
    length_m=105.0, beam_m=26.0, draught_m=8.0,
    entry_angle_deg=18.0, stem_angle_deg=26.0, friction_coefficient=0.05,
    name="transport",
)


def crushing_resistance(ice: IceState, hull: HullForm) -> float:
    """Stem crushing force ``R_c`` [N] (total force, implicit contact width h)."""
    alpha = math.radians(hull.entry_angle_deg)
    mu = hull.friction_coefficient
    if math.cos(alpha) <= 0.0:  # pragma: no cover - guarded by HullForm
        raise ValueError("degenerate entry angle")
    return (
        COEFFICIENT_LEDGER["crushing_efficiency"][0]
        * ice.compressive_strength_pa
        * ice.thickness_m**2
        * (math.tan(alpha) + mu / math.cos(alpha))
    )


def bending_resistance(ice: IceState, hull: HullForm) -> float:
    """Bending (flexural failure) force ``R_b`` [N]."""
    return (
        COEFFICIENT_LEDGER["bending_efficiency"][0]
        * ice.flexural_strength_pa
        * hull.beam_m
        * ice.thickness_m**2
        / hull.length_m
    )


def submersion_resistance(
    ice: IceState,
    hull: HullForm,
    water_density_kg_m3: float = WATER_DENSITY,
    gravity_m_s2: float = GRAVITY,
) -> float:
    """Submersion force ``R_s`` [N]: buoyant depression plus channel friction."""
    _finite("water_density_kg_m3", water_density_kg_m3, positive=True)
    _finite("gravity_m_s2", gravity_m_s2, positive=True)
    h = ice.thickness_m
    mu = hull.friction_coefficient
    factor = (
        COEFFICIENT_LEDGER["submersion_buoyancy"][0]
        + COEFFICIENT_LEDGER["submersion_friction"][0] * mu * hull.length_m / hull.beam_m
    )
    return water_density_kg_m3 * gravity_m_s2 * h * h * hull.beam_m * factor


def added_drag(
    ice: IceState,
    hull: HullForm,
    speed_m_s: float,
    water_density_kg_m3: float = WATER_DENSITY,
) -> float:
    """Quadratic added drag ``R_v`` [N] on the displaced sheet area ``B*h``."""
    speed = _finite("speed_m_s", speed_m_s, nonnegative=True)
    return (
        COEFFICIENT_LEDGER["drag_coefficient"][0]
        * water_density_kg_m3 * hull.beam_m * ice.thickness_m * speed**2
    )


def lindqvist_components(
    ice: IceState,
    hull: HullForm = ICEBREAKER,
    speed_m_s: float = 0.0,
    *,
    water_density_kg_m3: float = WATER_DENSITY,
    gravity_m_s2: float = GRAVITY,
) -> dict[str, Any]:
    """Three-component Lindqvist anchor with a machine-checkable provenance hash."""
    speed = _finite("speed_m_s", speed_m_s, nonnegative=True)
    r_c = crushing_resistance(ice, hull)
    r_b = bending_resistance(ice, hull)
    r_s = submersion_resistance(ice, hull, water_density_kg_m3, gravity_m_s2)
    r_v = added_drag(ice, hull, speed, water_density_kg_m3)
    total = r_c + r_b + r_s + r_v
    return {
        "model": "lindqvist-3component",
        "hull": hull.as_dict(),
        "ice": ice.as_dict(),
        "speed_m_s": speed,
        "crushing_N": r_c,
        "bending_N": r_b,
        "submersion_N": r_s,
        "added_drag_N": r_v,
        "static_N": r_c + r_b + r_s,
        "total_N": total,
        "crushing_over_bending": r_c / r_b,
        "crushing_fraction": r_c / total,
        "bending_fraction": r_b / total,
        "submersion_fraction": r_s / total,
        "ledger_sha256": ledger_sha256(),
        "provenance": {
            name: grade for name, (_v, grade, _t) in COEFFICIENT_LEDGER.items()
        },
    }


def riska_components(
    ice: IceState,
    hull: HullForm = ICEBREAKER,
    speed_m_s: float = 0.0,
    *,
    water_density_kg_m3: float = WATER_DENSITY,
    gravity_m_s2: float = GRAVITY,
) -> dict[str, Any]:
    """Riska-type speed-linear anchor ``R = C1 + C2 * V``.

    The C1/C2 assembly is the repository's documented three-field baseline
    (:func:`tensordem.threefield.resistance.riska_resistance`), so the two
    anchors share no duplicated source of truth.  Its own coefficient ledger is
    graded in ``docs/threefield/FORMULAS.md``; like the Lindqvist anchor above,
    literature agreement is **not** claimed.
    """
    from tensordem.threefield.resistance import (
        Environment,
        HullGeometry,
        IceProperties,
        riska_resistance,
    )

    speed = _finite("speed_m_s", speed_m_s, nonnegative=True)
    payload = riska_resistance(
        IceProperties(
            thickness_m=ice.thickness_m,
            compressive_strength_pa=ice.compressive_strength_pa,
            flexural_strength_pa=ice.flexural_strength_pa,
            elastic_modulus_pa=ice.elastic_modulus_pa,
            density_kg_m3=ice.density_kg_m3,
        ),
        HullGeometry(
            length_m=hull.length_m,
            beam_m=hull.beam_m,
            draught_m=hull.draught_m,
            waterline_entry_angle_deg=hull.entry_angle_deg,
            stem_angle_deg=hull.stem_angle_deg,
            buttock_angle_deg=20.0,
            friction_coefficient=hull.friction_coefficient,
        ),
        speed,
        Environment(water_density_kg_m3=water_density_kg_m3, gravity_m_s2=gravity_m_s2),
    )
    payload = dict(payload)
    payload["model"] = "riska-speed-linear"
    payload["hull"] = hull.as_dict()
    payload["ledger_sha256"] = ledger_sha256()
    payload["provenance"] = {
        "riska_c1_c2": "WITHHELD (docs/threefield/FORMULAS.md coefficient ledger)"
    }
    return payload


# ---------------------------------------------------------------------------
# Experimental reference table (MT Uikku level-ice model tests)
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_CSV = (
    Path(__file__).resolve().parents[3]
    / "benchmarks" / "dem3d" / "ship_ice_validation" / "mt_uikku_level_ice.csv"
)

REFERENCE_COLUMNS = (
    "test_id", "ice_sheet", "ice_type", "ice_thickness_m",
    "flexural_strength_kPa", "compressive_strength_kPa",
    "ice_elastic_modulus_MPa", "ice_drift_speed_m_s",
    "measured_mean_resistance_table3_kN", "measured_std_kN",
    "measured_max_kN", "measured_min_kN",
    "experimental_reference_table8_kN", "lindqvist_prediction_kN",
    "riska_prediction_kN", "jeong_prediction_kN", "keinonen_prediction_kN",
)


def _optional_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


def reference_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the MT Uikku level-ice case table with a SHA-256 content hash.

    Missing cells stay ``None`` (the source paper does not report every
    statistic for every case) - they are never estimated.
    """
    csv_path = Path(path) if path is not None else DEFAULT_REFERENCE_CSV
    raw = csv_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REFERENCE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"reference CSV is missing columns: {missing}")
        for raw_row in reader:
            row: dict[str, Any] = {
                "test_id": raw_row["test_id"].strip(),
                "ice_sheet": raw_row["ice_sheet"].strip(),
                "ice_type": raw_row["ice_type"].strip(),
            }
            for column in REFERENCE_COLUMNS[3:]:
                row[column] = _optional_float(raw_row[column])
            rows.append(row)
    if not rows:
        raise ValueError("reference CSV is empty")
    for row in rows:
        row["source_sha256"] = digest
        row["source_path"] = str(csv_path)
    return rows


def level_ice_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Reference rows restricted to level ice (channel cases are excluded)."""
    return [row for row in reference_cases(path) if row["ice_type"] == "level"]


def _lindqvist_total_for_hull(
    row: dict[str, Any], beam: float, length: float, entry_deg: float,
    friction: float = 0.05,
) -> float:
    ice = IceState(
        thickness_m=row["ice_thickness_m"],
        flexural_strength_pa=1e3 * row["flexural_strength_kPa"],
        compressive_strength_pa=1e3 * row["compressive_strength_kPa"],
        elastic_modulus_pa=1e6 * row["ice_elastic_modulus_MPa"],
    )
    hull = HullForm(
        length_m=length, beam_m=beam, draught_m=8.0,
        entry_angle_deg=entry_deg, stem_angle_deg=30.0,
        friction_coefficient=friction,
    )
    return lindqvist_components(ice, hull, row["ice_drift_speed_m_s"])["total_N"]


def fit_hull_to_published(
    rows: Iterable[dict[str, Any]] | None = None,
    *,
    column: str = "lindqvist_prediction_kN",
    initial: tuple[float, float, float] = (26.0, 105.0, 20.0),
) -> dict[str, Any]:
    """Least-squares fit of ``(B, L, alpha)`` to a published formula column.

    This is the only literature-facing check available offline.  It asks:
    *can this three-component structure reproduce the numbers the paper
    attribute to Lindqvist (1989) as the hull shape varies?*  A small
    residual says the structure spans the published column; it says nothing
    about the coefficient values (which stay ``WITHHELD``).
    """
    cases = [r for r in (rows if rows is not None else level_ice_cases())
             if r.get(column)]
    if len(cases) < 4:
        raise ValueError("need at least four cases with the published column")
    targets = [1e3 * r[column] for r in cases]

    def residual(theta: "list[float]") -> list[float]:
        beam, length, entry = theta
        if not (5.0 < beam < 60.0 and 40.0 < length < 400.0 and 5.0 < entry < 60.0):
            return [1e9] * len(cases)
        return [
            _lindqvist_total_for_hull(r, beam, length, entry) - t
            for r, t in zip(cases, targets)
        ]

    try:
        from scipy.optimize import least_squares

        solution = least_squares(
            residual, list(initial), bounds=([5.0, 40.0, 5.0], [60.0, 400.0, 60.0]),
            x_scale=[10.0, 100.0, 10.0], xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        theta = list(solution.x)
        success = bool(solution.success)
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        theta = list(initial)
        success = False
    predictions = [sum(_lindqvist_total_for_hull(r, *theta) for r in [c]) for c in cases]
    errors = [100.0 * (p - t) / t for p, t in zip(predictions, targets)]
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    mape = sum(abs(e) for e in errors) / len(errors)
    return {
        "column": column,
        "n_cases": len(cases),
        "fitted": {"beam_m": theta[0], "length_m": theta[1],
                   "entry_angle_deg": theta[2]},
        "success": success,
        "rms_error_pct": rms,
        "mape_pct": mape,
        "max_abs_error_pct": max(abs(e) for e in errors),
        "case_ids": [r["test_id"] for r in cases],
        "per_case_error_pct": errors,
        "source_sha256": cases[0]["source_sha256"],
        "grade": "STRUCTURE-LEVEL ONLY (coefficient values stay WITHHELD)",
    }


def reference_summary(path: str | Path | None = None) -> dict[str, Any]:
    """Published-formula-versus-measurement audit of the reference table."""
    cases = level_ice_cases(path)
    audit: dict[str, Any] = {"n_cases": len(cases), "source_sha256": None,
                             "columns": {}}
    if cases:
        audit["source_sha256"] = cases[0]["source_sha256"]
        audit["source_path"] = cases[0]["source_path"]
    for column in ("lindqvist_prediction_kN", "riska_prediction_kN",
                   "jeong_prediction_kN", "keinonen_prediction_kN"):
        pairs = [
            (r[column], r["experimental_reference_table8_kN"])
            for r in cases
            if r.get(column) and r.get("experimental_reference_table8_kN")
        ]
        if not pairs:
            continue
        errors = [100.0 * (p - t) / t for p, t in pairs]
        audit["columns"][column] = {
            "n": len(pairs),
            "mape_pct": sum(abs(e) for e in errors) / len(errors),
            "bias_pct": sum(errors) / len(errors),
            "max_abs_error_pct": max(abs(e) for e in errors),
        }
    return audit


def relative_error_pct(predicted: float, reference: float) -> float:
    """Signed percentage error ``100 * (predicted - reference) / reference``."""
    reference = _finite("reference", reference, positive=True)
    predicted = _finite("predicted", predicted)
    return 100.0 * (predicted - reference) / reference


def power_law_exponent(x: Iterable[float], y: Iterable[float]) -> float:
    """Least-squares exponent ``n`` of ``y ~ x**n`` (log-log, no offset)."""
    xs = [_finite("x", v, positive=True) for v in x]
    ys = [_finite("y", v, positive=True) for v in y]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need matching sequences of length >= 2")
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    sxx = sum((v - mx) ** 2 for v in lx)
    if sxx <= 0.0:
        raise ValueError("x values must not be identical")
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    """Pearson product-moment correlation coefficient."""
    xs = [_finite("x", v) for v in x]
    ys = [_finite("y", v) for v in y]
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need matching sequences of length >= 2")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((v - mx) ** 2 for v in xs)
    syy = sum((v - my) ** 2 for v in ys)
    if sxx <= 0.0 or syy <= 0.0:
        raise ValueError("zero variance in a sequence")
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / math.sqrt(sxx * syy)