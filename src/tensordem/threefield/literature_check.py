"""Identifiability audit of the shipped Lindqvist / Riska chains against published
predictions for the MT Uikku level-ice cases.

Why this module exists
----------------------
``tensordem.threefield.resistance`` ships two engineering baselines whose
component *structure* follows Lindqvist (1989) and Riska (1987/1989), while the
coefficient set is a documented convention -- ``docs/threefield/FORMULAS.md``
grades literature agreement as **WITHHELD** because the primary papers were not
reachable offline.  A ``WITHHELD`` grade is honest but operationally weak: the
two baselines are used as the *reference gate* for the DEM ice-resistance work,
so a reader cannot tell whether the gate is literature-anchored or an
unverified curve set.

This module replaces the unavailable papers with data the repository *already*
carries: ``benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv``
transcribes the ten level-ice cases of Hu and Zhou (2016), including their
published Lindqvist and Riska predictions (kN).  Published predictions computed
from the real formulas are exactly the reference a paraphrase must be tested
against, so literature agreement becomes a *computable* question:

1. **Is the published Riska column structurally ``R = C1 + C2 V``?**  If yes, its
   speed slope ``C2`` and static part ``C1`` can be read off the data without
   knowing the hull, and the *relative* speed dependence ``C2 / C1`` is a
   hull-insensitive signature of the published model.
2. **Can the shipped chain reproduce the published columns at *any* hull inside
   a declared, physically defensible envelope?**  Both chains carry the hull only
   through ``(L, B, alpha, mu, phi)``; if the best achievable maximum relative
   error over that envelope still exceeds the platform's 15 % tolerance, then the
   shipped coefficient set is *not* a reproduction of the published model under
   any admissible geometry -- a much stronger statement than ``WITHHELD``.
3. **How far apart are the signatures?**  The best (largest) ``C2 / C1`` the
   shipped Riska chain can reach inside the envelope is compared with the
   published value.  This is the decisive, geometry-robust test: it does not
   depend on identifying the real ship.

Evidence grades produced (see ``run_audit``):

* ``published_riska_linearity`` -- REAL (data-internal, machine-checked);
* ``shipped_chain_vs_published_as_shipped`` -- REAL (numbers computed here);
* ``shipped_chain_vs_published_best_feasible_hull`` -- REAL (bounded optimum);
* ``literature_agreement_of_shipped_coefficients`` -- **REFUTED-IN-ENVELOPE**
  when the bounded best fit misses the 15 % gate, else WITHHELD (not proven).

Scope: nothing here validates the shipped coefficients against the *text* of the
primary papers (still unavailable); it validates them against published
*predictions*, which is what the repository uses them for.  The envelope is an
assumption and is declared as such.

All forces in N internally, displayed in kN; SI everywhere.

Reproducer: ``scripts/audit_threefield_resistance_literature.py`` (JSON + figures).
Regression ratchet: ``tests/test_threefield_resistance_literature_check.py``.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .resistance import (
    Environment,
    HullGeometry,
    IceProperties,
    lindqvist_resistance,
    riska_resistance,
)

PROTOCOL = "tensordem-threefield-literature-identifiability-v1"

#: Repository-relative location of the transcribed Hu & Zhou (2016) case set.
REFERENCE_CSV = Path("benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv")

#: Platform-wide engineering tolerance (percent) used as the acceptance gate.
TOLERANCE_PCT = 15.0

#: Declared hull envelope for the identifiability question.  Bounds are broad
#: enough to contain any icebreaking tanker of MT Uikku's class (length between
#: perpendiculars 120-260 m, beam 18-46 m, beam/length 0.09-0.26, waterline
#: entry angle 10-36 deg, stem angle 15-48 deg, steel-ice friction 0.02-0.15)
#: yet narrow enough that a fit cannot hide behind unphysical geometry.
ENVELOPE: dict[str, tuple[float, float]] = {
    "length_m": (120.0, 260.0),
    "beam_m": (18.0, 46.0),
    "waterline_entry_angle_deg": (10.0, 36.0),
    "stem_angle_deg": (15.0, 48.0),
    "friction_coefficient": (0.02, 0.15),
}
BEAM_LENGTH_RATIO: tuple[float, float] = (0.09, 0.26)
DRAUGHT_M = 8.0
ENVELOPE_EVIDENCE_GRADE = "CONVENTION (declared assumption, not a literature value)"


@dataclass(frozen=True)
class LiteratureCase:
    """One transcribed MT Uikku level-ice case (SI, forces in newtons)."""

    test_id: str
    thickness_m: float
    flexural_strength_pa: float
    compressive_strength_pa: float
    elastic_modulus_pa: float
    speed_m_s: float
    measured_resistance_n: float | None
    lindqvist_n: float
    riska_n: float

    def ice(self) -> IceProperties:
        return IceProperties(
            thickness_m=self.thickness_m,
            compressive_strength_pa=self.compressive_strength_pa,
            flexural_strength_pa=self.flexural_strength_pa,
            elastic_modulus_pa=self.elastic_modulus_pa,
        )


def _optional_float(text: str | None) -> float | None:
    if text is None or text.strip() == "":
        return None
    return float(text)


def load_cases(path: str | Path | None = None) -> list[LiteratureCase]:
    """Parse the transcribed case set; missing source values stay missing (None)."""
    target = Path(path) if path is not None else REFERENCE_CSV
    if not target.is_absolute():
        target = Path(__file__).resolve().parents[3] / target
    cases: list[LiteratureCase] = []
    with target.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ice_type"].strip() != "level":
                raise ValueError(f"case {row['test_id']} is not a level-ice case")
            cases.append(
                LiteratureCase(
                    test_id=row["test_id"].strip(),
                    thickness_m=float(row["ice_thickness_m"]),
                    flexural_strength_pa=float(row["flexural_strength_kPa"]) * 1e3,
                    compressive_strength_pa=float(row["compressive_strength_kPa"]) * 1e3,
                    elastic_modulus_pa=float(row["ice_elastic_modulus_MPa"]) * 1e6,
                    speed_m_s=float(row["ice_drift_speed_m_s"]),
                    measured_resistance_n=(
                        None if _optional_float(row["experimental_reference_table8_kN"]) is None
                        else _optional_float(row["experimental_reference_table8_kN"]) * 1e3
                    ),
                    lindqvist_n=float(row["lindqvist_prediction_kN"]) * 1e3,
                    riska_n=float(row["riska_prediction_kN"]) * 1e3,
                )
            )
    if len(cases) != 10:
        raise ValueError(f"expected 10 level-ice cases, parsed {len(cases)}")
    return cases


def published_riska_linearity(cases: Sequence[LiteratureCase]) -> dict[str, Any]:
    """Fit ``R = C1 + C2 V`` to the published Riska column, grouped by thickness.

    Only groups with at least two distinct speeds carry slope information; the
    published column is rounded to 1 kN, so the residual of an exact linear model
    is reported in kN and compared with that rounding floor.
    """
    groups: dict[float, list[LiteratureCase]] = {}
    for case in cases:
        groups.setdefault(round(case.thickness_m, 6), []).append(case)
    fits: list[dict[str, Any]] = []
    for thickness, members in sorted(groups.items()):
        speeds = {round(m.speed_m_s, 6) for m in members}
        if len(speeds) < 2:
            continue
        v = np.array([m.speed_m_s for m in members], dtype=float)
        r = np.array([m.riska_n for m in members], dtype=float)
        c2, c1 = np.polyfit(v, r, 1)
        residual = r - (c1 + c2 * v)
        fits.append(
            {
                "thickness_m": thickness,
                "test_ids": [m.test_id for m in members],
                "c1_n": float(c1),
                "c2_n_per_m_s": float(c2),
                "c2_over_c1_s_per_m": float(c2 / c1),
                "max_abs_residual_n": float(np.max(np.abs(residual))),
                "flexural_strength_spread_pa": float(
                    max(m.flexural_strength_pa for m in members)
                    - min(m.flexural_strength_pa for m in members)
                ),
            }
        )
    if not fits:
        raise ValueError("no thickness group carries two speeds; the slope is unidentified")
    return {
        "groups": fits,
        "rounding_floor_n": 1000.0,
        "max_abs_residual_n": max(f["max_abs_residual_n"] for f in fits),
        "is_linear_within_rounding": all(f["max_abs_residual_n"] <= 150.0 for f in fits),
        "c2_over_c1_s_per_m": [f["c2_over_c1_s_per_m"] for f in fits],
    }


@dataclass(frozen=True)
class HullParameters:
    """Flattened hull point used by the bounded search."""

    length_m: float
    beam_m: float
    entry_angle_deg: float
    stem_angle_deg: float
    friction_coefficient: float

    def hull(self) -> HullGeometry:
        return HullGeometry(
            length_m=self.length_m,
            beam_m=self.beam_m,
            draught_m=DRAUGHT_M,
            waterline_entry_angle_deg=self.entry_angle_deg,
            stem_angle_deg=self.stem_angle_deg,
            friction_coefficient=self.friction_coefficient,
        )


def _decode(vector: Sequence[float]) -> HullParameters | None:
    length, beam, entry, stem, friction = (float(x) for x in vector)
    ratio = beam / length
    if not (BEAM_LENGTH_RATIO[0] <= ratio <= BEAM_LENGTH_RATIO[1]):
        return None
    return HullParameters(length, beam, entry, stem, friction)


def _encode(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(*ENVELOPE["length_m"]),
            rng.uniform(*ENVELOPE["beam_m"]),
            rng.uniform(*ENVELOPE["waterline_entry_angle_deg"]),
            rng.uniform(*ENVELOPE["stem_angle_deg"]),
            rng.uniform(*ENVELOPE["friction_coefficient"]),
        ]
    )


def _scan_bounds() -> np.ndarray:
    return np.array([ENVELOPE[key] for key in
                     ("length_m", "beam_m", "waterline_entry_angle_deg",
                      "stem_angle_deg", "friction_coefficient")])


def relative_errors_pct(
    cases: Iterable[LiteratureCase],
    hull: HullGeometry,
    model: str,
    environment: Environment = Environment(),
) -> np.ndarray:
    """Signed ``(shipped - published) / published`` per case, in percent."""
    values: list[float] = []
    for case in cases:
        ice = case.ice()
        if model == "lindqvist":
            predicted = lindqvist_resistance(ice, hull, case.speed_m_s, environment)["total_N"]
            reference = case.lindqvist_n
        elif model == "riska":
            predicted = riska_resistance(ice, hull, case.speed_m_s, environment)["total_N"]
            reference = case.riska_n
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown model {model!r}")
        values.append(100.0 * (predicted - reference) / reference)
    return np.array(values, dtype=float)


def default_hull_errors(
    cases: Sequence[LiteratureCase], model: str
) -> dict[str, Any]:
    """Errors at the package default hull (the configuration the tests pin)."""
    hull = HullGeometry()
    errors = relative_errors_pct(cases, hull, model)
    return {
        "model": model,
        "hull": hull.as_dict(),
        "per_case_pct": [float(e) for e in errors],
        "max_abs_pct": float(np.max(np.abs(errors))),
        "mean_signed_pct": float(np.mean(errors)),
        "mape_pct": float(np.mean(np.abs(errors))),
        "within_tolerance": bool(np.max(np.abs(errors)) <= TOLERANCE_PCT),
    }


def best_feasible_fit(
    cases: Sequence[LiteratureCase],
    model: str,
    *,
    samples: int = 4000,
    refinement_rounds: int = 6,
    seed: int = 20261104,
) -> dict[str, Any]:
    """Minimise the maximum absolute relative error over the declared envelope.

    Deterministic: a seeded random scan followed by coordinate refinement.  The
    objective is ``max_i |error_i|`` because a chain used as an acceptance gate
    must not be screened by an aggregate that hides one bad case.
    """
    rng = np.random.default_rng(seed)
    best_vector = None
    best_value = math.inf
    for _ in range(samples):
        vector = _encode(rng)
        decoded = _decode(vector)
        if decoded is None:
            continue
        value = float(np.max(np.abs(relative_errors_pct(cases, decoded.hull(), model))))
        if value < best_value:
            best_value, best_vector = value, vector.copy()
    if best_vector is None:  # pragma: no cover - envelope always admits points
        raise RuntimeError("envelope produced no admissible hull")

    lows, highs = _scan_bounds()[:, 0], _scan_bounds()[:, 1]
    span = (highs - lows) / 8.0
    for _ in range(refinement_rounds):
        improved = False
        for axis in range(best_vector.size):
            for step in (-span[axis], span[axis]):
                trial = best_vector.copy()
                trial[axis] = min(max(trial[axis] + step, lows[axis]), highs[axis])
                decoded = _decode(trial)
                if decoded is None:
                    continue
                value = float(np.max(np.abs(relative_errors_pct(cases, decoded.hull(), model))))
                if value < best_value - 1e-9:
                    best_value, best_vector, improved = value, trial, True
        span *= 0.5
        if not improved and span.max() < 1e-6:  # pragma: no cover - safety valve
            break

    decoded = _decode(best_vector)
    assert decoded is not None
    errors = relative_errors_pct(cases, decoded.hull(), model)
    return {
        "model": model,
        "max_abs_pct": float(np.max(np.abs(errors))),
        "mape_pct": float(np.mean(np.abs(errors))),
        "mean_signed_pct": float(np.mean(errors)),
        "per_case_pct": [float(e) for e in errors],
        "hull": {
            "length_m": decoded.length_m,
            "beam_m": decoded.beam_m,
            "waterline_entry_angle_deg": decoded.entry_angle_deg,
            "stem_angle_deg": decoded.stem_angle_deg,
            "friction_coefficient": decoded.friction_coefficient,
            "beam_length_ratio": decoded.beam_m / decoded.length_m,
        },
        "seed": seed,
        "samples": samples,
        "within_tolerance": bool(float(np.max(np.abs(errors))) <= TOLERANCE_PCT),
    }


def shipped_speed_signature(
    case: LiteratureCase, hull: HullGeometry, *,
    speed_span: tuple[float, float] = (0.2, 1.0),
) -> dict[str, float]:
    """``C2 / C1`` of the shipped Riska chain for one ice state.

    ``R(V) = C1 + C2 V`` is evaluated at two speeds and decomposed, which is exact
    for the shipped law (it is affine in ``V``) and keeps the comparison with the
    published column free of any assumption about the code's internals.
    """
    v_low, v_high = speed_span
    ice = case.ice()
    low = riska_resistance(ice, hull, v_low)["total_N"]
    high = riska_resistance(ice, hull, v_high)["total_N"]
    slope = (high - low) / (v_high - v_low)
    intercept = low - slope * v_low
    return {
        "c1_n": float(intercept),
        "c2_n_per_m_s": float(slope),
        "c2_over_c1_s_per_m": float(slope / intercept),
    }


def max_feasible_speed_signature(
    case: LiteratureCase, *, samples: int = 20000, seed: int = 20261104
) -> dict[str, Any]:
    """Largest ``C2 / C1`` the shipped Riska chain can reach inside the envelope."""
    rng = np.random.default_rng(seed)
    best = {"c2_over_c1_s_per_m": -math.inf}
    for _ in range(samples):
        decoded = _decode(_encode(rng))
        if decoded is None:
            continue
        signature = shipped_speed_signature(case, decoded.hull())
        if signature["c2_over_c1_s_per_m"] > best["c2_over_c1_s_per_m"]:
            best = {**signature, "hull": decoded.hull().as_dict()}
    best["samples"] = samples
    best["seed"] = seed
    return best


def run_audit(
    *,
    samples: int = 4000,
    signature_samples: int = 20000,
    seed: int = 20261104,
    cases: Sequence[LiteratureCase] | None = None,
) -> dict[str, Any]:
    """Full identifiability report (all numbers generated, none hand-typed)."""
    case_list = list(cases) if cases is not None else load_cases()
    linearity = published_riska_linearity(case_list)
    reference_case = next(c for c in case_list if abs(c.thickness_m - 1.04) < 1e-9)
    published_signature = float(
        np.mean([g["c2_over_c1_s_per_m"] for g in linearity["groups"]])
    )
    feasible_signature = max_feasible_speed_signature(
        reference_case, samples=signature_samples, seed=seed
    )
    lindqvist_fit = best_feasible_fit(case_list, "lindqvist", samples=samples, seed=seed)
    riska_fit = best_feasible_fit(case_list, "riska", samples=samples, seed=seed)
    as_shipped = {
        "lindqvist": default_hull_errors(case_list, "lindqvist"),
        "riska": default_hull_errors(case_list, "riska"),
    }
    refuted = not (lindqvist_fit["within_tolerance"] and riska_fit["within_tolerance"])
    signature_gap = (
        published_signature / feasible_signature["c2_over_c1_s_per_m"]
        if feasible_signature["c2_over_c1_s_per_m"] > 0.0 else math.inf
    )
    return {
        "protocol": PROTOCOL,
        "tolerance_pct": TOLERANCE_PCT,
        "envelope": {k: list(v) for k, v in ENVELOPE.items()},
        "beam_length_ratio_bounds": list(BEAM_LENGTH_RATIO),
        "envelope_evidence_grade": ENVELOPE_EVIDENCE_GRADE,
        "cases": [
            {
                "test_id": c.test_id,
                "thickness_m": c.thickness_m,
                "speed_m_s": c.speed_m_s,
                "flexural_strength_pa": c.flexural_strength_pa,
                "compressive_strength_pa": c.compressive_strength_pa,
                "published_lindqvist_n": c.lindqvist_n,
                "published_riska_n": c.riska_n,
                "measured_n": c.measured_resistance_n,
            }
            for c in case_list
        ],
        "published_riska_linearity": linearity,
        "published_speed_signature_s_per_m": published_signature,
        "feasible_speed_signature": feasible_signature,
        "speed_signature_gap_factor": float(signature_gap),
        "as_shipped": as_shipped,
        "best_feasible_fit": {"lindqvist": lindqvist_fit, "riska": riska_fit},
        "verdict": {
            "published_riska_linearity": "REAL" if linearity["is_linear_within_rounding"]
            else "WITHHELD",
            "shipped_chain_as_shipped": "REAL",
            "shipped_chain_best_feasible_hull": "REAL",
            "literature_agreement_of_shipped_coefficients": (
                "REFUTED-IN-ENVELOPE" if refuted else "WITHHELD"
            ),
            "speed_term_structure": (
                "REFUTED (published relative speed dependence exceeds the shipped "
                "chain's attainable maximum by %.2f x)" % signature_gap
                if signature_gap > 1.0 else "NOT REFUTED"
            ),
        },
    }


@dataclass
class AuditSummary:
    """Compact view of a report, used by the docs build and the tests."""

    report: Mapping[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        lind = self.report["as_shipped"]["lindqvist"]["per_case_pct"]
        risk = self.report["as_shipped"]["riska"]["per_case_pct"]
        for case, a, b in zip(self.report["cases"], lind, risk):
            self.rows.append(
                {
                    "test_id": case["test_id"],
                    "thickness_m": case["thickness_m"],
                    "speed_m_s": case["speed_m_s"],
                    "published_lindqvist_kN": case["published_lindqvist_n"] / 1e3,
                    "shipped_lindqvist_err_pct": a,
                    "published_riska_kN": case["published_riska_n"] / 1e3,
                    "shipped_riska_err_pct": b,
                }
            )


def format_markdown_table(report: Mapping[str, Any]) -> str:
    """Markdown table of the as-shipped comparison (digits never hand-typed)."""
    summary = AuditSummary(report)
    lines = [
        "| test | h [m] | V [m/s] | published Lindqvist [kN] | shipped Lindqvist error | "
        "published Riska [kN] | shipped Riska error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.rows:
        lines.append(
            "| {test_id} | {thickness_m:.2f} | {speed_m_s:.1f} | "
            "{published_lindqvist_kN:.0f} | {shipped_lindqvist_err_pct:+.1f} % | "
            "{published_riska_kN:.0f} | {shipped_riska_err_pct:+.1f} % |".format(**row)
        )
    return "\n".join(lines)