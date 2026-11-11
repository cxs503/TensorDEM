"""M4: MT Uikku comparison surface — provenance-corrected, honestly scored.

Audit of 2026-11-11 (this revision)
-----------------------------------
Three defects were found in the previous revision of this module and are fixed
here; the correction is *not* a calibration, it is a provenance and
evidence-integrity repair.

1. **Speed provenance.** The previous revision declared ``ship_speed_kn`` of
   3 / 5 / 7 kn (1.54 / 2.57 / 3.60 m/s) and used them as the ship speed.  The
   measured quantity in the experimental table is the *ice drift speed*
   ``ice_drift_speed_m_s`` = 0.2 / 0.5 / 1.0 m/s
   (``benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv``).  The
   DEM was therefore run 3-7x too fast.  Speeds are now read from the CSV,
   which is the single source of truth; the knot values exist only as a
   derived display quantity.
2. **Fabricated correlation.** ``verify_mt_uikku`` used to append the constant
   ``0.85`` (annotated as a placeholder that would need an actual time
   series) and then report
   it as the ``rho >= 0.7`` criterion result.  A constant is not a statistic:
   the time-history correlation is now declared ``WITHHELD`` (the repository
   ships no measured force time series) and the only correlation reported is a
   *data-driven* across-case Pearson/Spearman coefficient between the DEM and
   measured mean resistances, labelled as such.
3. **Regime confound (pre-existing, now declared).** The ten shipped MT Uikku
   cases are *level-ice* runs; the DEM field used here is a *broken-ice* field
   of concentration 0.6-0.9.  The maintained chain
   (``tensordem.broken_ice_resistance.validation``, criterion C4) declares the
   same score ``WITHHELD`` for exactly this reason, so M4 reports
   ``overall_verdict = "WITHHELD"`` even where a numeric error is computed.

The ``F_mean`` / ``F_peak`` errors are still computed and reported per case, as
an honest diagnostic, with the criteria flags derived from the numbers (never
asserted).

References:
- Riska, K. et al. (1998): "MT Uikku full-scale ice trials"
- Su, B. et al. (2010): "Ice resistance prediction methods"
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

#: Repository-relative path of the experimental anchor table (single source).
MT_UIKKU_CSV = ("benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv")

#: Hull length of the DEM stand-in hull (``3 B`` parallel body + bow wedge of
#: ``B / (2 tan 18 deg)``), used for the steady-state reachability test.
DEM_HULL_LENGTH_M = 3.0 * 26.0 + 0.5 * 26.0 / math.tan(math.radians(18.0))

#: Number of hull lengths the ship must travel before the last-half mean may be
#: called a steady-state statistic.
REQUIRED_TRAVEL_MULTIPLE = 3.0

#: Ice concentration assumed per MT Uikku ice sheet class in the legacy DEM
#: configuration.  The experimental table has *no* concentration column (the
#: tests are level ice, i.e. C = 1.0), so these values are ASSUMED and are one
#: of the reasons the M4 score is WITHHELD.
ASSUMED_CONCENTRATION_BY_SHEET = {"I": 0.9, "II": 0.8, "III": 0.7, "IV": 0.6}


@dataclass
class MTUikkuCase:
    """MT Uikku test case specification (fields traced to the anchor CSV)."""

    case_id: str
    ice_drift_speed_ms: float        # m/s, CSV field ``ice_drift_speed_m_s``
    ice_thickness_m: float           # m
    flexural_strength_kPa: float     # kPa
    compressive_strength_kPa: float  # kPa
    measured_mean_kN: float          # kN
    measured_std_kN: float           # kN
    measured_max_kN: float           # kN (0.0 when the table has no value)
    ice_sheet: str = ""              # CSV field ``ice_sheet`` (I-IV)
    ice_type: str = "level"          # CSV field ``ice_type``
    ice_elastic_modulus_MPa: float = 0.0
    concentration: float = 1.0       # ASSUMED, see ASSUMED_CONCENTRATION_BY_SHEET
    provenance: str = MT_UIKKU_CSV

    @property
    def ship_speed_ms(self) -> float:
        """Relative ship-ice speed, m/s (alias kept for API compatibility)."""
        return self.ice_drift_speed_ms

    @property
    def ship_speed_kn(self) -> float:
        """Derived display quantity, kn.  NOT an input of the experimental table."""
        return self.ice_drift_speed_ms / 0.5144

    def steady_state_reach(self, t_total: float) -> dict:
        """Can ``t_total`` seconds at this speed resolve a steady-state mean?"""
        travel = self.ship_speed_ms * t_total
        required = REQUIRED_TRAVEL_MULTIPLE * DEM_HULL_LENGTH_M
        ok = travel >= required
        return {
            "speed_m_s": self.ship_speed_ms,
            "t_total_s": t_total,
            "travel_m": travel,
            "hull_length_m": DEM_HULL_LENGTH_M,
            "required_travel_m": required,
            "travel_in_hull_lengths": travel / DEM_HULL_LENGTH_M,
            "verdict": "PASS" if ok else "WITHHELD",
            "reason": (None if ok else
                       f"ship travels {travel:.1f} m = "
                       f"{travel / DEM_HULL_LENGTH_M:.2f} hull lengths in "
                       f"{t_total:.0f} s; <{REQUIRED_TRAVEL_MULTIPLE:.0f} hull "
                       "lengths of swept path means the last-half mean is still "
                       "a start-up transient, not a steady-state resistance"),
        }


#: Embedded fallback, byte-identical in *meaning* to the CSV (speeds are the
#: tabulated drift speeds, not knots).  Used only when the CSV is unreachable,
#: and flagged as such through ``provenance``.
_FALLBACK_ROWS: list[dict[str, Any]] = [
    dict(test_id="103", ice_sheet="I", ice_thickness_m=0.77, flexural_strength_kPa=724,
         compressive_strength_kPa=1748, ice_elastic_modulus_MPa=929, ice_drift_speed_m_s=0.2,
         measured_mean_resistance_table3_kN=480, measured_std_kN=130, measured_max_kN=920),
    dict(test_id="104", ice_sheet="I", ice_thickness_m=0.76, flexural_strength_kPa=844,
         compressive_strength_kPa=2192, ice_elastic_modulus_MPa=984, ice_drift_speed_m_s=0.5,
         measured_mean_resistance_table3_kN=560, measured_std_kN=130, measured_max_kN=990),
    dict(test_id="205", ice_sheet="II", ice_thickness_m=0.96, flexural_strength_kPa=920,
         compressive_strength_kPa=1840, ice_elastic_modulus_MPa=1685, ice_drift_speed_m_s=0.2,
         measured_mean_resistance_table3_kN=670, measured_std_kN=220, measured_max_kN=1410),
    dict(test_id="206", ice_sheet="II", ice_thickness_m=0.95, flexural_strength_kPa=912,
         compressive_strength_kPa=1862, ice_elastic_modulus_MPa=1701, ice_drift_speed_m_s=0.5,
         measured_mean_resistance_table3_kN=720, measured_std_kN=193, measured_max_kN=1390),
    dict(test_id="301", ice_sheet="III", ice_thickness_m=1.04, flexural_strength_kPa=540,
         compressive_strength_kPa=2477, ice_elastic_modulus_MPa=1474, ice_drift_speed_m_s=0.2,
         measured_mean_resistance_table3_kN=540, measured_std_kN=88, measured_max_kN=850),
    dict(test_id="302", ice_sheet="III", ice_thickness_m=1.04, flexural_strength_kPa=669,
         compressive_strength_kPa=2485, ice_elastic_modulus_MPa=1273, ice_drift_speed_m_s=0.5,
         measured_mean_resistance_table3_kN=830, measured_std_kN=130, measured_max_kN=1200),
    dict(test_id="303", ice_sheet="III", ice_thickness_m=1.04, flexural_strength_kPa=592,
         compressive_strength_kPa=2397, ice_elastic_modulus_MPa=1390, ice_drift_speed_m_s=1.0,
         measured_mean_resistance_table3_kN=780, measured_std_kN=160, measured_max_kN=1260),
    dict(test_id="401", ice_sheet="IV", ice_thickness_m=0.63, flexural_strength_kPa=808,
         compressive_strength_kPa=4040, ice_elastic_modulus_MPa=1616, ice_drift_speed_m_s=0.2,
         measured_mean_resistance_table3_kN=150, measured_std_kN=0, measured_max_kN=0),
    dict(test_id="402", ice_sheet="IV", ice_thickness_m=0.63, flexural_strength_kPa=1029,
         compressive_strength_kPa=5389, ice_elastic_modulus_MPa=2058, ice_drift_speed_m_s=0.5,
         measured_mean_resistance_table3_kN=290, measured_std_kN=0, measured_max_kN=0),
    dict(test_id="403", ice_sheet="IV", ice_thickness_m=0.63, flexural_strength_kPa=903,
         compressive_strength_kPa=4616, ice_elastic_modulus_MPa=1805, ice_drift_speed_m_s=1.0,
         measured_mean_resistance_table3_kN=390, measured_std_kN=0, measured_max_kN=0),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _opt_float(value: Any) -> float:
    """CSV cell -> float, empty cell -> 0.0 (i.e. "the table has no value")."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def _row_to_case(row: dict[str, Any], provenance: str) -> MTUikkuCase:
    sheet = str(row.get("ice_sheet", "")).strip()
    return MTUikkuCase(
        case_id=str(row["test_id"]).strip(),
        ice_drift_speed_ms=float(row["ice_drift_speed_m_s"]),
        ice_thickness_m=float(row["ice_thickness_m"]),
        flexural_strength_kPa=float(row["flexural_strength_kPa"]),
        compressive_strength_kPa=float(row["compressive_strength_kPa"]),
        measured_mean_kN=float(row["measured_mean_resistance_table3_kN"]),
        measured_std_kN=_opt_float(row.get("measured_std_kN")),
        measured_max_kN=_opt_float(row.get("measured_max_kN")),
        ice_sheet=sheet,
        ice_type=str(row.get("ice_type", "level")).strip() or "level",
        ice_elastic_modulus_MPa=_opt_float(row.get("ice_elastic_modulus_MPa")),
        concentration=ASSUMED_CONCENTRATION_BY_SHEET.get(sheet, 1.0),
        provenance=provenance,
    )


def load_mt_uikku_cases(path: str | Path | None = None) -> list[MTUikkuCase]:
    """Load the ten level-ice cases from the anchor CSV (single source of truth).

    Falls back to the embedded rows only when the CSV is missing; the fallback
    is flagged through ``MTUikkuCase.provenance`` so a report can never mistake
    one for the other.
    """
    csv_path = Path(path) if path is not None else _repo_root() / MT_UIKKU_CSV
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [_row_to_case(row, str(csv_path)) for row in rows]
    return [_row_to_case(row, "FALLBACK_EMBEDDED") for row in _FALLBACK_ROWS]


#: Module-level case table (CSV-backed whenever the repository is present).
MT_UIKKU_CASES: list[MTUikkuCase] = load_mt_uikku_cases()


@dataclass
class MTUikkuResult:
    """Result of one MT Uikku case simulation."""

    case_id: str
    time: np.ndarray           # s
    F_ice: np.ndarray          # ice resistance, N
    F_mean_N: float            # mean resistance, N
    F_std_N: float             # std resistance, N
    F_peak_N: float            # peak resistance, N
    measured_mean_N: float     # measured mean, N
    measured_max_N: float      # measured max, N (0.0 when unavailable)
    speed_ms: float = 0.0      # simulated relative speed, m/s
    t_total_s: float = 0.0     # simulated duration, s


def run_mt_uikku_case(
    case: MTUikkuCase,
    dt: float = 0.01,
    t_total: float = 60.0,
) -> MTUikkuResult:
    """Run one MT Uikku case through the M3 broken-ice field demo."""
    from .channel_clearing import BrokenIceFieldConfig, broken_ice_field_simulation

    config = BrokenIceFieldConfig(
        domain_length=500.0,
        domain_width=100.0,
        concentration=case.concentration,
        mean_floe_radius=3.0,
        radius_std=1.0,
        ice_thickness=case.ice_thickness_m,
        compressive_strength=case.compressive_strength_kPa * 1000.0,  # kPa -> Pa
        friction_coefficient=0.1,
        ship_speed=case.ship_speed_ms,
        ship_beam=26.0,
        ship_entry_angle=18.0,
        ship_bow_radius=15.0,
        seed=42,
    )

    result = broken_ice_field_simulation(config, dt=dt, t_total=t_total)

    return MTUikkuResult(
        case_id=case.case_id,
        time=result.time,
        F_ice=result.R_ice,
        F_mean_N=result.R_ice_mean,
        F_std_N=result.R_ice_std,
        F_peak_N=result.R_ice_peak,
        measured_mean_N=case.measured_mean_kN * 1000.0,
        measured_max_N=case.measured_max_kN * 1000.0,
        speed_ms=case.ship_speed_ms,
        t_total_s=t_total,
    )


def run_all_mt_uikku_cases(
    cases: Sequence[MTUikkuCase] | None = None,
    dt: float = 0.01,
    t_total: float = 60.0,
) -> list[MTUikkuResult]:
    """Run every MT Uikku case (defaults to the CSV-backed table)."""
    table = list(cases) if cases is not None else MT_UIKKU_CASES
    return [run_mt_uikku_case(case, dt=dt, t_total=t_total) for case in table]


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson product-moment coefficient; NaN when undefined."""
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0.0 else float("nan")


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank coefficient (tie-aware)."""
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    return pearson(_rank(x), _rank(y))


def score_cases(case_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pure scoring function: rows -> criteria, flags and honest verdicts.

    Each row needs ``F_mean_kN``, ``measured_mean_kN``, ``F_peak_kN`` and
    ``measured_max_kN`` (0.0 = the table has no peak value).  No constant is
    ever substituted for a statistic here.
    """
    rows = list(case_rows)
    mean_errors: list[float] = []
    peak_errors: list[float] = []
    scored_mean: list[tuple[float, float]] = []
    scored_peak: list[tuple[float, float]] = []
    per_case: list[dict[str, Any]] = []

    for row in rows:
        f_mean = float(row["F_mean_kN"])
        m_mean = float(row["measured_mean_kN"])
        f_peak = float(row["F_peak_kN"])
        m_peak = float(row.get("measured_max_kN") or 0.0)

        mean_error = abs(f_mean - m_mean) / m_mean if m_mean > 0 else float("nan")
        peak_error = abs(f_peak - m_peak) / m_peak if m_peak > 0 else float("nan")
        if m_mean > 0:
            mean_errors.append(mean_error)
            scored_mean.append((f_mean, m_mean))
        if m_peak > 0:
            peak_errors.append(peak_error)
            scored_peak.append((f_peak, m_peak))

        per_case.append({
            **row,
            "F_mean_error": mean_error,
            "F_peak_error": peak_error,
        })

    f_mean_mean_error = float(np.mean(mean_errors)) if mean_errors else float("nan")
    f_peak_mean_error = float(np.mean(peak_errors)) if peak_errors else float("nan")
    r_mean = pearson([a for a, _ in scored_mean], [b for _, b in scored_mean])
    rho_mean = spearman([a for a, _ in scored_mean], [b for _, b in scored_mean])

    mean_ok = bool(mean_errors) and f_mean_mean_error <= 0.20
    peak_ok = bool(peak_errors) and f_peak_mean_error <= 0.25
    corr_ok = bool(mean_errors) and r_mean >= 0.70

    return {
        "n_cases": len(rows),
        "n_scored_mean": len(mean_errors),
        "n_scored_peak": len(peak_errors),
        "F_mean_mean_error": f_mean_mean_error,
        "F_peak_mean_error": f_peak_mean_error,
        "F_mean_threshold": 0.20,
        "F_peak_threshold": 0.25,
        # Flags are *derived* from the numbers, never asserted.
        "F_mean_criterion_pass": mean_ok,
        "F_peak_criterion_pass": peak_ok,
        "correlation_threshold": 0.70,
        "correlation_across_cases_pearson": r_mean,
        "correlation_across_cases_spearman": rho_mean,
        "correlation_basis": ("across-case Pearson/Spearman of DEM vs measured "
                              "*mean* resistance over the scored cases"),
        "correlation_criterion_pass": corr_ok,
        "correlation_time_series": {
            "verdict": "WITHHELD",
            "value": None,
            "reason": ("no measured force time series is available: "
                       f"{MT_UIKKU_CSV} ships table-3 statistics (mean/std/max) "
                       "only, so the rho >= 0.7 time-history criterion cannot be "
                       "evaluated. A constant must not stand in for it."),
        },
        "case_results": per_case,
    }


def negative_control_zero_thickness(dt: float = 0.01, t_total: float = 10.0) -> dict:
    """Negative control: h_i = 0 must give R_ice = 0."""
    from .channel_clearing import BrokenIceFieldConfig, broken_ice_field_simulation

    config_zero = BrokenIceFieldConfig(ice_thickness=0.0, concentration=0.5, seed=42)
    result_zero = broken_ice_field_simulation(config_zero, dt=dt, t_total=t_total)
    return {
        "F_zero_thickness_N": result_zero.R_ice_mean,
        "negative_control_pass": bool(result_zero.R_ice_mean < 1.0),
    }


def verify_mt_uikku(
    cases: Sequence[MTUikkuCase] | None = None,
    dt: float = 0.01,
    t_total: float = 60.0,
) -> dict:
    """Score the DEM broken-ice field against the MT Uikku level-ice table.

    Returns the numeric errors (diagnostic, honestly computed), the data-driven
    across-case correlation, an explicit ``WITHHELD`` verdict for the
    time-history correlation, and an ``overall_verdict`` that is ``WITHHELD``
    because the comparison confounds regime (broken-ice field vs level-ice
    measurement) and, at the tabulated drift speeds, does not reach steady
    state within ``t_total``.
    """
    table = list(cases) if cases is not None else MT_UIKKU_CASES
    results = run_all_mt_uikku_cases(table, dt=dt, t_total=t_total)

    rows = [{
        "case_id": res.case_id,
        "speed_kn": case.ship_speed_kn,
        "speed_m_s": case.ship_speed_ms,
        "ice_sheet": case.ice_sheet,
        "thickness_m": case.ice_thickness_m,
        "concentration_assumed": case.concentration,
        "F_mean_kN": res.F_mean_N / 1000.0,
        "F_peak_kN": res.F_peak_N / 1000.0,
        "F_mean_DEM_kN": res.F_mean_N / 1000.0,
        "F_mean_measured_kN": res.measured_mean_N / 1000.0,
        "F_peak_DEM_kN": res.F_peak_N / 1000.0,
        "measured_max_kN": res.measured_max_N / 1000.0,
        "measured_mean_kN": res.measured_mean_N / 1000.0,
    } for res, case in zip(results, table)]

    scored = score_cases(rows)
    negative = negative_control_zero_thickness()

    reach = [case.steady_state_reach(t_total) for case in table]
    steady_ok = all(item["verdict"] == "PASS" for item in reach)

    reasons = ["regime confound: the ten shipped MT Uikku cases are level-ice "
               "runs (C = 1.0) while the M3 field is broken ice at C = 0.6-0.9; "
               "the maintained chain (broken_ice_resistance C4) declares the same "
               "score WITHHELD"]
    if not steady_ok:
        worst = min(item["travel_in_hull_lengths"] for item in reach)
        reasons.append(f"steady state not reached at the tabulated drift speeds: "
                       f"worst case sweeps {worst:.2f} hull lengths < "
                       f"{REQUIRED_TRAVEL_MULTIPLE:.0f} in {t_total:.0f} s")
    if not scored["F_mean_criterion_pass"]:
        reasons.append("F_mean criterion not met: mean relative error "
                       f"{scored['F_mean_mean_error']:.1%} > 20%")
    reasons.append("time-history correlation WITHHELD (no measured time series)")

    return {
        **scored,
        **negative,
        "overall_verdict": "WITHHELD",
        "verdict_reasons": reasons,
        "provenance": {
            "case_source": table[0].provenance if table else None,
            "speed_field": "ice_drift_speed_m_s",
            "speed_note": ("previous revision used 3/5/7 kn (1.54/2.57/3.60 m/s), "
                           "3-7x the tabulated drift speed"),
            "concentration_note": ("no concentration column in the experimental "
                                   "table; 0.9/0.8/0.7/0.6 per ice sheet class are "
                                   "ASSUMED for the legacy DEM configuration"),
        },
        "steady_state": {"t_total_s": t_total, "all_pass": steady_ok,
                         "required_travel_multiple": REQUIRED_TRAVEL_MULTIPLE,
                         "per_case": reach},
    }