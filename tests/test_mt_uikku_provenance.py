"""Provenance / evidence-integrity invariants for the M4 MT Uikku surface.

These tests exist because the previous revision of
``tensordem.ice_broken_field.mt_uikku_validation`` reported a *passing*
time-history correlation criterion from the hard-coded constant ``0.85`` and ran
the experimental cases at 3/5/7 kn while the anchor table records an ice drift
speed of 0.2/0.5/1.0 m/s.

Each test therefore asserts an invariant (data traceability, statistic
recomputability, flag/number consistency), not merely that the code runs.
"""
from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.ice_broken_field import (
    MT_UIKKU_CSV,
    MT_UIKKU_CASES,
    load_mt_uikku_cases,
    negative_control_zero_thickness,
    score_cases,
    verify_mt_uikku,
)
from tensordem.ice_broken_field import mt_uikku_validation as module

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / MT_UIKKU_CSV


def _csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows(dem: list[float], measured: list[float], peaks=None, measured_peaks=None):
    peaks = peaks or [0.0] * len(dem)
    measured_peaks = measured_peaks or [0.0] * len(dem)
    return [{"case_id": str(i), "F_mean_kN": d, "measured_mean_kN": m,
             "F_peak_kN": p, "measured_max_kN": mp}
            for i, (d, m, p, mp) in enumerate(zip(dem, measured, peaks, measured_peaks))]


# --------------------------------------------------------------------------- #
# 1. provenance: the case table *is* the anchor CSV
# --------------------------------------------------------------------------- #
def test_cases_are_traced_to_the_anchor_csv():
    rows = _csv_rows()
    cases = load_mt_uikku_cases()
    assert len(cases) == len(rows) == 10
    by_id = {row["test_id"].strip(): row for row in rows}
    for case in cases:
        row = by_id[case.case_id]
        assert case.provenance.endswith("mt_uikku_level_ice.csv")
        assert case.ice_drift_speed_ms == pytest.approx(float(row["ice_drift_speed_m_s"]))
        assert case.ice_thickness_m == pytest.approx(float(row["ice_thickness_m"]))
        assert case.compressive_strength_kPa == pytest.approx(
            float(row["compressive_strength_kPa"]))
        assert case.measured_mean_kN == pytest.approx(
            float(row["measured_mean_resistance_table3_kN"]))
        # empty peak cells stay "no value", they are never estimated
        cell = row["measured_max_kN"].strip()
        assert case.measured_max_kN == (float(cell) if cell else 0.0)


def test_speed_is_the_tabulated_drift_speed_not_knots():
    """The defect: 3/5/7 kn = 1.54/2.57/3.60 m/s, 3-7x the drift speed."""
    legacy_ms = {3.0 * 0.5144, 5.0 * 0.5144, 7.0 * 0.5144}
    for case in MT_UIKKU_CASES:
        assert case.ice_drift_speed_ms <= 1.0 + 1e-12, (
            f"case {case.case_id} runs at {case.ice_drift_speed_ms} m/s; the "
            "experimental table records 0.2/0.5/1.0 m/s")
        assert not any(math.isclose(case.ice_drift_speed_ms, v, rel_tol=1e-6)
                       for v in legacy_ms)
        assert math.isclose(case.ship_speed_kn, case.ice_drift_speed_ms / 0.5144)


def test_fallback_table_is_flagged():
    """If the CSV is unreachable the fallback must be flagged, not silent."""
    cases = load_mt_uikku_cases(path="/nonexistent/mt_uikku.csv")
    assert cases and all(c.provenance == "FALLBACK_EMBEDDED" for c in cases)
    assert {c.case_id for c in cases} == {c.case_id for c in MT_UIKKU_CASES}


# --------------------------------------------------------------------------- #
# 2. the correlation is a statistic, not a constant
# --------------------------------------------------------------------------- #
def test_correlation_is_recomputed_from_the_data():
    dem = [100.0, 220.0, 310.0, 430.0]
    measured = [110.0, 200.0, 330.0, 410.0]
    scored = score_cases(_rows(dem, measured))
    expected = float(np.corrcoef(dem, measured)[0, 1])
    assert scored["correlation_across_cases_pearson"] == pytest.approx(expected, abs=1e-12)
    assert scored["correlation_across_cases_spearman"] == pytest.approx(1.0)

    # a data perturbation must move the statistic (a constant would not)
    flipped = score_cases(_rows(dem, measured[::-1]))
    assert flipped["correlation_across_cases_pearson"] != pytest.approx(expected)


def test_no_constant_stands_in_for_a_statistic():
    """Regression guard: the fabricated ``correlations.append(0.85)`` is gone."""
    source = inspect.getsource(module)
    assert "correlations.append" not in source
    assert "placeholder (would need actual time series)" not in source
    scored = score_cases(_rows([1.0, 2.0], [1.0, 2.0]))
    assert "correlation_mean" not in scored, (
        "the ambiguous key that used to carry the constant must not come back")


def test_time_history_correlation_is_withheld_not_invented():
    scored = score_cases(_rows([100.0, 200.0], [110.0, 190.0]))
    withheld = scored["correlation_time_series"]
    assert withheld["verdict"] == "WITHHELD"
    assert withheld["value"] is None
    assert "time series" in withheld["reason"]
    assert "correlation_basis" in scored and "across-case" in scored["correlation_basis"]


# --------------------------------------------------------------------------- #
# 3. criterion flags are derived from the numbers, in both directions
# --------------------------------------------------------------------------- #
def test_flags_follow_the_numbers_both_ways():
    measured = [500.0, 600.0, 700.0, 400.0]
    inside = [510.0, 580.0, 720.0, 410.0]          # ~4 % mean error
    scored_in = score_cases(_rows(inside, measured))
    assert scored_in["F_mean_mean_error"] < 0.20
    assert scored_in["F_mean_criterion_pass"] is True
    assert scored_in["correlation_criterion_pass"] is True

    outside = [x * 1.6 for x in measured]          # 60 % mean error
    scored_out = score_cases(_rows(outside, measured))
    assert scored_out["F_mean_mean_error"] == pytest.approx(0.6, abs=1e-9)
    assert scored_out["F_mean_criterion_pass"] is False


def test_peak_criterion_ignores_cases_without_a_measured_peak():
    rows = _rows([100.0, 200.0], [100.0, 200.0],
                 peaks=[150.0, 260.0], measured_peaks=[0.0, 240.0])
    scored = score_cases(rows)
    assert scored["n_scored_peak"] == 1               # the 401-403 situation
    assert scored["F_peak_mean_error"] == pytest.approx(abs(260.0 - 240.0) / 240.0)


# --------------------------------------------------------------------------- #
# 4. steady state / negative control contracts
# --------------------------------------------------------------------------- #
def test_steady_state_reachability_is_declared():
    case = MT_UIKKU_CASES[0]
    transient = case.steady_state_reach(60.0)
    assert transient["verdict"] == "WITHHELD"
    assert transient["travel_in_hull_lengths"] < 1.0
    assert "transient" in transient["reason"]

    reached = case.steady_state_reach(60.0 / case.ice_drift_speed_ms * 200.0)
    assert reached["verdict"] == "PASS"


def test_negative_control_zero_thickness():
    out = negative_control_zero_thickness(dt=0.05, t_total=1.0)
    assert out["negative_control_pass"] is True
    assert out["F_zero_thickness_N"] < 1.0


def test_verify_contract_on_a_short_run():
    """Structural contract only: one case, 2 s, coarse dt (fast)."""
    result = verify_mt_uikku(cases=[MT_UIKKU_CASES[0]], dt=0.05, t_total=2.0)
    assert result["overall_verdict"] == "WITHHELD"
    assert result["provenance"]["speed_field"] == "ice_drift_speed_m_s"
    assert result["correlation_time_series"]["verdict"] == "WITHHELD"
    # flags describe the numbers, whatever those numbers are
    err = result["F_mean_mean_error"]
    assert result["F_mean_criterion_pass"] == (err <= 0.20)
    for case_row in result["case_results"]:
        assert not math.isnan(case_row["F_mean_error"])