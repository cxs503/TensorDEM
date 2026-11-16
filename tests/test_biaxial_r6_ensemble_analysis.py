"""Assembler-level tests for the R6 ensemble of CASE-BIAXIAL-SHEAR-BAND.

The R6 driver (:mod:`benchmarks.biaxial_shear_band.run_r6_peak_relative_ensemble`)
re-runs the eight R5 anchor specimens with the peak-relative stopping rule and
compares them *paired* against the R5 baseline.  The DEM half of that run is
expensive, so the paired-comparison / stopping-summary / prediction logic is
exercised here on synthetic records with the same nesting as a real specimen
record (``cumulative -> late_window -> {radon, strain_span, resolved}`` plus
``n_late_snapshots`` and ``stopping``).

What is asserted (invariants, not just execution):

* a specimen that failed to resolve in R5 (empty late window) and resolves in
  R6 is counted as *newly resolved*, and the reverse case as *lost resolution*;
* the predeclared prediction P-R6 is only ``met`` when **all three** dead-window
  specimens acquire a span >= 0.02 -- one straggler falsifies it;
* a specimen that raises is carried as an error row and never silently dropped
  from the paired table;
* the stopping summary distinguishes the two stop reasons and reports the
  final-strain range.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.biaxial_shear_band.run_r6_peak_relative_ensemble import (  # noqa: E402
    P_R6_MIN_SPAN,
    R5_DEAD_WINDOW,
    _paired,
    _stopping_summary,
)

SEEDS = [53, 67, 79, 83, 89, 97, 101, 103]


def _cum(theta, span, resolved):
    return {"late_window": {"radon": theta, "strain_span": span,
                            "resolved": bool(resolved)}}


def _r6(name, seed, theta, span, resolved, stop="post_peak", final_log=0.15):
    return {"name": name, "seed": seed, "cumulative": _cum(theta, span, resolved),
            "n_late_snapshots": int(round(span / 0.01)) + 1 if span else 1,
            "stopping": {"stop_reason": stop,
                         "final_axial_strain_log": final_log}}


def _r5(name, seed, theta, span, resolved):
    return {"name": name, "seed": seed, "cumulative": _cum(theta, span, resolved),
            "n_late_snapshots": int(round(span / 0.01)) + 1 if span else 1,
            "final_axial_strain": 0.12}


#: R5 (the R4 artefact) -- 5/8 resolved, s89/s97/s103 have an empty late window.
R5_ROWS = [
    _r5("anchor_s53", 53, 52.0, 0.02, True),
    _r5("anchor_s67", 67, 55.0, 0.03, True),
    _r5("anchor_s79", 79, 56.0, 0.03, True),
    _r5("anchor_s83", 83, 55.0, 0.01, True),
    _r5("anchor_s89", 89, None, 0.0, False),
    _r5("anchor_s97", 97, None, 0.0, False),
    _r5("anchor_s101", 101, 57.0, 0.01, True),
    _r5("anchor_s103", 103, None, 0.0, False),
]

BASELINE = {"specimens": R5_ROWS}


def _r6_full(acquired_span=0.03):
    rows = []
    for seed in SEEDS:
        name = f"anchor_s{seed}"
        old = {r["name"]: r for r in R5_ROWS}[name]
        span = acquired_span if name in R5_DEAD_WINDOW else 0.03
        rows.append(_r6(name, seed, 55.0, span, True))
    return rows


def test_newly_resolved_is_counted_and_prediction_met():
    paired = _paired(_r6_full(), BASELINE)
    assert paired["n_newly_resolved"] == 3
    assert sorted(paired["newly_resolved"]) == sorted(R5_DEAD_WINDOW)
    assert paired["n_lost_resolution"] == 0
    prediction = paired["prediction_P_R6"]
    assert prediction["n_acquired"] == 3
    assert prediction["prediction_met"] is True
    assert all(row["acquired"] for row in prediction["acquired_window"])
    assert all(row["r6_span"] >= P_R6_MIN_SPAN for row in prediction["acquired_window"])


def test_one_straggler_falsifies_the_prediction():
    rows = _r6_full()
    for rec in rows:
        if rec["name"] == "anchor_s97":
            rec["cumulative"] = _cum(55.0, 0.01, True)   # window still too short
    prediction = _paired(rows, BASELINE)["prediction_P_R6"]
    assert prediction["n_acquired"] == 2
    assert prediction["prediction_met"] is False


def test_lost_resolution_is_reported_not_hidden():
    rows = _r6_full()
    for rec in rows:
        if rec["name"] == "anchor_s53":
            rec["cumulative"] = _cum(None, 0.0, False)
    paired = _paired(rows, BASELINE)
    assert paired["lost_resolution"] == ["anchor_s53"]
    assert paired["n_lost_resolution"] == 1


def test_error_row_is_carried_through():
    rows = _r6_full()
    rows[2] = {"name": "anchor_s79", "seed": 79, "error": "boom"}
    paired = _paired(rows, BASELINE)
    row = {r["name"]: r for r in paired["per_specimen"]}["anchor_s79"]
    assert row["error"] == "boom"
    assert "r6_resolved" not in row


def test_paired_rows_are_sorted_by_seed_and_keep_both_angles():
    paired = _paired(_r6_full(), BASELINE)
    seeds = [r["seed"] for r in paired["per_specimen"]]
    assert seeds == sorted(SEEDS)
    first = paired["per_specimen"][0]
    assert first["r5_theta_deg"] == pytest.approx(52.0)
    assert first["r6_theta_deg"] == pytest.approx(55.0)
    assert first["r6_stop_reason"] == "post_peak"


def test_stopping_summary_splits_reasons_and_reports_range():
    rows = _r6_full()
    rows[0]["stopping"] = {"stop_reason": "strain_cap",
                           "final_axial_strain_log": 0.18}
    summary = _stopping_summary(rows)
    assert summary["stop_reasons"]["strain_cap"]["n"] == 1
    assert summary["stop_reasons"]["post_peak"]["n"] == 7
    assert summary["final_strain_max"] == pytest.approx(0.18)
    assert summary["final_strain_min"] == pytest.approx(0.15)


def test_stopping_summary_survives_all_error_rows():
    summary = _stopping_summary([{"name": "x", "error": "boom"}])
    assert summary["stop_reasons"] == {}
    assert summary["final_strain_mean"] is None