"""Acceptance tests for the moist-particle (evaporation) drying closure.

Mission p380w2-n05 / m14592: extends the dry thermo-mechanical rotary-drum
closure to wet particles.  The seven machine-checkable criteria (a)..(g) are
asserted through the single entry point
:func:`tensordem.rotary_dryer.moist_drying_case.run_acceptance_suite`, i.e. the
tests and the reporting artefact are produced by the *same* code path (no
hand-copied numbers).

Resolution: profile ``test`` (dt = 0.2 s, 8 axial cells, tau = 300 s).  The
benchmark rows of the same suite carry the discretisation check: dt 0.2 vs 0.1
moves the outlet moisture by 0.1 %, so the coarse profile is adequate for the
assertions made here; the reporting profile is ``full`` (dt = 0.1, 24 cells).
"""
from __future__ import annotations

import json

import pytest

from tensordem.rotary_dryer import calculate_rotary_dryer_with_drying
from tensordem.rotary_dryer.moist_drying_case import (
    MAIN_CASE,
    PROFILES,
    THRESHOLDS,
    run_acceptance_suite,
)

#: benchmark rows that must hold (the axial grid-convergence row is reported
#: separately as a limitation of the cell discretisation, see the docs).
REQUIRED_VALIDATION = {
    "Sh(Re->0)",
    "Sh(Re=100, Sc=0.6)",
    "X_out (transient cells vs RK4 steady)",
    "T_p_out (transient cells vs RK4 steady)",
    "water ledger residual",
    "energy ledger residual",
    "degenerate limit vs dry baseline",
    "Page fit R^2",
    "Page dX/dt R^2",
}


@pytest.fixture(scope="module")
def bundle():
    return run_acceptance_suite(profile="test")


# ---------------------------------------------------------------------------
# criteria (a)..(g)
# ---------------------------------------------------------------------------
def test_all_seven_criteria_pass(bundle):
    assert bundle["criterion_pass"] == {"a": True, "b": True, "c": True,
                                        "d": True, "e": True, "f": True, "g": True}
    assert bundle["all_passed"] is True


def test_criterion_a_wet_bulb_pinning(bundle):
    a = bundle["criteria"]["a"]
    assert a["deviation_K"]["value"] < THRESHOLDS["a_pinning"]["dev_K"]
    assert a["slope_K_per_s"]["value"] < THRESHOLDS["a_pinning"]["slope_K_per_s"]
    # the pinning target is the Lewis-factor-corrected closed form, not a fit
    assert abs(a["T_p_end_K"] - a["T_wet_bulb_closures_K"]) < 0.1
    # Le != 1 here, so the chart value must be measurably different
    assert 2.0 < abs(a["lewis_induced_offset_K"]) < 5.0
    assert a["lewis_number"] > 1.0


def test_criterion_b_mass_transfer(bundle):
    b = bundle["criteria"]["b"]
    assert abs(b["Sh_stokes"]["value"]) < THRESHOLDS["b_mass_transfer"]["sh_limit_rel"]
    assert b["Sh_at_Re100"]["value"] < THRESHOLDS["b_mass_transfer"]["ranz_marshall_rel"]
    assert 99.0 < b["Re_probe"] < 101.0
    assert b["monotone_in_Re"] is True
    # independent correlations bracket Ranz-Marshall within their spread band
    assert b["Sh_whitaker_rel_dev"] < 0.15
    assert b["Sh_froessling_rel_dev"] < 0.15


def test_criterion_c_water_ledger(bundle):
    cd = bundle["criteria"]["cd"]
    assert abs(cd["water"]["value"]) < THRESHOLDS["c_water_ledger"]["residual_rel"]
    assert cd["water_evaporated_kg"] > 0.0


def test_criterion_d_energy_ledger_and_negative_control(bundle):
    cd = bundle["criteria"]["cd"]
    assert abs(cd["energy"]["value"]) < THRESHOLDS["d_energy_ledger"]["residual_rel"]
    nc = cd["negative_control"]
    assert nc["ok"] is True, "energy ledger check has no teeth"
    assert abs(nc["latent_sink_disabled_energy_residual_rel"]) > 0.1


def test_criterion_e_thin_layer_kinetics(bundle):
    e = bundle["criteria"]["e"]
    assert e["r2_moisture"]["value"] > THRESHOLDS["e_kinetics"]["r2"]
    assert e["r2_rate"]["value"] > THRESHOLDS["e_kinetics"]["r2"]
    assert 0.0 < e["page"]["n"] < 5.0


def test_criterion_f_degenerate_limit(bundle):
    f = bundle["criteria"]["f"]
    assert f["delta"]["value"] < THRESHOLDS["f_degenerate"]["delta"]
    assert f["residual_moisture_kg"] == pytest.approx(0.0, abs=1e-15)


def test_criterion_g_monotonicity(bundle):
    g = bundle["criteria"]["g"]
    assert g["n_points_min"] >= THRESHOLDS["g_monotonicity"]["n_points"]
    assert g["n_flips_total"] == THRESHOLDS["g_monotonicity"]["n_flips"]
    for name, sweep in g["sweeps"].items():
        values = sweep["X_out"]
        assert values == sorted(values, reverse=(sweep["expected_direction"] == "decreasing")), name


# ---------------------------------------------------------------------------
# reference-vs-computed benchmark table
# ---------------------------------------------------------------------------
def test_benchmark_required_rows_pass(bundle):
    rows = {r["quantity"]: r for r in bundle["benchmark"]}
    for quantity in REQUIRED_VALIDATION:
        assert quantity in rows, f"missing benchmark row {quantity}"
        assert rows[quantity]["passed"], (quantity, rows[quantity])
        assert rows[quantity]["reference_source"], quantity


def test_benchmark_row_schema_and_serialisability(bundle):
    assert bundle["benchmark"]
    for row in bundle["benchmark"]:
        assert set(row) >= {"quantity", "reference", "computed", "unit", "error",
                            "tolerance", "reference_source", "passed"}
        assert isinstance(row["passed"], bool)
    json.dumps(bundle)  # the bundle must be a plain-JSON artefact


def test_falsified_claim_row_is_documented(bundle):
    """The old '0.05 K' Le=1 coincidence claim is measurement-refuted."""
    claims = {r["quantity"]: r for r in bundle["benchmark"]
              if r.get("class") == "falsified_claim"}
    assert claims, "the refuted docstring claim must stay in the audit table"
    row = next(iter(claims.values()))
    assert row["passed"] is False
    assert 1.0 < row["error"] < 3.0
    assert "film" in row["note"]


# ---------------------------------------------------------------------------
# integration caller
# ---------------------------------------------------------------------------
def test_integration_caller_returns_drying_and_drum_faces():
    out = calculate_rotary_dryer_with_drying(
        profile={"n_cells": 6, "dt": 0.5, "t_end": 120.0, "residence_time": 120.0})
    assert set(out) >= {"drum", "drying", "wet_bulb", "ledger", "evidence"}
    assert out["drying"]["moisture_dry_basis"] < MAIN_CASE["initial_moisture"]
    assert out["drying"]["evaporation_rate_kg_per_s"] > 0.0
    assert out["drying"]["latent_duty_W"] > 0.0
    assert abs(out["ledger"]["water_residual_rel"]) < 1e-9
    assert abs(out["ledger"]["energy_residual_rel"]) < 1e-2
    assert out["evidence"]["drying"] == "REAL"
    assert "kinematics" in out["drum"] and "rtd" in out["drum"]


def test_integration_caller_rejects_unknown_profile_keys():
    with pytest.raises(ValueError):
        calculate_rotary_dryer_with_drying(profile={"bogus": 1})


def test_profile_table_is_consistent():
    for name, prof in PROFILES.items():
        assert prof["residence_time"] > 0
        assert prof["bench_factor"] >= 1.0
        assert prof["n_cells"] >= 1
        assert prof["dt"] > 0, name
