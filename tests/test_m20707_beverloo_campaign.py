"""Tests for the m20707 Beverloo campaign: parameter validation, formula checks, and protocol integrity.

These tests verify the campaign infrastructure without running the full DEM
(which takes minutes per case and belongs on the cluster).  The actual
Beverloo-law agreement is checked by the campaign + analysis scripts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")

import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from tensordem.silo_flow.beverloo import (
    BEVERLOO_C, BEVERLOO_K, DischargeCase, beverloo_prediction, fit_law, fit_power_law,
)
from tensordem.silo_flow.kernel import Material, SiloGeometry, damping_ratio


# ---------------------------------------------------------------------------
# 1. Reference coefficients
# ---------------------------------------------------------------------------
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"


def test_reference_json_exists_and_is_valid():
    assert REF_PATH.exists(), f"Reference JSON not found: {REF_PATH}"
    ref = json.loads(REF_PATH.read_text())
    assert ref["law"].startswith("W = C")
    C = ref["coefficients"]["C"]
    k = ref["coefficients"]["k"]
    assert 0.55 <= C <= 0.60, f"C = {C} outside literature range [0.55, 0.60]"
    assert 1.2 <= k <= 1.6, f"k = {k} outside literature range [1.2, 1.6]"


def test_module_constants_match_reference():
    ref = json.loads(REF_PATH.read_text())
    assert BEVERLOO_C == pytest.approx(ref["coefficients"]["C"], rel=1e-6)
    assert BEVERLOO_K == pytest.approx(ref["coefficients"]["k"], rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Beverloo prediction formula
# ---------------------------------------------------------------------------
def test_beverloo_prediction_matches_manual():
    C, k, d, rho_b, g = 0.58, 1.4, 0.003, 1400.0, 9.81
    D = 0.015  # D/d = 5
    Deff = D - k * d
    W_manual = C * rho_b * math.sqrt(g) * Deff ** 2.5
    W_func = beverloo_prediction(D, d, rho_b, c=C, k=k, gravity=g)
    assert W_func == pytest.approx(W_manual, rel=1e-12)


def test_beverloo_prediction_zero_below_critical_orifice():
    # D = k*d -> Deff = 0 -> W = 0
    assert beverloo_prediction(1.4 * 0.003, 0.003, 1400.0) == 0.0
    # D < k*d -> Deff < 0 -> W = 0
    assert beverloo_prediction(0.003, 0.003, 1400.0) == 0.0


def test_beverloo_prediction_scales_with_sqrt_gravity():
    D, d, rho_b = 0.015, 0.003, 1400.0
    W1 = beverloo_prediction(D, d, rho_b, gravity=9.81)
    W4 = beverloo_prediction(D, d, rho_b, gravity=9.81 / 4.0)
    assert W4 / W1 == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Power-law fit
# ---------------------------------------------------------------------------
def test_fit_recovers_exact_beverloo_exponent():
    C, k, d, rho_b = 0.58, 1.4, 0.003, 1400.0
    D = np.array([0.012, 0.015, 0.018])
    W = np.array([beverloo_prediction(Di, d, rho_b, c=C, k=k) for Di in D])
    fits = fit_law(D, d, W, rho_b)
    fk = fits["fixed_k"]
    assert fk["exponent"] == pytest.approx(2.5, abs=0.01)
    assert fk["c"] == pytest.approx(C, rel=0.01)
    assert fk["r2"] > 0.999


def test_fit_flags_wrong_exponent():
    d, rho_b = 0.003, 1400.0
    D = np.array([0.012, 0.015, 0.018])
    # Artificial data with exponent 2.0 instead of 2.5
    W = 0.58 * rho_b * math.sqrt(9.81) * (D - 1.4 * d) ** 2.0
    fits = fit_law(D, d, W, rho_b)
    fk = fits["fixed_k"]
    assert abs(fk["exponent"] - 2.5) > 0.2  # should flag as wrong


# ---------------------------------------------------------------------------
# 4. Campaign case parameters
# ---------------------------------------------------------------------------
def test_campaign_cases_satisfy_beverloo_validity():
    """All D/d ratios must be >= 4 and silo_D/D >= 2.5."""
    from benchmarks.m20707_beverloo.campaign import BASE, CASES

    for cid, overrides in CASES.items():
        params = dict(BASE)
        params.update(overrides)
        d = params["particle_diameter"]
        D = params["orifice_diameter"]
        R = params["silo_radius"]
        D_over_d = D / d
        silo_D_over_D = 2 * R / D
        assert D_over_d >= 4.0, f"{cid}: D/d = {D_over_d:.1f} < 4"
        assert silo_D_over_D >= 2.5, f"{cid}: silo_D/D = {silo_D_over_D:.1f} < 2.5"


def test_primary_case_is_D5():
    from benchmarks.m20707_beverloo.campaign import CASES
    assert "D5" in CASES
    assert CASES["D5"]["orifice_diameter"] / 0.003 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 5. DischargeCase validation
# ---------------------------------------------------------------------------
def test_discharge_case_rejects_orifice_larger_than_silo():
    with pytest.raises(ValueError):
        DischargeCase(orifice_diameter=0.060, particle_diameter=0.003,
                      silo_radius=0.025, fill_height=0.060)


def test_discharge_case_rejects_zero_orifice():
    with pytest.raises(ValueError):
        DischargeCase(orifice_diameter=0.0, particle_diameter=0.003,
                      silo_radius=0.025, fill_height=0.060)


def test_discharge_case_rejects_too_few_windows():
    with pytest.raises(ValueError):
        DischargeCase(orifice_diameter=0.015, particle_diameter=0.003,
                      silo_radius=0.025, fill_height=0.060, n_windows=1)


# ---------------------------------------------------------------------------
# 6. Material / geometry consistency
# ---------------------------------------------------------------------------
def test_material_damping_ratio_is_physical():
    # e = 0.3 -> zeta ≈ 0.22
    zeta = damping_ratio(0.3)
    assert 0.1 < zeta < 0.4


def test_material_dt_is_stable():
    mat = Material(radius=0.0015, kn=5e3, mu=0.5)
    # dt must be < 0.25 * contact period
    T = 2 * math.pi * math.sqrt(mat.mass_eff / mat.kn)
    assert mat.dt < 0.25 * T


def test_silo_geometry_volume_is_positive():
    geom = SiloGeometry(cylinder_radius=0.025, cylinder_height=0.075,
                        outlet_radius=0.0075)
    assert geom.volume_below(0.06) > 0.0
    assert geom.volume_below(0.0) == 0.0


# ---------------------------------------------------------------------------
# 6. Preset reachability (regression: the pilot config quoted in the note was
#    not runnable from the committed script, so its numbers were unverifiable)
# ---------------------------------------------------------------------------
BENCH_DIR = ROOT / "benchmarks" / "m20707_beverloo"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import campaign as CAMP  # noqa: E402


def test_pilot_preset_matches_documented_parameters():
    """docs/m20707_beverloo_verification.md §1 pilot table must be runnable."""
    for cid, ratio in (("D4", 4.0), ("D5", 5.0), ("D6", 6.0)):
        case, g = CAMP.build_case(cid, "pilot")
        assert case.particle_diameter == pytest.approx(0.002)
        assert case.silo_radius == pytest.approx(0.012)
        assert case.fill_height == pytest.approx(0.020)
        assert case.kn == pytest.approx(2.0e3)
        assert (case.settle_time, case.discard_time, case.window_time) == (
            pytest.approx(0.03), pytest.approx(0.05), pytest.approx(0.08))
        assert case.n_windows == 4
        assert case.mu == pytest.approx(0.50)
        assert case.wall_mu == pytest.approx(0.0)
        assert case.seed == 20707
        assert case.orifice_diameter / case.particle_diameter == pytest.approx(ratio)
        assert g == pytest.approx(9.81)


def test_full_preset_matches_documented_parameters():
    """docs §1 full-campaign table must be runnable too."""
    for cid, ratio in (("D4", 4.0), ("D5", 5.0), ("D6", 6.0)):
        case, g = CAMP.build_case(cid, "full")
        assert case.particle_diameter == pytest.approx(0.003)
        assert case.silo_radius == pytest.approx(0.025)
        assert case.fill_height == pytest.approx(0.060)
        assert case.kn == pytest.approx(5.0e3)
        assert case.n_windows == 6
        assert case.orifice_diameter / case.particle_diameter == pytest.approx(ratio)
        assert g == pytest.approx(9.81)


def test_mu_and_gravity_variants_are_distinct():
    assert CAMP.build_case("D5_mu40", "full")[0].mu == pytest.approx(0.40)
    assert CAMP.build_case("D5_mu55", "full")[0].mu == pytest.approx(0.55)
    # only the g/4 case departs from terrestrial gravity
    assert CAMP.build_case("D5_g4", "full")[1] == pytest.approx(9.81 / 4.0)
    for cid in ("D4", "D5", "D6", "D5_mu40", "D5_mu55"):
        assert CAMP.build_case(cid, "full")[1] == pytest.approx(9.81)
    # the g/4 case must remain geometrically identical to D5
    a, _ = CAMP.build_case("D5", "full")
    b, _ = CAMP.build_case("D5_g4", "full")
    assert a == b


def test_build_case_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        CAMP.build_case("D5", "nonexistent_preset")
    with pytest.raises(ValueError):
        CAMP.build_case("D9", "full")
    with pytest.raises(ValueError):
        CAMP.build_case("D5_mu45", "pilot")   # mu = 0.45 is not in any preset


# ---------------------------------------------------------------------------
# 7. Sharding: the work list must partition without gaps or duplication
# ---------------------------------------------------------------------------
def test_job_list_covers_every_preset_case():
    jobs = CAMP.job_list(["full", "pilot"])
    assert len(jobs) == sum(len(CAMP.PRESETS[p][1]) for p in ("full", "pilot"))
    assert ("full", "D5") in jobs and ("pilot", "D5") in jobs


@pytest.mark.parametrize("nsh", [1, 2, 3, 4, 9])
def test_shard_partition_is_disjoint_and_complete(nsh):
    jobs = CAMP.job_list(["full", "pilot"])
    shards = [CAMP.shard_jobs(jobs, i, nsh) for i in range(nsh)]
    flat = [j for s in shards for j in s]
    assert sorted(flat) == sorted(jobs), "shards must be a partition of the work list"
    assert all(isinstance(s, list) for s in shards)
    # determinism
    assert shards == [CAMP.shard_jobs(jobs, i, nsh) for i in range(nsh)]


def test_parse_shard_validates_range():
    assert CAMP.parse_shard("0/6") == (0, 6)
    assert CAMP.parse_shard("5/6") == (5, 6)
    for bad in ("6/6", "-1/6", "a/b", "0"):
        with pytest.raises(Exception):
            CAMP.parse_shard(bad)


# ---------------------------------------------------------------------------
# 8. analyse(): the pass/fail decision is a pure function of the artefacts
# ---------------------------------------------------------------------------
def test_analyse_recomputes_reference_and_flags():
    case, gravity = CAMP.build_case("D5", "full")
    # deliberately construct a measurement 5 % above the Beverloo prediction
    d, D = case.particle_diameter, case.orifice_diameter
    rho_b = 1400.0
    W_ref = BEVERLOO_C * rho_b * math.sqrt(gravity) * (D - BEVERLOO_K * d) ** 2.5
    res = {
        "n_bed": 5000, "rho_bulk": rho_b, "flow_rate_kg_s": 1.05 * W_ref,
        "flow_rate_std_kg_s": 1e-4, "flow_rate_windows_kg_s": [1.05 * W_ref] * 6,
        "particles_exited": 100, "dt": 1e-5, "steps": 1000,
    }
    out = CAMP.analyse("D5", "full", case, gravity, res, wall=1.0)
    assert out["preset"] == "full"
    assert out["beverloo"]["W_ref_kg_s"] == pytest.approx(W_ref, rel=1e-12)
    assert out["beverloo"]["rel_err_pct"] == pytest.approx(5.0, abs=1e-9)
    assert out["beverloo"]["pass_10pct"] is True
    assert out["beverloo"]["C_recovered"] == pytest.approx(1.05 * BEVERLOO_C, rel=1e-12)


def test_analyse_flags_out_of_tolerance_case():
    case, gravity = CAMP.build_case("D5", "full")
    rho_b = 1400.0
    d, D = case.particle_diameter, case.orifice_diameter
    W_ref = BEVERLOO_C * rho_b * math.sqrt(gravity) * (D - BEVERLOO_K * d) ** 2.5
    res = {
        "n_bed": 5000, "rho_bulk": rho_b, "flow_rate_kg_s": 0.80 * W_ref,
        "flow_rate_std_kg_s": 1e-4, "flow_rate_windows_kg_s": [0.80 * W_ref] * 6,
        "particles_exited": 100, "dt": 1e-5, "steps": 1000,
    }
    out = CAMP.analyse("D5", "full", case, gravity, res, wall=1.0)
    assert out["beverloo"]["rel_err_pct"] == pytest.approx(-20.0, abs=1e-9)
    assert out["beverloo"]["pass_10pct"] is False
    assert out["beverloo"]["pass_15pct"] is False


def test_analyse_records_gravity_in_denominator():
    """W_ref must scale as sqrt(g) — the g/4 case is otherwise silently wrong."""
    case, g = CAMP.build_case("D5", "full")
    case4, g4 = CAMP.build_case("D5_g4", "full")
    res = {"n_bed": 1, "rho_bulk": 1400.0, "flow_rate_kg_s": 1e-3,
           "flow_rate_std_kg_s": 0.0, "flow_rate_windows_kg_s": [1e-3],
           "particles_exited": 1, "dt": 1e-5, "steps": 1}
    a = CAMP.analyse("D5", "full", case, g, res, 1.0)
    b = CAMP.analyse("D5_g4", "full", case4, g4, res, 1.0)
    assert a["beverloo"]["W_ref_kg_s"] / b["beverloo"]["W_ref_kg_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 9. Committed artefacts (skipped until the campaign has been run)
# ---------------------------------------------------------------------------
PILOT_D5 = BENCH_DIR / "results" / "pilot" / "D5.json"


@pytest.mark.skipif(not PILOT_D5.exists(), reason="pilot artefacts not produced yet")
def test_committed_pilot_artefact_reproduces_documented_verdict():
    d = json.loads(PILOT_D5.read_text())
    assert d["preset"] == "pilot"
    assert d["case"]["D_over_d"] == pytest.approx(5.0)
    assert d["result"]["n_bed"] == 1140
    assert d["result"]["rho_bulk"] == pytest.approx(1319, abs=1)
    assert d["beverloo"]["C_recovered"] == pytest.approx(0.540, abs=0.002)
    assert d["beverloo"]["rel_err_pct"] == pytest.approx(-6.9, abs=0.1)
    assert d["beverloo"]["pass_10pct"] is True
