"""Provenance of the round-3 isochoric critical-state numbers.

Round 3 (``docs/CASE-DEM-BIAXIAL-FRICTION_R3_ISOCHORIC.md``) published
``phi_cv = 14.31 deg`` for the anchor configuration and an *honest-FAIL*
verdict against the predeclared Rowe (1962) limit, but the probe record lived
only on the cluster: on a clean checkout the published numbers could not be
recomputed at all.  This module closes that hole:

1. ``benchmarks/case_dem_biaxial_friction/results_m22748/iso_paths.json.gz``
   holds the anchor stress path decimated with a fixed stride (13) to 930
   samples and rounded to 7 significant figures (~45 kB, reviewable in a PR).
   The tests re-run the published analysis (:mod:`analyse_isochoric`) on that
   committed bundle -- no cluster artefact, no re-simulation -- and pin the
   numbers the document quotes.
2. The pre-registered sensitivity rules
   (``prereg/PREREG_m22748_ISOCHORIC_SENSITIVITY.json``) are checked to be
   present and self-consistent, so round 4 cannot quietly re-declare them after
   seeing the data.

What the compaction costs is measured, not assumed: re-analysing the decimated
path returns ``phi_cv`` 0.02 deg away from the published value (which came from
the full 120 716-step record), far below the +-1.5 deg window uncertainty the
document reports.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSER = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                        "analyse_isochoric.py")
BUNDLE = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                      "results_m22748", "iso_paths.json.gz")
PREREG = os.path.join(REPO, "prereg", "PREREG_m22748_ISOCHORIC_SENSITIVITY.json")

ANCHOR = "anchor_24x48_mu055_p100"
# deliberately absent: the provenance test must always read the *committed*
# bundle, even when a developer has fresh probe records on disk
PROBE_DIR_ABSENT = os.path.join(REPO, "runs", "m20537__absent")

# numbers published in the round-3 document (full 120 716-step record)
PHI_ROWE_DEG = 28.81
PUBLISHED_ANCHOR_PHI_CV_DEG = 14.31
PHI_WINDOW_UNCERTAINTY_DEG = 1.5       # document §5.4 (three-window spread bound)
RECOMPUTE_TOLERANCE_DEG = 0.10         # published vs decimated-path recomputation


def _load_analyser():
    spec = importlib.util.spec_from_file_location("analyse_isochoric", ANALYSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyser():
    return _load_analyser()


@pytest.fixture(scope="module")
def runs(analyser) -> list[dict]:
    loaded, names = analyser.load_runs(PROBE_DIR_ABSENT)
    assert names == [os.path.basename(BUNDLE)], (
        "the committed bundle must be the provenance source when no probe "
        "records are on disk")
    return loaded


@pytest.fixture(scope="module")
def case(analyser, runs) -> dict:
    assert len(runs) == 1 and runs[0]["tag"] == ANCHOR
    return analyser.analyse_one(runs[0])


# ---------------------------------------------------------------------------
# the committed record
# ---------------------------------------------------------------------------
def test_bundle_is_reviewable_and_complete(runs) -> None:
    raw, = runs
    comp = raw["compaction"]
    assert comp["n_original"] == 12073           # 10:1 subsampled probe record
    assert comp["stride"] == 13 and comp["n_kept"] == 930
    for key in ("axial_strain", "axial_strain_log", "sigma_v", "sigma_h",
                "deviator", "volumetric_strain", "volumetric_strain_log",
                "sigma_v_internal", "sigma_h_internal"):
        assert len(raw["stress_path"][key]) == comp["n_kept"], key
    assert raw["config"]["lateral_control"] == "isochoric"
    assert raw["loading"]["steps"] == 120716
    assert os.path.getsize(BUNDLE) < 400_000


def test_compaction_preserves_phi_cv(analyser, runs, tmp_path) -> None:
    """A 2x coarser compaction must not move the critical-state angle."""
    out = os.path.join(str(tmp_path), "iso_roundtrip_coarse.json.gz")
    try:
        analyser.compact_runs(runs, out, max_samples=400)
        coarse, _ = analyser.load_runs(str(tmp_path))
        coarse = [r for r in coarse if r["tag"] == ANCHOR]
        assert len(coarse) == 1
        assert coarse[0]["compaction"]["n_kept"] < runs[0]["compaction"]["n_kept"]
        phi_coarse = analyser.analyse_one(coarse[0])["critical_state"]["phi_cv_deg"]
    finally:
        if os.path.exists(out):
            os.remove(out)
    phi_fine = analyser.analyse_one(runs[0])["critical_state"]["phi_cv_deg"]
    assert abs(phi_coarse - phi_fine) < RECOMPUTE_TOLERANCE_DEG


# ---------------------------------------------------------------------------
# the published anchor numbers
# ---------------------------------------------------------------------------
def test_anchor_reproduces_the_published_round3_angle(case) -> None:
    cs = case["critical_state"]
    assert cs["strain_measure"] == "log"
    assert cs["boundary_control"] == "isochoric"
    assert abs(cs["phi_cv_deg"] - PUBLISHED_ANCHOR_PHI_CV_DEG) <= RECOMPUTE_TOLERANCE_DEG
    # and the published number itself sits inside the reported uncertainty
    assert abs(cs["phi_cv_deg"] - PUBLISHED_ANCHOR_PHI_CV_DEG) <= PHI_WINDOW_UNCERTAINTY_DEG
    assert cs["phi_cv_deg"] == pytest.approx(14.29, abs=0.02)
    assert case["peak_phi_deg"] == pytest.approx(18.13, abs=0.02)


def test_anchor_gate_flags_are_the_published_ones(case) -> None:
    cs = case["critical_state"]
    flags = cs["flags"]
    # constant volume holds to the integrator's truncation error
    assert flags["isochoric"] is True
    assert abs(cs["d_eps_v_d_eps_a"]) <= 0.02
    assert abs(cs["d_eps_v_d_eps_a"]) < 1e-4
    assert flags["lateral_walls_loaded"] is True
    assert cs["sigma_h_cv_Pa"] > 0.0
    assert flags["window_independent"] is True
    assert cs["window_spread_deg"] <= PHI_WINDOW_UNCERTAINTY_DEG
    # ... but the deviator is NOT stationary: the honest-FAIL of round 3
    assert flags["deviator_stationary"] is False
    assert cs["q_drift"] > 0.05
    assert cs["converged"] is False
    assert cs["thresholds"]["deviator_tolerance"] == pytest.approx(0.05)


def test_anchor_verdict_against_both_references_stays_a_fail(case) -> None:
    rowe = case["judge_primary_rowe"]
    bolton = case["judge_secondary_bolton"]
    assert rowe["ref_deg"] == pytest.approx(PHI_ROWE_DEG, abs=0.01)
    assert rowe["passed"] is False and rowe["relative_error"] > 0.4
    assert bolton["passed"] is False and bolton["relative_error"] > 0.4
    # the wall-traction / Love-Weber closure diagnostic stays inside a few %
    closure = case["stress_definition_closure"]
    assert abs(closure["relative_difference"]) < 0.10


def test_two_dimensional_mobilisation_ratio_is_reported_not_assumed(case) -> None:
    """sin(phi_cv)/mu must be the measured ~0.45, not the 3D Rowe value 1.0."""
    phi = case["critical_state"]["phi_cv_deg"]
    mu = case["config"]["mu"]
    ratio = math.sin(math.radians(phi)) / mu
    assert 0.40 < ratio < 0.50, ratio


# ---------------------------------------------------------------------------
# the pre-registered sensitivity matrix (runs pending, rules fixed)
# ---------------------------------------------------------------------------
def test_prereg_declares_the_decision_rules_before_the_runs() -> None:
    with open(PREREG) as fh:
        prereg = json.load(fh)
    rules = prereg["decision_rules_predeclared"]
    assert set(rules) == {"resolution_invariance", "friction_monotonicity",
                          "pressure_invariance", "rate_invariance"}
    assert prereg["preregistered_before_sensitivity_runs_landed"] is True
    assert "2.0 deg" in rules["resolution_invariance"]
    assert "2.0 deg" in rules["pressure_invariance"]
    assert "2.0 deg" in rules["rate_invariance"]
    assert "sin(phi_cv)/mu" in rules["friction_monotonicity"]
    # the four configurations that are still to be run
    assert set(prereg["pending_configurations"]) == {
        "res_16x32_mu055_p100", "mu_24x48_mu030_p100",
        "pressure_24x48_mu055_p200", "rate_24x48_mu055_p100_slow"}


# ---------------------------------------------------------------------------
# the document's table is generated by the analyser, not typed by hand
# ---------------------------------------------------------------------------
def test_markdown_table_reports_the_anchor(analyser, case) -> None:
    table = analyser.markdown_table([case])
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert len(lines) == 3                     # header + separator + one row
    assert ANCHOR in table and "14.29" in table and "50.4 %" in table