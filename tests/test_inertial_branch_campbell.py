"""Tests for R4 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT.

Round 4 asks whether the elastic -> inertial transition of Campbell (2002) can be
reached by walking ``kappa`` downward, and whether a two-leg
``tau* = A kappa + B`` stress partition describes the curve better than a single
power law.  The answer that came back is negative on both counts, so these tests
exist to stop the negative result from being quietly converted into a positive
one later, and to pin the arithmetic that produced it:

1. **The estimator recovers what it puts in** -- synthetic data from
   ``tau* = A(kappa + c)`` must return ``A`` and ``c`` (so a "no crossover" result
   is a property of the data, not of a broken optimiser).
2. **Model selection is not rigged** -- on exact power-law data the two-leg model
   must *lose* on AIC.  If this ever passes the other way, every R4 verdict is
   suspect.
3. **Refusals and gates** -- too few gated points is a refusal, not a fit;
   rejected points are reported, never dropped; the published verdicts are the
   pre-registered ones (no PASS is upgraded after the fact).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_dem_shear_campbell_elastic_limit"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, two_term_stress_fit)

import inertial_branch as ib  # noqa: E402

RUNS = ib.RUNS
ANALYSIS = RUNS / "analysis.json"
PREREG = RUNS / "prereg.json"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _pt(kappa: float, tau_star: float, *, gated: bool = True, tag: str = "x") -> dict:
    return {"tag": tag, "kappa": float(kappa), "tau_star": float(tau_star),
            "gates": {"all": gated}, "kinetic_stress_share": 1e-3}


# ---------------------------------------------------------------------------
# 1. the estimator recovers known parameters
# ---------------------------------------------------------------------------
def test_two_term_fit_recovers_synthetic_parameters():
    A, c = 0.02, 30.0
    kappa = np.logspace(0.5, 6, 14)
    pts = [_pt(k, A * (k + c), tag=f"s{i}") for i, k in enumerate(kappa)]
    fit = two_term_stress_fit(pts, min_points=5, n_boot=50)
    assert fit["status"] == "ok"
    assert fit["A"] == pytest.approx(A, rel=1e-3)
    assert fit["kappa_crossover"] == pytest.approx(c, rel=5e-3)
    assert fit["B"] == pytest.approx(A * c, rel=5e-3)
    assert fit["B"] / fit["A"] == pytest.approx(fit["kappa_crossover"], rel=1e-9)
    for p in fit["points"]:
        assert p["model_beta"] == pytest.approx(
            p["kappa"] / (p["kappa"] + c), rel=1e-6)
        assert abs(p["residual_log"]) < 1e-3


def test_model_selection_prefers_the_power_law_on_power_law_data():
    """On exact power-law data the extra 'inertial leg' must not win.

    This is the guard against a fit that manufactures curvature: the two-leg
    model has the same parameter count as the power law, so AIC is a fair
    comparison, and the data-generating process here has *no* crossover.
    """
    beta = 0.9
    kappa = np.logspace(0.5, 6, 16)
    pts = [_pt(k, 0.02 * k ** beta, tag=f"p{i}") for i, k in enumerate(kappa)]
    fit = two_term_stress_fit(pts, min_points=5, n_boot=20)
    assert fit["status"] == "ok"
    assert fit["preferred_model"] == "single_power_law"
    assert fit["delta_aic_two_term_minus_power"] > 0
    assert fit["single_power_law"]["beta"] == pytest.approx(beta, abs=1e-6)


def test_two_term_fit_refuses_when_gates_leave_too_few_points():
    pts = [_pt(k, 1.0, gated=(i < 3)) for i, k in enumerate(np.logspace(1, 3, 8))]
    fit = two_term_stress_fit(pts, min_points=5)
    assert fit["status"] == "refused"
    assert "reason" in fit
    assert fit["n_gated"] == 3


def test_gated_points_only_enter_the_fit():
    """A rejected point must not move A, B or the crossover."""
    kappa = np.logspace(1.5, 5.5, 10)
    clean = [_pt(k, 0.015 * (k + 20.0), tag=f"c{i}") for i, k in enumerate(kappa)]
    dirty = clean + [_pt(3.0, 1e6, gated=False, tag="crush")]
    a = two_term_stress_fit(clean, min_points=5, n_boot=20)
    b = two_term_stress_fit(dirty, min_points=5, n_boot=20)
    assert b["n_submitted"] == 11 and b["n_gated"] == 10
    assert b["A"] == pytest.approx(a["A"], rel=1e-12)
    assert b["kappa_crossover"] == pytest.approx(a["kappa_crossover"], rel=1e-12)


def test_duplicate_kappa_is_averaged_not_double_counted():
    """Iso-kappa pairs are one abscissa: the fit must group them."""
    kappa = np.logspace(1.5, 5.5, 8)
    base = [_pt(k, 0.015 * (k + 20.0), tag=f"d{i}") for i, k in enumerate(kappa)]
    twinned = base + [{**_pt(k, 0.015 * (k + 20.0) * 1.05, tag=f"t{i}")}
                      for i, k in enumerate(kappa[:3])]
    a = two_term_stress_fit(base, min_points=5, n_boot=20)
    b = two_term_stress_fit(twinned, min_points=5, n_boot=20)
    assert b["n_gated"] == 11
    assert b["n_distinct_kappa"] == 8
    assert b["A"] != pytest.approx(a["A"], rel=1e-12)  # the twins do carry weight
    assert len(b["points"]) == 8


# ---------------------------------------------------------------------------
# 2. design integrity
# ---------------------------------------------------------------------------
def test_design_is_below_the_r3_floor_and_reaches_a_decade():
    rows = ib.design_table()
    grid = [r for r in rows if r.get("role") != "noise_twin"]
    below = [r for r in grid if r["kappa_predicted"] < ib.KAPPA_FLOOR_R3]
    assert len(below) >= 6
    deepest = min(r["kappa_predicted"] for r in below)
    assert ib.KAPPA_FLOOR_R3 / deepest > 10.0 * 0.9, (
        f"design must reach ~1 decade below the R3 floor, got {ib.KAPPA_FLOOR_R3/ deepest}")


def test_iso_kappa_pair_is_exact_in_the_kernel():
    """kn=1e4/gdot=16 and kn=2.5e3/gdot=8 are the same kappa by construction."""
    a = ElasticLimitShearCase(n_particles=200, kn=1e4, shear_rate=16.0)
    b = ElasticLimitShearCase(n_particles=200, kn=2.5e3, shear_rate=8.0)
    assert a.kappa == pytest.approx(b.kappa, rel=1e-12)
    assert a.kappa == pytest.approx(1e4 / (1.0 * a.d_mean ** 3 * 16.0 ** 2), rel=1e-12)


def test_design_kappas_match_the_documented_formula():
    """kappa = k_n / (rho d^3 gdot^2) with rho = 1 and d the number-average diameter."""
    for r in ib.design_table():
        case = ib._case_for(r["kn"], r["gdot"])
        expected = r["kn"] / (case.density * case.d_mean ** 3 * r["gdot"] ** 2)
        assert r["kappa_predicted"] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. published results are bound to the pre-registration and stay honest
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ANALYSIS.exists(), reason="R4 analysis not present")
def test_prereg_is_bound_to_the_analysis():
    pre = json.loads(PREREG.read_text())
    a = json.loads(ANALYSIS.read_text())
    assert a["prereg_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert pre["predictions"].keys() >= {"P1_reach", "P2_inertial", "P3_partition",
                                         "P4_boundary", "P5_kinetic_crosscheck"}
    assert set(a["verdicts"]) == {"P1_verdict", "P2_verdict", "P3_verdict",
                                  "P4_verdict", "P5_verdict"}


@pytest.mark.skipif(not ANALYSIS.exists(), reason="R4 analysis not present")
def test_negative_verdicts_are_not_upgraded():
    """The two-leg partition and the Campbell boundary claim both came back
    negative; if a later edit flips them without new runs, this fails."""
    a = json.loads(ANALYSIS.read_text())
    assert a["partition"]["status"] == "ok"
    assert a["partition"]["preferred_model"] == "single_power_law"
    assert a["partition"]["delta_aic_two_term_minus_power"] > 0
    assert a["P3_verdict"] in ("NOT_SUPPORTED", "PARTIAL")
    assert a["P4_verdict"] in ("NOT_SUPPORTED", "PARTIAL")
    # the crossover, whatever it is, must equal B/A exactly
    assert a["partition"]["kappa_crossover"] == pytest.approx(
        a["partition"]["B"] / a["partition"]["A"], rel=1e-9)


@pytest.mark.skipif(not ANALYSIS.exists(), reason="R4 analysis not present")
def test_rejected_points_are_reported_with_their_failing_gate():
    a = json.loads(ANALYSIS.read_text())
    assert a["n_gated"] + len(a["rejected"]) == a["n_points"]
    for r in a["rejected"]:
        assert r["failing_gates"], "a rejected point must name the gate it failed"
    wall = a["reach"]["wall"]
    if wall is not None:
        assert wall["tag"] in [r["tag"] for r in a["rejected"]]


@pytest.mark.skipif(not ANALYSIS.exists(), reason="R4 analysis not present")
def test_noise_floor_is_measured_and_propagated():
    a = json.loads(ANALYSIS.read_text())
    nf = a["noise_floor"]
    assert nf["n_seeds"] >= 3
    assert math.isfinite(nf["relative_std"]) and 0 < nf["relative_std"] < 0.5
    for p in a["local_slopes"]:
        assert p["sigma_n"] is not None and p["sigma_n"] > 0
    # the fit must carry the interval it claims to
    assert a["partition"]["A_ci68"] is not None
    assert a["partition"]["kappa_crossover_ci68"] is not None


@pytest.mark.skipif(not ANALYSIS.exists(), reason="R4 analysis not present")
def test_single_power_law_residual_is_at_the_noise_floor():
    """The claim 'one power law fits across the whole reachable window' must be
    checkable: its RMS log residual cannot exceed the measured seed scatter by
    more than a factor 2."""
    a = json.loads(ANALYSIS.read_text())
    rms = a["partition"]["single_power_law"]["rms_log_residual"]
    nf = a["noise_floor"]["relative_std"]
    assert rms <= 2.0 * nf, f"rms={rms:.3f} noise={nf:.3f}: power law not adequate"


@pytest.mark.skipif(not ANALYSIS.exists(), reason="R4 analysis not present")
def test_reused_r3_points_are_declared():
    a = json.loads(ANALYSIS.read_text())
    assert a["fit_input"]["n_r3"] >= 5
    assert "reused_r3_points" in a["declared_deviations"]