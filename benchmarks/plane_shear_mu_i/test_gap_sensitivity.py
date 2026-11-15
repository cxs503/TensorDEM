"""Tests for round 21 of CASE-DEM-PLANE-SHEAR-MUIDI (finite-gap sensitivity).

Two things are tested, in this order of importance:

1. the *correction machinery* (`fit_wall_correction`, `residual_table`,
   `verdict`) against synthetic data with a known dead-layer thickness -- so a
   fitted ``delta`` is never read as a measurement unless the algebra that
   produced it is right, and the identifiability guards actually fire;
2. the *driver* (`run_point` / `assemble`) end-to-end on a deliberately tiny,
   fast case, asserting that the archived fields the fit consumes are present,
   finite and that the reference-law provenance travels with the artifact.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gap_sensitivity as gs  # noqa: E402


# --------------------------------------------------------------- correction
def test_fit_recovers_known_dead_layer():
    """phi(h) = phi_inf (1 - 2 delta/h) with phi_inf = 0.64, delta = d/2."""
    phi_inf, delta_d = 0.64, 0.5
    pts = [{"gap_d": h, "phi_bed": phi_inf * (1.0 - 2.0 * delta_d / h)}
           for h in (3.0, 6.0, 10.0, 16.0)]
    fit = gs.fit_wall_correction(pts, "phi_bed")
    assert fit["identified"] is True
    assert fit["n_points"] == 4
    assert fit["v_inf"] == pytest.approx(phi_inf, abs=1e-9)
    assert fit["delta_over_d"] == pytest.approx(delta_d, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    # delta in metres is delta_over_d * particle diameter
    assert fit["delta_m"] == pytest.approx(delta_d * 2.0 * gs.R, abs=1e-12)


def test_fit_requires_two_distinct_gaps():
    one = [{"gap_d": 4.0, "phi_bed": 0.55}]
    assert gs.fit_wall_correction(one, "phi_bed")["identified"] is False
    same = [{"gap_d": 4.0, "phi_bed": 0.55}, {"gap_d": 4.0, "phi_bed": 0.56}]
    assert gs.fit_wall_correction(same, "phi_bed")["identified"] is False


def test_fit_rejects_nonpositive_intercept():
    """Two points on a *decreasing* line extrapolate to v_inf < 0 -> no quote."""
    pts = [{"gap_d": 3.0, "phi_bed": 0.01}, {"gap_d": 6.0, "phi_bed": 0.005}]
    fit = gs.fit_wall_correction(pts, "phi_bed")
    assert fit["identified"] is False


def test_fit_ignores_missing_and_nonfinite_values():
    pts = [{"gap_d": 3.0, "phi_bed": 0.55},
           {"gap_d": 6.0, "phi_bed": None},
           {"gap_d": 10.0, "phi_bed": float("nan")},
           {"gap_d": 16.0, "phi_bed": 0.62}]
    fit = gs.fit_wall_correction(pts, "phi_bed")
    assert fit["n_points"] == 2
    assert fit["identified"] is True


def test_residual_table_errors_are_percentages_against_reference():
    i = 0.01
    pts = [{"gap_d": 4.0, "I_abscissa": i, "phi_bed": gs.ref_phi(i) * 0.9}]
    rows = gs.residual_table(pts, "phi_bed", {"v_inf": 1.0})
    assert rows[0]["err_pct"] == pytest.approx(-10.0, abs=1e-6)
    assert rows[0]["ref"] == pytest.approx(float(gs.ref_phi(i)))


def test_verdict_counts_band_membership_before_and_after():
    analysis = {"per_I": {
        "0.01": {"err_pct_thinnest_gap": {"phi": -15.0, "mu_wall": -20.0,
                                          "mu_bulk": 5.0},
                 "err_pct_gap_extrapolated": {"phi": -3.0, "mu_wall": -4.0,
                                              "mu_bulk": 4.0}}}}
    v = gs.verdict(analysis)
    assert v["n_comparisons"] == 3
    assert v["n_inside_band_thinnest"] == 1          # only mu_bulk
    assert v["n_inside_band_extrapolated"] == 3      # all three
    assert v["correction_improves_majority"] is True


# ------------------------------------------------------------------ assemble
def _point(I, gap, phi, mu_w, mu_b, *, budget_ok=True, drift=1.0):
    return {"I_target": I, "gap_d": gap, "phi_bed": phi, "mu_wall": mu_w,
            "mu_bulk": mu_b, "I_abscissa": I,
            "strain_budget_respected": budget_ok,
            "mu_drift_pct_halfsplit": drift}


def test_assemble_structure_and_provenance():
    pts = [_point(0.01, h, 0.64 * (1 - 1.0 / h), 0.36 * (1 - 0.4 / h),
                  0.36 * (1 - 0.2 / h)) for h in (3.0, 6.0, 10.0, 16.0)]
    a = gs.assemble(pts)
    blk = a["per_I"]["0.01"]
    assert blk["n_points"] == 4
    assert blk["phi_fit"]["v_inf"] == pytest.approx(0.64, abs=1e-9)
    assert blk["mu_wall_fit"]["v_inf"] == pytest.approx(0.36, abs=1e-9)
    # the synthetic v_inf is 0.64, so the extrapolated error is against
    # ref_phi(I_abscissa) -- not necessarily zero, and it must match the law
    expected = 100.0 * (0.64 / float(gs.ref_phi(0.01)) - 1.0)
    assert blk["err_pct_gap_extrapolated"]["phi"] == pytest.approx(expected, abs=1e-6)
    assert blk["strain_budget_respected_all"] is True
    # the artifact carries the reference-law provenance, not a bare number
    assert a["reference"]["mu"]["constants"]["mu_s"] == pytest.approx(gs.reference()["mu"]["constants"]["mu_s"])
    assert a["reference_status"]["phi"] == gs.PHI_REF_STATUS


def test_assemble_flags_violated_strain_budget():
    pts = [_point(0.03, 3.0, 0.55, 0.3, 0.34, budget_ok=True),
           _point(0.03, 6.0, 0.58, 0.32, 0.35, budget_ok=False)]
    blk = gs.assemble(pts)["per_I"]["0.03"]
    assert blk["strain_budget_respected_all"] is False
    assert blk["n_points"] == 2


def test_assemble_groups_by_I():
    pts = [_point(0.01, 3.0, 0.55, 0.3, 0.34), _point(0.1, 3.0, 0.55, 0.4, 0.42),
           _point(0.01, 6.0, 0.58, 0.32, 0.35)]
    a = gs.assemble(pts)
    assert set(a["per_I"]) == {"0.01", "0.1"}
    assert a["per_I"]["0.01"]["n_points"] == 2


# --------------------------------------------------------------- driver smoke
@pytest.mark.parametrize("fill", [0.04])
def test_run_point_smoke_tiny_case(fill):
    """End-to-end on a tiny, fast case: fields must exist, be finite, and be
    consistent (measured gap positive, phi in (0,1), mu >= 0)."""
    rec = gs.run_point(0.1, fill, strain=0.2, settle_steps=600,
                       sample_stride=20, box_width=0.06, box_depth=0.03)
    for k in ("gap_d", "mu_wall", "phi_bed", "mu_drift_pct_halfsplit"):
        assert k in rec, k
    assert rec["n_particles"] > 20
    assert rec["gap_d"] > 1.0
    assert 0.0 < rec["phi_bed"] < 1.0
    assert rec["mu_wall"] >= 0.0
    # round-17 ceiling: request + a recovery allowance of one request unit
    assert rec["strain_achieved"] <= 2.0 * 0.2 + 0.02   # one-step overshoot
    assert rec["reference"]["phi"]["status"] == gs.PHI_REF_STATUS
    # json-serialisable archive record (the analysis reads these back)
    json.loads(json.dumps(rec))


def test_load_points_skips_broken_files(tmp_path):
    good = {"I_target": 0.01, "gap_d": 3.0, "phi_bed": 0.55,
            "mu_wall": 0.3, "mu_bulk": 0.34, "I_abscissa": 0.01,
            "strain_budget_respected": True, "mu_drift_pct_halfsplit": 1.0}
    (tmp_path / "gap0.01__fill0.040.json").write_text(json.dumps(good))
    (tmp_path / "gapbad.json").write_text("{not json")
    pts = gs._load_points(tmp_path)
    assert len(pts) == 1 and pts[0]["gap_d"] == 3.0