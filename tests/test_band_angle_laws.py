"""Tests for the state-matched band-angle law test (R9 of CASE-BIAXIAL-SHEAR-BAND).

Three groups:

* the arithmetic of the laws themselves (closed forms, inversions, monotonicity);
* the state reconstruction from a stress path and a band history, on synthetic
  input whose answer is known by construction;
* the per-specimen reanalysis of ``run_r9_law_test`` on a fabricated result dict
  — that keeps the plumbing (window matching, purity check, adjudication wiring)
  under test without running a DEM cell.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensordem.band_angle_laws import (  # noqa: E402
    LAWS,
    adjudicate,
    anchor_implied_angles,
    dilation_angle_deg,
    late_window_band_angle,
    late_window_state,
    law_errors,
    secant_friction_deg,
    stress_dilatancy_psi_deg,
)


# ---------------------------------------------------------------------------
# laws
# ---------------------------------------------------------------------------
def test_secant_friction_roundtrip():
    for phi in (0.0, 15.0, 30.0, 36.0, 45.0):
        q_over_p = 2.0 * math.sin(math.radians(phi))
        assert secant_friction_deg(q_over_p) == pytest.approx(phi, abs=1e-9)


def test_secant_friction_clips():
    assert secant_friction_deg(10.0) == pytest.approx(math.degrees(math.asin(0.95)))
    assert secant_friction_deg(-1.0) == pytest.approx(0.0)


def test_dilation_angle_sign_convention():
    eps_a = np.linspace(0.0, 0.1, 21)
    # dilation: the log volumetric strain decreases while axial strain increases
    eps_v = -0.05 * eps_a
    psi, slope = dilation_angle_deg(eps_a, eps_v)
    assert slope == pytest.approx(-0.05, abs=1e-12)
    assert psi == pytest.approx(math.degrees(math.asin(0.05)), abs=1e-9)
    # compaction gives a compaction-positive slope and hence no dilation angle
    psi_c, slope_c = dilation_angle_deg(eps_a, 0.05 * eps_a)
    assert slope_c == pytest.approx(0.05, abs=1e-12)
    assert psi_c == pytest.approx(-math.degrees(math.asin(0.05)), abs=1e-9)


def test_dilation_angle_rejects_degenerate_segments():
    with pytest.raises(ValueError):
        dilation_angle_deg([0.0, 0.01, 0.02], [0.0, -0.001, -0.002])
    with pytest.raises(ValueError):
        dilation_angle_deg(np.zeros(10), np.zeros(10))


def test_stress_dilatancy_matches_closed_form():
    phi, phi_cv = 36.0, 32.0
    s_phi, s_cv = math.sin(math.radians(phi)), math.sin(math.radians(phi_cv))
    expected = math.degrees(math.asin((s_phi - s_cv) / (1.0 - s_phi * s_cv)))
    assert stress_dilatancy_psi_deg(phi, phi_cv) == pytest.approx(expected, abs=1e-12)


def test_stress_dilatancy_zero_at_critical_state_and_monotone():
    assert stress_dilatancy_psi_deg(32.0, 32.0) == pytest.approx(0.0, abs=1e-12)
    # below the critical state the relation returns no dilation, never a negative one
    assert stress_dilatancy_psi_deg(30.0, 32.0) == pytest.approx(0.0, abs=1e-12)
    values = [stress_dilatancy_psi_deg(p, 30.0) for p in (32.0, 34.0, 36.0, 38.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_law_errors_are_exact_on_their_own_prediction():
    phi, psi = 34.0, 6.0
    predictions = {"mohr_coulomb": 45.0 + 0.5 * phi,
                   "roscoe": 45.0 + 0.5 * psi,
                   "arthur": 45.0 + 0.25 * (phi + psi)}
    for law, theta in predictions.items():
        errors = law_errors(theta, phi, psi)
        assert errors[law]["abs_relative_error"] == pytest.approx(0.0, abs=1e-15)
        assert set(errors) == set(LAWS)


def test_adjudicate_picks_the_generating_law_with_margin():
    phi, psi = 34.0, 6.0
    theta = 45.0 + 0.25 * (phi + psi)          # Arthur's law generated it
    verdict = adjudicate(theta, phi, psi, tolerance=0.15, precision=0.03)
    assert verdict["closest_law"] == "arthur"
    assert verdict["errors"]["arthur"]["abs_relative_error"] == pytest.approx(0.0, abs=1e-15)
    assert verdict["discrimination_deg"] > 2.0
    assert verdict["within_tolerance"] and verdict["within_precision"]
    assert "arthur" in verdict["within_precision"]


def test_adjudicate_reports_no_discrimination_when_laws_coincide():
    # phi = psi makes Roscoe (45 + psi/2) and Arthur (45 + (phi+psi)/4) identical
    verdict = adjudicate(51.0, 12.0, 12.0)
    assert verdict["discrimination_deg"] == pytest.approx(0.0, abs=1e-12)
    assert len(verdict["within_precision"]) >= 2


def test_anchor_provenance_is_mohr_coulomb_at_phi_30():
    anchor = anchor_implied_angles(60.0)
    assert anchor["mohr_coulomb_implied_phi_deg"] == pytest.approx(30.0)
    assert anchor["roscoe_implied_psi_deg"] == pytest.approx(30.0)
    assert anchor["arthur_implied_phi_plus_psi_deg"] == pytest.approx(60.0)
    # the literature band's midpoint and upper bound assume different material states
    assert anchor_implied_angles(55.0)["mohr_coulomb_implied_phi_deg"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# state reconstruction
# ---------------------------------------------------------------------------
def _synthetic_path(*, phi_deg=35.0, slope=-0.06, n=400, peak_row=100,
                    phi_peak_deg=38.0):
    """A path that peaks early and then holds a *different* mobilised angle.

    ``stress_ratio`` is ``q / p`` (``biaxial_shear_band`` convention), so the
    late-window mean ratio maps back to ``phi_deg`` exactly.  ``eps_v`` is built
    by integrating ``slope`` so the dilation fit recovers ``slope`` exactly.
    """
    peak_row = min(int(peak_row), max(1, n // 5))
    ratio = np.full(n, 2.0 * math.sin(math.radians(phi_deg)))
    ratio[:peak_row] = np.linspace(0.1, 2.0 * math.sin(math.radians(phi_peak_deg)),
                                   peak_row)
    deviator = ratio.copy()                      # p is irrelevant for the fit
    eps_a = np.linspace(0.0, 0.2, n)
    eps_v = slope * eps_a
    return {"deviator": deviator, "stress_ratio": ratio,
            "axial_strain_log": eps_a, "volumetric_strain_log": eps_v}


def test_late_window_state_matches_the_state_it_was_built_from():
    path = _synthetic_path(phi_deg=35.0, slope=-0.06)
    state = late_window_state(path, late_window_fraction=0.8)
    assert state["phi_deg"] == pytest.approx(35.0, abs=1e-9)
    assert state["psi_deg"] == pytest.approx(math.degrees(math.asin(0.06)), abs=1e-9)
    assert state["phi_peak_deg"] > state["phi_deg"]      # peak arm is a different state
    # the late window keeps the last 80 % of the post-peak record
    span_post = 1.0 - float(state["peak_index"]) / (len(path["deviator"]) - 1)
    assert state["rows"] / (len(path["deviator"]) - 1) == pytest.approx(
        span_post * 0.8, abs=0.01)


def test_late_window_state_rejects_bad_input():
    with pytest.raises(ValueError):
        late_window_state(_synthetic_path(), late_window_fraction=0.0)
    short = {k: np.asarray(v)[:5] for k, v in _synthetic_path(n=20).items()}
    with pytest.raises(ValueError):
        late_window_state(short)


def test_late_window_band_angle_averages_and_flags_resolution():
    history = [
        {"axial_strain": 0.02, "folded_angle_deg": 20.0, "cumulative": {"folded_angle_deg": 30.0}},
        {"axial_strain": 0.08, "folded_angle_deg": 54.0, "cumulative": {"folded_angle_deg": 55.0}},
        {"axial_strain": 0.09, "folded_angle_deg": 55.0, "cumulative": {"folded_angle_deg": 55.5}},
        {"axial_strain": 0.10, "folded_angle_deg": 56.0, "cumulative": {"folded_angle_deg": 56.0}},
    ]
    out = late_window_band_angle(history, tail_start_strain=0.05)
    assert out["n_snapshots"] == 3
    assert out["incremental_mean_deg"] == pytest.approx(55.0)
    assert out["cumulative_mean_deg"] == pytest.approx((55.0 + 55.5 + 56.0) / 3.0)
    assert out["resolved"] is True
    # a two-snapshot window cannot be resolved, however tight
    out2 = late_window_band_angle(history[:3], tail_start_strain=0.05)
    assert out2["n_snapshots"] == 2 and out2["resolved"] is False
    # disagreement beyond 10 deg is not resolved either
    wide = [{"axial_strain": 0.06 + 0.01 * i, "folded_angle_deg": a,
             "cumulative": {"folded_angle_deg": a}}
            for i, a in enumerate((30.0, 45.0, 60.0))]
    assert late_window_band_angle(wide, tail_start_strain=0.05)["resolved"] is False


# ---------------------------------------------------------------------------
# per-specimen reanalysis plumbing
# ---------------------------------------------------------------------------
def _fake_result(*, theta=55.0, phi_late=34.0, phi_peak=38.0, psi=2.0):
    from benchmarks.biaxial_shear_band.run_r7_wide_window_ensemble import (
        LATE_WINDOW_FRACTION,
    )

    path = _synthetic_path(phi_deg=phi_late, phi_peak_deg=phi_peak,
                           slope=-math.sin(math.radians(psi)))
    peak_strain, final_strain = 0.05, 0.15
    tail_start = peak_strain + (1.0 - LATE_WINDOW_FRACTION) * (final_strain - peak_strain)
    history = [{"axial_strain": tail_start + 0.01 * i,
                "folded_angle_deg": theta + (0.2, -0.2, 0.0)[i],
                "cumulative": {"folded_angle_deg": theta, "eta2": 0.4, "z_score": 11.0}}
               for i in range(3)]
    return {
        "name": "synthetic", "seed": 1,
        "stopping": {"stop_reason": "post_peak"},
        "loading": {"final_axial_strain": final_strain, "inertial_number": 0.005},
        "peak": {"axial_strain": peak_strain, "friction_angle_deg": phi_peak,
                 "dilation_angle_deg": psi},
        "band": {"angle_mean_deg": theta, "history": history},
        "stress_path": path,
        "broken_bonds": 7,
        "stress_definitions": {},
    }


def test_analyse_specimen_reproduces_the_published_angle():
    from benchmarks.biaxial_shear_band.run_r9_law_test import _analyse_specimen

    row = _analyse_specimen(_fake_result(theta=55.0, phi_late=34.0, psi=2.0))
    assert row["error"] is None
    assert row["theta_deg"] == pytest.approx(55.0)
    assert row["purity_vs_published_deg"] == pytest.approx(0.0, abs=1e-12)
    assert row["resolved"] is True
    assert row["state_primary"]["phi_deg"] == pytest.approx(34.0, abs=1e-9)
    # the published peak detector smooths the deviator, so the peak state is
    # located within the ramp: close to 38 deg but not exactly at the ramp top
    assert row["state_primary"]["phi_peak_deg"] == pytest.approx(38.0, abs=1.0)
    # Arthur at (34, 2) = 54 deg, Roscoe = 46, Mohr-Coulomb = 62
    assert row["primary"]["errors"]["arthur"]["predicted_deg"] == pytest.approx(54.0)
    assert row["primary"]["closest_law"] == "arthur"
    assert row["primary"]["errors"]["arthur"]["abs_relative_error"] < 0.03
    # the peak arm uses a different (higher) friction angle and must differ
    assert (row["peak_arm"]["errors"]["mohr_coulomb"]["predicted_deg"]
            > row["primary"]["errors"]["mohr_coulomb"]["predicted_deg"])
    # the stress-dilatancy arm is reported for every declared phi_cv
    from benchmarks.biaxial_shear_band.run_r9_law_test import PHI_CV_RANGE
    assert set(row["stress_dilatancy_arm"]) == {str(v) for v in PHI_CV_RANGE}


def test_analyse_specimen_marks_errors_instead_of_crashing():
    from benchmarks.biaxial_shear_band.run_r9_law_test import _analyse_specimen

    row = _analyse_specimen({"name": "x", "seed": 2, "error": "boom"})
    assert row["error"] == "boom"


def test_ensemble_adjudication_detects_arthur_from_arthur_like_specimens():
    from benchmarks.biaxial_shear_band.run_r9_law_test import _analyse_specimen, _ensemble

    rows = []
    for i, (phi, psi) in enumerate([(34.0, 2.0), (35.0, 3.0), (33.0, 1.0), (36.0, 4.0),
                                    (34.0, 2.0), (35.0, 3.0), (33.0, 1.0), (36.0, 4.0)]):
        theta = 45.0 + 0.25 * (phi + psi)
        rows.append(_analyse_specimen(_fake_result(theta=theta, phi_late=phi, psi=psi)))
    ensemble = _ensemble(rows)
    assert ensemble["n"] == 8
    # Arthur is the only law that survives the precision bar, in every
    # stress-dilatancy arm (phi_cv never fitted)
    assert ensemble["stress_dilatancy_supported_at_every_phi_cv"] == {
        "mohr_coulomb": False, "roscoe": False, "arthur": True}
    block = ensemble["arms"]["stress_dilatancy[32]"]
    assert block["arthur"]["abs_relative_error"]["mean"] < 0.05
    assert block["arthur"]["closest_count"] == 8
    assert block["mohr_coulomb"]["abs_relative_error"]["mean"] > 0.10
    assert ensemble["verdicts"]["stress_dilatancy[32]"]["mohr_coulomb"]["supported"] is False
    # a deck built on Arthur's law sits ~ (45+10) = 55 deg, i.e. inside the anchor tolerance
    assert ensemble["anchor_within_tolerance"] is True
    assert ensemble["theta_deg"]["n"] == 8

# ---------------------------------------------------------------------------
# published (pruned) cache path and the dilation clip
# ---------------------------------------------------------------------------
def test_saturation_flag_catches_the_publication_clip():
    from tensordem.band_angle_laws import __name__ as _  # noqa: F401
    from benchmarks.biaxial_shear_band.run_r9_law_test import _saturation

    clip_deg = math.degrees(math.asin(0.95))
    assert _saturation(clip_deg)["saturated"] is True
    assert _saturation(71.81)["saturated"] is True         # the value R7 published
    flagged = _saturation(71.81)
    assert flagged["implied_d_eps_v_d_eps_a"] == pytest.approx(-0.95, abs=1e-4)
    assert _saturation(38.77)["saturated"] is False
    assert _saturation(-clip_deg)["saturated"] is True


def test_analyse_specimen_accepts_a_pruned_published_cache():
    from benchmarks.biaxial_shear_band.run_r9_law_test import _analyse_specimen

    cache = {
        "name": "anchor_s53", "seed": 53,
        "friction_angle_deg": 35.61, "dilation_angle_deg": 71.81,
        "theory_angles_deg": {"mohr_coulomb": 62.8, "roscoe": 80.9, "arthur": 71.85},
        "cumulative": {"late_window": {"radon": 55.0, "resolved": True}},
        "inertial_number": 0.0072,
    }
    row = _analyse_specimen(cache)
    assert row["state_source"].startswith("peak_from_pruned_cache")
    assert row["theta_deg"] == pytest.approx(55.0)
    assert row["resolved"] is True
    assert row["dilation_measured"]["saturated"] is True
    # the published psi saturates the fit, so the psi-based laws are far off
    assert row["primary"]["errors"]["roscoe"]["predicted_deg"] == pytest.approx(80.9, abs=0.1)
    assert row["primary"]["closest_law"] == "mohr_coulomb"
    # ... while stress-dilatancy puts Arthur on the measured angle
    assert row["stress_dilatancy_arm"]["32.0"]["errors"]["arthur"]["predicted_deg"] == pytest.approx(
        54.99, abs=0.05)
    assert row["stress_dilatancy_arm"]["32.0"]["closest_law"] == "arthur"
    assert row["theory_angles_deg"]["arthur"] == pytest.approx(71.85)


def test_analyse_specimen_reports_unusable_input():
    from benchmarks.biaxial_shear_band.run_r9_law_test import _analyse_specimen

    row = _analyse_specimen({"name": "nothing", "seed": 0})
    assert row["error"]
