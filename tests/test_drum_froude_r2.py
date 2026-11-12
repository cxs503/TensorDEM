"""Tests for the m15649 CASE-ROTATING-DRUM round-2 validation surface.

The tests check *invariants*, not just execution:

* the Froude reference anchors are the closed forms (theta_d = arccos(-Fr),
  Fr_c = 1) and the literature band table is monotone;
* the measured-regime classifier consumes only kinematics and applies the
  frozen rule order;
* the drum kernel now (i) counts contacts per particle, (ii) exposes the
  kinematic observables used by the classifier, and (iii) actually reaches the
  rest criterion during bed preparation -- the round-1 defect.
"""
import math

import numpy as np
import pytest

from tensordem.drum_segregation.kernel2d import DrumConfig, simulate
from tensordem.drum_flow.regimes import (
    MELLMANN_BANDS, MEASURED_REGIME_THRESHOLDS, classify_froude_literature,
    classify_measured, detachment_angle_deg)


def test_detachment_angle_is_arccos_minus_froude():
    for fr in (0.0, 0.1, 0.25, 0.5, 0.70710678, 0.9):
        assert detachment_angle_deg(fr) == pytest.approx(
            math.degrees(math.acos(-fr)), abs=1e-9)
    assert detachment_angle_deg(1.0) == 180.0
    assert detachment_angle_deg(1.5) == 180.0


def test_literature_bands_are_ordered_and_cover_the_axis():
    lo = [b[0] for b in MELLMANN_BANDS]
    hi = [b[1] for b in MELLMANN_BANDS]
    assert lo[0] == 0.0 and math.isinf(hi[-1])
    for a, b in zip(hi[:-1], lo[1:]):
        assert a == pytest.approx(b)
    assert classify_froude_literature(1e-5) == "sliding"
    assert classify_froude_literature(5e-4) == "surging"
    assert classify_froude_literature(5e-3) == "rolling"
    assert classify_froude_literature(5e-2) == "cascading"
    assert classify_froude_literature(0.5) == "cataracting"
    assert classify_froude_literature(1.0) == "centrifuging"
    with pytest.raises(ValueError):
        classify_froude_literature(-1.0)


def test_measured_classifier_uses_frozen_rule_order():
    th = MEASURED_REGIME_THRESHOLDS
    # pinned load wins over everything (no free surface)
    assert classify_measured({"r_p05_over_R": th["pinned_r_p05_over_R"] + 0.01,
                              "upper_flight_fraction": 0.5,
                              "wall_velocity_ratio": 0.1}) == "centrifuging"
    assert classify_measured({"r_p05_over_R": 0.3,
                              "upper_flight_fraction": th["ballistic_upper_fraction"] + 0.01,
                              "wall_velocity_ratio": 0.8}) == "cataracting"
    assert classify_measured({"r_p05_over_R": 0.3, "upper_flight_fraction": 0.0,
                              "wall_velocity_ratio": th["wall_slip_ratio"] - 0.01}) == "slipping"
    assert classify_measured({"r_p05_over_R": 0.3, "upper_flight_fraction": 0.0,
                              "wall_velocity_ratio": th["cascading_wall_ratio"] - 0.01}) == "cascading"
    assert classify_measured({"r_p05_over_R": 0.3, "upper_flight_fraction": 0.0,
                              "wall_velocity_ratio": 0.95}) == "rolling"


def _small_cfg(**kw):
    base = dict(d_small=0.010, dt=1.0e-4, revs=0.05, sample_revs=0.025,
                n_bins=6, settle_steps=3000)
    base.update(kw)
    return DrumConfig(**base)


def test_bed_preparation_reaches_the_rest_criterion():
    """Round-1 defect: the settle phase reported rms >> threshold (6.2e-2)."""
    run = simulate(_small_cfg(revs=0.02))
    d = run.diagnostics
    assert d["resting_ok"] is True
    assert d["settle_rms_speed"] < d["settle_speed_threshold"]
    assert d["quench_rounds"] <= 20
    assert d["settle_steps_used"] > 0


def test_contact_counting_and_kinematic_observables_are_produced():
    run = simulate(_small_cfg())
    series = run.series
    for key in ("wall_velocity_ratio", "wall_contact_arc_upper_deg",
                "airborne_fraction", "upper_flight_fraction",
                "r_p05_over_R", "bed_omega_over_wall"):
        assert key in series
        assert np.asarray(series[key]).size == np.asarray(series["rev"]).size
    # a rotating bed must be in contact part of the time, and the wall layer
    # must move in the direction of the shell
    assert 0.0 <= float(np.nanmean(series["airborne_fraction"])) < 0.95
    assert float(np.nanmean(series["wall_velocity_ratio"])) > 0.05
    assert 0.0 < float(np.nanmean(series["r_p05_over_R"])) <= 1.0
    arc = float(np.nanmean(series["wall_contact_arc_upper_deg"]))
    assert 60.0 < arc < 180.0


def test_centrifuging_at_froude_above_one_is_measured_not_asserted():
    cfg = _small_cfg()
    omega = math.sqrt(1.15 * cfg.gravity / cfg.radius)
    run = simulate(cfg.with_(omega=omega, revs=0.6, sample_revs=0.1))
    obs = {
        "r_p05_over_R": float(np.nanmean(run.series["r_p05_over_R"])),
        "upper_flight_fraction": float(np.nanmean(run.series["upper_flight_fraction"])),
        "wall_velocity_ratio": float(np.nanmean(run.series["wall_velocity_ratio"])),
    }
    assert obs["r_p05_over_R"] > 0.75          # load pinned as an annulus
    assert classify_measured(obs) == "centrifuging"