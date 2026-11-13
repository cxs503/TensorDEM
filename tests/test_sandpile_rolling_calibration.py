"""Tests for the rolling-friction calibration surface on the sandpile protocol.

Two properties are load-bearing and are tested first:

1. the prior guard of the frozen protocol is **not weakened**: a hand-tuned
   ``mu_r`` is still refused unless the run is explicitly flagged as a
   calibration point, and the exemption covers ``mu_s``/``mu_r`` only;
2. a calibration point can never be read as a validation baseline (its record
   carries ``kind="calibration_sweep"``, ``calibration_disabled=False`` and
   ``prior_guard="bypassed:calibration_sweep"``).

The rest tests the surrogate machinery on synthetic surfaces (inversion,
saturation fit, out-of-sample scoring) and the zero-hand-copy literature
parser.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from tensordem.protocols.sandpile import (
    PRIORS,
    SandpileConfig,
    SandpileProtocolError,
    SandpileRun,
)
from tensordem.protocols.sandpile_calibration import (
    REFERENCE_DOC,
    CalibrationCondition,
    CalibrationPoint,
    UnattainableTarget,
    fit_saturation,
    invert_target,
    parse_wk2012_reference,
    run_point,
    transfer_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def synth_point(mu_r: float, theta_deg: float, *, seed: int = 11,
                n_particles: int = 1600, dt_s: float = 1e-4) -> CalibrationPoint:
    """A calibration point carrying only the fields the surface maths uses."""
    return CalibrationPoint(
        mu_r=float(mu_r), mu_s=PRIORS["mu_s"], theta_deg=float(theta_deg),
        theta_stderr_deg=0.1, r2=0.99, packing_bulk=0.5, apex_z=0.05,
        toe_radius=0.15, radius=0.005, steps=100, simulated_time_s=0.1,
        injection_velocity_m_s=0.3, wall_seconds=1.0,
        condition={"n_particles": n_particles, "seed": seed, "dt_s": dt_s},
        record={})


def synthetic_surface(theta_inf=40.0, amplitude=25.0, mu_c=0.25,
                      mu_rs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)) -> list[CalibrationPoint]:
    return [synth_point(m, theta_inf - amplitude * math.exp(-m / mu_c))
            for m in mu_rs]


# ---------------------------------------------------------------------------
# 1. the prior guard is not weakened
# ---------------------------------------------------------------------------

def test_baseline_still_refuses_hand_tuned_mu_r():
    """Regression: without the calibration flag a tuned mu_r must be refused."""
    with pytest.raises(SandpileProtocolError, match="prior guard"):
        SandpileRun(SandpileConfig(n_particles=64, mu_r=0.3))


def test_baseline_still_refuses_hand_tuned_mu_s():
    with pytest.raises(SandpileProtocolError, match="prior guard"):
        SandpileRun(SandpileConfig(n_particles=64, mu_s=0.2))


def test_calibration_flag_is_exempt_and_tagged():
    cfg = SandpileConfig(n_particles=64, mu_r=0.3, calibration_sweep=True)
    record = cfg.as_record()
    assert record["kind"] == "calibration_sweep"
    assert record["calibration_sweep"] is True
    assert record["calibration_disabled"] is False
    assert record["prior_guard"] == "bypassed:calibration_sweep"
    assert record["priors_used"] == {"mu_s": PRIORS["mu_s"], "mu_r": 0.3}
    # a calibration point is visibly *not* the guarded baseline
    assert record["priors_match_file"] is False
    assert record["kind"] != "baseline"


def test_prior_guard_scope_is_exactly_the_two_coefficients():
    """The sweep frees ``mu_s``/``mu_r``; a plain baseline frees neither."""
    from tensordem.protocols.sandpile import assert_priors

    swept = SandpileConfig(n_particles=64, mu_s=0.2, mu_r=0.3,
                           calibration_sweep=True)
    assert_priors(swept)                      # calibration surface: allowed
    for kwargs in ({"mu_r": 0.3}, {"mu_s": 0.2}):
        with pytest.raises(SandpileProtocolError, match="prior guard"):
            assert_priors(SandpileConfig(n_particles=64, **kwargs))


def test_calibration_sweep_cannot_be_a_negative_control():
    with pytest.raises(SandpileProtocolError, match="calibration_sweep"):
        SandpileConfig(n_particles=64, mu_r=0.0, control="mu_r_zero",
                       calibration_sweep=True)


# ---------------------------------------------------------------------------
# 2. an actual calibration point runs through the frozen measurement code
# ---------------------------------------------------------------------------

def test_calibration_point_runs_and_is_measured():
    condition = CalibrationCondition(n_particles=300, pour_rate=2.0e4,
                                     settle_time=0.03, seed=5)
    point = run_point(0.3, condition)
    assert point.record["kind"] == "calibration_sweep"
    assert point.record["prior_guard"] == "bypassed:calibration_sweep"
    assert 0.0 < point.theta_deg < 90.0
    assert point.record["poured"] == 300
    assert point.height_over_radius == pytest.approx(
        point.apex_z / point.radius)


# ---------------------------------------------------------------------------
# 3. inversion
# ---------------------------------------------------------------------------

def test_invert_target_recovers_the_synthetic_coefficient():
    surface = synthetic_surface()
    for mu_r, target in ((0.1, None), (0.2, None), (0.3, None)):
        theta = float(fit_saturation(surface)(mu_r))
        inv = invert_target(surface, theta)
        # piecewise-linear inversion between measured points is exact on a
        # secant of the exponential, so allow the chord error of the surface
        assert inv["mu_r_star"] == pytest.approx(mu_r, abs=0.03)
        assert inv["attainable_deg"][0] <= theta <= inv["attainable_deg"][1]


def test_invert_target_is_exact_at_measured_points():
    surface = synthetic_surface()
    point = surface[3]
    inv = invert_target(surface, point.theta_deg)
    assert inv["mu_r_star"] == pytest.approx(point.mu_r, abs=1e-12)


def test_invert_target_reports_unattainable_band():
    surface = synthetic_surface()
    lo = min(p.theta_deg for p in surface)
    hi = max(p.theta_deg for p in surface)
    with pytest.raises(UnattainableTarget) as excinfo:
        invert_target(surface, hi + 1.0)
    assert excinfo.value.attainable == pytest.approx((lo, hi))
    assert "attainable range" in str(excinfo.value)


def test_invert_rejects_duplicate_mu_r():
    surface = synthetic_surface() + [synth_point(0.3, 20.0)]
    with pytest.raises(ValueError, match="duplicate mu_r"):
        invert_target(surface, 25.0)


# ---------------------------------------------------------------------------
# 4. saturation fit and out-of-sample transfer
# ---------------------------------------------------------------------------

def test_saturation_fit_recovers_synthetic_parameters():
    theta_inf, amplitude, mu_c = 40.0, 25.0, 0.25
    surface = synthetic_surface(theta_inf, amplitude, mu_c)
    fit = fit_saturation(surface)
    assert fit.theta_inf == pytest.approx(theta_inf, rel=1e-3)
    assert fit.amplitude == pytest.approx(amplitude, rel=1e-3)
    assert fit.mu_c == pytest.approx(mu_c, rel=1e-3)
    assert fit.rmse_deg < 1e-6
    assert fit.as_record()["form"].startswith("theta_inf")


def test_transfer_error_scores_held_out_conditions():
    fit_surface = synthetic_surface(mu_rs=(0.0, 0.05, 0.1, 0.2, 0.4))
    # held-out point under a different condition, with a known 5 % offset
    base = synth_point(0.3, 40.0 - 25.0 * math.exp(-0.3 / 0.25), seed=23)
    shifted = synth_point(0.3, base.theta_deg * 1.05, seed=23, n_particles=800)
    report = transfer_error(fit_surface, [shifted])
    row = report["points"][0]
    assert row["theta_predicted_deg"] == pytest.approx(base.theta_deg, rel=1e-9)
    assert row["abs_error_deg"] == pytest.approx(-0.05 * base.theta_deg, rel=1e-9)
    assert report["worst_rel_error"] == pytest.approx(0.05 / 1.05, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. literature reference is parsed, never retyped
# ---------------------------------------------------------------------------

def test_wk2012_reference_parses_from_the_repository_doc():
    ref = parse_wk2012_reference(REPO_ROOT)
    assert ref.mu_r == (0.0, 0.5)
    assert ref.theta_deg == (21.0, 32.0)
    assert ref.sensitivity_deg_per_unit_mu_r == pytest.approx(22.0)
    raw = (REPO_ROOT / REFERENCE_DOC).read_bytes()
    assert ref.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert ref.source_line > 0


def test_reference_parser_fails_loudly_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_wk2012_reference(tmp_path)


def test_reference_parser_fails_loudly_without_the_row(tmp_path):
    doc = tmp_path / REFERENCE_DOC
    doc.parent.mkdir(parents=True)
    doc.write_text("| nothing here |\n")
    with pytest.raises(SandpileProtocolError, match="Wensrich"):
        parse_wk2012_reference(tmp_path)