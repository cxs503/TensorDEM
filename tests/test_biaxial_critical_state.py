"""Invariant tests for the biaxial critical-state detector (m20304).

The tests build *synthetic* stress paths with a known critical-state angle and
known dilatancy, so every assertion is against an analytic value rather than
against a previously recorded run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_critical_state import (
    critical_state_from_stress_path,
    judge_against_reference,
    phi_deg_from_stresses,
    rowe_stress_dilatancy,
)


def _path(*, eps_max=0.40, n=4001, sigma3=100e3, q_peak=360e3, q_cv=300e3,
          peak_strain=0.08, dil_rate=0.0, sigma3_drift=0.0):
    """Synthetic biaxial path: hardening to a peak, softening to a plateau."""
    eps = np.linspace(0.0, eps_max, n)
    # deviator: rapid rise, peak at peak_strain, exponential decay to q_cv
    rise = np.clip(eps / peak_strain, 0.0, 1.0)
    q = q_peak * rise
    decay = np.where(eps > peak_strain,
                     1.0 - (1.0 - q_cv / q_peak) * (1.0 - np.exp(
                         -(eps - peak_strain) / 0.05)), 1.0)
    q = q * decay
    sh = sigma3 * (1.0 + sigma3_drift * np.clip((eps - 0.2) / 0.2, 0.0, 1.0))
    sv = sh + q
    # volumetric strain: compaction early, dilation between peak and eps=0.2,
    # then isochoric (constant volume) -- i.e. a real critical state at the tail
    ev = np.where(eps < peak_strain, 0.5 * eps,
                  np.where(eps < 0.20,
                           0.5 * peak_strain + dil_rate * (eps - peak_strain),
                           0.5 * peak_strain + dil_rate * (0.20 - peak_strain)))
    return {"axial_strain": eps, "sigma_v": sv, "sigma_h": sh,
            "deviator": sv - sh, "volumetric_strain": ev}


def test_phi_from_stresses_matches_mohr_coulomb():
    phis = phi_deg_from_stresses([400e3], [100e3])
    assert phis[0] == pytest.approx(math.degrees(math.asin(0.6)), rel=1e-12)


def test_detector_recovers_known_critical_state_angle():
    path = _path(dil_rate=-0.05)          # finite pre-critical dilation
    res = critical_state_from_stress_path(path, sigma3_target=100e3)
    # at the tail q = q_cv = 300 kPa on sigma_h = 100 kPa -> sin phi = 0.6
    assert res.phi_cv_deg == pytest.approx(math.degrees(math.asin(0.6)), abs=0.05)
    assert res.converged, res.flags
    assert res.flags["isochoric"]
    assert res.flags["deviator_stationary"]
    assert res.flags["window_independent"]


def test_window_rule_is_declared_last_fraction_above_min_strain():
    path = _path(dil_rate=-0.05)
    res = critical_state_from_stress_path(path, sigma3_target=100e3,
                                          min_strain=0.20, window_fraction=0.20)
    lo, hi = res.window
    assert hi == pytest.approx(0.40, abs=1e-9)
    assert lo == pytest.approx(0.40 - 0.20 * (0.40 - 0.20), abs=1e-9)


def test_peak_is_not_reported_as_critical_state():
    path = _path(dil_rate=-0.05)
    res = critical_state_from_stress_path(path, sigma3_target=100e3)
    assert res.peak_phi_deg > res.phi_cv_deg + 2.0     # peak is distinctly higher
    assert res.peak_strain < 0.10


def test_ongoing_dilation_fails_isochoric_flag():
    # dilation continues all the way to the end of shearing: no critical state
    path = _path(dil_rate=-0.30)
    res = critical_state_from_stress_path(path, sigma3_target=100e3,
                                          vol_tolerance=0.02)
    # the tail is still dilating? No: the synthesiser freezes ev at eps=0.2.
    # Build the pathological case explicitly instead.
    eps = path["axial_strain"]
    path["volumetric_strain"] = -0.10 * eps      # steady dilation to the end
    res = critical_state_from_stress_path(path, sigma3_target=100e3,
                                          vol_tolerance=0.02)
    assert not res.flags["isochoric"]
    assert not res.converged


def test_servo_drift_is_flagged():
    path = _path(dil_rate=-0.05, sigma3_drift=0.25)
    res = critical_state_from_stress_path(path, sigma3_target=100e3,
                                          servo_tolerance=0.10)
    assert not res.flags["servo_on_target"]
    assert res.servo_relative_error == pytest.approx(
        float(np.mean(np.clip((res.window[0] + res.window[1]) / 2 - 0.2, 0, 0.2)
                      / 0.2)) * 0.25, abs=0.05)
    assert res.servo_relative_error > 0.10
    assert not res.converged


def test_softening_branch_breaks_deviator_stationary_flag():
    # q still falling by >5 % inside the averaging window
    path = _path(dil_rate=-0.05, q_cv=300e3)
    eps = path["axial_strain"]
    sh = path["sigma_h"]
    # strongly softening tail: q falls by ~55 % across the averaging window
    q = 400e3 * np.exp(-20.0 * eps)
    path["sigma_v"] = sh + q
    path["deviator"] = q
    res = critical_state_from_stress_path(path, sigma3_target=100e3)
    assert res.q_drift > 0.30
    assert not res.flags["deviator_stationary"]


def test_window_sensitivity_reports_10_20_30_percent():
    path = _path(dil_rate=-0.05)
    res = critical_state_from_stress_path(path, sigma3_target=100e3)
    assert set(res.window_sensitivity) == {"last_10pct", "last_20pct", "last_30pct"}
    assert res.window_spread_deg >= 0.0


def test_rowe_identity_recovers_interparticle_friction():
    mu = 0.55
    phi_mu = math.degrees(math.atan(mu))
    kappa = math.tan(math.radians(45.0 + 0.5 * phi_mu)) ** 2
    eps = np.linspace(0.0, 0.40, 4001)
    sigma3 = 100e3
    sigma1 = kappa * sigma3                       # D = 0 exactly
    path = {"axial_strain": eps, "sigma_v": np.full_like(eps, sigma1),
            "sigma_h": np.full_like(eps, sigma3),
            "deviator": np.full_like(eps, sigma1 - sigma3),
            "volumetric_strain": np.zeros_like(eps)}
    out = rowe_stress_dilatancy(path, mu=mu)
    assert out["phi_cv_predicted_deg"] == pytest.approx(phi_mu, rel=1e-12)
    assert out["phi_from_rowe_relation_mean_deg"] == pytest.approx(phi_mu, abs=1e-6)
    assert abs(out["measured_minus_rowe_mean"]) < 1e-9


def test_rowe_check_flags_a_dilating_assembly_as_stronger_than_rowe():
    mu = 0.55
    phi_mu = math.degrees(math.atan(mu))
    kappa = math.tan(math.radians(45.0 + 0.5 * phi_mu)) ** 2
    eps = np.linspace(0.0, 0.40, 4001)
    sigma3 = 100e3
    # a dilating assembly mobilises more than tan^2(45+phi_mu/2)
    sigma1 = 1.35 * kappa * sigma3
    ev = -0.05 * eps
    path = {"axial_strain": eps, "sigma_v": np.full_like(eps, sigma1),
            "sigma_h": np.full_like(eps, sigma3),
            "deviator": np.full_like(eps, sigma1 - sigma3),
            "volumetric_strain": ev}
    out = rowe_stress_dilatancy(path, mu=mu)
    assert out["measured_minus_rowe_mean"] > 0.2


def test_short_path_raises():
    with pytest.raises(ValueError):
        critical_state_from_stress_path(
            {"axial_strain": np.linspace(0, 0.05, 100),
             "sigma_v": np.full(100, 200e3), "sigma_h": np.full(100, 100e3),
             "volumetric_strain": np.zeros(100)}, min_strain=0.20)


def test_judge_is_relative_error_with_tolerance():
    ok = judge_against_reference(28.8, 28.81, 0.08)
    assert ok["passed"] and ok["relative_error"] < 1e-3
    bad = judge_against_reference(33.0, 28.81, 0.08)
    assert not bad["passed"] and bad["relative_error"] > 0.08