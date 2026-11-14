"""Unit tests for the constant-pressure (R5) leg of the Campbell anchor.

The round's numeric claims rest on three pieces of arithmetic that can be
pinned without running a DEM simulation:

1. the dimensionless observables of the constant-pressure device are the *same*
   functions the constant-volume rounds used (``kappa_of`` / ``tau_star_of``),
   and they satisfy ``tau* == mu / I^2`` -- otherwise the two devices could not
   be compared at all;
2. the log-log fit used for the exponent is unbiased (recovers a synthetic
   exponent/coefficient exactly, and stays inside 0.1 on 5 % multiplicative
   noise);
3. the validity gates reject a soft/high-rate point (the overlap wall) and
   accept a nominal one, including the pre-registered ``kappa I^2 = k_n/(dP)``
   scaling of the overlap proxy.
"""
from __future__ import annotations

import math

import pytest

from case_dem_shear_campbell_elastic_limit import constant_pressure as cp


def test_observables_match_library_definitions():
    rho, d, kn, gdot, tau, p = 2500.0, 0.01, 6.25e4, 158.1, 2.2e5, 4.6e5
    obs = cp.point_observables(kn=kn, rho=rho, d=d, gdot=gdot, tau=tau, pressure=p)
    assert obs["kappa"] == pytest.approx(kn / (rho * d ** 3 * gdot ** 2))
    assert obs["tau_star"] == pytest.approx(tau / (rho * d ** 2 * gdot ** 2))
    assert obs["inertial_number"] == pytest.approx(gdot * d / math.sqrt(p / rho))
    # tau* = mu / I^2 exactly, by construction
    assert obs["tau_star"] == pytest.approx(obs["tau_star_from_mu"], rel=1e-12)


def test_overlap_proxy_is_stiffness_over_pressure():
    """``kappa I^2 == k_n/(d P)`` for a linear spring -- the proxy's premise."""
    rho, d, kn, p = 2500.0, 0.01, 6.25e4, 6.25e5
    for gdot in (15.8, 158.1, 632.5):
        obs = cp.point_observables(kn=kn, rho=rho, d=d, gdot=gdot,
                                   tau=1.0, pressure=p)
        assert obs["kappa"] * obs["inertial_number"] ** 2 == pytest.approx(
            kn / (d * p), rel=1e-10)


def test_gate_rejects_soft_high_rate_point():
    """kappa I^2 = 1 (k_n = d P) is a 33 % overlap -> must fail the gate."""
    gt = cp.gate_point(pressure_error=0.0, gdot=100.0, gdot_imposed=100.0,
                       profile_r2=0.99, phi=0.70, kappa=100.0,
                       inertial=0.1)          # kappa I^2 = 1
    assert gt["gated"] is False
    assert gt["gates"]["overlap_proxy"] is False


def test_gate_accepts_nominal_point():
    gt = cp.gate_point(pressure_error=0.02, gdot=100.0, gdot_imposed=98.0,
                       profile_r2=0.97, phi=0.71, kappa=1000.0,
                       inertial=0.1)          # kappa I^2 = 10
    assert gt["gated"] is True
    assert gt["overlap_proxy"] == pytest.approx(1.0 / 30.0)


def test_fit_power_recovers_synthetic_law():
    kappa = [1e2, 3e2, 1e3, 3e3, 1e4]
    fit = cp._fit_power(kappa, [0.043 * k ** 0.894 for k in kappa])
    assert fit["beta"] == pytest.approx(0.894, abs=1e-6)
    assert fit["C"] == pytest.approx(0.043, rel=1e-6)
    assert fit["rms_log_residual"] < 1e-6


def test_fit_power_is_robust_to_multiplicative_noise():
    import random
    random.seed(7)
    kappa = [1e2, 2e2, 4e2, 1e3, 2e3, 4e3, 1e4, 2e4]
    tau = [0.043 * k ** 0.894 * math.exp(random.gauss(0.0, 0.05)) for k in kappa]
    fit = cp._fit_power(kappa, tau)
    assert abs(fit["beta"] - 0.894) < 0.1
    assert fit["sigma_beta"] > 0.0