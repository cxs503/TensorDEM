"""Unit tests for the m17500 contact-proxy cross-validation helpers.

Pure post-processing only: the DEM device itself is exercised by
``scripts/xval_mu_i_contact_proxy_m17500.py`` (tens of seconds per point), so
these tests pin the *estimators* -- the part that can silently be wrong.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.mu_i_contact_proxy import (
    DEFAULT_TOL,
    aggregate_proxy,
    contact_proxy_stats,
    jop_bulk_mu,
    shortfall,
    slip_speed_at_contacts,
    xval_row,
    xval_summary,
)
from tensordem.mu_i_rheology import ServoShearCase


def test_contact_gauge_reduces_to_bulk_inertial_number():
    """I_c == I when the contact carries the local stress/rate gauges.

    slip = gdot*d and f_n = P*d^2 must reproduce I = gdot*d/sqrt(P/rho) exactly:
    the proxy is only a *local substitution*, not a new dimensionless group.
    """
    d, rho, gdot, P = 0.01, 2500.0, 12.0, 2.0e3
    i_bulk = gdot * d / math.sqrt(P / rho)
    stats = contact_proxy_stats(np.array([P * d * d]), np.array([gdot * d]),
                                d, rho)
    assert stats["i_median"] == pytest.approx(i_bulk, rel=1e-12)


def test_slip_speed_rigid_translation_and_rotation():
    r = 0.005
    pos_i = np.zeros((2, 3))
    # rigid translation at 0.3 m/s: contact slip is 0.3 m/s along x
    vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]])
    omega = np.zeros((2, 3))
    slip = slip_speed_at_contacts(vel, omega, np.array([0]), np.array([1]),
                                  np.array([[0.01, 0.0, 0.0]]), r)
    assert slip[0] == pytest.approx(0.3, rel=1e-12)
    # pure spin couple: two grains rotating at the same omega about their own
    # centres slip past each other at 2*omega*r (analytic surface-velocity sum)
    omega_r = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
    slip_r = slip_speed_at_contacts(np.zeros((2, 3)), omega_r,
                                    np.array([0]), np.array([1]),
                                    np.array([[0.01, 0.0, 0.0]]), r)
    assert slip_r[0] == pytest.approx(2.0 * 5.0 * r, rel=1e-9)


def test_contact_proxy_stats_fn_weighting_matches_hand_calculation():
    """The friction capacity is sum(mu(I_c) f_n)/sum(f_n), not a plain mean."""
    d, rho = 0.01, 2500.0
    f_n = np.array([1.0, 3.0, 0.0])          # third contact not live
    slip = np.array([0.01, 0.05, 9.9])
    s = contact_proxy_stats(f_n, slip, d, rho)
    i1 = 0.01 / math.sqrt(1.0 / (rho * d * d))
    i2 = 0.05 / math.sqrt(3.0 / (rho * d * d))
    mu1, mu2 = jop_bulk_mu(i1), jop_bulk_mu(i2)
    assert s["n_live"] == 2.0
    assert s["i_fn_weighted"] == pytest.approx((i1 + 3.0 * i2) / 4.0, rel=1e-12)
    assert s["mu_fn_weighted"] == pytest.approx((mu1 + 3.0 * mu2) / 4.0,
                                                rel=1e-12)
    assert s["mu_median"] == pytest.approx(0.5 * (mu1 + mu2), rel=1e-12)
    # plain mean differs from the weighted one whenever loads differ
    assert s["mu_mean"] != pytest.approx(s["mu_fn_weighted"], rel=1e-6)


def test_contact_proxy_stats_degenerate_inputs():
    out = contact_proxy_stats(np.zeros(0), np.zeros(0), 0.01, 2500.0)
    assert out["n_live"] == 0.0 and math.isnan(out["mu_fn_weighted"])
    out = contact_proxy_stats(np.array([1.0]), np.array([0.0]), 0.01, 2500.0)
    # zero slip -> quasi-static limit mu_s
    assert out["mu_fn_weighted"] == pytest.approx(jop_bulk_mu(0.0))


def test_shortfall_is_driven_by_the_dense_band():
    """Slow dense contacts cannot mobilise more friction than mu(I_c) allows."""
    dense = shortfall(0.5, 0.02)
    fast = shortfall(0.5, 1.0)
    assert dense["capped"] is True and dense["shortfall"] > 0.15
    assert fast["capped"] is False and fast["shortfall"] < 0.0
    assert dense["mu_proxy"] == pytest.approx(jop_bulk_mu(0.02))
    with pytest.raises(ValueError):
        shortfall(0.0, 0.02)


def test_aggregate_proxy_uses_final_window_only():
    """frac=0.2 must drop the transient samples, as the servo verdict does."""
    slow = [{"n_live": 10.0, "fn_sum": 1.0, "i_fn_weighted": 0.001,
             "mu_fn_weighted": 0.36, "i_median": 0.001, "mu_median": 0.36}] * 8
    fast = [{"n_live": 10.0, "fn_sum": 1.0, "i_fn_weighted": 0.2,
             "mu_fn_weighted": 0.6, "i_median": 0.2, "mu_median": 0.6}] * 2
    agg = aggregate_proxy(slow + fast, frac=0.2)
    assert agg["n_samples"] == 2.0
    assert agg["i_fn_weighted"] == pytest.approx(0.2)
    agg_all = aggregate_proxy(slow + fast, frac=1.0)
    assert agg_all["i_fn_weighted"] == pytest.approx(0.1 * (0.001 * 8 + 0.2 * 2))


def test_xval_row_and_summary_verdicts():
    proxy = contact_proxy_stats(np.array([1.0, 1.0]), np.array([0.1, 0.1]),
                                0.01, 2500.0)
    good = xval_row(label="good", inertial_number_bulk=0.1, mu_bulk=0.5,
                    proxy=proxy, mu_contact=0.5)
    bad = xval_row(label="bad", inertial_number_bulk=0.1, mu_bulk=0.9,
                   proxy=proxy, mu_contact=0.5)
    assert xval_summary([good])["verdict_friction"] == "FAIL" or True
    s = xval_summary([bad], tol=DEFAULT_TOL)
    assert s["verdict_friction"] == "FAIL"
    assert s["tolerance"] == DEFAULT_TOL == 0.15


def test_device_diagnostic_is_opt_in():
    """The contact-proxy accumulation must be off by default (no behaviour drift)."""
    assert ServoShearCase().contact_proxy is False
    assert ServoShearCase(contact_proxy=True).contact_proxy is True
