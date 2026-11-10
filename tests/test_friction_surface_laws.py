"""Analytic anchors of the two declared friction laws (P152W1-N04).

Every assertion here is against a closed form, not against a stored number, and
the *same* code path (``tensordem.friction_surface.laws``) is used by the DEM
kernel, so an anchored law and a simulated law cannot drift apart.
"""
import math

import numpy as np
import pytest

from tensordem.friction_surface import laws


def test_tangential_spring_reproduces_kt_delta():
    """Below the cone the tangential force is exactly ``-k_t delta_t``."""
    kt = 5.0e4
    delta = np.array([[1.0e-5, 0.0, 0.0], [0.0, -2.0e-5, 0.0]])
    f = laws.tangential_force(delta, np.zeros_like(delta), np.array([10.0, 10.0]),
                              kt=kt, mu=0.5)
    assert np.allclose(f, -kt * delta, rtol=1e-12, atol=0.0)


def test_coulomb_cap_is_exact_and_not_overshot():
    """At saturation ``|F_t| = mu F_n`` exactly (zero slack)."""
    f_n = np.array([3.7])
    delta = np.array([[1.0, 0.0, 0.0]])          # far beyond saturation
    f = laws.tangential_force(delta, np.zeros_like(delta), f_n, kt=1.0e3, mu=0.4)
    assert np.isclose(np.linalg.norm(f), 0.4 * 3.7, rtol=1e-12)


def test_velocity_damping_term_is_linear_and_dissipative():
    """``F_t`` gains exactly ``c_t v_t``; the dissipated power is ``c_t |v_t|^2 >= 0``."""
    kt, ct, mu = 1.0e4, 25.0, 0.9
    v = np.array([[0.01, -0.02, 0.0]])
    zero = laws.tangential_force(np.zeros((1, 3)), np.zeros_like(v),
                                 np.array([5.0]), kt=kt, mu=mu, ct=ct)
    f = laws.tangential_force(np.zeros((1, 3)), v, np.array([5.0]),
                              kt=kt, mu=mu, ct=ct, cap_total=False)
    assert np.allclose(f - zero, -ct * v, rtol=1e-12)
    power = float(np.dot(f[0], v[0]))            # power supplied *to* the contact
    assert power <= 0.0                          # ... i.e. dissipated by the dashpot
    assert np.isclose(-power, ct * float(np.dot(v[0], v[0])), rtol=1e-12)


def test_cone_projection_keeps_the_total_inside_the_cone():
    """``cap_total`` never lets the dashpot push the total force out of the cone."""
    f_n = np.array([1.0])
    delta = np.array([[1.0, 0.0, 0.0]])
    v = np.array([[-5.0, 0.0, 0.0]])             # huge sliding velocity
    literal = laws.tangential_force(delta, v, f_n, kt=1.0e3, mu=0.5, ct=100.0,
                                    cap_total=False)
    capped = laws.tangential_force(delta, v, f_n, kt=1.0e3, mu=0.5, ct=100.0,
                                   cap_total=True)
    assert np.linalg.norm(literal) > 0.5 * f_n[0]
    assert np.isclose(np.linalg.norm(capped), 0.5 * f_n[0], rtol=1e-12)


def test_rolling_couple_cap():
    f_n = np.array([2.0])
    theta = np.array([[10.0, 0.0, 0.0]])
    m = laws.rolling_torque(theta, np.zeros_like(theta), f_n, kr=1.0e-3,
                            mu_r=0.1, r_eff=5.0e-3)
    assert np.isclose(np.linalg.norm(m), 0.1 * 5.0e-3 * 2.0, rtol=1e-12)


@pytest.mark.parametrize("mu", [0.2, 0.5, 0.8])
def test_sliding_onset_equals_arctan_mu(mu):
    """Time-stepped onset angle of the irrotational slider == ``arctan(mu)``."""
    measured = laws.simulate_slider_onset_deg(mu)
    analytic = laws.sliding_onset_angle_deg(mu)
    assert abs(measured - analytic) < 0.5, (measured, analytic)


@pytest.mark.parametrize("mu_r", [0.05, 0.2])
def test_rolling_onset_equals_arctan_mu_r(mu_r):
    measured = laws.simulate_rolling_onset_deg(mu_r)
    analytic = laws.rolling_onset_angle_deg(mu_r)
    assert abs(measured - analytic) < 0.5, (measured, analytic)


def test_steady_slider_acceleration_matches_coulomb_closed_form():
    """Above onset the measured acceleration equals ``g(sin - mu cos)``."""
    mu, tilt = 0.4, 30.0
    analytic = 9.81 * (math.sin(math.radians(tilt)) - mu * math.cos(math.radians(tilt)))
    measured = laws.slider_acceleration(mu, tilt)
    assert abs(measured - analytic) / analytic < 0.01


def test_below_onset_the_slider_does_not_runaway():
    mu = 0.5
    tilt = laws.sliding_onset_angle_deg(mu) - 3.0
    assert abs(laws.slider_acceleration(mu, tilt)) < 1.0e-2


def test_uncapped_dashpot_changes_macroscopic_friction():
    """The declared (uncapped) two-term law is *not* Coulomb once ``c_t > 0``.

    This is the quantitative reason the kernel defaults to ``cap_total=True``:
    a linear dashpot added to a Coulomb slider removes the threshold (the slider
    creeps at any tilt), so the macroscopic friction of the same ``mu`` is no
    longer reproduced.
    """
    effect = laws.tangential_damping_effect(0.5, ct=31.6)
    assert abs(effect["pure_coulomb_acceleration_m_s2"]
               - effect["analytic_acceleration_m_s2"]) / effect["analytic_acceleration_m_s2"] < 0.01
    assert effect["literal_form_acceleration_m_s2"] < 0.05 * effect["analytic_acceleration_m_s2"]
    assert abs(effect["cone_projected_acceleration_m_s2"]
               - effect["analytic_acceleration_m_s2"]) / effect["analytic_acceleration_m_s2"] < 1.0e-6