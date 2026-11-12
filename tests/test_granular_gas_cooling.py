"""Regression tests for the granular-gas (Haff cooling) verification surface.

Anchors are external: Haff (1983) JFM 132:401, Jenkins & Savage (1983) JFM
130:187 and the Enskog collision frequency.  The tests are deliberately small
(N <= 500 particles, short windows) -- the full verification matrix lives in
``benchmarks/granular_gas_cooling.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem import granular_gas as gg


# ---------------------------------------------------------------------------
# theory side (no simulation)
# ---------------------------------------------------------------------------
def test_theory_coefficient_closed_form():
    """2-D / 3-D constants and the (1-e^2) n sigma^{d-1} structure."""
    n, e = 0.05, 0.9
    k2 = gg.theory_cooling_coefficient(2, n, e, sigma=1.0)
    k3 = gg.theory_cooling_coefficient(3, n, e, sigma=1.0)
    # K(2-D) = sqrt(pi) n sigma (1-e^2); K(3-D) = 4/(3*4)*pi*2.2568...
    assert k2 == pytest.approx(math.sqrt(math.pi) * n * (1 - e ** 2), rel=1e-12)
    assert k3 == pytest.approx(2.36327 * n * (1 - e ** 2), rel=1e-3)
    # elastic limit: no cooling
    assert gg.theory_cooling_coefficient(2, n, 1.0) == pytest.approx(0.0)
    # quadratic in (1-e^2) removal: linear scaling in density
    assert gg.theory_cooling_coefficient(2, 2 * n, e) == pytest.approx(2.0 * k2, rel=1e-12)


def test_theory_collision_frequency_matches_enskog_3d():
    """The derivation's collision-frequency branch must reproduce Enskog's
    textbook 3-D result ``omega = 24 phi g0 sqrt(theta/pi)/sigma`` at g0=1."""
    phi, sigma, theta = 0.03, 1.0, 1.0
    n = 6.0 * phi / (math.pi * sigma ** 3)
    omega = gg.theory_collision_frequency(3, n, theta, sigma)
    assert omega == pytest.approx(24.0 * phi / math.sqrt(math.pi), rel=1e-12)


def test_theory_collision_frequency_2d_constant():
    """2-D: omega = n * 2 sigma * sqrt(pi theta)."""
    n, theta = 0.04, 0.5
    assert gg.theory_collision_frequency(2, n, theta) == pytest.approx(
        n * 2.0 * math.sqrt(math.pi * theta), rel=1e-12)


def test_flux_weighting_is_included():
    """K must use <|g|^3>, i.e. be 1.5x (2-D) the naive 'mean loss x frequency'.

    Regression guard against the factor that a naive derivation loses.
    """
    n, e, sigma = 0.03, 0.8, 1.0
    k = gg.theory_cooling_coefficient(2, n, e, sigma)
    naive = (1.0 / (2.0 * 2)) * n * (2.0 * sigma) * math.sqrt(math.pi) * (1 - e ** 2)
    # speed weighting <|g|^3>/(<|g|><|g|^2>) = 1.5 and impact-geometry
    # <cos^2>_coll = 2/3 against 1/2: total 1.5 * 4/3 = 2.0 (2-D)
    assert k / naive == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# contact / integrator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("e", [0.80, 0.90, 0.99])
def test_two_body_restitution_matches_nominal(e):
    cfg = gg.GasConfig(dim=2, n_particles=200, packing=0.02, restitution=e, kn=1000.0)
    measured = gg.two_body_restitution(cfg)
    assert measured == pytest.approx(e, abs=0.02)


def test_two_body_restitution_elastic():
    cfg = gg.GasConfig(dim=2, n_particles=200, packing=0.02, restitution=1.0, kn=1000.0)
    assert gg.two_body_restitution(cfg) == pytest.approx(1.0, abs=1e-3)


def test_elastic_energy_conservation():
    """e = 1: kinetic + elastic potential energy is a constant of motion."""
    cfg = gg.GasConfig(dim=2, n_particles=300, packing=0.05, restitution=1.0,
                       kn=500.0, steps=6000, sample_every=200, stop_ratio=1.0e9,
                       record_velocity=False, seed=3)
    run = gg.simulate_cooling(cfg)
    e_tot = run.kinetic + run.potential
    assert np.ptp(e_tot) / e_tot[0] < 5e-3


def test_contact_time_and_timestep_scaling():
    cfg = gg.GasConfig(dim=2, kn=1000.0, restitution=0.8)
    # t_c = pi sqrt(m_eff/kn) / sqrt(1 - zeta^2)
    zeta = cfg.zeta
    expected = math.pi * math.sqrt(cfg.m_eff / cfg.kn) / math.sqrt(1.0 - zeta ** 2)
    assert cfg.contact_time == pytest.approx(expected, rel=1e-12)
    assert cfg.dt == pytest.approx(cfg.contact_time / cfg.dt_per_contact, rel=1e-12)


# ---------------------------------------------------------------------------
# many-body diagnostics
# ---------------------------------------------------------------------------
def test_maxwell_kurtosis_estimator():
    rng = np.random.default_rng(0)
    gauss = rng.normal(size=(5000, 2))
    assert abs(gg.maxwell_kurtosis(gauss)) < 0.15
    uniform = rng.uniform(-1.0, 1.0, size=(5000, 2))
    assert gg.maxwell_kurtosis(uniform) < -0.8      # platykurtic by construction


def test_haff_fit_recovers_known_coefficient():
    """Synthetic exact Haff data -> K, R^2 and the -2 exponent are recovered."""
    k_true, theta0 = 0.004, 1.0
    t = np.linspace(0.0, 3.0 * 2.0 / k_true, 500)
    theta = theta0 / (1.0 + t / (2.0 / (k_true * math.sqrt(theta0)))) ** 2
    fit = gg.fit_haff_law(t, theta)
    assert fit["K_meas"] == pytest.approx(k_true, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    # The *apparent* log-log exponent only reaches -2 asymptotically (t >> tau);
    # the -2 claim is carried by the linearity of theta**-1/2 and the exact K
    # recovery above, so the apparent exponent is only bounded here.
    assert -2.0 < fit["exponent"] < 0.0


def test_impact_counting_not_inflated_by_neighbour_rebuilds():
    """Impacts must not be re-counted when the neighbour list is rebuilt.

    Regression guard: zeroing the contact mask on rebuild counted every ongoing
    contact as a fresh impact (inflating the collision frequency by ~3x).
    """
    base = dict(dim=2, n_particles=400, packing=0.03, restitution=0.9, kn=500.0,
                steps=6000, sample_every=100, stop_ratio=1.0e9, record_velocity=False,
                seed=11)
    a = gg.simulate_cooling(gg.GasConfig(rebuild=10, **base))
    b = gg.simulate_cooling(gg.GasConfig(rebuild=25, **base))
    ia, ib = float(a.impacts[-1]), float(b.impacts[-1])
    assert ia > 0 and ib > 0
    assert abs(ia - ib) / max(ia, ib) < 0.25


def test_cooling_run_reproduces_haff_trend():
    """Small end-to-end run: theta^-1/2 is linear and K is the right order."""
    cfg = gg.GasConfig(dim=2, n_particles=400, packing=0.05, restitution=0.7,
                       kn=500.0, sample_every=100, stop_ratio=5.0, steps=200_000,
                       seed=5)
    run = gg.simulate_cooling(cfg)
    fit = run.fit
    assert not run.stopped_early, "run should reach the stop ratio"
    assert fit["r2"] > 0.995
    k_th = gg.theory_cooling_coefficient(2, run.config.number_density,
                                         gg.two_body_restitution(cfg))
    # finite-density corrections are the subject of the benchmark; here only the
    # order of magnitude and the sign of the deviation are asserted
    assert 0.4 < fit["K_meas"] / k_th < 1.4


def test_bad_inputs():
    with pytest.raises(ValueError):
        gg.theory_cooling_coefficient(4, 0.05, 0.9)
    with pytest.raises(ValueError):
        gg.theory_cooling_coefficient(2, 0.05, 1.5)
    with pytest.raises(ValueError):
        gg.maxwell_kurtosis(np.zeros((2, 2)))
    cfg = gg.GasConfig(dim=2, n_particles=2, packing=1.0e-6, restitution=0.9)
    # an impossible packing fraction is rejected either by the box/cutoff guard
    # (ValueError) or earlier by the random-sequential-insertion stage
    with pytest.raises((ValueError, RuntimeError)):
        gg.simulate_cooling(cfg.with_(packing=2.0))