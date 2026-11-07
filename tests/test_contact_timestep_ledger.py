"""Time-integration layer verification: contact time, restitution, timestep, ledger.

Every assertion here is one of the campaign acceptance criteria; the numbers are
reproduced (with the full 1e5-step ledger) by
``scripts/verify_nonspherical_contact_and_timestep.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.contact_timestep_ledger import (
    SpringOscillator,
    cundall_critical_timestep,
    gas_energy_ledger,
    hertz_contact_time_weber_stillinger,
    hookean_contact_time,
    hunt_crossley_semi_analytic_restitution,
    measure_critical_timestep,
    measure_linear_contact_time,
    measure_restitution_sweep,
)


def test_hookean_contact_time_is_half_a_period():
    """t_c = pi sqrt(m/k); the per-body convention gives pi sqrt(m/(2 k_n))."""
    m, k = 1.0, 2000.0
    assert hookean_contact_time(m, k) == pytest.approx(math.pi * math.sqrt(m / k), rel=1e-15)
    assert hookean_contact_time(m, k / 2.0, per_body_stiffness=True) == pytest.approx(
        math.pi * math.sqrt(m / k), rel=1e-15)


@pytest.mark.parametrize("stiffness", [500.0, 2000.0, 8000.0])
def test_measured_linear_contact_time_within_two_percent(stiffness):
    """Three stiffness levels: measured contact duration vs pi sqrt(m/k)."""
    result = measure_linear_contact_time(0.5, 1.0, stiffness, samples_per_contact=800)
    assert abs(result["relative_error"]) <= 0.02
    assert result["measured_tc"] > 0.0


def test_hertz_contact_time_matches_weber_stillinger():
    """Hertzian contact duration against the 2.9427 (5m/4k)^{2/5} v^{-1/5} law."""
    cases = [(0.10, 4.0e4), (0.20, 9.0e4)]
    for v, k in cases:
        analytic = hertz_contact_time_weber_stillinger(v, 1.0, k)
        result = measure_linear_contact_time(v, 1.0, k, law="hertz", samples_per_contact=800)
        assert abs(result["relative_error"]) <= 0.02
        assert result["analytic_tc"] == pytest.approx(analytic, rel=1e-15)
    # the exponent is the shape anchor no linear spring can fake
    fast = measure_linear_contact_time(0.4, 1.0, 4.0e4, law="hertz", samples_per_contact=800)
    slow = measure_linear_contact_time(0.1, 1.0, 4.0e4, law="hertz", samples_per_contact=800)
    exponent = math.log(slow["measured_tc"] / fast["measured_tc"]) / math.log(0.4 / 0.1)
    # t_c ~ v^{-1/5}: slower impact -> longer contact, exponent +0.2
    assert exponent == pytest.approx(0.2, abs=0.02)


def test_restitution_sweep_against_semi_analytic():
    """e(v_imp): monotone decreasing, <= 5 % from the semi-analytic balance, R^2 >= 0.99."""
    sweep = measure_restitution_sweep(np.array([0.05, 0.1, 0.2, 0.4, 0.8]), 1.0, 1.0e4, 0.002,
                                      samples_per_contact=600)
    assert sweep["monotone"]
    assert sweep["max_relative_deviation"] <= 0.05
    assert sweep["fit_r2"] >= 0.99
    # v^{1/5} signature of a Hertzian contact
    assert sweep["fit_exponent"] == pytest.approx(0.2, abs=0.03)


def test_semi_analytic_restitution_limits():
    """Zero damping -> e = 1; stronger damping -> smaller e, monotonically."""
    assert hunt_crossley_semi_analytic_restitution(0.2, 1.0, 1.0e4, 0.0) == 1.0
    values = [hunt_crossley_semi_analytic_restitution(0.2, 1.0, 1.0e4, a)
              for a in (0.0005, 0.002, 0.005)]
    assert values[0] > values[1] > values[2]


def test_cundall_strack_critical_timestep_is_measured_exactly():
    """Bisection on the integrator vs 2 sqrt(m_min/k_max): ratio in [0.9, 1.1]."""
    result = measure_critical_timestep(1.0, 2000.0)
    assert result["theory_dt_cr"] == pytest.approx(cundall_critical_timestep(1.0, 2000.0))
    assert 0.9 <= result["ratio"] <= 1.1
    assert result["stable_bracket"] <= result["measured_dt_cr"] <= result["divergent_bracket"]


def test_timestep_above_the_bound_diverges():
    """Negative control: 1.5 dt_cr must blow up while 0.5 dt_cr stays bounded."""
    dt_cr = cundall_critical_timestep(1.0, 2.0e4)
    unstable = SpringOscillator(mass=1.0, stiffness=2.0e4, displacement=0.05).run(1.5 * dt_cr, 400)
    stable = SpringOscillator(mass=1.0, stiffness=2.0e4, displacement=0.05).run(0.5 * dt_cr, 400)
    assert unstable > 1.0e3
    assert stable < 1.1


def test_undamped_gas_energy_is_conserved():
    """64 spheres, no damping: total energy drift stays below 0.1 %."""
    ledger = gas_energy_ledger(count=64, steps=20_000, check_every=2_000, safety=0.01)
    assert abs(ledger["energy_drift_relative"]) < 1.0e-3
    assert ledger["dissipated"] == 0.0
    assert ledger["ledger_residual_relative"] == pytest.approx(ledger["energy_drift_relative"])


def test_damped_gas_ledger_closes():
    """Damped gas: |dE_mech + W_diss| / E0 stays below 1e-6."""
    ledger = gas_energy_ledger(count=64, steps=8_000, check_every=2_000, safety=3.0e-4,
                               damping=5.0)
    assert ledger["dissipated"] > 0.0
    assert abs(ledger["ledger_residual_relative"]) < 1.0e-6


def test_negative_damping_adds_energy():
    """Negative control: a negative damping coefficient must *increase* the energy."""
    ledger = gas_energy_ledger(count=32, steps=2_000, check_every=500, safety=0.01,
                               damping=-2.0)
    assert ledger["energy_drift_relative"] > 0.0


def test_torn_ledger_is_detected():
    """Negative control: dropping the dissipative term leaves a detectable imbalance."""
    ledger = gas_energy_ledger(count=32, steps=2_000, check_every=500, safety=0.01, damping=2.0)
    unaccounted = abs(ledger["energy_drift_relative"])
    assert unaccounted > 1.0e-4
    assert abs(ledger["ledger_residual_relative"]) < unaccounted / 10.0