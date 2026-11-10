"""The 1D contact conductor against analytic references.

Three independent checks, none of which can pass by accident:

* **Blok semi-infinite flash** -- the interface rise under a constant flux must
  approach ``2 q sqrt(t/pi)/(e_i + e_s)``; the backward-Euler scheme is
  first-order, so the deviation must *fall* when ``dt`` is refined (a fixed
  offset would mean the wrong physics, not a discretisation error);
* **steady state** -- with a constant flux the interface settles at
  ``q R`` with ``1/R = k_i/d_i + k_s/d_s`` (two columns in parallel between the
  shared interface and their own reservoirs), which the discretisation
  represents exactly;
* **energy** -- ``int q dt`` equals sensible enthalpy plus the flux leaving
  through the Dirichlet faces, to round-off, at every step.

Plus the structural invariants a heat conductor must satisfy (positive
impedance, no flux/no heating, continuity of temperature across the interface).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_friction_thermo import (
    ICE,
    STEEL,
    TwoBodyContactConductor,
    blok_interface_rise_k,
    steady_state_resistance_m2_k_w,
)

FLUX = 1.0e6


def test_interface_impedance_is_positive():
    """Second law: adding heat at the interface cannot cool it."""
    conductor = TwoBodyContactConductor(ice_cells=40, steel_cells=40,
                                        initial_temperature_c=-5.0)
    zero, sensitivity = conductor.trial_response(1e-3)
    assert sensitivity[0] > 0.0
    assert zero[0] == pytest.approx(-5.0, abs=1e-12)


def test_commit_matches_the_affine_response():
    """``commit(q)`` must equal ``trial(q)`` exactly (same linear operator)."""
    conductor = TwoBodyContactConductor(ice_cells=30, steel_cells=30,
                                        initial_temperature_c=-3.0)
    zero, sensitivity = conductor.trial_response(1e-3)
    expected = zero[0] + FLUX * sensitivity[0]
    got = conductor.commit(1e-3, FLUX)
    assert got == pytest.approx(expected, rel=1e-12)


def test_zero_flux_leaves_the_state_untouched():
    conductor = TwoBodyContactConductor(ice_cells=20, steel_cells=20,
                                        initial_temperature_c=-7.0)
    before = conductor.state.copy()
    conductor.commit(1e-2, 0.0)
    assert np.allclose(conductor.state, before, atol=1e-12)
    assert conductor.heat_in_j_m2 == 0.0


def test_early_time_rise_converges_to_blok_with_the_step():
    """Backward Euler is first order: the Blok deviation falls with ``dt``."""
    errors = []
    for dt in (2.0e-3, 1.0e-3, 2.5e-4):
        steps = int(round(5.0e-3 / dt))
        conductor = TwoBodyContactConductor(ice_depth_m=2.0e-3, steel_depth_m=2.0e-3,
                                           ice_cells=200, steel_cells=200,
                                           initial_temperature_c=-10.0)
        for _ in range(steps):
            conductor.commit(dt, FLUX)
        numerical = conductor.interface_temperature_c + 10.0
        analytic = blok_interface_rise_k(steps * dt, FLUX)
        errors.append(abs(numerical / analytic - 1.0))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1.0e-2           # < 1 % with a fine enough step


def test_steady_state_matches_the_parallel_resistance():
    depth = 2.0e-3
    conductor = TwoBodyContactConductor(ice_depth_m=depth, steel_depth_m=depth,
                                        ice_cells=100, steel_cells=100,
                                        initial_temperature_c=-10.0)
    for _ in range(6000):
        conductor.commit(1.0e-2, FLUX)
    numerical = conductor.interface_temperature_c + 10.0
    analytic = FLUX * steady_state_resistance_m2_k_w(depth, depth)
    assert numerical == pytest.approx(analytic, rel=1e-8)
    # the analytic resistance must itself be the parallel combination
    assert steady_state_resistance_m2_k_w(depth, depth) == pytest.approx(
        1.0 / (ICE.k_w_m_k / depth + STEEL.k_w_m_k / depth), rel=1e-15)


def test_energy_is_conserved_including_the_boundary_outflow():
    conductor = TwoBodyContactConductor(ice_cells=60, steel_cells=60,
                                        initial_temperature_c=-10.0)
    worst = 0.0
    for _ in range(400):
        conductor.commit(1.0e-2, FLUX)
        worst = max(worst, abs(conductor.energy_residual_j_m2()) / conductor.heat_in_j_m2)
    assert worst < 1.0e-9


def test_interface_temperature_lies_between_the_two_bulk_values():
    """With one column warmer than the other, T_0 must sit between them."""
    conductor = TwoBodyContactConductor(ice_cells=40, steel_cells=40,
                                        initial_temperature_c=-10.0,
                                        steel_temperature_c=-2.0)
    assert -10.0 <= conductor.interface_temperature_c <= -2.0
    for _ in range(50):
        conductor.commit(1.0e-2, 0.0)
    assert -10.0 <= conductor.interface_temperature_c <= -2.0
    # with no flux the interface relaxes towards the steel (better conductor)
    assert conductor.interface_temperature_c > -6.0


def test_invalid_construction_and_steps_are_rejected():
    with pytest.raises(ValueError):
        TwoBodyContactConductor(ice_depth_m=0.0)
    with pytest.raises(ValueError):
        TwoBodyContactConductor(ice_cells=1)
    with pytest.raises(ValueError):
        TwoBodyContactConductor(initial_temperature_c=1.0)
    conductor = TwoBodyContactConductor(ice_cells=10, steel_cells=10)
    with pytest.raises(ValueError):
        conductor.commit(0.0, FLUX)
    with pytest.raises(ValueError):
        conductor.trial_response(-1.0)


def test_blok_reference_is_the_effusivity_combination():
    t = 1.0e-3
    expected = 2.0 * FLUX * math.sqrt(t / math.pi) / (
        ICE.effusivity_j_m2_k_sqrt_s + STEEL.effusivity_j_m2_k_sqrt_s)
    assert blok_interface_rise_k(t, FLUX) == pytest.approx(expected, rel=1e-15)
    assert blok_interface_rise_k(0.0, FLUX) == 0.0
    with pytest.raises(ValueError):
        blok_interface_rise_k(-1.0, FLUX)