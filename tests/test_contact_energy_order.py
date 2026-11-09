"""Regression guards for the contact energy-ledger conventions and their orders.

Each test pins one *claim* of ``tensordem.contact_energy_order`` so that a change
which silently downgrades a convention's order, breaks the discrete work--energy
identity, or perturbs the m5404 verdict fails loudly.

Failure modes these guards are designed to catch
------------------------------------------------
* the average-velocity (trapezoid) channel losing its exact closure -- e.g. by
  pairing the force and the velocity at different time levels, which is exactly
  the defect ``floe_rigid6dof._accumulate`` carries
  (:func:`test_staggered_pairing_costs_one_order`);
* the explicit channel drifting off first order;
* an injected integrator defect (a dropped half kick, a stale force) that the
  exact-closure guard must see even though *convergence* still looks healthy
  (:func:`test_closure_guard_catches_dropped_half_kick`);
* the restitution reference and the event-driven computation drifting apart;
* the Newton cradle losing momentum conservation;
* the m5404 re-judgement numbers moving (the reported 5.111e-07 must be
  reproducible bit for bit on this machine).
"""
from __future__ import annotations

import math

import pytest

from tensordem.contact_energy_order import (
    ContactLaw,
    M5404_CASE,
    event_driven_impact,
    fit_order,
    leapfrog_work_identity,
    ledger_conventions,
    m5404_level,
    newton_cradle,
    order_sweep,
    staggered_pairing_order,
    stiffness_sweep,
    two_body_impact,
)

HERTZ = ContactLaw(stiffness=1.0e6, exponent=1.5)
LINEAR = ContactLaw(stiffness=1.0e5, exponent=1.0)
ZETA = 0.2
DAMPED = ContactLaw(stiffness=1.0e5, exponent=1.0, damping_model="linear",
                    damping=2.0 * ZETA * math.sqrt(1.0e5))


# ---------------------------------------------------------------------------
# The discrete work--energy identity (round-off closure, any dt)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("refinements", [4, 8, 64])
def test_average_velocity_closes_to_round_off_at_any_dt(refinements):
    """Trapezoid impulse x displacement is an identity for velocity-Verlet."""
    impact = two_body_impact(HERTZ, refinements=refinements)
    conv = ledger_conventions(impact)
    assert conv["max_overlap"] > 0.0, "no contact: the test would be vacuous"
    assert conv["residual_average_velocity"] < 1e-12
    # the explicit (left endpoint) rule is *not* an identity
    assert conv["residual_explicit"] > 1e-3


def test_leapfrog_identity_on_a_grossly_under_resolved_step():
    """The identity must hold even when the contact is resolved by ~4 steps."""
    ident = leapfrog_work_identity(
        positions=[-0.02, 0.02], velocities=[0.5, -0.5], masses=[2.0, 2.0],
        radii=[0.01, 0.01], law=HERTZ, dt=2.5e-3, t_end=0.2)
    assert ident["relative_gap"] < 1e-12
    assert ident["explicit_gap"] > 1e-2


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def test_explicit_convention_is_first_order():
    sweep = order_sweep(HERTZ, refinements=4, levels=5)
    slope = sweep["fits"]["residual_explicit"]["slope"]
    assert abs(slope - 1.0) < 0.2, slope
    assert sweep["fits"]["residual_explicit"]["r2"] > 0.99


def test_state_convention_collapses_with_dt():
    sweep = order_sweep(HERTZ, refinements=4, levels=5)
    rows = sweep["levels"]
    assert rows[-1]["residual_state"] < rows[0]["residual_state"] / 100.0


def test_staggered_pairing_costs_one_order():
    """Mixing time levels (end force x half-step velocity) degrades exact -> O(dt)."""
    hy = staggered_pairing_order(HERTZ, refinements=4, levels=5)
    slope = hy["fits"]["residual_staggered_power"]["slope"]
    assert abs(slope - 1.0) < 0.25, slope
    conv = ledger_conventions(two_body_impact(HERTZ, refinements=8))
    assert conv["residual_average_velocity"] * 1e6 < conv["residual_staggered_power"]


# ---------------------------------------------------------------------------
# The negative controls: a defect must show, convergence notwithstanding
# ---------------------------------------------------------------------------

def test_closure_guard_catches_dropped_half_kick():
    """A dropped second half kick still converges but breaks the exact closure."""
    clean = order_sweep(HERTZ, refinements=4, levels=4)
    broken = order_sweep(HERTZ, refinements=4, levels=4, defect="dropped_second_kick")
    clean_guard = max(r["residual_average_velocity"] for r in clean["levels"])
    broken_guard = min(r["residual_average_velocity"] for r in broken["levels"])
    assert clean_guard < 1e-12
    # the guard channel jumps by many orders of magnitude
    assert broken_guard > 1e4 * clean_guard
    assert broken_guard > 1e-6
    # and the stale-force defect is caught by the explicit channel too
    stale = order_sweep(HERTZ, refinements=4, levels=4, defect="explicit_kick")
    assert stale["fits"]["residual_explicit"]["slope"] < 0.5


def test_constant_ledger_offset_has_zero_slope():
    """A stale initial-energy offset (constant +1e-2 E0) must NOT look convergent."""
    sweep = order_sweep(HERTZ, refinements=8, levels=4)
    dts = [r["dt"] for r in sweep["levels"]]
    offset = [max(r["residual_state"], 0.0) + 1e-2 for r in sweep["levels"]]
    fit = fit_order(dts, offset)
    assert abs(fit["slope"]) < 0.1, fit  # a defect: flat, not converging
    genuine = sweep["fits"]["residual_state"]
    assert genuine["slope"] > 1.0  # the real channel does converge
    assert genuine["ratio_coarse_over_fine"] > 30.0


# ---------------------------------------------------------------------------
# Restitution against the clamped closed form
# ---------------------------------------------------------------------------

def test_event_driven_restitution_matches_closed_form():
    for zeta in (0.05, 0.2, 0.4):
        law = ContactLaw(stiffness=1.0e5, exponent=1.0, damping_model="linear",
                         damping=2.0 * zeta * math.sqrt(1.0e5))
        res = event_driven_impact(law, substeps=1024)
        assert res["separated"]
        assert res["residual_restitution"] < 1e-9, res
        assert res["restitution_error"] < 1e-9
        # contact duration must match the analytic half period at the release angle
        assert 0.6 < res["contact_phase_time_s"] / res["contact_phase_time_theory_s"] < 1.0


def test_fixed_step_restitution_within_acceptance():
    """With the contact resolved by >=2**10 outer steps, |e - e_theory| < 1e-3."""
    e_theory = DAMPED.restitution(1.0)
    impact = two_body_impact(DAMPED, refinements=1024)
    conv = ledger_conventions(impact, e_theory=e_theory)
    assert conv["steps_per_contact"] >= 1024
    assert conv["restitution_error"] < 1e-3, conv["restitution_error"]
    # and it is a discretisation error, not a bias: the coarser run is worse
    coarse = ledger_conventions(two_body_impact(DAMPED, refinements=16),
                                e_theory=e_theory)
    assert coarse["restitution_error"] > conv["restitution_error"]


def test_contact_law_rejects_bad_input():
    with pytest.raises(ValueError):
        ContactLaw(stiffness=-1.0)
    with pytest.raises(ValueError):
        ContactLaw(stiffness=1.0, exponent=2.0)
    with pytest.raises(ValueError):
        ContactLaw(stiffness=1.0, damping_model="linear", damping=0.0)
    with pytest.raises(ValueError):
        two_body_impact(HERTZ, m_eff=0.0)
    with pytest.raises(ValueError):
        event_driven_impact(HERTZ)  # non-linear law has no closed form here


# ---------------------------------------------------------------------------
# Newton cradle
# ---------------------------------------------------------------------------

def test_newton_cradle_momentum_and_closure_guards():
    res = newton_cradle(n=5, stiffness=1.0e7, refinements=8, levels=3)
    assert res["momentum_error_max"] < 1e-12
    assert res["residual_average_velocity_max"] < 1e-12
    assert res["fits"]["residual_explicit"]["slope"] > 0.8
    # a real multi-body impact: the last ball carries most of the momentum
    last = res["results"][-1]
    assert last.kinetic_final > 0.5 * last.kinetic_initial
    assert last.velocities_final[-1] > 0.5


# ---------------------------------------------------------------------------
# Stiffness sweep: the explicit channel collapses on steps-per-contact
# ---------------------------------------------------------------------------

def test_explicit_channel_collapses_on_steps_per_contact():
    sweep = stiffness_sweep((2.5e5, 4.0e6), refinements=4, levels=3)
    for nc, entry in sweep["collapse_explicit"].items():
        assert entry["ratio_max_over_min"] < 1.1, (nc, entry)


# ---------------------------------------------------------------------------
# m5404 re-judgement (drives the shipped solver; ~10 s)
# ---------------------------------------------------------------------------

def test_m5404_reported_residual_is_reproduced():
    row = m5404_level(0)
    assert row["dt"] == pytest.approx(M5404_CASE["dt"])
    assert row["residual_state"] == pytest.approx(
        M5404_CASE["reported_residual_state"], rel=1e-6)
    assert row["residual_trapezoid"] == pytest.approx(
        M5404_CASE["reported_trapezoid"], rel=1e-6)
    # the state convention meets m5404's own acceptance line
    assert row["residual_state"] < 1e-6
    # the injected 1e-3 state-function defect is *larger* than the genuine
    # residual at this dt: the two classes are separated by ~2 decades here
    assert row["residual_state_injected_defect"] > 50.0 * row["residual_state"]