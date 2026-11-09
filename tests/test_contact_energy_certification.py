"""Acceptance gates for the rigid-contact energy-ledger certification (p54w3-n03).

Every item of the node's machine criteria is one test:

(a) three conventions implemented side by side, flux convention residual < 1e-9
(b) explicit slope |s-1| < 0.2 and trapezoid slope |s-2| < 0.2
(c) measured vs target restitution < 1e-3
(d) five-ball Newton cradle total momentum < 1e-12
(e) explicit verdict on mission m5404's 5.111290246e-07 residual

The negative controls (dropped half kick, injected 1e-3 state-function defect)
are part of the gates on purpose: a threshold that no defect can break would
certify nothing.
"""

import math

import numpy as np
import pytest

from tensordem.contact_energy_certification import (
    FLUX_TOL,
    MOMENTUM_TOL,
    RESTITUTION_TOL,
    SLOPE_TOL,
    certify_all,
    flux_closure,
    m5404_certificate,
    newton_cradle_certificate,
    order_study,
    restitution_certificate,
)
from tensordem.contact_energy_order import ContactLaw, two_body_impact

LINEAR = ContactLaw(stiffness=1.0e5, exponent=1.0)


# ---------------------------------------------------------------------------
# (a) flux convention: structural closure < 1e-9  (+ negative control)
# ---------------------------------------------------------------------------

def test_flux_convention_closes_to_round_off_at_every_dt():
    study = order_study(refinements=8, levels=5)
    assert study["flux_closes"] is True
    for row in study["rows"]:
        assert row["residual_flux_full"] < FLUX_TOL, row
    # a closure that does not depend on dt distinguishes an identity from a
    # discretisation error: the residual must not shrink with the step
    residuals = [row["residual_flux_full"] for row in study["rows"]]
    assert max(residuals) < 1e-12


def test_flux_convention_catches_a_bookkeeping_defect():
    """The 1e-9 gate is only meaningful if a real defect breaks it."""
    clean = flux_closure(two_body_impact(LINEAR, refinements=64))
    dropped = flux_closure(two_body_impact(LINEAR, refinements=64,
                                           defect="dropped_second_kick"))
    stale = flux_closure(two_body_impact(LINEAR, refinements=64,
                                         defect="explicit_kick"))
    assert clean < FLUX_TOL
    assert dropped > 1e-3, dropped
    assert stale > 1e-3, stale


# ---------------------------------------------------------------------------
# (b) explicit slope 1, trapezoid slope 2 at the interior checkpoint
# ---------------------------------------------------------------------------

def test_explicit_and_trapezoid_orders_at_the_interior_checkpoint():
    study = order_study(refinements=8, levels=7)
    assert abs(study["slope_explicit"] - 1.0) < SLOPE_TOL, study["slope_explicit"]
    assert abs(study["slope_trapezoid"] - 2.0) < SLOPE_TOL, study["slope_trapezoid"]
    # the two channels must actually differ by an order, not just by constants
    assert study["slope_trapezoid"] - study["slope_explicit"] > 0.8
    assert study["explicit_first_order"] and study["trapezoid_second_order"]


def test_full_episode_trapezoid_is_an_identity_hence_the_checkpoint():
    """Documented finding (in54w1) that the maturation had to resolve.

    At the end of a complete contact episode the force-trapezoid work *is* the
    discrete dKE, so its residual is round-off at every dt and no order can be
    fitted there; the order is measurable only at an interior checkpoint.
    """
    contributions = []
    for refinements in (16, 64, 256):
        impact = two_body_impact(LINEAR, refinements=refinements)
        trace = impact.trace
        force = trace.forces[0]
        dgap = np.diff(trace.gaps[0])
        w_trap = float(np.sum(0.5 * (force[:-1] + force[1:]) * dgap))
        dke = float(trace.kinetic[-1] - trace.kinetic[0])
        e0 = 0.5 * impact.m_eff * impact.v0 ** 2
        contributions.append(abs(w_trap - dke) / e0)
    assert max(contributions) < 1e-14, contributions
    # ... and the interior checkpoint does show a converging residual
    study = order_study(refinements=8, levels=5)
    assert study["rows"][-1]["residual_trapezoid"] < 1e-3
    assert study["rows"][-1]["residual_trapezoid"] < study["rows"][0]["residual_trapezoid"]


# ---------------------------------------------------------------------------
# (c) restitution: measured vs target < 1e-3, and it is a discretisation error
# ---------------------------------------------------------------------------

def test_restitution_within_tolerance_and_error_refines():
    cert = restitution_certificate(refinements=1024, coarse_refinements=32)
    assert cert["within_tolerance"] is True
    assert cert["restitution_error"] < RESTITUTION_TOL
    assert cert["error_shrinks_with_refinement"] is True, cert
    assert cert["e_measured"] == pytest.approx(cert["e_target"], abs=1e-3)
    # the closed-form target must not be reproduced to machine precision by a
    # fixed-step run -- that would mean the check is vacuous
    assert cert["restitution_error"] > 1e-9


# ---------------------------------------------------------------------------
# (d) five-ball Newton cradle: total momentum < 1e-12
# ---------------------------------------------------------------------------

def test_newton_cradle_momentum_below_tolerance():
    cert = newton_cradle_certificate(n=5, refinements=8, levels=3)
    assert cert["momentum_conserved"] is True
    assert cert["momentum_error_max"] < MOMENTUM_TOL
    assert cert["flux_closes"] is True
    assert cert["flux_residual_max"] < FLUX_TOL
    # the impact must actually be a multi-body one (last ball carries momentum)
    assert cert["v_last_ball_final"][-1] > 0.5
    # ... and the momentum gate must be an absolute one, not relative to a
    # vanishing momentum: the initial momentum is 1 kg m/s
    assert cert["mass_kg"] * cert["velocity_m_s"] > 0.9


# ---------------------------------------------------------------------------
# (e) m5404: explicit verdict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [3])
def test_m5404_residual_is_a_convention_not_an_implementation_defect(levels):
    cert = m5404_certificate(levels=levels)
    # the level-0 run reproduces the reported 5.111290246e-07 exactly
    assert cert["reproduced_residual"] == pytest.approx(cert["reported_residual"],
                                                       rel=1e-6)
    assert cert["reproduced_residual"] == pytest.approx(5.111290246e-07, rel=1e-6)
    # verdict + the evidence it rests on
    assert cert["verdict"] == "convention"
    assert cert["convention_identified"] is True
    assert cert["slope_state"] >= 0.8
    assert cert["state_residual_ratio"] > 2.0
    # control: a 1e-3 state-function defect plateaus (slope ~ 0) at a level far
    # above the genuine residual, so the slope test discriminates
    assert abs(cert["slope_injected_defect"]) < 0.3
    assert cert["defect_plateau_ratio"] == pytest.approx(1.0, abs=0.1)
    assert cert["rows"][0]["residual_state_injected_defect"] > 10.0 * 5.111290246e-07
    assert "口径误差" in cert["statement"]


def test_two_levels_are_refused_for_the_m5404_verdict():
    """Pre-asymptotic guard: 2 levels would return a false 'implementation'."""
    with pytest.raises(ValueError):
        m5404_certificate(levels=2)
    with pytest.raises(ValueError):
        certify_all(levels=3, m5404_levels=2)


# ---------------------------------------------------------------------------
# the folded certificate
# ---------------------------------------------------------------------------

def test_certify_all_reports_every_criterion():
    cert = certify_all(levels=5, refinements=8, cradle_levels=2,
                       m5404_levels=3, restitution_refinements=1024)
    assert cert["all_passed"] is True, cert["criteria"]
    criteria = cert["criteria"]
    assert set(criteria) == {
        "a_flux_convention_below_1e-9",
        "b_explicit_slope_one",
        "b_trapezoid_slope_two",
        "c_restitution_within_1e-3",
        "d_cradle_momentum_below_1e-12",
        "e_m5404_verdict_given",
    }
    assert all(criteria.values())
    key = cert["key_numbers"]
    assert key["flux_residual_max"] < FLUX_TOL
    assert abs(key["slope_explicit"] - 1.0) < SLOPE_TOL
    assert abs(key["slope_trapezoid"] - 2.0) < SLOPE_TOL
    assert key["restitution_error"] < RESTITUTION_TOL
    assert key["cradle_momentum_error_max"] < MOMENTUM_TOL
    assert key["m5404_verdict"] == "convention"
    # three conventions, defined side by side, no aliasing
    assert set(cert["conventions"]) == {"flux", "explicit", "trapezoid"}
    assert math.isfinite(cert["study"]["slope_trapezoid"])