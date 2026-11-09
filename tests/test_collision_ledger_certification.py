"""Acceptance guards for the collision energy-ledger certificate (p54w2-N06).

Each test pins one clause of the pre-registration
(``prereg/p54w2_n06_collision_ledger_certification.json``).  The interesting
one is :func:`test_flux_convention_two_axes`: the previous node's honest FAIL
was produced by scoring a *closure identity* on an order axis; the guard asserts
that both axes exist, are different in magnitude, and that the order the
criterion asked for is recovered on the accuracy axis.

Slow cases (the shipped ``floe_rigid6dof`` re-run) are skipped unless
``TENSORDEM_SLOW=1``; they are covered by the driver instead.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tensordem.collision_ledger_certification import (
    CONVENTIONS,
    CRITERIA,
    classify,
    closure_audit,
    m5404_certificate,
    m5404_level,
    momentum_audit,
    quadrature_sweep,
    reference_pulse,
    restitution_sweep,
)
from tensordem.contact_energy_order import ContactLaw, M5404_CASE

HERTZ = ContactLaw(stiffness=1.0e6, exponent=1.5)
LINEAR = ContactLaw(stiffness=1.0e6, exponent=1.0)
DASHPOT = ContactLaw(stiffness=1.0e5, exponent=1.0, damping_model="linear",
                     damping=0.2 * 2.0 * math.sqrt(1.0e5 * 1.0), clip=True)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
CERTIFICATE = ARTIFACTS / "collision_ledger_certification" / "certification.json"


# ----------------------------------------------------------------------
# taxonomy
# ----------------------------------------------------------------------

def test_taxonomy_declares_two_axes():
    """Every convention must state a closure *and* an accuracy prediction."""
    assert set(CONVENTIONS) == {"flux", "explicit", "state", "staggered"}
    for key, conv in CONVENTIONS.items():
        assert conv.key == key
        assert conv.formula.strip()
        assert conv.closure_expectation.strip()
        assert conv.accuracy_expectation.strip()
        assert "A" in conv.audit_axis


def test_flux_accuracy_expectation_is_law_qualified():
    """The order claimed for the integral convention must name the force law."""
    text = CONVENTIONS["flux"].accuracy_expectation
    assert "O(h^2)" in text and "linear" in text


# ----------------------------------------------------------------------
# Axis A: closure against the discrete state
# ----------------------------------------------------------------------

def test_flux_closure_is_roundoff_at_every_dt():
    """(a) velocity-Verlet work-energy identity: closure to round-off, no order."""
    audit = closure_audit(HERTZ, levels=5, refinements=8)
    assert audit["max_residual_flux"] < CRITERIA["flux_closure_max"]
    verdict = classify([r["dt"] for r in audit["rows"]],
                       [r["residual_flux"] for r in audit["rows"]])
    assert verdict["klass"] == "roundoff_exact"
    # the identity must not degrade with dt: even the coarsest ladder closes
    assert audit["rows"][0]["residual_flux"] < 1e-12


def test_explicit_closure_loses_one_order():
    """(b, closure side) left-endpoint impulse x displacement is O(dt)."""
    audit = closure_audit(HERTZ, levels=5, refinements=8)
    slope = audit["fits"]["explicit"]["slope"]
    assert abs(slope - 1.0) < CRITERIA["explicit_slope_tol"]


def test_staggered_pairing_costs_one_order():
    """The shipped contact_work pairing is O(dt), not exact."""
    audit = closure_audit(HERTZ, levels=5, refinements=8)
    assert abs(audit["fits"]["staggered"]["slope"] - 1.0) < 0.2
    assert audit["max_residual_staggered"] > 1e-3


# ----------------------------------------------------------------------
# Axis B: quadrature accuracy against the exact work
# ----------------------------------------------------------------------

def test_reference_pulse_is_exact_for_a_linear_law():
    pulse = reference_pulse(LINEAR, samples=2001)
    assert pulse.source.startswith("analytic")
    # energy of the harmonic solution: U(peak) must be E0 to round-off
    u_peak = LINEAR.stiffness * pulse.delta.max() ** 2 / 2.0
    assert abs(u_peak - pulse.e0) / pulse.e0 < 1e-12


def test_explicit_quadrature_slope_is_one():
    """(b1) |slope - 1| < 0.2 on the Hertzian reference pulse."""
    quad = quadrature_sweep(HERTZ, levels=6, contacts=16, half="compression",
                            samples=20001)
    slope = quad["fits"]["explicit"]["slope"]
    assert abs(slope - 1.0) < CRITERIA["explicit_slope_tol"]


def test_flux_quadrature_slope_is_two():
    """(b2) |slope - 2| < 0.2 -- the criterion the previous node failed.

    It passes here because the residual is measured against the *exact*
    continuum work, not against the discrete kinetic-energy change.
    """
    quad = quadrature_sweep(HERTZ, levels=6, contacts=16, half="compression",
                            samples=20001)
    slope = quad["fits"]["flux"]["slope"]
    assert abs(slope - 2.0) < CRITERIA["flux_accuracy_slope_tol"]
    # local slopes must drift towards 2, not away from it
    rows = quad["rows"]
    local = [math.log(rows[i]["residual_flux"] / rows[i + 1]["residual_flux"])
             / math.log(2.0) for i in range(len(rows) - 1)]
    assert local[-1] > local[0]
    assert abs(local[-1] - 2.0) < 0.1


def test_flux_convention_two_axes():
    """The same convention: exact closure, second-order quadrature.

    This is the well-posedness result that replaces the previous node's FAIL.
    """
    quad = quadrature_sweep(HERTZ, levels=5, contacts=16, half="compression",
                            samples=20001)
    audit = closure_audit(HERTZ, levels=3, refinements=16)
    closure = max(r["residual_flux"] for r in audit["rows"])
    accuracy = quad["rows"][0]["residual_flux"]
    assert closure < 1e-12            # identity
    assert accuracy > 1e-4            # genuine quadrature error at coarse h
    assert accuracy / closure > 1e8   # the two axes are not the same question


def test_trapezoid_is_exact_for_a_linear_force_law():
    """The legal order of the flux convention is law-dependent (order infinity)."""
    quad = quadrature_sweep(LINEAR, levels=4, contacts=16, half="compression",
                            samples=2001)
    assert max(r["residual_flux"] for r in quad["rows"]) <= 1e-12
    verdict = classify([r["h"] for r in quad["rows"]],
                       [r["residual_flux"] for r in quad["rows"]])
    assert verdict["klass"] == "roundoff_exact"
    # the explicit convention is still first order on the same case
    assert abs(quad["fits"]["explicit"]["slope"] - 1.0) < 0.2


# ----------------------------------------------------------------------
# Restitution
# ----------------------------------------------------------------------

def test_restitution_phase_averaged_meets_tolerance():
    """(c) |e - e_target| < 1e-3 in the resolved segment, phase averaged."""
    sweep = restitution_sweep(DASHPOT, levels=5, contacts=8, phases=8)
    assert sweep["resolved_segment"], "resolved segment must not be empty"
    assert sweep["max_error_in_resolved_segment"] < CRITERIA["restitution_tol"]
    # phase averaging must remove phase noise: the trend is first order
    assert abs(sweep["fits"]["phase_averaged"]["slope"] - 1.0) < 0.35


def test_single_phase_restitution_is_phase_noisy():
    """Why the criterion needs phase averaging: single-phase is non-monotone."""
    sweep = restitution_sweep(DASHPOT, levels=5, contacts=8, phases=4)
    errors = [r["error_single_phase"] for r in sweep["rows"]]
    monotone = all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))
    assert not monotone
    assert max(r["phase_spread"] for r in sweep["rows"]) > 0.0


# ----------------------------------------------------------------------
# Momentum
# ----------------------------------------------------------------------

def test_five_ball_cradle_conserves_momentum():
    """(d) 5-ball cradle: total momentum to round-off, and a closing ledger."""
    audit = momentum_audit(levels=2, refinements=16)
    assert audit["n"] == 5
    assert audit["momentum_error_max"] < CRITERIA["momentum_max"]
    assert audit["residual_flux_max"] < CRITERIA["flux_closure_max"]
    # momentum is conserved structurally: it must not grow when dt shrinks
    errs = [r["momentum_error"] for r in audit["rows"]]
    assert max(errs) < 1e-12 and min(errs) < 1e-14


# ----------------------------------------------------------------------
# decision procedure
# ----------------------------------------------------------------------

def test_classify_categories():
    dt = [1.0 / 2 ** k for k in range(6)]
    conv = classify(dt, [3e-2 * h for h in dt])
    assert conv["klass"] == "convention_error"
    assert abs(conv["slope"] - 1.0) < 0.05

    plateau = classify(dt, [4.8e-5] * 6)
    assert plateau["klass"] == "implementation_error"

    exact = classify(dt, [1e-15] * 6)
    assert exact["klass"] == "roundoff_exact"

    ambiguous = classify(dt, [1e-3 * h ** 0.5 for h in dt])
    assert ambiguous["klass"] == "undetermined"


def test_m5404_reference_is_pinned():
    assert M5404_CASE["reported_residual_state"] == pytest.approx(
        5.111290246429849e-07, rel=0.0, abs=0.0)


@pytest.mark.skipif(os.environ.get("TENSORDEM_SLOW") != "1",
                    reason="drives the shipped floe solver (~30 s); the driver "
                           "covers it and caches the number")
def test_m5404_residual_reproduces():
    """The 5.111e-07 must be reproducible on the shipped solver, not a legend."""
    row = m5404_level(0)
    assert row["residual_state"] == pytest.approx(5.111290246429849e-07,
                                                  rel=1e-9)
    assert row["residual_state_injected_defect"] > 50.0 * row["residual_state"]


# ----------------------------------------------------------------------
# the reported certificate (present only after the driver ran)
# ----------------------------------------------------------------------

@pytest.mark.skipif(not CERTIFICATE.exists(),
                    reason="certification.json is produced by "
                           "scripts/p54w2_n06_collision_ledger_certification.py")
def test_certificate_all_criteria_pass():
    cert = json.loads(CERTIFICATE.read_text())
    failed = [c for c in cert["criteria"] if not c["pass"]]
    assert not failed, f"failed criteria: {[c['id'] for c in failed]}"
    assert cert["m5404"]["verdict"] == "convention_error"
    assert cert["m5404"]["reproduction_rel_dev"] < CRITERIA["m5404_reproduction_tol"]