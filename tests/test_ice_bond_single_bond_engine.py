"""Single-bond engine driver: the emergent rate exponent at the engine level.

Closes the gap-ledger "next node" left open by the specimen-level campaign
(``docs/ice_bond_specimen_rate_dependence.md``): the 4x4x3 opposed-platen
specimen is inertia-dominated at 0.25-4 s^-1, so its emergent strength
exponent could not be compared to the analytic ductile envelope ``1/(m+1)``.
This test isolates the constitutive response with a single bond driven
kinematically (no inertia) and checks that the *engine's* call sites
(``_bond_force`` history advance + ``_commit_creep_damage`` with the engine's
``F_ref``) reproduce the analytic exponent.

Constitutive correctness of the raw law is covered by
``tests/test_ice_bond_viscoelastic.py``; the engine *wiring* (force, branch
advance, dissipation ledger, checkpointing) by ``tests/test_ice_bond_engine.py``.
This test bridges the two: the engine applies the law correctly, in a regime
where the inertial term is absent.
"""
from __future__ import annotations

import math

import pytest

from scripts.verify_ice_bond_single_bond_engine import (
    analytic_brittle_force,
    analytic_creep_force,
    analytic_exponent,
    run_one,
)


def _slope(rates, strengths) -> float:
    xs = [math.log10(r) for r in rates]
    ys = [math.log10(s) for s in strengths]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def test_linear_bond_is_rate_independent_negative_control():
    """Arm A: the legacy linear bond must be rate independent (brittle)."""
    rows = [run_one("linear", r) for r in (0.1, 1.0, 3.0)]
    for row in rows:
        assert row["failure_mode"] == 1, "linear bond must fail by brittle cleavage"
    strengths = [r["F_f_N"] for r in rows]
    spread = (max(strengths) - min(strengths)) / min(strengths)
    assert spread < 1e-3, f"linear bond strength must be rate independent, got {spread}"
    for row in rows:
        assert row["F_f_N"] == pytest.approx(analytic_brittle_force(), rel=1e-3)


def test_linear_creep_arm_recovers_the_ductile_exponent():
    """Arm B: F_f ~ edot**(1/(m+1)) in the creep-dominated window."""
    rates = (0.03, 0.1, 0.3)
    rows = [run_one("linear+creep", r) for r in rates]
    for row in rows:
        assert row["failure_mode"] == 4, "creep-dominated window must fail by creep"
    strengths = [r["F_f_N"] for r in rows]
    exponent = _slope(list(rates), strengths)
    assert exponent == pytest.approx(analytic_exponent(), abs=0.02), (
        f"measured exponent {exponent} vs analytic {analytic_exponent()}")
    # And each point sits on the closed-form creep force, not just the slope.
    for rate, row in zip(rates, rows):
        assert row["F_f_N"] == pytest.approx(analytic_creep_force(rate), rel=2e-3)


def test_creep_brittle_transition_matches_the_time_comparison():
    """Below the crossover the bond fails by creep; above it, by cleavage."""
    below = run_one("linear+creep", 0.3)
    above = run_one("linear+creep", 3.0)
    assert below["failure_mode"] == 4
    assert above["failure_mode"] == 1
    assert above["F_f_N"] == pytest.approx(analytic_brittle_force(), rel=1e-3)


def test_maxwell_wiechert_arm_is_rate_stiffening():
    """Arm C: relaxation spectrum stiffens the instantaneous response."""
    low = run_one("maxwell_wiechert", 0.03)
    high = run_one("maxwell_wiechert", 3.0)
    assert low["failure_mode"] == 1 and high["failure_mode"] == 1
    assert high["F_f_N"] > low["F_f_N"], "MW bond must be stiffer at higher rate"
    # The relaxed (slow) strength is below the unrelaxed brittle bound.
    assert low["F_f_N"] < analytic_brittle_force()


def test_maxwell_wiechert_creep_arm_exponent_is_near_ductile():
    """Arm D: the physical model's creep-window exponent is near 1/(m+1)."""
    rates = (0.03, 0.1, 0.3)
    rows = [run_one("maxwell_wiechert+creep", r) for r in rates]
    for row in rows:
        assert row["failure_mode"] == 4
    exponent = _slope(list(rates), [r["F_f_N"] for r in rows])
    # The relaxation spectrum adds rate stiffening, so the exponent is at or
    # above the pure-creep value; it must not collapse to the brittle plateau.
    assert analytic_exponent() - 0.02 <= exponent <= analytic_exponent() + 0.15
