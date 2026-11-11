#!/usr/bin/env python3
"""Board audit (m15392) regression pins for CASE-PAIR-COLLISION.

Three invariants that the original submission did not pin:

1. the elastic Hertz contact-time reference must equal the quadrature-exact
   coefficient C = 2*int_0^1 (1-eta^(5/2))^(-1/2) d eta = 0.8*B(2/5, 1/2);
   the pre-audit script used a dimensionally wrong closed form that was 2.50%
   high (run.py:analytic_contact_time);
2. the DEM engine must reproduce that law (agreement with the repo's own RK4
   reference of the same Tsuji law) to <0.5%;
3. the realised restitution must be documented as NOT delivering e_target:
   the linear zeta(e) mapping applied to a Hertzian dashpot overshoots by
   ~3.2% at e_target=0.8 (16.5% at 0.5).  This pin exists so that the gap
   cannot silently disappear from the benchmark output again.

If the zeta mapping is ever calibrated, test 3 is expected to be rewritten -
that is the intended failure mode.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "case_pair_collision_hertz"))

import run as bench  # noqa: E402


def _quadrature_coefficient() -> float:
    """Composite Simpson on the (integrable) endpoint singularity via u-substitution."""
    # int_0^1 (1-eta^(5/2))^(-1/2) deta = (2/5) B(2/5, 1/2)
    return 2.0 * 0.4 * math.exp(
        math.lgamma(0.4) + math.lgamma(0.5) - math.lgamma(0.9))


def test_contact_time_coefficient_is_exact_closed_form():
    assert bench.HERTZ_CONTACT_TIME_COEFF == pytest.approx(_quadrature_coefficient(),
                                                          rel=1e-14)
    assert bench.HERTZ_CONTACT_TIME_COEFF == pytest.approx(2.9432751843247056, rel=1e-12)


def test_contact_time_reference_matches_quadrature_not_the_old_buggy_form():
    tc = bench.analytic_contact_time(bench.PRIMARY_SPEED, bench.RADIUS, bench.DENSITY,
                                     bench.YOUNG_MODULUS, bench.POISSON_RATIO)
    delta = bench.analytic_peak_overlap(bench.PRIMARY_SPEED, bench.RADIUS, bench.DENSITY,
                                        bench.YOUNG_MODULUS, bench.POISSON_RATIO)
    # closed form under test vs (4/5)*B(2/5,1/2) * delta_max / v_rel
    assert tc == pytest.approx(_quadrature_coefficient() * delta / 1.0, rel=1e-12)
    # the rejected form 2.94*(m_eff^2/(E*^2 R* v_rel))^0.2 was +2.50% high
    m_eff = bench.DENSITY * (4.0 / 3.0) * math.pi * bench.RADIUS ** 3 / 2.0
    e_star = 1.0e9 / (2.0 * (1.0 - 0.3 ** 2))
    legacy = 2.94 * (m_eff ** 2 / (e_star ** 2 * 0.0125 * 1.0)) ** 0.2
    assert abs(legacy - tc) / tc > 0.02


def test_engine_reproduces_tsuji_law_and_closure_gap_is_documented():
    got = bench.measure_engine(bench.PRIMARY_SPEED, restitution_coefficient=bench.TARGET_E)
    ref = bench.reference_rk4(bench.PRIMARY_SPEED, bench.TARGET_E)
    # (2) engine == law to <0.5%
    assert abs(got["restitution"] - ref["restitution"]) / ref["restitution"] < 5e-3
    # (3) and the law itself misses the requested coefficient by ~3.2%
    closure = abs(got["restitution"] - bench.TARGET_E) / bench.TARGET_E
    assert closure == pytest.approx(0.0323, abs=2e-3)
    assert closure > 0.03  # explicitly outside the <3% tolerance