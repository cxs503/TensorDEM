"""Tests for the leave-one-level-out check of a refinement extrapolation.

``stiffness_limit`` always returns an intercept; these tests pin down when that
intercept is *checkable*, i.e. when a fit made on the coarser levels predicts
the withheld stiffest one.  The negative controls matter more than the positive
ones: a check that cannot fail is not a check (cf. the three-level fit of the
RCP case, which by construction has zero residual).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tensordem.deposition_packing import (
    extrapolation_selfconsistency, stiffness_limit)


def _ladder(limit: float, amp: float, order: float, levels=(4000, 16000, 64000, 256000)):
    return [(float(k), limit + amp * float(k) ** (-order)) for k in levels]


def test_exact_power_law_is_predicted_and_stable():
    """A clean ``limit + A k^-p`` ladder is reproduced and nothing moves."""
    sc = extrapolation_selfconsistency(_ladder(0.6366, 120.0, 1.0))
    assert sc["prediction_error"] == pytest.approx(0.0, abs=1e-12)
    assert sc["limit_shift"] == pytest.approx(0.0, abs=1e-12)
    assert sc["full"]["limit"] == pytest.approx(0.6366, abs=1e-12)
    assert sc["fit"]["limit"] == pytest.approx(0.6366, abs=1e-12)
    for row in sc["pair_orders"]:
        assert row["order"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("order", [0.5, 1.0, 2.0])
def test_observed_order_is_recovered(order):
    """The adjacent-pair exponent is measured, not assumed."""
    sc = extrapolation_selfconsistency(_ladder(0.60, 40.0, order))
    for row in sc["pair_orders"]:
        assert row["order"] == pytest.approx(order, rel=1e-9)
    assert sc["full"]["limit"] == pytest.approx(0.60, abs=1e-10)


def test_three_levels_cannot_be_self_checked():
    """Three levels fix three parameters -> no out-of-sample content exists."""
    with pytest.raises(ValueError):
        extrapolation_selfconsistency(_ladder(0.6366, 120.0, 1.0)[:3])


@pytest.mark.parametrize("offset,gate_passes", [(1e-3, True), (5e-3, False)])
def test_deviating_level_is_detected_and_sized(offset, gate_passes):
    """Negative control: a level that leaves the power law must show up, and a
    deviation of half the gate must still pass -- the check has a resolution."""
    clean = _ladder(0.6366, 120.0, 1.0)
    polluted = clean[:3] + [(clean[-1][0], clean[-1][1] + offset)]
    sc = extrapolation_selfconsistency(polluted)
    # the withheld level differs from the 3-level model by exactly the injection
    assert sc["prediction_error"] == pytest.approx(offset, rel=1e-6)
    # and admitting it drags the intercept
    assert sc["limit_shift"] > 0.4 * offset
    detected = sc["prediction_error"] < 2e-3 and sc["limit_shift"] < 2e-3
    assert detected == gate_passes


def test_order_drift_is_visible_in_the_pair_orders():
    """A ladder that changes exponent shows one order per pair, i.e. no single p.

    Constructed so that pair 1 is exactly ``k^-1`` and pair 2 exactly
    ``k^-1/2`` (the shape the frictional RCP ladder actually has).
    """
    lim, amp = 0.6366, 120.0
    v0 = lim + amp / 4000 ** 1
    v1 = lim + amp / 16000 ** 1
    v2 = lim + amp / 64000 ** 1
    v3 = v2 + 0.5 * (v2 - v1)          # halved decrement -> observed order 0.5
    sc = extrapolation_selfconsistency([(4000.0, v0), (16000.0, v1),
                                        (64000.0, v2), (256000.0, v3)])
    orders = [row["order"] for row in sc["pair_orders"]]
    assert orders[0] == pytest.approx(1.0, abs=1e-9)
    assert orders[1] == pytest.approx(0.5, abs=1e-9)
    assert abs(orders[0] - orders[1]) > 0.4
    # a drifting order means the intercept is NOT the clean limit
    assert sc["limit_shift"] > 1e-3


def test_non_monotone_ladder_does_not_crash_and_is_flagged():
    """Non-monotone measurements must fall back, not invent an order."""
    pts = [(4000.0, 0.6360), (16000.0, 0.6364), (64000.0, 0.6364), (256000.0, 0.6360)]
    sc = extrapolation_selfconsistency(pts)
    assert sc["full"]["order"] is None            # no clean order exists
    assert sc["full"]["order_used"] == pytest.approx(1.0)   # documented default
    assert np.isfinite(sc["prediction_error"]) and np.isfinite(sc["limit_shift"])


@pytest.mark.parametrize("pts", [
    [(4000.0, 0.63), (4000.0, 0.64), (16000.0, 0.635), (64000.0, 0.63)],
    [(-4000.0, 0.63), (16000.0, 0.635), (64000.0, 0.63), (256000.0, 0.63)],
])
def test_invalid_ladders_raise(pts):
    with pytest.raises(ValueError):
        extrapolation_selfconsistency(pts)


def test_pure_function_agrees_with_stiffness_limit_on_the_full_ladder():
    """No second model: ``full`` is literally ``stiffness_limit`` on all levels."""
    pts = _ladder(0.6366, 120.0, 1.0)
    sc = extrapolation_selfconsistency(pts)
    ref = stiffness_limit(pts)
    assert sc["full"]["limit"] == pytest.approx(ref["limit"], abs=0)
    assert sc["full"]["order_used"] == pytest.approx(ref["order_used"], abs=0)


# ---------------------------------------------------------------------------
# served-data check: the numbers in the audit document are reproducible
# ---------------------------------------------------------------------------
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "benchmarks", "case_rcp_deposition", "runs")


def _audit():
    path = os.path.join(RUNS, "order_audit.json")
    if not os.path.exists(path):
        pytest.skip("order_audit.json not present (run order_audit.py first)")
    return json.load(open(path))


def test_served_audit_recomputes_from_the_run_files():
    """Recompute each tier's self-check straight from ``runs/*.json`` and compare."""
    audit = _audit()
    import glob
    runs = {}
    for p in glob.glob(os.path.join(RUNS, "deposition_*.json")):
        d = json.load(open(p))
        spec = d["spec"]
        key = (round(spec["mu"], 9), round(spec["kn"], 6), spec["n_particles"],
               round(spec.get("dt_frac", 0.05), 9))
        runs.setdefault(key, []).append(d["metrics"]["phi_bulk"])
    for tag, tier in audit["tiers"].items():
        mu = float(tag)
        pts = []
        for kn, phi in tier["points"]:
            phis = runs[(round(mu, 9), round(kn, 6), audit["n_primary"], 0.05)]
            assert np.mean(phis) == pytest.approx(phi, abs=1e-12)
            pts.append((kn, phi))
        if "selfcheck" not in tier:
            continue
        sc = extrapolation_selfconsistency(pts)
        assert sc["prediction_error"] == pytest.approx(
            tier["selfcheck"]["prediction_error"], abs=1e-12)
        assert sc["limit_shift"] == pytest.approx(
            tier["selfcheck"]["limit_shift"], abs=1e-12)


def test_served_gate_verdicts_are_consistent_with_their_numbers():
    """A PASS must be a PASS of the stated criterion, not of a relaxed one."""
    audit = _audit()
    gates = {g["id"]: g for g in audit["gates"]}
    for gid, g in gates.items():
        if "selfconsistency" in gid:
            got = g["prediction_error"] < 0.002 and g["limit_shift"] < 0.002
            assert got == g["pass"], gid
        elif "timestep_at_stiffest" in gid:
            assert (g["got"] < 0.002) == g["pass"], gid