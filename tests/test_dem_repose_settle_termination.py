"""Tests of the adaptive rest criterion of the repose forward model.

``tensordem.dem_calibration.heap.run_heap`` historically integrated a *fixed*
``hold + lift + settle`` window, so the reported repose angle was whatever the
heap looked like when the clock ran out - not the angle of a heap at rest.  The
optional ``ke_stop`` argument terminates on the physical criterion
``K / released_PE < ke_stop`` instead.  These tests pin down

* backward compatibility: with ``ke_stop=None`` the trajectory and the
  reproducibility digest are unchanged;
* the termination semantics: ``settled`` is True only if the criterion was
  actually met, and ``steps_used`` never exceeds the step budget;
* a monotone consequence: a looser tolerance cannot stop *later* than a
  stricter one;
* the residual-energy check does not perturb the integrator.
"""
from __future__ import annotations

import numpy as np

from tensordem.dem_calibration.heap import HeapCase, run_heap


def _column_case(**overrides) -> HeapCase:
    """Cone relaxation of the shipped sand case (fittable flank, ~1.5 s)."""
    base = dict(radius=0.003, density=2650.0, mu=0.45, mu_rolling=0.10,
                restitution=0.38, stiffness_ratio=2.2, kn=800.0, gravity=9.81,
                cylinder_radius=0.021, fill_height=0.042, packing_gap=0.03,
                jitter=0.02, hold_time=0.03, settle_time=0.0654,
                shape="cone", seed=32001)
    base.update(overrides)
    return HeapCase(**base)


def _cone_case(**overrides) -> HeapCase:
    """Alias of the fast cone case above (measurable at every instant)."""
    return _column_case(**overrides)


def test_default_window_is_unchanged_and_reproducible():
    case = _column_case()
    a = run_heap(case, max_steps=300)
    b = run_heap(case, max_steps=300)
    assert a["digest"] == b["digest"]
    assert a["settled"] is False            # no criterion requested
    assert a["steps_used"] == a["n_steps"]
    assert a["ke_stop"] is None


def test_loose_tolerance_settles_and_stops_inside_budget():
    case = _cone_case(settle_time=0.05)
    out = run_heap(case, ke_stop=2.0, settle_check_every=10, max_steps=20000)
    assert out["settled"] is True
    assert out["ke_ratio"] < 2.0
    assert out["steps_used"] <= out["n_steps"]
    # the criterion may only be applied after the prescribed driving phase
    assert out["time_end"] >= (case.hold_time if case.shape == "cone"
                               else case.hold_time + case.lift_time)
    # the response is still measurable (a settled cone keeps its flank)
    assert np.isfinite(out["theta_r_deg"])


def test_criterion_never_met_reports_not_settled():
    case = _column_case()
    out = run_heap(case, ke_stop=0.0, settle_check_every=10, max_steps=400)
    assert out["settled"] is False
    assert out["steps_used"] == 400
    assert out["time_end"] == 400 * out["dt"]


def test_tolerance_is_monotone():
    """A looser tolerance cannot stop later than a stricter one: the stop time
    is the first check where the residual energy is below tolerance."""
    case = _cone_case()
    strict = run_heap(case, ke_stop=1e-9, settle_check_every=10, max_steps=20000)
    loose = run_heap(case, ke_stop=5.0, settle_check_every=10, max_steps=20000)
    assert strict["settled"] is False
    assert loose["settled"] is True
    assert loose["steps_used"] < strict["steps_used"]
    assert loose["ke_ratio"] < 5.0


def test_check_interval_does_not_change_the_physics():
    """The integrator is untouched: same case, same digest, whatever the
    residual-energy check interval is."""
    case = _column_case()
    a = run_heap(case, ke_stop=0.0, settle_check_every=7, max_steps=300)
    b = run_heap(case, ke_stop=None, max_steps=300)
    assert a["digest"] == b["digest"]
    assert np.isclose(a["theta_r_deg"], b["theta_r_deg"])