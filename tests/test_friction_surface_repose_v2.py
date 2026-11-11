"""Pour-protocol (v2) regression tests for the repose anchor.

Three things are pinned here, each of which was a measured defect or a measured
pathology of protocol v1 (`src/tensordem/friction_surface/repose.py`):

1. **spawn overflow** — v1 wrote ``pos[active]`` from a burst loop whose only
   guard was checked once per step, so a burst that had *partially* refused its
   spheres (the normal case once a heap exists) ran one index past ``n_max`` and
   raised ``IndexError``.  That is one of the reasons anchor A/B of node
   P152W1-N04 came back WITHHELD.
2. **climbing nozzle** — v1 raised the source to ``max(drop_height, apex + 5 R)``
   at every burst, the protocol the node report itself lists as rejected
   (§5.3).  Measured: the nozzle climbs to 1.44 m, the "heap" becomes a spire and
   the at-rest criterion is never reached.  v2 keeps the nozzle at
   ``drop_height`` and paces the bursts by the charge fall time.
3. **pour closure** — the pour must stop when the deposit fills the nozzle, and
   must not be resumed; a fixed nozzle cannot pour over its own heap.

The tests use small systems and short windows on purpose: they are unit-level
regressions of the protocol, not the anchor itself (the anchor numbers live in
``artifacts/P152W1-N04/repose.json``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.friction_surface.repose import (
    ReposeCase, heap_surface_angle, nozzle_clear_time, simulate_repose,
)


def _fast(**kw) -> ReposeCase:
    """A small, short case: same physics, a few hundred steps."""
    base = dict(radius=2.0e-3, mu=0.5, mu_rolling=0.10, spawn_batch=4,
                spawn_every=20, source_radius=4.0e-3, settle_time=0.02,
                max_time=0.06, velocity_rest_threshold=0.05)
    base.update(kw)
    return ReposeCase(**base)


def test_spawn_never_overflows_the_particle_array():
    """A batch that is partly refused must not write past ``n_max`` (v1 IndexError)."""
    case = _fast(n_particles=7)          # not a multiple of spawn_batch
    res = simulate_repose(case)          # would raise IndexError under v1
    assert res["n_particles"] <= case.n_particles
    assert res["final_positions"].shape[0] == res["n_particles"]


def test_fixed_nozzle_does_not_climb_with_the_heap():
    """v2: the nozzle height is a constant of the case, independent of the heap."""
    case = _fast(n_particles=60, drop_height=0.020)
    res = simulate_repose(case)
    assert res["source_mode"] == "fixed"
    assert res["n_particles"] > 0
    # exact: the nozzle is never moved, whatever the deposit does
    assert res["nozzle_z_max"] == pytest.approx(case.drop_height, abs=1e-12)


def test_legacy_rising_source_reproduces_the_spire_pathology():
    """v1 behaviour is kept and is measurably wrong: the nozzle climbs each burst."""
    case = _fast(n_particles=40, source_mode="rising", drop_height=0.020)
    res = simulate_repose(case)
    assert res["source_mode"] == "rising"
    # 10 bursts, each raising the source by 5 R above the current apex
    assert res["nozzle_z_max"] > 2.0 * case.drop_height
    assert res["nozzle_z_max"] > res["apex_height_m"]


def test_pour_closes_when_the_deposit_reaches_the_nozzle():
    """A fixed nozzle stops pouring once the deposit sits in the nozzle slab."""
    # unit test of the closure rule, not of the physics: a low nozzle (3 radii)
    # puts the nozzle slab just above the floor, and a fast fat charge fills it
    # within the short window.
    case = _fast(n_particles=400, drop_height=3.0 * 2.0e-3, spawn_batch=8,
                 release_speed=1.0, max_time=0.02)
    res = simulate_repose(case)
    assert res["pour_closed"] is True
    assert res["bursts_released"] < case.n_particles // case.spawn_batch
    # the nozzle is closed by geometry, so it is never buried by the deposit
    assert res["nozzle_z_max"] == pytest.approx(case.drop_height, abs=1e-12)


def test_bursts_are_paced_by_the_charge_fall_time():
    """The v2 burst pacing is the physical fall time, not a tuned step count."""
    case = _fast(release_speed=0.05, nozzle_clearance=2.0)
    r = case.radius
    g = case.model.gravity
    s_fall = (2.0 + 2.2) * r
    v0 = case.release_speed
    t_exact = (-v0 + math.sqrt(v0 * v0 + 2.0 * g * s_fall)) / g
    assert nozzle_clear_time(case) == pytest.approx(t_exact, rel=1e-12)

    res = simulate_repose(case)
    assert res["clear_steps"] == max(
        int(round(t_exact / res["dt"])), case.spawn_every)
    assert res["bursts_skipped"] >= 0


def test_at_rest_bookkeeping_is_self_consistent():
    """``reached_rest`` must mean "the RMS criterion fired", not "the clock ran out"."""
    case = _fast(n_particles=120, mu=0.8, mu_rolling=0.30, settle_time=0.15,
                 max_time=0.60, velocity_rest_threshold=0.03, source_radius=3.0e-3)
    res = simulate_repose(case)
    assert res["reached_rest"] == (res["t_at_rest"] is not None)
    if res["reached_rest"]:
        assert res["t_at_rest"] < res["t_end"] + res["dt"]
    assert res["n_particles"] > 0
    m = res["measurement"]
    if m["n_bins_used"] >= 3:
        assert np.isfinite(m["angle_deg"])


def test_heap_surface_angle_recovers_a_synthetic_cone():
    """The measurement layer is unchanged and still recovers a known slope."""
    theta = math.radians(30.0)
    rng = np.random.default_rng(7)
    pts = []
    for _ in range(4000):
        r = 0.10 * math.sqrt(rng.random())
        # a heap falls with radius: z = (r_max - r) tan(theta)
        z = (0.10 - r) * math.tan(theta)
        a = rng.random() * 2.0 * math.pi
        pts.append((0.10 + r * math.cos(a), 0.10 + r * math.sin(a),
                    z + rng.random() * 2.0e-3))
    m = heap_surface_angle(np.asarray(pts), 2.0e-3)
    assert m["angle_deg"] == pytest.approx(30.0, abs=2.0)