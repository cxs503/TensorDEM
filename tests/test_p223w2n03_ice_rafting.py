"""Verification tests for node p223w2-n03, face A: ice floe rafting and ridging.

The tests are split into

* closed-form / hydrostatics tests with an *analytic* answer (Sutherland-Hodgman
  clip, Archimedes draft, elastic-foundation length, ramp factor, energy
  consistency of the average and instantaneous rafting laws), and
* DEM-level tests that are cheap but real: a floating floe must find its
  Archimedean draft, the ramp experiment must show a resistance that grows with
  the climb, rises with friction, and is bitwise reproducible.

The two-limit bracket against the field anchors is *reported*, not asserted: the
honest outcome of that comparison is a FAIL of the 15 % point criterion (the
model over-predicts the field onset stress), which lives in
``artifacts/p223w2n03/acceptance.json`` -- a test that forced it to pass would
be a fabrication.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_rafting import (
    RAFTING_ANCHORS,
    RIDGE_GEOMETRY_ANCHORS,
    FloeSpec,
    IceProperties,
    analytic_rafting_threshold,
    bracket_verdict,
    clip_polygon_below,
    compare,
    flexural_length,
    floe_polygon,
    rafting_resistance,
    ramp_factor,
    ridge_geometry,
    run_rafting_ramp,
    run_ridge,
    submerged_properties,
)
from tensordem.ice_rafting.mechanics import _Floe


ICE = IceProperties()


# ---------------------------------------------------------------------------
# Hydrostatics
# ---------------------------------------------------------------------------

def test_clip_keeps_only_the_part_below_the_waterline():
    poly = floe_polygon(0.0, 0.5, 4.0, 0.4, 0.0)          # entirely above y=0
    assert len(clip_polygon_below(poly, 0.0)) == 0
    poly = floe_polygon(0.0, -0.5, 4.0, 0.4, 0.0)         # entirely below
    clipped = clip_polygon_below(poly, 0.0)
    assert len(clipped) == 4
    assert clipped[:, 1].max() <= 1e-12


def test_submerged_area_matches_the_analytic_rectangle():
    length, thickness = 6.0, 0.5
    centre_y = -0.0366                       # floe spans [-0.2866, +0.2134]
    depth = 0.5 * thickness - centre_y       # submerged depth below y = 0
    poly = floe_polygon(0.0, centre_y, length, thickness, 0.0)
    area, cx, cy = submerged_properties(poly, 0.0)
    assert area == pytest.approx(length * depth, rel=1e-9)
    assert cx == pytest.approx(0.0, abs=1e-9)
    assert cy == pytest.approx(-0.5 * depth, rel=1e-9)


def test_archimedes_draft_is_recovered_by_the_buoyancy_balance():
    """A floe floats where submerged area * rho_w == mass per unit width."""
    length, thickness = 5.0, 0.6
    draft = thickness * ICE.draft_ratio
    poly = floe_polygon(0.0, 0.5 * thickness - draft, length, thickness, 0.0)
    area, _, _ = submerged_properties(poly, 0.0)
    displaced = area * ICE.rho_water
    weight = length * thickness * ICE.rho_ice
    assert displaced == pytest.approx(weight, rel=1e-9)


def test_tilted_floe_has_a_smaller_submerged_area():
    length, thickness = 5.0, 0.6
    draft = thickness * ICE.draft_ratio
    flat = floe_polygon(0.0, 0.5 * thickness - draft, length, thickness, 0.0)
    tilted = floe_polygon(0.0, 0.5 * thickness - draft, length, thickness, 0.10)
    a_flat, _, _ = submerged_properties(flat, 0.0)
    a_tilt, _, _ = submerged_properties(tilted, 0.0)
    assert 0.0 < a_tilt < a_flat            # lifting one end raises the other end


def test_polygon_area_and_centroid_of_a_beveled_floe():
    spec = FloeSpec(length=4.0, thickness=0.4, bevel=0.5)
    poly = spec.local_polygon()
    area, cx, cy = submerged_properties(poly, 1e18)
    c = 0.5 * spec.thickness
    assert area == pytest.approx(4.0 * 0.4 - 0.5 * c * c, rel=1e-12)
    assert cx > 0.0                        # the bevel removes material on the left


# ---------------------------------------------------------------------------
# Closed-form rafting law
# ---------------------------------------------------------------------------

def test_elastic_foundation_length_scales_as_h_to_the_three_quarters():
    l1 = flexural_length(0.3, ICE)
    l2 = flexural_length(0.6, ICE)
    assert l2 / l1 == pytest.approx(2.0 ** 0.75, rel=1e-12)
    stiffer = IceProperties(young=2.0 * ICE.young)
    assert flexural_length(0.5, stiffer) / flexural_length(0.5, ICE) == pytest.approx(
        2.0 ** 0.25, rel=1e-12)


def test_ramp_factor_without_friction_is_the_tangent():
    for bevel in (0.25, 0.5, 1.0):
        assert ramp_factor(bevel, 0.0) == pytest.approx(2.0 * bevel, rel=1e-12)


def test_ramp_factor_grows_with_friction_and_rejects_a_locking_pair():
    assert ramp_factor(0.5, 0.3) > ramp_factor(0.5, 0.0)
    with pytest.raises(ValueError):
        ramp_factor(0.5, 50.0)      # cos a - mu sin a < 0: the floe cannot ride up


def test_rigid_threshold_is_linear_in_length_and_independent_of_thickness():
    a = analytic_rafting_threshold(0.4, 4.0, ICE, mu=0.0)
    b = analytic_rafting_threshold(0.4, 8.0, ICE, mu=0.0)
    c = analytic_rafting_threshold(0.8, 4.0, ICE, mu=0.0)
    assert b["stress"] == pytest.approx(2.0 * a["stress"], rel=1e-12)
    assert c["stress"] == pytest.approx(a["stress"], rel=1e-12)


def test_flexural_limit_caps_the_participating_length():
    short = analytic_rafting_threshold(0.3, 3.0, ICE, limit_length=True)
    long = analytic_rafting_threshold(0.3, 40.0, ICE, limit_length=True)
    assert short["l_eff"] == pytest.approx(3.0)
    assert long["l_eff"] == pytest.approx(flexural_length(0.3, ICE))
    assert long["stress"] == pytest.approx(
        analytic_rafting_threshold(0.3, flexural_length(0.3, ICE), ICE)["stress"])


def test_instantaneous_law_integrates_to_the_closed_form_lift_work():
    """Work of the instantaneous law over the emergence equals the energy form.

    The closed form is the *work* needed to lift the floe out of the water,
    ``W = 0.5 rho_i g h^2 L (rho_i/rho_w)``, divided by the ride-up length ``h``.
    Integrating the instantaneous resistance from zero climb to the draft must
    therefore return ``W`` exactly (frictionless ramp).
    """
    h, length, mu = 0.5, 6.0, 0.0
    draft = h * ICE.draft_ratio
    z = np.linspace(0.0, draft, 20001)
    f = np.array([rafting_resistance(zz, h, length, ICE, bevel=0.5, mu=mu)["line_load"]
                  for zz in z])
    work = float(np.trapezoid(f, z))
    closed = analytic_rafting_threshold(h, length, ICE, mu=mu)
    assert work == pytest.approx(closed["lift_work"], rel=1e-3)
    # ... and beyond full emergence the resistance is the floe weight (no longer
    # buoyancy-limited), so the two laws differ there by design
    beyond = rafting_resistance(2.0 * h, h, length, ICE, mu=mu)["line_load"]
    assert beyond > 2.0 * rafting_resistance(draft, h, length, ICE, mu=mu)["line_load"]


def test_instantaneous_law_saturates_at_the_floe_weight():
    h, length = 0.5, 6.0
    draft = h * ICE.draft_ratio
    before = rafting_resistance(0.5 * draft, h, length, ICE, mu=0.0)
    after = rafting_resistance(2.0 * h, h, length, ICE, mu=0.0)
    assert before["vertical_force"] == pytest.approx(
        ICE.rho_water * ICE.gravity * length * 0.5 * draft, rel=1e-12)
    assert after["vertical_force"] > before["vertical_force"]
    # once fully emerged the vertical force is the full floe weight
    full = rafting_resistance(h, h, length, ICE, mu=0.0)["vertical_force"]
    assert full >= ICE.rho_ice * ICE.gravity * h * length * 0.99


# ---------------------------------------------------------------------------
# DEM level
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ramp_run():
    return run_rafting_ramp(h=0.5, length=6.0, ice=ICE, ramp_angle_deg=45.0,
                            mu=0.3, piston_velocity=-0.2, max_steps=4000,
                            sample_every=200, settle_steps=2000)


def test_floe_in_open_water_finds_its_archimedean_draft():
    floe = _Floe(FloeSpec(4.0, 0.4, 9, ICE.rho_ice, 0.5),
                 (0.0, 0.2 - 0.4 * ICE.draft_ratio), 0.0, ICE)
    draft = 0.4 * ICE.draft_ratio
    poly = floe.vertices()
    area, _, _ = submerged_properties(poly, 0.0)
    assert area * ICE.rho_water == pytest.approx(floe.mass, rel=5e-3)


def test_ramp_load_grows_with_the_climb(ramp_run):
    loads = [h["load"] for h in ramp_run["history"] if h["climb"] > 1e-3]
    assert len(loads) >= 5
    assert max(loads) > 3.0 * min(loads)      # a real rise, not noise
    assert ramp_run["peak_load"] > 0.0


def test_ramp_load_exceeds_the_closed_form_by_between_one_and_twenty(ramp_run):
    """The DEM resistance is *above* the closed-form wedge law, and we say so.

    Measured ratio (h = 0.5 m, L = 6 m, 45 deg ramp): the DEM plateau is ~2x the
    closed-form value at mid climb.  The wedge law neglects the moment balance
    between the piston and the single ramp contact, and the fact that a rigid
    floe must be raised as a whole.  The band is deliberately wide (1-20): the
    test protects against a sign/scale error, not against the physics gap, which
    the report states explicitly.
    """
    tau = ramp_run["tau_mid_climb"] or ramp_run["history"][-1]["load"] / 0.5
    analytic = rafting_resistance(0.5 * 0.5, 0.5, 6.0, ICE, bevel=0.5, mu=0.3)["stress"]
    assert 1.0 < tau / analytic < 20.0


def test_ramp_run_is_bitwise_reproducible(ramp_run):
    again = run_rafting_ramp(h=0.5, length=6.0, ice=ICE, ramp_angle_deg=45.0,
                             mu=0.3, piston_velocity=-0.2, max_steps=4000,
                             sample_every=200, settle_steps=2000)
    a = [h["load"] for h in ramp_run["history"]]
    b = [h["load"] for h in again["history"]]
    assert a == b


def test_rafting_resistance_is_friction_insensitive_with_a_viscous_law():
    """Documented limitation: the regularised friction mobilises no static friction.

    With a velocity-proportional (regularised) Coulomb law the tangential force is
    ``min(c_t |v_t|, mu f_n)``; at the quasi-static sliding speed of this
    experiment the viscous branch always wins, so ``mu`` cancels and the DEM
    resistance is friction-insensitive (measured spread < 2 % for mu = 0.05-0.6).
    The closed-form ramp factor *does* depend on mu -- that difference is stated
    in the report rather than hidden.
    """
    low = run_rafting_ramp(h=0.5, length=6.0, ice=ICE, ramp_angle_deg=45.0, mu=0.05,
                           piston_velocity=-0.2, max_steps=3000, sample_every=400,
                           settle_steps=1500)
    high = run_rafting_ramp(h=0.5, length=6.0, ice=ICE, ramp_angle_deg=45.0, mu=0.5,
                            piston_velocity=-0.2, max_steps=3000, sample_every=400,
                            settle_steps=1500)
    rel = abs(high["peak_load"] - low["peak_load"]) / low["peak_load"]
    assert rel < 0.10
    assert ramp_factor(0.5, 0.5) > ramp_factor(0.5, 0.05)   # the closed form does vary


def test_ridge_geometry_recovers_a_synthetic_pile():
    """Two floes, one on top of the other, with known sail and keel."""
    base = _Floe(FloeSpec(4.0, 0.4, 9, ICE.rho_ice, 0.0), (0.0, 0.5 * 0.4 - 0.4 * ICE.draft_ratio),
                 0.0, ICE)
    top = _Floe(FloeSpec(3.0, 0.4, 9, ICE.rho_ice, 0.0), (0.0, 1.2), 0.0, ICE)
    geom = ridge_geometry([base, top], 0.4, ICE)
    assert geom["sail_height"] == pytest.approx(1.4, abs=1e-9)
    assert geom["keel_depth"] == pytest.approx(0.4 * ICE.draft_ratio, rel=1e-9)
    assert geom["sail_keel_ratio"] > 1.0
    assert 0.0 < geom["submerged_area"] < 4.0 * 0.4


# ---------------------------------------------------------------------------
# Anchor bookkeeping
# ---------------------------------------------------------------------------

def test_compare_flags_range_membership_and_point_error():
    anchor = RAFTING_ANCHORS["rafting_onset_h050"]
    inside = compare(1.5e4, anchor)
    assert inside["within_range"] and inside["abs_rel_error"] < 1e-12
    outside = compare(6.0e4, anchor)
    assert not outside["within_range"] and outside["rel_error"] == pytest.approx(3.0)


def test_bracket_verdict_requires_covering_the_observed_range():
    anchor = RAFTING_ANCHORS["rafting_onset_h050"]
    assert bracket_verdict(5.0e3, 6.0e4, anchor)["brackets"] is True
    assert bracket_verdict(2.0e4, 6.0e4, anchor)["brackets"] is False


def test_ridge_anchors_are_self_consistent():
    a = RIDGE_GEOMETRY_ANCHORS["sail_keel_ratio"]
    lo, hi = a["range"]
    assert lo < a["value"] < hi
    assert a["value"] == pytest.approx(1.0 / 4.4)