"""Tests of the steady-state transit surface (``tensordem.ice_transit``).

The suite is deliberately cheap: every simulation uses a fraction of a ship
length of route, and the physics assertions are formulated as identities
(``dU/ddelta == F``, ledger closure, ``W_thrust = R V T``) rather than as
reference resistances, so nothing here can drift with a tuning change.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.broken_ice_resistance.hull import ShipForm
from tensordem.ice_channel import WaterEnvironment
from tensordem.ice_transit.campaign import (HEXAGONAL_LIMIT, STANDARD_CASE,
                                            _build_sim, case_spec,
                                            evaluate_criteria,
                                            reference_anchors, run_case)
from tensordem.ice_transit.laws import (CrushingLaw, floe_floe_contacts,
                                        hull_contacts, polygon_distance,
                                        wall_contacts)
from tensordem.ice_transit.transit import SteadyTransit, _max_radius_spread

SIGMA_C = 2.477e6
MODULUS = 1.474e9


# -- contact law ------------------------------------------------------------
def test_crushing_law_zero_force_at_zero_penetration():
    law = CrushingLaw(SIGMA_C, MODULUS)
    pen = torch.tensor([0.0], dtype=torch.float64)
    radius = torch.tensor([5.0], dtype=torch.float64)
    thick = torch.tensor([1.0], dtype=torch.float64)
    assert float(law.force(pen, radius, thick)[0]) == 0.0
    assert float(law.potential(pen, radius, thick)[0]) == 0.0


def test_crushing_law_offset_is_removed_contrary_to_the_legacy_law():
    """The legacy law carries sigma_c h 2 sqrt(2 R eps) at zero penetration."""
    law = CrushingLaw(SIGMA_C, MODULUS)
    radius = torch.tensor([5.0], dtype=torch.float64)
    thick = torch.tensor([1.0], dtype=torch.float64)
    eps = float(law.eps(radius))
    legacy_zero_pen_force = SIGMA_C * 1.0 * 2.0 * math.sqrt(2.0 * 5.0 * eps)
    assert legacy_zero_pen_force > 1.0e6           # 1.1 MN of spurious pre-load
    assert float(law.force(torch.tensor([0.0], dtype=torch.float64),
                           radius, thick)[0]) == 0.0


def test_crushing_law_potential_derivative_equals_force():
    law = CrushingLaw(SIGMA_C, MODULUS)
    pen = torch.linspace(0.0, 1.2, 4001, dtype=torch.float64)
    radius = torch.full_like(pen, 5.0)
    thick = torch.ones_like(pen)
    force = law.force(pen, radius, thick)
    potential = law.potential(pen, radius, thick)
    derivative = torch.gradient(potential, spacing=(pen,))[0]
    assert float((derivative - force).abs().max()) < 2.0e-3 * float(force.max())


def test_crushing_law_tangent_stiffness_is_finite_at_contact():
    law = CrushingLaw(SIGMA_C, MODULUS)
    pen = torch.zeros(1, dtype=torch.float64)
    radius = torch.tensor([5.0], dtype=torch.float64)
    thick = torch.tensor([1.0], dtype=torch.float64)
    stiffness = law.tangent_stiffness(pen, radius, thick)
    assert math.isfinite(float(stiffness[0])) and float(stiffness[0]) > 0.0


def test_crushing_law_saturates_at_the_floe_diameter():
    law = CrushingLaw(SIGMA_C, MODULUS)
    radius = torch.tensor([0.5], dtype=torch.float64)
    thick = torch.tensor([1.0], dtype=torch.float64)
    deep = law.force(torch.tensor([2.0], dtype=torch.float64), radius, thick)
    deeper = law.force(torch.tensor([20.0], dtype=torch.float64), radius, thick)
    assert float(deep[0]) == pytest.approx(float(deeper[0]))
    # saturated force = sigma_c h (2R - offset) <= sigma_c h 2R
    assert float(deep[0]) <= SIGMA_C * 1.0 * 2.0 * 0.5


def test_energy_consistent_contact_reports_its_potential():
    law = CrushingLaw(SIGMA_C, MODULUS)
    pos = torch.tensor([[0.0, 0.0], [9.5, 0.0]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    omega = torch.zeros(2, dtype=torch.float64)
    radius = torch.tensor([5.0, 5.0], dtype=torch.float64)
    thick = torch.ones(2, dtype=torch.float64)
    mass = 917.0 * math.pi * radius ** 2 * thick
    out = floe_floe_contacts(pos, vel, omega, radius, thick, mass, law, 0.1, 0.2, 0.2)
    assert out["count"] == 1
    assert float(out["potential"]) > 0.0
    assert float(out["p_normal_damp"]) == pytest.approx(0.0, abs=1e-9)


# -- hull geometry ----------------------------------------------------------
def test_polygon_distance_fixes_the_bow_apex_overestimation():
    """Face-plane distance puts a floe 7 m ahead of the stem in contact."""
    hull = ShipForm(length=105.0, beam=26.0, speed=3.0, entry_angle_deg=18.0,
                    x_start=0.0, y_position=0.0)
    normals, points = hull.outline(0.0)
    pos = torch.tensor([[7.0, 0.0]], dtype=torch.float64)
    face_plane, _ = hull.signed_distance(pos, 0.0)
    exact, normal = polygon_distance(pos, points, normals)
    assert float(face_plane[0]) < 0.5 * float(exact[0])      # the defect
    assert float(exact[0]) == pytest.approx(7.0, rel=1e-6)   # true distance
    assert float(torch.linalg.norm(normal[0])) == pytest.approx(1.0, rel=1e-6)


def test_polygon_distance_signs_inside_points():
    hull = ShipForm(length=105.0, beam=26.0, speed=3.0, entry_angle_deg=18.0,
                    x_start=0.0, y_position=0.0)
    normals, points = hull.outline(0.0)
    pos = torch.tensor([[-50.0, 0.0]], dtype=torch.float64)
    signed, _ = polygon_distance(pos, points, normals)
    assert float(signed[0]) < 0.0


# -- field construction -----------------------------------------------------
@pytest.mark.parametrize("concentration", [0.5, 0.7, 0.9])
def test_field_realises_the_requested_concentration(concentration):
    spec = STANDARD_CASE.with_(name="field", concentration=concentration,
                               route_ship_lengths=0.1)
    sim = _build_sim(spec)
    assert sim.realised_concentration() == pytest.approx(concentration, rel=0.01)
    assert sim.max_initial_overlap() < 1e-3


def test_field_refuses_concentration_above_the_hexagonal_limit():
    assert HEXAGONAL_LIMIT == pytest.approx(math.pi / (2.0 * math.sqrt(3.0)))
    with pytest.raises(ValueError):
        STANDARD_CASE.with_(concentration=0.99)


def test_radius_spread_limit_is_consistent_with_the_hexagonal_limit():
    assert _max_radius_spread(HEXAGONAL_LIMIT) == pytest.approx(0.0, abs=1e-6)
    assert _max_radius_spread(0.5) > _max_radius_spread(0.7) > _max_radius_spread(0.9)


def test_initial_state_has_no_contact_or_pack_energy():
    sim = _build_sim(STANDARD_CASE.with_(name="field", concentration=0.9,
                                         route_ship_lengths=0.1))
    energy = sim.stored_energy()
    assert energy["contact"] < 1.0e3      # < 1 kJ against ~10^5 kJ of thrust work
    assert energy["pack_wall"] < 1.0e3


# -- ledger and dynamics ----------------------------------------------------
def test_energy_ledger_closes_on_a_short_transit():
    spec = STANDARD_CASE.with_(name="ledger", concentration=0.5,
                               route_ship_lengths=0.25, steady_start_s=5.0)
    payload = run_case(spec)
    assert payload["ledger_residual_rel"] < 0.10
    total_diss = sum(payload["dissipation_kJ"].values())
    assert total_diss > 0.0


def test_thrust_work_is_the_integral_of_resistance_times_speed():
    spec = STANDARD_CASE.with_(name="thrust", concentration=0.5,
                               route_ship_lengths=0.25, steady_start_s=5.0)
    payload = run_case(spec)
    start, end = payload["steady_window_s"]
    window = [row for row in payload["series"] if start <= row["time_s"] <= end]
    integral = sum(row["resistance_kN"] for row in window) * 1e3 \
        * spec.speed_m_s * (window[-1]["time_s"] - window[0]["time_s"]) / len(window)
    mean_based = payload["mean_kN"] * 1e3 * spec.speed_m_s * (end - start)
    assert integral == pytest.approx(mean_based, rel=0.10)
    # the thrust work of the whole run exceeds the steady-window integral
    assert payload["thrust_work_kJ"] * 1e3 > integral


def test_null_field_has_zero_resistance():
    payload = run_case(case_spec("no_ice").with_(route_ship_lengths=0.2,
                                                steady_start_s=4.0))
    assert abs(payload["mean_kN"]) < 1.0
    assert payload["floe_count"] == 1


def test_recycling_inlet_conserves_floe_count_and_books_the_removal():
    """A floe abaft the stern re-enters ahead of the stem, its energy booked."""
    sim = _build_sim(STANDARD_CASE.with_(name="recycle", concentration=0.5,
                                         route_ship_lengths=0.1))
    n0 = sim.n
    sim.vel = torch.full((sim.n, 2), 0.4, dtype=torch.float64)
    # move the *first lattice column* behind the stern: those floes share an x but
    # have distinct y, so shifting them leaves the field itself non-overlapping
    first_column = torch.argsort(sim.pos[:, 0])[:4]
    sim.pos[first_column, 0] = sim.x_stem() - sim.window_downstream - 5.0
    assert sim.max_initial_overlap() < 1e-3
    energy_before = sim.stored_energy()
    moved = sim._recycle()
    assert 0 < moved <= n0
    assert sim.n == n0
    assert sim.recycle_count == moved
    speeds = torch.linalg.norm(sim.vel, dim=1)
    assert int((speeds == 0.0).sum()) >= moved
    assert float(sim.pos[:, 0].max()) > sim.x_stem()
    # a recycled floe never lands on top of another one
    assert sim.max_initial_overlap() < 1e-3
    removed = sim.w_removed
    # every recycled floe carries its kinetic energy out of the window
    per_floe = float(0.5 * sim.mass[first_column[0]]
                     * (torch.tensor([0.4, 0.4], dtype=torch.float64) ** 2).sum())
    assert removed == pytest.approx(moved * per_floe, rel=0.35)
    assert float(sim.pos[:, 0].max()) > sim.x_stem()


def test_pack_boundary_is_inactive_inside_the_pack_width():
    sim = _build_sim(STANDARD_CASE.with_(name="wall", concentration=0.5,
                                         route_ship_lengths=0.1))
    wc = wall_contacts(sim.pos, sim.vel, sim.radius, sim.pack_width,
                       sim.wall_stiffness, 0.2, sim.mass)
    assert wc["count"] == 0
    shifted = sim.pos.clone()
    shifted[:, 1] = 0.1
    wc2 = wall_contacts(shifted, sim.vel, sim.radius, sim.pack_width,
                        sim.wall_stiffness, 0.2, sim.mass)
    assert wc2["count"] > 0 and float(wc2["potential"]) > 0.0


def test_transit_is_deterministic_for_a_fixed_seed():
    spec = STANDARD_CASE.with_(name="det", concentration=0.5,
                               route_ship_lengths=0.15, steady_start_s=4.0)
    first = run_case(spec)
    second = run_case(spec)
    assert first["state_digest"] == second["state_digest"]


# -- references and criteria ------------------------------------------------
def test_lindqvist_anchor_is_the_sum_of_its_four_components():
    anchors = reference_anchors(STANDARD_CASE)
    lq = anchors["lindqvist"]
    total = (lq["crushing_N"] + lq["bending_N"] + lq["submersion_N"]
             + lq["added_drag_N"])
    assert lq["total_N"] == pytest.approx(total, rel=1e-12)
    assert anchors["lindqvist_total_kN"] > 500.0


def test_ledger_hash_is_stable():
    assert reference_anchors(STANDARD_CASE)["lindqvist"]["ledger_sha256"] \
        == reference_anchors(STANDARD_CASE)["lindqvist"]["ledger_sha256"]


def test_criteria_evaluation_is_a_pure_function():
    payload = {
        "mean_kN": 100.0, "ledger_residual_rel": 0.01,
        "stationarity_drift": 0.05, "route_length_m": 1050.0,
        "spec": STANDARD_CASE.as_dict(),
        "anchors": reference_anchors(STANDARD_CASE),
    }
    results = {"standard": payload,
               "no_ice": {"mean_kN": 0.0},
               "conc_0.50": {**payload, "mean_kN": 50.0,
                             "spec": {**payload["spec"], "concentration": 0.5}},
               "conc_0.90": {**payload, "mean_kN": 120.0,
                             "spec": {**payload["spec"], "concentration": 0.9}},
               "seed_11": {**payload, "mean_kN": 101.0,
                           "spec": {**payload["spec"], "seed": 11}},
               "seed_23": {**payload, "mean_kN": 99.0,
                           "spec": {**payload["spec"], "seed": 23}}}
    out = evaluate_criteria(results, test_count=20)
    assert out["criteria"]["E1"]["verdict"] in ("PASS", "FAIL")
    assert out["criteria"]["E2"]["verdict"] == "PASS"
    assert out["criteria"]["E9"]["verdict"] == "PASS"
    assert out["summary"]["n_criteria"] >= 7


def test_spec_refuses_impossible_inputs():
    with pytest.raises(ValueError):
        STANDARD_CASE.with_(thickness_m=-1.0)
    with pytest.raises(ValueError):
        STANDARD_CASE.with_(pack_width_m=20.0)