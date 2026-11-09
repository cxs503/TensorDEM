"""Contract tests for the DEM contact-force time-history export.

Covers: the three verification cases in ``physics_complete`` mode, the energy
budget self-check (positive and negative), the ``statistics_only`` mode, JSON
round trip, and the fail-closed behaviour on empty histories.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.fatigue_chain import cases
from tensordem.fatigue_chain.contract import (
    HULL,
    SCHEMA_ID,
    ContactForceHistory,
    ForceHistoryError,
    MissingForceHistory,
)


@pytest.fixture(scope="module")
def impact():
    return cases.single_ball_impact()


@pytest.fixture(scope="module")
def collision():
    return cases.two_ball_collision()


@pytest.fixture(scope="module")
def floe_field():
    # short run: the case is O(1e4) explicit steps, the contract checks do not
    # depend on the record length beyond the sample count per contact
    return cases.floe_field_bow(duration_s=0.35, dt_s=4.0e-5, n_floes=24)


def test_single_ball_impact_is_admissible(impact):
    report = impact.history.validate()
    assert report["admissible"], report["checks"]
    assert report["mode"] == "physics_complete"
    assert report["n_contacts"] == 1
    assert report["energy"]["max_rel_budget_error"] < 1e-3
    assert report["skipped_checks"] == []


def test_single_ball_impact_matches_independent_ode(impact):
    ode = cases.integrate_impact_ode(mass=0.25, velocity=0.5, stiffness=2.0e5,
                                     damping_ratio=0.05)
    peak_err = abs(impact.analytic["peak_force_N"] - ode["peak_force_N"]) / ode["peak_force_N"]
    dur_err = (abs(impact.analytic["contact_duration_s"] - ode["contact_duration_s"])
               / ode["contact_duration_s"])
    res_err = (abs(impact.analytic["restitution"] - ode["restitution"])
               / ode["restitution"])
    assert peak_err < 0.03, peak_err
    assert dur_err < 0.03, dur_err
    assert res_err < 0.03, res_err
    assert ode["restitution"] == pytest.approx(ode["analytic_restitution"], rel=0.03)


def test_two_ball_collision_conserves_momentum(collision):
    report = collision.history.validate()
    assert report["admissible"], report["checks"]
    assert collision.analytic["momentum_final_kg_m_s"] == pytest.approx(
        collision.analytic["momentum_initial_kg_m_s"], rel=1e-6)
    # relative velocity reverses and is damped by the measured restitution
    e_meas = (-collision.analytic["relative_velocity_final_m_s"]
              / collision.analytic["relative_velocity_initial_m_s"])
    assert e_meas == pytest.approx(collision.analytic["restitution"], rel=1e-3)


def test_floe_field_contract(floe_field):
    history = floe_field.history
    # the floe-field record interval is 40 us and the contacts last ~2 ms, so
    # the trapezoidal budget closes to O(dt) -- 10 % is the resolution-bound
    # tolerance (measured 6.9 % at 25 kHz, see the node report)
    report = history.validate(tol_energy=0.10)
    assert report["admissible"], [c for c in report["checks"] if not c["passed"]]
    assert history.n_contacts() > 5
    assert report["energy"]["max_rel_state_function_error"] < 1e-2
    assert report["energy"]["spring_monotone_in_delta"]
    t, f = history.to_uniform(1000.0)
    assert t.size > 10
    assert f.max() > 0.0
    stats = history.statistics(1000.0)
    assert stats["peak_N"] == pytest.approx(f.max(), rel=1e-6)


def test_contract_detects_corrupted_energy_column(impact):
    h = impact.history
    bad = ContactForceHistory(
        h.t, h.contact_id, h.floe_i, h.floe_j, h.F_n, h.F_t, h.patch_area,
        delta=h.delta, v_n=h.v_n, e_spring=h.e_spring, e_diss=h.e_diss * 2.0,
    )
    report = bad.validate()
    assert not report["admissible"]
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert "energy_budget_consistent" in failed


def test_contract_detects_non_monotone_time(impact):
    h = impact.history
    t = h.t.copy()
    t[5], t[6] = t[6], t[5]
    bad = ContactForceHistory(t, h.contact_id, h.floe_i, h.floe_j, h.F_n, h.F_t,
                              h.patch_area, delta=h.delta, v_n=h.v_n,
                              e_spring=h.e_spring, e_diss=h.e_diss)
    report = bad.validate()
    assert not report["admissible"]
    assert any(c["name"] == "t_strictly_increasing_per_contact" and not c["passed"]
               for c in report["checks"])


def test_contract_detects_negative_normal_force(impact):
    h = impact.history
    f_n = h.F_n.copy()
    f_n[3] = -1.0
    bad = ContactForceHistory(h.t, h.contact_id, h.floe_i, h.floe_j, f_n, h.F_t,
                              h.patch_area, delta=h.delta, v_n=h.v_n,
                              e_spring=h.e_spring, e_diss=h.e_diss)
    assert not bad.validate()["admissible"]


def test_statistics_only_mode_reports_skipped_checks(impact):
    h = impact.history
    stats_only = ContactForceHistory(
        h.t, h.contact_id, h.floe_i, h.floe_j, h.F_n, h.F_t, h.patch_area,
        meta={"note": "no energy group"},
    )
    report = stats_only.validate()
    assert report["mode"] == "statistics_only"
    assert report["admissible"]
    assert set(report["skipped_checks"]) == {
        "delta_non_negative", "energy_spring_state_function", "energy_budget_consistent"}


def test_json_round_trip_preserves_physics(impact, tmp_path):
    path = impact.history.save(tmp_path / "ft.json")
    back = ContactForceHistory.load(path)
    assert back.physics_complete
    assert back.meta["case"] == "single_ball_impact"
    for name in ("t", "F_n", "delta", "v_n", "e_spring", "e_diss"):
        assert np.allclose(getattr(back, name), getattr(impact.history, name), rtol=0, atol=0)
    assert back.validate()["admissible"]


def test_empty_history_is_fail_closed():
    empty = ContactForceHistory([], [], [], [], [], [], [])
    assert empty.n_samples == 0
    report = empty.validate()
    assert not report["admissible"]
    with pytest.raises(MissingForceHistory):
        empty.to_uniform(1000.0)


def test_schema_id_is_declared():
    assert SCHEMA_ID == "tensordem.contact-force-history/1"
    assert HULL == -1


def test_energy_group_must_be_complete(impact):
    h = impact.history
    with pytest.raises(ForceHistoryError):
        ContactForceHistory(h.t, h.contact_id, h.floe_i, h.floe_j, h.F_n, h.F_t,
                            h.patch_area, delta=h.delta)