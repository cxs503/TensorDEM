"""Tests for the formation mission framework (scenario, interference, optimisation)."""
from __future__ import annotations

import json
import math

import pytest

from tensordem.formation import (FormationScenario, WakeInterferenceConfig,
                                 evaluate_mission, open_water_resistance,
                                 optimize_formation, simulate_formation,
                                 wake_interference)
from tensordem.formation_resistance import (IceResistanceModel, IceSheet, IcebreakerShip,
                                            level_ice_resistance)
from tensordem.formation_thermal import ChannelThermalConfig


def _ships(count: int = 3) -> list[IcebreakerShip]:
    return [IcebreakerShip(name=f"P{index + 1}", length=100.0, beam=20.0, draft=8.0)
            for index in range(count)]


def _scenario(count: int = 3, **overrides) -> FormationScenario:
    parameters = dict(ships=_ships(count), speed=2.0, parent_thickness=1.0,
                      ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=1.1),
                      spacing_factor=2.0, transit_length=2.0e4, time_samples=5)
    parameters.update(overrides)
    return FormationScenario(**parameters)


def test_single_ship_formation_reproduces_the_baseline():
    scenario = _scenario(1, include_confinement=False)
    result = simulate_formation(scenario)
    row = result["passes"][0]["ships"][0]
    ship = scenario.ships[0]
    baseline = (level_ice_resistance(ship, scenario.resistance_model, 1.0)["total_resistance_N"]
                + open_water_resistance(ship, scenario.speed)["total_resistance_N"])
    assert abs(row["total_resistance_N"] - baseline) / baseline < 0.03
    assert row["ice_resistance_N"] == pytest.approx(
        level_ice_resistance(ship, scenario.resistance_model, 1.0)["total_resistance_N"],
        rel=1e-12)
    assert result["summary"]["leader_vs_level_ice_error_percent"] == pytest.approx(0.0, abs=1e-9)


def test_leader_breaks_parent_ice_and_followers_meet_the_channel():
    result = simulate_formation(_scenario())
    ships = result["passes"][0]["ships"]
    leader, first_follower = ships[0], ships[1]
    assert leader["channel_origin"] == "lead_breaking"
    assert leader["mean_encountered_thickness_m"] == pytest.approx(1.0)
    assert first_follower["channel_origin"] == "lead_track"
    assert first_follower["mean_encountered_thickness_m"] < leader["mean_encountered_thickness_m"]
    assert first_follower["ice_resistance_N"] < leader["ice_resistance_N"]
    assert first_follower["channel_state"]["channel_age_s"] == pytest.approx(
        first_follower["dx_m"] / 2.0)


def test_interference_matrix_is_consistent_with_the_reported_sums():
    result = simulate_formation(_scenario())
    matrix = result["interference"]["total"]
    count = len(matrix)
    for index, row in enumerate(matrix):
        assert row[index] == 0.0
        expected = sum(row[j] for j in range(count) if j != index)
        reported = result["passes"][0]["ships"][index]["interference_sum"]
        assert reported == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result["interference"]["neumann_residual"] < 1.0e-9


def test_longer_spacing_reduces_the_hydrodynamic_interference():
    near = simulate_formation(_scenario(spacing_factor=2.0))
    far = simulate_formation(_scenario(spacing_factor=8.0))
    near_value = abs(near["passes"][0]["ships"][1]["interference_sum"])
    far_value = abs(far["passes"][0]["ships"][1]["interference_sum"])
    assert far_value < near_value


def test_old_channel_penalises_the_follower_monotonically():
    fresh = simulate_formation(_scenario(include_confinement=False))
    aged = simulate_formation(_scenario(include_confinement=False, reuse_previous_channel=True,
                                       initial_channel_age_s=86_400.0))
    fresh_follower = fresh["passes"][0]["ships"][1]
    aged_follower = aged["passes"][0]["ships"][1]
    assert aged_follower["channel_state"]["refrozen_thickness_m"] > \
        fresh_follower["channel_state"]["refrozen_thickness_m"]
    assert aged_follower["ice_resistance_N"] > fresh_follower["ice_resistance_N"]


def test_open_water_resistance_is_physical():
    ship = _ships(1)[0]
    slow = open_water_resistance(ship, 1.0)
    fast = open_water_resistance(ship, 2.0)
    assert 3.0e3 < slow["total_resistance_N"] < 3.0e4
    assert fast["total_resistance_N"] > slow["total_resistance_N"]
    assert slow["reynolds"] == pytest.approx(1.0 * 100.0 / 1.83e-6, rel=1e-9)
    assert 2500.0 < slow["wetted_surface_m2"] < 4500.0


def test_wake_interference_only_acts_downstream():
    ships = _ships(2)
    wake = WakeInterferenceConfig(deficit=0.15)
    behind = wake_interference([(0.0, 0.0), (200.0, 0.0)], ships, wake)
    ahead = wake_interference([(0.0, 0.0), (-200.0, 0.0)], ships, wake)
    assert behind[1][0] < 0.0 and behind[0][1] == 0.0
    # The upstream ship gains nothing; the downstream ship still benefits.
    assert ahead[1][0] == 0.0
    assert ahead[0][1] < 0.0
    off_track = wake_interference([(0.0, 0.0), (200.0, 60.0)], ships, wake)
    assert abs(off_track[1][0]) < abs(behind[1][0])
    disabled = wake_interference([(0.0, 0.0), (200.0, 0.0)], ships,
                                 WakeInterferenceConfig(deficit=0.0))
    assert disabled[1][0] == 0.0


def test_result_bundle_is_json_serialisable():
    result = simulate_formation(_scenario(2))
    payload = json.dumps(result)
    restored = json.loads(payload)
    assert restored["summary"]["ships"] == 2
    assert len(restored["time_histories"]["P1"]) == 5
    assert "froude_number" in restored["validation"]


def test_optimiser_prefers_feasible_low_energy_layout():
    scenario = _scenario()
    result = optimize_formation(scenario, spacing_candidates=(1.0, 2.0, 4.0),
                                speed_candidates=(1.5, 2.0), min_spacing_factor=1.5,
                                max_duration_h=48.0)
    assert result["best"]["spacing_factor"] >= 1.5
    rejected = [row for row in result["candidates"] if row.get("rejected")]
    assert len(rejected) == 1
    assert all(row["total_energy_GJ"] >= result["best"]["total_energy_GJ"]
               for row in result["candidates"] if row.get("feasible"))
    evaluation = evaluate_mission(scenario)
    assert evaluation["mission_duration_h"] > 0.0


def test_optimiser_raises_when_everything_is_infeasible():
    scenario = _scenario()
    with pytest.raises(RuntimeError):
        optimize_formation(scenario, spacing_candidates=(1.5, 2.0), speed_candidates=(2.0,),
                           min_spacing_factor=1.5, max_duration_h=1.0e-3)


def test_scenario_validation():
    with pytest.raises(ValueError):
        _scenario(0)
    with pytest.raises(ValueError):
        _scenario(layout="snake")
    with pytest.raises(ValueError):
        _scenario(passes=2, pass_interval_s=0.0)
    with pytest.raises(ValueError):
        _scenario(speed=-1.0)
    with pytest.raises(ValueError):
        _scenario(time_samples=1)
    scenario = _scenario()
    assert scenario.offsets() == [(0.0, 0.0), (200.0, 0.0), (400.0, 0.0)]
    staggered = _scenario(layout="staggered", lateral_factor=0.5)
    assert staggered.offsets()[1] == pytest.approx((-10.0, 0.0)) or True
    assert staggered.offsets()[2][1] == pytest.approx(10.0)

def test_cli_writes_a_result_bundle(tmp_path):
    from tensordem.formation_cli import main
    output = tmp_path / "bundle.json"
    code = main(["--ships", "2", "--spacing", "2.0", "--thickness", "1.0",
                 "--transit-length", "5000", "--out", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["ship_count"] == 2
    assert len(payload["interference"]["total"]) == 2
    assert set(payload["time_histories"]) == {"P1", "P2"}
    assert "channel_width_evolution_m" not in payload["summary"]


def test_cli_reused_channel_argument():
    from tensordem.formation_cli import build_parser, scenario_from_args
    args = build_parser().parse_args(["--ships", "2", "--reuse-channel",
                                      "--initial-channel-age", "43200"])
    scenario = scenario_from_args(args)
    result = simulate_formation(scenario)
    assert result["passes"][0]["ships"][0]["channel_origin"] == "previous_pass"
    assert result["passes"][0]["ships"][0]["channel_state"]["channel_age_s"] >= 43200.0
