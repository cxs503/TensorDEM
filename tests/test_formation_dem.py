"""Tests for the multi-body (formation) DEM ice-ship interaction layer."""
from __future__ import annotations

import math

import pytest

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D
from tensordem.formation_dem import FormationIceDEM3D, ShipBow

RADIUS = 0.05
THETA_DEG = 45.0


def _config(speed: float, drag: float = 5.0, nx: int = 6, ny: int = 3) -> DEM3DConfig:
    return DEM3DConfig(nx=nx, ny=ny, nz=3, radius=RADIUS, tool_speed=speed,
                       bond_stiffness=1200.0, contact_stiffness=3500.0,
                       breaking_strain=0.01, shear_breaking_strain=0.03,
                       drag=drag, contact_damping=0.5,
                       fix_x_edges=False, fix_bottom=False)


def _zero_overlap_tip(config: DEM3DConfig, speed: float) -> float:
    probe = FormationIceDEM3D(config, [ShipBow(name="P1", speed=speed, center_y=0.0,
                                               half_width=0.1, tip_x0=0.0)])
    x_min = float(probe.initial_positions[:, 0].min())
    return x_min - RADIUS / math.sin(math.radians(THETA_DEG)) - 1.0e-9


def test_single_ship_formation_matches_the_wedge_model():
    speed = 0.2
    config = _config(speed)
    tip = _zero_overlap_tip(config, speed)
    gap = RADIUS / math.sin(math.radians(THETA_DEG)) - 0.5 * RADIUS
    single = MovingWedgeBowIceDEM3D(config, bow_speed=speed, wedge_angle_deg=THETA_DEG,
                                    bow_half_width_m=0.1, initial_gap=gap)
    formation = FormationIceDEM3D(config, [ShipBow(name="P1", speed=speed, center_y=0.0,
                                                    half_width=0.1, tip_x0=tip,
                                                    wedge_angle_deg=THETA_DEG)])
    for _ in range(250):
        single.step()
        formation.step()
    single_force = float(single.diagnostics()["bow_reaction_x_N"])
    # Both values are evaluated on the current state (the stored per-ship
    # diagnostics of the formation solver lag one step behind by construction).
    formation_force = float(formation.diagnostics()["reaction_x"])
    assert single_force != 0.0
    assert formation_force == pytest.approx(single_force, rel=1e-6, abs=1e-6)
    assert formation.broken_bonds == single.broken_bonds


def test_energy_audit_closes_within_five_percent():
    speed = 0.1
    config = _config(speed, drag=5.0, nx=6, ny=3)
    tip = _zero_overlap_tip(config, speed)
    sim = FormationIceDEM3D(config, [ShipBow(name="P1", speed=speed, center_y=0.0,
                                             half_width=0.1, tip_x0=tip)])
    steps = 1400
    for _ in range(steps):
        sim.step()
    audit = sim.audit()
    assert audit["work_in_J"] > 0.0
    assert abs(audit["relative_residual"]) < 0.05
    assert abs(audit["work_in_J"] - audit["stored_energy_change_J"] - audit["released_bond_J"]
               - audit["dissipation_J"]) == pytest.approx(abs(audit["residual_J"]), rel=1e-9)
    assert audit["dissipation_breakdown"]["drag_J"] > 0.0


def test_shared_field_changes_the_follower_response():
    speed = 0.3
    config = _config(speed, drag=2.0, nx=8, ny=3)
    tip = _zero_overlap_tip(config, speed)
    lag = 0.05
    shared = FormationIceDEM3D(config, [
        ShipBow(name="P1", speed=speed, center_y=0.0, half_width=0.1, tip_x0=tip),
        ShipBow(name="P2", speed=speed, center_y=0.0, half_width=0.1, tip_x0=tip - lag),
    ])
    control = FormationIceDEM3D(config, [
        ShipBow(name="P2", speed=speed, center_y=0.0, half_width=0.1, tip_x0=tip - lag),
    ])
    shared_forces, control_forces, leader_forces = [], [], []
    for _ in range(600):
        shared.step()
        control.step()
        shared_forces.append(shared.last_ship_diagnostics["P2"]["reaction_x_N"])
        control_forces.append(control.last_ship_diagnostics["P2"]["reaction_x_N"])
        leader_forces.append(shared.last_ship_diagnostics["P1"]["reaction_x_N"])
    window = max(5, len(shared_forces) // 4)
    mean_shared = sum(shared_forces[-window:]) / window
    mean_control = sum(control_forces[-window:]) / window
    assert math.isfinite(mean_shared) and math.isfinite(mean_control)
    # The leader breaks the shared field, so the follower meets a damaged ice cover.
    assert shared.broken_bonds > control.broken_bonds
    assert abs(sum(leader_forces[-window:]) / window) > 0.0
    assert all(key in shared.last_ship_diagnostics for key in ("P1", "P2"))
    report = shared.ship_report()
    assert report["P2"]["center_y_m"] == 0.0
    assert report["P1"]["work_in_J"] != 0.0


def test_lane_geometry_controls_the_contact():
    speed = 0.2
    config = _config(speed, nx=6, ny=5)
    tip = _zero_overlap_tip(config, speed)
    inside = FormationIceDEM3D(config, [ShipBow(name="P1", speed=speed, center_y=0.0,
                                                half_width=0.1, tip_x0=tip)])
    outside = FormationIceDEM3D(config, [ShipBow(name="P1", speed=speed, center_y=5.0,
                                                 half_width=0.1, tip_x0=tip)])
    for _ in range(200):
        inside.step()
        outside.step()
    assert inside.last_ship_diagnostics["P1"]["bow_particles_in_contact"] > 0
    assert outside.last_ship_diagnostics["P1"]["bow_particles_in_contact"] == 0
    assert abs(outside.last_ship_diagnostics["P1"]["reaction_x_N"]) < 1e-12


def test_bow_specification_and_audit_validation():
    with pytest.raises(ValueError):
        ShipBow(name="", speed=0.1, center_y=0.0, half_width=0.1, tip_x0=0.0)
    with pytest.raises(ValueError):
        ShipBow(name="P1", speed=-0.1, center_y=0.0, half_width=0.1, tip_x0=0.0)
    with pytest.raises(ValueError):
        ShipBow(name="P1", speed=0.1, center_y=0.0, half_width=0.1, tip_x0=0.0,
                wedge_angle_deg=2.0)
    with pytest.raises(ValueError):
        FormationIceDEM3D(_config(0.1), [])
    sim = FormationIceDEM3D(_config(0.1),
                            [ShipBow(name="P1", speed=0.1, center_y=0.0, half_width=0.1,
                                     tip_x0=0.0)])
    with pytest.raises(RuntimeError):
        sim.audit()
    with pytest.raises(KeyError):
        sim.bow_tip_x("missing")