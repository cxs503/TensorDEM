"""Tests for the double-body hydrodynamic interference layer."""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.formation_hydro import (FormationFlow, FormationFlowConfig, HullWaterline,
                                       froude_number, source_panel_velocity)


def _circle_flow(panels: int = 48, speed: float = 1.0) -> tuple[FormationFlow, dict]:
    hull = HullWaterline(length=2.0, beam=2.0, panels_per_side=panels)
    flow = FormationFlow([hull], FormationFlowConfig(speed=speed))
    return flow, flow.solve([(0.0, 0.0)])


def test_panel_solution_satisfies_the_no_penetration_condition():
    _, solution = _circle_flow()
    assert solution["normalized_neumann_residual"] < 1.0e-9


def test_circle_surface_speed_matches_exact_potential_flow():
    speed = 1.0
    _, solution = _circle_flow(panels=64, speed=speed)
    collocation = solution["collocation"]
    velocity = solution["onset"] + solution["induced"]
    theta = torch.atan2(collocation[:, 1], collocation[:, 0])
    exact = 2.0 * speed * torch.abs(torch.sin(theta))
    computed = torch.linalg.vector_norm(velocity, dim=1)
    mask = exact > 0.3  # exclude the stagnation region
    error = ((computed[mask] - exact[mask]).abs() / exact[mask])
    assert float(error.mean()) < 0.02
    assert float(error.max()) < 0.06


def test_far_field_is_the_exact_dipole_and_converges_with_panels():
    radius = 1.0
    errors = []
    for panels in (24, 96):
        _, solution = _circle_flow(panels=panels)
        point = torch.tensor([[0.0, 8.0]])
        velocity = source_panel_velocity(point, solution["starts"], solution["ends"],
                                        solution["strengths"])
        exact = radius ** 2 * 8.0 ** 2 / 8.0 ** 4  # U a^2 (y^2 - x^2)/r^4 at (0, d)
        errors.append(abs(float(velocity[0, 0]) - exact) / exact)
    assert errors[0] < 0.05
    assert errors[1] < errors[0]


def test_column_interference_decays_to_zero_with_spacing():
    hull = HullWaterline(length=100.0, beam=20.0, panels_per_side=24)
    flow = FormationFlow([hull, hull], FormationFlowConfig(speed=2.0, channel_half_width=None))
    near = flow.interference([(0.0, 0.0), (200.0, 0.0)])["coefficients"]
    mid = flow.interference([(0.0, 0.0), (500.0, 0.0)])["coefficients"]
    far = flow.interference([(0.0, 0.0), (5000.0, 0.0)])["coefficients"]
    assert abs(float(far[1, 0])) < 1.0e-3
    assert abs(float(mid[1, 0])) < abs(float(near[1, 0]))
    assert abs(float(far[1, 0])) < abs(float(mid[1, 0]))


def test_interference_is_symmetric_in_a_column_and_abeam_sign_is_positive():
    hull = HullWaterline(length=100.0, beam=20.0, panels_per_side=24)
    flow = FormationFlow([hull, hull], FormationFlowConfig(speed=2.0))
    column = flow.interference([(0.0, 0.0), (200.0, 0.0)])["coefficients"]
    assert float(column[0, 1]) == pytest.approx(float(column[1, 0]), rel=1e-6, abs=1e-9)
    abeam = flow.interference([(0.0, 0.0), (0.0, 40.0)])["coefficients"]
    assert float(abeam[1, 0]) > 0.0


def test_contact_limit_is_finite_and_bounded():
    hull = HullWaterline(length=100.0, beam=20.0, panels_per_side=24)
    flow = FormationFlow([hull, hull], FormationFlowConfig(speed=2.0))
    for offsets in ([(0.0, 0.0), (100.0 + 1e-3, 0.0)], [(0.0, 0.0), (0.0, 20.0 + 1e-3)],
                    [(0.0, 0.0), (60.0, 0.0)]):
        coefficients = flow.interference(offsets)["coefficients"]
        assert math.isfinite(float(coefficients[1, 0]))
        assert abs(float(coefficients[1, 0])) <= 1.0


def test_channel_walls_add_confinement_and_keep_the_solution_consistent():
    hull = HullWaterline(length=100.0, beam=20.0, panels_per_side=20)
    free = FormationFlow([hull], FormationFlowConfig(speed=2.0))
    confined = FormationFlow([hull], FormationFlowConfig(speed=2.0, channel_half_width=32.0))
    free_result = free.interference([(0.0, 0.0)])
    confined_result = confined.interference([(0.0, 0.0)])
    assert float(confined_result["confinement"][0]) > 0.05
    assert abs(float(free_result["confinement"][0])) < 1e-12
    assert confined_result["wall_panel_count"] > 0
    assert confined_result["neumann_residual"] < 1.0e-9


def test_three_ship_matrix_is_consistent_with_row_sums():
    hull = HullWaterline(length=100.0, beam=20.0, panels_per_side=18)
    flow = FormationFlow([hull, hull, hull], FormationFlowConfig(speed=2.0))
    offsets = [(0.0, 0.0), (200.0, 0.0), (400.0, 0.0)]
    result = flow.interference(offsets)
    matrix = result["coefficients"]
    assert matrix.shape == (3, 3)
    assert float(matrix[0, 0]) == 0.0 and float(matrix[2, 2]) == 0.0
    assert abs(float(matrix[0, 2])) < abs(float(matrix[0, 1]))


def test_interference_tables_cover_spacing_and_bearing():
    hull = HullWaterline(length=100.0, beam=20.0, panels_per_side=16)
    flow = FormationFlow([hull, hull], FormationFlowConfig(speed=2.0))
    rows = flow.interference_tables([2.0, 10.0], [0.0, 90.0])
    assert len(rows) == 4
    by_key = {(row["spacing_over_length"], row["bearing_deg"]): row for row in rows}
    assert by_key[(2.0, 0.0)]["dy_over_length"] == pytest.approx(0.0, abs=1e-9)
    assert by_key[(2.0, 90.0)]["dx_over_length"] == pytest.approx(0.0, abs=1e-9)
    assert abs(by_key[(10.0, 0.0)]["C_follower_from_leader"]) < abs(
        by_key[(2.0, 0.0)]["C_follower_from_leader"])


def test_configuration_and_geometry_validation():
    with pytest.raises(ValueError):
        FormationFlowConfig(speed=-1.0)
    with pytest.raises(ValueError):
        HullWaterline(length=0.0, beam=20.0)
    with pytest.raises(ValueError):
        HullWaterline(length=100.0, beam=20.0, panels_per_side=2)
    hull = HullWaterline(length=100.0, beam=20.0)
    flow = FormationFlow([hull], FormationFlowConfig(speed=2.0))
    with pytest.raises(ValueError):
        flow.interference([(0.0, 0.0), (10.0, 0.0)])


def test_froude_number_reference_values():
    assert froude_number(2.0, 100.0) == pytest.approx(2.0 / math.sqrt(980.665), rel=1e-9)
    with pytest.raises(ValueError):
        froude_number(-1.0, 100.0)