"""Ice-belt plate-panel FEM against classical plate benchmarks."""
from __future__ import annotations

import math

import pytest

from tensordem.myi_ridge import panel


E = 206.0e9
NU = 0.3
T = 0.030
A = 1.0
Q = 1.0e4
D = E * T ** 3 / (12.0 * (1.0 - NU ** 2))


def test_plate_rigidity_matches_closed_form():
    assert panel.plate_rigidity(E, T, NU) == pytest.approx(D, rel=1e-12)
    with pytest.raises(ValueError):
        panel.plate_rigidity(E, -T, NU)


def test_clamped_plate_centre_deflection_benchmark():
    result = panel.solve_panel(A, A, T, Q, elements_x=8, elements_y=8,
                               boundary="clamped")
    analytic = 0.00126 * Q * A ** 4 / D
    assert result.deflection_centre == pytest.approx(analytic, rel=0.015)


def test_clamped_plate_stress_benchmark():
    result = panel.solve_panel(A, A, T, Q, elements_x=12, elements_y=12,
                               boundary="clamped")
    analytic = 6.0 * 0.0513 * Q * A ** 2 / T ** 2
    assert result.stress_max == pytest.approx(analytic, rel=0.03)


def test_simply_supported_benchmarks():
    result = panel.solve_panel(A, A, T, Q, elements_x=10, elements_y=10,
                               boundary="simply_supported")
    assert result.deflection_centre == pytest.approx(0.00406 * Q * A ** 4 / D, rel=0.01)
    assert result.stress_max == pytest.approx(6.0 * 0.0479 * Q * A ** 2 / T ** 2, rel=0.02)


def test_mesh_convergence_of_the_clamped_panel():
    coarse = panel.solve_panel(A, A, T, Q, elements_x=6, elements_y=6, boundary="clamped")
    fine = panel.solve_panel(A, A, T, Q, elements_x=14, elements_y=14, boundary="clamped")
    analytic = 0.00126 * Q * A ** 4 / D
    assert coarse.deflection_centre == pytest.approx(analytic, rel=0.01)
    assert fine.deflection_centre == pytest.approx(analytic, rel=0.01)
    assert abs(fine.deflection_centre - coarse.deflection_centre) / analytic < 0.005


def test_thickness_scaling_is_inverse_square():
    thin = panel.solve_panel(A, A, 0.020, Q, elements_x=10, elements_y=10)
    thick = panel.solve_panel(A, A, 0.040, Q, elements_x=10, elements_y=10)
    assert thin.stress_max / thick.stress_max == pytest.approx(4.0, rel=1e-9)
    assert thin.deflection_max / thick.deflection_max == pytest.approx(8.0, rel=1e-9)


def test_patch_load_gives_smaller_deflection_than_full_load():
    full = panel.solve_panel(3.0, 3.0, T, 2.0e6, elements_x=10, elements_y=10)
    patch = panel.solve_panel(3.0, 3.0, T, 2.0e6, elements_x=10, elements_y=10,
                              patch=(1.0, 1.0, 2.0, 2.0))
    assert patch.deflection_max < full.deflection_max
    assert patch.load_total < full.load_total


def test_required_thickness_satisfies_the_allowable_stress():
    pressure = 1.5e6
    allowable = panel.STEEL_EH36["yield_strength"] / 1.15
    thickness = panel.required_thickness(2.4, 2.4, pressure, allowable_stress=allowable,
                                         reference_thickness=0.030, elements_x=10,
                                         elements_y=10)
    check = panel.check_ice_belt(pressure, span_x=2.4, span_y=2.4,
                                 thickness=thickness * 1.005,
                                 elements_x=10, elements_y=10)
    assert check["pass"] is True
    assert check["utilisation"] == pytest.approx(1.0, rel=0.02)


def test_ice_belt_check_pass_and_fail():
    small = panel.check_ice_belt(0.05e6, span_x=2.4, span_y=2.4, thickness=0.040,
                                 elements_x=10, elements_y=10)
    large = panel.check_ice_belt(2.0e6, span_x=2.4, span_y=2.4, thickness=0.030,
                                 elements_x=10, elements_y=10)
    assert small["pass"] is True
    assert large["pass"] is False
    assert large["thickness_required"] > 0.030


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        panel.solve_panel(1.0, 1.0, T, Q, boundary="free")
    with pytest.raises(ValueError):
        panel.solve_panel(-1.0, 1.0, T, Q)
    with pytest.raises(ValueError):
        panel.solve_panel(1.0, 1.0, T, Q, patch=(0.9, 0.9, 0.1, 0.1))
    with pytest.raises(ValueError):
        panel.required_thickness(1.0, 1.0, Q, allowable_stress=-1.0)
    with pytest.raises(ValueError):
        panel.check_ice_belt(Q, span_x=1.0, span_y=1.0, thickness=0.01, safety_factor=0.0)