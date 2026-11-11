"""Closed-form quadrature defect vs the engine ledger (benchmark pinning).

These tests bind three independent statements to the same numbers:

* the per-step defect ``legacy - exact`` equals the analytic
  ``sum(1/2 k n dz^2)`` (:mod:`tensordem.dem3d_energy_quadrature`) to better
  than 0.1 % at every timestep tested;
* the fitted convergence order of the defect is two (quadratic in ``dz``);
* the triaxial protocol surfaces the predicted/measured pair and only gates it
  when the platen dashpot is off (spring-only closed form).

Run: ``PYTHONPATH=src python -m pytest tests/test_dem3d_energy_quadrature_closure.py -q``
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D
from tensordem.dem3d_energy_quadrature import (
    platen_defect_components,
    quadrature_order,
    relative_error,
)
from tensordem.dem3d_triaxial import TriaxialConfig, run_triaxial

torch.set_num_threads(1)


def _sim(dt: float) -> EnergyAuditedIceDEM3D:
    cfg = DEM3DConfig(nx=4, ny=4, nz=3, radius=0.025, top_platen_enabled=True,
                      bottom_platen_enabled=True, top_platen_velocity=-2.0e-3)
    sim = EnergyAuditedIceDEM3D(cfg)
    sim.dt = dt
    return sim


def _measure(dt: float, steps: int = 120, warm: int = 40) -> tuple[float, float]:
    sim = _sim(dt)
    for _ in range(warm):
        sim.step()
    e0 = sim.cumulative_platen_work_exact_J
    l0 = sim.cumulative_platen_work_J
    p0 = sim.cumulative_platen_deficit_predicted_J
    for _ in range(steps):
        sim.step()
    measured = (sim.cumulative_platen_work_exact_J - e0) - (
        sim.cumulative_platen_work_J - l0)
    predicted = sim.cumulative_platen_deficit_predicted_J - p0
    assert math.isfinite(measured)
    return measured, predicted


@pytest.mark.parametrize("dt", [8.0e-5, 4.0e-5, 2.0e-5])
def test_measured_deficit_matches_closed_form(dt: float):
    """Engine measurement vs analytic 1/2 k n dz^2: agreement < 0.1 %."""
    measured, predicted = _measure(dt)
    assert predicted > 0.0
    assert relative_error(measured, predicted) < 1.0e-3


def test_deficit_convergence_order_is_two():
    """log-log slope of the defect over three dt levels is 2 (quadratic in dz)."""
    levels = [(dt, _measure(dt)[0]) for dt in (8.0e-5, 4.0e-5, 2.0e-5)]
    assert all(defect > 0.0 for _, defect in levels)
    assert quadrature_order(levels) == pytest.approx(2.0, abs=5.0e-3)


def test_quadrature_order_rejects_degenerate_input():
    with pytest.raises(ValueError):
        quadrature_order([(1.0e-4, 1.0e-9)])
    with pytest.raises(ValueError):
        quadrature_order([(1.0e-4, 1.0e-9), (1.0e-4, 2.0e-9)])


def test_lateral_defect_closed_form_hand_computed():
    """Two particles against the x_min plane: defect = 2 * 1/2 k dz^2 exactly."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.05, 0.05]])
    radius, k, dz = 0.01, 5000.0, -1.0e-5
    lat0 = torch.tensor([radius + 0.005, 1.0, radius + 0.005, 1.0])
    lat1 = lat0.clone()
    lat1[0] = lat0[0] + dz
    lat1[2] = lat0[2] + dz
    parts = platen_defect_components(positions, radius=radius, stiffness=k,
                                     lateral_pos0=lat0, lateral_pos1=lat1)
    # The module differences the float32 plane coordinates, so take the same
    # float32 difference here instead of the nominal float64 dz.
    dz_eff = float(lat1[0]) - float(lat0[0])
    expected = 2 * 0.5 * k * dz_eff * dz_eff
    assert parts["lateral"] == pytest.approx(expected, rel=1e-12)
    assert parts["top"] == 0.0 and parts["bottom"] == 0.0
    assert parts["total"] == pytest.approx(expected, rel=1e-12)


def test_contact_entering_the_step_is_counted():
    """A particle whose overlap becomes positive only after the advance counts."""
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    radius, k = 0.01, 5000.0
    top_z0 = radius + 1.0e-4      # no overlap before
    top_z1 = radius - 1.0e-4      # overlap after
    parts = platen_defect_components(positions, radius=radius, stiffness=k,
                                     top_z0=top_z0, top_z1=top_z1)
    dz = top_z1 - top_z0
    assert parts["top"] == pytest.approx(0.5 * k * dz * dz, rel=1e-12)


def test_accumulator_is_reset_by_legacy_checkpoint_load():
    """Loading a pre-quadrature checkpoint must not inherit a stale prediction."""
    sim = _sim(4.0e-5)
    for _ in range(10):
        sim.step()
    state = sim.state_dict()
    state["energy_audit"].pop("cumulative_platen_deficit_predicted_J", None)
    other = _sim(4.0e-5)
    other.load_state_dict(state)
    assert other.cumulative_platen_deficit_predicted_J == 0.0


def test_triaxial_ledger_reports_defect_pair_and_scope():
    """Damping-free triaxial run gates the defect; the default dashpot does not."""
    clean = run_triaxial(TriaxialConfig(loading_speed=0.002, confine_steps=20,
                                        max_axial_steps=60, peak_min_stress=1e9,
                                        platen_damping=0.0, sigma_x=0.5e6,
                                        sigma_y=0.5e6))["energy_ledger"]
    assert clean["legacy_deficit_applicable"] is True
    assert clean["pass_legacy_deficit"] is True
    assert clean["legacy_deficit_relative_error"] < 1.0e-3

    damped = run_triaxial(TriaxialConfig(loading_speed=0.002, confine_steps=20,
                                         max_axial_steps=60,
                                         peak_min_stress=1e9))["energy_ledger"]
    assert damped["legacy_deficit_applicable"] is False
    assert damped["pass_legacy_deficit"] is None