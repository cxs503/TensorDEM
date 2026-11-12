"""Step budget and arm wiring of the plane-shear sweep (m16497, round 3).

Round 2 of ``CASE-DEM-PLANE-SHEAR-MUIDI`` left two experiment controls owed
(doc note D3, rounds-planner clue 1/3): a 10x stiffer contact arm (``P/kn=0.1``)
and a thicker bed arm.  The stiff arm aborted with

    MuIRheologyError: servo did not reach the target normal stress

because ``ServoShearCase.timestep`` is ``2*pi*sqrt(0.5*m/kn)/20`` (it shrinks as
``1/sqrt(kn)``) while the settle budget was a *fixed step count*: at kn = 10x
the crate relaxes over 3.16x shorter physical time and the servo never reaches
its 8 % pressure window before ``max_settle``.

These tests pin the fix and the arm wiring:

* ``stiffness_step_scale`` == ``sqrt(kn/kn_ref)`` and exactly 1 for the
  baseline stiffness (the round-1/2 configuration stays bit-identical),
* the stiff arm really requests ``sqrt(10)`` times the step budget while the
  thick arm keeps the baseline stiffness *and* changes geometry only,
* ``run_point`` forwards the arm geometry/stiffness into the device case,
* (opt-in, ``MU_I_STIFF_DEFECT=1``) the unscaled budget really does abort on the
  production geometry -- the evidence for the defect the fix removes.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "benchmarks" / "plane_shear_mu_i"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_sweep as rs  # noqa: E402
from tensordem.mu_i_rheology import MuIRheologyError, simulate_servo_shear  # noqa: E402


# --------------------------------------------------------------------------
# 1. the step-budget law
# --------------------------------------------------------------------------
def test_step_scale_is_sqrt_of_stiffness_ratio():
    assert rs.stiffness_step_scale(rs.KN) == pytest.approx(1.0, rel=1e-12)
    assert rs.stiffness_step_scale(10.0 * rs.KN) == pytest.approx(math.sqrt(10.0),
                                                                 rel=1e-12)
    # inverse-square consistency with the device time step
    from tensordem.mu_i_rheology import ServoShearCase
    soft = ServoShearCase(kn=rs.KN)
    stiff = ServoShearCase(kn=10.0 * rs.KN)
    ratio = soft.timestep / stiff.timestep
    assert ratio == pytest.approx(rs.stiffness_step_scale(10.0 * rs.KN), rel=1e-12)


def test_step_scale_rejects_bad_input():
    with pytest.raises(ValueError):
        rs.stiffness_step_scale(0.0)
    with pytest.raises(ValueError):
        rs.stiffness_step_scale(1.0, kn_ref=-1.0)


# --------------------------------------------------------------------------
# 2. arm wiring: baseline unchanged, stiff scaled, thick geometry-only
# --------------------------------------------------------------------------
def _config(arm: str) -> dict:
    return rs.payload_for(arm, [], rs.I_TARGETS)["config"]


def test_baseline_arm_is_the_round1_configuration():
    cfg = _config("baseline")
    assert cfg["kn"] == rs.KN
    assert cfg["box"] == [rs.W, rs.D, rs.H]
    assert cfg["step_scale"] == pytest.approx(1.0, rel=1e-12)
    assert cfg["settle_steps_requested"] == rs.SETTLE_STEPS_BASE == 2500
    assert cfg["min_shear_steps_requested"] == rs.MIN_SHEAR_STEPS_BASE == 400


def test_stiff_arm_scales_budget_not_geometry():
    cfg = _config("stiff")
    assert cfg["kn"] == pytest.approx(10.0 * rs.KN)
    assert cfg["box"] == [rs.W, rs.D, rs.H]
    assert cfg["step_scale"] == pytest.approx(math.sqrt(10.0), rel=1e-12)
    assert cfg["settle_steps_requested"] == round(rs.SETTLE_STEPS_BASE * math.sqrt(10.0))
    assert cfg["min_shear_steps_requested"] == round(rs.MIN_SHEAR_STEPS_BASE
                                                     * math.sqrt(10.0))


def test_thick_arm_changes_geometry_only():
    cfg = _config("thick")
    assert cfg["kn"] == rs.KN                      # stiffness arm not engaged
    assert cfg["step_scale"] == pytest.approx(1.0, rel=1e-12)
    assert cfg["box"][1] > rs.D and cfg["box"][2] > rs.H
    assert cfg["box_depth_d"] == pytest.approx(8.0, rel=1e-12)
    assert cfg["layers"] == pytest.approx(9.0, rel=1e-12)


def test_stiff_thick_arm_combines_both_controls():
    cfg = _config("stiff_thick")
    assert cfg["kn"] == pytest.approx(10.0 * rs.KN)
    assert cfg["box_depth_d"] == pytest.approx(8.0, rel=1e-12)
    assert cfg["step_scale"] == pytest.approx(math.sqrt(10.0), rel=1e-12)


# --------------------------------------------------------------------------
# 3. run_point forwards arm parameters into the device case (no dynamics)
# --------------------------------------------------------------------------
def test_run_point_forwards_arm_kn_and_geometry(monkeypatch):
    seen = {}

    def fake_simulate(case, **kwargs):
        seen["case"] = case
        return {
            "inertial_number": 1e-2, "mu_effective": 0.4, "solid_fraction": 0.5,
            "phi_bed_settled": 0.5, "phi_bed_steady": 0.5,
            "plate_height_settled": 0.05, "plate_height_steady": 0.05,
            "dilatancy": 0.0, "coordination_number": 4.0,
            "normal_stress_measured_pa": rs.P, "normal_stress_applied_pa": rs.P,
            "settle_pressure_pa": rs.P, "shear_stress_pa": 0.4 * rs.P,
            "n_particles": 10, "plate_height": 0.05, "shear_steps": 500,
            "settle_steps": 7000, "dt": 1e-5, "max_velocity": 1e-3,
        }

    monkeypatch.setattr(rs, "simulate_servo_shear", fake_simulate)
    row = rs.run_point(1e-2, kn=10.0 * rs.KN, box_depth=0.08, fill_height=0.09)
    case = seen["case"]
    assert case.kn == pytest.approx(10.0 * rs.KN)
    assert case.box_depth == pytest.approx(0.08)
    assert case.fill_height == pytest.approx(0.09)
    assert case.settle_steps == round(rs.SETTLE_STEPS_BASE * math.sqrt(10.0))
    assert case.min_shear_steps == round(rs.MIN_SHEAR_STEPS_BASE * math.sqrt(10.0))
    # bookkeeping: compression is reported against the *fill* height of the arm
    assert row["settle_compression"] == pytest.approx(1.0 - 0.05 / 0.09, rel=1e-12)


# --------------------------------------------------------------------------
# 4. opt-in reproduction of the defect the scaled budget removes
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("MU_I_STIFF_DEFECT") != "1",
                    reason="opt-in: ~1 min, reproduces the stiff-arm servo abort")
def test_unscaled_budget_aborts_on_production_geometry():
    from tensordem.mu_i_rheology import ServoShearCase as Case

    case = Case(inertial_number=1e-2, mu=rs.MU_CONTACT, mu_rolling=rs.MU_ROLLING,
                restitution=rs.E, kn=10.0 * rs.KN, normal_stress=rs.P,
                shear_strain=rs.SHEAR_STRAIN,
                settle_steps=rs.SETTLE_STEPS_BASE,          # round-2 budget, unscaled
                min_shear_steps=rs.MIN_SHEAR_STEPS_BASE,
                packing_gap=0.02, jitter=0.01, seed=rs.SEED, radius=rs.R,
                density=rs.RHO, box_width=rs.W, box_depth=rs.D, fill_height=rs.H)
    with pytest.raises(MuIRheologyError, match="servo did not reach"):
        simulate_servo_shear(case)