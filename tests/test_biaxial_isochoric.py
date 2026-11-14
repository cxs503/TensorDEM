"""Invariants of the constant-volume (isochoric) biaxial shear mode.

Round 2 of CASE-DEM-BIAXIAL-FRICTION showed that the lateral *servo* cannot be
trusted in the large-strain regime (velocity-limit saturation on both sides), so
the critical-state test is run at constant volume with kinematic
volume-preserving walls instead.  These tests pin the invariants that make that
statement meaningful:

* ``V = W * H`` is preserved to round-off and the walls open as the platen
  descends (opposite sign, ``|dW/W| = |dH/H|``);
* the finite-strain volumetric measure is the one that vanishes, while the
  small-strain sum drifts (this is why the critical-state gate must use the log
  measure);
* the lateral stress is an output and stays in compression;
* the critical-state detector switches its boundary-closure gate accordingly,
  and refuses a log-measure path that does not carry the log keys;
* the default mode is still the original servo boundary condition.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_critical_state import (
    critical_state_from_stress_path,
    judge_against_reference,
)
from tensordem.biaxial_shear_band import run_biaxial_shear_band

SMALL = dict(confining_stress=100.0e3, columns=6, rows=10, radius=0.75,
             strain_rate=0.1, strain_target=0.05, consolidation_steps=300,
             snapshot_strain=1.0e9, wall_friction=0.0,
             material_overrides={"freeze_bond_fraction": 0.0,
                                 "friction_coefficient": 0.55})


@pytest.fixture(scope="module")
def isochoric_run() -> dict:
    return run_biaxial_shear_band(lateral_control="isochoric", **SMALL)


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def test_lateral_control_is_validated_before_any_work() -> None:
    with pytest.raises(ValueError, match="lateral_control"):
        run_biaxial_shear_band(lateral_control="pressure", **SMALL)


def test_volume_is_preserved_and_walls_open(isochoric_run: dict) -> None:
    h = isochoric_run["stress_path"]
    eps_v_log = np.asarray(h["volumetric_strain_log"], dtype=float)
    eps_a_log = np.asarray(h["axial_strain_log"], dtype=float)
    # volume preserving to the order of the explicit integrator's truncation
    # error (the residual grows linearly with the step count: ~2.5e-6 over 1e3
    # steps, i.e. 3e-9 per step, and does not drift with strain)
    assert eps_a_log[-1] > 0.03
    assert np.max(np.abs(eps_v_log)) < 1e-4 * eps_a_log[-1]
    assert abs(eps_v_log[-1] / eps_a_log[-1]) < 0.01

    specimen = isochoric_run["specimen"]
    # the two measures agree only to first order: the small-strain sum drifts
    # negative once eps_a is O(0.1), which is exactly why the gate uses log
    # for eps_a -> 0.05 the spurious small-strain volume change is
    # 1 - e^{-eps} + (1 - e^{+eps}) = -0.0025, i.e. two orders of magnitude above
    # the log residual: the gate would reject a perfectly isochoric path
    small = np.asarray(h["volumetric_strain"], dtype=float)
    assert small[-1] < 0.0
    assert abs(small[-1]) > 50.0 * abs(eps_v_log[-1])

    # kinematic check on the walls themselves: W/W0 = H0/H
    width0 = specimen["width0"]
    height0 = specimen["height0"]
    h_max = height0 * math.exp(-eps_a_log[-1])
    w_expected = width0 * height0 / h_max
    assert w_expected > width0            # walls opened, not closed
    assert abs(math.log(w_expected / width0) + math.log(h_max / height0)) < 1e-9


def test_lateral_stress_is_an_output_and_in_compression(isochoric_run: dict) -> None:
    h = isochoric_run["stress_path"]
    sh = np.asarray(h["sigma_h"], dtype=float)
    q = np.asarray(h["deviator"], dtype=float)
    assert np.all(np.isfinite(sh))
    assert np.all(sh > 0.0)               # walls never detach
    # after the servo release the platen load still has to build up, so the very
    # first steps can sit at q < 0; the sheared record is q > 0 throughout
    assert np.all(q[len(q) // 4:] > 0.0)
    assert isochoric_run["lateral_control"] == "isochoric"
    assert "constant volume" in isochoric_run["note"]


def test_default_mode_is_still_the_servo_cell() -> None:
    # only the validation path is exercised here (no run): the default must remain
    # the historically validated constant-confining-pressure cell
    with pytest.raises(ValueError):
        run_biaxial_shear_band(lateral_control="servo2", **SMALL)
    assert "lateral_control" in run_biaxial_shear_band.__code__.co_varnames


# ---------------------------------------------------------------------------
# critical-state detector
# ---------------------------------------------------------------------------
def _synthetic_path(*, log_volume_exact: bool) -> dict:
    """Idealised dense-sand path: peak, softening, then a constant-ratio plateau."""
    eps = np.linspace(0.0, 0.60, 1200)
    p0 = 100.0e3
    phi = 20.0 + 22.0 * np.exp(-((eps - 0.10) / 0.06) ** 2)   # peak ~42 deg at 0.10
    phi = np.minimum(phi, 34.0)                              # plateau at 34 deg
    # constant-volume hardening: p rises while the specimen dilates, then saturates
    p = p0 * (1.0 + 1.4 * (1.0 - np.exp(-eps / 0.15)))
    sin_phi = np.sin(np.radians(phi))
    sv = p * (1.0 + sin_phi)
    sh = p * (1.0 - sin_phi)
    small = -0.45 * np.clip(eps - 0.30, 0.0, None)            # spurious small-strain drift
    log_ev = np.zeros_like(eps) if log_volume_exact else small
    return {"axial_strain": eps, "axial_strain_log": eps,
            "sigma_v": sv, "sigma_h": sh, "deviator": sv - sh,
            "volumetric_strain": small, "volumetric_strain_log": log_ev}


def test_log_measure_is_required_for_the_isochoric_gate() -> None:
    path = _synthetic_path(log_volume_exact=True)
    cs_log = critical_state_from_stress_path(
        path, boundary_control="isochoric", strain_measure="log")
    assert cs_log.flags["isochoric"] is True
    # the same stresses judged on the small-strain sum are (wrongly) flagged
    cs_small = critical_state_from_stress_path(
        path, boundary_control="isochoric", strain_measure="small")
    assert cs_small.flags["isochoric"] is False
    assert cs_log.strain_measure == "log"
    assert cs_small.strain_measure == "small"
    # the angle itself must not depend on the strain measure (same window rule)
    assert cs_log.phi_cv_deg == pytest.approx(cs_small.phi_cv_deg, rel=1e-9)


def test_isochoric_control_swaps_the_closure_gate() -> None:
    path = _synthetic_path(log_volume_exact=True)
    cs = critical_state_from_stress_path(
        path, sigma3_target=100.0e3, boundary_control="isochoric",
        strain_measure="log")
    assert "lateral_walls_loaded" in cs.flags
    assert "servo_on_target" not in cs.flags
    assert cs.flags["lateral_walls_loaded"] is True      # sigma_h > 0
    assert cs.boundary_control == "isochoric"

    cs_servo = critical_state_from_stress_path(path, sigma3_target=100.0e3)
    assert "servo_on_target" in cs_servo.flags
    assert "lateral_walls_loaded" not in cs_servo.flags
    # the synthetic path holds sigma_h ~ 100 kPa early and ~150 kPa late, so the
    # servo closure check must fail there -- the gate is doing real work
    assert cs_servo.flags["servo_on_target"] is False


def test_log_measure_requires_log_keys() -> None:
    path = _synthetic_path(log_volume_exact=True)
    del path["axial_strain_log"]
    with pytest.raises(ValueError, match="axial_strain_log"):
        critical_state_from_stress_path(path, strain_measure="log")
    with pytest.raises(ValueError, match="strain_measure"):
        critical_state_from_stress_path(path, strain_measure="engineering")
    with pytest.raises(ValueError, match="boundary_control"):
        critical_state_from_stress_path(path, boundary_control="drained")


def test_plateau_is_detected_and_judged() -> None:
    path = _synthetic_path(log_volume_exact=True)
    cs = critical_state_from_stress_path(
        path, boundary_control="isochoric", strain_measure="log")
    # declared detector defaults: last 20 % of the path above eps_a = 0.20
    lo, hi = cs.window
    assert lo == pytest.approx(0.20 + 0.80 * 0.40, abs=1e-6)
    assert hi == pytest.approx(0.60, abs=1e-6)
    assert cs.flags["deviator_stationary"] is True
    assert cs.converged is True
    verdict = judge_against_reference(cs.phi_cv_deg, 28.81, 0.08)
    assert set(verdict) >= {"passed", "relative_error"}
    # the plateau in this synthetic path sits at 34 deg -> fails the 8 % band
    assert verdict["passed"] is False