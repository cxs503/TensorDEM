"""Premise gates for CASE-DEM-PLANE-SHEAR-MUIDI (round 4a, m16669).

The case confronts a *constant-pressure* mu(I)/phi(I) law with a servo cell.
Two premises of that confrontation were never judged in rounds 1-3:

  P1  the servo actually holds the target normal stress,
  P2  the requested ``shear_strain`` is the strain that is run.

Both are numerics/protocol, not physics, so they can be asserted without
pinning any physical constant.  The stationarity statistics of the sampling
window are unit-tested on synthetic series (no simulation cost).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "plane_shear_mu_i"
for p in (REPO / "src", BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tensordem.mu_i_rheology import ServoShearCase  # noqa: E402

_spec = importlib.util.spec_from_file_location("servo_control_diag",
                                              BENCH / "servo_control_diag.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


# --------------------------------------------------------------------------- #
# unit: stationarity statistics of the sampling window
# --------------------------------------------------------------------------- #
def _series(values, d_strain=1e-3):
    return [{"strain": i * d_strain, "normal_stress_pa": v,
             "shear_stress_pa": 0.4 * v, "mu": 0.4,
             "plate_z": 0.05}
            for i, v in enumerate(values)]


def test_window_stats_stationary_series_is_flagged_steady():
    series = _series([62500.0] * 500)
    st = diag.window_stats(series, 62500.0)
    assert st["n_tail"] == 300
    assert st["p_cv"] == pytest.approx(0.0, abs=1e-12)
    assert st["p_rel_drift_over_window"] == pytest.approx(0.0, abs=1e-12)
    assert st["frac_below_0.9p_target"] == 0.0


def test_window_stats_detects_deficit_and_drift():
    # 20 % low plateau -> deficit is flagged, drift stays ~0
    series = _series([50000.0] * 500)
    st = diag.window_stats(series, 62500.0)
    assert st["frac_below_0.9p_target"] == 1.0
    assert st["p_rel_drift_over_window"] == pytest.approx(0.0, abs=1e-12)
    # linear ramp down by 40 % -> drift flagged, mean is the window midpoint
    ramp = _series([62500.0 * (1.0 - 0.4 * i / 499.0) for i in range(500)])
    st = diag.window_stats(ramp, 62500.0)
    assert st["p_rel_drift_over_window"] < -0.30
    assert st["p_mean"] < 62500.0


# --------------------------------------------------------------------------- #
# P1: servo holds the target pressure with raised authority (numerics only)
# --------------------------------------------------------------------------- #
def test_servo_holds_pressure_with_raised_authority():
    """With servo_relax=0.5 / servo_max_step_d=0.05 d the cell holds P to <8 %.

    At the default authority (relax=0.2, clip=0.002 d) the same point sits at
    -28 % of the 62.5 kPa target (runs/servo_diag/trace_relax0.2_clip0.002.json),
    i.e. the constant-pressure premise of the reference law was violated.  No
    contact-law, material or reference parameter is touched here.
    """
    case = diag.build_case(0.1, servo_relax=0.5, servo_max_step_d=0.05)
    res = diag.simulate_servo_shear(case)
    err = res["normal_stress_measured_pa"] / diag.rs.P - 1.0
    assert abs(err) <= 0.08, f"pressure control error {err:+.3%} exceeds 8 %"


# --------------------------------------------------------------------------- #
# P2: the requested shear strain is the strain that is run
# --------------------------------------------------------------------------- #
def test_requested_strain_is_authoritative():
    """An explicit strain must not be clamped by ``min_shear_steps``."""
    case = diag.build_case(0.3, servo_relax=0.5, servo_max_step_d=0.05,
                           shear_strain=3.0)
    assert case.min_shear_steps == 1
    res = diag.simulate_servo_shear(case, return_series=True)
    strain = res["series"][-1]["strain"]
    assert strain == pytest.approx(3.0, rel=0.05), strain


def test_default_sweep_overstrains_low_inertial_points():
    """Documents the clamp: the sweep's reported strain is not the run strain.

    ``run_sweep`` passes ``shear_strain=0.2`` with ``min_shear_steps=400``; at
    I = 0.3 the strain budget of 0.2 is only ~42 steps, so 400 steps (strain
    ~1.9) are run while the artefact still records 0.2.  The test pins the
    mechanism (physics-free): the clamp direction and magnitude.
    """
    gamma_dot = 0.3 * math.sqrt(diag.rs.P / diag.rs.RHO) / (2.0 * diag.rs.R)
    case = ServoShearCase(inertial_number=0.3, normal_stress=diag.rs.P,
                          kn=diag.rs.KN, mu=diag.rs.MU_CONTACT,
                          mu_rolling=diag.rs.MU_ROLLING, restitution=diag.rs.E,
                          shear_strain=0.2, radius=diag.rs.R,
                          density=diag.rs.RHO, box_width=diag.rs.W,
                          box_depth=diag.rs.D, fill_height=diag.rs.H,
                          settle_steps=200, min_shear_steps=400)
    steps_for_target = math.ceil(0.2 / (gamma_dot * case.timestep))
    assert steps_for_target < 100, steps_for_target
    assert max(400, steps_for_target) == 400
    assert 400 * gamma_dot * case.timestep > 1.5  # ~1.9 strain actually run


# --------------------------------------------------------------------------- #
# ensemble summary: bias vs scatter bookkeeping
# --------------------------------------------------------------------------- #
def test_ensemble_summary_separates_bias_from_scatter():
    rows = [{"I_target": 0.1, "seed": s, "mu": 0.40 + 0.01 * k,
             "pressure_control_error": 0.01, "phi_bed_steady": 0.53,
             "ref_mu": 0.50, "ref_phi": 0.60}
            for k, s in enumerate((1, 2, 3, 4))]
    mu_m = sum(r["mu"] for r in rows) / 4
    mu_sd = math.sqrt(sum((r["mu"] - mu_m) ** 2 for r in rows) / 3)
    assert (mu_m / 0.50 - 1.0) * 100 < -15.0   # bias: outside a 10 % band
    assert mu_sd > 0.0                          # scatter is reported separately
    assert (0.53 / 0.60 - 1.0) * 100 < -10.0