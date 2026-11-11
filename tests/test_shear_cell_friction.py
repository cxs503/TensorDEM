"""Verification surface for the direct-shear cell and its Mohr-Coulomb fit.

Mission m13798 (2026-11-11).  The module `tensordem.shear_cell` shipped as dead
code: the first integration line indexed the scalar `mass` (`mass[:, None]`) and
raised ``TypeError`` before a single step ran, and it fitted a Mohr-Coulomb
*slope* inside one run where ``sigma_n`` is constant.  These tests pin the two
things that can be asserted without a working stress boundary:

* the envelope fitter is algebraically exact, and it *refuses* the ill-posed
  single-level regression (the original defect);
* the device runs end-to-end and its validity gate reports a detached platen
  instead of returning a made-up friction angle.

The physical steady state (``sigma_n`` held on target) is NOT reproduced yet --
see docs/GAP_DEM_FRICTION_SHEAR_CELL.md, honest FAIL carried forward.
"""
import math

import pytest

from tensordem.shear_cell import (
    ShearCellCase,
    _fit_mohr_coulomb,
    measure_macro_friction,
    simulate_shear_cell,
)


# --------------------------------------------------------------------- fitter
def test_mohr_coulomb_fit_is_exact_on_a_synthetic_line():
    sig = [400.0, 800.0, 1600.0]
    tau = [0.5 * s for s in sig]
    fit = _fit_mohr_coulomb(sig, tau)
    assert abs(fit["tan_phi"] - 0.5) < 1e-12
    assert abs(fit["cohesion_pa"]) < 1e-9
    assert fit["r_squared"] > 1.0 - 1e-12
    assert abs(fit["friction_angle_deg"] - math.degrees(math.atan(0.5))) < 1e-9
    assert fit["mu_level_spread_rel"] < 1e-12


def test_mohr_coulomb_fit_recovers_cohesion():
    sig = [400.0, 800.0, 1600.0]
    tau = [100.0 + 0.3 * s for s in sig]
    fit = _fit_mohr_coulomb(sig, tau)
    assert abs(fit["tan_phi"] - 0.3) < 1e-10
    assert abs(fit["cohesion_pa"] - 100.0) < 1e-6


def test_mohr_coulomb_refuses_a_single_stress_level():
    """A single level gives the point value tau/sigma, never a slope."""
    with pytest.raises(ValueError, match="at least|>="):
        _fit_mohr_coulomb([800.0], [400.0])


def test_mohr_coulomb_refuses_duplicate_stress_levels():
    with pytest.raises(ValueError, match="distinct"):
        _fit_mohr_coulomb([800.0, 800.0], [400.0, 410.0])


# --------------------------------------------------------------------- device
def _tiny_case() -> ShearCellCase:
    return ShearCellCase(settle_time=0.002, consol_time=0.004,
                         shear_strain_target=0.004, sample_every=10)


def test_device_runs_and_returns_a_complete_record():
    out = simulate_shear_cell(_tiny_case())
    for key in ("valid", "invalid_reason", "mu_macro", "shear_stress",
                "normal_stress_measured", "normal_stress_rel_err",
                "n_particles", "total_steps", "timestep",
                "bed_top_history", "platen_z_history"):
        assert key in out, key
    assert out["n_particles"] > 0
    assert out["total_steps"] > 0
    assert out["timestep"] > 0.0
    assert isinstance(out["valid"], bool)
    # the gate must always explain itself when it fires
    assert (out["invalid_reason"] == "") == out["valid"]


def test_short_run_reproduces_the_open_platen_detachment():
    """Negative control: the platen does not hold sigma_n in a short run.

    This is the honest FAIL recorded for m13798, not an acceptance criterion:
    the bed top falls away from the platen, so the measured normal stress
    collapses and the validity gate must fire (a NaN/0 "friction angle" would be
    the bug the gate exists to stop).
    """
    out = simulate_shear_cell(_tiny_case())
    assert out["normal_stress_rel_err"] < -0.5
    assert out["valid"] is False
    assert "platen" in out["invalid_reason"]


def test_measure_macro_friction_refuses_to_fit_unloaded_levels():
    res = measure_macro_friction(_tiny_case(), normal_stresses=(400.0, 800.0))
    assert res["valid"] is False
    assert res["invalid_levels_pa"]
    assert "friction_angle_deg" not in res      # no fabricated envelope
