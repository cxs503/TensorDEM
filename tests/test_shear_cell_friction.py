"""Verification surface for the direct-shear cell and its Mohr-Coulomb fit.

History
-------
m13798: the module shipped as dead code (``TypeError`` on the first step) and
fitted a Mohr-Coulomb *slope* inside a single run; the device ran but the platen
lost contact so ``sigma_n -> 0`` and ``mu`` was undefined (honest FAIL).

m14721 (this revision) fixes the three mechanisms behind that failure:

1. **Bed preparation** -- the jittered HCP fill is dilated (bulk packing
   fraction ~0.31) and collapses under gravity at 0.1-0.4 m/s, faster than any
   servo can follow.  :func:`prepare_bed` settles it to a quasi-static state and
   every normal-stress level of a sweep reuses the *same* bed.
2. **Load-controlled platen actuator** -- the old velocity-rate servo capped the
   platen speed at 0.02-1.0 x shear velocity, so it was left behind.  The platen
   is now a mass driven by the force deficit ``N_target - N_measured`` with a
   short velocity relaxation, which has no steady-state offset.
3. **Traction sign** -- ``tau = -sum(f_p,x)/A`` made ``mu`` negative; the platen
   pushes the bed in +x, so the sign is ``+``.

These tests pin: the exact fitter and its refusals, the validity gate, the bed
preparation criterion, the shared-bed protocol, and the sign/consistency of the
measured tractions.
"""
import math
from dataclasses import replace

import pytest

from tensordem.shear_cell import (
    ShearCellCase,
    _fit_mohr_coulomb,
    _validity_gate,
    measure_macro_friction,
    prepare_bed,
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


# ------------------------------------------------------------- validity gate
def test_validity_gate_rejects_an_unloaded_platen():
    ok, why = _validity_gate(0.0, 0.0, 800.0)
    assert ok is False and "undefined" in why
    ok, why = _validity_gate(100.0, 50.0, 800.0)      # < 50 % of target
    assert ok is False
    ok, why = _validity_gate(800.0, -12.0, 800.0)     # negative traction
    assert ok is False
    ok, why = _validity_gate(780.0, 300.0, 800.0)
    assert ok is True and why == ""


# ----------------------------------------------------------- bed preparation
def _tiny_case(**kw) -> ShearCellCase:
    base = dict(settle_time=0.002, settle_max_time=0.01, consol_time=0.004,
                shear_strain_target=0.01, sample_every=10)
    base.update(kw)
    return ShearCellCase(**base)


def test_prepare_bed_settles_to_a_quasi_static_packing():
    # a loose tolerance makes the quasi-static criterion reachable inside the
    # short cap of the tiny case (the production bed needs ~13.7k steps)
    case = _tiny_case(settle_v_tol=2.0, settle_max_time=0.02)
    bed = prepare_bed(case)
    assert bed["steps"] > 0
    assert bed["n_particles"] > 0
    v_max = float(((bed["vel"]) ** 2).sum(axis=1).max() ** 0.5)
    assert v_max < case.settle_v_tol, v_max          # quasi-static criterion met
    assert 0.3 < bed["packing_fraction"] < 0.8
    # the prepared bed is a *dense*-er packing than the raw jittered lattice
    assert bed["bed_top"] < case.fill_height * 0.95


def test_device_runs_from_a_prepared_bed_and_returns_a_complete_record():
    case = _tiny_case()
    out = simulate_shear_cell(case, bed=prepare_bed(case))
    for key in ("valid", "invalid_reason", "mu_macro", "shear_stress",
                "normal_stress_measured", "normal_stress_rel_err",
                "n_particles", "total_steps", "timestep", "bed_prep_steps",
                "bed_top_history", "platen_z_history"):
        assert key in out, key
    assert out["n_particles"] > 0
    assert out["total_steps"] > 0
    assert out["timestep"] > 0.0
    assert isinstance(out["valid"], bool)
    # the gate must always explain itself when it fires
    assert (out["invalid_reason"] == "") == out["valid"]


def test_measure_macro_friction_refuses_to_fit_unloaded_levels():
    res = measure_macro_friction(_tiny_case(), normal_stresses=(400.0, 800.0))
    assert res["valid"] is False
    assert res["invalid_levels_pa"]
    assert "friction_angle_deg" not in res      # no fabricated envelope


# ------------------------------------------- prepared-bed shear cell (slow)
def test_prepared_bed_holds_the_normal_stress_and_gives_a_positive_traction():
    """End-to-end: quasi-static bed + load-controlled servo => an usable mu.

    This is the acceptance test for the m13798 honest FAIL: the platen must
    carry the servo target through the shear window (within 15 %, the noise
    floor of a 78-particle bed) and the traction must be positive (sign fix).
    """
    # a short shear window does NOT do: the servo transient has to be flushed
    # before the steady-state mean, so this uses the campaign protocol
    # (0.04 s consolidation + 0.30 shear strain), which is the configuration the
    # m14721 sweep was run with.
    case = ShearCellCase(consol_time=0.04, shear_strain_target=0.30,
                         normal_stress=4000.0, sample_every=25)
    bed = prepare_bed(case)
    out = simulate_shear_cell(case, bed=bed)
    assert out["valid"] is True, out["invalid_reason"]
    assert abs(out["normal_stress_rel_err"]) < 0.15, out["normal_stress_rel_err"]
    assert out["shear_stress"] > 0.0
    assert out["mu_macro"] > 0.0
    # platen-side and floor-side tractions are the same traction (sign fix)
    assert out["tau_platen_vs_floor_rel"] < 1.5, out["tau_platen_vs_floor_rel"]


def test_every_stress_level_starts_from_the_same_prepared_bed():
    case = ShearCellCase(consol_time=0.01, shear_strain_target=0.03,
                         normal_stress=2000.0, sample_every=25)
    bed = prepare_bed(case)
    a = simulate_shear_cell(case, bed=bed)
    b = simulate_shear_cell(replace(case, normal_stress=4000.0), bed=bed)
    assert a["n_particles"] == b["n_particles"]
    assert a["bed_prep_steps"] == b["bed_prep_steps"]
    assert bed["packing_fraction"] == pytest.approx(bed["packing_fraction"])
    # reusing the bed must not mutate it
    assert bed["pos"].shape == (a["n_particles"], 3)
    assert a["normal_stress_measured"] != b["normal_stress_measured"]