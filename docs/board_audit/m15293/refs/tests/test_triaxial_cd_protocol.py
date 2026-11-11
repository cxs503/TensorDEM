"""Protocol tests for the drained (CD) triaxial driver (tensordem.triaxial_cd)."""
from __future__ import annotations

import math

import pytest

from tensordem.triaxial_cd import CDTriaxialCase, friction_angle_deg, run_cd_triaxial


def _tiny(**kw) -> CDTriaxialCase:
    base = dict(nx=4, ny=4, nz=3, sigma3=1.0e5, lattice_gap=0.005, jitter=0.05,
                axial_strain_rate=1.0,
                axial_strain_target=0.06, sample_every=50,
                consolidate_steps=2500)
    base.update(kw)
    return CDTriaxialCase(**base)


def test_friction_angle_matches_mohr_coulomb():
    # sigma1 = 3 sigma3  ->  sin(phi) = 1/2  ->  phi = 30 deg exactly
    assert friction_angle_deg(3.0, 1.0) == pytest.approx(30.0, abs=1e-9)
    # hydrostatic state has no shear strength
    assert friction_angle_deg(2.0, 2.0) == pytest.approx(0.0, abs=1e-9)
    # degenerate input must not raise
    assert math.isnan(friction_angle_deg(0.0, 0.0))


def test_case_rejects_inadmissible_input():
    with pytest.raises(ValueError):
        CDTriaxialCase(sigma3=0.0)          # drained test needs sigma3 > 0
    with pytest.raises(ValueError):
        CDTriaxialCase(nx=2)                # specimen too small


def test_consolidation_reaches_confining_pressure():
    """The lateral barostat + platen servo must realise sigma3 before shear."""
    res = run_cd_triaxial(_tiny())
    conf = res["consolidation"]
    assert abs(conf["lateral_relative_error"]) <= 0.03, conf
    # wall-branch Love-Weber closure: the axial platen pressure must agree with
    # the internal sigma_zz (this is the anchor that the W-L sum is complete)
    assert abs(conf["axial_relative_error"]) <= 0.05, conf


def test_axial_shear_develops_strength_and_dilates():
    """Shearing must raise q > 0 and report the drained gate + dilatancy."""
    res = run_cd_triaxial(_tiny())
    assert res["peak"]["q_Pa"] > 0.0, res["peak"]
    assert res["peak"]["phi_deg"] > 0.0
    assert res["axial_strain_reached"] == pytest.approx(0.06, rel=0.05)
    assert res["drained_gate"]["pass"], res["drained_gate"]
    assert set(res["dilatancy"]) >= {"volumetric_strain_min", "volumetric_strain_end"}
    assert res["quasi_static"]["pass"]


def test_run_is_deterministic():
    a = run_cd_triaxial(_tiny(consolidate_steps=1200, axial_strain_target=0.03))
    b = run_cd_triaxial(_tiny(consolidate_steps=1200, axial_strain_target=0.03))
    assert a["peak"]["phi_deg"] == pytest.approx(b["peak"]["phi_deg"], rel=1e-12)

