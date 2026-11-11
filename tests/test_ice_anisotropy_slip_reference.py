"""Tests for the Schmid slip-system reference (ice_anisotropy.slip_systems)."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_anisotropy.fabric import FabricODF
from tensordem.ice_anisotropy.slip_systems import (
    SlipSystemModel,
    calibrate_pyramidal_ratio,
    load_direction,
    resolved_shear_factor,
    schmid_factor,
    uniaxial_stress_tensor,
)


def test_schmid_factor_closed_form_and_limits():
    planes = np.array([[0.0, 0.0, 1.0]])      # plane normal along z
    assert schmid_factor(planes, [0.0, 0.0, 1.0]) == pytest.approx(0.0)   # no shear
    assert schmid_factor(planes, [1.0, 0.0, 0.0]) == pytest.approx(0.0)   # in-plane
    d45 = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    assert schmid_factor(planes, d45) == pytest.approx(0.5)               # maximum
    # general stress version agrees with the closed form for uniaxial loading
    dirs = load_direction(37.0)
    S = uniaxial_stress_tensor(dirs)
    assert resolved_shear_factor(planes, S) == pytest.approx(
        schmid_factor(planes, dirs), rel=1e-12)


def test_basal_slip_is_inactive_along_the_columns():
    """Zero resolved shear on every basal plane for a column-parallel load: the
    micro-mechanical reason the column direction is the strong one."""
    odf = FabricODF.columnar(n_azimuth=180)
    model = SlipSystemModel(pyramids_tau_ratio=5.0)
    fac = model.mechanism_factors(odf, load_direction(0.0))
    assert fac["basal"] == pytest.approx(0.0, abs=1e-12)
    assert fac["pyramidal"] == pytest.approx(0.5, rel=1e-12)
    strengths = model.mechanism_strengths(odf, load_direction(0.0))
    assert strengths["basal"] is None


def test_calibration_reproduces_the_target_ratio():
    odf = FabricODF.columnar(n_azimuth=180)
    for target in (3.0, 3.5, 5.0):
        r = calibrate_pyramidal_ratio(odf, target)
        model = SlipSystemModel(pyramids_tau_ratio=r)
        assert model.strength_ratio(odf) == pytest.approx(target, rel=1e-8)


def test_transverse_shear_strength_is_half_the_transverse_strength():
    """<|cos phi|> = 2/pi for transverse shear vs 1/pi for horizontal uniaxial
    loading, so tau_transverse / sigma_perp = 1/2 for a columnar fabric."""
    odf = FabricODF.columnar(n_azimuth=720)
    model = SlipSystemModel(pyramids_tau_ratio=4.0)
    shear = model.transverse_shear_strength(odf)
    sperp = model.uniaxial_strength(odf, load_direction(90.0))
    assert shear["schmid_basal"] == pytest.approx(2.0 / np.pi, rel=1e-4)
    assert shear["shear_strength_Pa"] / sperp == pytest.approx(0.5, rel=1e-4)


def test_sachs_bound_is_below_the_mean_field_bound():
    odf = FabricODF.columnar(n_azimuth=360)
    r = calibrate_pyramidal_ratio(odf, 3.5, convention="mean")
    mean = SlipSystemModel(pyramids_tau_ratio=r, convention="mean")
    sachs = SlipSystemModel(pyramids_tau_ratio=r, convention="max")
    for th in (10.0, 30.0, 45.0, 60.0, 90.0):
        assert sachs.uniaxial_strength(odf, load_direction(th)) <= mean.uniaxial_strength(
            odf, load_direction(th)) * (1.0 + 1e-12)


def test_uniform_fabric_gives_an_isotropic_slip_response():
    # a nonlinear angular average needs the dense isotropic fabric: the
    # cubic-orbit set is exact in a_ij but quadrature biased for this average
    odf = FabricODF.uniform_dense()
    model = SlipSystemModel(pyramids_tau_ratio=5.5)
    sperp = model.uniaxial_strength(odf, load_direction(90.0))
    vals = [model.uniaxial_strength(odf, load_direction(th)) / sperp
            for th in (0.0, 30.0, 45.0, 60.0)]
    assert max(abs(v - 1.0) for v in vals) < 0.05
    # ... and the shear/axial ratio lands near the von Mises value 1/sqrt(3)
    shear = model.transverse_shear_strength(odf)["shear_strength_Pa"]
    assert shear / sperp == pytest.approx(1.0 / np.sqrt(3.0), rel=0.05)


def test_strength_curve_angles_and_governing_mechanism():
    odf = FabricODF.columnar(n_azimuth=180)
    model = SlipSystemModel(pyramids_tau_ratio=5.5)
    rows = model.strength_curve(odf, [0.0, 45.0, 90.0])
    assert [r["theta_deg"] for r in rows] == [0.0, 45.0, 90.0]
    assert rows[0]["governing"] == "pyramidal"
    assert rows[1]["governing"] == "basal"
    assert rows[2]["governing"] == "basal"
    # the 45 deg dip is shallower than the 0 deg peak
    assert rows[1]["strength_Pa"] < rows[2]["strength_Pa"] < rows[0]["strength_Pa"]


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        SlipSystemModel(basal_tau=0.0)
    with pytest.raises(ValueError):
        SlipSystemModel(pyramids_tau_ratio=-1.0)
    with pytest.raises(ValueError):
        SlipSystemModel(convention="taylor")
    with pytest.raises(ValueError):
        calibrate_pyramidal_ratio(FabricODF.columnar(), 1.0)