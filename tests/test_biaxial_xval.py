"""Tests for the biaxial-cell cross-validation operators and the virial stress.

Three things are asserted here:

1. the Love-Weber (virial) stress added to :class:`RidgeAssembly` reproduces an
   *analytic* uniaxial stress exactly on a bonded chain (so the comparison
   against the wall traction in the case is a comparison of definitions, not of
   two arbitrary numbers);
2. the OSS-route orientation operators (structure tensor / second moment)
   recover a synthetic band of known inclination on a *lattice* point set, i.e.
   they are usable as a reference for the production Radon operator;
3. the two operator families agree within the pre-declared XVAL tolerance on the
   same strain field, and the ``wall_friction`` switch documented for the A/B
   experiment actually reaches all four planes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.biaxial_shear_band import _make_specimen
from tensordem.biaxial_xval import (
    XVAL_TOLERANCE,
    cross_validate_synthetic,
    orientation_estimators,
    second_moment_angle,
    structure_tensor_angle,
)
from tensordem.biaxial_band_diagnostics import synthetic_band_configuration
from tensordem.ridge_dem import RidgeAssembly, RidgeMaterial


def _chain(stretch: float, *, tension: bool) -> tuple[RidgeAssembly, float, float]:
    """Three bonded discs in a vertical chain, both bonds strained by ``stretch``."""
    material = RidgeMaterial(particle_radius=0.5)
    radius = material.particle_radius
    length = 2.0 * radius
    positions = torch.tensor([[0.0, 0.0], [0.0, length], [0.0, 2.0 * length]],
                             dtype=torch.float64)
    bonds = torch.tensor([[0, 1], [1, 2]])
    assembly = RidgeAssembly(positions,
                             torch.full((3,), radius, dtype=torch.float64),
                             torch.full((3,), material.particle_mass, dtype=torch.float64),
                             bonds, material, thickness=1.0)
    sign = 1.0 if tension else -1.0
    assembly.pos = assembly.pos + torch.tensor(
        [[0.0, -sign * stretch], [0.0, 0.0], [0.0, sign * stretch]],
        dtype=torch.float64)
    height = 2.0 * length + (2.0 * stretch if tension else -2.0 * stretch)
    volume = (2.0 * radius) * height * 1.0
    area = 2.0 * radius * 1.0
    return assembly, volume, area


@pytest.mark.parametrize("tension", [True, False])
def test_love_weber_stress_matches_analytic_chain(tension: bool) -> None:
    material = RidgeMaterial(particle_radius=0.5)
    stretch = 1.0e-5
    assembly, volume, area = _chain(stretch, tension=tension)
    out = assembly.forces(dt=None, update_memory=False)
    # tension-positive convention: sigma = +virial / V
    sigma = float(out["virial"][1, 1]) / volume
    analytic = material.bond_stiffness * stretch / area * (1.0 if tension else -1.0)
    assert sigma == pytest.approx(analytic, rel=1e-9)
    # compression-positive convention used by the biaxial case
    assert -sigma == pytest.approx(-analytic, rel=1e-9)
    # no lateral load -> no lateral stress
    assert abs(float(out["virial"][0, 0])) < 1e-9 * abs(float(out["virial"][1, 1]))


def _lattice(columns: int = 16, rows: int = 32) -> tuple[np.ndarray, dict]:
    material = RidgeMaterial(particle_radius=0.75)
    assembly, geometry = _make_specimen(material, columns=columns, rows=rows,
                                       confining_stress=100.0e3)
    return assembly.pos.detach().numpy().copy(), geometry


def test_structure_tensor_recovers_known_band_on_lattice() -> None:
    pos, geometry = _lattice()
    margin = 2.0 * 2.0 * 0.75
    window = (-geometry["half_w"] + margin, geometry["half_w"] - margin,
              -geometry["half_h"] + margin, geometry["half_h"] - margin)
    moved = synthetic_band_configuration(pos, angle_deg=55.0, slip=3.0, band_width=4.5)
    got = orientation_estimators(pos, moved, bounds=window)
    truth = 55.0
    assert abs(got["structure_tensor"]["folded_angle_deg"] - truth) / truth \
        <= XVAL_TOLERANCE
    assert abs(got["ours"]["folded_angle_deg"] - truth) / truth <= XVAL_TOLERANCE


def test_two_operators_agree_within_tolerance_on_real_case_lattice() -> None:
    pos, geometry = _lattice(24, 48)
    margin = 3.0
    window = (-geometry["half_w"] + margin, geometry["half_w"] - margin,
              -geometry["half_h"] + margin, geometry["half_h"] - margin)
    moved = synthetic_band_configuration(pos, angle_deg=60.0, slip=4.5, band_width=4.5)
    got = orientation_estimators(pos, moved, bounds=window)
    ours = got["ours"]["folded_angle_deg"]
    tensor = got["structure_tensor"]["folded_angle_deg"]
    assert abs(ours - tensor) / tensor <= XVAL_TOLERANCE
    assert got["structure_tensor"]["cells_used"] > 5


def test_second_moment_orientation_on_isotropic_patch() -> None:
    # square patch, 45 deg band: the second moment must see the band, not the box
    xs, ys = np.meshgrid(np.linspace(0.0, 20.0, 21), np.linspace(0.0, 20.0, 21))
    x = xs.ravel()
    y = ys.ravel()
    s = (x * math.cos(math.radians(45.0)) + y * math.sin(math.radians(45.0)))
    values = np.exp(-0.5 * (s - s.mean()) ** 2)
    out = second_moment_angle(x, y, values)
    assert abs(out["folded_angle_deg"] - 45.0) < 5.0


def test_degenerate_inputs_raise() -> None:
    from tensordem.biaxial_shear_band import BandMeasurementError
    with pytest.raises(BandMeasurementError):
        structure_tensor_angle([0.0, 1.0], [0.0, 1.0], [1.0, 1.0])
    with pytest.raises(BandMeasurementError):
        second_moment_angle(np.zeros(3), np.zeros(3), np.zeros(3))


def test_synthetic_cross_validation_criterion_structure() -> None:
    pos, geometry = _lattice(12, 24)
    margin = 3.0
    window = (-geometry["half_w"] + margin, geometry["half_w"] - margin,
              -geometry["half_h"] + margin, geometry["half_h"] - margin)
    out = cross_validate_synthetic(pos, angles_deg=(45.0, 60.0),
                                   measurement_window=window, cells=30)
    assert set(out["criterion"]) == {
        "structure_tensor_vs_truth_pass", "second_moment_vs_truth_pass",
        "ours_vs_structure_tensor_pass", "ours_vs_second_moment_pass"}
    assert all(np.isfinite(row["ours_deg"]) for row in out["rows"])
    assert out["tolerance"] == XVAL_TOLERANCE
    # the band around the anchor is where the operators must agree
    row = [r for r in out["rows"] if r["injected_deg"] == 60.0][0]
    assert row["ours_vs_tensor_rel"] <= XVAL_TOLERANCE


@pytest.mark.parametrize("friction", [None, 0.0])
def test_wall_friction_switch_reaches_all_planes(friction: float | None) -> None:
    material = RidgeMaterial(particle_radius=0.75)
    assembly, geometry = _make_specimen(material, columns=8, rows=16,
                                       confining_stress=100.0e3,
                                       wall_friction=friction)
    expected = material.friction_coefficient if friction is None else friction
    assert [plane.friction for plane in assembly.planes] == [expected] * 4
    assert geometry["wall_friction"] == pytest.approx(expected)