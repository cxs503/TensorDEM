"""Tests for the structure-tensor shear-band operator (R3 cross-check).

The operator is validated on synthetic fields of *known* inclination, both on an
isotropic random cloud and on the elongated 24 x 48 lattice of the production
case, because the R2 regression showed that an operator validated only on a
square fixture can carry a large aspect-ratio bias into the case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.biaxial_shear_band import (
    BandMeasurementError,
    fold_angle_deg,
    local_shear_strain,
)
from tensordem.shear_band_structure_tensor import (
    grid_scalar_field,
    measure_band_structure_tensor,
    structure_tensor_band_angle,
)


def _case_lattice(columns: int = 24, rows: int = 48, seed: int = 20261031) -> np.ndarray:
    """Jittered triangular disc lattice identical to the biaxial specimen."""
    import torch

    from tensordem.biaxial_shear_band import _make_specimen
    from tensordem.ridge_dem import RidgeMaterial

    material = RidgeMaterial(particle_radius=0.75, freeze_bond_fraction=0.35,
                             seed=seed)
    assembly, _ = _make_specimen(material, columns=columns, rows=rows,
                                 confining_stress=100.0e3, jitter=0.03,
                                 servo_gain=3.0e-4, servo_velocity_limit=2.0,
                                 wall_friction=0.55)
    return assembly.pos.numpy()


def _slip_band(points: np.ndarray, angle_deg: float, slip: float = 0.02,
               width: float = 1.5) -> np.ndarray:
    """Displace a lattice with a tangential slip band of known inclination."""
    theta = math.radians(angle_deg)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    tangent = np.array([math.cos(theta), math.sin(theta)])
    s = points @ normal
    ramp = 0.5 * (1.0 + np.tanh(s / width))
    disp = np.outer(slip * ramp, tangent)
    disp[:, 0] += -1.0e-3 * points[:, 0]
    disp[:, 1] += 2.0e-3 * points[:, 1]
    return points + disp


@pytest.mark.parametrize("angle", [30.0, 45.0, 60.0, 70.0])
def test_structure_tensor_recovers_case_lattice_band(angle):
    """Known-angle bands on the elongated case lattice are recovered to <= 2.5 deg."""
    points = _case_lattice()
    now = _slip_band(points, angle)
    result = measure_band_structure_tensor(points, now)
    assert abs(result["folded_angle_deg"] - angle) <= 2.5
    assert result["coherence"] > 0.9
    assert result["folded_angle_deg"] == pytest.approx(
        fold_angle_deg(result["angle_deg"]))


def test_structure_tensor_is_aspect_ratio_neutral():
    """The same physical inclination is measured identically on 24x48 and 16x32."""
    wide = _case_lattice(columns=24, rows=48)
    narrow = _case_lattice(columns=16, rows=32)
    a = measure_band_structure_tensor(wide, _slip_band(wide, 60.0))
    b = measure_band_structure_tensor(narrow, _slip_band(narrow, 60.0))
    assert abs(a["folded_angle_deg"] - b["folded_angle_deg"]) <= 2.0


def test_structure_tensor_rejects_uniform_deformation():
    """A homogeneous deformation has no band: the operator must refuse, not guess."""
    points = _case_lattice()
    uniform = points.copy()
    uniform[:, 0] *= 1.0 - 5.0e-4
    uniform[:, 1] *= 1.0 + 1.0e-3
    with pytest.raises(BandMeasurementError):
        measure_band_structure_tensor(points, uniform)


def test_structure_tensor_rejects_degenerate_input():
    with pytest.raises(BandMeasurementError):
        grid_scalar_field([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])     # too few samples
    points = _case_lattice()
    with pytest.raises(BandMeasurementError):
        measure_band_structure_tensor(points, _slip_band(points, 60.0),
                                      bounds=(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(BandMeasurementError):
        structure_tensor_band_angle(np.ones((5, 5)), np.ones((5, 5), dtype=bool))


def test_grid_fills_empty_cells_without_nan():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 10.0, 300)
    y = rng.uniform(0.0, 10.0, 300)
    grid = grid_scalar_field(x, y, rng.normal(size=300), cell=0.4)
    assert np.isfinite(grid["field"]).all()
    assert grid["occupancy"].any()
    assert not grid["occupancy"].all()          # sparse scatter leaves empty cells


def test_local_shear_strain_rejects_shape_mismatch():
    points = _case_lattice()
    with pytest.raises(BandMeasurementError):
        local_shear_strain(points, points[:-1])