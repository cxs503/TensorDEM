"""Wall tangential velocity in the friction contact (needed by the shear cell)."""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.granular2d.engine import DiskConfig, DiskSystem, box_walls


def _config() -> DiskConfig:
    return DiskConfig(kn=1.0e5, kt_ratio=0.5, zeta=0.3, mu=0.5,
                      mu_rolling=0.0, density=2500.0, thickness=0.01,
                      gravity=(0.0, -9.81))


def test_moving_wall_drags_a_disk():
    walls = box_walls(0.2, 0.1, left=False, right=False)
    sys_ = DiskSystem([[0.1, 0.0051]], [0.005], _config(), walls=walls,
                      wall_velocity=[[0.5, 0.0], [0.5, 0.0]])
    sys_.step(3000)
    assert sys_.vel[0, 0] > 0.02          # driven by friction, not teleported


def test_static_wall_does_not_drag():
    walls = box_walls(0.2, 0.1, left=False, right=False)
    sys_ = DiskSystem([[0.1, 0.0051]], [0.005], _config(), walls=walls)
    sys_.step(3000)
    assert abs(float(sys_.vel[0, 0])) < 1e-9


def test_wall_velocity_shape_is_validated():
    walls = box_walls(0.2, 0.1, left=False, right=False)
    with pytest.raises(ValueError):
        DiskSystem([[0.1, 0.02]], [0.005], _config(), walls=walls,
                   wall_velocity=[[0.5, 0.0]])
    with pytest.raises(ValueError):
        DiskSystem([[0.1, 0.02]], [0.005], _config(), walls=walls,
                   wall_velocity=[[0.5, np.nan], [0.5, 0.0]])


def test_default_is_static():
    walls = box_walls(0.2, 0.1, left=False, right=False)
    sys_ = DiskSystem([[0.1, 0.02]], [0.005], _config(), walls=walls)
    assert sys_.wall_velocity is None
