"""Rolling-lever conventions: the arithmetic that the macro arm tests.

``docs/XVAL_DEM_ROUND2_METHOD_SELECTION.md`` §E3 measured on a pair scene that
TensorDEM's ``mu_r`` maps onto the LIGGGHTS EPSD2 convention as
``mu_r,OSS = 2 * mu_r,TD``.  The macro arm
(``docs/XVAL_DEM_ROUND2_COLUMN_ARMS_MACRO.md``) checks whether that mapping
also survives a full column collapse.  These tests pin the *arithmetic* it
relies on, per contact kind, so that a future change to the lever convention
cannot silently invalidate the macro conclusion:

* pair contact: cap ``mu_r * Fn * r_eff`` is equal under the mapping (exactly),
  the spring ``k_r = kt r_eff^2`` is **not** (4x softer) — the equivalence is
  Coulomb-saturated only;
* wall contact: ``r_eff = r`` under both conventions, so the mapping *doubles*
  the wall cap and leaves ``k_r`` unchanged.

No integration, no randomness: pure property checks (< 1 s).
"""
from __future__ import annotations

import math

import pytest

from tensordem.granular3d_core import Material3D


def test_pair_cap_is_equal_under_the_published_mapping():
    legacy = Material3D(radius=0.03, kn=6.0e4, kt_ratio=0.5, mu_rolling=0.5,
                        rolling_lever="particle_radius")
    mapped = Material3D(radius=0.03, kn=6.0e4, kt_ratio=0.5, mu_rolling=1.0,
                        rolling_lever="reduced_radius")
    fn = 1.0
    cap_legacy = legacy.mu_rolling * fn * legacy.pair_rolling_reff
    cap_mapped = mapped.mu_rolling * fn * mapped.pair_rolling_reff
    assert cap_mapped == pytest.approx(cap_legacy, rel=1e-15)


def test_pair_stiffness_is_not_preserved_by_the_mapping():
    legacy = Material3D(radius=0.03, kn=6.0e4, kt_ratio=0.5, rolling_lever="particle_radius")
    mapped = Material3D(radius=0.03, kn=6.0e4, kt_ratio=0.5, rolling_lever="reduced_radius")
    # kr = kt reff^2 with reff halved -> factor 4, independent of mu_r.
    assert mapped.pair_rolling_kr == pytest.approx(legacy.pair_rolling_kr / 4.0, rel=1e-15)


def test_wall_cap_is_doubled_by_the_mapping():
    legacy = Material3D(radius=0.03, kn=6.0e4, kt_ratio=0.5, mu_rolling=0.5,
                        rolling_lever="particle_radius")
    mapped = Material3D(radius=0.03, kn=6.0e4, kt_ratio=0.5, mu_rolling=1.0,
                        rolling_lever="reduced_radius")
    fn = 1.0
    cap_legacy = legacy.mu_rolling * fn * legacy.wall_rolling_reff
    cap_mapped = mapped.mu_rolling * fn * mapped.wall_rolling_reff
    assert cap_mapped == pytest.approx(2.0 * cap_legacy, rel=1e-15)
    assert mapped.wall_rolling_kr == pytest.approx(legacy.wall_rolling_kr, rel=1e-15)


def test_default_convention_is_the_legacy_one():
    mat = Material3D(radius=0.03)
    assert mat.rolling_lever == "particle_radius"
    assert mat.pair_rolling_reff == pytest.approx(mat.radius, rel=1e-15)
    assert mat.pair_rolling_kr == pytest.approx(mat.kt * mat.radius ** 2, rel=1e-15)


def test_unknown_lever_is_rejected():
    # The dataclass stores the string; validation happens when a lever arm is
    # requested, so both contact kinds must refuse it.
    mat = Material3D(radius=0.03, rolling_lever="centre_of_mass")
    with pytest.raises(ValueError):
        _ = mat.pair_rolling_reff
    with pytest.raises(ValueError):
        _ = mat.wall_rolling_reff