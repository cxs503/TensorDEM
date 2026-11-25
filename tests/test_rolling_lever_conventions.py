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

The pair-force kernel is also checked against the spring law and torque cap.
"""
from __future__ import annotations

import numpy as np

import pytest

from tensordem.granular3d_core import ContactState, Material3D, contact_forces


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


@pytest.mark.parametrize("lever,arm", [("particle_radius", .5), ("reduced_radius", .25)])
@pytest.mark.parametrize("torsion", [False, True])
@pytest.mark.parametrize("spin_rate", [0.1, 100.])
def test_kernel_rolling_couple(lever, arm, torsion, spin_rate, record_property):
    mat = Material3D(radius=.5, kn=100., kt_ratio=.5, mu=0., mu_rolling=.2,
                     rolling_lever=lever, rolling_torsion=torsion)
    pos = np.array([[0., 0., 0.], [.75, 0., 0.]])
    omega = np.array([[0., 0., 0.], [spin_rate, spin_rate, 0.]])
    dt = .1
    _, torque = contact_forces(pos, np.zeros_like(pos), omega, np.array([0]), np.array([1]),
                               ContactState(), mat, dt)
    twist = dt * np.array([spin_rate if torsion else 0., spin_rate, 0.])
    trial = 50. * arm**2 * twist
    cap = .2 * 25. * arm
    expected = trial * min(1., cap / np.linalg.norm(trial))
    record_property("max_abs_torque_error_Nm", np.max(np.abs(torque - [expected, -expected])))
    np.testing.assert_allclose(torque, [expected, -expected], atol=1e-14, rtol=1e-14)
