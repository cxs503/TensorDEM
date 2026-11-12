"""The 2-D plane-shear stress ordinate must come from the shared instrument.

Three things are pinned here:

1. ``pair_stress`` / ``pair_stress_components`` are *the instrument*, not a
   private re-derivation: they agree with
   ``granular2d.stress_arrays`` (hence with ``granular3d.stress.contact_outer``)
   to machine precision on arbitrary contact sets.
2. The gap between the two off-diagonal components is exactly the lever-arm
   torque of the tangential forces: zero for central forces, and equal to
   ``sum |l| f_t / A`` analytically for a frictional contact set.
3. In a real plane-shear run the two ordinates agree to machine precision when
   friction is switched off, and separate when it is not -- so the reported
   ``tau_symmetry_defect`` is a measurement, not a constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.elastic_limit_shear import (
    ElasticLimitShearCase,
    pair_stress,
    pair_stress_components,
    simulate_plane_shear,
)
from tensordem.granular2d.stress_arrays import (
    contact_tensors,
    pressure_2d,
    shear_components,
    virial,
)
from tensordem.granular3d.stress import contact_outer


def _random_contacts(n=64, seed=7):
    rng = np.random.default_rng(seed)
    dx = rng.normal(size=n)
    dy = rng.normal(size=n)
    # arbitrary (non-central) force set: normal + tangential parts
    ux, uy = dx / np.hypot(dx, dy), dy / np.hypot(dx, dy)
    fn = rng.uniform(0.5, 3.0, size=n)
    ft = rng.uniform(-0.5, 0.5, size=n)
    fx = -fn * ux + ft * (-uy)
    fy = -fn * uy + ft * ux
    return dx, dy, fx, fy, ux, uy, fn, ft


def test_pair_stress_is_the_instrument():
    dx, dy, fx, fy, *_ = _random_contacts()
    area = 3.5
    tau, pres = pair_stress(dx, dy, fx, fy, area)

    sigma = virial(np.stack([dx, dy], axis=1), np.stack([fx, fy], axis=1), area)
    assert tau == pytest.approx(float(sigma[1, 0]), rel=1e-13, abs=1e-13)
    assert pres == pytest.approx(pressure_2d(sigma), rel=1e-13, abs=1e-13)

    comp = pair_stress_components(dx, dy, fx, fy, area)
    assert comp["sigma_yx"] == pytest.approx(tau, rel=1e-13)
    assert comp["pressure"] == pytest.approx(pres, rel=1e-13)
    assert comp["tau_symmetric"] == pytest.approx(0.5 * (sigma[0, 1] + sigma[1, 0]),
                                                  rel=1e-13)


def test_2d_wrapper_is_the_3d_instrument_on_padded_vectors():
    dx, dy, fx, fy, *_ = _random_contacts(seed=11)
    branch2 = np.stack([dx, dy], axis=1)
    force2 = np.stack([fx, fy], axis=1)
    pad_l = np.zeros((branch2.shape[0], 3))
    pad_f = np.zeros_like(pad_l)
    pad_l[:, :2] = branch2
    pad_f[:, :2] = force2
    np.testing.assert_allclose(contact_tensors(branch2, force2),
                               contact_outer(pad_l, pad_f), rtol=0.0, atol=0.0)


def test_central_forces_give_a_symmetric_virial():
    dx, dy, fx, fy, ux, uy, _, _ = _random_contacts(seed=3)
    fn = np.hypot(fx, fy)
    fx_c, fy_c = -fn * ux, -fn * uy          # purely central (frictionless)
    comp = pair_stress_components(dx, dy, fx_c, fy_c, 2.0)
    assert comp["sigma_xy"] == pytest.approx(comp["sigma_yx"], rel=1e-14, abs=1e-14)
    assert comp["tau_antisymmetric"] == pytest.approx(0.0, abs=1e-14)
    assert comp["asymmetry_relative"] == pytest.approx(0.0, abs=1e-13)


def test_tangential_gap_is_the_lever_arm_torque():
    dx, dy, fx, fy, ux, uy, fn, ft = _random_contacts(seed=5)
    area = 4.0
    comp = pair_stress_components(dx, dy, fx, fy, area)
    lever = float(np.sum(dx * fy - dy * fx)) / area
    assert comp["sigma_xy"] - comp["sigma_yx"] == pytest.approx(lever, rel=1e-13)
    # and that equals the per-contact |l| f_t (documented identity)
    length = np.hypot(dx, dy)
    assert comp["sigma_xy"] - comp["sigma_yx"] == pytest.approx(
        float(np.sum(length * ft)) / area, rel=1e-12)


def test_single_repulsive_pair_pressure_and_shear_are_the_analytics():
    # i at origin, j at (1, 1); repulsive force on i along -(1,1)
    dx, dy = np.array([1.0]), np.array([1.0])
    d = math.hypot(1.0, 1.0)
    fn, area = 3.0, 2.0
    fx, fy = -fn * dx / d, -fn * dy / d
    comp = pair_stress_components(dx, dy, fx, fy, area)
    assert comp["pressure"] == pytest.approx(fn * d / (2.0 * area))
    assert comp["sigma_xy"] == pytest.approx(fx * dy[0] / area)
    assert comp["sigma_yx"] == pytest.approx(fx * dy[0] / area)


def _short_run(friction: float) -> dict:
    case = ElasticLimitShearCase(n_particles=200, shear_rate=1.0,
                                 friction=friction, strain_average_from=0.05)
    n_steps = max(200, int(0.5 / (case.shear_rate * case.dt)))
    return simulate_plane_shear(case, n_steps_override=n_steps)


def test_frictionless_run_has_no_ordinate_gap():
    res = _short_run(0.0)
    assert np.isfinite(res["tau"]) and np.isfinite(res["tau_symmetric"])
    assert res["tau_symmetry_defect"] == pytest.approx(0.0, abs=1e-12)
    assert res["tau"] == pytest.approx(res["tau_symmetric"], rel=1e-12)


def test_frictional_run_reports_a_measurable_gap():
    res = _short_run(0.5)
    assert np.isfinite(res["tau_symmetry_defect"])
    # the raw virial is not symmetric, so the two ordinates must differ, and the
    # per-sample asymmetry must be visible even where the time average cancels
    assert 0.0 < abs(res["tau_minus_symmetric"]) < 0.5 * abs(res["tau_symmetric"])
    assert res["stress_asymmetry_relative"] > 1e-4
    assert res["stress_instrument"].startswith("granular2d.stress_arrays")
    assert np.isfinite(res["mu_eff"])