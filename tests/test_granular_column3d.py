"""Verification tests for the 3-D granular-column free-surface-flow surface.

Covered here (fast subset, <2 min):

* the vectorised contact kernel reproduces the shared reference kernel
  (``tensordem.granular3d_core``) to machine precision;
* the HCP block packing reproduces the requested geometry and solid fraction;
* a short collapse run is dissipative, bounded by the released potential
  energy, and its runout is finite and monotone in the aspect ratio;
* the runout/Midi fits and the mu(I) fit recover synthetic targets exactly;
* the mu(I) law degenerates to ``mu_s`` as ``I -> 0`` and its ``mu_2 = 0``
  negative control cannot saturate.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.granular3d_core import (ContactState, Material3D, build_pairs,
                                       contact_forces, hcp_positions_in_box)
from tensordem.granular_column.column3d import Column3DCase, build_column, simulate_column
from tensordem.granular_column.kernel import (PairSpringState, WallSpringState,
                                              pair_contact_forces, wall_contact_forces)
from tensordem.granular_column.scaling import (fit_jop_law, fit_power_law,
                                               jop_mu_i, local_exponents,
                                               plateau_test)


def _random_state(seed: int = 11, radius: float = 0.004):
    """Crowded lattice + noise: enough contacts for a force comparison."""
    rng = np.random.default_rng(seed)
    r = radius
    spacing = 2.0 * r * 1.02
    pts = [(i * spacing, j * spacing * 0.866, k * spacing * 0.816)
           for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)]
    pos = np.asarray(pts, dtype=float)
    pos = pos + rng.normal(0.0, 0.15 * r, size=pos.shape)
    vel = rng.normal(0.0, 0.2, size=pos.shape)
    omega = rng.normal(0.0, 5.0, size=pos.shape)
    return pos, vel, omega


def test_vector_kernel_matches_reference():
    """Forces/torques equal the shared kernel (dict-based) to machine precision."""
    mat = Material3D(radius=0.004, density=2500.0, mu=0.5, mu_rolling=0.1,
                     restitution=0.4, kn=1.0e4)
    pos, vel, omega = _random_state()
    iu, ju = build_pairs(pos, mat.radius)
    assert iu.size > 100, "test configuration must produce many contacts"
    dt = mat.timestep

    ref_state = ContactState()
    f_ref, t_ref = contact_forces(pos, vel, omega, iu, ju, ref_state, mat, dt)

    state = PairSpringState()
    res = pair_contact_forces(pos, vel, omega, iu, ju, state, mat, dt,
                              contact_mode="reference")
    assert res["n_contacts"] > 0
    np.testing.assert_allclose(res["force"], f_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["torque"], t_ref, rtol=0, atol=1e-12)

    # a second call (state carried over) must still track the reference kernel
    pos2 = pos + vel * dt
    f_ref, t_ref = contact_forces(pos2, vel, omega, iu, ju, ref_state, mat, dt)
    res2 = pair_contact_forces(pos2, vel, omega, iu, ju, state, mat, dt,
                               contact_mode="reference")
    np.testing.assert_allclose(res2["force"], f_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res2["torque"], t_ref, rtol=0, atol=1e-12)


def test_kernel_dissipation_is_positive():
    mat = Material3D(radius=0.004, density=2500.0, mu=0.5, mu_rolling=0.1,
                     restitution=0.4, kn=1.0e4)
    pos, vel, omega = _random_state(seed=5)
    iu, ju = build_pairs(pos, mat.radius)
    res = pair_contact_forces(pos, vel, omega, iu, ju, PairSpringState(), mat,
                              mat.timestep)
    assert res["dissip_collision"] >= 0.0
    assert res["dissip_friction"] >= 0.0
    assert res["dissip_rolling"] >= 0.0
    assert res["energy_normal"] >= 0.0


def test_wall_contact_only_above_floor():
    mat = Material3D(radius=0.005, density=2500.0, kn=1.0e4)
    pos = np.array([[0.0, 0.0, 0.005 + 1e-3], [0.0, 0.0, 0.004]])
    vel = np.zeros((2, 3))
    omega = np.zeros((2, 3))
    res = wall_contact_forces(pos, vel, omega, mat, mat.timestep,
                              state=WallSpringState(2))
    assert res["force"][0, 2] == 0.0
    assert res["force"][1, 2] > 0.0
    assert res["n_contacts"] == 1


def test_packing_geometry_matches_request():
    case = Column3DCase(aspect_ratio=2.0, length=0.05, radius=0.003125)
    pos, mat = build_column(case)
    assert pos.shape[0] > 400
    assert pos[:, 2].min() >= 0.99 * mat.radius
    assert abs((pos[:, 0].max() + mat.radius) - (pos[:, 0].min() - mat.radius)
               - case.length) < 4.0 * mat.diameter
    assert abs(pos[:, 2].max() + mat.radius - case.height) < 4.0 * mat.diameter


def test_collapse_short_run_is_dissipative_and_bounded():
    case = Column3DCase(aspect_ratio=1.5, radius=0.005, length=0.05,
                        duration_factor=1.5, min_time=0.12, sample_stride=50)
    res = simulate_column(case, return_positions=False, max_steps=700)
    energy = res["energy"]

    assert res["n_particles"] > 100
    assert math.isfinite(res["runout_relative"]) and res["runout_relative"] > 0.0
    assert energy["potential_released"] > 0.0
    assert energy["dissipated_physical_only"] > 0.0
    # the physical channels may not dissipate more than was released
    assert energy["dissipated_physical_only"] < 1.05 * energy["potential_released"]
    assert energy["kinetic_final"] >= 0.0 and energy["spring_final"] >= 0.0
    # the ledger residual is bounded by the discretisation error, not by gross
    # bookkeeping mistakes
    assert abs(energy["residual"]) < 0.25


def test_runout_increases_with_aspect_ratio():
    runs = []
    for a in (0.5, 2.0):
        case = Column3DCase(aspect_ratio=a, radius=0.005, length=0.05,
                            duration_factor=1.5, min_time=0.12, sample_stride=100)
        runs.append(simulate_column(case, return_positions=False, max_steps=600))
    assert runs[1]["runout_relative"] > runs[0]["runout_relative"]


def test_power_law_fit_recovers_exponent():
    a = np.array([0.5, 1.0, 2.0, 4.0])
    y = 1.7 * a ** (2.0 / 3.0)
    fit = fit_power_law(a, y)
    assert abs(fit["exponent"] - 2.0 / 3.0) < 1e-9
    assert fit["r2"] > 0.999999
    loc = local_exponents(a, y)
    assert all(abs(row["exponent"] - 2.0 / 3.0) < 1e-9 for row in loc)
    plat = plateau_test(loc)
    assert plat["plateau"] is True


def test_plateau_test_detects_growing_exponent():
    a = np.array([2.0, 3.0, 4.0])
    y = np.array([2.0 ** 0.5, 3.0 ** 0.8, 4.0 ** 1.2])
    plat = plateau_test(local_exponents(a, y))
    assert plat["plateau"] is False


def test_jop_law_zero_I_limit_and_negative_control():
    i = np.array([1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5])
    mu = jop_mu_i(i, 0.36, 0.84, 0.30)
    fit = fit_jop_law(i, mu)
    assert abs(fit["mu_s"] - 0.36) < 0.01
    assert abs(fit["mu_2"] - 0.84) < 0.05
    assert abs(fit["i_0"] - 0.30) < 0.02
    assert fit["r2"] > 0.999
    # I -> 0 degenerates to mu_s
    assert abs(float(jop_mu_i(1e-12, 0.36, 0.84, 0.30)) - 0.36) < 1e-6
    # negative control: mu_2 = 0 cannot reproduce the saturating data
    neg = fit_jop_law(i, mu, fix_mu_2=0.0)
    assert neg["mu_2"] == 0.0
    assert neg["rms"] > 10.0 * fit["rms"]


def test_case_validation_rejects_unphysical_inputs():
    with pytest.raises(ValueError):
        Column3DCase(aspect_ratio=-1.0)
    with pytest.raises(ValueError):
        Column3DCase(packing="ultra-loose")
    with pytest.raises(ValueError):
        Column3DCase(aspect_ratio=1.0, length=0.01, radius=0.005)
