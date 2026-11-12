"""Analytic validation of the slab Love-Weber stress estimator (round 6).

The device-level question ("is the wall traction mu the bulk mu?") is answered
in ``benchmarks/plane_shear_mu_i/bulk_stress_rheometry.py``; these tests check
that the estimator itself is exactly the textbook sum on configurations whose
stress is known in closed form, so a device-level disagreement cannot be blamed
on the estimator.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.granular3d_core import Material3D, ContactState, contact_forces
from tensordem.mu_i_rheology import bulk_virial_stress, shear_rate_profile


def _pairs(iu, ju, f_on_i, branch):
    iu = np.asarray(iu, dtype=np.int64)
    ju = np.asarray(ju, dtype=np.int64)
    return {"iu": iu, "ju": ju,
            "force_on_i": np.asarray(f_on_i, dtype=float).reshape(-1, 3),
            "branch": np.asarray(branch, dtype=float).reshape(-1, 3),
            "f_n": np.zeros(len(iu)), "f_t": np.zeros((len(iu), 3))}


def test_axial_doublet_gives_exact_sigma_zz():
    """Two particles pressed together along z: sigma_zz = f r_z / V exactly."""
    pos = np.array([[0.01, 0.01, 0.010], [0.01, 0.01, 0.018]])
    vel = np.zeros((2, 3))
    f = 125.0                      # = kn*overlap for kn=6.25e4, overlap=2 mm
    branch = np.array([[0.0, 0.0, 0.008]])
    pairs = _pairs([0], [1], [[0.0, 0.0, -f]], branch)
    out = bulk_virial_stress(pos, vel, pairs, z_lo=0.0, z_hi=0.05,
                             y_lo=0.0, y_hi=0.05, box_width=0.05, n=2,
                             mass=0.0)
    volume = 0.05 * 0.05 * 0.05
    assert out["n_contacts"] == 1
    assert out["sigma_zz"] == pytest.approx(-f * 0.008 / volume, rel=1e-12)
    assert out["sigma_xx"] == pytest.approx(0.0, abs=1e-12)
    assert out["pressure"] == pytest.approx(f * 0.008 / volume / 3.0, rel=1e-12)
    assert out["mu"] == pytest.approx(0.0, abs=1e-12)   # no shear component


def test_kinetic_streaming_term_is_mass_times_velocity_variance():
    """No contacts, +/- vx: sigma_xx = m sum v'^2 / V (pressure = trace/3)."""
    pos = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    pairs = _pairs([], [], np.zeros((0, 3)), np.zeros((0, 3)))
    out = bulk_virial_stress(pos, vel, pairs, z_lo=0.0, z_hi=0.05,
                             y_lo=0.0, y_hi=0.05, box_width=0.05, n=2,
                             mass=1.0)
    volume = 0.05 * 0.05 * 0.05
    assert out["n_contacts"] == 0
    assert out["sigma_xx"] == pytest.approx(-2.0 / volume, rel=1e-12)
    assert out["kinetic_pressure"] == pytest.approx(2.0 / volume / 3.0, rel=1e-12)
    assert out["pressure"] == pytest.approx(2.0 / volume / 3.0, rel=1e-12)
    assert out["contact_pressure"] == pytest.approx(0.0, abs=1e-12)


def test_contacts_crossing_the_slab_boundary_are_not_counted():
    """A contact with one endpoint outside the slab must not enter the sum."""
    pos = np.array([[0.01, 0.01, 0.045], [0.01, 0.01, 0.055]])
    vel = np.zeros((2, 3))
    pairs = _pairs([0], [1], [[0.0, 0.0, -100.0]], [[0.0, 0.0, 0.01]])
    out = bulk_virial_stress(pos, vel, pairs, z_lo=0.0, z_hi=0.048,
                             y_lo=0.0, y_hi=0.05, box_width=0.05, n=2,
                             mass=0.0)
    import math
    assert out["n_contacts"] == 0
    assert out["n_particles"] == 1
    assert math.isnan(out["sigma_zz"])   # <2 particles: no slab stress


def test_shear_rate_profile_recovers_an_imposed_linear_profile():
    n = 400
    # one particle per bin centre (plus replicates) so the binned profile is an
    # exact sample of the linear field
    edges = np.linspace(0.0, 0.05, 7)
    centres = 0.5 * (edges[:-1] + edges[1:])
    z = np.repeat(centres, 60) + np.tile(np.linspace(-1e-4, 1e-4, 60), 6)
    x = np.zeros_like(z)
    y = np.full_like(z, 0.025)
    pos = np.column_stack([x, y, z])
    vel = np.column_stack([2.5 * z, np.zeros(n), np.zeros(n)])
    prof = shear_rate_profile(pos, vel, z_lo=0.0, z_hi=0.05, y_lo=0.0,
                              y_hi=0.05, n_bins=6)
    assert prof["shear_rate"] == pytest.approx(2.5, rel=1e-10)
    assert prof["r2"] == pytest.approx(1.0, abs=1e-10)


def test_contact_forces_pair_payload_is_consistent_with_the_solver():
    """return_pairs must expose the per-pair force, not a per-particle sum."""
    mat = Material3D(radius=0.005, density=2500.0, mu=0.0, mu_rolling=0.0,
                     restitution=0.0, kn=6.25e4, kt_ratio=0.5, gravity=0.0)
    pos = np.array([[0.01, 0.01, 0.010], [0.01, 0.01, 0.018]])
    vel = np.zeros((2, 3))
    omega = np.zeros((2, 3))
    state = ContactState()
    f, t, pairs = contact_forces(pos, vel, omega, np.array([0]),
                                 np.array([1]), state, mat, 1e-5,
                                 return_pairs=True)
    overlap = 0.010 - 0.008
    assert pairs["force_on_i"].shape == (1, 3)
    assert pairs["force_on_i"][0, 2] == pytest.approx(-6.25e4 * overlap, rel=1e-12)
    assert pairs["branch"][0, 2] == pytest.approx(0.008, rel=1e-12)
    # the per-particle force array is the negation of the pair force for i
    assert f[0, 2] == pytest.approx(pairs["force_on_i"][0, 2], rel=1e-12)
    assert f[1, 2] == pytest.approx(-pairs["force_on_i"][0, 2], rel=1e-12)
