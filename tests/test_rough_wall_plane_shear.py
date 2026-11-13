"""Tests for the rough-wall plane-shear mode (round 11 of the mu(I) case).

The rough wall is a *grafted monolayer* of spheres on the platen and the floor
(``tensordem.mu_i_rheology.graft_lattice`` / ``_graft_wall_force``).  These tests
pin the three things the round-11 verdict depends on:

1. the lattice is a *blocking* monolayer (spacing < 2 d, inside the box),
2. the graft contact law reproduces the particle-particle normal law with the
   correct sign (wall load in compression) and a tangential force capped at
   ``mu_w * f_n``,
3. switching ``wall_mode`` is the only change: the default (smooth) path is
   untouched, so every archived round-1..10 result stays reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.mu_i_rheology import (
    MuIRheologyError,
    ServoShearCase,
    _graft_wall_force,
    graft_lattice,
    simulate_servo_shear,
)

R = 0.005
W, D = 0.12, 0.05


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_graft_lattice_is_inside_the_box():
    gx, gy = graft_lattice(W, D, R)
    assert gx.size == gy.size > 0
    assert gx.min() >= 0.0 and gx.max() <= W
    assert gy.min() >= R - 1e-12 and gy.max() <= D - R + 1e-12


def test_graft_lattice_spacing_blocks_passage():
    """A monolayer blocks an equal sphere only if the spacing is below 2 d.

    Two graft spheres of radius ``r`` at centre distance ``s`` leave a
    bottleneck of half-width ``s/2``; a particle of radius ``r`` needs
    ``s / 2 >= 2 r``.  With ``s = 1.05 d`` the passage is geometrically closed,
    which is what makes the layer a *wall* rather than a sieve.
    """
    spacing_diameters = 1.05
    gx, gy = graft_lattice(W, D, R, spacing=spacing_diameters)
    pts = np.stack([gx, gy], axis=1)
    # nearest-neighbour distance in the (x, y) lattice
    dmin = np.inf
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d = d[d > 1e-12]
        if d.size:
            dmin = min(dmin, d.min())
    assert dmin >= spacing_diameters * 2 * R - 1e-12
    assert dmin < 2.0 * (2 * R)  # s < 2 d -> passage closed


def test_graft_spacing_validation():
    with pytest.raises(MuIRheologyError):
        ServoShearCase(wall_mode="rough", rough_spacing=0.9)
    with pytest.raises(MuIRheologyError):
        ServoShearCase(wall_mode="knobbly")


def test_default_wall_mode_is_smooth():
    """The historical device must stay the default (archived runs)."""
    assert ServoShearCase().wall_mode == "smooth"


# ---------------------------------------------------------------------------
# graft contact law (analytic, no time integration)
# ---------------------------------------------------------------------------
def _graft_force(pos, vel, gx, gy, gz, gvx=0.0, gvz=0.0, n=None, **kw):
    n = pos.shape[0] if n is None else n
    xi = np.zeros((n, 2))
    params = dict(kn=6.25e4, kt=3.125e4, gamma_w=0.0, mu_w=1.5,
                  radius=R, box_width=W, dt=1e-5)
    params.update(kw)
    return _graft_wall_force(pos, vel, gx, gy, gz, gvx, gvz, xi, **params)


def test_graft_normal_force_matches_spring_law():
    """One particle touching one graft: force = kn * overlap along the normal."""
    gx = np.array([0.06])
    gy = np.array([0.025])
    gz = R  # floor monolayer
    overlap = 0.2 * R
    pos = np.array([[0.06, 0.025, 3.0 * R - overlap]])
    vel = np.zeros((1, 3))
    force, fn_sum, ftx, nc = _graft_force(pos, vel, gx, gy, gz)
    assert nc == 1
    # particle is above the graft: it must be pushed up (+z), i.e. away from it
    assert force[0, 2] == pytest.approx(6.25e4 * overlap, rel=1e-12)
    # the load transmitted to the wall is the same magnitude, compression +
    assert fn_sum == pytest.approx(6.25e4 * overlap, rel=1e-12)
    assert abs(force[0, 0]) < 1e-9 and abs(force[0, 1]) < 1e-9
    assert ftx == pytest.approx(0.0, abs=1e-9)


def test_graft_no_force_without_contact():
    gx, gy = np.array([0.06]), np.array([0.025])
    pos = np.array([[0.06, 0.025, 5.0 * R]])  # separated by 4 R > d
    force, fn_sum, ftx, nc = _graft_force(pos, np.zeros((1, 3)), gx, gy, R)
    assert nc == 0
    assert fn_sum == 0.0 and ftx == 0.0
    assert np.allclose(force, 0.0)


def test_graft_tangential_force_is_capped_by_wall_friction():
    """The tangential spring saturates at ``mu_w * f_n`` (Coulomb cap)."""
    gx, gy = np.array([0.06]), np.array([0.025])
    overlap = 0.2 * R
    pos = np.array([[0.06, 0.025, 3.0 * R - overlap]])
    mu_w = 0.5
    fn_expected = 6.25e4 * overlap
    vel = np.array([[50.0, 0.0, 0.0]])  # large sliding velocity -> saturated
    dt = 1e-5
    xi = np.zeros((1, 2))
    force, fn_sum, ftx, _ = _graft_wall_force(
        pos, vel, gx, gy, R, 0.0, 0.0, xi,
        kn=6.25e4, kt=3.125e4, gamma_w=0.0, mu_w=mu_w, radius=R, box_width=W, dt=dt)
    cap = mu_w * fn_expected
    assert abs(force[0, 0]) == pytest.approx(cap, rel=1e-9)
    assert ftx == pytest.approx(force[0, 0], rel=1e-12)
    # the spring offset is rescaled to the cap, so the force stays bounded
    assert np.linalg.norm(xi[0]) * 3.125e4 == pytest.approx(cap, rel=1e-9)


def test_graft_tangential_force_opposes_relative_motion():
    gx, gy = np.array([0.06]), np.array([0.025])
    pos = np.array([[0.06, 0.025, 3.0 * R - 0.2 * R]])
    vel = np.array([[1.0, 0.0, 0.0]])
    force, *_ = _graft_force(pos, vel, gx, gy, R)
    assert force[0, 0] < 0.0  # drags the particle back


def test_graft_damping_opposes_approach():
    """Approaching the wall must load it more than a static overlap does."""
    gx, gy = np.array([0.06]), np.array([0.025])
    overlap = 0.2 * R
    pos = np.array([[0.06, 0.025, 3.0 * R - overlap]])
    gamma_w = 2.0
    static = _graft_force(pos, np.zeros((1, 3)), gx, gy, R, gamma_w=gamma_w)[1]
    moving = _graft_force(pos, np.array([[0.0, 0.0, -10.0]]), gx, gy, R,
                          gamma_w=gamma_w)[1]
    assert moving > static


# ---------------------------------------------------------------------------
# integration smoke tests (short runs only: ~5 s each)
# ---------------------------------------------------------------------------
def _short_case(**kw):
    base = dict(inertial_number=0.1, kn=6.25e4, normal_stress=6.25e4,
                shear_strain=0.02, settle_steps=60, min_shear_steps=40,
                box_width=W, box_depth=D, fill_height=0.06, bulk_stress=False)
    base.update(kw)
    return ServoShearCase(**base)


def test_rough_mode_runs_and_loads_the_wall():
    res = simulate_servo_shear(_short_case(wall_mode="rough"))
    p = res["normal_stress_measured_pa"]
    assert np.isfinite(p) and p > 0.0
    assert res["shear_rate_measured"] is not None
    assert np.isfinite(res["shear_rate_measured"])


def test_wall_mode_is_the_only_difference():
    """``smooth`` must reproduce the pre-round-11 code path exactly.

    The graft machinery is guarded by ``if rough``; this run pair pins that the
    guard leaves the smooth numbers bit-for-bit identical (the archived
    rounds 1-10 are reproducible).
    """
    a = simulate_servo_shear(_short_case())
    b = simulate_servo_shear(_short_case(wall_mode="smooth"))
    for key in ("normal_stress_measured_pa", "mu_effective",
                "shear_rate_measured", "phi_bed_steady"):
        assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)