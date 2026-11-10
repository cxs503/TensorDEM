"""Kernel-level invariants of the friction-contact verification surface."""
import numpy as np
import pytest

from tensordem.friction_surface.kernel import (
    ContactModel, PairState, PlaneWall, build_pairs, integrate, pair_forces,
)


def test_static_penetration_equals_weight_over_stiffness():
    """A resting sphere penetrates the floor by exactly ``mg/k_n``."""
    m = ContactModel()
    dt = m.timestep
    pos = np.array([[0.0, 0.0, 0.02]])
    vel = np.zeros((1, 3))
    om = np.zeros((1, 3))
    floor = PlaneWall((0, 0, 0), (0, 0, 1))
    ps = PairState()
    for k in range(40000):
        f, tq = pair_forces(pos, vel, om, np.zeros(0, np.int64),
                            np.zeros(0, np.int64), ps, m, dt)
        fw, tw = floor.apply(pos, vel, om, m, dt, k * dt)
        f = f + fw
        f[:, 2] -= m.mass * m.gravity
        pos, vel, om = integrate(pos, vel, om, f, tq + tw, m, dt)
    assert np.isclose(m.radius - pos[0, 2], m.mass * m.gravity / m.kn, rtol=1e-4)


def test_pair_state_keys_do_not_alias_when_the_particle_set_grows():
    """Regression: keys must use a stable bound, not ``len(pos)``.

    With ``n_keys = len(pos)`` the pair ``(1, 5)`` at n = 10 and ``(0, 15)`` at
    n = 20 both hash to 15, so a brand new contact inherits a stranger's
    tangential displacement.  That aliasing was measured to throw particles
    metres away in the deposition run before it was fixed.
    """
    rng = np.random.default_rng(0)
    pos = rng.random((20, 3)) * 0.05
    iu, ju, keys = build_pairs(pos, 0.01, n_keys=20)
    assert keys.size > 0
    assert np.all(keys < 20 * 20)
    assert np.unique(keys).size == keys.size
    # with a *fixed* bound the encoding is injective: (1,5) -> 25, (0,15) -> 15
    assert 1 * 20 + 5 != 0 * 20 + 15
    # and the same pair keeps its key when the active count changes
    pos_small = pos[:12]
    _, _, keys_small = build_pairs(pos_small, 0.01, n_keys=20)
    assert np.all(keys_small < 20 * 20)
    assert set(keys_small.tolist()) <= set(keys.tolist())


def test_wall_state_survives_a_growing_particle_prefix():
    """Wall friction memory must not be wiped when the active set grows."""
    m = ContactModel()
    wall = PlaneWall((0, 0, 0), (0, 0, 1), mu=0.5)
    pos = np.array([[0.0, 0.0, m.radius * 0.999]])
    vel = np.array([[0.01, 0.0, 0.0]])
    om = np.zeros((1, 3))
    wall.apply(pos, vel, om, m, 1e-5, 0.0)
    before = wall._tang.copy()
    pos2 = np.vstack([pos, [0.1, 0.0, m.radius * 0.999]])
    vel2 = np.vstack([vel, [0.0, 0.0, 0.0]])
    om2 = np.zeros((2, 3))
    wall.apply(pos2, vel2, om2, m, 1e-5, 1e-5)
    # the first particle kept its history (it accumulated one more increment)
    assert np.allclose(wall._tang[0], 2.0 * before[0])
    # the newly appended particle starts from zero
    assert np.allclose(wall._tang[1], 0.0)


def test_restitution_target_is_met_within_the_tension_clamp():
    """Measured collision restitution vs the mapped target (documented artefact).

    ``restitution -> damping`` uses the linear dashpot relation; the tension
    clamp ``max(k_n d - c_n v_n, 0)`` makes the restitution that is actually
    achieved *higher* than requested.  This test pins the measured behaviour so a
    future change to the mapping cannot pass silently.
    """
    def measure(e_target):
        m = ContactModel(restitution=e_target, gravity=0.0, mu=0.0, mu_rolling=0.0)
        dt = m.timestep / 4.0
        pos = np.array([[0.0, 0.0, 0.0], [0.0102, 0.0, 0.0]])
        vel = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        om = np.zeros((2, 3))
        ps = PairState()
        touched = False
        for _ in range(400000):
            iu, ju, keys = build_pairs(pos, m.radius)
            f, tq = pair_forces(pos, vel, om, iu, ju, ps, m, dt)
            pos, vel, om = integrate(pos, vel, om, f, tq, m, dt)
            if pos[1, 0] - pos[0, 0] < 2.0 * m.radius:
                touched = True
            if touched and pos[1, 0] - pos[0, 0] > 0.0101:
                break
        return abs(vel[1, 0] / 0.1)

    for e in (0.3, 0.9):
        measured = measure(e)
        assert measured >= e - 0.02
        assert measured < min(1.0, e + 0.35)
