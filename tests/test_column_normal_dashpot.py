"""Isolate normal spring storage and dashpot work in each column contact geometry."""
import math

import numpy as np
import pytest

from tensordem.granular3d_core import Material3D
from tensordem.granular_column.kernel import PairSpringState, pair_contact_forces, wall_contact_forces
from tensordem.granular_column.pluviation import cylinder_wall_contact_forces


@pytest.mark.parametrize("geometry", ["pair", "floor", "cylinder"])
@pytest.mark.parametrize("mode", ["one_sided", "reference"])
@pytest.mark.parametrize("speed", [-.3, 0., .05, .3, 3.])
def test_normal_work(geometry, mode, speed, record_property):
    mat = Material3D(radius=.5, density=10., kn=200., restitution=.5, mu=0., mu_rolling=0.)
    overlap, dt = .01, .02
    mass = 10. * 4. * math.pi * .5**3 / 3.
    reduced_mass = mass / 2. if geometry == "pair" else mass
    zeta = math.log(2.) / math.hypot(math.pi, math.log(2.))
    gamma = 2. * zeta * math.sqrt(200. * reduced_mass)
    if geometry == "pair":
        pos = np.array([[0., 0., 0.], [1. - overlap, 0., 0.]])
        vel = np.array([[0., 0., 0.], [speed, 0., 0.]])
        result = pair_contact_forces(pos, vel, np.zeros_like(pos), np.array([0]),
                                     np.array([1]), PairSpringState(), mat, dt, contact_mode=mode)
    elif geometry == "floor":
        pos = np.array([[0., 0., .5 - overlap]])
        vel = np.array([[0., 0., speed]])
        result = wall_contact_forces(pos, vel, np.zeros_like(pos), mat, dt, contact_mode=mode)
    else:
        radial = np.array([[.6, .8, 0.]])
        pos = (.5 + overlap) * radial
        result = cylinder_wall_contact_forces(pos, -speed * radial, np.zeros_like(pos),
                                              mat, dt, contact_mode=mode, wall_radius=1.)
    spring = 200. * overlap
    trial = spring - gamma * speed
    force = max(0., trial) if mode == "reference" else max(spring, trial)
    work = (spring - force) * speed * dt if force > 0. else 0.
    record_property("abs_work_error_J", abs(result["dissip_collision"] - work))
    assert result["normal_force_sum"] == pytest.approx(force, abs=1e-13)
    assert result["energy_normal"] == pytest.approx(.5 * spring * overlap)
    assert result["dissip_collision"] == pytest.approx(work, abs=1e-14)


@pytest.mark.parametrize("mode,active_fraction", [("reference", 1.), ("one_sided", .5)])
def test_cylinder_work_over_a_closed_indentation_cycle(mode, active_fraction, record_property):
    mat = Material3D(radius=.5, density=10., kn=200., restitution=.5, mu=0.)
    period, amplitude, mean_overlap = 1., .005, .05
    frequency = 2. * math.pi / period
    steps = 64
    dt = period / steps
    radial = np.array([[.6, .8, 0.]])
    work = 0.
    for step in range(steps):
        phase = frequency * (step + .5) * dt
        overlap = mean_overlap + amplitude * math.sin(phase)
        velocity = amplitude * frequency * math.cos(phase) * radial
        pos = (.5 + overlap) * radial
        contact = cylinder_wall_contact_forces(pos, velocity, np.zeros_like(pos), mat, dt,
                                               contact_mode=mode, wall_radius=1.)
        assert contact["normal_force_sum"] > 0.
        work += contact["dissip_collision"]
    # The elastic spring returns to its initial energy. The integral of
    # gamma * overlap_rate**2 is half as large for compression-only damping.
    expected = active_fraction * mat.gamma_wall * (amplitude * frequency)**2 * period / 2.
    record_property("relative_cycle_work_error", abs(work / expected - 1.))
    assert work == pytest.approx(expected, rel=1e-13)
