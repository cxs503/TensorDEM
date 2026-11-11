"""H1-BROKEN-ICE-DEM, round 2: regression tests for the three kernel defects.

Round 1 shipped a runnable case but the energy ledger did not close and the
rubble pile never came to rest.  Three defects in the shared kernel
(``polyhedron_floe3d.dynamics``) explain both symptoms, and each one gets a
minimal, self-contained regression test here -- no case setup, no thresholds
borrowed from the case:

1. the buoyancy work was booked as a **power** (``F . v``) instead of work
   (missing ``dt``), so ``energy_residual`` of a freely floating floe read 212
   instead of ~1e-5;
2. the same unit slip existed on the moving-wall work term;
3. the pair dashpot had the **wrong sign** (``+ c v_n`` with ``normal`` running
   from a to b), i.e. negative damping: a confined, constant-speed drive fed
   energy in on every separating step and a rubble pile never came to rest.

The assertions are on invariants (an energy that may not rise, a driven body
whose work term has the right magnitude) rather than on the case numbers, so
they still hold if the case parameters change.
"""
from __future__ import annotations

import math

import numpy as np

from tensordem.polyhedron_floe3d.contact import ContactLaw
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody
from tensordem.polyhedron_floe3d.geometry import equivalent_disc_prism


def _prism(n_sides: int = 8, area: float = 0.04, thickness: float = 0.10):
    return equivalent_disc_prism(area * thickness, thickness, n_sides)


def _mechanical_energy(system: DEMSystem3D) -> float:
    return system.energy()


def test_pair_dashpot_is_dissipative_not_amplifying():
    """Two prisms colliding head-on: total mechanical energy may not rise.

    With ``f_n = k_n d + c_n v_n`` (wrong sign) the dashpot pushes the bodies
    *apart faster* while they separate, so the energy of this isolated pair grows
    by tens of percent.  A dissipative dashpot can only remove energy.
    """
    shape_a = _prism()
    shape_b = _prism()
    a = FloeBody(shape_a, 917.0, position=[-0.16, 0.0, 0.0], label="a")
    b = FloeBody(shape_b, 917.0, position=[0.16, 0.0, 0.0], label="b")
    a.velocity = np.array([0.5, 0.0, 0.0])
    b.velocity = np.array([-0.5, 0.0, 0.0])
    system = DEMSystem3D([a, b], law=ContactLaw(kn=2.0e5, zeta=0.3, mu=0.0,
                                                mu_rolling=0.0, kt_ratio=0.25),
                         gravity=0.0, water_level=None, dt=1.0e-4, record=False)
    e0 = _mechanical_energy(system)
    assert e0 > 0.0
    hit = False
    peak = e0
    for _ in range(3000):
        _ = system.step()
        peak = max(peak, _mechanical_energy(system))
        hit = hit or system.n_pair_contacts > 0
    assert hit, "the two bodies never touched; the test would prove nothing"
    assert peak <= e0 * (1.0 + 1.0e-3), (
        "pair dashpot injected energy: E0=%.6g peak=%.6g" % (e0, peak))
    assert _mechanical_energy(system) < e0, "collision must dissipate, not conserve"


def test_buoyancy_work_is_work_not_power():
    """A floe relaxing to its draft must close its energy ledger.

    Round 1 read ``energy_residual`` = 212 for this very case because the
    buoyancy work term was accumulated without ``dt``; the physical drift here
    is O(1e-5) of ``m g R``.
    """
    shape = _prism(n_sides=16, area=0.36, thickness=0.12)
    body = FloeBody(shape, 917.0, position=[0.0, 0.0, 0.0], label="floe")
    mass = body.mass
    omega0 = math.sqrt(1025.0 * 9.81 * 0.36 / mass)
    system = DEMSystem3D([body], law=ContactLaw(), gravity=9.81, water_level=0.0,
                         water_density=1025.0, dt=5.0e-4,
                         linear_drag=0.7 * 2.0 * mass * omega0,
                         angular_drag=0.21 * 2.0 * mass * omega0, record=False)
    system.record_initial_energy()
    for _ in range(3000):
        _ = system.step()
    assert abs(system.work_external) < 100.0 * system.energy_scale(), (
        "buoyancy work term is a power, not work: W=%.3g J, scale=%.3g J"
        % (system.work_external, system.energy_scale()))
    assert system.energy_residual() < 1.0e-2, (
        "energy ledger does not close: residual=%.3g" % system.energy_residual())


def test_moving_wall_work_term_is_per_step_work():
    """A wall advancing at constant speed into a body: |W| must be Jo(1) J.

    The work done over ``n`` steps is bounded by ``F_max * v_wall * t``; a term
    that is missing ``dt`` is ~1/dt = 5000x larger.
    """
    shape = _prism(n_sides=8, area=0.04, thickness=0.10)
    body = FloeBody(shape, 917.0, position=[0.0, 0.0, 0.0], label="floe")
    system = DEMSystem3D([body], law=ContactLaw(kn=2.0e5, zeta=0.2), gravity=0.0,
                         water_level=None, dt=1.0e-4, record=False)
    system.add_wall([1.0, 0.0, 0.0], -0.02, velocity=[0.2, 0.0, 0.0], label="hull")
    speed, n_steps = 0.2, 500
    t = n_steps * system.dt
    force_max = 0.0
    for _ in range(n_steps):
        system.walls[0].offset += speed * system.dt
        _ = system.step()
        force_max = max(force_max, float(np.linalg.norm(
            system.wall_force_by_label.get("hull", np.zeros(3)))))
    bound = max(force_max, 1.0) * speed * t * 10.0
    assert abs(system.work_external) <= bound, (
        "wall work term out of bounds: W=%.4g J, bound=%.4g J" %
        (system.work_external, bound))
    assert abs(system.work_external) > 0.0, "the wall did work; a zero ledger is wrong"