"""Guard the ``contact_damping`` contract for unbonded particle/particle contact.

Regression history: m670 replaced the raw dashpot ``contact_damping`` of the
particle/particle normal contact with a damper derived from
``restitution_coefficient``.  Because the default coefficient is ``1.0``, that
change set the pair damper to exactly zero for every default configuration, so
``contact_damping`` became a silent no-op and free pairs rebounded with
``e ~ 1.0`` whatever the configured dashpot was.  The merged analytic-impact
and engine-closure anchors (``tests/test_dem3d_contact_impact.py``,
``tests/test_restitution.py``) went red at the same time.

These tests pin the restored precedence explicitly:

* ``restitution_coefficient < 1``  -> dashpot inverted from the requested ``e``
  (clipped-law mapping, ``tensordem.restitution``);
* ``restitution_coefficient == 1`` -> the raw ``contact_damping`` dashpot, the
  same convention the tool and platen branches use.

The measured rebound is compared with ``restitution.unilateral_restitution``,
an independently coded RK4 solution of the *same* clipped law, so a pass means
the engine and the reference agree rather than that the test agrees with itself.
"""
import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.restitution import (
    classical_damping,
    damping_for_restitution,
    effective_mass,
    unilateral_restitution,
)

RADIUS = 0.025
DENSITY = 917.0
STIFFNESS = 5_000.0
IMPACT_SPEED = 0.5
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
M_PAIR = effective_mass(MASS, MASS)

# Idle sites must stay close to the impact pair: the contact search sizes its
# uniform cell list over the bounding box of all positions.
_IDLE_SITES = (
    (0.15, 0.15, 0.15), (-0.15, 0.15, 0.15), (0.15, -0.15, 0.15),
    (-0.15, -0.15, 0.15), (0.0, 0.15, 0.30), (0.0, -0.15, 0.30),
)


def _engine(damping: float, restitution_coefficient: float = 1.0) -> IceDEM3D:
    sim = IceDEM3D(DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_stiffness=STIFFNESS, contact_damping=damping,
        platen_stiffness=STIFFNESS, platen_damping=0.0,
        restitution_coefficient=restitution_coefficient,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=False, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
    ))
    # Drop every bond without releasing bond energy: contacts own the response.
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0
    return sim


def measure_pair(damping: float, restitution_coefficient: float = 1.0,
                 max_steps: int = 500_000) -> float:
    """Head-on equal-mass impact inside the engine; returns restitution ``e``."""
    torch.set_num_threads(1)
    sim = _engine(damping, restitution_coefficient)
    n = sim.positions.shape[0]
    with torch.no_grad():
        for idx in range(2, n):
            sim.positions[idx] = torch.tensor(_IDLE_SITES[(idx - 2) % len(_IDLE_SITES)],
                                              dtype=sim.dtype)
            sim.velocities[idx] = torch.zeros(3, dtype=sim.dtype)
        sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([2.0 * RADIUS + 0.001, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([IMPACT_SPEED, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1] = torch.tensor([-IMPACT_SPEED, 0.0, 0.0], dtype=sim.dtype)

    def overlap() -> float:
        return 2.0 * RADIUS - float(
            torch.linalg.vector_norm(sim.positions[1] - sim.positions[0]))

    def relative_velocity() -> float:
        return float(sim.velocities[1, 0] - sim.velocities[0, 0])

    v_in = v_out = None
    for _ in range(max_steps):
        if overlap() > 0.0:
            if v_in is None:
                v_in = relative_velocity()
        elif v_in is not None:
            v_out = relative_velocity()
            break
        sim.step()
    if v_in is None or v_out is None:
        raise RuntimeError("impact never completed")
    return v_out / (-v_in)


class PairDampingContractTests(unittest.TestCase):
    def test_contact_damping_is_not_a_no_op(self):
        """The m670 regression signature: e was 1.0 for any dashpot."""
        elastic = measure_pair(0.0)
        damped = measure_pair(5.0)
        self.assertLess(abs(elastic - 1.0), 0.01, msg=f"dashpot-free e={elastic}")
        self.assertLess(damped, 0.75, msg=f"contact_damping=5 gave e={damped}")

    def test_engine_matches_clipped_law_reference(self):
        """Engine vs an independent RK4 solution of the same clipped law."""
        for damping in (2.0, 5.0, 10.0):
            measured = measure_pair(damping)
            reference = unilateral_restitution(damping, STIFFNESS, M_PAIR)
            rel = abs(measured - reference) / reference
            self.assertLess(rel, 0.03,
                            msg=f"c={damping}: engine {measured:.5f} vs reference "
                                f"{reference:.5f} (rel {rel:.2%})")

    def test_requested_restitution_is_honoured(self):
        """``restitution_coefficient`` overrides the raw dashpot."""
        for target in (0.9, 0.7, 0.5):
            measured = measure_pair(0.0, restitution_coefficient=target)
            rel = abs(measured - target) / target
            self.assertLess(rel, 0.03,
                            msg=f"target e={target}: measured {measured:.5f} "
                                f"(rel {rel:.2%})")

    def test_restitution_branch_ignores_contact_damping(self):
        """Precedence: an explicit ``e`` wins over a contradictory dashpot."""
        from_e = measure_pair(50.0, restitution_coefficient=0.9)
        self.assertGreater(from_e, 0.8, msg=f"e from coefficient was {from_e}")

    def test_clipped_inversion_differs_from_bilateral_mapping(self):
        """Why an absolute dashpot cannot be set with the bilateral formula."""
        c_clip = damping_for_restitution(0.3, STIFFNESS, M_PAIR)
        c_bilateral = classical_damping(0.3, STIFFNESS, M_PAIR)
        self.assertGreater(c_clip / c_bilateral, 1.2)
        # Feeding the bilateral value back into the engine must miss the target,
        # i.e. the two mappings are not interchangeable at low restitution.
        measured = measure_pair(c_bilateral)
        self.assertGreater(measured - 0.3, 0.03)


if __name__ == "__main__":
    unittest.main()