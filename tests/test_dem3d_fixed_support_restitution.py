"""Fixed-support pairs must use the rigid-partner effective mass (``m``).

The pair dashpot is derived from ``restitution_coefficient`` through
``zeta = |ln e|/sqrt(pi^2+ln^2 e)`` and ``gamma = 2 zeta sqrt(k m_eff)``.  For a
pair of *mobile* equal spheres ``m_eff = m/2``; for a pair with one **pinned**
particle (hard support: velocity zeroed and position reset every step) the
partner is infinitely massive, so ``m_eff = m``
(``tensordem.restitution.effective_mass(m, None)``, the same rigid-counterpart
convention the tool/platen branches use).  Using ``m/2`` for a pinned pair
under-damps by ``sqrt(2)`` and the realised restitution overshoots ``e_target``
by 29 % (linear) / 35 % (Hertz) at ``e_target = 0.5``.

See ``docs/dem3d_fixed_support_reduced_mass.md``.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.restitution import effective_mass

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_dem3d_fixed_support_restitution import (  # noqa: E402
    build_impact,
    measure,
    reference_rk4,
)
from verify_dem3d_hertz_restitution import DENSITY, RADIUS  # noqa: E402

MASS = DEM3DConfig(nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY).mass


def _shoot(*, pinned: bool, e_target: float, model: str, speed: float) -> float:
    sim, closing = build_impact(speed, pinned=pinned, e_target=e_target, model=model)
    return measure(sim, closing)["restitution"]


class FixedSupportReducedMassTests(unittest.TestCase):
    def test_library_convention_is_m_for_a_rigid_partner(self) -> None:
        # The mass convention asserted here is the library's own contract.
        self.assertAlmostEqual(effective_mass(MASS, None) / MASS, 1.0, places=15)
        self.assertAlmostEqual(effective_mass(MASS, MASS) / MASS, 0.5, places=15)

    def test_pinned_impact_matches_the_free_free_realisation(self) -> None:
        """Same target, same law, only the support differs -> same realised e."""
        for model in ("linear", "hertz"):
            for e_target in (0.5, 0.8):
                with self.subTest(model=model, e_target=e_target):
                    pinned = _shoot(pinned=True, e_target=e_target, model=model, speed=0.5)
                    free = _shoot(pinned=False, e_target=e_target, model=model, speed=0.5)
                    self.assertLess(abs(pinned / free - 1.0), 0.01,
                                    f"pinned={pinned:.4f} free={free:.4f}")

    def test_pinned_impact_matches_independent_rk4_with_full_mass(self) -> None:
        for model in ("linear", "hertz"):
            with self.subTest(model=model):
                speed = 0.5
                pinned = _shoot(pinned=True, e_target=0.5, model=model, speed=speed)
                ref = reference_rk4(e_target=0.5, model=model, m_eff=MASS,
                                    closing_speed=speed)["restitution"]
                self.assertLess(abs(pinned / ref - 1.0), 0.01,
                                f"{model}: engine={pinned:.4f} rk4={ref:.4f}")

    def test_negative_control_half_mass_reference_is_rejected(self) -> None:
        """The pre-fix convention (m/2) predicts a visibly different restitution."""
        speed = 0.5
        # Pre-fix engine: pinned-pair dynamics with m, damping built from m/2.
        wrong = reference_rk4(e_target=0.5, model="linear", m_eff=MASS,
                              damping_m_eff=MASS / 2.0,
                              closing_speed=speed)["restitution"]
        pinned = _shoot(pinned=True, e_target=0.5, model="linear", speed=speed)
        self.assertGreater(abs(pinned / wrong - 1.0), 0.10,
                           f"engine={pinned:.4f} m/2 reference={wrong:.4f}")

    def test_mixed_topology_keeps_m_half_for_mobile_pairs(self) -> None:
        """A pinned *neighbour* must not change the mobile pair's m/2 convention."""
        e_target = 0.5
        sim, closing = build_impact(0.5, pinned=False, e_target=e_target, model="linear")
        # Pin every particle except the impacting pair, so the pair itself stays
        # fully mobile while the pair list contains pinned members.
        others = torch.arange(len(sim.positions)) > 1
        sim.fixed[others] = True
        sim.initial_positions[others] = sim.positions[others].clone()
        sim.velocities[others] = 0.0
        measured = measure(sim, closing)["restitution"]
        reference = _shoot(pinned=False, e_target=e_target, model="linear", speed=0.5)
        self.assertLess(abs(measured / reference - 1.0), 1e-12,
                        f"mixed={measured:.6f} pure_free={reference:.6f}")

    def test_elastic_pinned_impact_loses_no_energy(self) -> None:
        """e = 1 keeps gamma_n = 0, so the mass convention cannot leak in."""
        sim, closing = build_impact(0.5, pinned=True, e_target=1.0, model="linear")
        measured = measure(sim, closing)["restitution"]
        self.assertLess(abs(measured - 1.0), 1e-4)


if __name__ == "__main__":
    unittest.main()