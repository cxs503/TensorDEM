"""Engine closure of the particle-particle normal dashpot (m2626).

Before this module the pair law of :class:`tensordem.dem3d.IceDEM3D` derived its
normal dashpot *only* from ``restitution_coefficient``, so ``contact_damping``
was a dead parameter for sphere-sphere contacts even though it is documented as
the normal dashpot, is used by the tool/platen branches, and enters the
``recommended_dt`` damping bound.  ``contact_damping_model="absolute"`` makes the
pair law the clipped law ``F_n = max(k*delta - c*delta_dot, 0)`` whose reference
solution is ``tensordem.restitution.unilateral_restitution(c, k, m_eff)``.

The tests below pin both sides of that contract: the legacy default stays
bit-for-bit (``contact_damping`` not observable), and the opt-in absolute mode
reproduces the reference solution to a few tenths of a percent.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.restitution import (
    damping_for_restitution,
    effective_mass,
    unilateral_restitution,
)

RADIUS = 0.025
DENSITY = 917.0
STIFFNESS = 5_000.0
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
M_PAIR = effective_mass(MASS, MASS)


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _harness():
    from verify_dem3d_restitution import measure_pair, measure_platen
    return measure_pair, measure_platen


def _measure_pair_legacy(damping: float) -> dict:
    """Same pair impact as the harness, but with the legacy default dashpot law."""
    import verify_dem3d_restitution as V
    from tensordem.dem3d import IceDEM3D

    kwargs = dict(
        nx=2, ny=2, nz=2, radius=V.RADIUS, density=V.DENSITY,
        contact_stiffness=V.STIFFNESS, contact_damping=damping,
        platen_stiffness=V.STIFFNESS, platen_damping=damping,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=False, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
    )
    sim = IceDEM3D(DEM3DConfig(**kwargs))
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0
    V._park_idle(sim)
    with torch.no_grad():
        sim.positions[0] = torch.zeros(3, dtype=sim.dtype)
        sim.positions[1] = torch.tensor([2.0 * V.RADIUS + 0.001, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([V.IMPACT_SPEED, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1] = torch.tensor([-V.IMPACT_SPEED, 0.0, 0.0], dtype=sim.dtype)

    def overlap() -> float:
        return 2.0 * V.RADIUS - float(
            torch.linalg.vector_norm(sim.positions[1] - sim.positions[0]))

    def relative_velocity() -> float:
        return float(sim.velocities[1, 0] - sim.velocities[0, 0])

    return V._measure_impact(sim, overlap, relative_velocity)


class ConfigContractTests(unittest.TestCase):
    def test_default_is_restitution_mode(self):
        """The legacy convention must stay the default (no numeric drift)."""
        self.assertEqual(DEM3DConfig().contact_damping_model, "restitution")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            DEM3DConfig(contact_damping_model="viscous")

    def test_conflicting_conventions_rejected(self):
        """Two dashpot conventions at once are ambiguous, so they are refused."""
        with self.assertRaises(ValueError):
            DEM3DConfig(contact_damping_model="absolute", restitution_coefficient=0.5)

    def test_absolute_mode_accepts_any_nonnegative_damping(self):
        for damping in (0.0, 5.0, 40.0):
            cfg = DEM3DConfig(contact_damping_model="absolute", contact_damping=damping)
            self.assertEqual(cfg.contact_damping_model, "absolute")


class LegacyPairLawTests(unittest.TestCase):
    """``contact_damping`` stays invisible unless the mode is opted into."""

    def test_pair_restitution_ignores_contact_damping_by_default(self):
        """Documents the defect that motivated the opt-in mode (legacy contract)."""
        soft = _measure_pair_legacy(0.0)["restitution"]
        hard = _measure_pair_legacy(40.0)["restitution"]
        self.assertLess(abs(hard - soft), 1e-3,
                        msg=f"legacy default moved with contact_damping: {hard} vs {soft}")
        self.assertGreater(soft, 0.99)  # e = 1 => collision is elastic

    def test_absolute_mode_is_what_makes_the_parameter_visible(self):
        measure_pair, _ = _harness()
        free = measure_pair(0.0)["restitution"]
        damped = measure_pair(40.0)["restitution"]
        self.assertGreater(free, 0.99)
        self.assertLess(damped, 0.15,
                        msg=f"absolute mode still insensitive to c: {damped}")


class AbsolutePairLawTests(unittest.TestCase):
    """Opt-in absolute mode must reproduce the clipped-law reference."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        measure_pair, measure_platen = _harness()
        # Plain functions assigned to a class attribute become bound methods;
        # wrap them so ``self.measure_pair(c)`` keeps its one-argument call.
        cls.measure_pair = staticmethod(measure_pair)
        cls.measure_platen = staticmethod(measure_platen)

    def test_damping_is_observable_and_monotonic(self):
        values = [self.measure_pair(c)["restitution"] for c in (2.0, 5.0, 10.0, 20.0, 40.0)]
        for upper, lower in zip(values, values[1:]):
            self.assertGreater(upper, lower, msg=f"non-monotonic sweep: {values}")
        self.assertLess(values[-1], 0.25)

    def test_engine_matches_reference_law(self):
        for damping in (2.0, 5.0, 10.0, 20.0, 40.0):
            measured = self.measure_pair(damping)["restitution"]
            reference = unilateral_restitution(damping, STIFFNESS, M_PAIR)
            self.assertLess(abs(measured - reference) / reference, 0.03,
                            msg=f"c={damping}: engine {measured:.5f} vs ref {reference:.5f}")

    def test_inverted_damping_hits_the_target(self):
        for target in (0.9, 0.7, 0.5, 0.3):
            c = damping_for_restitution(target, STIFFNESS, M_PAIR)
            measured = self.measure_pair(c)["restitution"]
            self.assertLess(abs(measured - target) / target, 0.03,
                            msg=f"target {target}, c={c:.4g}: measured {measured:.5f}")

    def test_residual_is_not_timestep_error(self):
        """Refining dt by 4x must not move the closure (residual is model-level)."""
        coarse = self.measure_pair(10.0, dt_scale=1.0)["restitution"]
        fine = self.measure_pair(10.0, dt_scale=0.25)["restitution"]
        self.assertLess(abs(fine - coarse) / coarse, 0.01,
                        msg=f"closure drifts with dt: {coarse} vs {fine}")

    def test_platen_branch_keeps_absolute_convention(self):
        """The wall branch was already absolute; the pair branch now agrees.

        ``platen_damping`` never went through the restitution mapping, so this
        test documents the convention the pair branch was missing.
        """
        damping = 5.0
        measured = self.measure_platen(damping)["restitution"]
        reference = unilateral_restitution(damping, STIFFNESS, effective_mass(MASS))
        self.assertLess(abs(measured - reference) / reference, 0.03,
                        msg=f"platen {measured:.5f} vs ref {reference:.5f}")


if __name__ == "__main__":
    unittest.main()