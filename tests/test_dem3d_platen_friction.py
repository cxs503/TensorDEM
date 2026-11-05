"""Analytic verification of Coulomb friction at the DEM3D platen boundaries.

The three anchors are exact for the *discrete* solver (closed forms in
``tensordem.platen_friction``), so agreement cannot be a small-dt artefact:

* settled normal load: ``N = F_n`` and ``delta_z = -/+ F_n / k_p``
* stick branch:        ``delta_x = F_h / k_t``   (Cundall-Strack static solution)
* slip branch:         ``a = (F_h - mu_p N) / m`` (saturated, drag-free)

Controls keep the anchors from collapsing into identities: ``mu_p = 0`` must
destroy the stick solution (no equilibrium exists), the Coulomb onset must
bracket, and a probe call (``update_history=False``) must not advance the
friction history.
"""
from __future__ import annotations

import functools
import math
import os
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.platen_friction import (
    ACCURACY_GATE,
    platen_friction_case,
    platen_friction_limit,
    platen_slip_acceleration,
    platen_static_offset,
)

#: Cheap settle budget for the unit tests; the benchmark script uses more steps.
FAST = dict(settle_steps=8000, steps=1800, ramp_steps=600)
SETTLE_TOL = 1.0e-6

#: The anchor cases settle the normal load to 1e-8 m/s (~30 s each).  They are the
#: content of ``benchmarks/platen_friction_verification.py``; set TENSORDEM_SLOW=1 to
#: run them in the unit-test selection as well.
SLOW = unittest.skipUnless(os.environ.get("TENSORDEM_SLOW"),
                          "heavy settle; run benchmarks/platen_friction_verification.py or set TENSORDEM_SLOW=1")


@functools.lru_cache(maxsize=None)
def _cached(params: tuple) -> dict:
    return platen_friction_case(**dict(params))


def _case(**kwargs) -> dict:
    params = dict(FAST)
    params.update(kwargs)
    return _cached(tuple(sorted(params.items())))


class PlatenFrictionAnchorTests(unittest.TestCase):
    @SLOW
    def test_bottom_stick_offset_matches_cundall_strack(self):
        case = _case(phase="stick", mu_p=0.3, normal_force=10.0, tangential_force=1.8)
        self.assertEqual(case["anchor"], "tangential_offset_m")
        self.assertLess(case["relative_error"]["tangential_offset_m"], ACCURACY_GATE)
        self.assertLess(case["relative_error"]["sink"], ACCURACY_GATE)
        # the stick branch must not have touched the Coulomb limit
        self.assertLess(case["measured"]["peak_friction_ratio"], 1.0)
        self.assertLess(abs(case["measured"]["final_speed_m_per_s"]), 1.0e-2)

    @SLOW
    def test_settled_normal_load_is_the_applied_force(self):
        case = _case(phase="stick", mu_p=0.3, normal_force=10.0, tangential_force=1.8)
        normal = case["measured"]["plane_reaction_N"] / 4.0
        self.assertLess(abs(normal - 10.0) / 10.0, ACCURACY_GATE)
        # the closed forms use physics, not the measured reaction: they agree anyway
        self.assertAlmostEqual(case["normal_force_N"], 10.0, delta=0.1)

    @SLOW
    def test_top_platen_carries_the_mirror_image(self):
        case = _case(phase="stick", mu_p=0.5, normal_force=12.0,
                     tangential_force=5.0, platen="top")
        self.assertGreater(case["measured"]["sink_m"], 0.0)  # pushed up, not down
        self.assertLess(case["relative_error"]["tangential_offset_m"], ACCURACY_GATE)
        self.assertLess(case["relative_error"]["sink"], ACCURACY_GATE)

    @SLOW
    def test_slip_acceleration_matches_newtonian_closed_form(self):
        case = _case(phase="slip", mu_p=0.3, normal_force=10.0, tangential_force=6.0,
                     drag=0.0, ramp_steps=50, steps=900)
        self.assertEqual(case["anchor"], "velocity_slope_m_per_s2")
        self.assertLess(case["relative_error"]["velocity_slope_m_per_s2"], ACCURACY_GATE)
        # sliding is the saturated branch: |F_t| equals mu_p*N at every sliding step
        self.assertGreater(case["measured"]["peak_friction_ratio"], 0.99)
        self.assertLess(case["measured"]["peak_friction_ratio"], 1.0 + 1.0e-3)
        self.assertGreater(case["measured"]["final_speed_m_per_s"], 1.0)

    @SLOW
    def test_stick_and_slip_differ_by_more_than_the_gate(self):
        stick = _case(phase="stick", mu_p=0.3, normal_force=10.0, tangential_force=1.8)
        slip = _case(phase="slip", mu_p=0.3, normal_force=10.0, tangential_force=6.0,
                     drag=0.0, ramp_steps=50, steps=600, platen_damping=5.0)
        self.assertGreater(slip["measured"]["tangential_offset_m"],
                           100.0 * stick["measured"]["tangential_offset_m"])


class PlatenFrictionOnsetTests(unittest.TestCase):
    def test_onset_brackets_the_coulomb_limit(self):
        limit = platen_friction_limit(0.3, 10.0)
        below = _case(phase="stick", mu_p=0.3, normal_force=10.0,
                      tangential_force=0.9 * limit)
        above = _case(phase="slip", mu_p=0.3, normal_force=10.0,
                      tangential_force=1.1 * limit, drag=0.0, ramp_steps=50, steps=600,
                      platen_damping=5.0)
        self.assertLess(abs(below["measured"]["final_speed_m_per_s"]), 1.0e-2)
        self.assertGreater(above["measured"]["final_speed_m_per_s"], 0.1)
        self.assertLess(below["relative_error"]["tangential_offset_m"], ACCURACY_GATE)


class PlatenFrictionControlTests(unittest.TestCase):
    def test_zero_friction_has_no_static_equilibrium(self):
        with_friction = _case(phase="stick", mu_p=0.3, normal_force=10.0,
                              tangential_force=1.8)
        frictionless = _case(phase="stick", mu_p=0.0, normal_force=10.0,
                             tangential_force=1.8)
        predicted = platen_static_offset(1.8, 5000.0)
        self.assertLess(with_friction["relative_error"]["tangential_offset_m"],
                        ACCURACY_GATE)
        # mu_p = 0 removes the restoring force entirely: the raft runs away
        self.assertGreater(frictionless["measured"]["tangential_offset_m"],
                           100.0 * predicted)
        self.assertEqual(frictionless["measured"]["peak_friction_ratio"], 0.0)

    def test_probe_does_not_advance_the_friction_history(self):
        cfg = DEM3DConfig(
            nx=2, ny=2, nz=2, radius=0.025, density=917.0,
            platen_stiffness=5000.0, platen_damping=0.5,
            bottom_platen_enabled=True, bottom_platen_gap=0.0,
            fix_x_edges=False, drag=2.0, platen_friction_coefficient=0.3,
        )
        sim = IceDEM3D(cfg)
        sim.tool_start[:] = torch.tensor([1.0e6] * 3, dtype=sim.dtype)
        sim.tool_velocity.zero_()
        raft = sim.positions[:, 2] <= sim.positions[:, 2].min() + 0.5 * cfg.radius
        sim.fixed |= ~raft
        loads = torch.zeros_like(sim.positions)
        loads[raft, 0] = 1.0
        loads[raft, 2] = -10.0
        for _ in range(50):
            sim.step(loads)
        before = sim._bottom_platen_tang_disp.clone()
        for _ in range(5):
            sim.forces(loads, update_history=False)
            sim.diagnostics(loads)
        self.assertTrue(torch.equal(before, sim._bottom_platen_tang_disp))

    def test_legacy_default_is_frictionless(self):
        cfg = DEM3DConfig(nx=3, ny=3, nz=2, bottom_platen_enabled=True,
                          top_platen_enabled=True, top_platen_velocity=-0.01,
                          platen_damping=0.5)
        self.assertEqual(cfg.platen_friction_coefficient, 0.0)
        sim = IceDEM3D(cfg)
        sim.tool_start[:] = torch.tensor([1.0e6] * 3, dtype=sim.dtype)
        sim.tool_velocity.zero_()
        for _ in range(40):
            sim.step()
        self.assertEqual(float(sim.last_bottom_platen_friction.abs().sum()), 0.0)
        self.assertEqual(float(sim.last_top_platen_friction.abs().sum()), 0.0)
        self.assertEqual(float(sim._bottom_platen_tang_disp.abs().sum()), 0.0)
        self.assertEqual(float(sim._top_platen_tang_disp.abs().sum()), 0.0)

    def test_invalid_friction_parameters_are_rejected(self):
        for kwargs in (dict(platen_friction_coefficient=-0.1),
                       dict(platen_friction_coefficient=math.nan),
                       dict(platen_friction_stiffness=0.0),
                       dict(platen_friction_stiffness=-5.0)):
            with self.assertRaises(ValueError):
                DEM3DConfig(**kwargs)

    def test_tangential_stiffness_changes_the_stick_offset(self):
        stiff = _case(phase="stick", mu_p=0.3, normal_force=10.0,
                      tangential_force=1.8, friction_stiffness=9000.0)
        self.assertLess(stiff["relative_error"]["tangential_offset_m"], ACCURACY_GATE)
        self.assertAlmostEqual(
            stiff["measured"]["tangential_offset_m"] / (1.8 / 9000.0), 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()