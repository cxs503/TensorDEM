"""Repeated-impact (bounce-chain) closure tests for the 3-D DEM contact law.

The single-impact verifications in ``tests/test_dem3d_restitution.py`` and
``tests/test_dem3d_contact_impact.py`` cannot see accumulated state between
contact episodes.  These tests make the engine hit two rigid planes several
times in a row and check the chain against the per-episode reference solution
of the same clipped law (``tensordem.restitution.unilateral_contact_detail``).

Acceptance criteria (all machine-checked below):

1. per-episode restitution ``|v_out|/|v_in|`` within 3 % of the reference
   clipped-law solution at the *measured* impact speed;
2. chain propagation ``v_n`` within 3 % of ``e**n v_0``;
3. force-window duration within 3 % of the reference contact time;
4. zero reaction while the particle is outside the plane (no dashpot leak);
5. exactly zero velocity change during force-free flight;
6. cumulative dissipation per episode within 3 % of ``(1 - e_ref**2) KE``;
7. halving ``dt`` changes the chain closure by less than 1 % (no dt artefact,
   so the agreement above is not a coincidence of one time step).
"""

from __future__ import annotations

import unittest

import torch

from tensordem.bounce_chain import (
    BounceChainConfig,
    chain_prediction,
    run_bounce_chain,
)


def _cfg(**overrides) -> BounceChainConfig:
    base = dict(
        radius=0.025,
        density=917.0,
        stiffness=5_000.0,
        damping=5.0,
        travel=0.02,          # short free path keeps the test fast
        impact_speed=1.0,
        bounces=3,
        dt_fraction=0.5,
    )
    base.update(overrides)
    return BounceChainConfig(**base)


class BounceChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.result = run_bounce_chain(_cfg())

    # -- 0. the chain actually happened ------------------------------------
    def test_chain_engaged_every_bounce(self):
        res = self.result
        self.assertEqual(len(res.events), 3)
        self.assertEqual([e.plane for e in res.events], ["bottom", "top", "bottom"])
        self.assertTrue(all(e.restitution is not None for e in res.events))
        # every episode in this configuration detaches while still overlapped
        self.assertTrue(res.checks["detached_all_events"])
        self.assertGreater(res.checks["n_closed_events"], 0)

    # -- 1. per-episode law closure ---------------------------------------
    def test_restitution_closure_per_episode(self):
        res = self.result
        for event in res.events:
            ref = event.reference["restitution"]
            rel = abs(event.restitution - ref) / ref
            self.assertLess(rel, 0.03,
                            msg=f"bounce {event.index} on {event.plane}: "
                                f"engine e={event.restitution:.5f} "
                                f"vs reference e={ref:.5f} ({rel:.2%})")

    # -- 2. chain propagation ---------------------------------------------
    def test_chain_propagation(self):
        res = self.result
        e0 = res.events[0].restitution
        predicted = chain_prediction(res.config["impact_speed"], e0, len(res.events))
        for event, expect in zip(res.events, predicted):
            rel = abs(event.rebound_speed - expect) / expect
            self.assertLess(rel, 0.03,
                            msg=f"bounce {event.index}: v_out={event.rebound_speed:.5f} "
                                f"vs e**n v0={expect:.5f} ({rel:.2%})")
        self.assertLess(res.checks["chain_worst_rel_error"], 0.03)

    # -- 3. contact time ---------------------------------------------------
    def test_force_window_duration_matches_reference(self):
        res = self.result
        for event in res.events:
            ref = event.reference["duration"]
            rel = abs(event.force_duration - ref) / ref
            self.assertLess(rel, 0.03,
                            msg=f"bounce {event.index}: force window "
                                f"{event.force_duration:.6g} s vs {ref:.6g} s")
        # the geometric overlap time of an identical episode must not drift
        self.assertLess(res.checks["geometric_overlap_duration_spread"], 0.01)

    # -- 4. no force outside the plane ------------------------------------
    def test_no_reaction_while_separated(self):
        res = self.result
        self.assertEqual(res.checks["separated_force_leak_steps"], 0)
        self.assertEqual(res.checks["separated_force_leak_worst_N"], 0.0)

    # -- 5. free flight is force free --------------------------------------
    def test_free_flight_has_zero_velocity_leak(self):
        # No gravity and no drag in this configuration: between episodes the
        # only possible force is a leaking contact, so v_z must be constant.
        self.assertEqual(self.result.free_flight_velocity_leak, 0.0)

    # -- 6. energy closure -------------------------------------------------
    def test_energy_loss_per_episode_matches_reference(self):
        res = self.result
        mass = res.mass
        for event in res.events:
            ref = event.reference["restitution"]
            loss = 0.5 * mass * (event.impact_speed ** 2 - event.rebound_speed ** 2)
            expect = 0.5 * mass * event.impact_speed ** 2 * (1.0 - ref ** 2)
            self.assertGreater(loss, 0.0)
            self.assertLess(abs(loss - expect) / expect, 0.03)
        speeds = [e.rebound_speed for e in res.events]
        self.assertEqual(speeds, sorted(speeds, reverse=True))

    # -- 7. time-step sensitivity -----------------------------------------
    def test_dt_halving_does_not_move_the_closure(self):
        fine = run_bounce_chain(_cfg(dt_fraction=0.25))
        coarse = self.result
        self.assertAlmostEqual(fine.dt, 0.5 * coarse.dt, places=18)
        self.assertEqual(len(fine.events), len(coarse.events))
        for a, b in zip(coarse.events, fine.events):
            rel = abs(a.restitution - b.restitution) / b.restitution
            self.assertLess(rel, 0.01,
                            msg=f"bounce {a.index}: e={a.restitution:.5f} at "
                                f"dt={coarse.dt:.3g} vs {b.restitution:.5f} at dt={fine.dt:.3g}")
        self.assertLess(fine.checks["restitution_worst_rel_error"], 0.03)


class BounceChainConfigTests(unittest.TestCase):
    def test_rejects_non_positive_inputs(self):
        for bad in (dict(radius=0.0), dict(travel=-1.0), dict(damping=-1.0),
                    dict(bounces=0), dict(dt_fraction=0.0)):
            with self.assertRaises(ValueError):
                run_bounce_chain(_cfg(**bad))

    def test_chain_prediction_is_geometric(self):
        self.assertEqual(chain_prediction(1.0, 0.5, 3), [0.5, 0.25, 0.125])


if __name__ == "__main__":
    unittest.main()