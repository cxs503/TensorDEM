"""Regression guard: the contact-impact harness must request the dashpot it validates.

The merged harness ``tensordem.contact_impact`` documents and integrates

    F_n = max(0, k_n * delta - c_n * v_n)                                  (1)

with ``c_n`` handed to the engine through ``DEM3DConfig.contact_damping``.  The
engine, however, derives the pair dashpot from ``restitution_coefficient`` in its
default ``contact_damping_model="restitution"`` branch, so an un-flagged harness
integrates an *undamped* pair and silently detaches from its own reference
solution (measured ``e = 1.0003`` against a clamped reference of ``0.7182`` for
``e_target = 0.7``).  These tests pin the coupling from both sides:

* the harness' config declares the absolute convention and carries the dashpot;
* the engine reproduces Eq. (1) under that convention, including the truncation;
* the legacy convention demonstrably ignores ``contact_damping`` -- the negative
  control that makes the flag load-bearing rather than cosmetic;
* the closure claims (restitution and contact duration, 3 % band) hold, and the
  residual shrinks with the time step, i.e. it is discretisation, not physics.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.contact_impact import _build_pair, pair_impact

BAND = 0.03
RADIUS = 0.05
DENSITY = 900.0
K_N = 5.0e4
V0 = 0.5
C_N = 55.4153  # the dashpot the harness installs for e_target = 0.5


def _pair(convention: str = "absolute"):
    """Harness pair, optionally rewritten to the legacy dashpot convention."""
    dem, i, j = _build_pair(K_N, C_N, RADIUS, DENSITY, 1.0)
    if dem.config.contact_damping_model == convention:
        return dem, i, j
    variant = dem.__class__(
        dataclasses.replace(dem.config, contact_damping_model=convention)
    )
    variant.positions = dem.initial_positions.clone()
    variant.initial_positions = dem.initial_positions.clone()
    variant.fixed = dem.fixed.clone()
    variant.velocities = torch.zeros_like(dem.positions)
    variant.forces()  # break the lattice bonds, as the harness itself does
    if int(variant.alive.sum().item()) != 0:
        raise RuntimeError("bonded lattice did not fully break")
    return variant, i, j


def _probe(dem, i, j, delta: float, closing_speed: float) -> float:
    positions = dem.positions.clone()
    positions[i] = torch.tensor([0.0, 0.0, -(RADIUS - delta / 2)], dtype=dem.dtype)
    positions[j] = torch.tensor([0.0, 0.0, RADIUS - delta / 2], dtype=dem.dtype)
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[i] = torch.tensor([0.0, 0.0, closing_speed / 2], dtype=dem.dtype)
    dem.velocities[j] = torch.tensor([0.0, 0.0, -closing_speed / 2], dtype=dem.dtype)
    saved = dem.positions.clone()
    dem.positions = positions
    force = dem.forces(update_fracture=False)[0]
    dem.positions = saved
    return abs(float(force[i, 2]))


class HarnessConventionTests(unittest.TestCase):
    def test_harness_declares_absolute_dashpot(self):
        dem, _, _ = _build_pair(K_N, C_N, RADIUS, DENSITY, 1.0)
        self.assertEqual(dem.config.contact_damping_model, "absolute")
        self.assertEqual(dem.config.contact_damping, C_N)
        # the absolute branch forbids a simultaneous restitution target
        self.assertEqual(dem.config.restitution_coefficient, 1.0)

    def test_engine_reproduces_clipped_force_law(self):
        dem, i, j = _pair("absolute")
        for delta in (1.0e-4, 5.0e-4, 2.0e-3, 5.0e-3):
            for closing in (0.5, 1.0):
                # closing > 0 means approaching, so -c_n*v_n = +c_n*closing
                expected = max(0.0, K_N * delta + C_N * closing)
                measured = _probe(dem, i, j, delta, closing)
                self.assertAlmostEqual(
                    measured, expected, delta=1e-9 * max(expected, 1.0),
                    msg=f"delta={delta}, closing={closing}: "
                        f"engine {measured} vs law {expected}",
                )

    def test_force_truncates_when_separating_fast(self):
        """Eq. (1) clamps to zero: a fast-separating pair feels no tensile force."""
        dem, i, j = _pair("absolute")
        delta = 1.0e-5
        separating = 2.0  # k*delta = 0.5 N < c*v_n = 110.8 N
        self.assertGreater(C_N * separating, K_N * delta)
        self.assertEqual(_probe(dem, i, j, delta, -separating), 0.0)

    def test_legacy_convention_ignores_the_installed_dashpot(self):
        """Negative control: without the flag the harness validates nothing."""
        absolute, i, j = _pair("absolute")
        legacy, _, _ = _pair("restitution")
        delta, closing = 1.0e-4, 1.0
        f_abs = _probe(absolute, i, j, delta, closing)
        f_rest = _probe(legacy, i, j, delta, closing)
        law = max(0.0, K_N * delta + C_N * closing)
        self.assertAlmostEqual(f_abs, law, delta=1e-9 * max(law, 1.0))
        self.assertLess(f_rest, 0.5 * law)
        self.assertAlmostEqual(f_rest, K_N * delta, delta=1e-9 * K_N * delta)


class ClosureAfterFixTests(unittest.TestCase):
    def test_restitution_within_band_and_monotone_in_dashpot(self):
        measured = []
        for e_target in (0.9, 0.5, 0.3):
            r = pair_impact(e_target=e_target)
            self.assertLess(
                abs(r.e_error_vs_clamped), BAND,
                msg=f"e_target={e_target}: {r.e_measured:.5f} vs clamped "
                    f"{r.e_clamped:.5f} ({100 * r.e_error_vs_clamped:+.3f}%)",
            )
            measured.append(r.e_measured)
        self.assertGreater(measured[0], measured[1])
        self.assertGreater(measured[1], measured[2])

    def test_contact_durations_within_band(self):
        r = pair_impact(e_target=0.5)
        self.assertLess(abs(r.contact_time_force_error), BAND)
        self.assertLess(abs(r.contact_time_error), BAND)
        # the force acts for less time than the overlap lasts
        self.assertLess(r.contact_time_force, r.contact_time)
        self.assertGreater(r.steps_per_contact, 10.0)

    def test_duration_residual_is_discretisation(self):
        """Refining dt reduces both duration residuals: first-order, not physics."""
        coarse = pair_impact(e_target=0.5, dt_scale=1.0)
        fine = pair_impact(e_target=0.5, dt_scale=0.25)
        self.assertLess(abs(fine.e_error_vs_clamped), abs(coarse.e_error_vs_clamped))
        self.assertLess(
            abs(fine.contact_time_force_error), abs(coarse.contact_time_force_error)
        )
        self.assertLess(abs(fine.contact_time_error), abs(coarse.contact_time_error))
        self.assertLess(abs(fine.contact_time_force_error), BAND)
        # observed order stays in the first-order neighbourhood (dt to dt/4)
        ratio = abs(coarse.contact_time_error) / max(abs(fine.contact_time_error), 1e-15)
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 8.0)

    def test_force_law_and_momentum_residuals_are_zero(self):
        r = pair_impact(e_target=0.5)
        self.assertLess(r.force_law_residual, 1e-12)
        self.assertLess(abs(r.momentum_error), 1e-12)


if __name__ == "__main__":
    unittest.main()