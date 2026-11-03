"""Closed-loop verification of the restitution <-> contact-damping mapping.

The shipped 3-D solver uses a compressive-only normal contact, while
``tensordem.contact_tangential.restitution_to_damping`` returns the damping of
the *unclamped* linear spring-dashpot oscillator. These tests pin down, with
independent references, where the two agree and where they do not:

* the solver must reproduce the clamped one-dimensional impact model,
* the classic closed form must agree with the clamped model only in the
  weak-damping limit (the known 3 % gate up to zeta ~ 0.1),
* ``damping_for_target_restitution`` must hit a requested coefficient of
  restitution in the real solver far better than the classic mapping, and
* the equal-mass pair impact must use the reduced mass ``m / 2``.
"""
import math
import unittest

import torch

from tensordem.contact_restitution import (
    clamped_restitution_ratio,
    classic_restitution_ratio,
    damping_for_target_restitution,
    measure_dem3d_impact,
)
from tensordem.contact_tangential import damping_to_restitution, restitution_to_damping

RADIUS = 0.025
DENSITY = 917.0
K_N = 5_000.0
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3


class RestitutionVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def impact(self, gamma_n, **kwargs):
        return measure_dem3d_impact(gamma_n, k_n=K_N, radius=RADIUS, density=DENSITY, **kwargs)

    def test_classic_formula_agrees_with_existing_mapping_helper(self):
        for zeta in (0.01, 0.05, 0.15, 0.3):
            gamma = 2.0 * zeta * math.sqrt(K_N * MASS)
            self.assertAlmostEqual(classic_restitution_ratio(zeta),
                                   damping_to_restitution(gamma, K_N, MASS), places=9)

    def test_classic_formula_is_the_weak_damping_limit_of_clamped_model(self):
        for zeta in (0.005, 0.01, 0.02):
            gamma = 2.0 * zeta * math.sqrt(K_N * MASS)
            reference = clamped_restitution_ratio(gamma, K_N, MASS)["e_reference"]
            self.assertLess(abs(reference - classic_restitution_ratio(zeta)) / reference, 1e-3)

    def test_reference_integration_is_step_independent(self):
        coarse = clamped_restitution_ratio(5.0, K_N, MASS, steps=1_000)["e_reference"]
        fine = clamped_restitution_ratio(5.0, K_N, MASS, steps=8_000)["e_reference"]
        self.assertLess(abs(coarse - fine) / fine, 1e-3)

    def test_clamping_raises_restitution_for_moderate_damping(self):
        # The force clamp releases the contact while spheres still overlap, so
        # the clamped model must rebound faster than the classic oscillator.
        zeta = 0.289  # gamma = 10 N s/m for the default-like stiffness/mass
        gamma = 2.0 * zeta * math.sqrt(K_N * MASS)
        reference = clamped_restitution_ratio(gamma, K_N, MASS)["e_reference"]
        classic = classic_restitution_ratio(zeta)
        self.assertGreater(reference, classic)
        self.assertGreater((reference - classic) / classic, 0.05)
        self.assertGreater(reference and clamped_restitution_ratio(gamma, K_N, MASS)["release_overlap_m"], 0.0)

    def test_solver_reproduces_clamped_reference_model(self):
        for gamma in (1.0, 3.0, 5.0):
            reference = clamped_restitution_ratio(gamma, K_N, MASS)["e_reference"]
            run = self.impact(gamma)
            relative = abs(run["e_measured"] - reference) / reference
            self.assertLess(relative, 0.01,
                            msg=f"gamma={gamma}: solver {run['e_measured']:.4f} "
                                f"vs reference {reference:.4f} ({relative:.2%})")

    def test_classic_mapping_is_within_3pct_only_for_small_damping(self):
        # Weak damping: the mapping helper is usable as documented.
        gamma = 2.0 * 0.058 * math.sqrt(K_N * MASS)
        weak = self.impact(gamma)
        weak_zeta = clamped_restitution_ratio(gamma, K_N, MASS)["zeta"]
        weak_error = abs(weak["e_measured"] - classic_restitution_ratio(weak_zeta)) / weak["e_measured"]
        self.assertLess(weak_error, 0.03)
        # Moderate damping: the same mapping is off by far more than 3 %, and
        # the deviation survives time-step refinement (it is model, not mesh).
        moderate = self.impact(10.0)
        refined = self.impact(10.0, dt_scale=0.5)
        classic = classic_restitution_ratio(moderate["zeta"])
        self.assertGreater(abs(moderate["e_measured"] - classic) / classic, 0.15)
        self.assertLess(abs(refined["e_measured"] - moderate["e_measured"]) / moderate["e_measured"], 0.01)

    def test_target_restitution_mapping_closes_the_loop(self):
        target = 0.6
        calibrated = damping_for_target_restitution(target, K_N, MASS)
        run = self.impact(calibrated["gamma_n"])
        self.assertLess(abs(run["e_measured"] - target) / target, 0.015)
        # The classic mapping is measurably worse at the same target.
        classic_gamma = restitution_to_damping(target, K_N, MASS)
        classic_run = self.impact(classic_gamma)
        self.assertGreater(abs(classic_run["e_measured"] - target),
                           abs(run["e_measured"] - target))

    def test_equal_mass_pair_uses_reduced_mass(self):
        run = self.impact(2.0, both_free=True)
        self.assertAlmostEqual(run["m_eff"], 0.5 * MASS, places=12)
        reference = clamped_restitution_ratio(2.0, K_N, 0.5 * MASS)["e_reference"]
        self.assertLess(abs(run["e_measured"] - reference) / reference, 0.015)

    def test_global_drag_depresses_measured_restitution(self):
        # DEM3DConfig applies -drag * v to every particle; with the default
        # drag = 2 N s/m that channel, not the contact damping, dominates the
        # observed rebound. Documented as a confound in the study note.
        clean = self.impact(2.0, drag=0.0)
        default_drag = self.impact(2.0, drag=2.0)
        self.assertLess(default_drag["e_measured"], clean["e_measured"] - 0.10)


if __name__ == "__main__":
    unittest.main()