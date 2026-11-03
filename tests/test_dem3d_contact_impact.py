"""Analytic contact-impact validation of the DEM3D normal contact model.

Verification debt targeted here: ``dem3d.py`` truncates every normal force at
zero (``max(0, k*overlap - c*v_n)``), whereas
``contact_tangential.restitution_to_damping`` inverts the *unclamped*
spring-dashpot restitution identity.  These tests pin down the difference
against two independent analytic references and guard against regressions in
either direction:

* the solver must reproduce the **clamped** closed form to better than the 3 %
  acceptance band;
* the unclamped mapping must stay measurably biased, so the known limitation
  cannot be silently "fixed" by weakening the reference.
"""
import math
import unittest

from tensordem.contact_impact import (
    clamped_restitution,
    impact_analytic,
    pair_impact,
    restitution_to_damping_clamped,
    rk4_clamped_restitution,
    unclamped_restitution,
)
from tensordem.contact_tangential import damping_to_restitution, restitution_to_damping

ACCURACY_BAND = 0.03  # 3 % engineering acceptance band

_CACHE: dict[tuple, object] = {}


def _run(**kwargs):
    """Memoise solver runs: several tests reuse the same physical case."""
    key = tuple(sorted(kwargs.items()))
    if key not in _CACHE:
        _CACHE[key] = pair_impact(**kwargs)
    return _CACHE[key]


class DerivationTests(unittest.TestCase):
    """The clamped closed form must match an independent RK4 integration."""

    def test_closed_form_matches_rk4(self):
        for zeta in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.69):
            self.assertAlmostEqual(
                clamped_restitution(zeta),
                rk4_clamped_restitution(zeta),
                delta=1e-8,
                msg=f"closed form vs RK4 disagree at zeta={zeta}",
            )

    def test_elastic_limit(self):
        self.assertEqual(clamped_restitution(0.0), 1.0)
        self.assertEqual(unclamped_restitution(0.0), 1.0)

    def test_clamped_never_below_unclamped(self):
        """Removing the retarding force can only increase the rebound."""
        for zeta in (0.05, 0.2, 0.4, 0.6):
            self.assertGreater(clamped_restitution(zeta), unclamped_restitution(zeta))

    def test_monotone_decreasing(self):
        values = [clamped_restitution(z) for z in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)]
        for coarser, finer in zip(values, values[1:]):
            self.assertGreater(coarser, finer)

    def test_force_and_geometric_durations_ordering(self):
        """The pair keeps overlapping after the force clamps to zero."""
        a = impact_analytic(5.0e4, 46.82, 0.2356)
        self.assertGreater(a.contact_time_geometric, a.contact_time_force)
        self.assertAlmostEqual(a.contact_time_force, a.theta_star / a.omega_d)


class CorrectedMappingTests(unittest.TestCase):
    """The corrected mapping must invert the clamped model, not the free one."""

    def test_round_trip_hits_target(self):
        k_n, m_eff = 5.0e4, 0.2356
        for e_target in (0.9, 0.7, 0.5, 0.3):
            c = restitution_to_damping_clamped(e_target, k_n, m_eff)
            zeta = c / (2.0 * math.sqrt(k_n * m_eff))
            self.assertAlmostEqual(clamped_restitution(zeta) / e_target, 1.0, delta=1e-8)

    def test_corrected_damping_is_larger(self):
        """Truncating the force is less dissipative, so more damping is needed.

        At equal ``gamma_n`` the clamped model rebounds faster than the free
        oscillator; reaching the same target restitution therefore requires a
        larger ``gamma_n`` than the unclamped identity returns.
        """
        k_n, m_eff = 5.0e4, 0.2356
        for e_target in (0.8, 0.5, 0.3):
            self.assertGreater(
                restitution_to_damping_clamped(e_target, k_n, m_eff),
                restitution_to_damping(e_target, k_n, m_eff),
            )

    def test_known_bias_of_existing_mapping(self):
        """The unclamped identity is off by >3 % for e_target <= 0.7 (the debt)."""
        k_n, m_eff = 5.0e4, 0.2356
        for e_target in (0.5, 0.3):
            c = restitution_to_damping(e_target, k_n, m_eff)
            zeta = c / (2.0 * math.sqrt(k_n * m_eff))
            realised = clamped_restitution(zeta)
            self.assertGreater(
                realised / e_target - 1.0, ACCURACY_BAND,
                msg=f"expected >3% bias for e_target={e_target}, got {realised}",
            )
            # The existing inverse is still exact for the model it documents.
            self.assertAlmostEqual(damping_to_restitution(c, k_n, m_eff), e_target, places=12)

    def test_elastic_and_invalid_inputs(self):
        self.assertEqual(restitution_to_damping_clamped(1.0, 5.0e4, 0.2356), 0.0)
        with self.assertRaises(ValueError):
            restitution_to_damping_clamped(0.0, 5.0e4, 0.2356)
        with self.assertRaises(ValueError):
            restitution_to_damping_clamped(1.2, 5.0e4, 0.2356)


class SolverImpactTests(unittest.TestCase):
    """The DEM3D solver against the clamped analytic reference."""

    def test_force_law_conformance(self):
        """Assembled normal force equals max(0, k*overlap - c*v_n) exactly."""
        r = _run(e_target=0.5)
        self.assertLess(r.force_law_residual, 1e-12)

    def test_momentum_conserved(self):
        """Internal pair forces are equal and opposite: zero momentum drift."""
        r = _run(e_target=0.5)
        self.assertEqual(r.momentum_error, 0.0)

    def test_restitution_matches_clamped_analytic(self):
        for e_target in (0.7, 0.5, 0.3):
            r = _run(e_target=e_target)
            self.assertLess(
                abs(r.e_error_vs_clamped), ACCURACY_BAND,
                msg=f"e_target={e_target}: solver {r.e_measured:.5f} vs clamped "
                    f"{r.e_clamped:.5f} ({100*r.e_error_vs_clamped:+.3f}%)",
            )

    def test_known_bias_is_visible_in_solver(self):
        """Solver confirms the >3 % bias of the unclamped mapping at e_target=0.5."""
        r = _run(e_target=0.5)
        self.assertGreater(r.e_error_vs_unclamped, ACCURACY_BAND)

    def test_corrected_mapping_realises_target(self):
        r = _run(e_target=0.5, damping="clamped_mapping")
        self.assertLess(abs(r.e_measured / 0.5 - 1.0), ACCURACY_BAND)

    def test_contact_durations(self):
        r = _run(e_target=0.5)
        self.assertLess(abs(r.contact_time_force_error), ACCURACY_BAND)
        self.assertLess(abs(r.contact_time_error), ACCURACY_BAND)

    def test_rate_independence(self):
        """Linear contact model: restitution must not depend on impact speed."""
        slow = _run(e_target=0.5, v0=0.25)
        fast = _run(e_target=0.5, v0=1.0)
        self.assertLess(abs(fast.e_measured / slow.e_measured - 1.0), 0.01)

    def test_timestep_convergence(self):
        """Halving dt must reduce the restitution error."""
        coarse = _run(e_target=0.5)
        fine = _run(e_target=0.5, dt_scale=0.5)
        self.assertLess(abs(fine.e_error_vs_clamped), abs(coarse.e_error_vs_clamped))
        self.assertLess(abs(fine.e_error_vs_clamped), ACCURACY_BAND)


if __name__ == "__main__":
    unittest.main()