"""Regression guard for the dashpot convention used by the impact harness.

Verification debt targeted here: ``DEM3DConfig.contact_damping_model`` defaults to
``"restitution"``, which rebuilds the normal dashpot from
``restitution_coefficient`` and *ignores* ``contact_damping`` on the
particle-particle branch.  A harness that installs an absolute dashpot
``c_n = restitution_to_damping(e, k_n, m_eff)`` [N s/m] but leaves the default
model in place therefore measures a purely elastic impact (``e ~ 1.0``) and every
contact-duration baseline collapses.  ``contact_impact._build_pair`` must be
explicit about ``contact_damping_model="absolute"``.

The tests below pin the consequences, so the silent-elastic failure mode cannot
come back unnoticed:

* the measured restitution must sit on the *dissipative* side (``e < 0.95`` for a
  target of 0.5) and match the clamped closed form within the 3 % band;
* both contact-duration baselines must close within 3 %;
* refining ``dt`` must reduce the duration residual (first-order convergence),
  i.e. the agreement is discretisation-limited, not luck.
"""
import unittest

from tensordem.contact_impact import pair_impact

ACCURACY_BAND = 0.03


class DashpotConventionTests(unittest.TestCase):
    def test_harness_installs_an_absolute_dashpot(self):
        """e_target=0.5 must produce a dissipative impact, not e ~ 1.0."""
        r = pair_impact(e_target=0.5)
        self.assertLess(r.e_measured, 0.95, "dashpot did not act: engine fell back to "
                                           "contact_damping_model='restitution'")
        self.assertGreater(r.e_measured, 0.30, "dashpot far stronger than requested")

    def test_measured_restitution_matches_clamped_closed_form(self):
        for e_target in (0.7, 0.5, 0.3):
            r = pair_impact(e_target=e_target)
            self.assertLess(
                abs(r.e_error_vs_clamped), ACCURACY_BAND,
                msg=f"e_target={e_target}: {100 * r.e_error_vs_clamped:+.3f}% vs clamped form",
            )

    def test_contact_durations_close_within_band(self):
        for e_target in (0.7, 0.5, 0.3):
            r = pair_impact(e_target=e_target)
            self.assertLess(
                abs(r.contact_time_force_error), ACCURACY_BAND,
                msg=f"e_target={e_target}: force duration {100 * r.contact_time_force_error:+.3f}%",
            )
            self.assertLess(
                abs(r.contact_time_error), ACCURACY_BAND,
                msg=f"e_target={e_target}: overlap duration {100 * r.contact_time_error:+.3f}%",
            )

    def test_force_duration_is_shorter_than_overlap_duration(self):
        """After the normal force truncates at zero the pair still overlaps."""
        r = pair_impact(e_target=0.5)
        self.assertLess(r.contact_time_force, r.contact_time)
        self.assertLessEqual(r.force_steps, r.contact_steps)

    def test_duration_residual_converges_under_dt_refinement(self):
        coarse = pair_impact(e_target=0.5, dt_scale=1.0)
        fine = pair_impact(e_target=0.5, dt_scale=0.25)
        self.assertGreater(abs(coarse.contact_time_force_error), 0.0)
        self.assertLess(
            abs(fine.contact_time_force_error), abs(coarse.contact_time_force_error),
            "halving dt twice must shrink the duration residual (first-order scheme)",
        )
        self.assertLess(abs(fine.contact_time_force_error), ACCURACY_BAND)


if __name__ == "__main__":
    unittest.main()