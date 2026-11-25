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

def test_force_release_against_integrated_trajectory(record_property):
    from scipy.integrate import solve_ivp

    worst_e = worst_time = 0.
    for zeta in (.71, .9999, 1., 1.0001, 2., 8.):
        # Normalized mass, stiffness and initial closing speed are one.
        def rhs(t, state):
            x, v = state
            return v, -x-2*zeta*v

        def force_zero(t, state):
            return state[0]+2*zeta*state[1]

        force_zero.terminal = True
        force_zero.direction = -1
        result = solve_ivp(rhs, (0., 4.), (0., 1.), events=force_zero,
                           rtol=2e-12, atol=2e-14)
        release, = result.t_events[0]
        (overlap, velocity), = result.y_events[0]
        model = impact_analytic(1., 2*zeta, 1.)
        assert velocity < 0 and overlap > 0
        worst_e = max(worst_e, abs(model.e_clamped+velocity))
        worst_time = max(worst_time, abs(model.contact_time_force-release),
                         abs(model.contact_time_geometric-(release-overlap/velocity)))
        damping = restitution_to_damping_clamped(model.e_clamped, 1., 1.)
        assert abs(damping-2*zeta) < 2e-10
    assert clamped_restitution(1.) == math.exp(-2.)
    record_property("max_restitution_error", worst_e)
    record_property("max_normalized_duration_error", worst_time)
    assert worst_e < 1e-11 and worst_time < 1e-10


def test_overdamped_dem_pair_at_three_time_steps(record_property):
    target = .08
    errors = []
    for scale in (.2, .1, .05):
        impact = pair_impact(e_target=target, damping="clamped_mapping", dt_scale=scale)
        m_eff = impact.pair_mass_kg/2
        damping = restitution_to_damping_clamped(target, 5.e4, m_eff)
        reference = impact_analytic(5.e4, damping, m_eff, 2*impact.impact_speed)
        peak_error = abs(impact.max_overlap/reference.peak_overlap_m-1)
        time_error = abs(impact.time_to_peak_overlap_s/reference.time_to_peak_overlap_s-1)
        record_property(f"peak_overlap_relative_error_{scale}", peak_error)
        record_property(f"peak_time_relative_error_{scale}", time_error)
        assert peak_error < .01 and time_error < .01
        force_peak_error = abs(impact.peak_force/reference.peak_force_n-1)
        record_property(f"peak_load_relative_error_{scale}", force_peak_error)
        record_property(f"peak_load_delay_in_steps_{scale}", impact.time_to_peak_force_s/impact.dt)
        assert force_peak_error < .01
        assert 0 <= impact.time_to_peak_force_s <= impact.dt
        loss = reference.dashpot_work_j + reference.release_elastic_energy_j
        energy_error = abs(impact.dissipated_energy-loss)/loss
        record_property(f"total_loss_relative_error_{scale}", energy_error)
        assert energy_error < 5e-5
        error = abs(impact.e_measured-target)
        force_error = abs(impact.contact_time_force/impact.contact_time_force_analytic-1)
        geometric_error = abs(impact.contact_time/impact.contact_time_analytic-1)
        record_property(f"restitution_error_{scale}", error)
        record_property(f"force_duration_relative_error_{scale}", force_error)
        record_property(f"geometric_duration_relative_error_{scale}", geometric_error)
        assert error < .0003
        assert abs(impact.momentum_error) < 1e-12
        errors.append((force_error, geometric_error))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(b < a for a,b in zip(coarse, fine))


def test_active_dashpot_work_and_residual_spring_energy(record_property):
    from scipy.integrate import solve_ivp

    k, m, v = 73., 1.7, 2.3
    worst = 0.
    for zeta in (0., 1e-9, .3, 1., 4.):
        c = 2*zeta*math.sqrt(k*m)
        ref = impact_analytic(k, c, m, v)
        # Integrate the active oscillator and its cumulative dashpot work.
        def rhs(t, state):
            overlap, rate, work = state
            return rate, -(k*overlap+c*rate)/m, c*rate**2
        def release(t, state):
            return k*state[0]+c*state[1]
        release.terminal = True
        release.direction = -1
        run = solve_ivp(rhs, (0., 2*math.pi*math.sqrt(m/k)), (0., v, 0.),
                        events=release, atol=1e-13, rtol=1e-12)
        (x, rate, work), = run.y_events[0]
        initial = .5*m*v*v
        worst = max(worst, abs(work-ref.dashpot_work_j)/initial,
                    abs(.5*k*x*x-ref.release_elastic_energy_j)/initial)
        assert abs(x-ref.release_overlap_m) < 2e-12
        assert abs(.5*m*rate**2+ref.dashpot_work_j+ref.release_elastic_energy_j-initial) < 2e-11
        assert ref.dashpot_work_j >= 0
    record_property("max_component_error_over_incident_energy", worst)
    assert worst < 2e-11

def test_peak_compression_against_velocity_zero_event(record_property):
    from scipy.integrate import solve_ivp

    k, m, closing_speed = 73., 1.7, 2.3
    worst_depth = worst_time = 0.
    for zeta in (0., .25, .999999, 1., 1.000001, 3., 8.):
        damping = 2*zeta*math.sqrt(k*m)
        ref = impact_analytic(k, damping, m, closing_speed)
        def rhs(t, state):
            depth, speed = state
            return speed, -(k*depth+damping*speed)/m
        def peak(t, state):
            return state[1]
        peak.terminal = True
        peak.direction = -1
        run = solve_ivp(rhs, (0., math.pi*math.sqrt(m/k)), (0., closing_speed),
                        events=peak, atol=1e-14, rtol=1e-12)
        time, = run.t_events[0]
        (depth, _), = run.y_events[0]
        worst_depth = max(worst_depth, abs(ref.peak_overlap_m/depth-1))
        worst_time = max(worst_time, abs(ref.time_to_peak_overlap_s/time-1))
        assert 0 <= ref.release_overlap_m < ref.peak_overlap_m
    record_property("max_relative_peak_overlap_error", worst_depth)
    record_property("max_relative_peak_time_error", worst_time)
    assert max(worst_time, worst_depth) < 2e-11

def test_peak_force_against_integrated_force_turning_point(record_property):
    from scipy.integrate import solve_ivp

    k, m, v = 73., 1.7, 2.3
    worst_force = worst_time = 0.
    for zeta in (0., .2, .49, .499999, .5, .500001, .8, 1., 4.):
        c = 2*zeta*math.sqrt(k*m)
        ref = impact_analytic(k, c, m, v)
        def rhs(t, state):
            x, rate = state
            return rate, -(k*x+c*rate)/m
        def turning(t, state):
            rate, acc = rhs(t, state)
            return k*rate+c*acc
        turning.direction = -1
        run = solve_ivp(rhs, (0., ref.contact_time_force), (0., v),
                        events=turning, rtol=1e-12, atol=1e-14)
        candidates = [(c*v, 0.)]
        for time, (x, rate) in zip(run.t_events[0], run.y_events[0]):
            candidates.append((k*x+c*rate, time))
        force, time = max(candidates)
        worst_force = max(worst_force, abs(ref.peak_force_n/force-1))
        worst_time = max(worst_time, abs(ref.time_to_peak_force_s-time)*ref.omega_n)
        assert ref.time_to_peak_force_s <= ref.time_to_peak_overlap_s
    record_property("max_relative_peak_load_error", worst_force)
    record_property("max_dimensionless_peak_time_error", worst_time)
    assert max(worst_force,worst_time) < 3e-11


def test_dem_pair_has_interior_load_peak_at_weak_damping(record_property):
    impact = _run(e_target=.5, damping="clamped_mapping")
    mass = impact.pair_mass_kg/2
    damping = restitution_to_damping_clamped(.5, 5.e4, mass)
    ref = impact_analytic(5.e4, damping, mass, 2*impact.impact_speed)
    load_error = abs(impact.peak_force/ref.peak_force_n-1)
    delay_error = abs(impact.time_to_peak_force_s-ref.time_to_peak_force_s)/impact.dt
    record_property("relative_peak_load_error", load_error)
    record_property("peak_time_error_in_steps", delay_error)
    assert 0 < impact.time_to_peak_force_s < impact.time_to_peak_overlap_s
    assert load_error < .03 and delay_error < 2
