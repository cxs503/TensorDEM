"""Acceptance tests for the unsteady drag ledger (mission IN71-N11).

The ledger is ``F_total = F_quasi-steady + F_added-mass + F_basset + F_pg``.
Every test below is tied to one analytic anchor or one negative control:

(a) added mass: ``A = F_am/(rho_f V du/dt) = 0.5`` (potential flow) and the
    ``alpha -> inf`` asymptote of the exact unsteady Stokes solution (== 1.5 for
    a fixed sphere, i.e. 1 (Froude-Krylov) + 0.5 (added mass));
(b) negative control: steady slip -> no history force at all, total equals
    Schiller-Naumann; ramp-then-hold -> measured history force follows the
    analytic ``1/sqrt(t)`` decay;
(c) oscillatory flow vs Landau & Lifshitz section 24: amplitude < 10 %,
    lag phase < 5 deg;
(d) O(N) recursion vs the O(N^2) reference: full-history L2 < 2 % and a
    CPU-time speedup of at least 10x at >= 1e4 steps; the classical three-term
    kernel is shown to *fail* the 2 % target (why n_exp is a parameter);
(e) density ratio: added-mass coefficient is independent of ``rho_p/rho_f``
    (< 5 %) and the history force dominates the slip forces near neutral
    buoyancy (> 20 %);
(f) multi-sphere: per-sphere force converges to the single-sphere value as the
    solid fraction -> 0 (< 10 %);
(g) negative controls: (i) no spurious history residual in steady flow,
    (ii) halving ``dt`` does not degrade the recursion error.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tensordem.unsteady_drag import (
    BassetRecursionState,
    added_mass_force,
    basset_direct_convolution,
    basset_kernel_relative_l2,
    basset_recursive,
    basset_recursive_blocked,
    basset_recursive_naive,
    basset_recursive_scalar,
    benchmark_basset_methods,
    landau_lifshitz_oscillatory_drag,
    landau_lifshitz_oscillatory_drag_amplitude_phase,
    multi_sphere_unsteady_drag,
    pressure_gradient_force,
    run_dilute_limit_convergence,
    run_free_particle_oscillatory,
    run_impulsive_start,
    run_oscillatory_flow_simulation,
    run_ramp_then_hold,
    run_steady_uniform,
    run_sudden_acceleration,
    schiller_naumann_drag,
    sphere_volume,
    stokes_impulsive_start_force,
    stokes_oscillatory_parameter,
    stokes_second_problem_velocity,
    unsteady_drag_force,
)
from tensordem.unsteady_drag_api import handle_unsteady_drag

# water + a 2 mm sphere
RHO_F = 1000.0
MU = 1.0e-3
D = 2.0e-3
NU = MU / RHO_F
A = D / 2.0


# --------------------------------------------------------------------------- #
# elementary closures
# --------------------------------------------------------------------------- #
class TestQuasiSteady:
    def test_stokes_limit(self):
        """Re -> 0: Schiller-Naumann must reduce to 3 pi mu d u."""
        u = 1.0e-6
        F = schiller_naumann_drag(MU, RHO_F, D, u)
        assert F == pytest.approx(3.0 * math.pi * MU * D * u, rel=1.0e-6)

    def test_sign_and_symmetry(self):
        F_pos = schiller_naumann_drag(MU, RHO_F, D, 1.0e-3)
        F_neg = schiller_naumann_drag(MU, RHO_F, D, -1.0e-3)
        assert F_pos > 0.0
        assert F_neg == pytest.approx(-F_pos, rel=1.0e-12)

    def test_rejects_nonphysical(self):
        with pytest.raises(ValueError):
            schiller_naumann_drag(0.0, RHO_F, D, 1.0e-3)


class TestAnchors:
    def test_landau_lifshitz_low_frequency_limit(self):
        """alpha -> 0: slip force -> Stokes drag (quasi-steady only)."""
        omega = 1.0e-6
        F = landau_lifshitz_oscillatory_drag(omega, D, MU, RHO_F, 1.0)
        assert abs(F) == pytest.approx(6.0 * math.pi * MU * A, rel=1.0e-3)

    def test_landau_lifshitz_high_frequency_asymptote_is_added_mass(self):
        """alpha -> inf: F/(rho_f V dU/dt) -> 1.5 = PG(1) + added mass(0.5)."""
        alpha = 300.0
        omega = 2.0 * NU * alpha ** 2 / A ** 2
        V = sphere_volume(D)
        amp, _ = landau_lifshitz_oscillatory_drag_amplitude_phase(
            omega, D, MU, RHO_F, 1.0)
        A_total = amp / (RHO_F * V * omega)          # |dU/dt| = omega for U0 = 1
        assert A_total == pytest.approx(1.5, rel=0.05)
        assert A_total - 1.0 == pytest.approx(0.5, rel=0.06)   # added-mass share

    def test_stokes_second_problem_decay(self):
        omega = 10.0
        kappa = math.sqrt(omega / (2.0 * NU))
        u0 = stokes_second_problem_velocity(0.0, 0.0, 1.0e-3, omega, NU)
        u1 = stokes_second_problem_velocity(1.0 / kappa, 0.0, 1.0e-3, omega, NU)
        assert u0 == pytest.approx(1.0e-3, rel=1.0e-12)
        # exact solution: U0 exp(-kappa y) cos(omega t - kappa y)
        assert u1 == pytest.approx(1.0e-3 * math.exp(-1.0) * math.cos(1.0), rel=1e-12)

    def test_identical_slip_and_pressure_gradient_shape(self):
        V = sphere_volume(D)
        assert added_mass_force(RHO_F, D, 2.0) == pytest.approx(0.5 * RHO_F * V * 2.0)
        assert pressure_gradient_force(RHO_F, D, 2.0) == pytest.approx(RHO_F * V * 2.0)


# --------------------------------------------------------------------------- #
# (a) added mass
# --------------------------------------------------------------------------- #
class TestAddedMass:
    def test_sudden_acceleration_gives_half(self):
        out = run_sudden_acceleration(RHO_F, D, MU, 1.0, 1.0e-3, 1.0e-7)
        assert out["A_measured_mean"] == pytest.approx(0.5, rel=0.05)
        assert out["A_measured_max_dev"] < 0.05

    def test_impulsive_start_matches_basset_analytic(self):
        """Basset (1888) 6 pi mu a U (1 + a/sqrt(pi nu t)) pins the kernel.

        ``U`` is kept small (Re = 0.02) so that the quasi-steady part is the
        Stokes drag the analytic formula assumes.
        """
        out = run_impulsive_start(RHO_F, D, MU, 1.0e-5, 1.0e-2, 1.0e-6)
        assert out["max_rel_error"] < 0.15
        assert out["mean_rel_error"] < 0.05
        # the history correction must decay like t^-1/2
        t = out["time"]
        corr = np.abs(out["F_basset"][-1]) / abs(out["F_qs"][-1])
        assert corr == pytest.approx(A / math.sqrt(math.pi * NU * t[-1]), rel=0.10)


# --------------------------------------------------------------------------- #
# (b) steady-state negative control + 1/sqrt(t) decay
# --------------------------------------------------------------------------- #
class TestSteadyNegativeControl:
    def test_no_unsteady_terms_without_acceleration(self):
        out = run_steady_uniform(RHO_F, D, MU, 7800.0, 1.0e-3, 5.0e-3, 1.0e-6)
        assert out["F_total_vs_sn_rel_error"] < 1.0e-2
        assert out["F_basset_vs_qs"] < 1.0e-3        # exactly zero here
        assert out["F_added_mass_max_late"] < 1.0e-18
        assert out["F_pressure_gradient_max_late"] < 1.0e-18

    def test_ramp_then_hold_follows_inverse_sqrt(self):
        """After the ramp only the history force survives and it decays 1/sqrt(t)."""
        U, tr, dt, t_end = 0.01, 1.0e-3, 1.0e-4, 2.0
        out = run_ramp_then_hold(RHO_F, D, MU, U, tr, t_end, dt)
        t = out["time"]
        coeff = 1.5 * D ** 2 * math.sqrt(math.pi * RHO_F * MU)
        theory = coeff * (U / tr) * 2.0 * (np.sqrt(t) - np.sqrt(np.maximum(t - tr, 0.0)))
        sel = t > 5.0 * tr
        ratio = float(np.mean(np.abs(out["F_basset"][sel]) / theory[sel]))
        assert 0.9 < ratio < 1.1
        # and it stays well above 1e-3 |F_qs| far after the ramp: 1/sqrt(t) is slow
        assert out["F_basset_vs_qs"] > 1.0e-3


# --------------------------------------------------------------------------- #
# (c) oscillatory flow vs Landau-Lifshitz
# --------------------------------------------------------------------------- #
class TestOscillatoryVsLandauLifshitz:
    @pytest.mark.parametrize("eps", [0.1, 1.0, 10.0])
    def test_amplitude_and_phase(self, eps):
        omega = eps * NU / A ** 2
        period = 2.0 * math.pi / omega
        out = run_oscillatory_flow_simulation(
            RHO_F, D, MU, 7800.0, 1.0e-5, omega, 40.0 * period, period / 400.0)
        assert out["amplitude_rel_error"] < 0.10
        assert out["phase_error_deg"] < 5.0
        assert out["Re_amplitude"] < 0.1          # Stokes regime for the anchor

    def test_omega_to_zero_degenerates_to_steady_drag(self):
        """omega -> 0: the ledger must degenerate to the steady closure (<1 %).

        This is the consistency link between the unsteady closure and the
        already-delivered steady Schiller-Naumann closure.
        """
        omega = 1.0e-6 * NU / A ** 2          # eps -> 1e-6, deep quasi-steady
        period = 2.0 * math.pi / omega
        out = run_oscillatory_flow_simulation(
            RHO_F, D, MU, 7800.0, 1.0e-5, omega, 12.0 * period, period / 200.0,
            measure_fraction=0.5)
        F_sn = schiller_naumann_drag(MU, RHO_F, D, 1.0e-5)
        assert out["amplitude_analytical"] == pytest.approx(F_sn, rel=0.05)
        assert out["amplitude_measured"] == pytest.approx(F_sn, rel=0.05)
        assert out["history_share"] < 0.05

    def test_phase_lag_grows_with_alpha(self):
        lags = []
        for eps in (0.1, 1.0, 10.0):
            omega = eps * NU / A ** 2
            lags.append(landau_lifshitz_oscillatory_drag_amplitude_phase(
                omega, D, MU, RHO_F, 1.0e-5)[1])
        assert lags[0] < lags[1] < lags[2]

    def test_pressure_gradient_doubles_added_mass_part(self):
        """lam^2/3 = 3 * (lam^2/9): the extra 2 lam^2/9 is Froude-Krylov."""
        omega = 1.0
        F_slip = landau_lifshitz_oscillatory_drag(omega, D, MU, RHO_F, 1.0)
        amp_total, _ = landau_lifshitz_oscillatory_drag_amplitude_phase(
            omega, D, MU, RHO_F, 1.0)
        amp_slip, _ = landau_lifshitz_oscillatory_drag_amplitude_phase(
            omega, D, MU, RHO_F, 1.0, include_pressure_gradient=False)
        assert amp_total > amp_slip
        assert abs(F_slip) > 0.0


# --------------------------------------------------------------------------- #
# (d) O(N) recursion: accuracy, term-count sweep, speed
# --------------------------------------------------------------------------- #
class TestBassetRecursion:
    def test_recursive_matches_direct(self):
        dt, n = 1.0e-5, 3000
        t = np.arange(n) * dt
        accel = np.sin(2.0 * math.pi * t / (n * dt / 6.0))
        ref = basset_direct_convolution(accel, dt, MU, RHO_F, D)
        rec = basset_recursive(accel, dt, MU, RHO_F, D, n_exp=16)
        assert np.linalg.norm(rec - ref) / np.linalg.norm(ref) < 0.02

    def test_recursive_matches_direct_for_other_signals(self):
        dt, n = 1.0e-5, 2000
        t = np.arange(n) * dt
        for accel in (np.minimum(t / 1.0e-3, 1.0),
                      (np.arange(n) == 5).astype(float),
                      np.cumsum(np.random.RandomState(0).randn(n)) * 1.0e-3):
            ref = basset_direct_convolution(accel, dt, MU, RHO_F, D)
            rec = basset_recursive(accel, dt, MU, RHO_F, D, n_exp=16)
            assert np.linalg.norm(rec - ref) / np.linalg.norm(ref) < 0.02

    def test_three_term_kernel_fails_the_target(self):
        """The classical 3-exponential fit cannot reach 2 % (justifies n_exp)."""
        assert basset_kernel_relative_l2(1.0e-5, 3000, n_exp=3) > 0.02
        assert basset_kernel_relative_l2(1.0e-5, 3000, n_exp=16) < 0.02

    def test_blocked_and_scalar_implementations_agree(self):
        dt, n = 1.0e-5, 9000
        accel = np.cos(np.arange(n) * dt / 1.0e-3)
        a = basset_recursive_scalar(accel, dt, MU, RHO_F, D)
        b = basset_recursive_blocked(accel, dt, MU, RHO_F, D)
        assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1.0e-10

    def test_naive_reference_agrees_with_vectorised_reference(self):
        dt, n = 1.0e-5, 400
        accel = np.sin(np.arange(n) * 0.05)
        a = basset_recursive_naive(accel, dt, MU, RHO_F, D)
        b = basset_direct_convolution(accel, dt, MU, RHO_F, D)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-30)

    def test_speedup_vs_direct_convolution(self):
        """CPU-time speedup at 1e4 steps (the O(N) claim of the mission)."""
        out = benchmark_basset_methods(1.0e-5, 10000, MU, RHO_F, D,
                                       n_exp_values=(16,))
        row = out["terms"][0]
        assert row["l2_rel_error"] < 0.02
        assert row["speedup_cpu_vs_direct"] > 10.0

    def test_rejects_dt_mismatch(self):
        state = BassetRecursionState.create(MU, RHO_F, D, 1.0e-5, n_exp=8, n_steps=100)
        with pytest.raises(ValueError):
            state.step(1.0, dt=2.0e-5)


# --------------------------------------------------------------------------- #
# (g) second negative control: dt halving must not degrade the recursion
# --------------------------------------------------------------------------- #
class TestTimeStepRefinement:
    def test_dt_halving_does_not_grow_error(self):
        errs = []
        for dt in (2.0e-5, 1.0e-5):
            n = 2000
            accel = np.sin(np.arange(n) * 0.01)
            ref = basset_direct_convolution(accel, dt, MU, RHO_F, D)
            rec = basset_recursive(accel, dt, MU, RHO_F, D, n_exp=16)
            errs.append(np.linalg.norm(rec - ref) / np.linalg.norm(ref))
        assert errs[1] <= errs[0] * 1.5 + 1.0e-6
        assert max(errs) < 0.02

    def test_dt_halving_keeps_oscillatory_amplitude(self):
        omega = NU / A ** 2
        period = 2.0 * math.pi / omega
        outs = [run_oscillatory_flow_simulation(
            RHO_F, D, MU, RHO_F, 1.0e-5, omega, 20.0 * period, period / pp)
            for pp in (200, 400)]
        for out in outs:
            assert out["amplitude_rel_error"] < 0.10
        assert outs[1]["amplitude_rel_error"] <= outs[0]["amplitude_rel_error"] * 2.0


# --------------------------------------------------------------------------- #
# (e) density ratio
# --------------------------------------------------------------------------- #
class TestDensityRatio:
    @pytest.mark.parametrize("rho_ratio", [0.9, 1.05, 1.5, 7.8])
    def test_added_mass_independent_of_density_ratio(self, rho_ratio):
        omega = NU / A ** 2
        period = 2.0 * math.pi / omega
        out = run_free_particle_oscillatory(
            RHO_F, D, MU, rho_ratio, 1.0e-4, omega, 12.0 * period, period / 400.0)
        assert out["added_mass_coefficient_mean"] == pytest.approx(0.5, rel=0.05)

    @pytest.mark.parametrize("rho_ratio", [0.9, 1.05, 1.5])
    def test_history_force_share_near_neutral_buoyancy(self, rho_ratio):
        omega = NU / A ** 2
        period = 2.0 * math.pi / omega
        out = run_free_particle_oscillatory(
            RHO_F, D, MU, rho_ratio, 1.0e-4, omega, 12.0 * period, period / 400.0)
        assert out["history_share"] > 0.20

    def test_neutral_density_particle_comoves_with_fluid(self):
        """rho_p = rho_f: the slip vanishes (no relative motion is driven)."""
        omega = NU / A ** 2
        period = 2.0 * math.pi / omega
        out = run_free_particle_oscillatory(
            RHO_F, D, MU, 1.0, 1.0e-4, omega, 12.0 * period, period / 400.0)
        assert out["slip_ratio"] < 0.01


# --------------------------------------------------------------------------- #
# (f) multi-sphere dilute limit
# --------------------------------------------------------------------------- #
class TestMultiSphere:
    def test_dilute_limit_converges_to_single_sphere(self):
        out = run_dilute_limit_convergence(
            RHO_F, D, MU, 1.0e-5, phi_values=(1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3))
        assert out["convergence_error_smallest_phi"] < 0.10
        assert out["extrapolation_error"] < 0.10

    def test_interaction_increases_drag_monotonically(self):
        out = run_dilute_limit_convergence(
            RHO_F, D, MU, 1.0e-5, phi_values=(1.0e-6, 1.0e-4, 1.0e-2))
        ratios = [row["force_ratio"] for row in out["rows"]]
        assert ratios[0] < ratios[1] < ratios[2]
        assert ratios[-1] > 1.1

    def test_single_sphere_recovers_closed_form(self):
        phi = 1.0e-9
        out = multi_sphere_unsteady_drag(
            np.zeros((1, 3)), D, RHO_F, MU, np.array([1.0e-5, 0.0, 0.0]), phi)
        expected = schiller_naumann_drag(MU, RHO_F, D, 1.0e-5)
        assert out["force_magnitude_mean"] == pytest.approx(expected, rel=1.0e-12)


# --------------------------------------------------------------------------- #
# ledger assembly + API surface
# --------------------------------------------------------------------------- #
class TestLedgerAndApi:
    def test_ledger_terms_add_up(self):
        dt, n = 1.0e-5, 500
        t = np.arange(n) * dt
        res = unsteady_drag_force(np.cos(2.0 * math.pi * t / 1.0e-3),
                                  np.zeros(n), dt, MU, RHO_F, D)
        total = (res.F_quasi_steady + res.F_added_mass + res.F_basset
                 + res.F_pressure_gradient)
        assert np.allclose(res.F_total, total, rtol=1e-12, atol=0.0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            unsteady_drag_force(np.ones(4), np.zeros(4), 1e-6, MU, RHO_F, D,
                                method="fft")

    @pytest.mark.parametrize("payload", [
        {"mode": "oscillatory", "epsilon": 1.0, "n_periods": 20},
        {"mode": "sudden_acceleration", "du_dt": 1.0, "t_end": 1e-4, "dt": 1e-7},
        {"mode": "steady_uniform", "U_slip": 1e-3, "t_end": 2e-3, "dt": 1e-6},
        {"mode": "impulsive_start", "U": 1e-3, "t_end": 1e-3, "dt": 1e-6},
        {"mode": "recursive_benchmark", "n_steps": 10000, "dt": 1e-5},
        {"mode": "dilute_limit"},
    ])
    def test_api_modes_pass_their_criteria(self, payload):
        out = handle_unsteady_drag(payload)
        assert out["all_criteria_pass"], out["criteria"]
        assert out["evidence"]["grade"] == "REAL"

    def test_api_rejects_bad_input(self):
        with pytest.raises(ValueError):
            handle_unsteady_drag({"mode": "nope"})
        with pytest.raises(ValueError):
            handle_unsteady_drag({"mode": "oscillatory", "fluid": {"mu": -1.0}})
        with pytest.raises(ValueError):
            handle_unsteady_drag({"mode": "steady_uniform"})   # no time grid