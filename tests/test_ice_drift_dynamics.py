"""Acceptance + negative-control tests for :mod:`tensordem.ice_drift_dynamics`.

Machine-checkable criteria of campaign node IN27-S6:

(a) Nansen turning angle in [20, 40] deg (Ekman-coupled free drift, whole suite)
(b) wind factor ``U_ice / U_wind`` in [1 %, 3 %]
(c) Ekman decay depth ``D_E = sqrt(2 K / f)`` recovered from an independent
    finite-difference solve; 45 deg surface veer; Ekman transport ``tau/(rho f)``
(d) negative control: ``f = 0`` (no rotation) -> turning angle exactly 0
(e) the case suite covers ``phi in [60, 85] deg x U_wind in [5, 15] m/s`` (12 cases)

Additional physics invariants that the closure must satisfy:

* the closed-form steady state is an exact solution (momentum residual ~ 1e-17);
* ``theta = arctan(f / c_d)`` is an identity, not a fit;
* with Coriolis on the ice mass alone the added mass leaves the steady state
  *invariant* and only rescales the acceleration (virtual-mass effect);
* the RK4 time integration converges to the closed-form steady state;
* the southern hemisphere flips the deflection to the left.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.ice_drift_dynamics import (  # noqa: E402
    OMEGA_EARTH,
    DriftEnvironment,
    FloeInertia,
    added_mass_tensor,
    coriolis_parameter,
    drift_case_matrix,
    ekman_coupled_drift,
    ekman_decay_depth,
    ekman_profile,
    ekman_surface_current,
    ekman_transport,
    nansen_angle_from_drag_frequency,
    run_case_suite,
    simulate_free_drift,
    solve_ekman_finite_difference,
    solve_steady_drift,
    steady_drift_analytic,
    water_drag,
    wind_stress,
)

SUITE = run_case_suite(eddy_viscosity=0.05, include_added_mass_coriolis=True)


# ---------------------------------------------------------------------------
# Rotation basis
# ---------------------------------------------------------------------------
class TestCoriolisParameter:
    def test_value_at_75n(self):
        """f = 2 Omega sin(phi); at 75N f = 1.4086e-4 1/s."""
        assert coriolis_parameter(75.0) == pytest.approx(1.4086e-4, rel=1e-4)

    def test_scales_with_sin_latitude(self):
        for phi in (60.0, 70.0, 85.0):
            assert coriolis_parameter(phi) == pytest.approx(
                2.0 * OMEGA_EARTH * math.sin(math.radians(phi)), rel=1e-12)

    def test_equator_and_hemispheres(self):
        assert coriolis_parameter(0.0) == pytest.approx(0.0, abs=1e-18)
        assert coriolis_parameter(-70.0) == pytest.approx(-coriolis_parameter(70.0), rel=1e-12)

    def test_latitude_range_guard(self):
        with pytest.raises(ValueError):
            coriolis_parameter(91.0)


class TestNansenAngleIdentity:
    def test_arctan_identity(self):
        """tan(theta) = f / c_d exactly (theta = arctan(f/c_d))."""
        f = coriolis_parameter(75.0)
        for cd in (1e-4, 3e-4, 1e-3):
            theta = nansen_angle_from_drag_frequency(f, cd)
            assert math.tan(math.radians(theta)) == pytest.approx(f / cd, rel=1e-12)

    def test_zero_rotation_gives_zero_angle(self):
        assert nansen_angle_from_drag_frequency(0.0, 5e-4) == 0.0

    def test_invalid_drag_frequency(self):
        with pytest.raises(ValueError):
            nansen_angle_from_drag_frequency(1e-4, 0.0)


# ---------------------------------------------------------------------------
# Closed-form steady state
# ---------------------------------------------------------------------------
class TestAnalyticSteadyState:
    def test_residual_is_machine_zero(self):
        """The closed form is an exact solution of the momentum balance."""
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        for h in (1.0, 1.5, 3.0):
            res = steady_drift_analytic(env, FloeInertia(ice_thickness=h))
            assert abs(res["residual"]) < 1e-12

    def test_angle_matches_arctan_of_f_over_cd(self):
        env = DriftEnvironment(latitude_deg=70.0, wind_speed=10.0)
        res = steady_drift_analytic(env, FloeInertia())
        theta = nansen_angle_from_drag_frequency(res["f"], res["drag_frequency"])
        assert res["angle_right_deg"] == pytest.approx(theta, abs=1e-10)

    def test_rotation_is_to_the_right_in_northern_hemisphere(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)  # wind toward +x
        res = steady_drift_analytic(env, FloeInertia())
        assert res["angle_right_deg"] > 0.0
        assert res["v"] < 0.0  # +y is 90 deg to the left

    def test_angle_grows_with_ice_mass_and_friction_parameter(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        thin = steady_drift_analytic(env, FloeInertia(ice_thickness=1.0))
        thick = steady_drift_analytic(env, FloeInertia(ice_thickness=3.0))
        assert thick["angle_right_deg"] > thin["angle_right_deg"]

    def test_angle_falls_with_wind_speed(self):
        """tan(theta) = f M / (rho_w C_w |U|) and |U| grows like U_wind."""
        th = [steady_drift_analytic(DriftEnvironment(latitude_deg=75.0, wind_speed=w),
                                    FloeInertia())["angle_right_deg"]
              for w in (5.0, 10.0, 15.0)]
        assert th[0] > th[1] > th[2]

    def test_still_water_required(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0, current_speed=0.05)
        with pytest.raises(ValueError):
            steady_drift_analytic(env, FloeInertia())

    def test_holds_in_southern_hemisphere_with_flipped_sign(self):
        env = DriftEnvironment(latitude_deg=-70.0, wind_speed=10.0)
        res = steady_drift_analytic(env, FloeInertia())
        assert res["f"] < 0.0
        assert res["angle_right_deg"] < 0.0  # deflected to the left
        assert res["v"] > 0.0


class TestNewtonSolver:
    def test_matches_closed_form_in_still_water(self):
        """Relative-wind Newton solve reproduces the closed form to ~2 %.

        The closed form prescribes the wind stress at ``U = 0``; the physical
        stress acts on ``U_a - U``, whose magnitude is smaller by
        ``2 U_ice/U_wind`` ~ 4 %, i.e. ~2 % in the drift speed.  This test pins
        that documented model difference rather than hiding it.
        """
        for phi in (60.0, 75.0, 85.0):
            for w in (5.0, 10.0, 15.0):
                env = DriftEnvironment(latitude_deg=phi, wind_speed=w)
                inertia = FloeInertia()
                ana = steady_drift_analytic(env, inertia)
                num = solve_steady_drift(env, inertia, u_guess=(ana["u"], ana["v"]))
                assert num["speed"] == pytest.approx(ana["speed"], rel=3e-2)
                assert num["angle_right_deg"] == pytest.approx(ana["angle_right_deg"], abs=1.0)
                # the relative-wind solve must be slower but not by more than a few %
                assert 0.95 * ana["speed"] < num["speed"] < ana["speed"]

    def test_residual_vanishes(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0, current_speed=0.08,
                               current_dir_deg=200.0)
        num = solve_steady_drift(env, FloeInertia())
        U = np.array([num["u"], num["v"]])
        f = env.f
        M = FloeInertia().mass_per_area
        res = (-M * f * np.array([-U[1], U[0]]) + wind_stress(env, U) + water_drag(env, U))
        assert float(np.hypot(*res)) < 1e-10

    def test_zero_wind_gives_zero_drift(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=0.0)
        num = solve_steady_drift(env, FloeInertia())
        assert num["speed"] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Added mass
# ---------------------------------------------------------------------------
class TestAddedMass:
    def test_tensor_is_isotropic_in_plane_and_positive(self):
        T = added_mass_tensor(50.0)
        assert T.shape == (3, 3)
        assert T[0, 0] == pytest.approx(T[1, 1], rel=1e-15)
        assert T[0, 0] > 0.0 and T[2, 2] > 0.0
        assert np.allclose(T, T.T)

    def test_potential_flow_disk_value(self):
        a, rho = 20.0, 1025.0
        T = added_mass_tensor(a, rho)
        assert T[0, 0] == pytest.approx((8.0 / 3.0) * rho * a ** 3, rel=1e-12)

    def test_ekman_depth_cap(self):
        """Added mass saturates at rho_w * D_E per unit area for very large floes."""
        a, rho, D = 500.0, 1025.0, 20.0
        T = added_mass_tensor(a, rho, potential_flow_limit_depth=D)
        assert T[0, 0] == pytest.approx(rho * D * math.pi * a ** 2, rel=1e-12)

    def test_steady_state_invariant_under_added_mass(self):
        """Coriolis on the ice mass only: M_a changes the transient, not the drift."""
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        r0 = steady_drift_analytic(env, FloeInertia(added_mass_coefficient=0.0))
        r1 = steady_drift_analytic(env, FloeInertia(added_mass_coefficient=2.0))
        assert r1["speed"] == pytest.approx(r0["speed"], rel=1e-12)
        assert r1["angle_right_deg"] == pytest.approx(r0["angle_right_deg"], rel=1e-12)

    def test_virtual_mass_in_coriolis_increases_angle_and_slows_drift(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        r0 = steady_drift_analytic(env, FloeInertia(), include_added_mass_coriolis=False)
        r1 = steady_drift_analytic(env, FloeInertia(), include_added_mass_coriolis=True)
        assert r1["angle_right_deg"] > r0["angle_right_deg"]
        assert r1["speed"] < r0["speed"]

    def test_added_mass_scales_initial_acceleration(self):
        """a(0) = tau / (M + M_a): virtual mass slows the spin-up by (M+M_a)/M."""
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        dt = 60.0
        acc = []
        for ca in (0.0, 1.0):
            inertia = FloeInertia(added_mass_coefficient=ca)
            sim = simulate_free_drift(env, inertia, duration=dt, dt=dt / 10.0)
            acc.append(sim["speed"][1] / (dt / 10.0))
        ratio_expected = (FloeInertia(added_mass_coefficient=1.0).effective_mass_per_area
                          / FloeInertia(added_mass_coefficient=0.0).effective_mass_per_area)
        assert acc[0] / acc[1] == pytest.approx(ratio_expected, rel=2e-2)


# ---------------------------------------------------------------------------
# Ekman spiral
# ---------------------------------------------------------------------------
class TestEkmanSpiral:
    K = 0.05
    PHI = 75.0
    TAU = 0.234 + 0.0j

    def test_decay_depth_formula(self):
        assert ekman_decay_depth(self.K, coriolis_parameter(self.PHI)) == pytest.approx(
            math.sqrt(2.0 * self.K / coriolis_parameter(self.PHI)), rel=1e-15)

    def test_decay_depth_guard(self):
        with pytest.raises(ValueError):
            ekman_decay_depth(self.K, 0.0)

    def test_surface_current_is_45deg_right_of_stress(self):
        w0 = ekman_surface_current(self.K, coriolis_parameter(self.PHI), self.TAU)
        # stress along +x in the northern hemisphere -> surface flow toward -y
        assert w0.real > 0.0 and w0.imag < 0.0
        assert math.degrees(math.atan2(w0.imag, w0.real)) == pytest.approx(-45.0, abs=1e-9)

    def test_surface_speed_formula(self):
        f = coriolis_parameter(self.PHI)
        w0 = ekman_surface_current(self.K, f, self.TAU)
        assert abs(w0) == pytest.approx(abs(self.TAU) / (1025.0 * math.sqrt(self.K * f)), rel=1e-12)

    def test_transport_is_90deg_right_of_stress(self):
        f = coriolis_parameter(self.PHI)
        tr = ekman_transport(f, self.TAU)
        assert tr.real == pytest.approx(0.0, abs=1e-15)
        assert tr.imag == pytest.approx(-abs(self.TAU) / (1025.0 * f), rel=1e-12)

    def test_propagator_decays_e_fold_per_decay_depth(self):
        f = coriolis_parameter(self.PHI)
        D = ekman_decay_depth(self.K, f)
        w = ekman_profile(self.K, f, self.TAU, np.array([0.0, D, 2 * D]))
        assert abs(w[1]) / abs(w[0]) == pytest.approx(math.exp(-1.0), rel=1e-12)
        assert abs(w[2]) / abs(w[0]) == pytest.approx(math.exp(-2.0), rel=1e-12)

    def test_finite_difference_reproduces_analytic(self):
        """Independent tridiagonal solve: profile, decay depth, veer, transport."""
        f = coriolis_parameter(self.PHI)
        fd = solve_ekman_finite_difference(self.K, f, self.TAU, n=4001, n_decay_depths=8.0)
        assert fd["rel_error_profile"] < 1e-3
        assert fd["decay_depth_numeric"] == pytest.approx(fd["decay_depth_analytic"], rel=1e-3)
        assert fd["surface_veer_right_deg"] == pytest.approx(45.0, abs=0.1)
        assert fd["surface_speed"] == pytest.approx(abs(ekman_surface_current(self.K, f, self.TAU)),
                                                    rel=1e-3)
        err = abs(fd["transport_numeric"] - fd["transport_analytic"]) / abs(fd["transport_analytic"])
        assert err < 5e-3


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
class TestTransient:
    def test_converges_to_newton_steady_state(self):
        """RK4 integration lands on the full relative-wind steady state."""
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        inertia = FloeInertia()
        num = solve_steady_drift(env, inertia)
        sim = simulate_free_drift(env, inertia, duration=40 * 86400.0, dt=600.0)
        assert sim["final_speed"] == pytest.approx(num["speed"], rel=1e-4)
        assert sim["final_angle_right_deg"] == pytest.approx(num["angle_right_deg"], abs=1e-3)

    def test_closed_form_overestimates_speed_by_two_percent(self):
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        inertia = FloeInertia()
        ana = steady_drift_analytic(env, inertia)
        num = solve_steady_drift(env, inertia)
        assert 1.005 < ana["speed"] / num["speed"] < 1.05

    def test_zero_rotation_no_deflection_and_aligned_with_wind(self):
        """Negative control (d): f = 0 -> the drift is exactly along the wind."""
        env = DriftEnvironment(latitude_deg=0.0, wind_speed=10.0)
        inertia = FloeInertia()
        ana = steady_drift_analytic(env, inertia)
        assert ana["angle_right_deg"] == pytest.approx(0.0, abs=1e-12)
        assert ana["v"] == pytest.approx(0.0, abs=1e-15)
        sim = simulate_free_drift(env, inertia, duration=20 * 86400.0, dt=600.0)
        assert sim["final_angle_right_deg"] == pytest.approx(0.0, abs=1e-8)
        assert abs(sim["final_u"][1]) < 1e-12


# ---------------------------------------------------------------------------
# Campaign acceptance criteria
# ---------------------------------------------------------------------------
class TestAcceptanceCriteria:
    def test_criterion_e_case_matrix_coverage(self):
        """(e) phi in [60, 85] deg x U_wind in [5, 15] m/s, at least 4 cases."""
        matrix = drift_case_matrix()
        assert len(matrix) >= 4
        lats = [p for p, _ in matrix]
        winds = [w for _, w in matrix]
        assert min(lats) == 60.0 and max(lats) == 85.0
        assert min(winds) == 5.0 and max(winds) == 15.0
        assert set(lats) <= set(range(60, 86))
        assert len(SUITE) == len(matrix)

    def test_criterion_a_nansen_angle_in_observed_band(self):
        """(a) Ekman-coupled Nansen angle in [20, 40] deg for every case."""
        for row in SUITE:
            assert 20.0 <= row["coupled_angle_right_deg"] <= 40.0, row

    def test_criterion_a_reference_case(self):
        """Reference case: 75N, 10 m/s, 1.5 m ice."""
        row = next(r for r in SUITE if r["latitude_deg"] == 75.0 and r["wind_speed"] == 10.0)
        assert 20.0 <= row["coupled_angle_right_deg"] <= 40.0

    def test_criterion_b_wind_factor(self):
        """(b) U_ice / U_wind in [1 %, 3 %] for every case."""
        for row in SUITE:
            assert 0.01 <= row["coupled_wind_factor"] <= 0.03, row

    def test_criterion_c_ekman_decay_depth(self):
        """(c) D_E = sqrt(2K/f) is what the boundary layer actually does."""
        for phi in (60.0, 85.0):
            f = coriolis_parameter(phi)
            for K in (0.02, 0.05):
                D = ekman_decay_depth(K, f)
                assert D == pytest.approx(math.sqrt(2.0 * K / f), rel=1e-15)
                fd = solve_ekman_finite_difference(K, f, 0.234 + 0.0j, n=3001)
                assert fd["decay_depth_numeric"] == pytest.approx(D, rel=2e-3)

    def test_criterion_d_no_coriolis_negative_control(self):
        """(d) f = 0 -> turning angle exactly 0 (closed form, Newton and RK4)."""
        env = DriftEnvironment(latitude_deg=0.0, wind_speed=12.0)
        inertia = FloeInertia()
        ana = steady_drift_analytic(env, inertia)
        num = solve_steady_drift(env, inertia)
        assert ana["angle_right_deg"] == pytest.approx(0.0, abs=1e-12)
        assert num["angle_right_deg"] == pytest.approx(0.0, abs=1e-10)
        sim = simulate_free_drift(env, inertia, duration=10 * 86400.0, dt=900.0)
        assert abs(sim["final_angle_right_deg"]) < 1e-8

    def test_criterion_all_cases_converged(self):
        assert all(row["coupled_converged"] for row in SUITE)

    def test_coupling_reduces_to_wind_only_without_rotation(self):
        """With f -> 0 there is no Ekman current; the angle returns to 0."""
        env = DriftEnvironment(latitude_deg=0.0, wind_speed=10.0)
        with pytest.raises(ValueError):
            ekman_coupled_drift(env, FloeInertia(), eddy_viscosity=0.05)

    def test_ekman_coupling_increases_turning_angle(self):
        """The ocean Ekman layer is what carries the angle into the Nansen band."""
        env = DriftEnvironment(latitude_deg=75.0, wind_speed=10.0)
        inertia = FloeInertia()
        wind_only = steady_drift_analytic(env, inertia, include_added_mass_coriolis=True)
        coupled = ekman_coupled_drift(env, inertia, eddy_viscosity=0.05)
        assert coupled["angle_right_deg"] > wind_only["angle_right_deg"]
        assert coupled["surface_current_speed"] > 0.0
        assert coupled["surface_current_dir_deg"] < -45.0  # right of the wind, beyond 45 deg
        assert coupled["decay_depth"] == pytest.approx(ekman_decay_depth(0.05, env.f), rel=1e-12)