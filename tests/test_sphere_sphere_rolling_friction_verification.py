"""Sphere–sphere rolling-friction verification (P27W2-N06).

Validates three rolling-friction mechanisms on equal-radius sphere pairs:
  (1) Cundall-Strack tangential spring  F_t = k_t * delta_t  +  Coulomb limit
  (2) Rolling resistance moment  M_r = mu_r * R_eff * F_n
  (3) Rotational damping  omega(t) = omega_0 * exp(-c_r * t / I)

Machine-verifiable acceptance criteria
---------------------------------------
C1  Tangential spring slope k_t recovery ≤ 1 %
C2  Coulomb limit |F_t| ≤ mu_s * F_n hard assertion (10 cases)
C3  Rolling resistance moment error ≤ 2 % (9 cases)
C4  Rotational decay R² ≥ 0.999
C5  Rolling–sliding transition error ≤ 3 %
C6  Incline stability critical angle error ≤ 5 %
C7  Energy conservation: e → γ_n → e round-trip residual ≤ 1e-10
C8  mu_r = 0 → M_r = 0 (error ≤ 1e-12)
C9  m5052 regression all green  (covered by full-suite run)
C10 pytest ≥ 18 all green       (covered by full-suite run)

References
----------
Cundall & Strack (1979), Geotechnique 29, 47-65.
Ai et al. (2011), Powder Technology 206, 269-282.
Iwashita & Oda (1998), J. Eng. Mech. 124, 285-292.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.contact_tangential import (
    advance_tangential_contact,
    compute_tangential_force,
    damping_to_restitution,
    integrate_tangential_displacement,
    project_tangential_displacement,
    restitution_to_damping,
    saturation_displacement,
    tangential_cycle_dissipated_work,
)
from tensordem.rolling_resistance import (
    RollingResistanceConfig,
    SphereInclineConfig,
    critical_incline_angle_deg,
    fit_acceleration,
    incline_rolling_acceleration,
    pure_rolling_feasible,
    rolling_component,
    rolling_resistance_torque,
    simulate_sphere_on_incline,
)


# =====================================================================
# C1: Cundall-Strack tangential spring slope recovery ≤ 1%
# =====================================================================
class TestC1_TangentialSpringSlope(unittest.TestCase):
    """Verify F_t = k_t * delta_t in the linear (below-Coulomb) regime."""

    def test_slope_recovery_sweep(self):
        """Sweep k_t over 3 decades; recover slope from F_t vs delta_t."""
        k_n_values = [1e3, 1e4, 1e5]
        for k_t in k_n_values:
            mu = 0.5
            F_n = 100.0
            # Coulomb limit displacement: delta_s = mu * F_n / k_t
            delta_s = mu * F_n / k_t
            # Use displacements well below the limit (10% of delta_s)
            deltas = torch.linspace(0.001 * delta_s, 0.1 * delta_s, 20, dtype=torch.float64)
            forces = []
            for d in deltas:
                normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
                tang_disp = torch.tensor([[0.0, d.item()]], dtype=torch.float64)
                F_t, _ = compute_tangential_force(
                    normal, tang_disp, torch.tensor([F_n]), k_t, mu
                )
                forces.append(float(F_t.norm(dim=1).item()))
            forces_t = torch.tensor(forces, dtype=torch.float64)
            # Linear fit: F = k * delta
            slope = (deltas * forces_t).sum() / (deltas ** 2).sum()
            rel_err = abs(float(slope) - k_t) / k_t
            self.assertLess(rel_err, 0.01,
                            msg=f"k_t={k_t}: slope={float(slope):.6e}, rel_err={rel_err:.4e}")

    def test_advance_tangential_contact_slope(self):
        """advance_tangential_contact: F_t vs accumulated slip gives k_t."""
        k_t = 5000.0
        mu = 0.8
        F_n = 200.0
        dt = 1e-5
        v_t = 0.1  # tangential velocity
        normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        rel_vel = torch.tensor([[0.0, v_t, 0.0]], dtype=torch.float64)
        disp = torch.zeros(1, 3, dtype=torch.float64)
        forces = []
        slips = []
        for _ in range(50):
            F_t, disp, _ = advance_tangential_contact(
                disp, rel_vel, normal, torch.tensor([F_n]), k_t, mu, dt
            )
            slip_mag = float(torch.linalg.vector_norm(disp, dim=1).item())
            f_mag = float(torch.linalg.vector_norm(F_t, dim=1).item())
            forces.append(f_mag)
            slips.append(slip_mag)
        # Linear regime check
        slips_t = torch.tensor(slips[:20], dtype=torch.float64)
        forces_t = torch.tensor(forces[:20], dtype=torch.float64)
        slope = (slips_t * forces_t).sum() / (slips_t ** 2).sum()
        rel_err = abs(float(slope) - k_t) / k_t
        self.assertLess(rel_err, 0.01,
                        msg=f"advance slope={float(slope):.6e}, rel_err={rel_err:.4e}")


# =====================================================================
# C2: Coulomb limit hard assertion (10 cases)
# =====================================================================
class TestC2_CoulombLimit(unittest.TestCase):
    """|F_t| ≤ mu_s * F_n must hold for 10 parameter combinations."""

    def test_coulomb_limit_10_cases(self):
        cases = [
            (1000.0, 0.3, 50.0),
            (5000.0, 0.5, 100.0),
            (1e4, 0.1, 200.0),
            (2e4, 0.8, 10.0),
            (1e5, 0.2, 500.0),
            (500.0, 0.9, 1.0),
            (1e3, 0.4, 1000.0),
            (3e4, 0.6, 50.0),
            (8e3, 0.15, 300.0),
            (2e5, 0.7, 25.0),
        ]
        for k_t, mu, F_n in cases:
            # Large displacement to ensure saturation
            normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
            tang_disp = torch.tensor([[0.0, 10.0]], dtype=torch.float64)
            F_t, _ = compute_tangential_force(
                normal, tang_disp, torch.tensor([F_n]), k_t, mu
            )
            f_mag = float(F_t.norm(dim=1).item())
            limit = mu * F_n
            # Allow small floating-point tolerance (relative 1e-6)
            self.assertLessEqual(f_mag, limit * (1 + 1e-6),
                                 msg=f"k_t={k_t}, mu={mu}, F_n={F_n}: "
                                     f"|F_t|={f_mag:.6e} > limit={limit:.6e}")
            # Should be at the limit (saturated)
            self.assertAlmostEqual(f_mag, limit, delta=limit * 1e-5,
                                   msg=f"Not saturated: f_mag={f_mag}, limit={limit}")


# =====================================================================
# C3: Rolling resistance moment error ≤ 2% (9 cases)
# =====================================================================
class TestC3_RollingResistanceMoment(unittest.TestCase):
    """M_r = mu_r * R_eff * F_n for the CDT (saturated) model."""

    def test_cdt_moment_9_cases(self):
        cases = [
            (0.05, 0.01, 10.0),
            (0.10, 0.02, 50.0),
            (0.20, 0.005, 100.0),
            (0.01, 0.05, 200.0),
            (0.50, 0.01, 5.0),
            (0.03, 0.03, 500.0),
            (0.15, 0.008, 1000.0),
            (0.08, 0.015, 25.0),
            (0.25, 0.02, 80.0),
        ]
        for mu_r, R, F_n in cases:
            config = RollingResistanceConfig(model="cdt", mu_r=mu_r)
            R_eff = torch.tensor([R], dtype=torch.float64)
            f_n = torch.tensor([F_n], dtype=torch.float64)
            normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
            # Use rolling component perpendicular to normal
            omega_roll = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
            torque, _ = rolling_resistance_torque(
                omega_roll, f_n, R_eff, config, normal=normal
            )
            m_mag = float(torch.linalg.vector_norm(torque, dim=1).item())
            m_expected = mu_r * R * F_n
            rel_err = abs(m_mag - m_expected) / m_expected
            self.assertLess(rel_err, 0.02,
                            msg=f"mu_r={mu_r}, R={R}, F_n={F_n}: "
                                f"M_r={m_mag:.6e}, expected={m_expected:.6e}, err={rel_err:.4e}")


# =====================================================================
# C4: Rotational decay R² ≥ 0.999
# =====================================================================
class TestC4_RotationalDecay(unittest.TestCase):
    """Free-rotating sphere with viscous rolling resistance:
    omega(t) = omega_0 * exp(-gamma_r * t / I)
    where gamma_r = mu_r * F_n * R_eff / omega_ref.
    """

    def test_exponential_decay_r_squared(self):
        mu_r = 0.01  # small to get long decay time
        R = 0.05  # larger sphere
        F_n = 10.0  # smaller force
        omega_ref = 1.0
        mass = 4.0 / 3.0 * math.pi * R**3 * 900.0
        I = 0.4 * mass * R**2
        gamma_r = mu_r * F_n * R / omega_ref  # viscous damping coeff
        tau = I / gamma_r  # decay time constant

        config = RollingResistanceConfig(model="viscous", mu_r=mu_r, omega_ref=omega_ref)
        R_eff = torch.tensor([R], dtype=torch.float64)
        f_n = torch.tensor([F_n], dtype=torch.float64)
        normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

        omega_0 = 5.0  # rad/s
        omega_vec = torch.tensor([[omega_0, 0.0, 0.0]], dtype=torch.float64)
        # Use dt << tau for accurate integration
        dt = tau / 100
        n_steps = int(5 * tau / dt)  # 5 time constants

        omega_history = [omega_0]
        time_history = [0.0]

        for step in range(n_steps):
            torque, _ = rolling_resistance_torque(
                omega_vec, f_n, R_eff, config, normal=normal
            )
            # Angular acceleration: alpha = torque / I
            alpha = torque / I
            omega_vec = omega_vec + alpha * dt
            w = float(omega_vec[0, 0].item())
            omega_history.append(w)
            time_history.append((step + 1) * dt)

        omega_t = torch.tensor(omega_history, dtype=torch.float64)
        time_t = torch.tensor(time_history, dtype=torch.float64)

        # Analytic: omega(t) = omega_0 * exp(-t / tau)
        omega_analytic = omega_0 * torch.exp(-time_t / tau)

        # R² between simulated and analytic
        ss_res = ((omega_t - omega_analytic) ** 2).sum()
        ss_tot = ((omega_t - omega_t.mean()) ** 2).sum()
        r_squared = 1.0 - ss_res / ss_tot
        self.assertGreater(float(r_squared), 0.999,
                           msg=f"R²={float(r_squared):.6f}")


# =====================================================================
# C5: Rolling-sliding transition error ≤ 3%
# =====================================================================
class TestC5_RollingSlidingTransition(unittest.TestCase):
    """Critical condition: pure rolling feasible when
    (I/(mR²) * tan(θ) + mu_r) / (1 + I/(mR²)) ≤ mu
    """

    def test_transition_critical_angle(self):
        mu_r = 0.1
        inertia_factor = 0.4
        for mu in [0.3, 0.5, 0.7, 0.9]:
            # Critical angle: (0.4 * tan(θ_c) + mu_r) / 1.4 = mu
            # tan(θ_c) = (mu * 1.4 - mu_r) / 0.4
            tan_tc = (mu * (1 + inertia_factor) - mu_r) / inertia_factor
            theta_c_analytic = math.degrees(math.atan(tan_tc))

            # Check just below and above
            theta_below = theta_c_analytic - 0.5
            theta_above = theta_c_analytic + 0.5
            self.assertTrue(pure_rolling_feasible(theta_below, mu_r, mu,
                                                   inertia_factor=inertia_factor))
            self.assertFalse(pure_rolling_feasible(theta_above, mu_r, mu,
                                                    inertia_factor=inertia_factor))

            # Verify the transition point within 3%
            # Binary search for the actual transition
            lo, hi = theta_below, theta_above
            for _ in range(50):
                mid = (lo + hi) / 2
                if pure_rolling_feasible(mid, mu_r, mu, inertia_factor=inertia_factor):
                    lo = mid
                else:
                    hi = mid
            theta_c_numerical = (lo + hi) / 2
            rel_err = abs(theta_c_numerical - theta_c_analytic) / theta_c_analytic
            self.assertLess(rel_err, 0.03,
                            msg=f"mu={mu}: theta_c_num={theta_c_numerical:.4f}, "
                                f"theta_c_an={theta_c_analytic:.4f}, err={rel_err:.4e}")


# =====================================================================
# C6: Incline stability critical angle error ≤ 5%
# =====================================================================
class TestC6_InclineStability(unittest.TestCase):
    """theta_c = arctan(mu_r); simulation should confirm stability below,
    instability above."""

    def test_critical_angle_formula(self):
        for mu_r in [0.01, 0.05, 0.1, 0.2, 0.3]:
            theta_c = critical_incline_angle_deg(mu_r)
            theta_c_expected = math.degrees(math.atan(mu_r))
            rel_err = abs(theta_c - theta_c_expected) / max(theta_c_expected, 1e-15)
            self.assertLess(rel_err, 0.05,
                            msg=f"mu_r={mu_r}: theta_c={theta_c:.4f}, "
                                f"expected={theta_c_expected:.4f}")

    def test_simulation_confirms_stability(self):
        """Sub-critical angle: sphere stays at rest."""
        mu_r = 0.1
        theta_c = critical_incline_angle_deg(mu_r)
        result = simulate_sphere_on_incline(
            [0.5 * theta_c],
            RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=1.0),
            config=SphereInclineConfig(),
            duration=0.06,
            settle_time=0.02,
        )
        R = SphereInclineConfig().radius
        drift = abs(float(result["x"][-1, 0])) / R
        self.assertLess(drift, 1e-3,
                        msg=f"Sub-critical drift={drift:.3e} R")

    def test_simulation_confirms_instability(self):
        """Super-critical angle: sphere accelerates downhill."""
        mu_r = 0.1
        theta_c = critical_incline_angle_deg(mu_r)
        result = simulate_sphere_on_incline(
            [1.5 * theta_c],
            RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=1.0),
            config=SphereInclineConfig(),
            duration=0.06,
            settle_time=0.02,
        )
        accel, r2 = fit_acceleration(result)
        expected_a = incline_rolling_acceleration(1.5 * theta_c, mu_r)
        self.assertGreater(float(accel[0]), 0.0,
                           msg="Should accelerate downhill")
        rel_err = abs(float(accel[0]) - expected_a) / expected_a
        self.assertLess(rel_err, 0.05,
                        msg=f"accel={float(accel[0]):.5f}, expected={expected_a:.5f}")


# =====================================================================
# C7: Energy conservation (analytic round-trip)
# =====================================================================
class TestC7_EnergyConservation(unittest.TestCase):
    """Energy conservation via analytic restitution-damping mapping.

    The spring-dashpot collision model conserves energy according to:
        e = exp(-π * ζ / sqrt(1 - ζ²))
    where ζ = γ_n / (2 * sqrt(k_n * m_eff)).

    Round-trip: e → γ_n → e should recover the original e within 1e-10.
    """

    def test_restitution_round_trip(self):
        """e → γ_n → e round-trip for multiple restitution coefficients."""
        for e_target in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            k_n = 1e5
            m_eff = 0.5
            gamma_n = restitution_to_damping(e_target, k_n, m_eff)
            e_back = damping_to_restitution(gamma_n, k_n, m_eff)
            residual = abs(e_back - e_target)
            self.assertLess(residual, 1e-10,
                            msg=f"e={e_target}: e_back={e_back:.15f}, residual={residual:.6e}")

    def test_tangential_friction_dissipation(self):
        """Friction cycle: dissipated work matches analytic formula."""
        mu = 0.3
        F_n = 100.0
        k_t = 5000.0
        amplitude = 0.05  # well above shakedown

        W_analytic = tangential_cycle_dissipated_work(mu, F_n, k_t, amplitude)
        a_stick = mu * F_n / k_t
        self.assertGreater(amplitude, a_stick)
        self.assertGreater(W_analytic, 0.0)
        # Analytic: W = 4 * mu * F_n * (a - mu * F_n / k_t)
        W_expected = 4 * mu * F_n * (amplitude - a_stick)
        self.assertAlmostEqual(W_analytic, W_expected, places=10)


# =====================================================================
# C8: mu_r = 0 → M_r = 0 (error ≤ 1e-12)
# =====================================================================
class TestC8_ZeroRollingResistance(unittest.TestCase):
    """When mu_r = 0, rolling resistance torque must be exactly zero."""

    def test_cdt_zero_mu_r(self):
        config = RollingResistanceConfig(model="cdt", mu_r=0.0)
        omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        f_n = torch.tensor([100.0], dtype=torch.float64)
        r = torch.tensor([0.01], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        torque, state = rolling_resistance_torque(omega, f_n, r, config, normal=normal)
        self.assertLess(float(torque.abs().max()), 1e-12)

    def test_viscous_zero_mu_r(self):
        config = RollingResistanceConfig(model="viscous", mu_r=0.0)
        omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        f_n = torch.tensor([100.0], dtype=torch.float64)
        r = torch.tensor([0.01], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        torque, state = rolling_resistance_torque(omega, f_n, r, config, normal=normal)
        self.assertLess(float(torque.abs().max()), 1e-12)

    def test_epsd_zero_mu_r(self):
        config = RollingResistanceConfig(model="epsd", mu_r=0.0, k_r=1.0)
        omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        f_n = torch.tensor([100.0], dtype=torch.float64)
        r = torch.tensor([0.01], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        torque, state = rolling_resistance_torque(
            omega, f_n, r, config, normal=normal, dt=1e-5
        )
        self.assertLess(float(torque.abs().max()), 1e-12)

    def test_epsd_zero_mu_r_state_stays_zero(self):
        """EPSD state must remain zero when mu_r=0."""
        config = RollingResistanceConfig(model="epsd", mu_r=0.0, k_r=1.0)
        omega = torch.tensor([[5.0, 3.0, 0.0]], dtype=torch.float64)
        f_n = torch.tensor([500.0], dtype=torch.float64)
        r = torch.tensor([0.02], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        state = torch.zeros(1, 3, dtype=torch.float64)
        for _ in range(100):
            torque, state = rolling_resistance_torque(
                omega, f_n, r, config, normal=normal,
                torque_state=state, dt=1e-5
            )
        self.assertLess(float(state.abs().max()), 1e-12)


# =====================================================================
# Additional: EPSD rolling spring verification
# =====================================================================
class TestEPSDRollingSpring(unittest.TestCase):
    """Verify the EPSD elastic-plastic spring-damper for rolling."""

    def test_epsd_linear_regime(self):
        """Below yield: M_r = k_r * theta_roll (linear spring)."""
        k_r = 0.1  # N*m/rad
        mu_r = 0.5  # high enough to not yield
        R = 0.01
        F_n = 100.0
        config = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
        R_eff = torch.tensor([R], dtype=torch.float64)
        f_n = torch.tensor([F_n], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

        omega_roll = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        dt = 1e-4
        state = torch.zeros(1, 3, dtype=torch.float64)
        # Small rotation: theta = omega * dt * n_steps
        n_steps = 10
        theta_total = 1.0 * dt * n_steps
        torque, state = rolling_resistance_torque(
            omega_roll, f_n, R_eff, config, normal=normal,
            torque_state=state, dt=dt
        )
        for _ in range(n_steps - 1):
            torque, state = rolling_resistance_torque(
                omega_roll, f_n, R_eff, config, normal=normal,
                torque_state=state, dt=dt
            )
        m_mag = float(torch.linalg.vector_norm(torque, dim=1).item())
        m_expected = k_r * theta_total
        # Should be in linear regime (below yield limit)
        yield_limit = mu_r * F_n * R
        self.assertLess(m_expected, yield_limit)
        rel_err = abs(m_mag - m_expected) / m_expected
        self.assertLess(rel_err, 0.02,
                        msg=f"EPSD linear: M={m_mag:.6e}, expected={m_expected:.6e}")

    def test_epsd_yield_clamp(self):
        """Above yield: |M_r| ≤ mu_r * F_n * R_eff."""
        k_r = 100.0  # very stiff
        mu_r = 0.01
        R = 0.01
        F_n = 100.0
        config = RollingResistanceConfig(model="epsd", mu_r=mu_r, k_r=k_r)
        R_eff = torch.tensor([R], dtype=torch.float64)
        f_n = torch.tensor([F_n], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        omega_roll = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)
        dt = 1e-3
        state = torch.zeros(1, 3, dtype=torch.float64)
        for _ in range(1000):
            torque, state = rolling_resistance_torque(
                omega_roll, f_n, R_eff, config, normal=normal,
                torque_state=state, dt=dt
            )
        m_mag = float(torch.linalg.vector_norm(torque, dim=1).item())
        m_limit = mu_r * F_n * R
        self.assertLessEqual(m_mag, m_limit + 1e-12)
        # Should be at the limit
        self.assertAlmostEqual(m_mag, m_limit, delta=m_limit * 0.01)


# =====================================================================
# Additional: Rolling component extraction
# =====================================================================
class TestRollingComponent(unittest.TestCase):
    """Verify that rolling_component strips the spin (drilling) component."""

    def test_pure_rolling_unchanged(self):
        omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        result = rolling_component(omega, normal)
        self.assertAlmostEqual(float(result[0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(result[0, 1]), 0.0, places=12)
        self.assertAlmostEqual(float(result[0, 2]), 0.0, places=12)

    def test_pure_spin_removed(self):
        omega = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        result = rolling_component(omega, normal)
        self.assertLess(float(result.norm()), 1e-12)

    def test_mixed(self):
        omega = torch.tensor([[3.0, 0.0, 4.0]], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        result = rolling_component(omega, normal)
        self.assertAlmostEqual(float(result[0, 0]), 3.0, places=12)
        self.assertAlmostEqual(float(result[0, 2]), 0.0, places=12)


# =====================================================================
# Additional: Saturation displacement
# =====================================================================
class TestSaturationDisplacement(unittest.TestCase):
    """delta_s = mu * F_n / k_t."""

    def test_formula(self):
        for mu, F_n, k_t in [(0.3, 100, 1000), (0.5, 200, 5000), (0.1, 50, 500)]:
            delta_s = saturation_displacement(torch.tensor([F_n], dtype=torch.float64), k_t, mu)
            expected = mu * F_n / k_t
            self.assertAlmostEqual(float(delta_s[0]), expected, places=12)


if __name__ == "__main__":
    unittest.main()
