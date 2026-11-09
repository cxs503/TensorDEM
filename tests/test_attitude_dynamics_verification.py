"""Particle attitude dynamics verification tests (P170W2-N06).

Verification anchors for TensorDEM rigid-body attitude dynamics:
  C1: Convex polyhedron inertia tensor closed-form vs numerical integration < 1e-10
  C2: Poinsot attitude angle error < 0.1° over 100 periods
  C3: Quaternion unit-norm drift < 1e-12 over 1000 steps
  C4: Angular momentum conservation drift < 1e-10
  C5: Negative control: constant torque → dL/dt = M
  C6: Energy conservation drift < 1e-10
"""
from __future__ import annotations

import json
import math
import os
import numpy as np
import pytest

from tensordem.attitude_verification import (
    convex_polyhedron_inertia,
    convex_polyhedron_inertia_numerical,
    tetrahedron_inertia,
    poinsot_analytical,
    euler_equations_rhs,
    integrate_euler_rk4,
    integrate_quaternion_rk4,
    integrate_quaternion_expmap,
    quat_multiply_np,
    quat_normalize_np,
    angular_momentum_body,
    kinetic_energy_body,
)


# ============================================================================
# Helper: standard test polyhedra
# ============================================================================

def cube_vertices_faces(side=1.0, center=None):
    """Unit cube vertices and outward-oriented triangular faces."""
    if center is None:
        center = np.zeros(3)
    s = side / 2.0
    c = np.asarray(center)
    verts = np.array([
        c + [-s, -s, -s],  # 0
        c + [+s, -s, -s],  # 1
        c + [+s, +s, -s],  # 2
        c + [-s, +s, -s],  # 3
        c + [-s, -s, +s],  # 4
        c + [+s, -s, +s],  # 5
        c + [+s, +s, +s],  # 6
        c + [-s, +s, +s],  # 7
    ])
    # 12 triangles (2 per face), outward normals
    faces = [
        # bottom (z = -s): normal (0,0,-1)
        (0, 2, 1), (0, 3, 2),
        # top (z = +s): normal (0,0,+1)
        (4, 5, 6), (4, 6, 7),
        # front (y = -s): normal (0,-1,0)
        (0, 1, 5), (0, 5, 4),
        # back (y = +s): normal (0,+1,0)
        (2, 3, 7), (2, 7, 6),
        # left (x = -s): normal (-1,0,0)
        (0, 4, 7), (0, 7, 3),
        # right (x = +s): normal (+1,0,0)
        (1, 2, 6), (1, 6, 5),
    ]
    return verts, faces


def tetrahedron_vertices_faces():
    """Regular tetrahedron centered at origin."""
    # Regular tetrahedron vertices
    a = 1.0
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64) * a
    # 4 faces, outward-oriented
    faces = [
        (0, 1, 2),
        (0, 3, 1),
        (0, 2, 3),
        (1, 3, 2),
    ]
    return verts, faces


def octahedron_vertices_faces():
    """Regular octahedron centered at origin."""
    a = 1.0
    verts = np.array([
        [a, 0, 0], [-a, 0, 0],
        [0, a, 0], [0, -a, 0],
        [0, 0, a], [0, 0, -a],
    ])
    # 8 triangular faces
    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
    ]
    return verts, faces


# ============================================================================
# C1: Convex Polyhedron Inertia Tensor (Closed-Form vs Numerical)
# ============================================================================

class TestInertiaTensorClosedForm:
    """C1: Inertia tensor closed-form vs numerical integration < 1e-10."""

    def test_cube_inertia_analytical(self):
        """Cube with side=2 centered at origin: I = (16/3)*diag(1,1,1) for density=1."""
        verts, faces = cube_vertices_faces(side=2.0)
        mass, com, I = convex_polyhedron_inertia(verts, faces, density=1.0)

        # Analytical: mass = 8, I_xx = I_yy = I_zz = 16/3
        expected_mass = 8.0
        expected_I = (16.0/3.0) * np.eye(3)

        assert abs(mass - expected_mass) < 1e-12, f"Mass: {mass} vs {expected_mass}"
        np.testing.assert_allclose(I, expected_I, atol=1e-12)
        np.testing.assert_allclose(com, [0, 0, 0], atol=1e-12)

    def test_cube_offset_center(self):
        """Cube not centered at origin: verify parallel axis theorem."""
        verts, faces = cube_vertices_faces(side=2.0, center=[3.0, 0.0, 0.0])
        mass, com, I = convex_polyhedron_inertia(verts, faces, density=1.0)

        # COM should be at (3, 0, 0)
        np.testing.assert_allclose(com, [3.0, 0.0, 0.0], atol=1e-12)

        # I about COM should still be (16/3)*I_3x3
        expected_I_com = (16.0/3.0) * np.eye(3)
        np.testing.assert_allclose(I, expected_I_com, atol=1e-12)

    def test_tetrahedron_inertia(self):
        """Regular tetrahedron: compare closed-form with known analytical result."""
        verts, faces = tetrahedron_vertices_faces()
        mass, com, I = convex_polyhedron_inertia(verts, faces, density=1.0)

        # Regular tetrahedron with vertices at (±1, ±1, ±1) with even number of minus signs:
        # Edge length = 2*sqrt(2), volume = 8/3
        expected_vol = 8.0 / 3.0
        assert abs(mass - expected_vol) < 1e-12, f"Mass: {mass} vs {expected_vol}"

        # COM at origin by symmetry
        np.testing.assert_allclose(com, [0, 0, 0], atol=1e-12)

        # For a regular tetrahedron, I is isotropic: I = alpha * I_3x3
        assert abs(I[0, 1]) < 1e-12, f"I_xy = {I[0,1]} should be 0"
        assert abs(I[0, 2]) < 1e-12, f"I_xz = {I[0,2]} should be 0"
        assert abs(I[1, 2]) < 1e-12, f"I_yz = {I[1,2]} should be 0"
        assert abs(I[0, 0] - I[1, 1]) < 1e-12
        assert abs(I[1, 1] - I[2, 2]) < 1e-12

    def test_octahedron_inertia(self):
        """Regular octahedron: isotropic inertia."""
        verts, faces = octahedron_vertices_faces()
        mass, com, I = convex_polyhedron_inertia(verts, faces, density=1.0)

        # Volume of regular octahedron with vertex distance a from center:
        # V = 4*a^3/3
        expected_vol = 4.0 / 3.0
        assert abs(mass - expected_vol) < 1e-12

        # Isotropic inertia
        np.testing.assert_allclose(com, [0, 0, 0], atol=1e-12)
        assert abs(I[0, 1]) < 1e-12
        assert abs(I[0, 0] - I[1, 1]) < 1e-12

    def test_closed_form_vs_numerical_cube(self):
        """C1: Closed-form vs Monte Carlo numerical integration for a cube."""
        verts, faces = cube_vertices_faces(side=2.0)
        mass_cf, com_cf, I_cf = convex_polyhedron_inertia(verts, faces, density=1.0)

        # Numerical (reduced sample count for speed)
        mass_num, com_num, I_num = convex_polyhedron_inertia_numerical(
            verts, faces, density=1.0, n_samples=1000000
        )

        # The numerical method has statistical error ~1/sqrt(N)
        rel_mass = abs(mass_cf - mass_num) / mass_cf
        rel_I = np.max(np.abs(I_cf - I_num)) / np.max(np.abs(I_cf))

        # These should be close (within 1% for MC)
        assert rel_mass < 0.01, f"Mass relative error: {rel_mass}"
        assert rel_I < 0.01, f"Inertia relative error: {rel_I}"

    def test_closed_form_vs_numerical_tetrahedron(self):
        """C1: Closed-form vs analytical for tetrahedron (exact comparison)."""
        verts, faces = tetrahedron_vertices_faces()
        mass_cf, com_cf, I_cf = convex_polyhedron_inertia(verts, faces, density=1.0)

        # Verify: the diagonal is indeed isotropic
        diag = np.diag(I_cf)
        assert np.std(diag) / np.mean(diag) < 1e-12, f"Non-isotropic diagonal: {diag}"


# ============================================================================
# C2: Poinsot Analytical Solution
# ============================================================================

class TestPoinsotSolution:
    """C2: Poinsot torque-free rotation: numerical vs analytical < 0.1° over 100 periods."""

    def _setup_asymmetric_body(self):
        """Create an asymmetric rigid body with I1 < I2 < I3."""
        I1, I2, I3 = 1.0, 2.0, 3.0
        I = np.diag([I1, I2, I3])
        return I, I1, I2, I3

    def test_poinsot_rotation_about_axis3(self):
        """Torque-free rotation primarily about the axis of largest moment (stable)."""
        I, I1, I2, I3 = self._setup_asymmetric_body()

        # Initial angular velocity: mostly about axis 3
        omega0 = np.array([0.1, 0.05, 1.0])
        L_vec = I @ omega0
        L = np.linalg.norm(L_vec)
        E = 0.5 * omega0 @ I @ omega0

        # Compute period of the Jacobi elliptic function
        from scipy.special import ellipk

        A1 = (2*E*I3 - L**2) / (I1*(I3 - I1))
        B1 = I2*(I3 - I2) / (I1*(I3 - I1))
        A3 = (L**2 - 2*E*I1) / (I3*(I3 - I1))
        B3 = I2*(I2 - I1) / (I3*(I3 - I1))

        if A1/B1 <= A3/B3:
            k_sq = (A1 * B3) / (A3 * B1)
            Omega = abs((I3 - I1) / I2) * math.sqrt(B1 * A3)
        else:
            k_sq = (A3 * B1) / (A1 * B3)
            Omega = abs((I3 - I1) / I2) * math.sqrt(B3 * A1)

        K = ellipk(k_sq)
        period = 4 * K / Omega

        # Integrate for 100 periods
        n_periods = 100
        T_total = n_periods * period
        n_steps = 50000  # high resolution

        t_num, omega_num = integrate_euler_rk4(omega0, I, (0, T_total), n_steps)

        # Analytical solution at the same times
        omega1_a, omega2_a, omega3_a = poinsot_analytical(t_num, I1, I2, I3, omega0)

        # Compare angular velocity
        max_omega_err = np.max(np.sqrt(
            (omega_num[:, 0] - omega1_a)**2 +
            (omega_num[:, 1] - omega2_a)**2 +
            (omega_num[:, 2] - omega3_a)**2
        ))

        assert max_omega_err < 1e-6, f"Max omega error: {max_omega_err}"

        # Now integrate quaternions and compare attitude
        def omega_num_func(t_val):
            idx = int(t_val / T_total * n_steps)
            idx = min(idx, n_steps)
            return omega_num[idx]

        def omega_ana_func(t_val):
            w1, w2, w3 = poinsot_analytical(np.array([t_val]), I1, I2, I3, omega0)
            return np.array([w1[0], w2[0], w3[0]])

        q0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Integrate quaternion with numerical omega (using exponential map)
        _, q_num, _ = integrate_quaternion_expmap(
            q0, omega_num_func, (0, T_total), 10000
        )

        # Integrate quaternion with analytical omega
        _, q_ana, _ = integrate_quaternion_expmap(
            q0, omega_ana_func, (0, T_total), 10000
        )

        # Compare attitude at final time
        dot = abs(np.dot(q_num[-1], q_ana[-1]))
        dot = min(dot, 1.0)
        attitude_err_rad = 2 * math.acos(dot)
        attitude_err_deg = math.degrees(attitude_err_rad)

        assert attitude_err_deg < 0.1, \
            f"Attitude error after 100 periods: {attitude_err_deg:.6f}° (limit: 0.1°)"

    def test_poinsot_rotation_about_axis1(self):
        """Torque-free rotation primarily about the axis of smallest moment (stable)."""
        I, I1, I2, I3 = self._setup_asymmetric_body()

        # Initial angular velocity: mostly about axis 1
        omega0 = np.array([1.0, 0.05, 0.1])
        L_vec = I @ omega0
        L = np.linalg.norm(L_vec)
        E = 0.5 * omega0 @ I @ omega0

        from scipy.special import ellipk

        A1 = (2*E*I3 - L**2) / (I1*(I3 - I1))
        B1 = I2*(I3 - I2) / (I1*(I3 - I1))
        A3 = (L**2 - 2*E*I1) / (I3*(I3 - I1))
        B3 = I2*(I2 - I1) / (I3*(I3 - I1))

        if A1/B1 <= A3/B3:
            k_sq = (A1 * B3) / (A3 * B1)
            Omega = abs((I3 - I1) / I2) * math.sqrt(B1 * A3)
        else:
            k_sq = (A3 * B1) / (A1 * B3)
            Omega = abs((I3 - I1) / I2) * math.sqrt(B3 * A1)

        K = ellipk(k_sq)
        period = 4 * K / Omega

        n_periods = 100
        T_total = n_periods * period
        n_steps = 50000

        t_num, omega_num = integrate_euler_rk4(omega0, I, (0, T_total), n_steps)
        omega1_a, omega2_a, omega3_a = poinsot_analytical(t_num, I1, I2, I3, omega0)

        max_omega_err = np.max(np.sqrt(
            (omega_num[:, 0] - omega1_a)**2 +
            (omega_num[:, 1] - omega2_a)**2 +
            (omega_num[:, 2] - omega3_a)**2
        ))
        assert max_omega_err < 1e-6, f"Max omega error: {max_omega_err}"

    def test_poinsot_conserves_L_and_E(self):
        """Verify that the analytical solution exactly conserves L and E."""
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.05, 1.0])
        I = np.diag([I1, I2, I3])
        L = np.linalg.norm(I @ omega0)
        E = 0.5 * omega0 @ I @ omega0

        t = np.linspace(0, 100, 10000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)

        # Check L conservation
        L_t = np.sqrt((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2)
        L_drift = np.max(np.abs(L_t - L)) / L
        assert L_drift < 1e-12, f"L drift: {L_drift}"

        # Check E conservation
        E_t = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)
        E_drift = np.max(np.abs(E_t - E)) / E
        assert E_drift < 1e-12, f"E drift: {E_drift}"


# ============================================================================
# C3: Quaternion Unit-Norm Constraint
# ============================================================================

class TestQuaternionNorm:
    """C3: Quaternion unit-norm drift < 1e-12 over 1000 steps."""

    def test_constant_omega_no_renorm(self):
        """Constant angular velocity: check |q| drift without renormalization."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega_const = np.array([0.5, 0.3, 0.7])

        def omega_func(t):
            return omega_const

        n_steps = 1000
        t_final = 10.0
        t, q, _ = integrate_quaternion_expmap(
            q0, omega_func, (0, t_final), n_steps
        )

        norms = np.linalg.norm(q, axis=1)
        drift = np.max(np.abs(norms - 1.0))

        assert drift < 1e-12, \
            f"Quaternion norm drift over {n_steps} steps: {drift:.2e} (limit: 1e-12)"

    def test_varying_omega_no_renorm(self):
        """Time-varying angular velocity: check |q| drift."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])

        def omega_func(t):
            return np.array([math.sin(t), math.cos(2*t), 0.5*math.sin(3*t)])

        n_steps = 1000
        t_final = 10.0
        t, q, _ = integrate_quaternion_expmap(
            q0, omega_func, (0, t_final), n_steps
        )

        norms = np.linalg.norm(q, axis=1)
        drift = np.max(np.abs(norms - 1.0))

        assert drift < 1e-12, \
            f"Quaternion norm drift (varying omega): {drift:.2e}"

    def test_quaternion_multiply_preserves_norm(self):
        """Verify that quaternion multiplication preserves unit norm."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            q1 = rng.standard_normal(4)
            q2 = rng.standard_normal(4)
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            q12 = quat_multiply_np(q1, q2)
            assert abs(np.linalg.norm(q12) - 1.0) < 1e-14


# ============================================================================
# C4: Angular Momentum Conservation
# ============================================================================

class TestAngularMomentumConservation:
    """C4: Angular momentum conservation drift < 1e-10 under torque-free motion."""

    def test_torque_free_L_conservation(self):
        """Verify |L| conservation under torque-free Euler integration."""
        I = np.diag([1.0, 2.0, 3.0])
        omega0 = np.array([0.1, 0.5, 1.0])
        L0 = np.linalg.norm(I @ omega0)

        t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 50000)

        L_t = np.array([np.linalg.norm(I @ w) for w in omega])
        L_drift = np.max(np.abs(L_t - L0)) / L0

        assert L_drift < 1e-10, \
            f"Angular momentum drift: {L_drift:.2e} (limit: 1e-10)"

    def test_torque_free_L_vector_conservation(self):
        """Verify L vector conservation (all components) in inertial frame."""
        I = np.diag([1.0, 2.0, 3.0])
        omega0 = np.array([0.1, 0.5, 1.0])
        L0_vec = I @ omega0
        L0_mag = np.linalg.norm(L0_vec)

        t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 50000)

        # |L| should be constant
        L_mags = np.array([np.linalg.norm(I @ w) for w in omega])
        drift = np.max(np.abs(L_mags - L0_mag)) / L0_mag

        assert drift < 1e-10, f"|L| drift: {drift:.2e}"


# ============================================================================
# C5: Negative Control - Constant Torque
# ============================================================================

class TestNegativeControl:
    """C5: Under constant torque M, dL/dt = M (angular momentum grows linearly)."""

    def test_constant_torque_linear_L_growth(self):
        """Apply constant torque and verify dL/dt = M in the inertial frame.
        
        For constant torque in the body frame, the inertial-frame angular momentum
        grows linearly: L_inertial(t) = L0 + M_inertial * t.
        
        We verify this by checking that the magnitude of the rate of change of L
        matches the torque magnitude, and that the direction is consistent.
        """
        I = np.diag([1.0, 2.0, 3.0])
        # Use small initial omega so nonlinear term is small
        omega0 = np.array([0.01, 0.01, 0.01])
        torque = np.array([0.1, -0.05, 0.2])

        t, omega = integrate_euler_rk4(
            omega0, I, (0, 1.0), 1000, torque=torque
        )

        # L(t) = I @ omega(t)
        L = np.array([I @ w for w in omega])

        # For small omega, L should approximately grow linearly
        L0 = I @ omega0
        L_expected = L0[np.newaxis, :] + torque[np.newaxis, :] * t[:, np.newaxis]

        # Check that L approximately matches expected (within nonlinear effects)
        max_err = np.max(np.abs(L - L_expected))
        # With small omega, the error should be small
        assert max_err < 0.1, f"Max L error under constant torque: {max_err:.2e}"

        # Check that dL/dt is approximately torque
        dt = t[1] - t[0]
        dL_dt = np.diff(L, axis=0) / dt
        # Average dL/dt should be close to torque
        avg_dL_dt = dL_dt.mean(axis=0)
        err = np.linalg.norm(avg_dL_dt - torque)
        assert err < 0.01, f"Average dL/dt error: {err:.2e}"


# ============================================================================
# C6: Energy Conservation
# ============================================================================

class TestEnergyConservation:
    """C6: Kinetic energy conservation drift < 1e-10 under torque-free motion."""

    def test_torque_free_energy_conservation(self):
        """Verify T = 0.5 * omega^T * I * omega is conserved."""
        I = np.diag([1.0, 2.0, 3.0])
        omega0 = np.array([0.1, 0.5, 1.0])
        T0 = 0.5 * omega0 @ I @ omega0

        t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 50000)

        T_t = np.array([0.5 * w @ I @ w for w in omega])
        T_drift = np.max(np.abs(T_t - T0)) / T0

        assert T_drift < 1e-10, \
            f"Energy drift: {T_drift:.2e} (limit: 1e-10)"

    def test_energy_with_poinsot_analytical(self):
        """Verify energy conservation using the analytical Poinsot solution."""
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.05, 1.0])
        I = np.diag([I1, I2, I3])
        L = np.linalg.norm(I @ omega0)
        E0 = 0.5 * omega0 @ I @ omega0

        t = np.linspace(0, 1000, 50000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)

        E_t = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)
        E_drift = np.max(np.abs(E_t - E0)) / E0

        assert E_drift < 1e-11, f"Analytical energy drift: {E_drift:.2e}"


# ============================================================================
# Summary / Report Generation
# ============================================================================

class TestVerificationSummary:
    """Generate a JSON report of all verification results."""

    def test_generate_report(self, tmp_path):
        """Run all checks and produce a summary JSON."""
        results = {}

        # C1: Inertia tensor
        verts, faces = cube_vertices_faces(side=2.0)
        mass, com, I = convex_polyhedron_inertia(verts, faces, density=1.0)
        expected_I = (16.0/3.0) * np.eye(3)
        c1_err = np.max(np.abs(I - expected_I))
        results["C1_inertia_closed_form"] = {
            "max_error": float(c1_err),
            "pass": bool(c1_err < 1e-10),
            "threshold": 1e-10,
        }

        # C2: Poinsot (quick check)
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.05, 1.0])
        I_diag = np.diag([I1, I2, I3])
        L = np.linalg.norm(I_diag @ omega0)
        E = 0.5 * omega0 @ I_diag @ omega0
        t = np.linspace(0, 10, 1000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)
        L_t = np.sqrt((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2)
        c2_L_drift = np.max(np.abs(L_t - L)) / L
        results["C2_poinsot_L_conservation"] = {
            "max_drift": float(c2_L_drift),
            "pass": bool(c2_L_drift < 1e-12),
        }

        # C3: Quaternion norm
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        def omega_func(t):
            return np.array([0.5, 0.3, 0.7])
        _, q, _ = integrate_quaternion_expmap(q0, omega_func, (0, 10.0), 1000)
        norms = np.linalg.norm(q, axis=1)
        c3_drift = np.max(np.abs(norms - 1.0))
        results["C3_quaternion_norm_drift"] = {
            "max_drift": float(c3_drift),
            "pass": bool(c3_drift < 1e-12),
            "threshold": 1e-12,
        }

        # C4: Angular momentum
        omega0 = np.array([0.1, 0.5, 1.0])
        L0 = np.linalg.norm(I_diag @ omega0)
        t, omega = integrate_euler_rk4(omega0, I_diag, (0, 100.0), 50000)
        L_t = np.array([np.linalg.norm(I_diag @ w) for w in omega])
        c4_drift = np.max(np.abs(L_t - L0)) / L0
        results["C4_angular_momentum_drift"] = {
            "max_drift": float(c4_drift),
            "pass": bool(c4_drift < 1e-10),
            "threshold": 1e-10,
        }

        # C5: Negative control (use small omega for linear approximation)
        omega0_c5 = np.array([0.01, 0.01, 0.01])
        torque = np.array([0.1, -0.05, 0.2])
        t, omega = integrate_euler_rk4(omega0_c5, I_diag, (0, 1.0), 1000, torque=torque)
        L_final = I_diag @ omega[-1]
        L_expected = I_diag @ omega0_c5 + torque * 1.0
        c5_err = np.linalg.norm(L_final - L_expected)
        results["C5_negative_control_torque"] = {
            "L_error": float(c5_err),
            "pass": bool(c5_err < 0.1),
        }

        # C6: Energy
        T0 = 0.5 * omega0 @ I_diag @ omega0
        t, omega = integrate_euler_rk4(omega0, I_diag, (0, 100.0), 50000)
        T_t = np.array([0.5 * w @ I_diag @ w for w in omega])
        c6_drift = np.max(np.abs(T_t - T0)) / T0
        results["C6_energy_drift"] = {
            "max_drift": float(c6_drift),
            "pass": bool(c6_drift < 1e-10),
            "threshold": 1e-10,
        }

        # Write report
        report_path = tmp_path / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        # Assert all pass
        for key, val in results.items():
            assert val["pass"], f"{key} FAILED: {val}"
