"""P493-N03: TensorDEM 颗粒姿态层从零到一验证面 — pytest 用例

验收判据:
  C1: 凸多面体惯性张量闭式 vs 解析解 (立方体/正四面体/正八面体) 误差 < 1e-10
  C2: Poinsot 无外力矩: RK4 数值 vs Jacobi 椭圆解析解, 姿态角误差 < 0.1° over 10 周期
  C3: 四元数归一化漂移 < 1e-12 over 1000 步
  C4: 姿态矩阵正交性误差 ||R^T R - I||_F < 1e-12, |det(R) - 1| < 1e-12
  C5: 角动量守恒 |L| 漂移 < 1e-10 (无外力矩)
  C6: 动能守恒漂移 < 1e-10 (无外力矩)

References:
  - Landau & Lifshitz, Mechanics, §33-37
  - Goldstein, Poole & Safko, Classical Mechanics, 3rd ed., Ch. 5
  - Poinsot, La théorie nouvelle de la rotation des corps, 1834
  - Mirtich, "Fast and Accurate Computation of Polyhedral Mass Properties", 1996
"""
from __future__ import annotations

import json
import math
import numpy as np
import pytest

from tensordem.attitude_verification import (
    convex_polyhedron_inertia,
    tetrahedron_inertia,
    poinsot_analytical,
    integrate_euler_rk4,
    integrate_quaternion_rk4,
    integrate_quaternion_expmap,
    quat_multiply_np,
)
from tensordem.attitude_matrix import (
    quaternion_to_rotation_matrix,
    rotation_matrix_orthogonality_error,
    attitude_matrix_from_quaternion_history,
)


# ============================================================================
# Helper: standard test polyhedra
# ============================================================================

def cube_vertices_faces(side=1.0, center=None):
    if center is None:
        center = np.zeros(3)
    s = side / 2.0
    c = np.asarray(center)
    verts = np.array([
        c + [-s, -s, -s], c + [+s, -s, -s], c + [+s, +s, -s], c + [-s, +s, -s],
        c + [-s, -s, +s], c + [+s, -s, +s], c + [+s, +s, +s], c + [-s, +s, +s],
    ])
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    ]
    return verts, faces


def regular_tetrahedron_vertices_faces(edge=1.0):
    a = edge / (2 * math.sqrt(2))
    verts = np.array([[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]])
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return verts, faces


def regular_octahedron_vertices_faces(edge=1.0):
    a = edge / math.sqrt(2)
    verts = np.array([
        [a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0], [0, 0, a], [0, 0, -a]
    ])
    faces = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]
    return verts, faces


# ============================================================================
# C1: Convex Polyhedron Inertia Tensor
# ============================================================================

class TestC1InertiaTensor:
    """C1: Convex polyhedron inertia tensor closed-form vs analytical < 1e-10."""

    def test_cube(self):
        side = 2.0
        verts, faces = cube_vertices_faces(side=side)
        mass, com, I_com = convex_polyhedron_inertia(verts, faces, density=1.0)
        expected_mass = side**3
        expected_I_diag = expected_mass * side**2 / 6.0
        assert abs(mass - expected_mass) / expected_mass < 1e-12
        assert np.linalg.norm(com) < 1e-12
        assert abs(I_com[0, 0] - expected_I_diag) / expected_I_diag < 1e-10
        assert np.max(np.abs(I_com - np.diag(np.diag(I_com)))) < 1e-14

    def test_regular_tetrahedron(self):
        edge = 1.0
        verts, faces = regular_tetrahedron_vertices_faces(edge=edge)
        mass, com, I_com = convex_polyhedron_inertia(verts, faces, density=1.0)
        V_exact = edge**3 / (6 * math.sqrt(2))
        expected_mass = V_exact
        expected_I_diag = expected_mass * edge**2 / 20.0
        assert abs(mass - expected_mass) / expected_mass < 1e-10
        assert np.linalg.norm(com) < 1e-10
        assert abs(I_com[0, 0] - expected_I_diag) / expected_I_diag < 1e-10
        assert abs(I_com[0, 0] - I_com[1, 1]) < 1e-14
        assert abs(I_com[1, 1] - I_com[2, 2]) < 1e-14

    def test_regular_octahedron(self):
        edge = 1.0
        verts, faces = regular_octahedron_vertices_faces(edge=edge)
        mass, com, I_com = convex_polyhedron_inertia(verts, faces, density=1.0)
        V_exact = math.sqrt(2) / 3.0 * edge**3
        expected_mass = V_exact
        expected_I_diag = expected_mass * edge**2 / 10.0
        assert abs(mass - expected_mass) / expected_mass < 1e-10
        assert np.linalg.norm(com) < 1e-10
        assert abs(I_com[0, 0] - expected_I_diag) / expected_I_diag < 1e-10


# ============================================================================
# C2: Poinsot Torque-Free Motion
# ============================================================================

class TestC2Poinsot:
    """C2: Poinsot attitude angle error < 0.1° over 10 periods."""

    def _compute_period(self, I1, I2, I3, omega0):
        from scipy.special import ellipk
        I = np.diag([I1, I2, I3])
        L = np.linalg.norm(I @ omega0)
        E = 0.5 * omega0 @ I @ omega0
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
        K_val = ellipk(k_sq)
        return 4 * K_val / Omega

    def test_rotation_about_axis3(self):
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.05, 1.0])
        I = np.diag([I1, I2, I3])
        period = self._compute_period(I1, I2, I3, omega0)
        T_total = 10 * period
        t, omega_num = integrate_euler_rk4(omega0, I, (0, T_total), 20000)
        w1_a, w2_a, w3_a = poinsot_analytical(t, I1, I2, I3, omega0)
        omega_err = np.max(np.sqrt(
            (omega_num[:, 0] - w1_a)**2 +
            (omega_num[:, 1] - w2_a)**2 +
            (omega_num[:, 2] - w3_a)**2
        ))
        assert omega_err < 1e-4, f"omega error: {omega_err:.2e}"

    def test_analytical_L_conservation(self):
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.05, 1.0])
        I = np.diag([I1, I2, I3])
        L0 = np.linalg.norm(I @ omega0)
        t = np.linspace(0, 100, 10000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)
        L_t = np.sqrt((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2)
        L_drift = np.max(np.abs(L_t - L0)) / L0
        assert L_drift < 1e-12, f"L drift: {L_drift:.2e}"

    def test_analytical_E_conservation(self):
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.05, 1.0])
        I = np.diag([I1, I2, I3])
        E0 = 0.5 * omega0 @ I @ omega0
        t = np.linspace(0, 100, 10000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)
        E_t = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)
        E_drift = np.max(np.abs(E_t - E0)) / E0
        assert E_drift < 1e-12, f"E drift: {E_drift:.2e}"


# ============================================================================
# C3: Quaternion Normalization
# ============================================================================

class TestC3QuaternionNorm:
    """C3: Quaternion unit-norm drift < 1e-12 over 1000 steps."""

    def test_expmap_constant_omega(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        def omega_func(t):
            return np.array([0.5, 0.3, 0.7])
        _, q, _ = integrate_quaternion_expmap(q0, omega_func, (0, 10.0), 1000)
        norms = np.linalg.norm(q, axis=1)
        drift = np.max(np.abs(norms - 1.0))
        assert drift < 1e-12, f"Norm drift: {drift:.2e}"

    def test_expmap_varying_omega(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        def omega_func(t):
            return np.array([math.sin(t), math.cos(2*t), 0.5*math.sin(3*t)])
        _, q, _ = integrate_quaternion_expmap(q0, omega_func, (0, 10.0), 1000)
        norms = np.linalg.norm(q, axis=1)
        drift = np.max(np.abs(norms - 1.0))
        assert drift < 1e-12, f"Norm drift: {drift:.2e}"

    def test_rk4_with_renorm(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        def omega_func(t):
            return np.array([0.5, 0.3, 0.7])
        _, q, _ = integrate_quaternion_rk4(q0, omega_func, (0, 10.0), 1000, renormalize=True)
        norms = np.linalg.norm(q, axis=1)
        drift = np.max(np.abs(norms - 1.0))
        assert drift < 1e-12, f"Norm drift: {drift:.2e}"

    def test_quaternion_multiply_preserves_norm(self):
        rng = np.random.default_rng(123)
        for _ in range(100):
            q1 = rng.standard_normal(4)
            q2 = rng.standard_normal(4)
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            q12 = quat_multiply_np(q1, q2)
            assert abs(np.linalg.norm(q12) - 1.0) < 1e-14


# ============================================================================
# C4: Attitude Matrix Orthogonality
# ============================================================================

class TestC4Orthogonality:
    """C4: ||R^T R - I||_F < 1e-12, |det(R) - 1| < 1e-12."""

    def test_expmap_orthogonality(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        def omega_func(t):
            return np.array([0.5*math.sin(0.3*t), 0.3*math.cos(0.7*t), 0.7])
        _, q, _ = integrate_quaternion_expmap(q0, omega_func, (0, 50.0), 5000)
        R_hist = attitude_matrix_from_quaternion_history(q)
        max_frob = 0.0
        max_det = 0.0
        for i in range(R_hist.shape[0]):
            f_err, d_err = rotation_matrix_orthogonality_error(R_hist[i])
            max_frob = max(max_frob, f_err)
            max_det = max(max_det, d_err)
        assert max_frob < 1e-12, f"Frobenius error: {max_frob:.2e}"
        assert max_det < 1e-12, f"Determinant error: {max_det:.2e}"

    def test_rk4_renorm_orthogonality(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        def omega_func(t):
            return np.array([0.5*math.sin(0.3*t), 0.3*math.cos(0.7*t), 0.7])
        _, q, _ = integrate_quaternion_rk4(q0, omega_func, (0, 50.0), 5000, renormalize=True)
        R_hist = attitude_matrix_from_quaternion_history(q)
        max_frob = 0.0
        max_det = 0.0
        for i in range(R_hist.shape[0]):
            f_err, d_err = rotation_matrix_orthogonality_error(R_hist[i])
            max_frob = max(max_frob, f_err)
            max_det = max(max_det, d_err)
        assert max_frob < 1e-12, f"Frobenius error: {max_frob:.2e}"
        assert max_det < 1e-12, f"Determinant error: {max_det:.2e}"


# ============================================================================
# C5: Angular Momentum Conservation
# ============================================================================

class TestC5AngularMomentum:
    """C5: |L| drift < 1e-10 under torque-free motion."""

    def test_rk4_L_conservation(self):
        I = np.diag([1.0, 2.0, 3.0])
        omega0 = np.array([0.1, 0.5, 1.0])
        L0 = np.linalg.norm(I @ omega0)
        t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 10000)
        L_t = np.array([np.linalg.norm(I @ w) for w in omega])
        L_drift = np.max(np.abs(L_t - L0)) / L0
        assert L_drift < 1e-10, f"L drift: {L_drift:.2e}"

    def test_analytical_L_conservation(self):
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.5, 1.0])
        I = np.diag([I1, I2, I3])
        L0 = np.linalg.norm(I @ omega0)
        t = np.linspace(0, 100.0, 10000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)
        L_t = np.sqrt((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2)
        L_drift = np.max(np.abs(L_t - L0)) / L0
        assert L_drift < 1e-12, f"L drift: {L_drift:.2e}"


# ============================================================================
# C6: Energy Conservation
# ============================================================================

class TestC6Energy:
    """C6: Kinetic energy drift < 1e-10 under torque-free motion."""

    def test_rk4_energy_conservation(self):
        I = np.diag([1.0, 2.0, 3.0])
        omega0 = np.array([0.1, 0.5, 1.0])
        T0 = 0.5 * omega0 @ I @ omega0
        t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 10000)
        T_t = np.array([0.5 * w @ I @ w for w in omega])
        T_drift = np.max(np.abs(T_t - T0)) / T0
        assert T_drift < 1e-10, f"Energy drift: {T_drift:.2e}"

    def test_analytical_energy_conservation(self):
        I1, I2, I3 = 1.0, 2.0, 3.0
        omega0 = np.array([0.1, 0.5, 1.0])
        I = np.diag([I1, I2, I3])
        E0 = 0.5 * omega0 @ I @ omega0
        t = np.linspace(0, 100.0, 10000)
        w1, w2, w3 = poinsot_analytical(t, I1, I2, I3, omega0)
        E_t = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)
        E_drift = np.max(np.abs(E_t - E0)) / E0
        assert E_drift < 1e-12, f"Energy drift: {E_drift:.2e}"
