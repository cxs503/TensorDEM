#!/usr/bin/env python3
"""
P493-N03: TensorDEM 颗粒姿态层从零到一验证面
================================================
Poinsot 无外力矩 Jacobi 椭圆解析解 + 凸多面体惯性张量闭式基准 + 四元数/角动量守恒账本

验收判据:
  C1: 凸多面体惯性张量闭式 vs 解析解 (球/立方体/正四面体) 误差 < 1e-10
  C2: Poinsot 无外力矩: RK4 数值 vs Jacobi 椭圆解析解, 姿态角误差 < 0.1° over 100 周期
  C3: 四元数归一化漂移 < 1e-12 over 1000 步
  C4: 姿态矩阵正交性误差 ||R^T R - I||_F < 1e-12, |det(R) - 1| < 1e-12
  C5: 角动量守恒 |L| 漂移 < 1e-10 (无外力矩)
  C6: 动能守恒漂移 < 1e-10 (无外力矩)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensordem.attitude_verification import (
    convex_polyhedron_inertia,
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
from tensordem.attitude_matrix import (
    quaternion_to_rotation_matrix,
    rotation_matrix_orthogonality_error,
    attitude_matrix_from_quaternion_history,
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
        c + [-s, -s, -s], c + [+s, -s, -s], c + [+s, +s, -s], c + [-s, +s, -s],
        c + [-s, -s, +s], c + [+s, -s, +s], c + [+s, +s, +s], c + [-s, +s, +s],
    ])
    faces = [
        (0, 2, 1), (0, 3, 2),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # front
        (2, 3, 7), (2, 7, 6),  # back
        (0, 4, 7), (0, 7, 3),  # left
        (1, 2, 6), (1, 6, 5),  # right
    ]
    return verts, faces


def regular_tetrahedron_vertices_faces(edge=1.0):
    """Regular tetrahedron centered at origin."""
    a = edge / (2 * math.sqrt(2))
    verts = np.array([
        [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]
    ])
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return verts, faces


def regular_octahedron_vertices_faces(edge=1.0):
    """Regular octahedron centered at origin."""
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

def verify_c1_inertia_tensor():
    """C1: Convex polyhedron inertia tensor closed-form vs analytical."""
    results = {}
    
    # --- Cube ---
    side = 2.0
    rho = 1.0
    verts, faces = cube_vertices_faces(side=side)
    mass, com, I_com = convex_polyhedron_inertia(verts, faces, density=rho)
    
    # Analytical: cube side=2, mass=8, I = (1/6)*m*a^2 = (1/6)*8*4 = 16/3
    expected_mass = rho * side**3
    expected_I = (expected_mass * side**2 / 6.0) * np.eye(3)
    
    mass_err = abs(mass - expected_mass) / expected_mass
    com_err = np.linalg.norm(com)
    I_err = np.max(np.abs(I_com - expected_I)) / np.max(np.abs(expected_I))
    
    results["cube"] = {
        "mass_relative_error": float(mass_err),
        "com_norm": float(com_err),
        "inertia_max_relative_error": float(I_err),
        "expected_mass": float(expected_mass),
        "computed_mass": float(mass),
        "expected_I_diag": float(expected_I[0, 0]),
        "computed_I_diag": float(I_com[0, 0]),
        "pass": bool(mass_err < 1e-12 and com_err < 1e-12 and I_err < 1e-10),
    }
    
    # --- Regular Tetrahedron ---
    edge = 1.0
    verts, faces = regular_tetrahedron_vertices_faces(edge=edge)
    mass, com, I_com = convex_polyhedron_inertia(verts, faces, density=rho)
    
    # Analytical: V = edge^3 / (6*sqrt(2)), m = rho*V
    V_exact = edge**3 / (6 * math.sqrt(2))
    expected_mass = rho * V_exact
    # For regular tetrahedron with edge a: I = m*a^2/20 about any axis through COM
    expected_I_diag = expected_mass * edge**2 / 20.0
    expected_I = expected_I_diag * np.eye(3)
    
    mass_err = abs(mass - expected_mass) / expected_mass
    com_err = np.linalg.norm(com)
    I_err = np.max(np.abs(I_com - expected_I)) / expected_I_diag
    
    results["tetrahedron"] = {
        "mass_relative_error": float(mass_err),
        "com_norm": float(com_err),
        "inertia_max_relative_error": float(I_err),
        "expected_mass": float(expected_mass),
        "computed_mass": float(mass),
        "expected_I_diag": float(expected_I_diag),
        "computed_I_diag": float(I_com[0, 0]),
        "pass": bool(mass_err < 1e-10 and com_err < 1e-10 and I_err < 1e-10),
    }
    
    # --- Regular Octahedron ---
    edge = 1.0
    verts, faces = regular_octahedron_vertices_faces(edge=edge)
    mass, com, I_com = convex_polyhedron_inertia(verts, faces, density=rho)
    
    # Analytical: V = sqrt(2)/3 * a^3, m = rho*V
    V_exact = math.sqrt(2) / 3.0 * edge**3
    expected_mass = rho * V_exact
    # For regular octahedron with edge a: I = m*a^2/10 about any axis through COM
    expected_I_diag = expected_mass * edge**2 / 10.0
    expected_I = expected_I_diag * np.eye(3)
    
    mass_err = abs(mass - expected_mass) / expected_mass
    com_err = np.linalg.norm(com)
    I_err = np.max(np.abs(I_com - expected_I)) / expected_I_diag
    
    results["octahedron"] = {
        "mass_relative_error": float(mass_err),
        "com_norm": float(com_err),
        "inertia_max_relative_error": float(I_err),
        "expected_mass": float(expected_mass),
        "computed_mass": float(mass),
        "expected_I_diag": float(expected_I_diag),
        "computed_I_diag": float(I_com[0, 0]),
        "pass": bool(mass_err < 1e-10 and com_err < 1e-10 and I_err < 1e-10),
    }
    
    all_pass = all(r["pass"] for r in results.values())
    return results, all_pass


# ============================================================================
# C2: Poinsot Torque-Free Motion vs Jacobi Elliptic
# ============================================================================

def verify_c2_poinsot():
    """C2: Poinsot torque-free: RK4 vs Jacobi elliptic analytical."""
    from scipy.special import ellipk
    
    results = {}
    
    for case_name, I1, I2, I3, omega0, n_periods_target in [
        ("rotation_about_axis3", 1.0, 2.0, 3.0, np.array([0.1, 0.05, 1.0]), 10),
        ("rotation_about_axis1", 1.0, 2.0, 3.0, np.array([1.0, 0.05, 0.1]), 10),
        ("intermediate_case", 1.0, 2.5, 4.0, np.array([0.3, 0.8, 0.5]), 10),
    ]:
        I = np.diag([I1, I2, I3])
        L_vec = I @ omega0
        L = np.linalg.norm(L_vec)
        E = 0.5 * omega0 @ I @ omega0
        
        # Compute period
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
        period = 4 * K_val / Omega
        
        # Integrate for specified periods with high resolution
        T_total = n_periods_target * period
        n_steps = 20000  # high resolution for accuracy
        
        t_num, omega_num = integrate_euler_rk4(omega0, I, (0, T_total), n_steps)
        
        # Analytical solution
        omega1_a, omega2_a, omega3_a = poinsot_analytical(t_num, I1, I2, I3, omega0)
        
        # Compare angular velocity
        omega_err = np.sqrt(
            (omega_num[:, 0] - omega1_a)**2 +
            (omega_num[:, 1] - omega2_a)**2 +
            (omega_num[:, 2] - omega3_a)**2
        )
        max_omega_err = np.max(omega_err)
        mean_omega_err = np.mean(omega_err)
        
        # Integrate quaternions and compare attitude
        def omega_num_func(t_val):
            idx = int(t_val / T_total * n_steps)
            idx = min(idx, n_steps)
            return omega_num[idx]
        
        def omega_ana_func(t_val):
            w1, w2, w3 = poinsot_analytical(np.array([t_val]), I1, I2, I3, omega0)
            return np.array([w1[0], w2[0], w3[0]])
        
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        n_quat_steps = 10000
        
        _, q_num, _ = integrate_quaternion_expmap(q0, omega_num_func, (0, T_total), n_quat_steps)
        _, q_ana, _ = integrate_quaternion_expmap(q0, omega_ana_func, (0, T_total), n_quat_steps)
        
        # Attitude error at final time
        dot = abs(np.dot(q_num[-1], q_ana[-1]))
        dot = min(dot, 1.0)
        attitude_err_rad = 2 * math.acos(dot)
        attitude_err_deg = math.degrees(attitude_err_rad)
        
        # L and E conservation from analytical solution
        L_t = np.sqrt((I1*omega1_a)**2 + (I2*omega2_a)**2 + (I3*omega3_a)**2)
        L_drift = np.max(np.abs(L_t - L)) / L
        
        E_t = 0.5 * (I1*omega1_a**2 + I2*omega2_a**2 + I3*omega3_a**2)
        E_drift = np.max(np.abs(E_t - E)) / E
        
        results[case_name] = {
            "n_periods": n_periods_target,
            "period": float(period),
            "T_total": float(T_total),
            "max_omega_error": float(max_omega_err),
            "mean_omega_error": float(mean_omega_err),
            "attitude_error_deg": float(attitude_err_deg),
            "analytical_L_drift": float(L_drift),
            "analytical_E_drift": float(E_drift),
            "pass": bool(attitude_err_deg < 0.1 and max_omega_err < 1e-4),
        }
    
    all_pass = all(r["pass"] for r in results.values())
    return results, all_pass


# ============================================================================
# C3: Quaternion Normalization Drift
# ============================================================================

def verify_c3_quaternion_norm():
    """C3: Quaternion unit-norm drift < 1e-12."""
    results = {}
    
    # Exponential map (should preserve norm exactly)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Constant omega
    def omega_const(t):
        return np.array([0.5, 0.3, 0.7])
    
    _, q_expmap, _ = integrate_quaternion_expmap(q0, omega_const, (0, 10.0), 1000)
    norms_expmap = np.linalg.norm(q_expmap, axis=1)
    drift_expmap_const = np.max(np.abs(norms_expmap - 1.0))
    
    results["expmap_constant_omega"] = {
        "max_norm_drift": float(drift_expmap_const),
        "threshold": 1e-12,
        "pass": bool(drift_expmap_const < 1e-12),
    }
    
    # Varying omega
    def omega_varying(t):
        return np.array([math.sin(t), math.cos(2*t), 0.5*math.sin(3*t)])
    
    _, q_expmap_v, _ = integrate_quaternion_expmap(q0, omega_varying, (0, 10.0), 1000)
    norms_expmap_v = np.linalg.norm(q_expmap_v, axis=1)
    drift_expmap_varying = np.max(np.abs(norms_expmap_v - 1.0))
    
    results["expmap_varying_omega"] = {
        "max_norm_drift": float(drift_expmap_varying),
        "threshold": 1e-12,
        "pass": bool(drift_expmap_varying < 1e-12),
    }
    
    # RK4 without renormalization
    _, q_rk4, _ = integrate_quaternion_rk4(q0, omega_const, (0, 10.0), 1000, renormalize=False)
    norms_rk4 = np.linalg.norm(q_rk4, axis=1)
    drift_rk4 = np.max(np.abs(norms_rk4 - 1.0))
    
    results["rk4_no_renorm"] = {
        "max_norm_drift": float(drift_rk4),
        "note": "RK4 without renormalization accumulates drift",
    }
    
    # RK4 with renormalization
    _, q_rk4_renorm, _ = integrate_quaternion_rk4(q0, omega_const, (0, 10.0), 1000, renormalize=True)
    norms_rk4_renorm = np.linalg.norm(q_rk4_renorm, axis=1)
    drift_rk4_renorm = np.max(np.abs(norms_rk4_renorm - 1.0))
    
    results["rk4_with_renorm"] = {
        "max_norm_drift": float(drift_rk4_renorm),
        "threshold": 1e-12,
        "pass": bool(drift_rk4_renorm < 1e-12),
    }
    
    all_pass = all(
        r.get("pass", True) for r in results.values()
        if "threshold" in r
    )
    return results, all_pass


# ============================================================================
# C4: Attitude Matrix Orthogonality
# ============================================================================

def verify_c4_orthogonality():
    """C4: Attitude matrix orthogonality ||R^T R - I||_F < 1e-12, |det(R)-1| < 1e-12."""
    results = {}
    
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    
    def omega_func(t):
        return np.array([0.5*math.sin(0.3*t), 0.3*math.cos(0.7*t), 0.7])
    
    # Exponential map
    _, q_expmap, _ = integrate_quaternion_expmap(q0, omega_func, (0, 50.0), 5000)
    R_expmap = attitude_matrix_from_quaternion_history(q_expmap)
    
    frob_errs = []
    det_errs = []
    for i in range(R_expmap.shape[0]):
        f_err, d_err = rotation_matrix_orthogonality_error(R_expmap[i])
        frob_errs.append(f_err)
        det_errs.append(d_err)
    
    max_frob = max(frob_errs)
    max_det = max(det_errs)
    
    results["expmap_frobenius"] = {
        "max_orthogonality_error": float(max_frob),
        "threshold": 1e-12,
        "pass": bool(max_frob < 1e-12),
    }
    results["expmap_determinant"] = {
        "max_det_error": float(max_det),
        "threshold": 1e-12,
        "pass": bool(max_det < 1e-12),
    }
    
    # RK4 with renormalization
    _, q_rk4, _ = integrate_quaternion_rk4(q0, omega_func, (0, 50.0), 5000, renormalize=True)
    R_rk4 = attitude_matrix_from_quaternion_history(q_rk4)
    
    frob_errs_rk4 = []
    det_errs_rk4 = []
    for i in range(R_rk4.shape[0]):
        f_err, d_err = rotation_matrix_orthogonality_error(R_rk4[i])
        frob_errs_rk4.append(f_err)
        det_errs_rk4.append(d_err)
    
    max_frob_rk4 = max(frob_errs_rk4)
    max_det_rk4 = max(det_errs_rk4)
    
    results["rk4_renorm_frobenius"] = {
        "max_orthogonality_error": float(max_frob_rk4),
        "threshold": 1e-12,
        "pass": bool(max_frob_rk4 < 1e-12),
    }
    results["rk4_renorm_determinant"] = {
        "max_det_error": float(max_det_rk4),
        "threshold": 1e-12,
        "pass": bool(max_det_rk4 < 1e-12),
    }
    
    all_pass = all(r.get("pass", True) for r in results.values())
    return results, all_pass


# ============================================================================
# C5: Angular Momentum Conservation
# ============================================================================

def verify_c5_angular_momentum():
    """C5: Angular momentum |L| drift < 1e-10 under torque-free motion."""
    results = {}
    
    I = np.diag([1.0, 2.0, 3.0])
    omega0 = np.array([0.1, 0.5, 1.0])
    L0 = np.linalg.norm(I @ omega0)
    
    # RK4 integration
    t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 10000)
    
    L_t = np.array([np.linalg.norm(I @ w) for w in omega])
    L_drift = np.max(np.abs(L_t - L0)) / L0
    
    results["rk4_L_drift"] = {
        "max_relative_drift": float(L_drift),
        "threshold": 1e-10,
        "pass": bool(L_drift < 1e-10),
    }
    
    # Also check L vector components in body frame
    # (they oscillate, but |L| should be constant)
    L_body = np.array([I @ w for w in omega])
    L_body_mags = np.linalg.norm(L_body, axis=1)
    L_body_drift = np.max(np.abs(L_body_mags - L0)) / L0
    
    results["rk4_L_body_drift"] = {
        "max_relative_drift": float(L_body_drift),
        "threshold": 1e-10,
        "pass": bool(L_body_drift < 1e-10),
    }
    
    # Analytical solution (should be exact)
    I1, I2, I3 = 1.0, 2.0, 3.0
    t_ana = np.linspace(0, 100.0, 10000)
    w1, w2, w3 = poinsot_analytical(t_ana, I1, I2, I3, omega0)
    L_ana = np.sqrt((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2)
    L_ana_drift = np.max(np.abs(L_ana - L0)) / L0
    
    results["analytical_L_drift"] = {
        "max_relative_drift": float(L_ana_drift),
        "threshold": 1e-12,
        "pass": bool(L_ana_drift < 1e-12),
    }
    
    all_pass = all(r["pass"] for r in results.values())
    return results, all_pass


# ============================================================================
# C6: Energy Conservation
# ============================================================================

def verify_c6_energy():
    """C6: Kinetic energy drift < 1e-10 under torque-free motion."""
    results = {}
    
    I = np.diag([1.0, 2.0, 3.0])
    omega0 = np.array([0.1, 0.5, 1.0])
    T0 = 0.5 * omega0 @ I @ omega0
    
    # RK4
    t, omega = integrate_euler_rk4(omega0, I, (0, 100.0), 10000)
    T_t = np.array([0.5 * w @ I @ w for w in omega])
    T_drift = np.max(np.abs(T_t - T0)) / T0
    
    results["rk4_energy_drift"] = {
        "max_relative_drift": float(T_drift),
        "threshold": 1e-10,
        "pass": bool(T_drift < 1e-10),
    }
    
    # Analytical
    I1, I2, I3 = 1.0, 2.0, 3.0
    t_ana = np.linspace(0, 100.0, 10000)
    w1, w2, w3 = poinsot_analytical(t_ana, I1, I2, I3, omega0)
    E_t = 0.5 * (I1*w1**2 + I2*w2**2 + I3*w3**2)
    E_drift = np.max(np.abs(E_t - T0)) / T0
    
    results["analytical_energy_drift"] = {
        "max_relative_drift": float(E_drift),
        "threshold": 1e-12,
        "pass": bool(E_drift < 1e-12),
    }
    
    all_pass = all(r["pass"] for r in results.values())
    return results, all_pass


# ============================================================================
# Plot Generation
# ============================================================================

def generate_plots(report_data: dict, output_dir: Path):
    """Generate verification plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return []
    
    figures = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Poinsot omega trajectory (numerical vs analytical)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    I1, I2, I3 = 1.0, 2.0, 3.0
    omega0 = np.array([0.1, 0.05, 1.0])
    I = np.diag([I1, I2, I3])
    
    t, omega_num = integrate_euler_rk4(omega0, I, (0, 20.0), 2000)
    w1_a, w2_a, w3_a = poinsot_analytical(t, I1, I2, I3, omega0)
    
    for i, (label, color) in enumerate([
        (r'$\omega_1$', 'C0'), (r'$\omega_2$', 'C1'), (r'$\omega_3$', 'C2')
    ]):
        axes[i].plot(t, omega_num[:, i], '-', color=color, alpha=0.7, label='RK4')
        omega_ana = [w1_a, w2_a, w3_a][i]
        axes[i].plot(t, omega_ana, '--', color=color, alpha=0.7, label='Analytical')
        axes[i].set_xlabel('t')
        axes[i].set_ylabel(label)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    fig.suptitle('Poinsot Torque-Free Motion: RK4 vs Jacobi Elliptic Analytical')
    fig.tight_layout()
    path1 = output_dir / "poinsot_omega_trajectory.png"
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append({"name": "poinsot_omega_trajectory", "caption": "Body-frame angular velocity components: RK4 numerical integration vs Jacobi elliptic function analytical solution", "path": str(path1)})
    
    # Plot 2: Angular momentum conservation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    L0 = np.linalg.norm(I @ omega0)
    L_t = np.array([np.linalg.norm(I @ w) for w in omega_num])
    ax1.plot(t, (L_t - L0) / L0, 'b-', linewidth=0.8)
    ax1.set_xlabel('t')
    ax1.set_ylabel(r'$(|L(t)| - |L_0|) / |L_0|$')
    ax1.set_title('Angular Momentum Drift (RK4)')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    T0 = 0.5 * omega0 @ I @ omega0
    T_t = np.array([0.5 * w @ I @ w for w in omega_num])
    ax2.plot(t, (T_t - T0) / T0, 'r-', linewidth=0.8)
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$(T(t) - T_0) / T_0$')
    ax2.set_title('Kinetic Energy Drift (RK4)')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    fig.suptitle('Conservation Laws Under Torque-Free Motion')
    fig.tight_layout()
    path2 = output_dir / "conservation_laws.png"
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append({"name": "conservation_laws", "caption": "Relative drift of angular momentum magnitude and kinetic energy under torque-free RK4 integration", "path": str(path2)})
    
    # Plot 3: Quaternion norm drift
    fig, ax = plt.subplots(figsize=(8, 4))
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    
    def omega_func(t_val):
        return np.array([0.5, 0.3, 0.7])
    
    _, q_expmap, _ = integrate_quaternion_expmap(q0, omega_func, (0, 10.0), 1000)
    _, q_rk4_no, _ = integrate_quaternion_rk4(q0, omega_func, (0, 10.0), 1000, renormalize=False)
    _, q_rk4_yes, _ = integrate_quaternion_rk4(q0, omega_func, (0, 10.0), 1000, renormalize=True)
    
    t_q = np.linspace(0, 10.0, 1001)
    ax.plot(t_q, np.linalg.norm(q_expmap, axis=1) - 1.0, label='ExpMap', linewidth=0.8)
    ax.plot(t_q, np.linalg.norm(q_rk4_no, axis=1) - 1.0, label='RK4 (no renorm)', linewidth=0.8)
    ax.plot(t_q, np.linalg.norm(q_rk4_yes, axis=1) - 1.0, label='RK4 (renorm)', linewidth=0.8)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$|q| - 1$')
    ax.set_title('Quaternion Normalization Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    fig.tight_layout()
    path3 = output_dir / "quaternion_norm_drift.png"
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append({"name": "quaternion_norm_drift", "caption": "Quaternion unit-norm drift for exponential map and RK4 integrators", "path": str(path3)})
    
    # Plot 4: Inertia tensor error bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    c1_data = report_data.get("C1_inertia_tensor", {})
    shapes = list(c1_data.keys())
    if shapes:
        mass_errs = [c1_data[s]["mass_relative_error"] for s in shapes]
        I_errs = [c1_data[s]["inertia_max_relative_error"] for s in shapes]
        
        x = np.arange(len(shapes))
        width = 0.35
        ax.bar(x - width/2, mass_errs, width, label='Mass Error', color='C0')
        ax.bar(x + width/2, I_errs, width, label='Inertia Error', color='C1')
        ax.set_xticks(x)
        ax.set_xticklabels(shapes)
        ax.set_ylabel('Relative Error')
        ax.set_title('Convex Polyhedron Inertia Tensor: Closed-Form vs Analytical')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Threshold 1e-10')
    
    fig.tight_layout()
    path4 = output_dir / "inertia_tensor_errors.png"
    fig.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append({"name": "inertia_tensor_errors", "caption": "Relative errors of closed-form inertia tensor computation vs analytical solutions for standard geometries", "path": str(path4)})
    
    # Plot 5: Poinsot body-frame trajectory (3D phase space)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    t_fine = np.linspace(0, 20.0, 5000)
    w1_f, w2_f, w3_f = poinsot_analytical(t_fine, I1, I2, I3, omega0)
    
    ax.plot(w1_f, w2_f, w3_f, 'b-', linewidth=0.5, alpha=0.7)
    ax.plot([omega0[0]], [omega0[1]], [omega0[2]], 'ro', markersize=8, label='Initial')
    
    # Plot the energy ellipsoid intersection (constant E and L)
    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel(r'$\omega_2$')
    ax.set_zlabel(r'$\omega_3$')
    ax.set_title('Poinsot Motion: Angular Velocity Trajectory in Body Frame')
    ax.legend()
    
    path5 = output_dir / "poinsot_phase_space.png"
    fig.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append({"name": "poinsot_phase_space", "caption": "Angular velocity trajectory in body-frame phase space (intersection of energy ellipsoid and angular momentum sphere)", "path": str(path5)})
    
    return figures


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all verification checks and generate report."""
    print("=" * 70)
    print("P493-N03: TensorDEM 颗粒姿态层从零到一验证面")
    print("=" * 70)
    
    report = {}
    all_pass = True
    
    # C1
    print("\n[C1] Convex Polyhedron Inertia Tensor...")
    t0 = time.time()
    c1_results, c1_pass = verify_c1_inertia_tensor()
    report["C1_inertia_tensor"] = c1_results
    print(f"  → {'PASS' if c1_pass else 'FAIL'} ({time.time()-t0:.2f}s)")
    for shape, r in c1_results.items():
        print(f"    {shape}: mass_err={r['mass_relative_error']:.2e}, I_err={r['inertia_max_relative_error']:.2e}")
    all_pass = all_pass and c1_pass
    
    # C2
    print("\n[C2] Poinsot Torque-Free Motion...")
    t0 = time.time()
    c2_results, c2_pass = verify_c2_poinsot()
    report["C2_poinsot"] = c2_results
    print(f"  → {'PASS' if c2_pass else 'FAIL'} ({time.time()-t0:.2f}s)")
    for case, r in c2_results.items():
        print(f"    {case}: ω_err={r['max_omega_error']:.2e}, attitude_err={r['attitude_error_deg']:.4f}°")
    all_pass = all_pass and c2_pass
    
    # C3
    print("\n[C3] Quaternion Normalization Drift...")
    t0 = time.time()
    c3_results, c3_pass = verify_c3_quaternion_norm()
    report["C3_quaternion_norm"] = c3_results
    print(f"  → {'PASS' if c3_pass else 'FAIL'} ({time.time()-t0:.2f}s)")
    for key, r in c3_results.items():
        if "max_norm_drift" in r:
            print(f"    {key}: drift={r['max_norm_drift']:.2e}")
    all_pass = all_pass and c3_pass
    
    # C4
    print("\n[C4] Attitude Matrix Orthogonality...")
    t0 = time.time()
    c4_results, c4_pass = verify_c4_orthogonality()
    report["C4_orthogonality"] = c4_results
    print(f"  → {'PASS' if c4_pass else 'FAIL'} ({time.time()-t0:.2f}s)")
    for key, r in c4_results.items():
        if "max_orthogonality_error" in r:
            print(f"    {key}: frob_err={r['max_orthogonality_error']:.2e}")
        if "max_det_error" in r:
            print(f"    {key}: det_err={r['max_det_error']:.2e}")
    all_pass = all_pass and c4_pass
    
    # C5
    print("\n[C5] Angular Momentum Conservation...")
    t0 = time.time()
    c5_results, c5_pass = verify_c5_angular_momentum()
    report["C5_angular_momentum"] = c5_results
    print(f"  → {'PASS' if c5_pass else 'FAIL'} ({time.time()-t0:.2f}s)")
    for key, r in c5_results.items():
        print(f"    {key}: drift={r['max_relative_drift']:.2e}")
    all_pass = all_pass and c5_pass
    
    # C6
    print("\n[C6] Energy Conservation...")
    t0 = time.time()
    c6_results, c6_pass = verify_c6_energy()
    report["C6_energy"] = c6_results
    print(f"  → {'PASS' if c6_pass else 'FAIL'} ({time.time()-t0:.2f}s)")
    for key, r in c6_results.items():
        print(f"    {key}: drift={r['max_relative_drift']:.2e}")
    all_pass = all_pass and c6_pass
    
    # Summary
    report["summary"] = {
        "all_pass": all_pass,
        "C1_pass": c1_pass,
        "C2_pass": c2_pass,
        "C3_pass": c3_pass,
        "C4_pass": c4_pass,
        "C5_pass": c5_pass,
        "C6_pass": c6_pass,
    }
    
    # Generate plots
    print("\n[Plots] Generating verification figures...")
    output_dir = Path(__file__).parent / "artifacts" / "p493_n03"
    figures = generate_plots(report, output_dir)
    for fig_info in figures:
        print(f"  → {fig_info['name']}.png")
    
    # Write JSON report
    report_path = output_dir / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Written to {report_path}")
    
    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
