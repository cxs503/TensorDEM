"""Particle attitude dynamics verification layer for TensorDEM (P170W2-N06).

This module provides from-zero-to-one verification anchors for rigid-body
attitude dynamics:

1. **Convex polyhedron inertia tensor (closed-form)**
   Analytical computation of the inertia tensor for a convex polyhedron given
   its vertex set, using the divergence theorem to reduce volume integrals to
   sums over tetrahedra formed with the origin.

2. **Poinsot torque-free rotation (Jacobi elliptic functions)**
   The exact analytical solution of Euler's equations for torque-free rigid
   body rotation, expressed in terms of Jacobi elliptic functions sn/cn/dn.

3. **Quaternion integration**
   Verification of the unit-quaternion constraint under the kinematic ODE
   dq/dt = 0.5 * q (x) omega.

4. **Angular momentum & energy conservation**
   Verification that |L| = |I*omega| and T = 0.5*omega^T*I*omega are
   conserved under torque-free motion.

References:
    - Landau & Lifshitz, Mechanics, §33-37
    - Goldstein, Poole & Safko, Classical Mechanics, 3rd ed., Ch. 5
    - Poinsot, La théorie nouvelle de la rotation des corps, 1834
    - Van Eetvelde, "Rigid body dynamics and quaternions", 2012
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc

__all__ = [
    "convex_polyhedron_inertia",
    "tetrahedron_inertia",
    "poinsot_analytical",
    "euler_equations_rhs",
    "integrate_euler_rk4",
    "integrate_quaternion_rk4",
    "integrate_quaternion_expmap",
    "quat_multiply_np",
    "quat_normalize_np",
    "angular_momentum_body",
    "kinetic_energy_body",
]


# ============================================================================
# 1. Convex Polyhedron Inertia Tensor (Closed-Form)
# ============================================================================

def tetrahedron_inertia(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
    density: float = 1.0
) -> Tuple[float, np.ndarray]:
    """Compute mass and inertia tensor of a tetrahedron with vertices v0..v3.

    Uses the closed-form formulas from:
    - Mirtich, "Fast and Accurate Computation of Polyhedral Mass Properties",
      J. Graphics Tools, 1(2), 1996.

    The inertia is computed about the origin; use parallel axis theorem to
    shift to center of mass.

    Parameters
    ----------
    v0, v1, v2, v3 : array_like, shape (3,)
        Vertices of the tetrahedron.
    density : float
        Uniform mass density.

    Returns
    -------
    mass : float
        Mass of the tetrahedron.
    I : ndarray, shape (3, 3)
        Inertia tensor about the origin.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v3 = np.asarray(v3, dtype=np.float64)

    # Edges from v0
    a = v1 - v0
    b = v2 - v0
    c = v3 - v0

    # Volume = |det([a, b, c])| / 6
    vol = abs(np.dot(a, np.cross(b, c))) / 6.0
    mass = density * vol

    verts = np.array([v0, v1, v2, v3])
    V = vol  # geometric volume (unsigned)

    # S1_i = sum of vertex coordinates
    S1 = verts.sum(axis=0)  # shape (3,)
    # S2_ij = sum of v_k_i * v_k_j
    S2 = verts.T @ verts  # shape (3,3), S2[i,j] = sum_k v_k[i]*v_k[j]

    # Second moment tensor: int x_i x_j dV = V/20 * (S2_ij + S1_i * S1_j)
    second_moment = (V / 20.0) * (S2 + np.outer(S1, S1))

    # Inertia tensor: I_ij = density * (trace(second_moment) * delta_ij - second_moment_ij)
    trace_sm = np.trace(second_moment)
    I = density * (trace_sm * np.eye(3) - second_moment)

    return mass, I


def convex_polyhedron_inertia(
    vertices: np.ndarray,
    faces: list,
    density: float = 1.0
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute mass, center of mass, and inertia tensor of a convex polyhedron.

    The polyhedron is decomposed into tetrahedra formed by each face triangle
    and the origin. The signed volumes ensure correct results even when the
    origin is outside the polyhedron.

    Parameters
    ----------
    vertices : ndarray, shape (n_vertices, 3)
        Vertex coordinates.
    faces : list of (int, int, int)
        Triangle face indices (outward-oriented normals).
    density : float
        Uniform mass density.

    Returns
    -------
    mass : float
        Total mass.
    com : ndarray, shape (3,)
        Center of mass.
    I_com : ndarray, shape (3, 3)
        Inertia tensor about the center of mass.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    total_mass = 0.0
    total_I_origin = np.zeros((3, 3))
    first_moment = np.zeros(3)  # for center of mass

    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        # Tetrahedron: origin, v0, v1, v2
        m, I = tetrahedron_inertia(
            np.zeros(3), v0, v1, v2, density
        )
        # Signed mass (based on orientation)
        signed_vol = np.dot(v0, np.cross(v1, v2)) / 6.0
        signed_mass = density * signed_vol

        total_mass += signed_mass
        total_I_origin += I * np.sign(signed_vol)
        # First moment: integral of r dV = V/4 * sum of vertices
        # For tetrahedron (0, v0, v1, v2):
        fm = (abs(signed_vol) / 4.0) * (v0 + v1 + v2) * np.sign(signed_vol)
        first_moment += density * fm

    if total_mass <= 0:
        raise ValueError("Non-positive total mass; check face orientations.")

    com = first_moment / total_mass

    # Shift inertia to center of mass using parallel axis theorem:
    # I_com = I_origin - M * (|com|^2 * I_3x3 - com (x) com)
    d = com
    I_com = total_I_origin - total_mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return total_mass, com, I_com


def convex_polyhedron_inertia_numerical(
    vertices: np.ndarray,
    faces: list,
    density: float = 1.0,
    n_samples: int = 1000000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Numerical (Monte Carlo) computation of inertia tensor for verification.

    Uses rejection sampling within the bounding box.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_vol = np.prod(bbox_max - bbox_min)

    # For point-in-convex-polyhedron test, we use the face normals.
    face_normals = []
    face_offsets = []
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n = n / np.linalg.norm(n)
        face_normals.append(n)
        face_offsets.append(np.dot(n, v0))

    rng = np.random.default_rng(42)
    total_mass = 0.0
    first_moment = np.zeros(3)
    second_moment_diag = np.zeros(3)  # xx, yy, zz
    second_moment_off = np.zeros(3)   # xy, yz, zx

    count_inside = 0
    batch_size = min(n_samples, 100000)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        pts = rng.uniform(bbox_min, bbox_max, size=(batch_size, 3))
        # Check if inside all faces
        inside = np.ones(batch_size, dtype=bool)
        for n, d in zip(face_normals, face_offsets):
            inside &= (np.dot(pts, n) - d) <= 1e-14

        pts_inside = pts[inside]
        count_inside += len(pts_inside)

        total_mass += density * len(pts_inside) * bbox_vol / n_samples
        first_moment += density * pts_inside.sum(axis=0) * bbox_vol / n_samples
        second_moment_diag += density * (pts_inside**2).sum(axis=0) * bbox_vol / n_samples
        second_moment_off[0] += density * (pts_inside[:, 0] * pts_inside[:, 1]).sum() * bbox_vol / n_samples
        second_moment_off[1] += density * (pts_inside[:, 1] * pts_inside[:, 2]).sum() * bbox_vol / n_samples
        second_moment_off[2] += density * (pts_inside[:, 0] * pts_inside[:, 2]).sum() * bbox_vol / n_samples

    com = first_moment / total_mass if total_mass > 0 else np.zeros(3)

    # Inertia about COM
    I_xx = density * (second_moment_diag[1] + second_moment_diag[2])
    I_yy = density * (second_moment_diag[0] + second_moment_diag[2])
    I_zz = density * (second_moment_diag[0] + second_moment_diag[1])
    I_xy = -density * second_moment_off[0]
    I_yz = -density * second_moment_off[1]
    I_xz = -density * second_moment_off[2]

    I_origin = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ])

    # Parallel axis theorem
    d = com
    I_com = I_origin - total_mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return total_mass, com, I_com


# ============================================================================
# 2. Poinsot Torque-Free Rotation (Jacobi Elliptic Functions)
# ============================================================================

def poinsot_analytical(
    t: np.ndarray,
    I1: float, I2: float, I3: float,
    omega0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical solution of torque-free Euler equations using Jacobi elliptic functions.

    For a rigid body with principal moments I1 < I2 < I3, the body-frame angular
    velocity is expressed in terms of Jacobi elliptic functions sn/cn/dn.

    Parameters
    ----------
    t : ndarray
        Time array.
    I1, I2, I3 : float
        Principal moments of inertia (I1 < I2 < I3 assumed).
    omega0 : ndarray, shape (3,)
        Initial angular velocity at t=0.

    Returns
    -------
    omega1, omega2, omega3 : ndarray
        Body-frame angular velocity components.
    """
    t = np.asarray(t, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)

    # Ensure I1 < I2 < I3
    assert I1 < I2 < I3, f"Need I1 < I2 < I3, got {I1}, {I2}, {I3}"

    # Compute conserved quantities
    I = np.diag([I1, I2, I3])
    L = np.linalg.norm(I @ omega0)
    E = 0.5 * omega0 @ I @ omega0

    # Parameters for the solution
    A1 = (2*E*I3 - L**2) / (I1*(I3 - I1))
    B1 = I2*(I3 - I2) / (I1*(I3 - I1))
    A3 = (L**2 - 2*E*I1) / (I3*(I3 - I1))
    B3 = I2*(I2 - I1) / (I3*(I3 - I1))

    # Determine which case based on energy
    if A1/B1 <= A3/B3:
        # Rotation about axis 3 (largest moment)
        w2_max = math.sqrt(A1/B1)
        k_sq = (A1 * B3) / (A3 * B1)
        Omega = abs((I3 - I1) / I2) * math.sqrt(B1 * A3)

        # Find phase shift tau0 to match initial conditions
        # w2(0) = w2_max * sn(tau0, k) = omega0[1]
        # sn(tau0, k) = omega0[1] / w2_max
        sn0 = omega0[1] / w2_max
        sn0 = np.clip(sn0, -1.0, 1.0)
        # tau0 = arcsin(sn0) in the elliptic sense
        # For Jacobi elliptic functions, if sn(u,k) = x, then u = F(arcsin(x), k)
        # where F is the incomplete elliptic integral of the first kind
        phi0 = math.asin(sn0)
        tau0 = ellipkinc(phi0, k_sq)

        # Adjust sign based on w1(0)
        # w1(0) = sqrt(B1) * w2_max * cn(tau0, k)
        # cn can be positive or negative, so we need to check the quadrant
        cn0 = math.sqrt(1 - sn0**2)  # cn(tau0) = sqrt(1 - sn^2(tau0))
        if omega0[0] < 0:
            cn0 = -cn0
            # Adjust tau0 to the correct quadrant
            # If cn < 0, we're in the second or third quadrant
            # tau0 should be in [K, 3K] where cn < 0
            K = ellipk(k_sq)
            if sn0 >= 0:
                tau0 = 2*K - tau0
            else:
                tau0 = 2*K + tau0

        tau = Omega * t + tau0
        sn, cn, dn, _ = ellipj(tau, k_sq)

        w1 = math.sqrt(B1) * w2_max * cn
        w2 = w2_max * sn
        w3 = math.sqrt(A3) * dn

    else:
        # Rotation about axis 1 (smallest moment)
        w2_max = math.sqrt(A3/B3)
        k_sq = (A3 * B1) / (A1 * B3)
        Omega = abs((I3 - I1) / I2) * math.sqrt(B3 * A1)

        # Find phase shift
        sn0 = omega0[1] / w2_max
        sn0 = np.clip(sn0, -1.0, 1.0)
        phi0 = math.asin(sn0)
        tau0 = ellipkinc(phi0, k_sq)

        # Adjust for w3(0)
        cn0 = math.sqrt(1 - sn0**2)
        if omega0[2] < 0:
            cn0 = -cn0
            K = ellipk(k_sq)
            if sn0 >= 0:
                tau0 = 2*K - tau0
            else:
                tau0 = 2*K + tau0

        tau = Omega * t + tau0
        sn, cn, dn, _ = ellipj(tau, k_sq)

        w3 = math.sqrt(B3) * w2_max * cn
        w2 = w2_max * sn
        w1 = math.sqrt(A1) * dn

    return w1, w2, w3


def euler_equations_rhs(
    omega: np.ndarray,
    I: np.ndarray
) -> np.ndarray:
    """Right-hand side of Euler's equations: domega/dt = I^{-1} (omega x (I omega)).

    Parameters
    ----------
    omega : ndarray, shape (3,)
        Body-frame angular velocity.
    I : ndarray, shape (3, 3)
        Inertia tensor in body frame (diagonal = principal axes).

    Returns
    -------
    domega_dt : ndarray, shape (3,)
    """
    I_inv = np.linalg.inv(I)
    L = I @ omega
    return I_inv @ np.cross(omega, L)


def integrate_euler_rk4(
    omega0: np.ndarray,
    I: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    torque: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate Euler's equations using RK4.

    Parameters
    ----------
    omega0 : ndarray, shape (3,)
        Initial angular velocity.
    I : ndarray, shape (3, 3)
        Inertia tensor.
    t_span : (float, float)
        Start and end time.
    n_steps : int
        Number of time steps.
    torque : ndarray, shape (3,), optional
        Constant external torque in body frame.

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
        Time array.
    omega : ndarray, shape (n_steps+1, 3)
        Angular velocity history.
    """
    I_inv = np.linalg.inv(I)
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    omega = np.zeros((n_steps + 1, 3))
    omega[0] = omega0

    def rhs(w):
        L = I @ w
        if torque is not None:
            dw = I_inv @ (np.cross(L, w) + torque)
        else:
            dw = I_inv @ np.cross(L, w)
        return dw

    for i in range(n_steps):
        w = omega[i]
        k1 = rhs(w)
        k2 = rhs(w + 0.5 * dt * k1)
        k3 = rhs(w + 0.5 * dt * k2)
        k4 = rhs(w + dt * k3)
        omega[i + 1] = w + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return t, omega


# ============================================================================
# 3. Quaternion Integration
# ============================================================================

def quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (scalar-first convention).

    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize_np(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    return q / np.linalg.norm(q)


def integrate_quaternion_rk4(
    q0: np.ndarray,
    omega_body_func,
    t_span: Tuple[float, float],
    n_steps: int,
    renormalize: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate quaternion kinematic equation dq/dt = 0.5 * q (x) omega using RK4.

    Parameters
    ----------
    q0 : ndarray, shape (4,)
        Initial quaternion [w, x, y, z].
    omega_body_func : callable
        Function omega(t) -> ndarray(3,) giving body-frame angular velocity.
    t_span : (float, float)
        Start and end time.
    n_steps : int
        Number of time steps.
    renormalize : bool
        If True, renormalize quaternion after each step.

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
    q : ndarray, shape (n_steps+1, 4)
        Quaternion history.
    omega_hist : ndarray, shape (n_steps+1, 3)
        Angular velocity history.
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    q = np.zeros((n_steps + 1, 4))
    omega_hist = np.zeros((n_steps + 1, 3))
    q[0] = q0 / np.linalg.norm(q0)
    omega_hist[0] = omega_body_func(t0)

    def dqdt(qq, w):
        # dq/dt = 0.5 * q (x) [0, omega]
        omega_quat = np.array([0.0, w[0], w[1], w[2]])
        return 0.5 * quat_multiply_np(qq, omega_quat)

    for i in range(n_steps):
        ti = t[i]
        qi = q[i]
        wi = omega_hist[i]

        k1 = dqdt(qi, wi)
        w_mid1 = omega_body_func(ti + 0.5*dt)
        k2 = dqdt(qi + 0.5*dt*k1, w_mid1)
        w_mid2 = omega_body_func(ti + 0.5*dt)
        k3 = dqdt(qi + 0.5*dt*k2, w_mid2)
        w_end = omega_body_func(ti + dt)
        k4 = dqdt(qi + dt*k3, w_end)

        q[i+1] = qi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if renormalize:
            q[i+1] = q[i+1] / np.linalg.norm(q[i+1])
        omega_hist[i+1] = omega_body_func(t[i+1])

    return t, q, omega_hist


def integrate_quaternion_expmap(
    q0: np.ndarray,
    omega_body_func,
    t_span: Tuple[float, float],
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate quaternion using exponential map (exact for constant omega).

    This is the method used in TensorDEM's floe_6dof module. The exponential
    map q <- q * exp(omega*dt/2) exactly preserves the unit norm.

    Parameters
    ----------
    q0 : ndarray, shape (4,)
        Initial quaternion [w, x, y, z].
    omega_body_func : callable
        Function omega(t) -> ndarray(3,) giving body-frame angular velocity.
    t_span : (float, float)
        Start and end time.
    n_steps : int
        Number of time steps.

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
    q : ndarray, shape (n_steps+1, 4)
        Quaternion history.
    omega_hist : ndarray, shape (n_steps+1, 3)
        Angular velocity history.
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    q = np.zeros((n_steps + 1, 4))
    omega_hist = np.zeros((n_steps + 1, 3))
    q[0] = q0 / np.linalg.norm(q0)
    omega_hist[0] = omega_body_func(t0)

    for i in range(n_steps):
        ti = t[i]
        qi = q[i]
        # Use omega at the midpoint for better accuracy
        omega_mid = omega_body_func(ti + 0.5*dt)

        # Exponential map: q_new = q * exp(omega*dt/2)
        # exp(theta * n) = [cos(theta), sin(theta)*n] where theta = |omega|*dt/2
        angle = np.linalg.norm(omega_mid) * dt / 2.0
        if angle > 1e-12:
            axis = omega_mid / np.linalg.norm(omega_mid)
            delta = np.array([math.cos(angle),
                            math.sin(angle)*axis[0],
                            math.sin(angle)*axis[1],
                            math.sin(angle)*axis[2]])
        else:
            # Small angle approximation
            delta = np.array([1.0, omega_mid[0]*dt/2, omega_mid[1]*dt/2, omega_mid[2]*dt/2])
            delta = delta / np.linalg.norm(delta)

        q[i+1] = quat_multiply_np(qi, delta)
        # The exponential map preserves norm exactly, but renormalize for safety
        q[i+1] = q[i+1] / np.linalg.norm(q[i+1])

        omega_hist[i+1] = omega_body_func(t[i+1])

    return t, q, omega_hist


# ============================================================================
# 4. Conservation Quantities
# ============================================================================

def angular_momentum_body(
    omega: np.ndarray,
    I: np.ndarray
) -> np.ndarray:
    """Compute angular momentum L = I * omega in body frame."""
    return I @ omega


def kinetic_energy_body(
    omega: np.ndarray,
    I: np.ndarray
) -> float:
    """Compute rotational kinetic energy T = 0.5 * omega^T * I * omega."""
    return 0.5 * omega @ I @ omega
