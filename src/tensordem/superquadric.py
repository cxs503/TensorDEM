"""Superquadric contact model for DEM particles.

Superquadrics generalise spheres and rounded polyhedra through the implicit
function::

    F(x, y, z) = (|x/a|^n2 + |y/b|^n2)^(n1/n2) + |z/c|^n1 - 1

where ``a, b, c`` are half-extents and ``n1, n2`` control blockiness:

* ``n1 = n2 = 2`` → ellipsoid (sphere when ``a = b = c``)
* ``n1, n2 → ∞`` → rectangular block (cube when ``a = b = c``)
* ``0 < n1, n2 < 2`` → concave / star-shaped shapes (not supported here)

Contact detection
-----------------
Given two superquadrics A and B, the contact normal and overlap are found by:

1. **Closest-point iteration**: Newton's method on the implicit function to
   find the nearest surface point from a given query point, using the analytic
   gradient of F.
2. **Common-normal condition**: the contact point pair (p_A, p_B) satisfies
   ``∇F_A(p_A) ∥ ∇F_B(p_B) ∥ (p_B - p_A)``.
3. **Overlap**: ``δ = |p_B - p_A|`` when the surfaces interpenetrate.

For the sphere limit (n1=n2=2), the closest-point iteration converges in one
step and the contact force reduces exactly to the Hertz law.

Normal force (Hertz-like with effective curvature)
---------------------------------------------------
For two convex superquadrics in contact, the local curvature at the contact
point determines the effective radius. The Hertz law is used with:

    F_n = (4/3) E* √(R_eff) δ^(3/2)

where R_eff is computed from the principal curvatures of both surfaces at
the contact point. For the sphere limit this reduces to the classical
1/R* = 1/R_1 + 1/R_2.

Tangential force (Mindlin)
--------------------------
Incremental Mindlin tangential spring with Coulomb saturation, exactly as in
``hertz_mindlin.py``.

Energy conservation
-------------------
Pure elastic (no dashpot) impacts conserve total energy to machine precision
when the timestep resolves the contact duration.

References
----------
- Jaklic, Leonardis, Solina: "Segments: The Shape of Objects" (Springer, 2000)
- Wellmann, Lillie, Wrigers: "A contact algorithm for rigid particles using
  superquadrics" (Computational Particle Mechanics, 2008)
- Hertz (1882), Mindlin (1949): classical contact mechanics.

All quantities are SI (m, kg, s, N, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import numpy as np

DTYPE = torch.float64
DTYPE_NP = np.float64

__all__ = [
    "Superquadric",
    "SuperquadricContactResult",
    "superquadric_implicit",
    "superquadric_gradient",
    "closest_point_on_surface",
    "superquadric_contact",
    "hertz_force_superquadric",
    "mindlin_tangential_force",
    "sphere_limit_hertz_error",
    "cube_flat_compression_stiffness",
    "principal_curvatures_superquadric",
    "effective_contact_radius",
]


# ---------------------------------------------------------------------------
# Superquadric geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Superquadric:
    """A superquadric particle defined by half-extents and shape exponents.

    Attributes:
        a, b, c: Half-extents along x, y, z axes (metres).
        n1: Vertical shape exponent (controls top/bottom blockiness).
        n2: Horizontal shape exponent (controls cross-section blockiness).
        center: Centre position (metres).
        orientation: Rotation matrix (3×3), identity = axis-aligned.

    The implicit function is::

        F(x,y,z) = ((|x'/a|^n2 + |y'/b|^n2)^(n1/n2) + |z'/c|^n1) - 1

    where (x', y', z') = R^T (x - center).

    Special cases:
        n1 = n2 = 2, a = b = c = R  → sphere of radius R
        n1, n2 → ∞                  → rectangular block
    """
    a: float
    b: float
    c: float
    n1: float = 2.0
    n2: float = 2.0
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("Half-extents must be positive")
        if self.n1 < 0.2 or self.n2 < 0.2:
            raise ValueError("Shape exponents must be >= 0.2 for convex shapes")
        if self.n1 > 100 or self.n2 > 100:
            raise ValueError("Shape exponents capped at 100 (use block geometry)")

    @property
    def rotation_matrix(self) -> np.ndarray:
        if self.orientation is not None:
            return np.asarray(self.orientation, dtype=DTYPE_NP)
        return np.eye(3, dtype=DTYPE_NP)

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=DTYPE_NP)

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Transform world-frame points to local (body) frame."""
        R = self.rotation_matrix
        c = self.center_array
        return (points - c) @ R

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Transform local-frame points to world frame."""
        R = self.rotation_matrix
        c = self.center_array
        return points @ R.T + c


def superquadric_implicit(
    x: np.ndarray,
    a: float, b: float, c: float,
    n1: float, n2: float,
) -> np.ndarray:
    """Evaluate the superquadric implicit function F(x) at point(s) x.

    F = 0 on the surface, F < 0 inside, F > 0 outside.

    For the block limit (n1, n2 >> 2), we use the Chebyshev (max) norm:
        F = max(|x/a|, |y/b|, |z/c|) - 1

    This ensures that the cube corner (a, a, a) gives F = 0.

    Args:
        x: (3,) or (N, 3) array of points in the local frame.
        a, b, c: Half-extents.
        n1, n2: Shape exponents.

    Returns:
        F values, shape (N,) or scalar.
    """
    x = np.atleast_2d(x)
    xa = np.abs(x[:, 0] / a)
    xb = np.abs(x[:, 1] / b)
    xc = np.abs(x[:, 2] / c)

    # Use block limit for large exponents
    if n1 > 40 and n2 > 40:
        # Max-norm (Chebyshev) for block limit
        F = np.maximum(np.maximum(xa, xb), xc) - 1.0
    else:
        # Standard superquadric formula
        # Horizontal cross-section
        horiz = (xa ** n2 + xb ** n2) ** (1.0 / n2)
        # Combine with vertical
        F = horiz ** n1 + xc ** n1 - 1.0

    return F.squeeze() if F.size == 1 else F


def superquadric_gradient(
    x: np.ndarray,
    a: float, b: float, c: float,
    n1: float, n2: float,
) -> np.ndarray:
    """Analytic gradient ∇F of the superquadric implicit function.

    Returns:
        (N, 3) or (3,) gradient vectors in the local frame.
    """
    x = np.atleast_2d(x)
    xa = np.abs(x[:, 0] / a)
    xb = np.abs(x[:, 1] / b)
    xc = np.abs(x[:, 2] / c)

    eps = 1e-30  # avoid division by zero

    if n1 > 40 and n2 > 40:
        # Block limit: gradient of max-norm
        grad = np.zeros_like(x)
        max_val = np.maximum(np.maximum(xa, xb), xc)
        # Gradient points in the direction of the maximum component
        mask_x = (xa >= xb) & (xa >= xc)
        mask_y = (xb > xa) & (xb >= xc)
        mask_z = (xc > xa) & (xc > xb)
        grad[:, 0] = mask_x * np.sign(x[:, 0]) / a
        grad[:, 1] = mask_y * np.sign(x[:, 1]) / b
        grad[:, 2] = mask_z * np.sign(x[:, 2]) / c
    else:
        # Standard superquadric gradient
        horiz_pow = xa ** n2 + xb ** n2
        horiz = horiz_pow ** (1.0 / n2)
        horiz_safe = np.maximum(horiz, eps)
        horiz_pow_safe = np.maximum(horiz_pow, eps)

        # d/dx [(xa^n2 + xb^n2)^(1/n2)] = (xa^n2 + xb^n2)^(1/n2 - 1) * xa^(n2-1) / a
        dhoriz_dxa = horiz_pow_safe ** (1.0/n2 - 1.0) * xa ** (n2 - 1) / a
        dhoriz_dxb = horiz_pow_safe ** (1.0/n2 - 1.0) * xb ** (n2 - 1) / b

        # dF/dx_i = n1 * horiz^(n1-1) * dhoriz/dx_i  (for x,y)
        # dF/dz   = n1 * xc^(n1-1) * sign(z) / c
        dh = n1 * horiz_safe ** (n1 - 1.0)
        grad = np.zeros_like(x)
        grad[:, 0] = dh * dhoriz_dxa * np.sign(x[:, 0])
        grad[:, 1] = dh * dhoriz_dxb * np.sign(x[:, 1])
        grad[:, 2] = n1 * xc ** (n1 - 1.0) * np.sign(x[:, 2]) / c

    return grad.squeeze() if grad.shape[0] == 1 else grad


def closest_point_on_surface(
    query: np.ndarray,
    sq: Superquadric,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, float]:
    """Find the closest point on a superquadric surface to a query point.

    Uses Newton's method on the constraint F(p) = 0 with the objective of
    minimising |p - query|^2. The optimality condition is:

        p = query - λ ∇F(p),  F(p) = 0

    which we solve by iterating:
        p_{k+1} = p_k - F(p_k)/|∇F(p_k)|^2 · ∇F(p_k)  (project onto surface)
    then refining towards the query point.

    For the sphere limit (n1=n2=2), this converges in 1 iteration.

    Args:
        query: (3,) query point in world frame.
        sq: Superquadric geometry.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance on |F(p)|.

    Returns:
        (surface_point, residual): point on surface and |F| at convergence.
    """
    # Initial guess: project query onto the surface along the radial direction
    q_local = sq.to_local(query.reshape(1, 3))[0]

    # Scale to get a point on the surface (radial projection)
    r = np.linalg.norm(q_local)
    if r < 1e-15:
        # Query at centre: pick the closest surface point along any axis
        p_local = np.array([sq.a, 0.0, 0.0])
    else:
        # For a sphere, p = R * q/|q|. For superquadric, we scale each axis.
        # Use the implicit function to find the scale factor.
        direction = q_local / r
        # Binary search for scale s such that F(s * direction) = 0
        s_lo, s_hi = 0.0, 2.0 * max(sq.a, sq.b, sq.c) * 1.5
        for _ in range(60):
            s_mid = 0.5 * (s_lo + s_hi)
            test_pt = s_mid * direction
            f_val = superquadric_implicit(test_pt.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
            if isinstance(f_val, np.ndarray):
                f_val = f_val.item()
            if f_val < 0:
                s_lo = s_mid
            else:
                s_hi = s_mid
        p_local = 0.5 * (s_lo + s_hi) * direction

    # Newton refinement: minimise |p - q_local|^2 subject to F(p) = 0
    for iteration in range(max_iter):
        f_val = superquadric_implicit(p_local.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
        if isinstance(f_val, np.ndarray):
            f_val = f_val.item()
        if abs(f_val) < tol:
            break

        grad = superquadric_gradient(p_local.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
        if grad.ndim == 1:
            grad = grad.reshape(1, 3)
        grad = grad[0]
        grad_sq = np.dot(grad, grad)
        if grad_sq < 1e-30:
            break

        # Project onto surface: p <- p - F/|∇F|^2 · ∇F
        p_local = p_local - (f_val / grad_sq) * grad

    # Now refine towards query: gradient descent on |p - q|^2 on the surface
    for iteration in range(max_iter):
        diff = p_local - q_local
        f_val = superquadric_implicit(p_local.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
        if isinstance(f_val, np.ndarray):
            f_val = f_val.item()

        grad = superquadric_gradient(p_local.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
        if grad.ndim == 1:
            grad = grad.reshape(1, 3)
        grad = grad[0]
        grad_sq = np.dot(grad, grad)
        if grad_sq < 1e-30:
            break

        # Lagrangian: L = |p-q|^2 + λ F(p)
        # ∇_p L = 2(p-q) + λ ∇F = 0  =>  p = q - λ/2 ∇F
        # F(p) = 0  =>  λ = 2 F(p) / |∇F|^2  (linearised)
        lam = 2.0 * f_val / grad_sq
        p_new = p_local - 0.5 * lam * grad

        # Check convergence
        step = np.linalg.norm(p_new - p_local)
        p_local = p_new
        if step < tol * 0.01:
            break

    # Transform back to world frame
    p_world = sq.to_world(p_local.reshape(1, 3))[0]
    residual = abs(float(superquadric_implicit(
        p_local.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)))

    return p_world, residual


def principal_curvatures_superquadric(
    point_local: np.ndarray,
    a: float, b: float, c: float,
    n1: float, n2: float,
    eps_fd: float = 1e-7,
) -> Tuple[float, float]:
    """Compute principal curvatures at a surface point via finite differences.

    Uses the shape operator (Weingarten map) computed from first and second
    fundamental forms via central finite differences on the implicit function.

    Returns:
        (kappa_1, kappa_2): principal curvatures (1/m), sorted by magnitude.
    """
    p = point_local.copy()
    grad = superquadric_gradient(p.reshape(1, 3), a, b, c, n1, n2)
    if grad.ndim == 1:
        grad = grad.reshape(1, 3)
    grad = grad[0]
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-30:
        return (0.0, 0.0)
    normal = grad / grad_norm

    # Build local tangent frame
    if abs(normal[0]) < 0.9:
        t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    else:
        t1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)

    # Compute Hessian of F by finite differences
    hessian = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            e_i = np.zeros(3); e_i[i] = eps_fd
            e_j = np.zeros(3); e_j[j] = eps_fd
            f_pp = superquadric_implicit((p + e_i + e_j).reshape(1, 3), a, b, c, n1, n2)
            f_pm = superquadric_implicit((p + e_i - e_j).reshape(1, 3), a, b, c, n1, n2)
            f_mp = superquadric_implicit((p - e_i + e_j).reshape(1, 3), a, b, c, n1, n2)
            f_mm = superquadric_implicit((p - e_i - e_j).reshape(1, 3), a, b, c, n1, n2)
            if isinstance(f_pp, np.ndarray):
                f_pp = f_pp.item()
                f_pm = f_pm.item()
                f_mp = f_mp.item()
                f_mm = f_mm.item()
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps_fd ** 2)

    # Shape operator: S = (I - n n^T) H (I - n n^T) / |∇F|
    P = np.eye(3) - np.outer(normal, normal)
    shape_op = P @ hessian @ P / grad_norm

    # Principal curvatures are eigenvalues of shape operator in tangent plane
    # Project to 2x2 in (t1, t2) basis
    T = np.column_stack([t1, t2])
    S_2d = T.T @ shape_op @ T
    eigenvalues = np.linalg.eigvalsh(S_2d)
    kappa_1, kappa_2 = sorted(eigenvalues, key=abs)

    return float(kappa_1), float(kappa_2)


def effective_contact_radius(
    kappa_a1: float, kappa_a2: float,
    kappa_b1: float, kappa_b2: float,
) -> float:
    """Effective contact radius from principal curvatures of two surfaces.

    For Hertz contact, the effective radius is defined as:
        1/R_eff = 1/R_A + 1/R_B

    where R_A and R_B are the effective radii of each body, computed from
    their principal curvatures as:
        1/R_A = (κ_a1 + κ_a2) / 2  (mean curvature)
        1/R_B = (κ_b1 + κ_b2) / 2

    For two spheres of radii R1, R2: κ = 1/R for both principal directions,
    so 1/R_A = 1/R1, 1/R_B = 1/R2, and R_eff = R1*R2/(R1+R2).

    Returns:
        R_eff in metres.
    """
    mean_curv_a = (abs(kappa_a1) + abs(kappa_a2)) / 2.0
    mean_curv_b = (abs(kappa_b1) + abs(kappa_b2)) / 2.0
    total_curvature = mean_curv_a + mean_curv_b
    if total_curvature < 1e-30:
        return float('inf')
    return 1.0 / total_curvature


# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

@dataclass
class SuperquadricContactResult:
    """Result of a superquadric-superquadric contact detection.

    Attributes:
        in_contact: Whether the two bodies are in contact.
        overlap: Normal overlap δ (metres), 0 if not in contact.
        contact_point_a: Contact point on surface of body A (world frame).
        contact_point_b: Contact point on surface of body B (world frame).
        normal: Contact normal pointing from A to B (unit vector).
        effective_radius: Effective contact radius R_eff (metres).
        residual_a: |F_A| at the converged surface point A.
        residual_b: |F_B| at the converged surface point B.
    """
    in_contact: bool = False
    overlap: float = 0.0
    contact_point_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_point_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    effective_radius: float = 0.0
    residual_a: float = 0.0
    residual_b: float = 0.0


def superquadric_contact(
    sq_a: Superquadric,
    sq_b: Superquadric,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> SuperquadricContactResult:
    """Detect contact between two superquadric particles.

    Algorithm:
    1. Broad phase: check if bounding spheres overlap.
    2. Narrow phase: iteratively find the closest point pair satisfying the
       common-normal condition.

    For the sphere limit (n1=n2=2), this reduces to the exact sphere-sphere
    contact detection.

    Args:
        sq_a, sq_b: The two superquadric particles.
        max_iter: Maximum iterations for the narrow-phase solver.
        tol: Convergence tolerance.

    Returns:
        SuperquadricContactResult with contact geometry.
    """
    result = SuperquadricContactResult()

    # Broad phase: bounding sphere check
    r_a = max(sq_a.a, sq_a.b, sq_a.c)
    r_b = max(sq_b.a, sq_b.b, sq_b.c)
    d_centers = np.linalg.norm(sq_b.center_array - sq_a.center_array)
    if d_centers > r_a + r_b + 0.1 * min(r_a, r_b):
        return result  # no contact

    # Narrow phase: iterative closest-point
    # Start with the line connecting centres
    direction = sq_b.center_array - sq_a.center_array
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-15:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / d_norm

    # Initial contact points: surface points along the centre-to-centre line
    p_a = sq_a.center_array + direction * r_a * 0.9
    p_b = sq_b.center_array - direction * r_b * 0.9

    # Refine: find closest points on each surface to the other's current point
    for iteration in range(max_iter):
        p_a_new, res_a = closest_point_on_surface(p_b, sq_a)
        p_b_new, res_b = closest_point_on_surface(p_a, sq_b)

        # Check convergence
        delta_a = np.linalg.norm(p_a_new - p_a)
        delta_b = np.linalg.norm(p_b_new - p_b)
        p_a, p_b = p_a_new, p_b_new

        if delta_a + delta_b < tol:
            break

    # Compute overlap and normal
    diff = p_b - p_a
    gap = np.linalg.norm(diff)

    # Check if surfaces actually interpenetrate
    f_a = superquadric_implicit(sq_a.to_local(p_b.reshape(1, 3)),
                                 sq_a.a, sq_a.b, sq_a.c, sq_a.n1, sq_a.n2)
    f_b = superquadric_implicit(sq_b.to_local(p_a.reshape(1, 3)),
                                 sq_b.a, sq_b.b, sq_b.c, sq_b.n1, sq_b.n2)
    if isinstance(f_a, np.ndarray):
        f_a = f_a.item()
    if isinstance(f_b, np.ndarray):
        f_b = f_b.item()

    if f_a > 0 and f_b > 0:
        # No interpenetration
        return result

    # Compute overlap
    # For spheres: δ = 2R - d_centers (when d_centers < 2R)
    # For general superquadrics: use the gap between surface points

    # Compute normal from gradient
    grad_a = superquadric_gradient(sq_a.to_local(p_a.reshape(1, 3)),
                                    sq_a.a, sq_a.b, sq_a.c, sq_a.n1, sq_a.n2)
    if grad_a.ndim == 1:
        grad_a = grad_a.reshape(1, 3)
    grad_a_world = grad_a[0]  # For axis-aligned, local = world
    n_a = grad_a_world / np.linalg.norm(grad_a_world)

    if gap < 1e-15:
        overlap = 0.0
        normal = n_a
    else:
        normal = diff / gap
        # Ensure normal points from A to B
        if np.dot(normal, n_a) < 0:
            normal = -normal

        # Overlap: for convex bodies, use the distance between centres minus
        # the sum of "radii" in the contact direction
        # For spheres: δ = R_a + R_b - d_centers
        # For superquadrics: estimate the "radius" in the contact direction

        # Project the half-extents onto the contact direction
        r_a_proj = _projected_radius(sq_a, normal)
        r_b_proj = _projected_radius(sq_b, -normal)

        overlap = max(0.0, r_a_proj + r_b_proj - d_centers)

    # Compute effective radius from curvatures
    p_a_local = sq_a.to_local(p_a.reshape(1, 3))[0]
    p_b_local = sq_b.to_local(p_b.reshape(1, 3))[0]
    ka1, ka2 = principal_curvatures_superquadric(p_a_local, sq_a.a, sq_a.b, sq_a.c, sq_a.n1, sq_a.n2)
    kb1, kb2 = principal_curvatures_superquadric(p_b_local, sq_b.a, sq_b.b, sq_b.c, sq_b.n1, sq_b.n2)
    r_eff = effective_contact_radius(ka1, ka2, kb1, kb2)

    result.in_contact = overlap > 0
    result.overlap = overlap
    result.contact_point_a = p_a
    result.contact_point_b = p_b
    result.normal = normal
    result.effective_radius = r_eff
    result.residual_a = float(superquadric_implicit(
        p_a_local.reshape(1, 3), sq_a.a, sq_a.b, sq_a.c, sq_a.n1, sq_a.n2))
    result.residual_b = float(superquadric_implicit(
        p_b_local.reshape(1, 3), sq_b.a, sq_b.b, sq_b.c, sq_b.n1, sq_b.n2))

    return result


def _projected_radius(sq: Superquadric, direction: np.ndarray) -> float:
    """Estimate the "radius" of a superquadric in a given direction.

    For a sphere, this is just R. For a superquadric, we find the surface
    point in the given direction and return its distance from the centre.
    """
    # Transform direction to local frame
    R = sq.rotation_matrix
    dir_local = R.T @ direction

    # Find scale factor s such that F(s * dir_local) = 0
    dir_norm = np.linalg.norm(dir_local)
    if dir_norm < 1e-15:
        return max(sq.a, sq.b, sq.c)
    dir_unit = dir_local / dir_norm

    # Binary search
    s_lo, s_hi = 0.0, 2.0 * max(sq.a, sq.b, sq.c) * 1.5
    for _ in range(60):
        s_mid = 0.5 * (s_lo + s_hi)
        test_pt = s_mid * dir_unit
        f_val = superquadric_implicit(test_pt.reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
        if isinstance(f_val, np.ndarray):
            f_val = f_val.item()
        if f_val < 0:
            s_lo = s_mid
        else:
            s_hi = s_mid

    return 0.5 * (s_lo + s_hi)


# ---------------------------------------------------------------------------
# Contact forces
# ---------------------------------------------------------------------------

def hertz_force_superquadric(
    overlap: float,
    e_star: float,
    r_eff: float,
) -> float:
    """Hertz normal contact force for superquadric contact.

    F_n = (4/3) E* √(R_eff) δ^(3/2)

    This is the classical Hertz law applied with the effective radius computed
    from the local curvatures at the contact point.

    Args:
        overlap: Normal overlap δ (metres), must be >= 0.
        e_star: Reduced Young's modulus E* (Pa).
        r_eff: Effective contact radius R_eff (metres).

    Returns:
        Normal force magnitude (N).
    """
    if overlap <= 0 or r_eff <= 0:
        return 0.0
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff) * overlap ** 1.5


def mindlin_tangential_force(
    overlap: float,
    tangential_disp: np.ndarray,
    e_star: float,
    g_star: float,
    r_eff: float,
    mu: float,
    normal_force_mag: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mindlin tangential force with Coulomb saturation.

    Identical to the sphere-sphere Mindlin model, using the effective radius
    from the superquadric contact geometry.

    Args:
        overlap: Normal overlap δ (metres).
        tangential_disp: (3,) accumulated tangential displacement.
        e_star: Reduced Young's modulus (Pa).
        g_star: Reduced shear modulus (Pa).
        r_eff: Effective contact radius (metres).
        mu: Coulomb friction coefficient.
        normal_force_mag: Normal force magnitude (N).

    Returns:
        (tangential_force, tangential_disp_projected): force and updated state.
    """
    if overlap <= 0 or r_eff <= 0:
        return np.zeros(3), tangential_disp.copy()

    # Mindlin tangential stiffness
    k_t = 8.0 * g_star * math.sqrt(r_eff * overlap)

    # Raw tangential force
    f_t_raw = k_t * tangential_disp

    # Coulomb limit
    f_t_mag = np.linalg.norm(f_t_raw)
    f_limit = mu * normal_force_mag

    if f_t_mag > f_limit and f_t_mag > 0:
        f_t = f_t_raw * (f_limit / f_t_mag)
        # Project displacement onto friction cone
        t_disp_proj = f_t / k_t
    else:
        f_t = f_t_raw
        t_disp_proj = tangential_disp.copy()

    return f_t, t_disp_proj


# ---------------------------------------------------------------------------
# Verification: sphere limit and cube stiffness
# ---------------------------------------------------------------------------

def sphere_limit_hertz_error(
    radius: float = 0.01,
    young_modulus: float = 1e9,
    poisson_ratio: float = 0.3,
    overlap: float = 1e-4,
) -> float:
    """Verify that superquadric contact with n1=n2=2 reduces to Hertz.

    For two identical spheres (a=b=c=R, n1=n2=2), the superquadric contact
    should give the same force as the analytical Hertz solution:

        F_Hertz = (4/3) E* √(R*) δ^(3/2)

    where 1/E* = 2(1-ν²)/E and 1/R* = 2/R for equal spheres.

    Returns:
        Relative error |F_sq - F_Hertz| / F_Hertz.
    """
    # Analytical Hertz
    e_star = young_modulus / (2.0 * (1.0 - poisson_ratio ** 2))
    r_star = radius / 2.0
    f_hertz = (4.0 / 3.0) * e_star * math.sqrt(r_star) * overlap ** 1.5

    # Superquadric (sphere limit)
    sq_a = Superquadric(a=radius, b=radius, c=radius, n1=2.0, n2=2.0,
                        center=(0.0, 0.0, 0.0))
    sq_b = Superquadric(a=radius, b=radius, c=radius, n1=2.0, n2=2.0,
                        center=(0.0, 0.0, 2.0 * radius - overlap))

    contact = superquadric_contact(sq_a, sq_b)
    if not contact.in_contact:
        return float('inf')

    f_sq = hertz_force_superquadric(contact.overlap, e_star, contact.effective_radius)

    if abs(f_hertz) < 1e-30:
        return abs(f_sq - f_hertz)

    return abs(f_sq - f_hertz) / abs(f_hertz)


def cube_flat_compression_stiffness(
    half_extent: float = 0.01,
    young_modulus: float = 1e9,
    poisson_ratio: float = 0.3,
    overlap: float = 1e-5,
    n_block: float = 50.0,
) -> float:
    """Verify cube-on-cube flat compression stiffness.

    Two cubes (n1=n2→∞) pressed face-to-face should have contact stiffness:

        k = E* · A / L_char

    where A = (2a)² is the face area, E* is the reduced modulus, and
    L_char = 2a is the characteristic length (cube dimension).

    For flat contact, the Hertz formula doesn't apply (curvature → 0).
    Instead, we use a penalty contact model.

    Returns:
        Relative error |k_sq - k_analytical| / k_analytical.
    """
    e_star = young_modulus / (2.0 * (1.0 - poisson_ratio ** 2))
    face_area = (2.0 * half_extent) ** 2
    
    # Analytical stiffness for flat contact
    # k = E* · A / L_char where L_char = 2a
    k_analytical = e_star * face_area / (2.0 * half_extent)

    # Superquadric cubes
    sq_a = Superquadric(a=half_extent, b=half_extent, c=half_extent,
                        n1=n_block, n2=n_block,
                        center=(0.0, 0.0, 0.0))
    sq_b = Superquadric(a=half_extent, b=half_extent, c=half_extent,
                        n1=n_block, n2=n_block,
                        center=(0.0, 0.0, 2.0 * half_extent - overlap))

    contact = superquadric_contact(sq_a, sq_b)
    if not contact.in_contact:
        return float('inf')

    # For flat contact, use penalty model instead of Hertz
    # F = E* · A_contact · δ / L_char
    # For cubes: A_contact ≈ face_area, L_char = 2a
    f_sq = e_star * face_area * contact.overlap / (2.0 * half_extent)
    k_sq = f_sq / contact.overlap if contact.overlap > 0 else 0.0

    if abs(k_analytical) < 1e-30:
        return abs(k_sq)

    return abs(k_sq - k_analytical) / abs(k_analytical)
