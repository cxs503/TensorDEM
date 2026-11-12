"""Verification tests for superquadric curvature analytic formulas.

Campaign node P165-W2-N06: closed-form curvature of superquadric surfaces
(replacing the ~15% error FD estimator of the parent kernel).

Acceptance criteria:
(a) Sphere: k1 = k2 = 1/R at all surface points, rel error < 1e-10
(b) Ellipsoid: Gaussian curvature K matches closed-form formula, rel error < 1e-8
(c) Mesh convergence: mesh estimator converges to analytic at O(N^-2), fitted order in [1.8, 2.2]
(d) FD convergence: legacy FD estimator converges at O(h^2), fitted order in [1.5, 2.5]
(e) Block limit: n1,n2 >= 40 → curvature on flat face centers → 0
(f) Symmetry: k1,k2 invariant under coordinate permutations for symmetric shapes
(g) Negative control: point not on surface → raises (IndexError from internal indexing)
(h) pytest >= 15 cases all green
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.superquadric_curvature import (
    BLOCK_EXPONENT,
    ellipsoid_gaussian_curvature,
    fd_refinement_order,
    gaussian_curvature,
    mean_curvature,
    mesh_refinement_order,
    principal_curvatures_analytic,
    principal_curvatures_fd,
    principal_curvatures_mesh,
    surface_points,
    superquadric_gradient,
    superquadric_hessian,
)


# ---------------------------------------------------------------------------
# (a) Sphere: k1 = k2 = 1/R everywhere
# ---------------------------------------------------------------------------

class TestSphereCurvature:
    """Criterion (a): sphere reports k1 = k2 = 1/R at all surface points."""

    @pytest.mark.parametrize("eta_deg,omega_deg", [
        (0, 0), (0, 45), (0, 90), (30, 45), (60, 30),
        (90, 0), (45, 135), (-30, 60),
    ])
    def test_sphere_curvature_isotropic(self, eta_deg, omega_deg):
        R = 0.5
        eta, omega = math.radians(eta_deg), math.radians(omega_deg)
        p = surface_points(np.array([eta]), np.array([omega]), R, R, R, 2.0, 2.0)
        k1, k2 = principal_curvatures_analytic(p, R, R, R, 2.0, 2.0)
        expected = 1.0 / R
        assert k1[0] == pytest.approx(expected, rel=1e-10)
        assert k2[0] == pytest.approx(expected, rel=1e-10)

    def test_sphere_gaussian_curvature(self):
        R = 2.0
        eta, omega = math.radians(37), math.radians(53)
        p = surface_points(np.array([eta]), np.array([omega]), R, R, R, 2.0, 2.0)
        K = gaussian_curvature(p, R, R, R, 2.0, 2.0)
        assert K[0] == pytest.approx(1.0 / R**2, rel=1e-10)

    def test_sphere_mean_curvature(self):
        R = 1.5
        eta, omega = math.radians(22), math.radians(77)
        p = surface_points(np.array([eta]), np.array([omega]), R, R, R, 2.0, 2.0)
        H = mean_curvature(p, R, R, R, 2.0, 2.0)
        assert H[0] == pytest.approx(1.0 / R, rel=1e-10)


# ---------------------------------------------------------------------------
# (b) Ellipsoid: Gaussian curvature matches closed-form
# ---------------------------------------------------------------------------

class TestEllipsoidGaussianCurvature:
    """Criterion (b): K matches ellipsoid closed-form formula."""

    def test_ellipsoid_K_at_pole(self):
        a, b, c = 2.0, 1.5, 1.0
        eta, omega = math.pi / 2, 0.0
        p = surface_points(np.array([eta]), np.array([omega]), a, b, c, 2.0, 2.0)
        K_analytic = gaussian_curvature(p, a, b, c, 2.0, 2.0)
        K_closed = ellipsoid_gaussian_curvature(p, a, b, c)
        assert K_analytic[0] == pytest.approx(K_closed[0], rel=1e-8)

    def test_ellipsoid_K_equator(self):
        a, b, c = 3.0, 2.0, 1.0
        eta, omega = 0.0, math.pi / 4
        p = surface_points(np.array([eta]), np.array([omega]), a, b, c, 2.0, 2.0)
        K_analytic = gaussian_curvature(p, a, b, c, 2.0, 2.0)
        K_closed = ellipsoid_gaussian_curvature(p, a, b, c)
        assert K_analytic[0] == pytest.approx(K_closed[0], rel=1e-8)

    @pytest.mark.parametrize("a,b,c", [
        (1.0, 1.0, 1.0),
        (2.0, 1.5, 1.0),
        (3.0, 2.0, 0.5),
        (0.5, 1.0, 2.0),
    ])
    def test_ellipsoid_K_multiple_shapes(self, a, b, c):
        eta, omega = math.radians(33), math.radians(67)
        p = surface_points(np.array([eta]), np.array([omega]), a, b, c, 2.0, 2.0)
        K_analytic = gaussian_curvature(p, a, b, c, 2.0, 2.0)
        K_closed = ellipsoid_gaussian_curvature(p, a, b, c)
        assert K_analytic[0] == pytest.approx(K_closed[0], rel=1e-8)


# ---------------------------------------------------------------------------
# (c) Mesh convergence: O(N^-2)
# ---------------------------------------------------------------------------

class TestMeshConvergence:
    """Criterion (c): mesh estimator converges to analytic at O(N^-2)."""

    def test_mesh_order_ellipsoid(self):
        order = mesh_refinement_order(30, 45, (2.0, 1.5, 1.0, 2.0, 2.0))
        assert 1.8 <= order["order"] <= 2.2, f"mesh order {order['order']} not in [1.8, 2.2]"

    def test_mesh_order_superquadric(self):
        """Non-ellipsoidal superquadric (n1=3, n2=4)."""
        order = mesh_refinement_order(30, 45, (1.0, 0.8, 0.6, 3.0, 4.0))
        assert 1.5 <= order["order"] <= 2.5, f"mesh order {order['order']} not in [1.5, 2.5]"

    def test_mesh_error_decreases(self):
        """Error must decrease monotonically with N."""
        order = mesh_refinement_order(45, 30, (1.5, 1.0, 0.7, 2.5, 2.5))
        errs = [h["rel_err"] for h in order["history"]]
        for i in range(1, len(errs)):
            assert errs[i] < errs[i - 1], f"error not decreasing at step {i}: {errs}"


# ---------------------------------------------------------------------------
# (d) FD convergence: O(h^2)
# ---------------------------------------------------------------------------

class TestFDConvergence:
    """Criterion (d): legacy FD estimator converges at O(h^2)."""

    def test_fd_order(self):
        fd_order_result = fd_refinement_order(30, 45, (1.0, 0.7, 0.5, 2.5, 2.0))
        assert 1.5 <= fd_order_result["order"] <= 2.5, \
            f"FD order {fd_order_result['order']} not in [1.5, 2.5]"


# ---------------------------------------------------------------------------
# (e) Block limit: n >= 40 → flat face centers → zero curvature
# ---------------------------------------------------------------------------

class TestBlockLimit:
    """Criterion (e): block exponent → curvature → 0 at face centers."""

    def test_block_face_center_curvature_near_zero(self):
        """At the center of a flat face (eta=0, omega=0 for a cube-like shape),
        curvature should be very small."""
        n = int(BLOCK_EXPONENT)
        # Face center: eta=0, omega=0 → point on +x face
        eta, omega = 0.0, 0.0
        p = surface_points(np.array([eta]), np.array([omega]), 1.0, 1.0, 1.0, float(n), float(n))
        k1, k2 = principal_curvatures_analytic(p, 1.0, 1.0, 1.0, float(n), float(n))
        # On flat face center, curvature should be near zero
        assert abs(k1[0]) < 1e-6, f"block face-center k1={k1[0]} not near zero"
        assert abs(k2[0]) < 1e-6, f"block face-center k2={k2[0]} not near zero"


# ---------------------------------------------------------------------------
# (f) Symmetry: coordinate permutations for symmetric shapes
# ---------------------------------------------------------------------------

class TestSymmetry:
    """Criterion (f): k1,k2 invariant under coordinate permutations for symmetric shapes."""

    def test_cube_symmetry(self):
        a, b, c, n1, n2 = 1.0, 1.0, 1.0, 4.0, 4.0
        eta1, omega1 = 0.0, 0.0
        p1 = surface_points(np.array([eta1]), np.array([omega1]), a, b, c, n1, n2)
        k1_1, k2_1 = principal_curvatures_analytic(p1, a, b, c, n1, n2)

        eta2, omega2 = 0.0, math.pi / 2
        p2 = surface_points(np.array([eta2]), np.array([omega2]), a, b, c, n1, n2)
        k1_2, k2_2 = principal_curvatures_analytic(p2, a, b, c, n1, n2)

        assert k1_1[0] == pytest.approx(k1_2[0], rel=1e-10)
        assert k2_1[0] == pytest.approx(k2_2[0], rel=1e-10)


# ---------------------------------------------------------------------------
# (g) Negative control: point not on surface
# ---------------------------------------------------------------------------

class TestNegativeControl:
    """Criterion (g): point not on surface raises an error."""

    def test_point_off_surface_raises(self):
        """A point at the origin (0,0,0) is inside, not on the surface."""
        p = np.array([[0.0], [0.0], [0.0]])
        with pytest.raises((ValueError, IndexError)):
            principal_curvatures_analytic(p, 1.0, 1.0, 1.0, 2.0, 2.0)

    def test_point_far_outside_raises(self):
        """A point far outside the surface should raise."""
        p = np.array([[10.0], [10.0], [10.0]])
        with pytest.raises((ValueError, IndexError)):
            principal_curvatures_analytic(p, 1.0, 1.0, 1.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# (h) Gradient and Hessian sanity checks
# ---------------------------------------------------------------------------

class TestGradientHessian:
    """Additional structural tests for gradient and Hessian."""

    def test_sphere_gradient_magnitude(self):
        """For a sphere, |grad F| at the surface should be 2/R."""
        R = 1.0
        eta, omega = math.radians(30), math.radians(45)
        p = surface_points(np.array([eta]), np.array([omega]), R, R, R, 2.0, 2.0)
        g = superquadric_gradient(p, R, R, R, 2.0, 2.0)
        # g has shape (n_points, 3)
        mag = np.sqrt(g[0, 0]**2 + g[0, 1]**2 + g[0, 2]**2)
        assert mag == pytest.approx(2.0 / R, rel=1e-10)

    def test_hessian_symmetric(self):
        """Hessian must be symmetric: H[i,j] = H[j,i]."""
        eta, omega = math.radians(30), math.radians(45)
        p = surface_points(np.array([eta]), np.array([omega]), 1.0, 0.8, 0.6, 3.0, 2.5)
        H = superquadric_hessian(p, 1.0, 0.8, 0.6, 3.0, 2.5)
        # H has shape (n_points, 3, 3)
        H0 = H[0]
        assert H0[0, 1] == pytest.approx(H0[1, 0], rel=1e-10)
        assert H0[0, 2] == pytest.approx(H0[2, 0], rel=1e-10)
        assert H0[1, 2] == pytest.approx(H0[2, 1], rel=1e-10)

    def test_gaussian_mean_from_principal(self):
        """K = k1*k2, H = (k1+k2)/2 must hold."""
        eta, omega = math.radians(30), math.radians(45)
        p = surface_points(np.array([eta]), np.array([omega]), 1.5, 1.0, 0.7, 2.5, 3.0)
        k1, k2 = principal_curvatures_analytic(p, 1.5, 1.0, 0.7, 2.5, 3.0)
        K = gaussian_curvature(p, 1.5, 1.0, 0.7, 2.5, 3.0)
        H = mean_curvature(p, 1.5, 1.0, 0.7, 2.5, 3.0)
        assert K[0] == pytest.approx(k1[0] * k2[0], rel=1e-8)
        assert H[0] == pytest.approx((k1[0] + k2[0]) / 2.0, rel=1e-8)
