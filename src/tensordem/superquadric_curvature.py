"""Closed-form curvature of superquadric surfaces (no finite differences).

Campaign node **P165-W2-N06** (TensorDEM).  The parent maturity node
(:mod:`tensordem.superquadric_kernel`) computed the curvature of the implicit
surface ``F = 0`` by *central differences of the analytic gradient*.  That
estimator carries ~15 % relative error on the principal curvatures of
non-ellipsoidal shapes at the step size used in production (``eps_rel = 1e-6``
of the largest semi-axis); see ``docs/dem_contact_mechanics_verification_report.md``
(C4 and limitation 1).  This module replaces it with the exact analytic Hessian
of the implicit function.

Geometry
--------
Body frame, semi-axes ``a, b, c`` and shape exponents ``n1, n2``::

    S = |x/a|^n2 + |y/b|^n2,     H = S^(n1/n2)          (horizontal part)
    F = H + |z/c|^n1 - 1                               (F = 0 on the surface)

Writing ``u = |x|/a``, ``v = |y|/b``, ``w = |z|/c`` and ``p = n2``, ``q = n1``
the first and second derivatives are exactly::

    F_x  = q S^(q/p - 1) u^(p-1) sgn(x) / a
    F_xx = q (q-p) S^(q/p - 2) u^(2p-2) / a^2 + q (p-1) S^(q/p - 1) u^(p-2) / a^2
    F_xy = q (q-p) S^(q/p - 2) u^(p-1) v^(p-1) sgn(x)sgn(y) / (a b)
    F_xz = F_yz = 0
    F_zz = q (q-1) w^(q-2) / c^2

so no numerical differentiation of a numerical object is involved.  (At the
pole ``u = v = 0`` the two singular-looking terms combine into ``O(u^(q-2))``
and the implemented expressions are their finite limits for ``q >= 2``.)

Curvatures from the implicit surface (outward gradient ``g``, Hessian ``Hm``)::

    K     = g^T adj(Hm) g / |g|^4                     (Gaussian, 1/m^2)
    Hm_   = (|g|^2 tr(Hm) - g^T Hm g) / (2 |g|^3)     (mean, convex > 0)
    k1,2  = Hm_ +- sqrt(Hm_^2 - K)                    (principal, convex > 0)

The sign convention is the one of
:func:`tensordem.superquadric_kernel.principal_curvatures_local`: a sphere
reports ``k1 = k2 = +1/R``.

An independent *mesh-based* reference estimator is included
(:func:`principal_curvatures_mesh`): it samples the parametric surface

    x = a cos^e1(eta) cos^e2(omega),  y = b cos^e1(eta) sin^e2(omega),
    z = c sin^e1(eta),                e1 = 2/n1, e2 = 2/n2

and obtains the curvatures from the first and second fundamental forms with
central differences of parametric step ``pi/N``.  Its truncation error is
``O(N^-2)``, so it independently validates the analytic formulas and provides
the mesh-refinement convergence order used in the acceptance criteria.

All quantities are SI (m, 1/m).  For ``n1, n2 > 40`` the implicit model of
:func:`tensordem.superquadric_kernel.implicit_local` switches to the max-norm
(block) branch whose faces are flat by construction; the routines below return
zero curvature there.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "BLOCK_EXPONENT",
    "superquadric_gradient",
    "superquadric_hessian",
    "superquadric_gradient_hessian",
    "gaussian_curvature",
    "mean_curvature",
    "gaussian_mean_curvature",
    "principal_curvatures_analytic",
    "principal_curvatures_fd",
    "surface_points",
    "principal_curvatures_mesh",
    "ellipsoid_gaussian_curvature",
    "curvature_table",
    "mesh_refinement_order",
    "fd_refinement_order",
]

DTYPE_NP = np.float64
BLOCK_EXPONENT = 40.0     # implicit_local switches to the max-norm branch here


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_m(x: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(x, dtype=DTYPE_NP))


def _broadcast(v, m: int) -> np.ndarray:
    v = np.asarray(v, dtype=DTYPE_NP)
    if v.ndim == 0:
        return np.full(m, float(v), dtype=DTYPE_NP)
    return np.broadcast_to(v.reshape(-1), (m,)).astype(DTYPE_NP)


def _params(x: np.ndarray, a, b, c, n1, n2):
    m = x.shape[0]
    return (_broadcast(a, m), _broadcast(b, m), _broadcast(c, m),
            _broadcast(n1, m), _broadcast(n2, m))


def _normalised(x: np.ndarray, a, b, c, n2):
    u = np.abs(x[:, 0]) / a
    v = np.abs(x[:, 1]) / b
    w = np.abs(x[:, 2]) / c
    with np.errstate(over="ignore", invalid="ignore"):
        s = u ** n2 + v ** n2
    return u, v, w, np.maximum(s, 1e-300)


# ---------------------------------------------------------------------------
# gradient / Hessian
# ---------------------------------------------------------------------------

def superquadric_gradient(x: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Analytic gradient of the implicit function (body frame, ``(M,3)``).

    Algebraically identical to
    :func:`tensordem.superquadric_kernel.gradient_local`; written in normalised
    coordinates so it shares intermediates with the Hessian.
    """
    x = _as_m(x)
    a, b, c, n1, n2 = _params(x, a, b, c, n1, n2)
    u, v, w, s = _normalised(x, a, b, c, n2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sh = s ** (n1 / n2 - 1.0)
        gx = n1 * sh * u ** (n2 - 1.0) / a * np.sign(x[:, 0])
        gy = n1 * sh * v ** (n2 - 1.0) / b * np.sign(x[:, 1])
        gz = n1 * w ** (n1 - 1.0) / c * np.sign(x[:, 2])
    g = np.column_stack([gx, gy, gz])
    return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def superquadric_hessian(x: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Analytic Hessian of the implicit function (body frame, ``(M,3,3)``).

    ``F_xz`` and ``F_yz`` vanish identically because ``F`` separates into a
    horizontal part plus ``|z/c|^n1``.
    """
    x = _as_m(x)
    m = x.shape[0]
    a, b, c, n1, n2 = _params(x, a, b, c, n1, n2)
    u, v, w, s = _normalised(x, a, b, c, n2)
    hess = np.zeros((m, 3, 3), dtype=DTYPE_NP)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sh1 = s ** (n1 / n2 - 1.0)
        sh2 = s ** (n1 / n2 - 2.0)
        d_h = (n1 * (n1 - n2) * sh2 * u ** (2.0 * n2 - 2.0)
               + n1 * (n2 - 1.0) * sh1 * u ** (n2 - 2.0)) / a ** 2
        d_v = (n1 * (n1 - n2) * sh2 * v ** (2.0 * n2 - 2.0)
               + n1 * (n2 - 1.0) * sh1 * v ** (n2 - 2.0)) / b ** 2
        d_xy = (n1 * (n1 - n2) * sh2 * u ** (n2 - 1.0) * v ** (n2 - 1.0)
                * np.sign(x[:, 0]) * np.sign(x[:, 1])) / (a * b)
        d_z = n1 * (n1 - 1.0) * w ** (n1 - 2.0) / c ** 2
    hess[:, 0, 0] = d_h
    hess[:, 1, 1] = d_v
    hess[:, 2, 2] = d_z
    hess[:, 0, 1] = hess[:, 1, 0] = d_xy
    hess = np.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)
    block = (n1 > BLOCK_EXPONENT) & (n2 > BLOCK_EXPONENT)
    if np.any(block):
        hess[block] = 0.0
    grad = superquadric_gradient(x, a, b, c, n1, n2)
    if np.any(block):
        grad[block] = 0.0
    return hess


def superquadric_gradient_hessian(x: np.ndarray, a, b, c, n1, n2):
    """``(gradient, Hessian)`` in one call."""
    return (superquadric_gradient(x, a, b, c, n1, n2),
            superquadric_hessian(x, a, b, c, n1, n2))


# ---------------------------------------------------------------------------
# curvature formulas
# ---------------------------------------------------------------------------

def _gaussian_mean_from(g: np.ndarray, hess: np.ndarray):
    m = g.shape[0]
    gn2 = np.maximum(np.sum(g * g, axis=1), 1e-300)
    gn = np.sqrt(gn2)
    tr = np.trace(hess, axis1=1, axis2=2)
    hg = np.einsum("mij,mj->mi", hess, g)
    g_hg = np.sum(g * hg, axis=1)
    h = hess
    adj = np.empty((m, 3, 3), dtype=DTYPE_NP)
    adj[:, 0, 0] = h[:, 1, 1] * h[:, 2, 2] - h[:, 1, 2] * h[:, 2, 1]
    adj[:, 0, 1] = h[:, 0, 2] * h[:, 2, 1] - h[:, 0, 1] * h[:, 2, 2]
    adj[:, 0, 2] = h[:, 0, 1] * h[:, 1, 2] - h[:, 0, 2] * h[:, 1, 1]
    adj[:, 1, 0] = h[:, 1, 2] * h[:, 2, 0] - h[:, 1, 0] * h[:, 2, 2]
    adj[:, 1, 1] = h[:, 0, 0] * h[:, 2, 2] - h[:, 0, 2] * h[:, 2, 0]
    adj[:, 1, 2] = h[:, 0, 2] * h[:, 1, 0] - h[:, 0, 0] * h[:, 1, 2]
    adj[:, 2, 0] = h[:, 1, 0] * h[:, 2, 1] - h[:, 1, 1] * h[:, 2, 0]
    adj[:, 2, 1] = h[:, 0, 1] * h[:, 2, 0] - h[:, 0, 0] * h[:, 2, 1]
    adj[:, 2, 2] = h[:, 0, 0] * h[:, 1, 1] - h[:, 0, 1] * h[:, 1, 0]
    g_adj_g = np.einsum("mi,mij,mj->m", g, adj, g)
    k_gauss = g_adj_g / (gn2 * gn2)
    mean = (gn2 * tr - g_hg) / (2.0 * gn2 * gn)
    return k_gauss, mean


def gaussian_mean_curvature(x: np.ndarray, a, b, c, n1, n2):
    """``(K, Hm)`` closed form, ``Hm`` positive on a convex body."""
    g, hess = superquadric_gradient_hessian(x, a, b, c, n1, n2)
    return _gaussian_mean_from(g, hess)


def gaussian_curvature(x: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Gaussian curvature ``K`` (1/m^2)."""
    return gaussian_mean_curvature(x, a, b, c, n1, n2)[0]


def mean_curvature(x: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Mean curvature ``Hm`` (1/m), positive on a convex body."""
    return gaussian_mean_curvature(x, a, b, c, n1, n2)[1]


def principal_curvatures_analytic(x: np.ndarray, a, b, c, n1, n2):
    """Principal curvatures ``(k1, k2)``, ``k1 >= k2``, closed form."""
    k, mean = gaussian_mean_curvature(x, a, b, c, n1, n2)
    root = np.sqrt(np.maximum(mean * mean - k, 0.0))
    return mean + root, mean - root


def _principal_from_hessian(nrm: np.ndarray, hess: np.ndarray, gn: np.ndarray):
    m = nrm.shape[0]
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
    alt = np.abs(nrm[:, 0]) > 0.9
    ref[alt] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(nrm, ref)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1), 1e-30)[:, None]
    t2 = np.cross(nrm, t1)
    s11 = np.einsum("mi,mij,mj->m", t1, hess, t1) / gn
    s12 = np.einsum("mi,mij,mj->m", t1, hess, t2) / gn
    s22 = np.einsum("mi,mij,mj->m", t2, hess, t2) / gn
    tr = 0.5 * (s11 + s22)
    det = s11 * s22 - s12 * s12
    root = np.sqrt(np.maximum(tr * tr - det, 0.0))
    return tr + root, tr - root


def principal_curvatures_fd(x: np.ndarray, a, b, c, n1, n2,
                            eps_rel: float = 1e-6):
    """Legacy finite-difference estimator (baseline / negative control).

    Central differences of the analytic gradient build the Hessian; the shape
    operator is restricted to the tangent plane.  This is the parent node's
    estimator (~15 % error on non-ellipsoidal shapes) and is kept so the
    improvement can be measured against the original baseline.
    """
    x = _as_m(x)
    m = x.shape[0]
    a, b, c, n1, n2 = _params(x, a, b, c, n1, n2)
    g = superquadric_gradient(x, a, b, c, n1, n2)
    gn = np.maximum(np.linalg.norm(g, axis=1), 1e-30)
    nrm = g / gn[:, None]
    h = np.maximum(np.maximum(np.maximum(a, b), c) * eps_rel, 1e-15)
    hess = np.empty((m, 3, 3))
    for j in range(3):
        dx = np.zeros((m, 3))
        dx[:, j] = h
        gp = superquadric_gradient(x + dx, a, b, c, n1, n2)
        gm = superquadric_gradient(x - dx, a, b, c, n1, n2)
        hess[:, :, j] = (gp - gm) / (2.0 * h)[:, None]
    hess = 0.5 * (hess + np.transpose(hess, (0, 2, 1)))
    return _principal_from_hessian(nrm, hess, gn)


# ---------------------------------------------------------------------------
# independent mesh-based reference estimator
# ---------------------------------------------------------------------------

def _signed_pow(t: np.ndarray, e: float) -> np.ndarray:
    t = np.asarray(t, dtype=DTYPE_NP)
    return np.sign(t) * np.abs(t) ** e


def surface_points(eta, omega, a: float, b: float, c: float, n1: float,
                   n2: float) -> np.ndarray:
    """Parametric superquadric surface ``r(eta, omega)`` (body frame, ``(M,3)``)."""
    e1 = 2.0 / float(n1)
    e2 = 2.0 / float(n2)
    eta = np.atleast_1d(np.asarray(eta, dtype=DTYPE_NP))
    omega = np.atleast_1d(np.asarray(omega, dtype=DTYPE_NP))
    eta, omega = np.broadcast_arrays(eta, omega)
    ce = _signed_pow(np.cos(eta), e1)
    x = a * ce * _signed_pow(np.cos(omega), e2)
    y = b * ce * _signed_pow(np.sin(omega), e2)
    z = c * _signed_pow(np.sin(eta), e1)
    return np.column_stack([x, y, z])


def principal_curvatures_mesh(eta: float, omega: float, a: float, b: float,
                              c: float, n1: float, n2: float,
                              n_rings: int = 64):
    """Curvatures from a parametric mesh of resolution ``n_rings``.

    Central differences of width ``h = pi/n_rings`` (one mesh cell); first and
    second fundamental forms give the shape operator, whose eigenvalues are the
    principal curvatures.  Truncation error ``O(h^2)``.  Returns
    ``(k1, k2, K)`` with ``k1 >= k2``, convex-positive.
    """
    h = math.pi / float(n_rings)

    def f(e, o):
        e = np.array([e])
        o = np.array([o])
        return surface_points(e, o, a, b, c, n1, n2)[0]

    r0 = f(eta, omega)
    r_e_p, r_e_m = f(eta + h, omega), f(eta - h, omega)
    r_o_p, r_o_m = f(eta, omega + h), f(eta, omega - h)
    r_e = (r_e_p - r_e_m) / (2.0 * h)
    r_o = (r_o_p - r_o_m) / (2.0 * h)
    r_ee = (r_e_p - 2.0 * r0 + r_e_m) / h ** 2
    r_oo = (r_o_p - 2.0 * r0 + r_o_m) / h ** 2
    r_eo = (f(eta + h, omega + h) - f(eta + h, omega - h)
            - f(eta - h, omega + h) + f(eta - h, omega - h)) / (4.0 * h ** 2)
    nrm = np.cross(r_e, r_o)
    nn = float(np.linalg.norm(nrm))
    if nn < 1e-300:
        return 0.0, 0.0, 0.0
    nrm = nrm / nn
    E, F_, G = float(r_e @ r_e), float(r_e @ r_o), float(r_o @ r_o)
    L, M, N = float(nrm @ r_ee), float(nrm @ r_eo), float(nrm @ r_oo)
    det1 = E * G - F_ * F_
    if abs(det1) < 1e-300:
        return 0.0, 0.0, 0.0
    a11 = (G * L - F_ * M) / det1
    a12 = (G * M - F_ * N) / det1
    a21 = (E * M - F_ * L) / det1
    a22 = (E * N - F_ * M) / det1
    tr = 0.5 * (a11 + a22)
    det = a11 * a22 - a12 * a21
    root = math.sqrt(max(tr * tr - det, 0.0))
    return float(tr + root), float(tr - root), float(det)


def ellipsoid_gaussian_curvature(x, a: float, b: float, c: float) -> np.ndarray:
    """Textbook Gaussian curvature of the ellipsoid ``x^2/a^2+y^2/b^2+z^2/c^2=1``.

    ``K = 1 / (a^2 b^2 c^2 (x^2/a^4 + y^2/b^4 + z^2/c^4)^2)``; an exact
    independent reference for the analytic implicit formula at ``n1 = n2 = 2``.
    """
    x = _as_m(x)
    s = (x[:, 0] / a ** 2) ** 2 + (x[:, 1] / b ** 2) ** 2 + (x[:, 2] / c ** 2) ** 2
    return 1.0 / ((a * b * c) ** 2 * s ** 2)


# ---------------------------------------------------------------------------
# convergence / error studies
# ---------------------------------------------------------------------------

def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


def curvature_table(shapes: Sequence[Tuple[float, float, float, float, float]],
                    params: Sequence[Tuple[float, float]],
                    n_rings: int = 128) -> Dict[str, object]:
    """Analytic-vs-mesh curvature comparison over shapes and surface locations.

    ``shapes``: ``(a, b, c, n1, n2)``; ``params``: ``(eta_deg, omega_deg)``.
    """
    rows: List[Dict[str, float]] = []
    worst = 0.0
    for (a, b, c, n1, n2) in shapes:
        for eta_deg, omega_deg in params:
            eta, omega = math.radians(eta_deg), math.radians(omega_deg)
            p = surface_points(np.array([eta]), np.array([omega]), a, b, c, n1, n2)
            ka1, ka2 = principal_curvatures_analytic(p, a, b, c, n1, n2)
            km1, km2, kmK = principal_curvatures_mesh(eta, omega, a, b, c, n1, n2,
                                                     n_rings)
            scale = max(abs(float(ka1[0])), abs(float(ka2[0])), 1e-30)
            e1 = abs(float(ka1[0]) - km1) / scale
            e2 = abs(float(ka2[0]) - km2) / scale
            K_an = float(gaussian_curvature(p, a, b, c, n1, n2)[0])
            worst = max(worst, e1, e2)
            rows.append({
                "a": a, "b": b, "c": c, "n1": n1, "n2": n2,
                "eta_deg": eta_deg, "omega_deg": omega_deg,
                "k1_analytic": float(ka1[0]), "k2_analytic": float(ka2[0]),
                "k1_mesh": km1, "k2_mesh": km2,
                "K_analytic": K_an, "K_mesh": kmK,
                "rel_err_k1": e1, "rel_err_k2": e2, "rel_err_max": max(e1, e2),
            })
    return {"n_rings": n_rings, "rows": rows,
            "max_rel_err": worst, "n_cases": len(rows)}


def mesh_refinement_order(eta_deg: float, omega_deg: float, shape,
                          rings: Sequence[int] = (4, 8, 16, 32, 64)) -> Dict[str, object]:
    """Refinement study of the mesh estimator against the analytic value.

    Returns the error history and the fitted order ``p`` of ``err ~ N^-p``
    (log-log least squares on the finest half of the sequence).
    """
    a, b, c, n1, n2 = shape
    eta, omega = math.radians(eta_deg), math.radians(omega_deg)
    p = surface_points(np.array([eta]), np.array([omega]), a, b, c, n1, n2)
    ka1, ka2 = principal_curvatures_analytic(p, a, b, c, n1, n2)
    k1, k2 = float(ka1[0]), float(ka2[0])
    scale = max(abs(k1), abs(k2), 1e-30)
    hist = []
    for n in rings:
        m1, m2, _ = principal_curvatures_mesh(eta, omega, a, b, c, n1, n2, n)
        err = max(abs(k1 - m1), abs(k2 - m2)) / scale
        hist.append({"n_rings": int(n), "k1_mesh": m1, "k2_mesh": m2,
                     "rel_err": err})
    half = hist[len(hist) // 2:]
    x = np.log(np.array([h["n_rings"] for h in half], dtype=float))
    y = np.log(np.maximum(np.array([h["rel_err"] for h in half], dtype=float), 1e-300))
    slope = float(-np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")
    return {"shape": list(shape), "eta_deg": eta_deg, "omega_deg": omega_deg,
            "k1_analytic": k1, "k2_analytic": k2, "history": hist,
            "order": slope}


def fd_refinement_order(eta_deg: float, omega_deg: float, shape,
                        eps_seq: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5)) -> Dict[str, object]:
    """Step-size refinement of the legacy FD estimator against the analytic value.

    ``err(h) ~ h^p``; a value ``p`` close to 2 is the expected central-difference
    order.  This documents that the parent node's 15 % error is a *step-size*
    artefact of ``eps_rel = 1e-6`` (round-off dominated) rather than a formula
    error.
    """
    a, b, c, n1, n2 = shape
    eta, omega = math.radians(eta_deg), math.radians(omega_deg)
    p = surface_points(np.array([eta]), np.array([omega]), a, b, c, n1, n2)
    ka1, ka2 = principal_curvatures_analytic(p, a, b, c, n1, n2)
    k1, k2 = float(ka1[0]), float(ka2[0])
    scale = max(abs(k1), abs(k2), 1e-30)
    hist = []
    for eps in eps_seq:
        f1, f2 = principal_curvatures_fd(p, a, b, c, n1, n2, eps_rel=eps)
        err = max(abs(k1 - float(f1[0])), abs(k2 - float(f2[0]))) / scale
        hist.append({"eps_rel": eps, "k1_fd": float(f1[0]), "k2_fd": float(f2[0]),
                     "rel_err": err})
    x = np.log(np.array([h["eps_rel"] for h in hist], dtype=float))
    y = np.log(np.maximum(np.array([h["rel_err"] for h in hist], dtype=float), 1e-300))
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")
    prod_err = [h for h in hist if h["eps_rel"] == 1e-6]
    return {"shape": list(shape), "eta_deg": eta_deg, "omega_deg": omega_deg,
            "k1_analytic": k1, "k2_analytic": k2, "history": hist,
            "order": slope, "production_eps_rel_err": (
                prod_err[0]["rel_err"] if prod_err else
                max(h["rel_err"] for h in hist))}