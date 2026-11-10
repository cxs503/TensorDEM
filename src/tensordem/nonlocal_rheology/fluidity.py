"""Nonlocal granular fluidity (NGF) closure for dense granular flow.

Physics
-------
The local ``mu(I)`` rheology (GDR MiDi 2004; Jop, Forterre & Pouliquen 2006)
assumes a *bijective* relation between the stress ratio ``mu = tau/P`` and the
inertial number ``I = gammadot d / sqrt(P/rho_p)``.  It fails wherever the flow
is not locally determined by the stress state: below yield (``mu < mu_s``) the
local law predicts a rigid plug, while experiments and DEM show *creep* whose
rate depends on how far the flowing region is (the classic annular-Couette
"shear band wider than the local prediction" observation).

The nonlocal granular fluidity (NGF) model of Kamrin & Henann (2012) replaces
the strain rate by a *fluidity* field ``g``,

    gammadot = g * mu                                     (1)
    g        = g_loc(mu) + xi^2 * laplacian(g)            (2)
    xi       = A d / sqrt(|mu - mu_s|)   (or A d / sqrt(mu_2 - mu))

where ``g_loc(mu) = gammadot_loc(mu)/mu`` is the fluidity implied by the local
law (zero below yield).  Equation (2) is an elliptic *cooperativity* equation:
fluidity diffuses over the correlation length ``xi``, so a region below yield
embedded in a flowing neighbourhood creeps at a rate decaying over ``xi``, and
the correlation length diverges as the yield stress is approached
(``mu -> mu_s``), which produces the observed band widening.

Numerics
--------
Equation (2) is discretised with a finite-volume (FV) scheme and solved as the
symmetric positive-definite linear system

    (M + xi^2 K) g = M g_loc                                    (3)

on an annulus (1D radial, cylindrical measure ``M_i = r_i dr``) or on a 2D
Cartesian grid.  ``K`` is the symmetric FV stiffness matrix (face
conductances).  The system is solved with a hand-written **conjugate gradient
iteration with a Jacobi (diagonal) preconditioner**; convergence is reported as
the relative 2-norm of the true residual ``||b - A g||/||b||``.  ``xi`` may be
frozen (single linear solve, the default and the case used for calibration) or
re-evaluated from the current ``g`` field in an outer Picard loop.

All quantities are SI (m, kg, s, Pa) unless documented otherwise.  ``xi`` and
``d`` are lengths; ``A = xi/d`` is dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp


class NGFError(ValueError):
    """Base class for nonlocal-fluidity errors."""


# ---------------------------------------------------------------------------
# local rheology  (mu -> I -> gammadot -> g)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JopLaw:
    """Local ``mu(I) = mu_s + (mu_2 - mu_s)/(I_0/I + 1)`` (Jop et al. 2006)."""

    mu_s: float = 0.30
    mu_2: float = 0.70
    i_0: float = 0.30

    def __post_init__(self) -> None:
        if not self.mu_2 > self.mu_s:
            raise NGFError("mu_2 must exceed mu_s")
        if not self.i_0 > 0.0:
            raise NGFError("i_0 must be positive")

    def mu(self, inertia) -> np.ndarray:
        """Forward map ``mu(I)`` (vectorised)."""
        I = np.asarray(inertia, dtype=float)
        if np.any(I < 0.0):
            raise NGFError("inertial number must be nonnegative")
        return self.mu_s + (self.mu_2 - self.mu_s) / (self.i_0 / np.maximum(I, 1e-300) + 1.0)

    def inertia(self, mu) -> np.ndarray:
        """Inverse map ``I(mu)``; ``nan`` where the state is unreachable."""
        m = np.asarray(mu, dtype=float)
        span = self.mu_2 - self.mu_s
        ok = m > self.mu_s
        out = np.full(m.shape, np.nan)
        if not ok.any():
            return out
        frac = span / (m[ok] - self.mu_s) - 1.0
        val = np.full(frac.shape, np.inf)
        pos = frac > 0.0
        val[pos] = self.i_0 / frac[pos]
        out[ok] = val
        return out

    def slope(self) -> float:
        return (self.mu_2 - self.mu_s) / self.i_0


def local_fluidity(mu, pressure, d: float, rho: float, law: JopLaw) -> np.ndarray:
    """``g_loc = gammadot_loc(mu)/mu`` in 1/s; zero where ``mu <= mu_s``."""
    mu = np.asarray(mu, dtype=float)
    pressure = np.broadcast_to(np.asarray(pressure, dtype=float), mu.shape)
    if np.any(pressure <= 0.0):
        raise NGFError("pressure must be positive")
    if d <= 0.0 or rho <= 0.0:
        raise NGFError("d and rho must be positive")
    I_loc = law.inertia(mu)
    scale = np.sqrt(pressure / rho) / d          # 1/s per unit I
    g = np.zeros_like(mu)
    usable = np.isfinite(I_loc) & (mu > law.mu_s)
    g[usable] = I_loc[usable] * scale[usable] / mu[usable]
    return g


def local_strain_rate(mu, pressure, d: float, rho: float, law: JopLaw) -> np.ndarray:
    """``gammadot_loc(mu) = mu g_loc`` (1/s); zero below yield."""
    return np.asarray(mu, dtype=float) * local_fluidity(mu, pressure, d, rho, law)


@dataclass(frozen=True)
class NGFParams:
    """Nonlocal closure parameters: local ``law`` plus ``amplitude = A = xi/d``."""

    law: JopLaw = JopLaw()
    amplitude: float = 0.5
    length_law: str = "yield"      # 'yield' | 'saturation'
    xi_floor_d: float = 1.0e-3     # regularisation, in units of d

    def __post_init__(self) -> None:
        if self.amplitude < 0.0:
            raise NGFError("amplitude must be nonnegative")
        if self.length_law not in ("yield", "saturation"):
            raise NGFError("length_law must be 'yield' or 'saturation'")
        if self.xi_floor_d <= 0.0:
            raise NGFError("xi_floor_d must be positive")

    def xi(self, mu, d: float) -> np.ndarray:
        """Cooperativity length ``xi(mu)`` in metres."""
        mu = np.asarray(mu, dtype=float)
        floor = self.xi_floor_d ** 2
        if self.length_law == "yield":
            gap = np.maximum(np.abs(mu - self.law.mu_s), floor)
        else:
            gap = np.maximum(np.abs(self.law.mu_2 - mu), floor)
        return self.amplitude * d / np.sqrt(gap)


# ---------------------------------------------------------------------------
# finite-volume operators  (M = cell measure, K = symmetric stiffness)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RadialGrid:
    """1D radial grid of an annulus (per radian of azimuth, unit thickness)."""

    r_inner: float
    r_outer: float
    n_cells: int

    def __post_init__(self) -> None:
        if not (0.0 < self.r_inner < self.r_outer):
            raise NGFError("need 0 < r_inner < r_outer")
        if self.n_cells < 2:
            raise NGFError("need at least two radial cells")

    @property
    def dr(self) -> float:
        return (self.r_outer - self.r_inner) / self.n_cells

    @property
    def centers(self) -> np.ndarray:
        return self.r_inner + (np.arange(self.n_cells) + 0.5) * self.dr

    @property
    def faces(self) -> np.ndarray:
        """Interior face radii (n_cells - 1 values)."""
        return self.r_inner + np.arange(1, self.n_cells) * self.dr

    @property
    def measure(self) -> np.ndarray:
        """Cell measure ``r_i dr`` (proportional to the cell area)."""
        return self.centers * self.dr

    def operator(self, xi2=None) -> sp.csr_matrix:
        """Symmetric FV stiffness for ``(1/r) d/dr (r d/dr)``.

        With ``xi2`` (cell-centred ``xi^2``) the face conductance is weighted by
        the arithmetic face mean of ``xi^2``; the operator stays symmetric
        positive semi-definite, which is what the CG solve requires
        (``diag(xi^2) @ K`` would be non-symmetric and must not be used).
        """
        n = self.n_cells
        cond = self.faces / self.dr
        if xi2 is not None:
            w = np.asarray(xi2, dtype=float)
            if w.shape != (n,):
                raise NGFError(f"xi2 must have shape ({n},)")
            if np.any(w < 0.0):
                raise NGFError("xi2 must be nonnegative")
            cond = cond * 0.5 * (w[:-1] + w[1:])
        rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
        cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
        vals = np.concatenate([-cond, -cond])
        K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        return K - sp.diags(np.asarray(K.sum(axis=1)).ravel())

    def gradient(self, field) -> np.ndarray:
        """Cell-centred radial derivative ``dg/dr`` (units 1/m)."""
        return np.gradient(np.asarray(field, dtype=float), self.dr)


def cartesian_2d_operator(nx: int, ny: int, h: float, active=None, xi2=None):
    """2D Cartesian FV operator; returns ``(K, measure, index)``.

    ``K`` is the *volume-integrated* symmetric five-point stiffness (face
    conductance ``h/h = 1``, so ``K g`` discretises ``-int laplacian(g) dV``),
    ``measure`` is the cell area ``h^2`` and ``index`` maps ``(i, j) -> unknown
    id`` (``-1`` when inactive).  The two are dimensionally consistent: the
    assembled matrix of ``g - xi^2 laplacian(g) = s`` is ``M + xi^2 K``.
    """
    if active is None:
        active = np.ones((nx, ny), dtype=bool)
    active = np.asarray(active, dtype=bool)
    index = -np.ones((nx, ny), dtype=int)
    index[active] = np.arange(int(active.sum()))
    n = int(active.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for dx, dy in ((1, 0), (0, 1)):
        a = index[0:nx - dx, 0:ny - dy]
        b = index[dx:nx, dy:ny]
        both = (a >= 0) & (b >= 0)
        if not both.any():
            continue
        ai = a[both]
        bi = b[both]
        cond = np.ones(ai.size)                 # face length h / distance h
        if xi2 is not None:
            w = np.asarray(xi2, dtype=float).reshape(-1)
            cond = cond * 0.5 * (w[ai] + w[bi])
        rows.append(ai); cols.append(bi); vals.append(-cond)
        rows.append(bi); cols.append(ai); vals.append(-cond)
        diag += np.bincount(ai, weights=cond, minlength=n)
        diag += np.bincount(bi, weights=cond, minlength=n)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    return K, np.full(n, h * h), index


# ---------------------------------------------------------------------------
# conjugate gradient with a Jacobi preconditioner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CGReport:
    iterations: int
    residual: float                    # ||b - A x||_2 / ||b||_2
    residual_history: Tuple[float, ...]
    converged: bool


def cg_jacobi(A: sp.spmatrix, b: np.ndarray, x0=None, *, rtol: float = 1e-12,
              atol: float = 0.0, maxiter: int = 20000,
              diag=None) -> Tuple[np.ndarray, CGReport]:
    """Conjugate gradient with a diagonal (Jacobi) preconditioner.

    Convergence test: ``||b - A x||_2 <= rtol ||b||_2 + atol`` (true residual,
    recomputed from the recursion every iteration).
    """
    A = sp.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    n = b.size
    diag = np.asarray(A.diagonal() if diag is None else diag, dtype=float)
    if np.any(diag <= 0.0):
        raise NGFError("Jacobi preconditioner requires a positive diagonal")
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    b_norm = float(np.linalg.norm(b))
    target = rtol * b_norm + atol
    r = b - A @ x
    z = r / diag
    p = z.copy()
    rz = float(r @ z)
    hist = [float(np.linalg.norm(r))]
    if b_norm == 0.0 or hist[0] <= target:
        return x, CGReport(0, hist[0] / max(b_norm, 1e-300), tuple(hist), True)
    for it in range(1, maxiter + 1):
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            raise NGFError("operator is not positive definite (pAp <= 0)")
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        hist.append(float(np.linalg.norm(r)))
        rel = hist[-1] / b_norm
        if hist[-1] <= target or rel <= rtol:
            return x, CGReport(it, rel, tuple(hist), True)
        z = r / diag
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, CGReport(maxiter, hist[-1] / b_norm, tuple(hist), False)


# ---------------------------------------------------------------------------
# NGF solve
# ---------------------------------------------------------------------------
@dataclass
class NGFResult:
    g: np.ndarray
    g_loc: np.ndarray
    xi: np.ndarray
    iterations: int
    residual: float
    residual_history: Tuple[float, ...]
    converged: bool
    free: np.ndarray | None = None

    @property
    def amplification(self) -> np.ndarray:
        """``g/g_loc`` — the nonlocal amplification (``inf`` below yield)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.g_loc > 0.0, self.g / self.g_loc, np.inf)


def solve_ngf(mu, pressure, d: float, rho: float, params: NGFParams,
              grid: RadialGrid, *, inner: str = "neumann", outer: str = "neumann",
              rtol: float = 1e-12, maxiter: int = 20000,
              fixed_point_iters: int = 1, xi_field=None) -> NGFResult:
    """Solve ``g = g_loc(mu) + xi^2 laplacian(g)`` on an annular grid.

    ``mu`` and ``pressure`` are cell-centred.  Wall conditions: ``neumann``
    (zero fluidity flux — fluidity cannot leave the material; the physical
    default) or ``dirichlet`` (``g = g_loc`` at the wall).
    """
    mu = np.asarray(mu, dtype=float)
    pressure = np.broadcast_to(np.asarray(pressure, dtype=float), mu.shape)
    n = grid.n_cells
    if mu.shape != (n,):
        raise NGFError(f"mu must have shape ({n},)")

    g_loc = local_fluidity(mu, pressure, d, rho, params.law)
    M = sp.diags(grid.measure)
    for side, kind in (("inner", inner), ("outer", outer)):
        if kind not in ("neumann", "dirichlet"):
            raise NGFError(f"{side} wall condition must be 'neumann' or 'dirichlet'")
    fixed = np.zeros(n, dtype=bool)
    if inner == "dirichlet":
        fixed[0] = True
    if outer == "dirichlet":
        fixed[-1] = True
    free = ~fixed
    if not free.any():
        raise NGFError("no free cells in the NGF system")

    g = g_loc.copy()
    rep = CGReport(0, 0.0, (0.0,), True)
    xi_used = params.xi(mu, d) if xi_field is None else np.asarray(xi_field, dtype=float)
    for _ in range(max(1, int(fixed_point_iters))):
        if fixed_point_iters > 1:
            g_eff = np.maximum(g, 0.0)
            mu_eff = np.where(g_eff > 0.0, np.asarray(mu, dtype=float), params.law.mu_s)
            xi_used = params.xi(mu_eff, d) if xi_field is None else np.asarray(xi_field, dtype=float)
        # Dirichlet cells are prescribed g = g_loc and eliminated from the
        # unknown set; their contribution moves to the right-hand side.
        xi2 = np.asarray(xi_used, dtype=float) ** 2
        A_full = (M + grid.operator(xi2)).tocsr()
        rhs = M @ g_loc
        A = A_full[free][:, free].tocsr()
        rhs_eff = rhs[free] - A_full[free][:, fixed] @ g_loc[fixed]
        x0 = g[free] if fixed_point_iters > 1 else None
        sol, rep = cg_jacobi(A, rhs_eff, x0, rtol=rtol, maxiter=maxiter)
        g = g_loc.copy()
        g[free] = sol
        if not rep.converged:
            if fixed_point_iters > 1:
                raise NGFError("NGF Picard iteration failed to converge")
            break
    return NGFResult(g=g, g_loc=g_loc, xi=xi_used, iterations=rep.iterations,
                     residual=rep.residual, residual_history=rep.residual_history,
                     converged=rep.converged, free=free)


def mu_from_strain_rate(strain_rate, pressure, d: float, rho: float,
                        law: JopLaw) -> np.ndarray:
    """Predict ``mu`` from a shear rate through the local law (``I = gammadot d/sqrt(P/rho)``)."""
    gammadot = np.asarray(strain_rate, dtype=float)
    pressure = np.broadcast_to(np.asarray(pressure, dtype=float), gammadot.shape)
    I = np.maximum(gammadot, 0.0) * d / np.sqrt(pressure / rho)
    return law.mu(I)


def nonlocal_strain_rate(mu, pressure, d: float, rho: float, params: NGFParams,
                         grid: RadialGrid, **kwargs):
    """``gammadot = g mu`` with ``g`` from the NGF solve; returns (rate, result)."""
    res = solve_ngf(mu, pressure, d, rho, params, grid, **kwargs)
    return res.g * np.asarray(mu, dtype=float), res


__all__ = [
    "NGFError", "JopLaw", "NGFParams", "NGFResult", "RadialGrid", "CGReport",
    "cg_jacobi", "cartesian_2d_operator", "local_fluidity", "local_strain_rate",
    "solve_ngf", "mu_from_strain_rate", "nonlocal_strain_rate",
]