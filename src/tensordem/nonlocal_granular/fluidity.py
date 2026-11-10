"""Nonlocal granular fluidity (NGF) constitutive solver.

Physics
-------
Dense granular flows are described by the local :math:`\\mu(I)` rheology

.. math::
    \\mu(I)=\\mu_s+\\frac{\\mu_2-\\mu_s}{I_0/I+1},\\qquad
    I=\\frac{\\dot\\gamma d}{\\sqrt{P/\\rho_p}}

which has *no intrinsic length scale*: it cannot produce a shear band of a
finite, size-independent thickness.  The nonlocal granular fluidity model
(Kamrin & Koval, PRL 108, 178301 (2012); Henann & Kamrin, PNAS 110, 6730
(2013)) introduces the fluidity field

.. math::
    g \\equiv \\frac{\\dot\\gamma}{\\mu}

which obeys the elliptic (Helmholtz-type) balance

.. math::
    g - \\xi^2 \\nabla^2 g = g_{loc},\\qquad
    g_{loc}=\\frac{\\dot\\gamma_{loc}(\\mu)}{\\mu},\\;
    \\dot\\gamma_{loc}=\\frac{I(\\mu)\\,\\sqrt{P/\\rho_p}}{d}

with :math:`\\xi` the *cooperative length*.  Because the equation carries the
length :math:`\\xi`, the localisation width of the deformation (shear band)
scales with :math:`\\xi` -- i.e. with a few grain diameters -- and therefore
does **not** vanish as the grain size (or the cell size) is changed.  That is
the capability this module provides and quantifies.

Boundary conditions
-------------------
* ``dirichlet`` : wall fluidity imposed by the wall (rough wall drags material)
* ``neumann``   : zero fluidity flux -- free surface / symmetry plane

Discretisation
--------------
Finite volume on a uniform grid, face-flux form.  The linear system is solved
matrix-free with conjugate gradients preconditioned by the (exact) diagonal
(Jacobi), so the solver never forms a matrix and stays differentiable /
torch-native.  For a Dirichlet face at distance ``h/2`` from the cell centre
the face flux is ``(g_bc - g_0)/(h/2)`` which makes the boundary operator
consistent (no half-cell smearing of the band width).

All quantities are SI (m, kg, s, Pa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

import torch

DTYPE = torch.float64
BCKind = Literal["dirichlet", "neumann"]


class NGFError(ValueError):
    """Raised when the NGF problem is ill-posed."""


# --------------------------------------------------------------------------- #
# mu(I) rheology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MuIParams:
    """Jop/Forterre/Pouliquen mu(I) parameters (glass-bead default set)."""

    mu_s: float = 0.38   # static friction (I -> 0)
    mu_2: float = 0.64   # dynamic friction (I -> inf)
    I_0: float = 0.30    # characteristic inertial number

    def __post_init__(self) -> None:
        if not (0.0 < self.mu_s < self.mu_2):
            raise NGFError(f"need 0<mu_s<mu_2, got {self.mu_s}, {self.mu_2}")
        if self.I_0 <= 0.0:
            raise NGFError("I_0 must be > 0")


def mu_of_I(I: torch.Tensor | float, p: MuIParams = MuIParams()) -> torch.Tensor:
    """mu(I) law; ``I=0`` gives ``mu_s``, ``I->inf`` gives ``mu_2``."""
    I = torch.as_tensor(I, dtype=DTYPE)
    return p.mu_s + (p.mu_2 - p.mu_s) / (p.I_0 / torch.clamp(I, min=1e-300) + 1.0)


def I_of_mu(mu: torch.Tensor | float, p: MuIParams = MuIParams()) -> torch.Tensor:
    """Invert ``mu(I)``: ``I = I_0 (mu - mu_s)/(mu_2 - mu)``.

    Clamped to ``[0, inf)`` (no flow below yield) and capped below ``mu_2``.
    """
    mu = torch.as_tensor(mu, dtype=DTYPE)
    mu_clip = torch.clamp(mu, min=p.mu_s, max=p.mu_2 * (1.0 - 1e-9))
    I = p.I_0 * (mu_clip - p.mu_s) / (p.mu_2 - mu_clip)
    return torch.where(mu > p.mu_s, I, torch.zeros_like(I))


def local_strain_rate(
    mu: torch.Tensor | float,
    pressure: torch.Tensor | float,
    d: float,
    rho_p: float,
    p: MuIParams = MuIParams(),
) -> torch.Tensor:
    """Local (Bagnold-type) shear rate ``gamma_loc = I(mu) sqrt(P/rho_p)/d``."""
    P = torch.clamp(torch.as_tensor(pressure, dtype=DTYPE), min=1e-30)
    return I_of_mu(mu, p) * torch.sqrt(P / rho_p) / d


def local_fluidity(
    mu: torch.Tensor | float,
    pressure: torch.Tensor | float,
    d: float,
    rho_p: float,
    p: MuIParams = MuIParams(),
) -> torch.Tensor:
    """Local fluidity source ``g_loc = gamma_loc(mu)/mu`` (0 below yield)."""
    mu_t = torch.as_tensor(mu, dtype=DTYPE)
    g = local_strain_rate(mu_t, pressure, d, rho_p, p) / torch.clamp(mu_t, min=1e-12)
    return torch.where(mu_t > p.mu_s, g, torch.zeros_like(g))


# --------------------------------------------------------------------------- #
# NGF linear operator:  A g = g - xi^2 L g
# --------------------------------------------------------------------------- #
@dataclass
class Boundary:
    """One boundary of a 1D/2D NGF grid."""

    kind: BCKind = "neumann"
    value: float = 0.0


class NGFOperator1D:
    """Matrix-free ``A = I - xi^2 d2/dy2`` on a uniform 1D grid.

    ``lo`` is the boundary at ``y=0`` (wall) and ``hi`` the one at ``y=H``.
    Ghost-cell formulation: Dirichlet ghost ``2 g_bc - g_0``, Neumann ghost
    ``g_0`` (zero flux).  Interior faces use the plain 3-point Laplacian, so
    the band width is not smeared by half-cell boundary stencils.
    """

    def __init__(self, n: int, h: float, xi: float, lo: Boundary, hi: Boundary):
        if n < 2:
            raise NGFError("need at least 2 cells")
        if h <= 0.0:
            raise NGFError("h must be > 0")
        self.n, self.h, self.xi = int(n), float(h), float(xi)
        self.lo, self.hi = lo, hi

    def diagonal(self) -> torch.Tensor:
        """Diagonal of ``A`` (also the Jacobi preconditioner)."""
        inv_h2 = 1.0 / self.h**2
        d = torch.full((self.n,), 1.0 + self.xi**2 * 2.0 * inv_h2, dtype=DTYPE)
        d[0] = 1.0 + self.xi**2 * inv_h2 * (3.0 if self.lo.kind == "dirichlet" else 1.0)
        d[-1] = 1.0 + self.xi**2 * inv_h2 * (3.0 if self.hi.kind == "dirichlet" else 1.0)
        return d

    def _ghosts(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gl = 2.0 * self.lo.value - g[0] if self.lo.kind == "dirichlet" else g[0]
        gh = 2.0 * self.hi.value - g[-1] if self.hi.kind == "dirichlet" else g[-1]
        return gl, gh

    def bc_vector(self) -> torch.Tensor:
        """Constant term of the affine operator (inhomogeneous Dirichlet data).

        ``matvec(g) = A_hom g + bc_vector()``; solve ``A_hom g = b - bc_vector()``.
        """
        c = torch.zeros(self.n, dtype=DTYPE)
        if self.lo.kind == "dirichlet":
            c[0] = -self.xi**2 * 2.0 * self.lo.value / self.h**2
        if self.hi.kind == "dirichlet":
            c[-1] = -self.xi**2 * 2.0 * self.hi.value / self.h**2
        return c

    def matvec(self, g: torch.Tensor) -> torch.Tensor:
        gl, gh = self._ghosts(g)
        ext = torch.cat([gl.reshape(1), g, gh.reshape(1)])
        lap = (ext[2:] - 2.0 * ext[1:-1] + ext[:-2]) / self.h**2
        return g - self.xi**2 * lap


class NGFOperator2D:
    """Matrix-free ``A = I - xi^2 (d2/dx2 + d2/dy2)`` on a uniform 2D grid.

    ``y`` (rows) is the inhomogeneous band direction, ``x`` (columns) is
    periodic.
    """

    def __init__(self, ny: int, nx: int, hy: float, hx: float, xi: float,
                 lo: Boundary, hi: Boundary):
        if ny < 2 or nx < 2:
            raise NGFError("need at least 2x2 cells")
        self.ny, self.nx = int(ny), int(nx)
        self.hy, self.hx, self.xi = float(hy), float(hx), float(xi)
        self.lo, self.hi = lo, hi

    def diagonal(self) -> torch.Tensor:
        inv_hy2, inv_hx2 = 1.0 / self.hy**2, 1.0 / self.hx**2
        d = torch.full((self.ny, self.nx), 1.0 + self.xi**2 * (2.0 * inv_hy2 + 2.0 * inv_hx2),
                       dtype=DTYPE)
        d[0, :] = 1.0 + self.xi**2 * (2.0 * inv_hx2 + inv_hy2 * (3.0 if self.lo.kind == "dirichlet" else 1.0))
        d[-1, :] = 1.0 + self.xi**2 * (2.0 * inv_hx2 + inv_hy2 * (3.0 if self.hi.kind == "dirichlet" else 1.0))
        return d

    def bc_vector(self) -> torch.Tensor:
        c = torch.zeros((self.ny, self.nx), dtype=DTYPE)
        if self.lo.kind == "dirichlet":
            c[0, :] = -self.xi**2 * 2.0 * self.lo.value / self.hy**2
        if self.hi.kind == "dirichlet":
            c[-1, :] = -self.xi**2 * 2.0 * self.hi.value / self.hy**2
        return c

    def matvec(self, g: torch.Tensor) -> torch.Tensor:
        lap_x = (torch.roll(g, -1, dims=1) - 2.0 * g + torch.roll(g, 1, dims=1)) / self.hx**2
        gl = (2.0 * self.lo.value - g[0, :]) if self.lo.kind == "dirichlet" else g[0, :]
        gh = (2.0 * self.hi.value - g[-1, :]) if self.hi.kind == "dirichlet" else g[-1, :]
        ext = torch.cat([gl.reshape(1, -1), g, gh.reshape(1, -1)], dim=0)
        lap_y = (ext[2:, :] - 2.0 * ext[1:-1, :] + ext[:-2, :]) / self.hy**2
        return g - self.xi**2 * (lap_x + lap_y)


@dataclass
class CGResult:
    x: torch.Tensor
    iterations: int
    residual: float
    residual_history: list[float] = field(default_factory=list)
    converged: bool = True


def _linear_apply(op, x: torch.Tensor) -> torch.Tensor:
    """Apply the *linear* part of ``op`` (strips the inhomogeneous-Dirichlet term)."""
    bc = getattr(op, "bc_vector", None)
    ax = op.matvec(x)
    return ax - bc() if bc is not None else ax


def cg_jacobi(op, b: torch.Tensor, *, tol: float = 1e-12, maxiter: int = 20000,
              x0: torch.Tensor | None = None, atol: float = 0.0) -> CGResult:
    """Jacobi-preconditioned conjugate gradients for ``A x = b``.

    ``op`` needs ``matvec`` and ``diagonal``.  Stops when
    ``||r||/||b|| <= tol`` or ``||r|| <= atol``.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - _linear_apply(op, x)
    diag = op.diagonal()
    z = r / diag
    p = z.clone()
    rz = (r * z).sum()
    bnorm = torch.linalg.vector_norm(b).clamp(min=1e-300)
    hist: list[float] = []
    it = 0
    converged = False
    for it in range(1, maxiter + 1):
        Ap = _linear_apply(op, p)
        pAp = (p * Ap).sum()
        if float(pAp) <= 0.0:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rnorm = float(torch.linalg.vector_norm(r))
        hist.append(rnorm / float(bnorm))
        if rnorm / float(bnorm) <= tol or rnorm <= atol:
            converged = True
            break
        z = r / diag
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    return CGResult(x=x, iterations=it, residual=hist[-1] if hist else 0.0,
                    residual_history=hist, converged=converged)


# --------------------------------------------------------------------------- #
# 1D shear-band problem
# --------------------------------------------------------------------------- #
@dataclass
class BandCase:
    """Wall-driven shear band in a granular layer of thickness ``H``.

    ``y=0`` is the rough *moving* wall (Dirichlet fluidity ``g_wall``) and
    ``y=H`` the far boundary.  The bulk sits below yield (``mu_bulk < mu_s``,
    ``g_loc = 0``) which is the regime where the local model predicts *no*
    deformation at all while the nonlocal model predicts a creep band of
    thickness set by ``xi``.
    """

    d: float = 1.0e-3          # grain diameter [m]
    rho_p: float = 2500.0      # grain density [kg/m3]
    gap_over_d: float = 40.0   # cell gap H/d
    xi_over_d: float = 4.0     # cooperative length / d
    mu_bulk: float = 0.30      # bulk stress ratio (below mu_s -> no local flow)
    mu_wall: float = 0.55      # wall friction -> wall fluidity
    pressure: float = 1000.0   # uniform confining pressure [Pa]
    cells_per_d: float = 4.0   # grid resolution
    shell: Literal["dirichlet", "neumann"] = "dirichlet"
    params: MuIParams = field(default_factory=MuIParams)

    def __post_init__(self) -> None:
        if self.d <= 0 or self.gap_over_d <= 0 or self.cells_per_d <= 0:
            raise NGFError("d, gap_over_d, cells_per_d must be > 0")

    @property
    def H(self) -> float:
        return self.gap_over_d * self.d

    @property
    def xi(self) -> float:
        return self.xi_over_d * self.d

    def g_wall(self) -> float:
        gam = local_strain_rate(self.mu_wall, self.pressure, self.d, self.rho_p, self.params)
        return float(gam / self.mu_wall)

    def n_cells(self) -> int:
        return int(round(self.gap_over_d * self.cells_per_d))


def solve_band(case: BandCase, *, tol: float = 1e-12, maxiter: int = 50000) -> dict:
    """Solve the 1D NGF boundary-layer (creep) band problem.

    Returns the fluidity profile, the strain-rate profile and solver stats.
    """
    n = case.n_cells()
    h = case.H / n
    lo = Boundary("dirichlet", case.g_wall())
    hi = Boundary(case.shell, 0.0)
    op = NGFOperator1D(n, h, case.xi, lo, hi)
    g_loc = torch.full((n,), float(local_fluidity(case.mu_bulk, case.pressure, case.d,
                                                  case.rho_p, case.params)))
    res = cg_jacobi(op, g_loc - op.bc_vector(), tol=tol, maxiter=maxiter)
    y = (torch.arange(n, dtype=DTYPE) + 0.5) * h
    gamma_dot = res.x * case.mu_bulk
    return {
        "y": y,
        "g": res.x,
        "gamma_dot": gamma_dot,
        "g_loc": g_loc,
        "H": case.H,
        "xi": case.xi,
        "d": case.d,
        "h": h,
        "residual": res.residual,
        "iterations": res.iterations,
        "converged": res.converged,
    }


def band_width_90_10(y: torch.Tensor, profile: torch.Tensor) -> float:
    """Width of the 90%->10% decay of ``profile`` measured from the wall.

    Locates the first crossing of 0.9*peak and the first later crossing of
    0.1*peak; returns the distance between them.  Returns 0 if the profile
    never reaches 10% inside the domain (band clipped by the far boundary).
    """
    prof = profile / profile[0].clamp(min=1e-300) if abs(float(profile[0])) > 0 else profile
    n = prof.numel()
    i90 = i10 = None
    for i in range(n):
        if prof[i] <= 0.9 and i90 is None:
            i90 = i
        if prof[i] <= 0.1 and i10 is None:
            i10 = i
            break
    if i90 is None:
        return float(y[-1] - y[0])
    if i10 is None:
        # band clipped by the far boundary: the width is the remaining extent
        return float(y[-1] - y[i90])
    return float(y[i10] - y[i90])


def band_width_moment(y: torch.Tensor, profile: torch.Tensor) -> float:
    """Moment width ``int p dy / p(0)`` (robust, no threshold choice)."""
    dy = float(y[1] - y[0])
    return float(profile.sum() * dy / profile[0].clamp(min=1e-300))


def width_sweep(xi_over_d: Iterable[float] = (1.0, 2.0, 4.0, 6.0),
                gap_over_d: Iterable[float] = (20.0, 40.0, 80.0),
                **kwargs) -> list[dict]:
    """Band width table over (xi/d, H/d) -- the saturation study."""
    out: list[dict] = []
    for xd in xi_over_d:
        for gd in gap_over_d:
            case = BandCase(gap_over_d=float(gd), xi_over_d=float(xd), **kwargs)
            sol = solve_band(case)
            out.append({
                "xi_over_d": float(xd),
                "gap_over_d": float(gd),
                "w_over_d_9010": band_width_90_10(sol["y"], sol["g"]) / case.d,
                "w_over_d_moment": band_width_moment(sol["y"], sol["g"]) / case.d,
                "residual": sol["residual"],
            })
    return out


# --------------------------------------------------------------------------- #
# Coarse graining (used by the A3/A4 checks)
# --------------------------------------------------------------------------- #
def coarse_grain(y: torch.Tensor, field: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-average a 1D field with ``window`` cells per bin."""
    if window < 1:
        raise NGFError("window must be >= 1")
    n = (field.numel() // window) * window
    if n == 0:
        raise NGFError("window larger than the field")
    yb = y[:n].reshape(-1, window).mean(dim=1)
    fb = field[:n].reshape(-1, window).mean(dim=1)
    return yb, fb


def fit_xi_from_profile(y: torch.Tensor, g: torch.Tensor, d: float) -> dict:
    """Estimate ``xi`` from a fluidity profile by a log-linear fit of the tail.

    For the boundary-layer solution ``g ~ g_w exp(-y/xi)`` (when the layer is
    much thinner than the gap), so ``xi`` is the negative inverse slope of
    ``log g`` vs ``y`` over the range where ``g`` is between 1% and 60% of
    ``g(0)``.
    """
    g0 = float(g[0])
    if g0 <= 0.0:
        raise NGFError("profile must be positive at the wall")
    ratio = g / g0
    mask = (ratio <= 0.6) & (ratio >= 1e-2)
    if int(mask.sum()) < 3:
        raise NGFError("insufficient points in the 1%-60% band for the xi fit")
    yy = y[mask]
    lg = torch.log(ratio[mask])
    # least squares slope of lg vs yy
    A = torch.stack([yy, torch.ones_like(yy)], dim=1)
    sol = torch.linalg.lstsq(A, lg.unsqueeze(1)).solution.squeeze(1)
    slope = float(sol[0])
    if slope >= 0.0:
        raise NGFError("tail is not decaying; cannot fit xi")
    xi_est = -1.0 / slope
    pred = A @ sol
    ss_res = float(((lg - pred) ** 2).sum())
    ss_tot = float(((lg - lg.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    return {"xi": xi_est, "xi_over_d": xi_est / d, "r2": r2, "n_points": int(mask.sum())}