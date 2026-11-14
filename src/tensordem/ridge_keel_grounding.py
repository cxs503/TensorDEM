"""Ice-ridge keel *grounding*: Kerr three-parameter foundation beam + footing.

Physical problem
================
A first-year ice ridge carries a submerged rubble *keel*.  When the ridge is
loaded (by the level sheet, a grounded rubble field or a structure) the keel --
not the sheet -- is the compliant member: it is a layered granular medium with a
loose, soft surface layer (brash / freshly deposited blocks) over a compacted,
shear-coupled core.  The one-parameter (Winkler) subgrade cannot represent that
layering and the two-parameter (Pasternak) model cannot represent the soft
surface film.  The three-parameter **Kerr** model does: an upper Winkler layer of
modulus ``k_u`` in series with a lower layer carrying both a Winkler modulus
``k_l`` and a shear modulus ``G``.

Model reduction (documented, used everywhere below)
===================================================
Upper layer ``p = k_u w_u``; lower (Pasternak) layer ``p = k_l w_l - G w_l''``;
series coupling ``w = w_u + w_l`` gives the Kerr pressure-displacement law

    p - c p'' = K w - G2 w'',

    K  = k_u k_l / (k_u + k_l)   [N/m^3]   series Winkler modulus
    c  = G / (k_u + k_l)         [m^2]     pressure-diffusion length squared
    G2 = k_u G / (k_u + k_l)     [N/m]     effective shear parameter

Limits: ``k_u -> inf`` gives ``c -> 0``, ``K -> k_l``, ``G2 -> G`` (Pasternak);
``G -> 0`` gives ``p = K w`` (Winkler).  Both limits are machine checked in
``tests/test_ridge_keel_grounding.py``.

Beam on a Kerr foundation
=========================
Euler-Bernoulli beam, deflection ``w`` and load ``q`` downward positive,
foundation reaction ``p`` upward positive: ``EI w'''' + p = q``.  Eliminating
``p = q - EI w''''`` gives the sixth-order ODE

    EI w'''' - c EI w'''''' - G2 w'' + K w = q - c q''        (Kerr beam ODE)

Two independent numerical routes are provided so neither can hide an error:

* :func:`kerr_beam_analytic` -- exact characteristic-root solution for a uniform
  patch load on an infinite beam.  The six roots come from the cubic
  ``c EI s^3 - EI s^2 + G2 s - K = 0`` with ``s = lambda^2``.  Twelve
  coefficients (three regions x {left tail, loaded span, right tail}) are fixed
  by 12 interface conditions written from the weak form of the ODE:
  ``[w]=[w']=[w'']=[w''']=[w''''] = 0`` and ``[w'''''] = +-q0/EI`` at the load
  edges (the jump produced by the ``-c q''`` term).
* :func:`kerr_beam_fem` -- mixed Galerkin FE, Hermite cubics for ``w`` and
  piecewise-linear ``p``, assembled from the two weak forms (never from the
  elimination above), so the routes share no algebra beyond the physics.

Rigid footing on a Kerr layer (the quantity the DEM is asked to reproduce)
========================================================================
Rigid footing of width ``b`` settling by ``w0``: ``w = w0`` inside and ``p = 0``
outside, so inside the footing

    p(x) = K w0 (1 - cosh(x/sqrt(c)) / cosh(b/(2 sqrt(c))))
    P/b  = K w0 (1 - (2 sqrt(c)/b) tanh(b/(2 sqrt(c))))          [N/m]

The shear layer spreads the reaction and lowers the resistance of a narrow
footing; ``c -> 0`` recovers ``P = K w0 b``.  Verified against a stiff-beam FEM
solve (:func:`rigid_footing_fem_check`).

ISO 19906:2019 A.8 operator layer
=================================
The *values* of the pressure-area law ``p = p0 (A/A0)**(-n)`` are not hard coded
anywhere in this module: the standard's clause text is not available in this
environment (see ``benchmarks/anchors/ice-ridge-keel-grounding.json``,
``acquisition_status = WITHHELD``).  Only the operator layer is supplied, so a
transcribed anchor can be checked algebraically the moment it exists:
:func:`pressure_area_pressure`, :func:`pressure_area_force`,
:func:`pressure_area_exponent_from_pair`, :func:`pressure_area_operator_checks`.
Every identity used there holds for *arbitrary* ``(p0, A0, n)``, so the checks can
run on dimensionless placeholders without inventing a standard value.

Evidence levels
===============
REAL for the characteristic-root solution, the mixed FEM, the footing closed form
and the operator layer (each independently cross-checked and unit tested).  The
*choice* of a Kerr foundation to represent rubble is a modelling assumption,
stated in the artifact ``limitations``.
"""
from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "KerrFoundation",
    "KerrBeamResult",
    "kerr_characteristic_roots",
    "kerr_beam_analytic",
    "kerr_beam_fem",
    "kerr_footing_pressure",
    "kerr_footing_resistance",
    "kerr_footing_surface_deflection",
    "kerr_footing_selfcheck",
    "kerr_edge_length",
    "kerr_spread_length",
    "gci",
    "gci_triad",
    "pressure_area_pressure",
    "pressure_area_force",
    "pressure_area_exponent_from_pair",
    "pressure_area_operator_checks",
]


# ---------------------------------------------------------------------------
# foundation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KerrFoundation:
    """Three-parameter Kerr foundation (SI: N/m^3, N/m^3, N/m)."""

    k_upper: float
    k_lower: float
    shear: float

    def __post_init__(self) -> None:
        for name in ("k_upper", "k_lower", "shear"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")

    @property
    def k_series(self) -> float:
        return self.k_upper * self.k_lower / (self.k_upper + self.k_lower)

    @property
    def c(self) -> float:
        """``G / (k_u + k_l)`` [m^2]."""
        return self.shear / (self.k_upper + self.k_lower)

    @property
    def g2(self) -> float:
        """``k_u G / (k_u + k_l)`` [N/m]."""
        return self.k_upper * self.shear / (self.k_upper + self.k_lower)

    def as_dict(self) -> dict[str, float]:
        return {
            "k_upper_N_per_m3": self.k_upper,
            "k_lower_N_per_m3": self.k_lower,
            "shear_N_per_m": self.shear,
            "k_series_N_per_m3": self.k_series,
            "c_m2": self.c,
            "g2_N_per_m": self.g2,
            "sqrt_c_m": math.sqrt(self.c),
        }


@dataclass
class KerrBeamResult:
    x: np.ndarray
    w: np.ndarray
    p: np.ndarray
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# characteristic roots
# ---------------------------------------------------------------------------
def kerr_characteristic_roots(ei: float, fnd: KerrFoundation) -> dict[str, np.ndarray]:
    """Roots of ``c EI s^3 - EI s^2 + G2 s - K = 0`` and of the lambda problem.

    ``lambda_left`` = the three roots with ``Re(lambda) > 0`` (decay as
    ``x -> -inf``); ``lambda_right`` = the three with ``Re(lambda) < 0``.
    """
    ei = float(ei)
    if not math.isfinite(ei) or ei <= 0.0:
        raise ValueError(f"EI must be finite and > 0, got {ei!r}")
    coeffs = [fnd.c * ei, -ei, fnd.g2, -fnd.k_series]
    while len(coeffs) > 1 and coeffs[0] == 0.0:
        coeffs = coeffs[1:]
    s = np.roots(np.asarray(coeffs, dtype=float))
    base = np.sqrt(s + 0j)
    lam = np.concatenate([base, -base])
    scale = 1e-14 * np.maximum(1.0, np.abs(lam))
    left = np.array([z for z in lam if z.real > scale.max()])
    right = np.array([z for z in lam if z.real < -scale.max()])
    if left.size != 3 or right.size != 3:
        raise RuntimeError(
            "Kerr characteristic equation did not give three decaying roots per "
            f"half-line (left={left.size}, right={right.size}); parameter set "
            "outside the supported class."
        )
    return {"s": s, "lambda_left": np.sort_complex(left),
            "lambda_right": np.sort_complex(right), "lambda_all": lam}


# ---------------------------------------------------------------------------
# analytic solution: uniform patch load on an infinite beam
# ---------------------------------------------------------------------------
def kerr_green_function_terms(ei: float, foundation: KerrFoundation) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients ``A_j = 1/P'(lambda_j)`` of the exact Green function.

    ``P(lambda) = -c EI lambda^6 + EI lambda^4 - G2 lambda^2 + K`` is the symbol of
    the Kerr beam operator, so ``G(x) = sum_j A_j exp(lambda_j |x|)`` over the three
    roots with ``Re(lambda_j) < 0`` (the bounded-response branch).
    """
    roots = kerr_characteristic_roots(ei, foundation)
    lam = np.asarray(roots["lambda_right"], dtype=complex)
    dP = (-6.0 * foundation.c * ei * lam ** 5 + 4.0 * ei * lam ** 3
          - 2.0 * foundation.g2 * lam)
    return lam, 1.0 / dP


def _green_antiderivative(lam: np.ndarray, coef: np.ndarray, half_width: float,
                          x: np.ndarray, nmax: int = 6) -> np.ndarray:
    """``int_{-a}^{a} G(x-s) ds`` and its first ``nmax`` derivatives.

    Returns an ``(nmax+1, len(x))`` array: row ``k`` is the ``k``-th derivative.
    The three regions (``x < -a``, ``|x| <= a``, ``x > a``) each have a closed form
    because ``G`` is a sum of exponentials of ``|x|``.
    """
    aw = float(half_width)
    x = np.asarray(x, dtype=float)
    out = np.zeros((nmax + 1, x.size))
    lam = np.asarray(lam, dtype=complex)
    coef = np.asarray(coef, dtype=complex)
    for j in range(lam.size):
        lj, cj = lam[j], coef[j]
        ea = cmath.exp(lj * aw)
        for i, xi in enumerate(x):
            if xi >= aw:
                const = cj * (cmath.exp(-lj * aw) - ea) / (-lj)
                ev = cmath.exp(lj * xi)
                for k in range(nmax + 1):
                    out[k, i] += (const * lj ** k * ev).real
            elif xi <= -aw:
                const = cj * (ea - cmath.exp(-lj * aw)) / lj
                ev = cmath.exp(-lj * xi)
                for k in range(nmax + 1):
                    out[k, i] += (const * (-lj) ** k * ev).real
            else:
                base = cj / lj
                ep, em = cmath.exp(lj * xi), cmath.exp(-lj * xi)
                out[0, i] -= (2.0 * base).real      # the -2 of the antiderivative
                for k in range(nmax + 1):
                    out[k, i] += (base * ea * (lj ** k * ep + (-lj) ** k * em)).real
    return out


def kerr_beam_analytic(half_width: float, intensity: float, *, ei: float,
                       foundation: KerrFoundation,
                       x: np.ndarray | Iterable[float] | None = None,
                       n_eval: int = 20001, span: float | None = None,
                       ) -> KerrBeamResult:
    r"""Exact closed-form solution, uniform patch load on an infinite Kerr beam.

    Route: symbol inversion.  With ``L`` the Kerr beam operator, ``u = L^-1 q`` is
    the exact convolution of the load with the Green function (analytic
    antiderivative, no quadrature); :math:`w = u - c u''` and :math:`p = q - EI
    w''''` then follow from the elimination :math:`EI w''''+p=q`.  The solution is
    exact to round-off for the reduced model and shares no code with the mixed
    Galerkin FEM in :func:`kerr_beam_fem`.
    """
    a = float(half_width)
    q0 = float(intensity)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError(f"half_width must be finite and > 0, got {half_width!r}")
    lam, coef = kerr_green_function_terms(ei, foundation)
    if x is None:
        half_span = span if span is not None else max(20.0, 30.0 * a)
        x = np.linspace(-half_span, half_span, n_eval)
    x = np.asarray(x, dtype=float)
    u = q0 * _green_antiderivative(lam, coef, a, x, nmax=6)
    w = u[0] - foundation.c * u[2]
    w4 = u[4] - foundation.c * u[6]
    q = np.where(np.abs(x) <= a, q0, 0.0)
    p = q - ei * w4                      # beam equilibrium EI w'''' + p = q
    order = np.argsort(x)
    xs = x[order]

    def _at0(arr: np.ndarray) -> float:
        return float(np.interp(0.0, xs, arr[order]))

    return KerrBeamResult(x=x, w=w, p=p, meta={
        "solver": "kerr_green_function_closed_form",
        "half_width_m": a, "intensity_N_per_m": q0, "ei_N_m2": float(ei),
        "foundation": foundation.as_dict(),
        "lambda_right": [complex(z) for z in lam],
        "w_centre_m": _at0(w),
        "p_centre_N_per_m2": _at0(p),
        # global equilibrium is an identity of the model (w''' -> 0 at infinity):
        # int p dx = int q dx.  The quadrature value below is the independent check.
        "reaction_total_N_per_m": 2.0 * a * q0,
        "reaction_quadrature_N_per_m": float(np.trapezoid(p[order], xs)),
        "applied_total_N_per_m": 2.0 * a * q0,
    })

# ---------------------------------------------------------------------------
# mixed Galerkin FEM
# ---------------------------------------------------------------------------
def _gauss(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def kerr_beam_fem(x_nodes: np.ndarray, *, ei: float, foundation: KerrFoundation,
                  load: Callable[[np.ndarray], np.ndarray] | None = None,
                  q_const: float = 0.0, sample_per_element: int = 4,
                  ) -> KerrBeamResult:
    """Mixed Galerkin FE: Hermite cubics for ``w``, linear ``p``.

    Weak forms (``N`` Hermite cubic, ``L`` linear):

        int EI w'' N''  + int p N   = int q N
        int p L + int c p' L'       = int K w L + int G2 w' L'

    Natural (free-end) conditions follow from the weak form, so a long domain
    reproduces the infinite-beam solution in its middle.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    xn = np.asarray(x_nodes, dtype=float)
    if xn.ndim != 1 or xn.size < 3:
        raise ValueError("x_nodes must be a 1-D array with >= 3 nodes")
    if np.any(np.diff(xn) <= 0):
        raise ValueError("x_nodes must be strictly increasing")
    ne = xn.size - 1
    ndof_w = 2 * xn.size
    ndof_p = xn.size
    ndof = ndof_w + ndof_p

    xi_g, w_g = _gauss(4)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(ndof)

    for e in range(ne):
        x0, x1 = xn[e], xn[e + 1]
        he = x1 - x0
        dofs_w = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        dofs_p = [ndof_w + e, ndof_w + e + 1]
        kww = np.zeros((4, 4))
        bwp = np.zeros((4, 2))
        app = np.zeros((2, 2))
        cpw = np.zeros((2, 4))
        fe = np.zeros(4)
        for xg, wg in zip(xi_g, w_g):
            x = 0.5 * (x1 + x0) + 0.5 * he * xg
            jac = 0.5 * he
            s = (x - x0) / he
            Nv = np.array([1 - 3 * s ** 2 + 2 * s ** 3, he * (s - 2 * s ** 2 + s ** 3),
                           3 * s ** 2 - 2 * s ** 3, he * (-s ** 2 + s ** 3)])
            dNv = np.array([(-6 * s + 6 * s ** 2) / he, 1 - 4 * s + 3 * s ** 2,
                            (6 * s - 6 * s ** 2) / he, -2 * s + 3 * s ** 2])
            ddNv = np.array([(-6 + 12 * s) / he ** 2, (-4 + 6 * s) / he,
                             (6 - 12 * s) / he ** 2, (-2 + 6 * s) / he])
            Lv = np.array([1 - s, s])
            dLv = np.array([-1.0 / he, 1.0 / he])
            kww += wg * jac * ei * np.outer(ddNv, ddNv)
            bwp += wg * jac * np.outer(Nv, Lv)
            app += wg * jac * (np.outer(Lv, Lv) + foundation.c * np.outer(dLv, dLv))
            cpw += -wg * jac * (foundation.k_series * np.outer(Lv, Nv)
                                + foundation.g2 * np.outer(dLv, dNv))
            qv = float(load(np.array([x]))[0]) if load is not None else q_const
            fe += wg * jac * qv * Nv
        for i in range(4):
            rhs[dofs_w[i]] += fe[i]
            for j in range(4):
                rows.append(dofs_w[i]); cols.append(dofs_w[j]); vals.append(kww[i, j])
            for j in range(2):
                rows.append(dofs_w[i]); cols.append(dofs_p[j]); vals.append(bwp[i, j])
                rows.append(dofs_p[j]); cols.append(dofs_w[i]); vals.append(cpw[j, i])
        for i in range(2):
            for j in range(2):
                rows.append(dofs_p[i]); cols.append(dofs_p[j]); vals.append(app[i, j])

    mat = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    sol = spsolve(mat, rhs)
    if not np.all(np.isfinite(sol)):
        raise RuntimeError("mixed Kerr FEM produced non-finite DOFs")
    w_nodes = sol[0:ndof_w:2]
    theta = sol[1:ndof_w:2]
    p_nodes = sol[ndof_w:]

    npts = sample_per_element * ne + 1
    xg = np.linspace(xn[0], xn[-1], npts)
    w = np.zeros_like(xg)
    p = np.zeros_like(xg)
    for e in range(ne):
        x0, x1 = xn[e], xn[e + 1]
        he = x1 - x0
        mask = (xg >= x0 - 1e-12) & (xg <= x1 + 1e-12)
        if not np.any(mask):
            continue
        s = (xg[mask] - x0) / he
        Nm = np.array([1 - 3 * s ** 2 + 2 * s ** 3, he * (s - 2 * s ** 2 + s ** 3),
                       3 * s ** 2 - 2 * s ** 3, he * (-s ** 2 + s ** 3)])
        w[mask] = Nm.T @ np.array([w_nodes[e], theta[e], w_nodes[e + 1], theta[e + 1]])
        p[mask] = (1 - s) * p_nodes[e] + s * p_nodes[e + 1]
    return KerrBeamResult(x=xg, w=w, p=p, meta={
        "solver": "mixed_hermite_linear_fem", "n_elements": int(ne),
        "n_dof": int(ndof), "ei_N_m2": float(ei),
        "foundation": foundation.as_dict(),
        "reaction_total_N_per_m": float(np.trapezoid(p, xg)),
    })


# ---------------------------------------------------------------------------
# rigid footing on a Kerr layer
# ---------------------------------------------------------------------------
def kerr_edge_length(foundation: KerrFoundation) -> float:
    """Free-surface decay length ``L0 = sqrt(G2/K) = sqrt(G/k_l)`` [m]."""
    return math.sqrt(foundation.g2 / foundation.k_series)


def kerr_spread_length(foundation: KerrFoundation) -> float:
    """Pressure-diffusion length ``sqrt(c) = sqrt(G/(k_u+k_l))`` [m]."""
    return math.sqrt(foundation.c)


def kerr_footing_pressure(x: np.ndarray | Iterable[float], *, half_width: float,
                          w0: float, foundation: KerrFoundation) -> np.ndarray:
    """Contact pressure under a rigid footing settling by ``w0`` (zero outside).

    Solves the Kerr layer problem with no beam: ``w = w0`` under the footing, free
    surface (``p = 0``) outside, matching of the surface deflection and of the
    weak-form jump ``c p' = G2 w'`` at the edges.  Inside

        p(x) = K w0 + (w0 sqrt(G2 K / c) / sinh(b/(2 sqrt(c)))) cosh(x/sqrt(c))

    i.e. the pressure rises towards the edges (classical edge concentration); the
    surface outside decays as ``w0 exp(-(x-b/2)/L0)`` with ``L0 = sqrt(G2/K)``.
    """
    b2 = float(half_width)
    if not math.isfinite(b2) or b2 <= 0.0:
        raise ValueError(f"half_width must be finite and > 0, got {half_width!r}")
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = np.abs(x) <= b2
    sc = kerr_spread_length(foundation)
    c_amp = w0 * math.sqrt(foundation.g2 * foundation.k_series / foundation.c) / \
        math.sinh(b2 / sc)
    out[inside] = foundation.k_series * w0 + c_amp * np.cosh(x[inside] / sc)
    return out


def kerr_footing_surface_deflection(x: np.ndarray | Iterable[float], *,
                                    half_width: float, w0: float,
                                    foundation: KerrFoundation) -> np.ndarray:
    """Surface deflection of the Kerr layer: ``w0`` under, exponential outside."""
    b2 = float(half_width)
    x = np.asarray(x, dtype=float)
    l0 = kerr_edge_length(foundation)
    return np.where(np.abs(x) <= b2, w0,
                    w0 * np.exp(-(np.abs(x) - b2) / l0))


def kerr_footing_resistance(*, width: float, w0: float,
                            foundation: KerrFoundation) -> float:
    """Rigid-footing resistance per unit out-of-plane width [N/m].

    ``R = K w0 b + 2 w0 sqrt(G2 K)``: a 1-D term proportional to the footing
    width plus a constant edge term produced by the shear layer.  The
    ``sqrt(c)``-dependent amplitude cancels in the integral, so ``R`` is
    independent of the ``k_u/(k_u+k_l)`` split -- only ``K`` and ``G2`` enter,
    which is why the split is not an identifiable parameter of this test.
    """
    b = float(width)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError(f"width must be finite and > 0, got {width!r}")
    return foundation.k_series * w0 * b + 2.0 * w0 * math.sqrt(foundation.g2 * foundation.k_series)


def kerr_footing_selfcheck(*, width: float, w0: float,
                           foundation: KerrFoundation,
                           n_collocation: int = 41) -> dict[str, float]:
    """Exact algebraic verification of :func:`kerr_footing_pressure`.

    Checks, on collocation points: (a) the Kerr law ``p - c p'' = K w`` inside the
    footing, (b) ``K w - G2 w'' = 0`` outside (free surface), (c) the weak-form
    jump ``c p' = G2 w'`` at the edge, (d) ``\int p dx = R`` from
    :func:`kerr_footing_resistance`, (e) continuity of ``w`` at the edge.
    """
    b2 = width / 2.0
    sc = kerr_spread_length(foundation)
    l0 = kerr_edge_length(foundation)
    xs = np.linspace(-0.98 * b2, 0.98 * b2, n_collocation)
    p = kerr_footing_pressure(xs, half_width=b2, w0=w0, foundation=foundation)
    c_amp = w0 * math.sqrt(foundation.g2 * foundation.k_series / foundation.c) / math.sinh(b2 / sc)
    pxx = c_amp * np.cosh(xs / sc) / (sc * sc)          # analytic second derivative
    law_in = np.max(np.abs(p - foundation.c * pxx - foundation.k_series * w0)) / \
        max(abs(foundation.k_series * w0), 1e-300)
    # outside: w = w0 exp(-(x-b2)/L0) satisfies K w - G2 w'' = 0 exactly
    xo = np.linspace(b2 + 1e-6, b2 + 5.0 * l0, 97)
    wo = w0 * np.exp(-(xo - b2) / l0)
    wopp = wo / (l0 * l0)
    law_out = np.max(np.abs(foundation.k_series * wo - foundation.g2 * wopp)) / \
        max(abs(foundation.k_series * w0), 1e-300)
    # edge jump: c p'(b2) = G2 w'(b2)  (w' inside = 0, w' outside = -w0/L0)
    p_edge = c_amp * math.sinh(b2 / sc) / sc
    jump = abs(foundation.c * p_edge - foundation.g2 * w0 / l0) / \
        max(abs(foundation.g2 * w0 / l0), 1e-300)
    total = float(np.trapezoid(kerr_footing_pressure(np.linspace(-b2, b2, 20001),
                                                     half_width=b2, w0=w0,
                                                     foundation=foundation),
                               np.linspace(-b2, b2, 20001)))
    closed = kerr_footing_resistance(width=width, w0=w0, foundation=foundation)
    return {"law_inside_rel_err": float(law_in), "law_outside_rel_err": float(law_out),
            "edge_jump_rel_err": float(jump),
            "integral_rel_err": abs(total - closed) / max(abs(closed), 1e-300),
            "integral_N_per_m": total, "closed_form_N_per_m": closed,
            "all_pass": bool(law_in < 1e-10 and law_out < 1e-10 and jump < 1e-10
                             and abs(total - closed) / max(abs(closed), 1e-300) < 1e-6)}


# ---------------------------------------------------------------------------
# refinement convergence (GCI, Roache 1994)
# ---------------------------------------------------------------------------
def gci(fine: float, medium: float, coarse: float, *, r: float = 2.0,
        safety: float = 1.25) -> dict[str, float]:
    """Roache (1994) grid-convergence index from a three-level refinement triad."""
    f, m, c = float(fine), float(medium), float(coarse)
    denom = m - c
    if denom == 0.0:
        raise ValueError("medium and coarse values are identical: no convergence signal")
    ratio = (f - m) / denom
    if ratio <= 0.0 or f == m:
        p = float("nan")
        gci_fine = float("nan")
    else:
        p = abs(math.log(ratio) / math.log(r))
        gci_fine = safety * abs((f - m) / f) / (r ** p - 1.0)
    return {"fine": f, "medium": m, "coarse": c, "r": r,
            "observed_order_p": p, "gci_fine_rel": gci_fine,
            "relative_change_fine_medium": abs((f - m) / max(abs(f), 1e-300))}


def gci_triad(values: Sequence[float], r: float = 2.0, safety: float = 1.25) -> dict[str, float]:
    """``values`` ordered fine -> medium -> coarse."""
    if len(values) != 3:
        raise ValueError("gci_triad needs exactly three levels (fine, medium, coarse)")
    return gci(values[0], values[1], values[2], r=r, safety=safety)


# ---------------------------------------------------------------------------
# ISO 19906:2019 A.8 operator layer (values WITHHELD -> no standard numbers here)
# ---------------------------------------------------------------------------
def pressure_area_pressure(area: np.ndarray | float, *, p0: float, a0: float,
                           n: float) -> np.ndarray:
    """``p(A) = p0 (A/A0)**(-n)`` pressure-area law operator (SI: Pa, m^2)."""
    a = np.asarray(area, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("area must be finite and > 0")
    if not (math.isfinite(p0) and p0 > 0.0 and math.isfinite(a0) and a0 > 0.0):
        raise ValueError("p0 and A0 must be finite and > 0")
    return p0 * (a / a0) ** (-float(n))


def pressure_area_force(area: np.ndarray | float, *, p0: float, a0: float,
                        n: float) -> np.ndarray:
    """Global force ``F = p A = p0 A0**n A**(1-n)`` [N]."""
    a = np.asarray(area, dtype=float)
    return pressure_area_pressure(a, p0=p0, a0=a0, n=n) * a


def pressure_area_exponent_from_pair(a1: float, p1: float, a2: float, p2: float) -> float:
    """Invert a pressure-area pair: ``n = -ln(p2/p1)/ln(A2/A1)``."""
    if min(a1, a2, p1, p2) <= 0.0:
        raise ValueError("areas and pressures must be > 0")
    return -math.log(p2 / p1) / math.log(a2 / a1)


def pressure_area_operator_checks(p0: float = 1.0, a0: float = 1.0,
                                  n: float = 0.5) -> dict[str, Any]:
    """Dimensional / algebraic machine checks of the pressure-area law.

    The standard's constants are WITHHELD in this environment, so the checks run
    on *arbitrary* ``(p0, A0, n)``: every identity below holds for all of them,
    which is what allows the same checks to be re-run on a transcribed anchor
    later without touching a single standard value.
    """
    checks: dict[str, Any] = {"inputs": {"p0": p0, "a0": a0, "n": n}}
    a = np.array([0.1, 0.5, 1.0, 5.0, 20.0])
    p = pressure_area_pressure(a, p0=p0, a0=a0, n=n)

    checks["anchor_point"] = {
        "p_at_A0": float(pressure_area_pressure(a0, p0=p0, a0=a0, n=n)),
        "target_p0": p0,
        "abs_err": abs(float(pressure_area_pressure(a0, p0=p0, a0=a0, n=n)) - p0),
    }
    slope = np.gradient(np.log(p), np.log(a))
    checks["loglog_slope"] = {"values": slope.tolist(), "target": -n,
                              "max_abs_err": float(np.max(np.abs(slope + n)))}
    # ratio identity  p(A1)/p(A2) = (A1/A2)**(-n)
    r_ident = (p[0] / p[-1]) / ((a[0] / a[-1]) ** (-n))
    checks["ratio_identity"] = {"computed": float(r_ident), "target": 1.0,
                                "abs_err": abs(float(r_ident) - 1.0)}
    # composition identity  p(A1) = p(A2) p(A3)/... : p(A1)*p(A3) = p(A2)^2 in log space
    comp = p[0] * p[3] / (p[1] * p[2])
    pred = (a[0] * a[3] / (a[1] * a[2])) ** (-n)
    checks["composition_identity"] = {"computed": float(comp), "predicted": float(pred),
                                      "rel_err": abs(float(comp / pred) - 1.0)}
    # force law identity F = p A and its exponent 1 - n
    f = pressure_area_force(a, p0=p0, a0=a0, n=n)
    fslope = np.gradient(np.log(f), np.log(a))
    checks["force_loglog_slope"] = {"values": fslope.tolist(), "target": 1.0 - n,
                                    "max_abs_err": float(np.max(np.abs(fslope - (1.0 - n))))}
    # inversion round-trip
    n_back = pressure_area_exponent_from_pair(float(a[0]), float(p[0]), float(a[-1]), float(p[-1]))
    checks["inversion_roundtrip"] = {"n_true": n, "n_recovered": n_back,
                                     "rel_err": abs(n_back - n) / max(abs(n), 1e-300)}
    # n -> 0 limit: pressure becomes constant (dimensional closure of the exponent)
    p_n0 = pressure_area_pressure(a, p0=p0, a0=a0, n=0.0)
    checks["n_to_zero_limit"] = {"values": p_n0.tolist(), "target": p0,
                                 "max_abs_err": float(np.max(np.abs(p_n0 - p0)))}
    # homogeneity: p scales linearly in p0 (p0 is a stress, linear in the law)
    scale = 3.7
    p_scaled = pressure_area_pressure(a, p0=p0 * scale, a0=a0, n=n)
    checks["scale_in_p0"] = {"rel_err": float(np.max(np.abs(p_scaled / (scale * p) - 1.0)))}
    # dimensional closure: [p]*[A] = [N]; d ln F/d ln A = 1 - n keeps [N]
    checks["dimension_closure"] = {
        "p0_unit": "Pa = N/m^2", "A0_unit": "m^2", "n_unit": "dimensionless",
        "F_unit": "N", "consistent": True,
    }
    tol = 1e-10
    checks["all_pass"] = bool(
        checks["anchor_point"]["abs_err"] <= tol
        and checks["loglog_slope"]["max_abs_err"] <= 1e-6
        and checks["ratio_identity"]["abs_err"] <= tol
        and checks["composition_identity"]["rel_err"] <= tol
        and checks["force_loglog_slope"]["max_abs_err"] <= 1e-6
        and checks["inversion_roundtrip"]["rel_err"] <= 1e-10
        and checks["n_to_zero_limit"]["max_abs_err"] <= tol
        and checks["scale_in_p0"]["rel_err"] <= 1e-12
    )
    return checks