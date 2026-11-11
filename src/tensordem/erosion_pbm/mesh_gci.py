"""Discretisation-error metrology for the sectional breakage PBM.

Why this module exists
----------------------
The finite-volume sectional solver of :mod:`tensordem.erosion_pbm.pbm` is
*lossless* in two of its moment measures -- the two-pivot weights conserve
``sum_k v_k n_k`` and every fragment contributes unit number -- so a
"mass is conserved" test says nothing about whether the *mesh* is converged.
This module adds the missing metrology:

* a **log-log variable grid** family (:class:`LogLogGrid`) with an overall
  width grading ratio ``R`` -- bin edges are uniform in ``ln v`` when
  ``R == 1`` and are exponentially stretched (local width growing by a fixed
  factor across the domain) otherwise.  Refinement multiplies the bin count by
  ``f`` and divides *every* local width by ``f`` while keeping ``R`` fixed, so
  a three-level sequence has refinement ratio ``r = f`` for graded and uniform
  grids alike (this is the property that makes Richardson extrapolation valid
  on a graded mesh -- it is why the grading is parametrised by the overall
  ratio and not by a per-bin growth factor);
* **analytic bin integrals** (:meth:`LogLogGrid.moment`): with the within-bin
  number density constant in ``v``, ``int v**p n(v) dv`` over a bin is
  ``n_k (V_{k+1}^{p+1} - V_k^{p+1}) / ((p+1) (V_{k+1} - V_k))`` in closed form,
  so the reported moment integrates the reconstruction instead of collapsing
  every particle onto a pivot;
* an **exact-in-time propagator** (:func:`propagate`): the breakage operator is
  linear with constant coefficients, so the state is advanced by a
  uniformisation series ``exp(A t) n0 = e^{-lam t} sum_k (lam t)^k / k!  P^k``
  with ``P = I + A/lam >= 0``.  The time axis is thereby removed as a
  confounder -- the error a mesh study measures is the *sectional* error, not
  the ODE tolerance (``scipy.linalg.expm`` is NaN-prone on this strongly
  non-normal operator; ``solve_ivp`` with the ``atol`` needed for the tiny
  bins takes minutes);
* **Richardson / Roache GCI** on a three-level sequence
  (:func:`richardson_gci`), with honest flags for a non-monotone or degenerate
  sequence;
* the **fines floor** turned from an implicit assumption into an explicit
  criterion (:func:`floor_share`, :func:`floor_threshold`);
* the **grid contribution to an erosion rate** (:func:`erosion_removed_mass`),
  so it can be compared on the same axis with the cross-correlation
  (Finnie vs Oka vs DNV-RP-O501) deviation.

Anchor: the analytic-integral vs pivot measure factor
------------------------------------------------------
A daughter of volume ``v_d`` split between the section representatives ``v_k``
and ``v_{k+1}`` with weights ``w, 1 - w`` satisfies ``w v_k + (1 - w) v_{k+1} =
v_d`` exactly -- that identity *is* the mass conservation of the scheme, and it
is why ``sum_k v_k n_k`` is an invariant rather than a converged quantity.  The
analytic bin integral instead weights the same number by the constant-density
bin mean ``vbar_k = (V_k + V_{k+1}) / 2``, and because both are proportional to
``V_k`` (``v_k = V_k sqrt(r)``, ``vbar_k = V_k (1 + r) / 2``) the two measures
differ by the *distribution-independent* factor

    ``int v n dv / sum_k v_k n_k = (1 + r) / (2 sqrt(r)) = 1 + (r - 1)**2 / (8 r)``

with ``r`` the local bin ratio -- second order in the mesh size, closed form,
and independent of the PSD.  :func:`pivot_measure_bias` is *not* that factor
(it is ``(r - 1) / 2``, the naive linear estimate); the exact factor is checked
in the study to machine precision, which calibrates the GCI machinery against
theory instead of against itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .pbm import SPHERE_FACTOR, daughter_matrix

__all__ = [
    "LogLogGrid",
    "nested_family",
    "build_operator",
    "propagate",
    "pivot_measure_bias",
    "richardson_gci",
    "cascade_solution",
    "cascade_error",
    "mass_ledger",
    "floor_share",
    "floor_threshold",
    "erosion_removed_mass",
    "psd_fingerprint",
]

#: 4-point Gauss-Legendre nodes/weights on [-1, 1] (for the erosion integral).
_GAUSS_X = (-0.8611363115940526, -0.3399810435848563,
            0.3399810435848563, 0.8611363115940526)
_GAUSS_W = (0.3478548451374538, 0.6521451548625461,
            0.6521451548625461, 0.3478548451374538)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogLogGrid:
    """Log-spaced volume grid on ``[v_min, v_max]`` with an overall grading ``R``.

    Edge ``k`` sits at

        ``ln V_k = ln v_min + (ln v_max - ln v_min) * f(k / N)``,
        ``f(xi) = (exp(beta xi) - 1) / (exp(beta) - 1)``, ``beta = ln R``.

    ``R == 1`` is the uniform-in-``ln v`` ladder.  ``R > 1`` stretches the mesh
    exponentially: the last local width exceeds the first by exactly ``R``, so
    resolution is concentrated at the fines end -- the "log-log variable mesh"
    of campaign node ``p488w1-n09``.  Uniform refinement multiplies ``N`` by
    ``f`` **and keeps ``R``**, hence every local width scales by ``1/f`` and
    the family is a valid Richardson sequence.
    """

    v_min: float
    v_max: float
    n_bins: int
    grading: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.v_min < self.v_max):
            raise ValueError(f"need 0 < v_min < v_max, got {self.v_min}, {self.v_max}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.grading < 1.0:
            raise ValueError(f"grading must be >= 1, got {self.grading}")

    # -- construction ------------------------------------------------------
    @property
    def log_span(self) -> float:
        return math.log(self.v_max / self.v_min)

    @property
    def beta(self) -> float:
        return math.log(self.grading)

    def _shape(self, xi):
        xi = np.asarray(xi, dtype=float)
        b = self.beta
        if abs(b) < 1e-13:
            return xi
        return (np.exp(b * xi) - 1.0) / (math.exp(b) - 1.0)

    @property
    def edges(self) -> np.ndarray:
        k = np.arange(self.n_bins + 1) / self.n_bins
        return self.v_min * np.exp(self.log_span * self._shape(k))

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def volumes(self) -> np.ndarray:
        """Section representative: geometric centre ``sqrt(V_k V_{k+1})``."""
        e = self.edges
        return np.sqrt(e[:-1] * e[1:])

    @property
    def diameters(self) -> np.ndarray:
        return self.volumes ** (1.0 / 3.0)

    @property
    def mean_volumes(self) -> np.ndarray:
        """Analytic bin mean of ``v`` under a constant-in-``v`` density."""
        e = self.edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def local_ratio(self) -> np.ndarray:
        e = self.edges
        return e[1:] / e[:-1]

    @property
    def sections_per_octave(self) -> np.ndarray:
        """Local resolution ``ln 2 / ln(V_{k+1}/V_k)`` per bin."""
        return math.log(2.0) / np.log(self.local_ratio)

    @property
    def width_grading(self) -> float:
        """Realised grading ``max(log width) / min(log width)``.

        It equals ``R**(1 - 1/N)`` -- i.e. ``R`` in the continuum limit and
        ``R`` to within ``1/N`` on a finite mesh -- and it is *invariant under
        refinement* up to that ``O(1/N)`` correction, which is exactly what the
        Richardson argument needs.

        The *log* width is the mesh size of a log-spaced ladder (the arithmetic
        width ``V_{k+1} - V_k`` is dominated by the volume range itself, not by
        the grading), so this is the ratio the refinement argument uses.
        """
        lw = np.diff(np.log(self.edges))
        return float(lw.max() / lw.min())

    def refine(self, factor: int = 2) -> "LogLogGrid":
        """Uniform refinement: ``N -> f N``, all local widths ``/ f``, ``R`` fixed."""
        if factor < 1:
            raise ValueError("refinement factor must be an integer >= 1")
        return LogLogGrid(self.v_min, self.v_max, self.n_bins * int(factor), self.grading)

    def index_of_volume(self, v) -> np.ndarray:
        v = np.atleast_1d(np.asarray(v, dtype=float))
        idx = np.searchsorted(self.edges, v, side="right") - 1
        return np.clip(idx, 0, self.n_bins - 1)

    # -- moment measures ---------------------------------------------------
    def _check(self, n) -> np.ndarray:
        n = np.asarray(n, dtype=float)
        if n.shape != (self.n_bins,):
            raise ValueError(f"n must have {self.n_bins} entries, got shape {n.shape}")
        return n

    def moment(self, n, order: int = 0) -> float:
        """Analytic bin integral ``int v**order n(v) dv`` (constant-in-``v`` model)."""
        n = self._check(n)
        e = self.edges
        p = float(order)
        if abs(p + 1.0) < 1e-12:
            num = np.log(e[1:] / e[:-1])
        else:
            num = (e[1:] ** (p + 1.0) - e[:-1] ** (p + 1.0)) / (p + 1.0)
        return float(np.sum(n * num / self.widths))

    def pivot_moment(self, n, order: int = 0) -> float:
        """Pivot sum ``sum_k v_k**order n_k`` -- the measure the two-pivot rule keeps."""
        return float(np.sum(self.volumes ** float(order) * self._check(n)))

    def quantile_diameter(self, n, prob: float) -> float:
        """Number-weighted quantile diameter, cumulative linear in ``v`` per bin."""
        if not (0.0 < prob < 1.0):
            raise ValueError("prob must lie in (0, 1)")
        n = self._check(n)
        w = n / n.sum()
        c = np.concatenate(([0.0], np.cumsum(w)))
        k = int(np.clip(np.searchsorted(c, prob, side="right") - 1, 0, self.n_bins - 1))
        frac = 0.0 if w[k] == 0.0 else (prob - c[k]) / w[k]
        e = self.edges
        v = e[k] + float(np.clip(frac, 0.0, 1.0)) * (e[k + 1] - e[k])
        return float(v ** (1.0 / 3.0))

    def functional(self, n, d_ref: float) -> dict:
        """The reported functionals of one PSD (all moment measures, no weighting)."""
        n = self._check(n)
        return {
            "M0": self.moment(n, 0),
            "M1_pivot": self.pivot_moment(n, 1),
            "M1_integral": self.moment(n, 1),
            "M2_integral": self.moment(n, 2),
            "fines_number": float(np.sum(n[self.diameters < 0.5 * d_ref])),
            "d50": self.quantile_diameter(n, 0.5),
            "floor_mass_share": float(self.volumes[0] * n[0] / self.pivot_moment(n, 1)),
        }


def nested_family(sections_per_octave, octaves_below: int, octaves_above: int,
                  d_0: float) -> LogLogGrid:
    """Uniform log2 ladder with ``q`` sections per octave of volume.

    ``v_min = d_0**3 2**-octaves_below``, ``v_max = d_0**3 2**octaves_above``.
    On this family a halving daughter lands **exactly** on a section
    representative (``sqrt(V_k V_{k+1}) / 2 = sqrt(V_{k-q} V_{k-q+1})``), which
    is what makes the analytic binary cascade a genuine reference (criterion C2).
    """
    v_0 = float(d_0) ** 3
    return LogLogGrid(v_min=v_0 * 2.0 ** (-octaves_below),
                      v_max=v_0 * 2.0 ** octaves_above,
                      n_bins=(octaves_below + octaves_above) * int(sections_per_octave),
                      grading=1.0)


# ---------------------------------------------------------------------------
# exact-in-time propagation
# ---------------------------------------------------------------------------
def build_operator(selection, matrix) -> np.ndarray:
    """Breakage operator ``A = -diag(S) + M diag(S)`` for ``dn/dt = A n``."""
    S = np.asarray(selection, dtype=float)
    M = np.asarray(matrix, dtype=float)
    if S.ndim != 1 or M.shape != (S.size, S.size):
        raise ValueError("selection must be 1-D and matrix square of matching size")
    return -np.diag(S) + M @ np.diag(S)


def propagate(selection, matrix, n0, t_eval, *, tol: float = 1e-16,
              max_terms: int = 20000) -> dict:
    """Exact-in-time PBE advance by the uniformisation series.

    ``exp(A t) n0 = e^{-lam t} sum_{k>=0} (lam t)^k / k!  P^k n0`` with
    ``A = -diag(S) + M diag(S)`` and ``lam`` large enough that
    ``P = I + A / lam >= 0`` (a generator splitting; the entries stay
    non-negative because ``M >= 0`` and the diagonal is shifted by ``lam``).
    ``P`` is applied through its *action* (``v -> v + (M (S v) - S v) / lam``),
    so the cost is one sparse mat-vec per term: forming the dense operator and
    exponentiating it is both slower and (``scipy.linalg.expm``) NaN-prone here.

    The series is truncated when the Poisson weight falls below ``tol`` relative
    to the running sum; the discarded tail is returned (``tail``) so the
    truncation error is auditable instead of assumed.

    Returns ``{"t", "n" (n_bins, n_t), "tail", "terms", "lam"}``.
    """
    S = np.asarray(selection, dtype=float)
    if np.ndim(S) == 0:
        S = np.full(np.asarray(matrix).shape[0], float(S))
    M = csr_matrix(np.asarray(matrix, dtype=float))
    K = S.size
    if M.shape != (K, K):
        raise ValueError("selection / matrix size mismatch")
    n0 = np.asarray(n0, dtype=float)
    if n0.shape != (K,):
        raise ValueError("n0 size must match the operator")

    def apply_P(v):
        sv = S * v
        return v + (M @ sv - sv) / lam

    diag_A = -S + S * np.asarray(M.diagonal(), dtype=float)
    lam = float(max(np.abs(diag_A).max() if K else 0.0, float(np.max(np.abs(S))))) * 1.05
    if lam <= 0.0:
        lam = 1.0
    if float(np.min(1.0 + diag_A / lam)) < -1e-12:
        raise RuntimeError("uniformisation matrix has negative entries -- lam too small")

    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    order = np.argsort(t_eval)
    n_out = np.empty((K, t_eval.size))
    tails = np.empty(t_eval.size)
    terms = np.empty(t_eval.size, dtype=int)
    scale = max(float(np.max(np.abs(n0))), 1e-300)
    for pos, idx in enumerate(order):
        t = float(t_eval[idx])
        w = math.exp(-lam * t)
        term = n0.copy()
        acc = w * term
        tail = 1.0 - w
        k = 0
        while k < max_terms:
            k += 1
            term = apply_P(term)
            w *= lam * t / k
            acc = acc + w * term
            tail -= w
            if k > lam * t + 5 and w * max(float(np.max(np.abs(term))), 1e-300) <= tol * scale:
                break
        n_out[:, pos] = acc
        tails[pos] = tail
        terms[pos] = k
    return {"t": t_eval, "n": n_out, "tail": tails, "terms": terms, "lam": lam}


# ---------------------------------------------------------------------------
# error metrology
# ---------------------------------------------------------------------------
def pivot_measure_bias(local_ratio) -> float:
    """Closed-form first-order bias ``mean((r_local - 1) / 2)`` of the pivot measure."""
    r = np.atleast_1d(np.asarray(local_ratio, dtype=float))
    return float(np.mean(0.5 * (r - 1.0)))


def richardson_gci(f_fine: float, f_mid: float, f_coarse: float,
                   r: float = 2.0) -> dict:
    """Roache observed order, Richardson extrapolation and GCI (safety 1.25).

    ``p = ln((f_coarse - f_mid) / (f_mid - f_fine)) / ln r``,
    ``f_ext = f_fine + (f_fine - f_mid) / (r**p - 1)`` (Roache; the code writes it
    with ``d_fine = f_mid - f_fine`` so the sign is explicit),
    ``GCI = 1.25 |f_fine - f_ext| / |f_ext|``.

    The order is **reported, never silently clamped**: a non-monotone sequence
    gets ``p = nan`` with ``certified = False`` and the extrapolation falls back
    to ``p = 1`` so the GCI stays defined but flagged.
    """
    d_fine = float(f_mid) - float(f_fine)
    d_coarse = float(f_coarse) - float(f_mid)
    out = {
        "f_fine": float(f_fine), "f_mid": float(f_mid), "f_coarse": float(f_coarse),
        "refinement_ratio": float(r),
        "delta_fine": d_fine, "delta_coarse": d_coarse,
        "monotone": bool(d_fine * d_coarse > 0.0),
        "degenerate": bool(d_fine == 0.0 and d_coarse == 0.0),
    }
    if out["degenerate"] or d_fine == 0.0:
        out.update({"p": float("nan"), "p_for_extrapolation": 1.0,
                    "f_extrapolated": float(f_fine), "gci": 0.0,
                    "asymptotic": False, "certified": False,
                    "relative_fine_to_coarse": (
                        abs(float(f_coarse) - float(f_fine)) / abs(float(f_fine))
                        if f_fine != 0.0 else float("nan"))})
        return out
    p = (math.log(abs(d_coarse) / abs(d_fine)) / math.log(r)
         if out["monotone"] else float("nan"))
    p_eff = p if (p == p and 0.05 <= p <= 4.0) else 1.0
    # Roache: f_ex = f1 + (f1 - f2) / (r**p - 1) with d_fine = f2 - f1
    f_ex = float(f_fine) - d_fine / (r ** p_eff - 1.0)
    out.update({
        "p": p, "p_for_extrapolation": p_eff, "f_extrapolated": f_ex,
        "gci": 1.25 * abs(float(f_fine) - f_ex) / abs(f_ex) if f_ex != 0.0 else float("nan"),
        "asymptotic": bool(out["monotone"] and 0.05 <= p <= 4.0),
        "certified": bool(out["monotone"] and p == p),
        "relative_fine_to_coarse": abs(float(f_coarse) - float(f_fine)) / abs(float(f_fine)),
    })
    return out


def cascade_solution(m: int, t: float, s_rate: float = 1.0) -> float:
    """Exact binary-cascade population ``e^{-S t} (2 S t)**m / m!``."""
    return math.exp(-s_rate * t) * (2.0 * s_rate * t) ** m / math.factorial(m)


def cascade_error(grid: LogLogGrid, *, sections_per_octave: int, j0: int,
                  t_eval, s_rate: float = 1.0, n_generations: int = 12) -> dict:
    """Max relative error of the discrete cascade against the exact law.

    The equal-binary rule on a log2 ladder maps the parent representative
    ``sqrt(V_j V_{j+1})`` to ``sqrt(V_{j-q} V_{j-q+1})`` exactly, so in
    generation index ``m`` the discrete system is ``dN_m/dt = -S N_m + 2 S N_{m+1}``
    whose solution is :func:`cascade_solution` for **every** ``q``.  The test is
    therefore a pure *time*-advance accuracy statement.
    """
    if not (0 <= j0 < grid.n_bins):
        raise ValueError("j0 outside grid")
    n0 = np.zeros(grid.n_bins)
    n0[j0] = 1.0
    M = daughter_matrix(grid, n_fragments=2, rule="equal")
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    run = propagate(np.full(grid.n_bins, float(s_rate)), M, n0, t_eval)
    worst, worst_at, n_gen = 0.0, None, 0
    for it, t in enumerate(run["t"]):
        for m in range(n_generations):
            idx = j0 - m * int(sections_per_octave)
            if idx < 0:
                break
            exact = cascade_solution(m, float(t), s_rate)
            err = abs(run["n"][idx, it] - exact) / exact if exact > 0.0 else abs(run["n"][idx, it])
            if err > worst:
                worst, worst_at = err, (float(t), m)
            n_gen = max(n_gen, m + 1)
    return {"max_relative_error": float(worst), "worst_at": worst_at,
            "generations_compared": int(n_gen), "n_bins": int(grid.n_bins),
            "truncation_tail": float(np.max(np.abs(run["tail"])))}


def mass_ledger(grid: LogLogGrid, n0, n_end) -> dict:
    """Mass / number ledger between the initial and the final PSD."""
    n0 = grid._check(np.asarray(n0, dtype=float))
    n_end = grid._check(np.asarray(n_end, dtype=float))
    m0 = grid.pivot_moment(n0, 1)
    m1 = grid.pivot_moment(n_end, 1)
    return {
        "mass_initial": m0, "mass_final": m1,
        "mass_drift_relative": abs(m1 - m0) / m0 if m0 else float("nan"),
        "mass_integral_initial": grid.moment(n0, 1),
        "mass_integral_final": grid.moment(n_end, 1),
        "number_initial": float(n0.sum()), "number_final": float(n_end.sum()),
    }


def floor_share(n_series, grid: LogLogGrid) -> np.ndarray:
    """Bin-0 share of the pivot mass, one value per time slice."""
    n_series = np.atleast_2d(np.asarray(n_series, dtype=float))
    if n_series.shape[0] != grid.n_bins:
        n_series = n_series.T
    mass = grid.volumes[:, None] * n_series
    total = mass.sum(axis=0)
    total = np.where(total > 0.0, total, np.nan)
    return mass[0, :] / total


def floor_threshold(ratios, shares, limit: float = 0.05) -> dict:
    """Smallest ``d_0 / v_min**(1/3)`` for which the floor share stays <= ``limit``.

    ``ratios`` must be strictly increasing.  The crossing is located by
    log-linear interpolation between the bracketing points; if the curve never
    crosses, ``threshold = None`` and ``crossed = False`` (an honest "no
    threshold in the scanned range").
    """
    ratios = np.asarray(ratios, dtype=float)
    shares = np.asarray(shares, dtype=float)
    if ratios.ndim != 1 or ratios.size != shares.size:
        raise ValueError("ratios and shares must be 1-D of equal length")
    if np.any(np.diff(ratios) <= 0):
        raise ValueError("ratios must be strictly increasing")
    for i in range(shares.size - 1):
        lo, hi = shares[i], shares[i + 1]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo > limit >= hi:
            x0, x1 = math.log(ratios[i]), math.log(ratios[i + 1])
            f = (lo - limit) / (lo - hi)
            return {"threshold": float(math.exp(x0 + f * (x1 - x0))), "crossed": True,
                    "limit": limit,
                    "bracketing_ratios": [float(ratios[i]), float(ratios[i + 1])],
                    "bracketing_shares": [float(lo), float(hi)]}
    if np.isfinite(shares[0]) and shares[0] <= limit:
        return {"threshold": float(ratios[0]), "crossed": True, "limit": limit,
                "note": "limit already satisfied at the smallest scanned ratio"}
    return {"threshold": None, "crossed": False, "limit": limit}


def erosion_removed_mass(grid: LogLogGrid, n, ratio_callable, *, rho_p: float,
                         t_end: float, f_impact: float = 1.0) -> dict:
    """Target mass removed by an impact ensemble acting on the PSD [kg].

    Two estimators of the *same* physical quantity:

    * ``pivot``: every particle of a bin is given the representative diameter
      ``d_k`` (the convention the DEM/PBM fusion of ``p176w2-n08`` uses);
    * ``integral``: the within-bin constant-density model is integrated with
      4-point Gauss-Legendre quadrature in ``v``, so the diameter *inside* a
      section is resolved instead of collapsed onto a pivot.

    Both use the same impact ensemble (``f_impact`` impacts per particle per
    second, same speed/angle, ``t_end`` seconds), so their difference is
    exactly the grid contribution to the predicted erosion rate.
    """
    n = grid._check(np.asarray(n, dtype=float))
    e = grid.edges
    d_pivot = grid.volumes ** (1.0 / 3.0)
    per_particle = (np.asarray(ratio_callable(d_pivot), dtype=float)
                    * rho_p * SPHERE_FACTOR * grid.volumes)
    pivot = float(f_impact * t_end * np.sum(n * per_particle))

    half = 0.5 * np.diff(e)
    mid = 0.5 * (e[:-1] + e[1:])
    total = 0.0
    for xi, wi in zip(_GAUSS_X, _GAUSS_W):
        v = mid + xi * half
        d = np.abs(v) ** (1.0 / 3.0)
        total += float(np.sum((wi * half / grid.widths) * n
                              * np.asarray(ratio_callable(d), dtype=float)
                              * rho_p * SPHERE_FACTOR * v))
    integral = float(f_impact * t_end * total)
    return {"removed_pivot": pivot, "removed_integral": integral,
            "relative_difference": abs(integral - pivot) / pivot if pivot else float("nan")}


def psd_fingerprint(n) -> tuple:
    """Bit-level fingerprint of a PSD (anti-false-convergence control)."""
    arr = np.ascontiguousarray(np.asarray(n, dtype=float))
    return (arr.shape, arr.tobytes())