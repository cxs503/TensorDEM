"""Bias of the ``sphere + mu_r`` proxy against the polyhedral truth, and GSA.

The proxy question
------------------
A spherical fragment can only resist rolling through an imposed couple
``mu_r R F_n``.  A faceted fragment resists rolling because a facet must be
lifted over an edge -- a *geometric* effect with an a-priori coefficient
``mu_geo = <s_e / h_f>`` (see :mod:`tensordem.ice_contact.shapes`).  Replacing
the facet by a sphere plus ``mu_r`` is therefore a modelling *proxy*, and its
systematic error is

    ``delta(mu_r) = (F_poly - F_sphere(mu_r)) / F_poly``

evaluated on the structure normal force (the load the structure actually
carries) and, separately, on the tangential traction (where angularity is
expected to matter most).

Reported for every condition:

* ``delta_mean``, ``delta_rms`` -- level and fluctuation bias [dimensionless],
* ``r2_raw``, ``r2_block`` -- squared correlation of the polyhedral and proxy
  force histories, raw and block averaged over the crushing-zone turnover
  ``h / v`` (the scale a structural design cares about),
* ``mu_r_star`` -- the ``mu_r`` at which ``delta_mean`` vanishes (linear
  interpolation on the ``mu_r`` grid), compared with ``mu_geo``.

Global sensitivity
------------------
:func:`saltelli_sobol` is a self-contained Saltelli/Jansen estimator; it is
validated in the test suite against an analytic additive model whose indices are
exact.  It is applied to ``|delta|`` over
``(mu_r, v, h_ice, angularity, mu_pp)`` on a *reduced-fidelity* copy of the same
DEM (smaller domain, fewer fragments), and the reduced model is checked against
the full-fidelity runs at the nominal condition before its ranking is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ProxyBias",
    "proxy_bias",
    "block_average",
    "solve_equivalence",
    "geometric_rolling_reference",
    "saltelli_sobol",
]


def block_average(series: np.ndarray, block: int) -> np.ndarray:
    """Non-overlapping block means (drops the incomplete tail)."""
    s = np.asarray(series, dtype=float)
    if block < 1:
        raise ValueError("block must be >= 1")
    n = len(s) // block
    if n < 1:
        return s.copy()
    return s[: n * block].reshape(n, block).mean(axis=1)


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


@dataclass
class ProxyBias:
    """Proxy bias at one ``mu_r`` for one condition."""

    mu_r: float
    force_poly_mean: float
    force_sphere_mean: float
    delta_mean: float
    delta_rms: float
    r2_raw: float
    r2_block: float
    block_samples: int
    tangential_delta: float
    tangential_r2_block: float

    def to_json(self) -> dict:
        return {
            "mu_r": self.mu_r,
            "force_poly_mean": self.force_poly_mean,
            "force_sphere_mean": self.force_sphere_mean,
            "delta_mean": self.delta_mean,
            "delta_rms": self.delta_rms,
            "abs_delta_mean": abs(self.delta_mean),
            "r2_raw": self.r2_raw,
            "r2_block": self.r2_block,
            "block_samples": self.block_samples,
            "tangential_delta": self.tangential_delta,
            "tangential_r2_block": self.tangential_r2_block,
        }


def proxy_bias(
    force_poly: np.ndarray,
    force_sphere: np.ndarray,
    mu_r: float,
    *,
    block: int = 5,
    tangent_poly: np.ndarray | None = None,
    tangent_sphere: np.ndarray | None = None,
    transient: float = 0.25,
) -> ProxyBias:
    """Bias of the spherical proxy relative to the polyhedral truth.

    Both histories must be sampled on the same time grid (they are: the platen
    stroke, velocity and time step are shared by every realisation of a
    condition).  The initial ``transient`` fraction is discarded.
    """
    fp = np.asarray(force_poly, dtype=float)
    fs = np.asarray(force_sphere, dtype=float)
    if fp.shape != fs.shape:
        raise ValueError("force histories must share the time grid")
    start = int(round(transient * (len(fp) - 1)))
    fp, fs = fp[start:], fs[start:]
    ref = float(fp.mean())
    if ref <= 0.0:
        raise ValueError("polyhedral force is zero; no contact with the structure")
    delta_mean = float((ref - fs.mean()) / ref)
    delta_rms = float(np.sqrt(np.mean((fp - fs) ** 2)) / ref)
    b = max(1, int(block))
    r2_block = _r2(block_average(fp, b), block_average(fs, b))
    if tangent_poly is not None and tangent_sphere is not None:
        tp = np.asarray(tangent_poly, dtype=float)[start:]
        ts = np.asarray(tangent_sphere, dtype=float)[start:]
        tan_ref = float(np.abs(tp).mean())
        tangential_delta = (
            float((np.abs(tp).mean() - np.abs(ts).mean()) / tan_ref) if tan_ref > 0 else float("nan")
        )
        tangential_r2 = _r2(block_average(np.abs(tp), b), block_average(np.abs(ts), b))
    else:
        tangential_delta = float("nan")
        tangential_r2 = float("nan")
    return ProxyBias(
        mu_r=float(mu_r),
        force_poly_mean=ref,
        force_sphere_mean=float(fs.mean()),
        delta_mean=delta_mean,
        delta_rms=delta_rms,
        r2_raw=_r2(fp, fs),
        r2_block=r2_block,
        block_samples=b,
        tangential_delta=tangential_delta,
        tangential_r2_block=tangential_r2,
    )


def solve_equivalence(
    mu_r_grid: np.ndarray, delta: np.ndarray
) -> dict:
    """``mu_r*`` where the proxy bias changes sign, plus the residual bias.

    Returns ``{"mu_r_star", "residual", "bracketed"}``; ``bracketed`` is False
    when the bias does not change sign on the grid, in which case ``mu_r_star``
    is reported as ``nan`` and must not be quoted as a calibration.
    """
    grid = np.asarray(mu_r_grid, dtype=float)
    d = np.asarray(delta, dtype=float)
    order = np.argsort(grid)
    grid, d = grid[order], d[order]
    for k in range(len(grid) - 1):
        if d[k] == 0.0:
            return {"mu_r_star": float(grid[k]), "residual": 0.0, "bracketed": True}
        if d[k] * d[k + 1] < 0.0:
            frac = d[k] / (d[k] - d[k + 1])
            return {
                "mu_r_star": float(grid[k] + frac * (grid[k + 1] - grid[k])),
                "residual": 0.0,
                "bracketed": True,
            }
    nearest = int(np.argmin(np.abs(d)))
    return {"mu_r_star": float("nan"), "residual": float(d[nearest]), "bracketed": False,
            "best_grid_mu_r": float(grid[nearest])}


def geometric_rolling_reference(shapes) -> dict:
    """Area-weighted mean geometric rolling coefficient of an assembly."""
    values = np.array([s.rolling_geometry for s in shapes], dtype=float)
    weights = np.array([s.volume for s in shapes], dtype=float)
    keep = np.isfinite(values)
    if not np.any(keep):
        return {"value": float("nan"), "std": float("nan"), "n": 0}
    v = values[keep]
    w = weights[keep]
    mean = float((v * w).sum() / w.sum())
    std = float(np.sqrt((w * (v - mean) ** 2).sum() / w.sum()))
    return {"value": mean, "std": std, "n": int(keep.sum())}


def saltelli_sobol(evaluate, base: np.ndarray, other: np.ndarray) -> dict:
    """Saltelli/Jansen first-order and total Sobol indices.

    ``base`` and ``other`` are ``(N, d)`` matrices of model inputs drawn from the
    same distribution (usually uniform on the unit hypercube, and the caller maps
    them to physical bounds).  ``evaluate`` maps an ``(n, d)`` array to an
    ``(n,)`` array of scalar outputs.  Estimators:

        ``S1_i = <y_B (y_AB_i - y_A)> / Var``
        ``ST_i = <(y_A - y_AB_i)^2> / (2 Var)``

    exactly as in Saltelli et al. (2010).
    """
    a = np.atleast_2d(np.asarray(base, dtype=float))
    b = np.atleast_2d(np.asarray(other, dtype=float))
    if a.shape != b.shape:
        raise ValueError("base and other must share shape")
    n, d = a.shape
    y_a = np.asarray(evaluate(a), dtype=float).ravel()
    y_b = np.asarray(evaluate(b), dtype=float).ravel()
    variance = float(np.var(np.concatenate([y_a, y_b])))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("model output is constant; Sobol indices are undefined")
    s1 = np.zeros(d)
    st = np.zeros(d)
    for i in range(d):
        hybrid = a.copy()
        hybrid[:, i] = b[:, i]
        y_ab = np.asarray(evaluate(hybrid), dtype=float).ravel()
        s1[i] = float(np.mean(y_b * (y_ab - y_a)) / variance)
        st[i] = float(0.5 * np.mean((y_a - y_ab) ** 2) / variance)
    return {
        "S1": s1,
        "ST": st,
        "n_base": int(n),
        "variance": variance,
        "evaluations": int(n * (d + 2)),
        "mean_a": float(y_a.mean()),
        "mean_b": float(y_b.mean()),
    }