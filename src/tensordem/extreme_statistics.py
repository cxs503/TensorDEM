"""Extreme-value statistics and sampling-error analysis for wear/erosion fields.

Why this module exists
----------------------
:mod:`tensordem.ice_abrasion` produces a *depth field* on the waterline band and
reports its P95 ("survey-patch") depth as the design statistic.  W10-N06 then
compared P95 across three band-grid resolutions and reported a Roache
grid-convergence index of 19.8 %, which cannot be accepted as a converged
quantity -- but it is equally wrong to call it a numerical error without a
statistical yardstick.  Three distinct effects are entangled in that number:

1. **Sampling error** of the P95 estimator (finite contact ensemble, finite
   number of deposited pits, random cell placement).  This is a confidence
   interval question, answered here by replicate runs, block bootstrap and
   extreme-value (GEV/Gumbel) profile likelihood.
2. **Measure dependence** of a shot-noise quantile: for dilute pits of radius
   ``a`` in cells of size ``h`` the cell-averaged depth is a compound-Poisson
   functional whose quantiles scale with the cell area, so ``h -> 0`` has no
   finite limit.  This is an analytic property of the field, not a
   discretisation error; it is captured by :func:`shot_noise_quantile` and
   :func:`shot_noise_resolution_law`.
3. **Numerical error** of the pit projection, which conserves volume exactly and
   is verified separately (``ice_abrasion.project_contact_pits`` tests).

The module supplies (a) GEV/Gumbel MLE with *profile-likelihood* intervals,
(b) coverage checks of those intervals against Monte Carlo, (c) recovery of the
extreme-value index ``xi`` and of the *extremal index* ``theta`` with
block-length stability scans, (d) bootstrap / replicate intervals for field
quantiles, (e) the analytic shot-noise resolution law, and (f) the adjudication
rule that decides whether a grid-convergence index lies inside or outside the
statistical band.

Conventions
-----------
* GEV ``F(x) = exp(-(1 + xi*y) ** (-1/xi))``, ``y = (x - mu)/sigma``, with the
  Gumbel limit ``xi -> 0``.  ``xi > 0`` is Frechet (heavy tail), ``xi < 0`` the
  bounded Weibull domain.
* ``p`` is a probability (0.95 means P95), ``level`` a confidence level.
* Profile intervals use the likelihood-ratio cut ``2*(l_max - l_profile) <=
  chi2(1, level)``.
* Pure ``numpy``/``scipy``; field utilities take SI depth arrays (metres).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GEVParams", "gev_cdf", "gev_ppf", "gev_loglik", "gev_fit", "gumbel_fit",
    "gumbel_profile_ci", "gev_profile_ci", "delta_method_ci",
    "order_statistic_ci", "coverage_study", "hill_estimator", "block_maxima",
    "evi_recovery_study", "max_ar1_series", "extremal_index_runs",
    "extremal_index_blocks", "extremal_index_stability", "theta_recovery_study",
    "bootstrap_quantile_ci", "replicate_quantile_ci", "shot_noise_quantile",
    "shot_noise_resolution_law", "adjudicate_gci_vs_ci",
    "wear_field_p95_certificate", "acceptance_protocol_evaluate",
]

_CHI2_1 = {0.90: 2.7055434, 0.95: 3.8414588, 0.99: 6.6348966}
_XI_EPS = 1.0e-7


# ---------------------------------------------------------------------------
# 1. GEV / Gumbel distribution, log-likelihood and gradients
# ---------------------------------------------------------------------------


def _as_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("need at least two observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observations must be finite")
    return arr


def _chi2_cut(level: float) -> float:
    for lev, val in _CHI2_1.items():
        if abs(level - lev) < 1e-12:
            return val
    from scipy import stats

    return float(stats.chi2.ppf(level, 1))


def _log_p_term(p: float) -> float:
    """``-log(-log p)``: the Gumbel reduced variate of probability ``p``."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    return -math.log(-math.log(p))


def _quantile_term(xi: float, p: float) -> float:
    """``h(xi)`` in ``q_p = mu + sigma * h(xi)`` with the Gumbel limit."""
    g = _log_p_term(p)
    if abs(xi) < 1e-6:
        return g - 0.5 * xi * g * g
    lg = math.log(-math.log(p))
    return (math.exp(-xi * lg) - 1.0) / xi


@dataclass(frozen=True)
class GEVParams:
    """GEV location, scale and shape (``xi > 0`` Frechet, ``xi < 0`` Weibull)."""

    mu: float
    sigma: float
    xi: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not (math.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("sigma must be positive")
        if not math.isfinite(self.xi):
            raise ValueError("xi must be finite")

    def cdf(self, x: Any) -> np.ndarray:
        return gev_cdf(self, x)

    def ppf(self, p: Any) -> np.ndarray:
        return gev_ppf(self, p)

    def quantile(self, p: float) -> float:
        return float(gev_ppf(self, np.array([p]))[0])

    def return_level(self, period: float) -> float:
        if not period > 1.0:
            raise ValueError("period must exceed 1")
        return self.quantile(1.0 - 1.0 / float(period))

    def summary(self) -> dict[str, Any]:
        kind = "gumbel" if abs(self.xi) < 1e-6 else ("frechet" if self.xi > 0 else "weibull")
        return {"mu": self.mu, "sigma": self.sigma, "xi": self.xi, "domain": kind}


def gev_cdf(params: GEVParams, x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    y = (arr - params.mu) / params.sigma
    xi = params.xi
    if abs(xi) < _XI_EPS:
        return np.exp(-np.exp(-y))
    t = 1.0 + xi * y
    out = np.zeros_like(t)
    ok = t > 0.0
    out[ok] = np.exp(-np.power(t[ok], -1.0 / xi))
    return out


def gev_ppf(params: GEVParams, p: Any) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if np.any((arr <= 0.0) | (arr >= 1.0)):
        raise ValueError("p must lie in (0, 1)")
    xi = params.xi
    if abs(xi) < _XI_EPS:
        return params.mu - params.sigma * np.log(-np.log(arr))
    h = (np.power(-np.log(arr), -xi) - 1.0) / xi
    return params.mu + params.sigma * h


def gev_loglik(params: GEVParams, x: Any, *, gradient: bool = False):
    """GEV log-likelihood and its analytic gradient w.r.t. ``(mu, sigma, xi)``."""
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    y = (arr - params.mu) / params.sigma
    xi = params.xi
    if abs(xi) < _XI_EPS:
        e = np.exp(-y)
        ll = -n * math.log(params.sigma) - float(y.sum()) - float(e.sum())
        if not gradient:
            return ll
        g_mu = float((1.0 - e).sum()) / params.sigma
        g_sigma = -n / params.sigma + float((y * (1.0 - e)).sum()) / params.sigma
        g_xi = -float(y.sum()) + 0.5 * float((y * y * (1.0 - e)).sum())
        return ll, np.array([g_mu, g_sigma, g_xi])
    t = 1.0 + xi * y
    if np.any(t <= 0.0):
        return (-math.inf, np.full(3, np.nan)) if gradient else -math.inf
    tm = np.power(t, -1.0 / xi)
    ll = (-n * math.log(params.sigma)
          - (1.0 + 1.0 / xi) * float(np.log(t).sum()) - float(tm.sum()))
    if not gradient:
        return ll
    common = (1.0 + xi) / t - np.power(t, -1.0 / xi - 1.0)
    g_mu = float(common.sum()) / params.sigma
    g_sigma = -n / params.sigma + float((y * common).sum()) / params.sigma
    g_xi = (float((np.log(t) * (1.0 - tm)).sum()) / (xi * xi)
            - float((y / t).sum()) * (1.0 + 1.0 / xi)
            + float((y * np.power(t, -1.0 / xi - 1.0)).sum()) / xi)
    return ll, np.array([g_mu, g_sigma, g_xi])


def _gumbel_profile_sigma(x: np.ndarray, sigma: float) -> tuple[float, float]:
    """Profile of the Gumbel likelihood over ``mu`` at fixed ``sigma``.

    ``d l / d mu = 0`` gives ``sum exp(-y_i) = n``, i.e.
    ``mu_hat(sigma) = sigma * log(mean(exp(x / sigma)))``, so the profile is
    one-dimensional and the profile interval of any quantile is exact.
    """
    n = x.size
    lo = float(np.min(x))
    s = np.exp(-(x - lo) / sigma)
    mu = -sigma * math.log(float(s.mean())) + lo
    y = (x - mu) / sigma
    ll = -n * math.log(sigma) - float(y.sum()) - n
    return ll, mu


def gumbel_fit(x: Any) -> GEVParams:
    """Two-parameter Gumbel MLE (``xi = 0`` held fixed); location profiled out."""
    from scipy.optimize import minimize_scalar

    arr = _as_array(x)
    s0 = float(arr.std(ddof=1)) * 1.2825
    if not (np.isfinite(s0) and s0 > 0.0):
        s0 = max(float(np.ptp(arr)), 1.0) * 0.1

    def neg(ls: float) -> float:
        return -_gumbel_profile_sigma(arr, math.exp(float(ls)))[0]

    res = minimize_scalar(neg, bounds=(math.log(s0 * 0.2), math.log(s0 * 5.0)),
                          method="bounded", options={"xatol": 1e-10})
    sigma = math.exp(float(res.x))
    ll, mu = _gumbel_profile_sigma(arr, sigma)
    return GEVParams(mu=mu, sigma=sigma, xi=0.0)


def gev_fit(x: Any, *, xi: float | None = None, maxiter: int = 500) -> GEVParams:
    """GEV maximum likelihood; ``xi`` may be held fixed (used by the profile).

    Analytic-gradient L-BFGS-B from three shape starts (0, +0.1, -0.1); the
    gradient is exact, so the fit is reproducible and fast enough to be nested
    inside the profile-likelihood bisection.
    """
    from scipy.optimize import minimize

    arr = _as_array(x)
    g = gumbel_fit(arr)

    def fun(pp: np.ndarray, shape: float | None):
        mu, ls, xi_v = float(pp[0]), float(pp[1]), (shape if shape is not None
                                                    else float(pp[2]))
        ll, grad = gev_loglik(GEVParams(mu, math.exp(ls), xi_v), arr, gradient=True)
        if not math.isfinite(ll) or not np.all(np.isfinite(grad)):
            return 1e30, np.zeros_like(pp)
        jac = np.array([-grad[0], -grad[1] * math.exp(ls), -grad[2]])
        return -ll, jac

    if xi is not None:
        shape = float(xi)
        res = minimize(lambda pp: fun(np.array([pp[0], pp[1], 0.0]), shape)[0],
                       np.array([g.mu, math.log(g.sigma)]), method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 2000,
                                "maxfev": 2000})
        return GEVParams(float(res.x[0]), math.exp(float(res.x[1])), shape)

    bounds = [(-np.inf, np.inf), (math.log(g.sigma) - 8.0, math.log(g.sigma) + 8.0),
              (-2.5, 2.5)]
    best: tuple[float, GEVParams] | None = None
    for xi0 in (0.0, 0.1, -0.1):
        x0 = np.array([g.mu, math.log(g.sigma), xi0])
        res = minimize(lambda pp: fun(pp, None), x0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-15,
                                               "gtol": 1e-11})
        cand = GEVParams(float(res.x[0]), math.exp(float(res.x[1])), float(res.x[2]))
        ll = gev_loglik(cand, arr)
        if math.isfinite(ll) and (best is None or ll > best[0]):
            best = (ll, cand)
    if best is None:
        raise RuntimeError("GEV fit failed from every start point")
    return best[1]


# ---------------------------------------------------------------------------
# 2. Confidence intervals: profile likelihood, delta method, order statistics
# ---------------------------------------------------------------------------


def gumbel_profile_ci(x: Any, p: float = 0.95, level: float = 0.95, *,
                      n_scan: int = 240) -> dict[str, Any]:
    """Profile-likelihood interval of the ``p``-quantile of a Gumbel parent.

    Location is profiled out analytically (see :func:`_gumbel_profile_sigma`), so
    this is the exact profile interval of the two-parameter likelihood, and the
    quantile map ``sigma -> q`` is monotone -- the interval is obtained by
    bracketing the likelihood-ratio crossing on a ``log sigma`` scan and refining
    it with Brent's method.
    """
    from scipy.optimize import brentq

    arr = _as_array(x)
    g = _log_p_term(p)
    fit = gumbel_fit(arr)
    ll_max, _ = _gumbel_profile_sigma(arr, fit.sigma)
    cut = _chi2_cut(level)
    ls_hat = math.log(fit.sigma)

    def prof(ls: float) -> tuple[float, float]:
        ll, mu = _gumbel_profile_sigma(arr, math.exp(ls))
        return ll, mu + math.exp(ls) * g

    def dev(ls: float) -> float:
        return 2.0 * (ll_max - prof(ls)[0])

    def solve(direction: float) -> float:
        step = 1.0e-3
        inside = ls_hat
        for _ in range(80):
            trial = inside + direction * step
            if dev(trial) >= cut:
                a, b = sorted((inside, trial))
                if dev(a) - cut == 0.0:
                    return a
                return brentq(lambda z: dev(z) - cut, a, b, xtol=1e-14)
            inside = trial
            step *= 1.7
        return math.nan                      # pragma: no cover - never escaped

    ls_lo = solve(-1.0)
    ls_hi = solve(+1.0)
    if not (math.isfinite(ls_lo) and math.isfinite(ls_hi)):   # pragma: no cover
        return {"estimate": prof(ls_hat)[1], "ci_lo": math.nan, "ci_hi": math.nan,
                "method": "gumbel-profile-likelihood"}
    est = prof(ls_hat)[1]
    qs = [prof(ls_lo)[1], prof(ls_hi)[1]]
    lo, hi = min(qs), max(qs)
    return {"estimate": est, "ci_lo": lo, "ci_hi": hi,
            "ci_rel_width_pct": (hi - lo) / abs(est) * 100.0 if est else math.nan,
            "ci_rel_halfwidth_pct": 0.5 * (hi - lo) / abs(est) * 100.0 if est else math.nan,
            "loglik_max": ll_max, "level": level, "p": p,
            "method": "gumbel-profile-likelihood"}


def _quantile_term_deriv(xi: float, p: float, h: float = 1e-5) -> float:
    """``d h / d xi`` by central differences (``h`` is smooth and cheap)."""
    return (_quantile_term(xi + h, p) - _quantile_term(xi - h, p)) / (2.0 * h)


def gev_profile_ci(x: Any, p: float = 0.95, level: float = 0.95, *,
                   maxiter: int = 200) -> dict[str, Any]:
    """Profile-likelihood interval of the ``p``-quantile of a GEV parent.

    The quantile is the parameter of interest: for a trial ``q`` the location is
    eliminated by ``mu = q - sigma * h(xi, p)`` and the likelihood is maximised
    over ``(log sigma, xi)`` with the analytic gradient and a warm start from the
    previous trial point.  The interval is the set where the likelihood-ratio
    statistic ``2 (l_max - l_profile)`` stays below the chi-square(1) cut; both
    bounds are bracketed and bisected, so the asymmetry of a small-sample tail
    interval is retained instead of being symmetrised by a delta method.
    """
    from scipy.optimize import minimize

    arr = _as_array(x)
    full = gev_fit(arr)
    ll_max = gev_loglik(full, arr)
    q_hat = full.quantile(p)
    cut = _chi2_cut(level)
    h_hat = _quantile_term(full.xi, p)
    x0_hat = np.array([math.log(full.sigma), full.xi])
    bounds = [(math.log(full.sigma) - 8.0, math.log(full.sigma) + 8.0), (-2.5, 2.5)]

    def profile(trial: float, x0: np.ndarray | None = None):
        """Maximise the likelihood over (log sigma, xi) at a fixed trial quantile.

        A coarse two-dimensional scan locates the basin and Nelder-Mead polishes
        it: the constrained problem is non-convex on its support boundary (for
        ``xi > 0`` the location is bounded from above by the largest observation),
        and a gradient line search started inside the boundary can be trapped, so
        a scanning start is used instead of a warm start only.
        """
        from scipy.optimize import minimize

        best = (-math.inf, x0_hat)
        grid_xi = np.linspace(-1.2, 1.2, 13)
        relax = 1.0 if x0 is None else 0.4
        ls_hat = x0_hat[0] if x0 is None else float(x0[0])
        grid_ls = ls_hat + np.linspace(-4.0, 4.0, 11) * relax
        for xi in grid_xi:
            hh = _quantile_term(float(xi), p)
            for ls in grid_ls:
                ll = gev_loglik(GEVParams(trial - math.exp(float(ls)) * hh,
                                          math.exp(float(ls)), float(xi)), arr)
                if math.isfinite(ll) and ll > best[0]:
                    best = (ll, np.array([float(ls), float(xi)]))
        if not math.isfinite(best[0]):
            return best

        def obj(pp: np.ndarray) -> float:
            sigma = math.exp(float(pp[0]))
            mu = trial - sigma * _quantile_term(float(pp[1]), p)
            ll = gev_loglik(GEVParams(mu, sigma, float(pp[1])), arr)
            return -ll if math.isfinite(ll) else 1e30

        res = minimize(obj, best[1], method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 600,
                                "maxfev": 600})
        if -float(res.fun) > best[0] and math.isfinite(float(res.fun)):
            best = (-float(res.fun), res.x)
        return best

    def solve(direction: float) -> float:
        state = {"best": ll_max}
        # bracket with a step of 40 % of the (delta-method) standard error
        se = None
        try:
            dci = delta_method_ci(arr, p, level, fit=full)
            if math.isfinite(dci.get("se", math.nan)) and dci["se"] > 0:
                se = dci["se"]
        except Exception:  # pragma: no cover - fall through to the relative step
            se = None
        step = 0.4 * se if se else max(abs(q_hat) * 0.1, 1e-15)
        trial = q_hat
        x0 = None
        for k in range(200):
            trial = trial + direction * step
            ll, vec = profile(trial, x0)
            # the profile likelihood of a unimodal model cannot rise again as the
            # trial moves away from the MLE: enforce monotonicity so that an
            # occasional local optimum cannot truncate the interval
            ll = min(ll, state["best"])
            state["best"] = ll
            cur = 2.0 * (ll_max - ll) if math.isfinite(ll) else math.inf
            if cur >= cut:
                a, b = sorted((trial - direction * step, trial))
                for _ in range(45):
                    mid = 0.5 * (a + b)
                    llm, _v = profile(mid, vec)
                    cm = 2.0 * (ll_max - llm) if math.isfinite(llm) else math.inf
                    if cm >= cut:
                        if direction > 0:
                            b = mid
                        else:
                            a = mid
                    else:
                        if direction > 0:
                            a = mid
                        else:
                            b = mid
                return 0.5 * (a + b)
            x0 = vec
            step *= 1.7
            if abs(trial - q_hat) > 40.0 * max(abs(q_hat), 1e-30):
                break
        return math.nan

    lo = solve(-1.0)
    hi = solve(+1.0)
    ok = math.isfinite(lo) and math.isfinite(hi)
    return {"estimate": q_hat, "ci_lo": lo, "ci_hi": hi,
            "ci_rel_width_pct": ((hi - lo) / abs(q_hat) * 100.0) if ok and q_hat else math.nan,
            "ci_rel_halfwidth_pct": (0.5 * (hi - lo) / abs(q_hat) * 100.0
                                     if ok and q_hat else math.nan),
            "level": level, "p": p, "method": "gev-profile-likelihood",
            "params": full.summary()}


def delta_method_ci(x: Any, p: float = 0.95, level: float = 0.95, *,
                    fit: GEVParams | None = None) -> dict[str, Any]:
    """Normal-approximation (Hessian / delta-method) interval, for comparison."""
    from scipy import stats

    arr = _as_array(x)
    full = gev_fit(arr) if fit is None else fit
    q_hat = full.quantile(p)
    p0 = np.array([full.mu, math.log(full.sigma), full.xi])

    def nll(pp: np.ndarray) -> float:
        return -gev_loglik(GEVParams(pp[0], math.exp(pp[1]), pp[2]), arr)

    h = np.zeros((3, 3))
    eps = np.array([abs(full.sigma) * 1e-4, 1e-4, 1e-4])
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                pp = p0.copy()
                pp[i] += si * eps[i]
                pp[j] += sj * eps[j]
                acc += si * sj * nll(pp)
            h[i, j] = acc / (4.0 * eps[i] * eps[j])
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:               # pragma: no cover
        return {"estimate": q_hat, "ci_lo": math.nan, "ci_hi": math.nan,
                "method": "delta-method(singular)"}
    dq = np.array([1.0, full.sigma * _quantile_term(full.xi, p),
                   full.sigma * (_quantile_term(full.xi + 1e-5, p)
                                 - _quantile_term(full.xi - 1e-5, p)) / 2e-5])
    var = float(dq @ cov @ dq)
    se = math.sqrt(max(var, 0.0))
    z = float(stats.norm.ppf(0.5 + 0.5 * level))
    return {"estimate": q_hat, "ci_lo": q_hat - z * se, "ci_hi": q_hat + z * se,
            "ci_rel_halfwidth_pct": (z * se / abs(q_hat) * 100.0) if q_hat else math.nan,
            "se": se, "level": level, "p": p, "method": "delta-method"}


def order_statistic_ci(x: Any, p: float = 0.95, level: float = 0.95) -> dict[str, Any]:
    """Distribution-free (binomial order-statistic) interval of the ``p``-quantile."""
    from scipy import stats

    arr = np.sort(_as_array(x))
    n = arr.size
    lo_k = int(stats.binom.ppf((1.0 - level) / 2.0, n, p))
    hi_k = int(stats.binom.ppf(1.0 - (1.0 - level) / 2.0, n, p))
    lo_k = max(1, min(n, lo_k))
    hi_k = max(1, min(n, hi_k))
    est = float(np.quantile(arr, p))
    lo, hi = float(arr[lo_k - 1]), float(arr[hi_k - 1])
    return {"estimate": est, "ci_lo": lo, "ci_hi": hi,
            "ci_rel_halfwidth_pct": (0.5 * (hi - lo) / abs(est) * 100.0) if est else math.nan,
            "rank_lo": lo_k, "rank_hi": hi_k, "n": n, "level": level, "p": p,
            "method": "order-statistic(binomial)"}


def coverage_study(params: GEVParams, *, n: int = 200, n_rep: int = 200,
                   p: float = 0.95, level: float = 0.95, seed: int = 20261106,
                   methods: tuple[str, ...] = ("profile", "delta", "order"),
                   gumbel_parent: bool = True) -> dict[str, Any]:
    """Monte Carlo coverage of the candidate intervals (profile vs delta vs order).

    ``gumbel_parent=True`` draws from a Gumbel parent and uses the fast analytic
    profile; otherwise the GEV profile is used (slower, and reported with a
    reduced replicate count by callers).  The output is the empirical coverage of
    each method together with the mean relative half-width, which is the quantity
    that decides whether a confidence interval can be used as a verdict.
    """
    rng = np.random.default_rng(seed)
    if gumbel_parent:
        base = GEVParams(mu=params.mu, sigma=params.sigma, xi=0.0)
    else:
        base = params
    truth = base.quantile(p)
    hits = {m: 0 for m in methods}
    widths = {m: [] for m in methods}
    fails = {m: 0 for m in methods}
    for _ in range(int(n_rep)):
        u = rng.random(int(n))
        sample = gev_ppf(base, u)
        if methods and "profile" in methods:
            ci = (gumbel_profile_ci(sample, p, level) if gumbel_parent
                  else gev_profile_ci(sample, p, level))
            _tally(hits, widths, fails, "profile", ci, truth)
        if "delta" in methods:
            _tally(hits, widths, fails, "delta", delta_method_ci(sample, p, level), truth)
        if "order" in methods:
            _tally(hits, widths, fails, "order", order_statistic_ci(sample, p, level), truth)
    out = {"n": int(n), "n_rep": int(n_rep), "p": p, "level": level,
           "truth": truth, "parent": base.summary(), "methods": {}}
    for m in methods:
        used = int(n_rep) - fails[m]
        out["methods"][m] = {
            "coverage": hits[m] / used if used else math.nan,
            "coverage_se": (math.sqrt(max(hits[m] / used * (1 - hits[m] / used), 0.0)
                                      / used) if used else math.nan),
            "mean_rel_halfwidth_pct": float(np.mean(widths[m])) if widths[m] else math.nan,
            "n_unresolved": fails[m],
        }
    return out


def _tally(hits: dict, widths: dict, fails: dict, key: str, ci: dict, truth: float) -> None:
    lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        fails[key] += 1
        return
    if lo <= truth <= hi:
        hits[key] += 1
    w = ci.get("ci_rel_halfwidth_pct")
    if w is not None and math.isfinite(w):
        widths[key].append(float(w))


# ---------------------------------------------------------------------------
# 3. Extreme-value index (xi) and extremal index (theta)
# ---------------------------------------------------------------------------


def hill_estimator(x: Any, k: int) -> dict[str, Any]:
    """Hill estimator of the extreme-value index for a Frechet-type tail.

    ``xi = mean(log x_(i) - log x_(k+1))`` for the ``k`` largest order statistics;
    asymptotic standard error ``xi / sqrt(k)``.
    """
    arr = np.sort(_as_array(x))[::-1]
    if not 1 <= int(k) < arr.size:
        raise ValueError("k must satisfy 1 <= k < n")
    kk = int(k)
    if np.any(arr[:kk] <= 0.0) or arr[kk] <= 0.0:
        raise ValueError("Hill estimator needs strictly positive upper tail")
    xi = float(np.mean(np.log(arr[:kk]) - math.log(arr[kk])))
    return {"xi": xi, "k": kk, "n": int(arr.size),
            "se": xi / math.sqrt(kk), "threshold": float(arr[kk])}


def block_maxima(x: Any, block: int) -> np.ndarray:
    """Block maxima of a series (last partial block is dropped)."""
    arr = _as_array(x)
    b = int(block)
    if b < 2 or b > arr.size:
        raise ValueError("block size outside [2, n]")
    m = arr.size // b
    return arr[:m * b].reshape(m, b).max(axis=1)


def evi_recovery_study(*, parents: dict[str, GEVParams] | None = None,
                       n: int = 20000, blocks: tuple[int, ...] = (10, 50, 200),
                       seed: int = 20261106, n_rep: int = 5) -> dict[str, Any]:
    """Recover the extreme-value index ``xi`` of Gumbel / Frechet / Weibull parents.

    Block-maxima convergence: maxima of ``b`` i.i.d. draws are GEV with the *same*
    shape ``xi``, so fitting block maxima of increasing length must recover the
    parent shape while the location drifts.  The replicate scatter gives the
    sampling error of ``xi`` (which is exactly why a bare ``xi`` has to be quoted
    with a profile interval).
    """
    if parents is None:
        parents = {"gumbel": GEVParams(0.0, 1.0, 0.0),
                   "frechet": GEVParams(1.0, 1.0, 0.25),
                   "weibull": GEVParams(1.0, 1.0, -0.25)}
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"n": int(n), "blocks": list(blocks), "n_rep": int(n_rep),
                           "parents": {}}
    for name, parent in parents.items():
        rows = []
        for b in blocks:
            xi_hat, mu_hat, sig_hat = [], [], []
            for _ in range(int(n_rep)):
                sample = gev_ppf(parent, rng.random(int(n)))
                bm = block_maxima(sample, int(b))
                fit = gev_fit(bm)
                xi_hat.append(fit.xi)
                mu_hat.append(fit.mu)
                sig_hat.append(fit.sigma)
            rows.append({"block": int(b),
                         "xi_mean": float(np.mean(xi_hat)),
                         "xi_sd": float(np.std(xi_hat, ddof=1)) if n_rep > 1 else math.nan,
                         "xi_rel_error_pct": (abs(float(np.mean(xi_hat)) - parent.xi)
                                              / max(abs(parent.xi), 1e-3) * 100.0),
                         "xi_abs_error": float(np.mean(xi_hat)) - parent.xi,
                         "mu_mean": float(np.mean(mu_hat)),
                         "sigma_mean": float(np.mean(sig_hat)),
                         "n_block_maxima": int(n // b)})
        out["parents"][name] = {"true_xi": parent.xi, "params": parent.summary(),
                                "rows": rows}
    return out


def max_ar1_series(n: int, a: float, *, seed: int = 20261106,
                   burn: int = 500) -> np.ndarray:
    """Stationary max-autoregressive AR(1) with unit-Frechet marginals.

    ``X_t = max(a X_{t-1}, (1 - a) Z_t)`` with ``Z`` unit Frechet is exactly
    marginal unit Frechet (``P(max(aX, (1-a)Z) <= x) = exp(-1/x)``) and has
    extremal index ``theta = 1 - a``: consecutive exceedances cluster with mean
    size ``1/theta``.  It is the reference process for the theta estimators here.
    """
    if not 0.0 <= float(a) < 1.0:
        raise ValueError("a must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    z = -1.0 / np.log(rng.random(int(n) + int(burn)))
    x = np.empty(int(n) + int(burn))
    x[0] = z[0]
    ao = float(a)
    for t in range(1, x.size):
        x[t] = max(ao * x[t - 1], (1.0 - ao) * z[t])
    return x[int(burn):]


def extremal_index_runs(x: Any, *, u_quantile: float = 0.95,
                        run_length: int = 1) -> dict[str, Any]:
    """Runs (clusters) estimator of the extremal index ``theta``.

    ``theta = n_clusters / n_exceedances`` where exceedances of the empirical
    ``u_quantile`` threshold are declustered by requiring ``run_length``
    consecutive non-exceedances between clusters.  ``theta = 1`` for independent
    data, ``theta < 1`` when extremes cluster.
    """
    arr = _as_array(x)
    u = float(np.quantile(arr, u_quantile))
    exc = arr > u
    n_exc = int(exc.sum())
    if n_exc == 0:
        return {"theta": math.nan, "n_exceedances": 0, "threshold": u}
    r = max(1, int(run_length))
    clusters = 0
    gap = 0
    in_cluster = False
    for flag in exc:
        if flag:
            if not in_cluster:
                clusters += 1
                in_cluster = True
            gap = 0
        else:
            gap += 1
            if in_cluster and gap >= r:
                in_cluster = False
    theta = clusters / n_exc
    return {"theta": theta, "n_clusters": clusters, "n_exceedances": n_exc,
            "run_length": r, "threshold": u, "u_quantile": u_quantile,
            "method": "runs"}


def extremal_index_blocks(x: Any, block_size: int, *, u_quantile: float = 0.95) -> dict[str, Any]:
    """Blocks estimator of ``theta``: ``theta = log P(M_b <= u) / (b log F(u))``.

    Uses the empirical within-block exceedance probability for ``F(u)`` and the
    empirical probability that a block maximum stays below ``u``; consistent for
    ``b -> inf``, ``b/n -> 0`` with ``u`` in the tail.
    """
    arr = _as_array(x)
    b = int(block_size)
    if b < 2 or b > arr.size // 2:
        raise ValueError("block_size outside [2, n/2]")
    u = float(np.quantile(arr, u_quantile))
    p_exc = float(np.mean(arr > u))
    if not 0.0 < p_exc < 1.0 or p_exc >= 1.0:
        return {"theta": math.nan, "block_size": b, "threshold": u}
    p_block = float(np.mean(block_maxima(arr, b) <= u))
    p_block = min(max(p_block, 1e-12), 1.0 - 1e-12)
    denom = b * math.log(1.0 - p_exc)
    theta = math.log(p_block) / denom if denom else math.nan
    return {"theta": theta, "block_size": b, "threshold": u,
            "p_block_below_u": p_block, "p_exceedance": p_exc, "method": "blocks"}


def extremal_index_stability(x: Any, *, run_lengths: Sequence[int] = (1, 2, 3, 5, 8, 12, 20),
                             u_quantiles: Sequence[float] = (0.90, 0.95, 0.975),
                             plateau_band_pct: float = 15.0) -> dict[str, Any]:
    """Block-length (run-length) stability scan of ``theta`` with plateau pick.

    A trustworthy ``theta`` is the one that does not drift with the declustering
    parameter: the longest run of consecutive estimates inside
    ``plateau_band_pct`` of their median is reported as the plateau, together with
    the scan itself so a drifting estimator cannot masquerade as a plateau.
    """
    arr = _as_array(x)
    scan = []
    for q in u_quantiles:
        for r in run_lengths:
            res = extremal_index_runs(arr, u_quantile=float(q), run_length=int(r))
            scan.append({"u_quantile": float(q), "run_length": int(r),
                         "theta": res["theta"], "n_exceedances": res["n_exceedances"]})
    vals = [s["theta"] for s in scan if s["theta"] is not None and math.isfinite(s["theta"])]
    plateau: list[dict[str, Any]] = []
    if vals:
        med = float(np.median(vals))
        best: list[dict[str, Any]] = []
        cur: list[dict[str, Any]] = []
        for s in scan:
            t = s["theta"]
            ok = t is not None and math.isfinite(t) and abs(t - med) <= plateau_band_pct / 100.0 * abs(med)
            if ok:
                cur.append(s)
                if len(cur) > len(best):
                    best = list(cur)
            else:
                cur = []
        plateau = best
    theta_hat = float(np.median([s["theta"] for s in plateau])) if plateau else math.nan
    return {"theta": theta_hat, "plateau_size": len(plateau),
            "plateau_band_pct": plateau_band_pct, "scan": scan,
            "theta_spread_min_max": ([min(vals), max(vals)] if vals else None),
            "method": "runs-stability"}


def theta_recovery_study(*, n: int = 20000, a_values: Sequence[float] = (0.0, 0.2, 0.5, 0.7),
                        run_lengths: Sequence[int] = (1, 2, 3, 5, 8),
                        seed: int = 20261106) -> dict[str, Any]:
    """Recover the extremal index ``theta = 1 - a`` of the max-AR(1) process.

    ``a = 0`` is the independent case (``theta = 1``): that arm is the negative
    control that would expose a declustering estimator biased towards small
    ``theta``.
    """
    rows = []
    for a in a_values:
        x = max_ar1_series(int(n), float(a), seed=seed + int(round(a * 1000)))
        scan = [extremal_index_runs(x, u_quantile=0.95, run_length=int(r))["theta"]
                for r in run_lengths]
        stab = extremal_index_stability(x, run_lengths=tuple(run_lengths))
        rows.append({"a": float(a), "theta_true": 1.0 - float(a),
                     "theta_runs_vs_runlength": scan,
                     "theta_plateau": stab["theta"],
                     "theta_plateau_size": stab["plateau_size"],
                     "rel_error_pct": (abs(stab["theta"] - (1.0 - float(a))) / (1.0 - float(a)) * 100.0),
                     "block_estimator": extremal_index_blocks(x, max(4, int(n) // 200))["theta"],
                     "n": int(n)})
    return {"n": int(n), "run_lengths": list(run_lengths), "rows": rows}


# ---------------------------------------------------------------------------
# 4. Field quantile confidence intervals: bootstrap, replicates, shot-noise law
# ---------------------------------------------------------------------------


def _as_field(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("field must be a 1-D sample or a 2-D grid")
    if not np.all(np.isfinite(arr)):
        raise ValueError("field must be finite")
    return arr


def _block_bootstrap_indices(shape: tuple[int, int], block: tuple[int, int],
                             rng: np.random.Generator) -> np.ndarray:
    """Flat indices of one circular block-bootstrap resample of a 2-D grid."""
    nz, nx = shape
    bz, bx = int(block[0]), int(block[1])
    if bz < 1 or bx < 1:
        raise ValueError("block shape must be positive")
    nzb, nxb = int(math.ceil(nz / bz)), int(math.ceil(nx / bx))
    starts_z = rng.integers(0, nz, size=nzb)
    starts_x = rng.integers(0, nx, size=nxb)
    zz = (starts_z[:, None] + np.arange(bz)[None, :]) % nz      # (nzb, bz)
    xx = (starts_x[:, None] + np.arange(bx)[None, :]) % nx      # (nxb, bx)
    idx = zz[:, None, :, None] * nx + xx[None, :, None, :]
    return idx.reshape(nzb * bz, nxb * bx)[:nz, :nx]


def bootstrap_quantile_ci(values: Any, q: float = 0.95, *, n_boot: int = 2000,
                          level: float = 0.95, block: tuple[int, int] | None = None,
                          seed: int = 20261106) -> dict[str, Any]:
    """Bootstrap confidence interval of a field quantile.

    ``block=None`` resamples cells independently (valid only for uncorrelated
    fields); ``block=(bz, bx)`` runs a *circular block bootstrap* over the 2-D
    grid, which keeps the spatial correlation of the shot-noise field and is
    therefore the honest interval for a wear-depth patch statistic.
    """
    from scipy import stats

    arr = _as_field(values)
    flat = arr.ravel()
    est = float(np.quantile(flat, q))
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot))
    if block is None or arr.ndim == 1:
        n = flat.size
        for b in range(int(n_boot)):
            boots[b] = np.quantile(flat[rng.integers(0, n, n)], q)
        mode = "iid-cells"
    else:
        shape = arr.shape
        for b in range(int(n_boot)):
            idx = _block_bootstrap_indices(shape, (int(block[0]), int(block[1])), rng)
            boots[b] = np.quantile(arr.ravel()[idx.ravel()], q)
        mode = "circular-block(%dx%d)" % (int(block[0]), int(block[1]))
    se = float(np.std(boots, ddof=1))
    t = float(stats.t.ppf(0.5 + 0.5 * level, int(n_boot) - 1))
    lo_n, hi_n = est - t * se, est + t * se
    lo_p, hi_p = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": est, "se": se, "n_boot": int(n_boot), "mode": mode,
            "ci_normal": [float(lo_n), float(hi_n)],
            "ci_percentile": [float(lo_p), float(hi_p)],
            "ci_rel_halfwidth_pct": 0.5 * (hi_p - lo_p) / abs(est) * 100.0 if est else math.nan,
            "level": level, "q": q}


def replicate_quantile_ci(values: Any, *, level: float = 0.95) -> dict[str, Any]:
    """Confidence interval of a quantile from independent replicate estimates.

    This is the *ground-truth* sampling interval of the field statistic: each
    replicate is an independent realisation (new contact ensemble and new cell
    placement), so the scatter of the per-replicate P95 is the sampling error that
    any single report inherits.
    """
    from scipy import stats

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("need at least two replicates")
    n = arr.size
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    t = float(stats.t.ppf(0.5 + 0.5 * level, n - 1))
    rel = lambda v: v / abs(mean) * 100.0 if mean else math.nan
    return {"n_rep": int(n), "mean": mean, "sd": sd, "se": se,
            "cv_pct": rel(sd),
            "ci_normal": [mean - t * se, mean + t * se],
            "ci_percentile": [float(np.quantile(arr, (1 - level) / 2)),
                              float(np.quantile(arr, 1 - (1 - level) / 2))],
            "ci_rel_halfwidth_pct": rel(0.5 * (2 * t * se)),
            "level": level}


def shot_noise_quantile(nu: float, shape_sample: Any, q: float = 0.95, *,
                        n_mc: int = 200000, seed: int = 20261106) -> dict[str, Any]:
    """Quantile of the normalised shot-noise depth at a mean hit count ``nu``.

    In the dilute-contact limit a cell of area ``A`` receives ``k ~ Poisson(nu)``
    pits with i.i.d. volumes ``V_i`` (``nu = lambda A``) and its average depth is
    ``D = (1/A) sum V_i``.  With ``X = V / <V>`` and the area-weighted mean depth
    ``mbar = lambda <V>`` this becomes ``D / mbar = (1/nu) sum_{i<=k} X_i``, so the
    whole cell-size dependence enters through ``nu`` alone and the quantile of
    ``D`` is ``mbar * Q_q(nu)``.
    """
    if not nu > 0.0:
        raise ValueError("nu must be positive")
    x = np.asarray(shape_sample, dtype=float).ravel()
    if x.size < 10 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("shape_sample must be >= 10 positive finite volumes")
    x = x / x.mean()
    rng = np.random.default_rng(seed)
    k = rng.poisson(float(nu), int(n_mc))
    total = int(k.sum())
    draws = x[rng.integers(0, x.size, total)]
    csum = np.concatenate(([0.0], np.cumsum(draws)))
    ends = np.cumsum(k)
    starts = ends - k
    norm = (csum[ends] - csum[starts]) / float(nu)
    return {"nu": float(nu), "q": q,
            "normalised_quantile": float(np.quantile(norm, q)),
            "normalised_mean": float(norm.mean()),
            "zero_probability": math.exp(-float(nu)),
            "n_mc": int(n_mc)}


def shot_noise_resolution_law(cell_sizes: Any, nu_by_level: Any, mean_depth_by_level: Any,
                              shape_sample: Any, *, q: float = 0.95,
                              n_mc: int = 200000, seed: int = 20261106) -> dict[str, Any]:
    """Analytic P95 cell-size law ``P95(A) = mbar(A) * Q_q(nu(A))``.

    Returns the predicted P95 at every resolution together with the ratios to the
    finest level and the local scaling exponent ``d ln P95 / d ln h``; those are
    the numbers a grid-convergence verdict has to be confronted with, because a
    Roache index assumes a continuum limit that this statistic does not have.
    """
    cells = [float(c) for c in cell_sizes]
    nus = [float(v) for v in nu_by_level]
    means = [float(v) for v in mean_depth_by_level]
    if not (len(cells) == len(nus) == len(means)):
        raise ValueError("cell_sizes, nu_by_level and mean_depth_by_level must match")
    rows = []
    for h, nu, mbar in zip(cells, nus, means):
        qn = shot_noise_quantile(nu, shape_sample, q, n_mc=n_mc, seed=seed)
        rows.append({"cell_m": h, "nu": nu, "mean_depth_m": mbar,
                     "normalised_quantile": qn["normalised_quantile"],
                     "predicted_p95_m": mbar * qn["normalised_quantile"]})
    ref = rows[0]
    for r in rows:
        r["ratio_to_finest"] = r["predicted_p95_m"] / ref["predicted_p95_m"]
        r["log_slope_vs_cell"] = (math.log(r["predicted_p95_m"] / ref["predicted_p95_m"])
                                  / math.log(r["cell_m"] / ref["cell_m"])
                                  if r["cell_m"] != ref["cell_m"] else math.nan)
    return {"q": q, "rows": rows,
            "note": ("cell-averaged shot-noise depth has no h -> 0 limit: the "
                     "quantile is measure dependent by construction")}


def adjudicate_gci_vs_ci(*, gci_pct: float, paired_diff_ci_pct: Sequence[float],
                         sampling_halfwidth_pct: float,
                         model_prediction_pct: float | None = None,
                         model_tolerance_pct: float = 15.0) -> dict[str, Any]:
    """Decide whether a resolution difference is noise, measure dependence or error.

    ``gci_pct`` is the reported grid-convergence index (relative difference between
    the fine and coarse level), ``paired_diff_ci_pct`` the confidence interval of
    the *paired* (same-seed) resolution difference, ``sampling_halfwidth_pct`` the
    half-width of a single-level P95 interval and ``model_prediction_pct`` the
    resolution difference predicted by the shot-noise law.  Verdicts:

    * ``sampling-noise``            -- the paired difference interval contains 0;
    * ``measure-dependence``        -- the difference is significant and the
      shot-noise law reproduces it within ``model_tolerance_pct`` (then it is a
      property of the statistic, not a numerical error);
    * ``unexplained``               -- significant, and not explained by the law:
      the conservative reading is a discretisation error to be reduced.
    """
    lo, hi = (float(paired_diff_ci_pct[0]), float(paired_diff_ci_pct[1]))
    significant = (lo > 0.0) or (hi < 0.0)
    explained = (model_prediction_pct is not None
                 and abs(gci_pct - float(model_prediction_pct)) <= model_tolerance_pct)
    if not significant:
        verdict = "sampling-noise"
    elif explained:
        verdict = "measure-dependence"
    else:
        verdict = "unexplained"
    return {"verdict": verdict, "gci_pct": float(gci_pct),
            "paired_diff_ci_pct": [lo, hi], "paired_diff_significant": bool(significant),
            "sampling_halfwidth_pct": float(sampling_halfwidth_pct),
            "gci_over_sampling_halfwidth": (abs(float(gci_pct)) / float(sampling_halfwidth_pct)
                                            if sampling_halfwidth_pct else math.nan),
            "model_prediction_pct": (None if model_prediction_pct is None
                                     else float(model_prediction_pct)),
            "model_misfit_pct": (None if model_prediction_pct is None
                                 else abs(float(gci_pct) - float(model_prediction_pct))),
            "model_tolerance_pct": float(model_tolerance_pct),
            "model_explains_difference": bool(explained),
            "reading": ("resolution difference is inside the paired sampling interval"
                        if not significant else
                        ("resolution difference is significant and fully accounted for "
                         "by the shot-noise quantile law (measure dependence, not a "
                         "numerical error)" if explained else
                         "resolution difference is significant and NOT explained by the "
                         "shot-noise law: treat as discretisation error"))}


# ---------------------------------------------------------------------------
# 5. Certification of a wear-field P95 and the acceptance protocol
# ---------------------------------------------------------------------------


def wear_field_p95_certificate(records: Any, fields: Mapping[Any, Any], *,
                               q: float = 0.95, level: float = 0.95,
                               block_shape: tuple[int, int] = (1, 4),
                               n_boot: int = 2000, shape_sample: Any | None = None,
                               gci_pct: float | None = None) -> dict[str, Any]:
    """Certify a field P95: sampling interval, resolution law and GCI adjudication.

    ``records`` is a sequence of per-run dicts with keys ``cell_m``, ``rep``,
    ``p95_m``, ``mean_m``, ``n_cells``, ``n_nonzero`` (the shape produced by
    ``scripts/extreme_stats_certification.py``); ``fields`` maps
    ``(cell_m, rep)`` to the 2-D depth field.  The certificate is a pure function
    of those inputs, so it can be unit-tested on synthetic shot-noise fields.
    """
    recs = list(records)
    if not recs:
        raise ValueError("records must not be empty")
    levels = sorted({float(r["cell_m"]) for r in recs})
    per_level: dict[str, Any] = {}
    p95_by_level: dict[float, list[float]] = {}
    for h in levels:
        rows = [r for r in recs if float(r["cell_m"]) == h]
        p95 = [float(r["p95_m"]) for r in rows]
        p95_by_level[h] = p95
        rep_ci = replicate_quantile_ci(p95, level=level)
        zeros = [1.0 - float(r["n_nonzero"]) / float(r["n_cells"]) for r in rows]
        nu = -math.log(max(float(np.mean(zeros)), 1e-12)) if float(np.mean(zeros)) > 0 else math.nan
        boots: dict[str, Any] = {}
        first = min(rows, key=lambda r: int(r["rep"]))
        key = (h, int(first["rep"]))
        if key in fields:
            field = np.asarray(fields[key], dtype=float)
            boots["iid"] = bootstrap_quantile_ci(field.ravel(), q, n_boot=n_boot,
                                                 level=level, block=None)
            if field.ndim == 2:
                boots["block"] = bootstrap_quantile_ci(field, q, n_boot=n_boot,
                                                       level=level, block=block_shape)
        gumbel = gumbel_profile_ci(p95, p=q, level=level)
        per_level["%.4g" % h] = {
            "cell_m": h, "n_rep": len(rows),
            "p95_mean_m": rep_ci["mean"], "p95_sd_m": rep_ci["sd"],
            "p95_cv_pct": rep_ci["cv_pct"],
            "p95_ci_normal_m": rep_ci["ci_normal"],
            "p95_ci_percentile_m": rep_ci["ci_percentile"],
            "p95_ci_rel_halfwidth_pct": rep_ci["ci_rel_halfwidth_pct"],
            "mean_depth_m": float(np.mean([float(r["mean_m"]) for r in rows])),
            "zero_cell_fraction": float(np.mean(zeros)),
            "nu_from_zero_cells": nu,
            "n_cells": int(rows[0]["n_cells"]),
            "bootstrap": boots,
            "gumbel_profile_on_replicates": {
                "estimate_m": gumbel["estimate"],
                "ci_lo_m": gumbel["ci_lo"], "ci_hi_m": gumbel["ci_hi"],
                "ci_rel_halfwidth_pct": gumbel.get("ci_rel_halfwidth_pct")},
        }
    # paired resolution differences (same seed pair -> common random numbers)
    reps = sorted({int(r["rep"]) for r in recs})
    paired: dict[str, Any] = {}
    fine, coarse = levels[0], levels[-1]
    diffs = []
    for r in reps:
        vals = {float(row["cell_m"]): float(row["p95_m"]) for row in recs
                if int(row["rep"]) == r}
        if fine in vals and coarse in vals:
            diffs.append((vals[coarse] - vals[fine]) / vals[fine] * 100.0)
    if diffs:
        paired_str = "%.4g_vs_%.4g" % (coarse, fine)
        paired[paired_str] = {
            "mean_diff_pct": float(np.mean(diffs)),
            "sd_diff_pct": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else math.nan,
            "ci_normal_pct": list(replicate_quantile_ci(diffs, level=level)["ci_normal"]),
            "n_pairs": len(diffs),
        }
    # cell-average scaling law q(h) = q_inf - B h**beta on the replicate means
    law = resolution_scaling_law({h: p95_by_level[h] for h in levels})
    # the empty-cell fraction fixes the mean hit count per cell only where empty
    # cells exist at all; coarser levels are scaled by the cell count (nu ~ 1/A)
    nu_finite = [(h, per_level["%.4g" % h]["nu_from_zero_cells"],
                  per_level["%.4g" % h]["n_cells"]) for h in levels
                 if math.isfinite(per_level["%.4g" % h]["nu_from_zero_cells"])]
    nu_source = "empty-cell fraction"
    nus: list[float] = []
    if nu_finite:
        h_ref, nu_ref, n_ref = min(nu_finite, key=lambda r: r[0])
        for h in levels:
            v = per_level["%.4g" % h]["nu_from_zero_cells"]
            if math.isfinite(v) and v > 0.0:
                nus.append(v)
            else:
                nus.append(nu_ref * n_ref / per_level["%.4g" % h]["n_cells"])
                nu_source = "empty-cell fraction (finest level) scaled by cell count"
    shot = None
    if shape_sample is not None and nus:
        shot = shot_noise_resolution_law(
            levels, nus,
            [per_level["%.4g" % h]["mean_depth_m"] for h in levels],
            shape_sample, q=q)
        shot["nu_source"] = nu_source
        for row in shot["rows"]:
            meas = per_level["%.4g" % row["cell_m"]]["p95_mean_m"]
            row["measured_p95_m"] = meas
            row["model_rel_deviation_pct"] = ((row["predicted_p95_m"] - meas) / meas * 100.0
                                              if meas else math.nan)
    return {"q": q, "level": level, "levels_m": levels,
            "per_level": per_level, "paired_differences_pct": paired,
            "gci_input_pct": gci_pct, "resolution_law": law,
            "shot_noise_law": shot}


def acceptance_protocol_evaluate(certificate: Mapping[str, Any], *,
                                 coverage: Mapping[str, Any] | None = None,
                                 theta: Mapping[str, Any] | None = None,
                                 criteria: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Machine-checkable evaluation of the extreme-statistic acceptance protocol.

    Criteria (defaults in ``criteria``):

    * ``ci_agreement_pct`` (3.0)  -- replicate, bootstrap and Gumbel-profile
      centres of P95 must agree within 3 % (the platform error standard);
    * ``coverage_min`` (0.90)     -- Monte Carlo coverage of the profile interval
      at nominal 95 % must be at least 0.90;
    * ``ci_agreement_pct`` also gates the finest-pair resolution difference: the
      statistic either has to be converged to within 3 % or its extrapolated
      continuum value (with the smoothing exponent) has to be declared;
    * ``theta_plateau_min`` (3)   -- the extremal index needs a stability plateau;
    * ``ci_quoted`` (1)           -- every P95 must carry an interval.
    """
    crit = {"ci_agreement_pct": 3.0, "coverage_min": 0.90,
            "gci_model_tolerance_pct": 15.0, "theta_plateau_min": 3.0,
            "ci_quoted": 1.0}
    if criteria:
        crit.update({k: float(v) for k, v in criteria.items()})
    checks: dict[str, Any] = {}

    # 1. three-way agreement of the P95 centre at the finest level
    fine_key = "%.4g" % certificate["levels_m"][0]
    lvl = certificate["per_level"][fine_key]
    centres = {"replicate": lvl["p95_mean_m"],
               "gumbel_profile": lvl["gumbel_profile_on_replicates"]["estimate_m"]}
    if "block" in lvl.get("bootstrap", {}):
        centres["block_bootstrap"] = lvl["bootstrap"]["block"]["estimate"]
    if "iid" in lvl.get("bootstrap", {}):
        centres["iid_bootstrap"] = lvl["bootstrap"]["iid"]["estimate"]
    base = centres["replicate"]
    spread = max(abs(v - base) for v in centres.values()) / abs(base) * 100.0
    checks["P1_three_way_ci_agreement"] = {
        "value_pct": spread, "criterion_pct": crit["ci_agreement_pct"],
        "pass": bool(spread <= crit["ci_agreement_pct"]), "centres_m": centres}

    # 2. profile interval coverage against Monte Carlo
    if coverage:
        prof = coverage["methods"].get("profile", coverage["methods"].get("gumbel"))
        cov = float(prof["coverage"]) if prof else math.nan
        checks["P2_profile_coverage"] = {
            "coverage": cov, "nominal": float(coverage["level"]),
            "criterion_min": crit["coverage_min"], "n_rep": int(coverage["n_rep"]),
            "pass": bool(cov >= crit["coverage_min"])}
    else:
        checks["P2_profile_coverage"] = {"pass": False, "note": "no coverage study supplied"}

    # 3. every P95 carries an interval
    missing = [k for k, v in certificate["per_level"].items()
               if not v.get("p95_ci_normal_m") or not np.all(np.isfinite(v["p95_ci_normal_m"]))]
    checks["P3_ci_quoted_everywhere"] = {"missing_levels": missing,
                                         "pass": not missing}

    # 4. the resolution status of the statistic must be declared, not hidden
    law = certificate.get("resolution_law") or {}
    levels = certificate["levels_m"]
    q_fine = certificate["per_level"]["%.4g" % levels[0]]["p95_mean_m"]
    q_next = certificate["per_level"]["%.4g" % levels[1]]["p95_mean_m"]
    delta = abs(q_fine - q_next) / q_next * 100.0
    converged = delta <= crit["ci_agreement_pct"]
    declared = (math.isfinite(law.get("continuum_value_m", math.nan))
                and 0.0 < law.get("beta", -1.0) < 3.0)
    scale_law = (math.isfinite(law.get("power_law_exponent", math.nan))
                 and all(math.isfinite(v) for v in (law.get("power_law_ci") or [math.nan])))
    checks["P4_resolution_status_declared"] = {
        "finest_pair_delta_pct": delta,
        "convergence_criterion_pct": crit["ci_agreement_pct"],
        "converged": bool(converged),
        "continuum_declared": bool(declared),
        "continuum_identifiable": bool(law.get("continuum_identifiable", False)),
        "continuum_value_m": law.get("continuum_value_m"),
        "finest_shortfall_pct": law.get("finest_shortfall_pct"),
        "beta": law.get("beta"),
        "power_law_exponent": law.get("power_law_exponent"),
        "power_law_ci": law.get("power_law_ci"),
        "scale_law_declared": bool(scale_law),
        "pass": bool(converged or declared or scale_law)}

    # 5. extremal index plateau
    if theta:
        size = int(theta.get("plateau_size", 0))
        checks["P5_theta_plateau"] = {
            "theta": theta.get("theta"), "plateau_size": size,
            "criterion_min": crit["theta_plateau_min"],
            "pass": bool(size >= crit["theta_plateau_min"])}
    else:
        checks["P5_theta_plateau"] = {"pass": False, "note": "no theta study supplied"}

    return {"criteria": crit, "checks": checks,
            "all_pass": all(bool(c.get("pass")) for c in checks.values())}


# ---------------------------------------------------------------------------
# 6. Resolution scaling law of a cell-averaged quantile
# ---------------------------------------------------------------------------


def local_scaling_exponent(cell_sizes: Sequence[float], values: Sequence[float]) -> list[dict[str, Any]]:
    """Local exponent ``beta = -d ln q / d ln h`` between adjacent resolutions.

    For a cell-averaged field statistic a *constant* exponent means the statistic
    obeys a power law in the cell size (no continuum limit), while exponents that
    decay towards zero indicate a transient smoothing that does converge.  The
    distinction is exactly what a Roache grid-convergence index cannot make.
    """
    cells = [float(c) for c in cell_sizes]
    vals = [float(v) for v in values]
    if len(cells) != len(vals) or len(cells) < 2:
        raise ValueError("need matching cell sizes and values, at least two")
    order = sorted(range(len(cells)), key=lambda i: cells[i], reverse=True)
    rows = []
    for a, b in zip(order[:-1], order[1:]):
        rows.append({"h_coarse_m": cells[a], "h_fine_m": cells[b],
                     "q_coarse": vals[a], "q_fine": vals[b],
                     "ratio_fine_over_coarse": vals[b] / vals[a],
                     "beta": (math.log(vals[b] / vals[a])
                              / math.log(cells[a] / cells[b])),
                     "refinement_ratio": cells[a] / cells[b]})
    return rows


def resolution_scaling_law(values_by_cell: Mapping[float, Any], *,
                           n_boot: int = 400, level: float = 0.95,
                           seed: int = 20261106,
                           weight_by_inverse_variance: bool = True,
                           continuum_bound_factor: float = 1.6) -> dict[str, Any]:
    """Fit the cell-size law of a cell-averaged quantile, two ways.

    ``values_by_cell`` maps the cell size to the replicate values produced at that
    resolution.  Two models are reported because they answer different questions:

    * ``q(h) = q_inf - B h**beta`` (saturating): ``q_inf`` is the extrapolated
      *continuum* (point-wise) value, bounded by construction to
      ``[q(h_min), continuum_bound_factor * q(h_min)]`` -- without that bound the
      model has a degenerate ``beta -> 0`` ridge in which ``B`` grows without limit
      and ``q_inf`` becomes arbitrary.  ``beta`` is the smoothing exponent and the
      pair ``(q_inf, beta)`` is what a grid-convergence index cannot report: how far
      the finest grid still is from the limit.
    * ``q(h) = C h**(-beta_power)`` (pure power law): the *measure-dependence*
      fallback.  A constant local exponent
      (:func:`local_scaling_exponent`) over the available levels means the
      continuum value is **not identifiable** from the data range and the statistic
      must be quoted as a scale-of-record quantity instead of an extrapolation.

    Sign convention: a cell-averaged quantile of a peaked field *rises* as the cell
    shrinks (less peak is averaged away), so the finite limit is approached from
    below.
    """
    from scipy.optimize import least_squares

    cells = np.array(sorted(float(c) for c in values_by_cell), dtype=float)
    means = np.array([float(np.mean(np.asarray(values_by_cell[c], dtype=float)))
                      for c in cells], dtype=float)
    reps = [np.asarray(values_by_cell[float(c)], dtype=float) for c in cells]
    sds = np.array([float(r.std(ddof=1)) if r.size > 1 else 0.0 for r in reps])
    if weight_by_inverse_variance and np.all(sds > 0):
        sigma = sds
    else:
        sigma = np.full_like(means, max(float(means.mean()) * 1e-3, 1e-30))
    q_top = float(means.max())
    scale = float(means.mean())
    q_hi = q_top * float(continuum_bound_factor)

    def fit_offset(h: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
        def resid(p: np.ndarray) -> np.ndarray:
            return (p[0] - p[1] * np.power(h, p[2]) - q) / s

        p0 = np.array([q_top * 1.02 + 0.02 * (q_hi - q_top),
                       max((q_top - float(q.min())) * 2.0, scale * 1e-3), 0.3])
        res = least_squares(resid, p0,
                            bounds=([q_top, 0.0, 0.02], [q_hi, np.inf, 3.0]),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        return res.x

    def fit_power(h: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        slope, intercept = np.polyfit(np.log(h), np.log(q), 1)
        return float(-slope), float(math.exp(intercept))

    q_inf, amp, beta = (float(v) for v in fit_offset(cells, means, sigma))
    beta_power, c_power = fit_power(cells, means)
    rng = np.random.default_rng(seed)
    boot_q, boot_b, boot_p = [], [], []
    for _ in range(int(n_boot)):
        q_b = np.array([float(np.mean(rng.choice(r, size=r.size, replace=True)))
                        for r in reps])
        try:
            a_b, _b, beta_b = (float(v) for v in fit_offset(cells, q_b, sigma))
            boot_q.append(a_b)
            boot_b.append(beta_b)
            boot_p.append(fit_power(cells, q_b)[0])
        except Exception:                     # pragma: no cover - fit did not move
            continue

    def ci(vals: list[float]) -> list[float]:
        if len(vals) < 10:
            return [math.nan, math.nan]
        return [float(np.quantile(vals, (1 - level) / 2)),
                float(np.quantile(vals, 1 - (1 - level) / 2))]

    q_finest = float(means[0])
    local = local_scaling_exponent(cells.tolist(), means.tolist())
    # identifiability: a saturating fit is only trusted when the local exponent
    # actually decays over the available range (otherwise only the scale law is).
    decaying = all(local[i]["beta"] >= local[i + 1]["beta"] - 0.02
                   for i in range(len(local) - 1))
    at_bound = abs(q_inf - q_hi) <= 1e-9 * max(q_hi, 1.0)
    return {"cell_sizes_m": cells.tolist(),
            "q_mean_by_cell": means.tolist(), "q_sd_by_cell": sds.tolist(),
            "q_finest_m": q_finest,
            "continuum_value_m": q_inf, "continuum_ci_m": ci(boot_q),
            "continuum_bound_m": [q_top, q_hi],
            "continuum_at_bound": bool(at_bound),
            "smoothing_amplitude_B": amp, "beta": beta, "beta_ci": ci(boot_b),
            "power_law_exponent": beta_power, "power_law_ci": ci(boot_p),
            "power_law_coefficient": c_power,
            "continuum_identifiable": bool(decaying and not at_bound),
            "n_boot": int(n_boot),
            "continuum_over_finest": q_inf / q_finest if q_finest else math.nan,
            "finest_shortfall_pct": ((q_inf - q_finest) / q_finest * 100.0
                                     if q_finest else math.nan),
            "residual_rel_pct": [float(r) for r in
                                 ((q_inf - amp * cells ** beta - means) / means * 100.0)],
            "local_exponents": local}


def p95_resolution_verdict(certificate: Mapping[str, Any], *,
                           sampling_tolerance_pct: float = 3.0) -> dict[str, Any]:
    """Adjudicate resolution sensitivity against sampling error for one statistic.

    Reads a :func:`wear_field_p95_certificate` result and answers the W10-N06
    question directly: is the finest-pair resolution difference inside the
    sampling interval (then it is noise), or is it a systematic measure effect
    that must either be extrapolated or declared as the scale of record?
    """
    levels = certificate["levels_m"]
    per = certificate["per_level"]
    keys = ["%.4g" % h for h in levels]
    finest, next_finest = keys[0], keys[1]
    q_fine = per[finest]["p95_mean_m"]
    q_next = per[next_finest]["p95_mean_m"]
    delta = (q_fine - q_next) / q_next * 100.0
    half = per[finest]["p95_ci_rel_halfwidth_pct"]
    law = certificate.get("resolution_law")
    out = {"finest_cell_m": levels[0], "next_cell_m": levels[1],
           "delta_finest_pair_pct": delta,
           "sampling_halfwidth_pct": half,
           "delta_over_sampling_halfwidth": (abs(delta) / half if half else math.nan),
           "significant": bool(abs(delta) > half),
           "within_sampling_tolerance": bool(abs(delta) <= sampling_tolerance_pct)}
    if law:
        out["continuum_value_m"] = law["continuum_value_m"]
        out["continuum_ci_m"] = law["continuum_ci_m"]
        out["beta"] = law["beta"]
        out["beta_ci"] = law["beta_ci"]
        out["continuum_identifiable"] = law["continuum_identifiable"]
        out["power_law_exponent"] = law["power_law_exponent"]
        out["power_law_ci"] = law["power_law_ci"]
        out["finest_shortfall_pct"] = law["finest_shortfall_pct"]
        converged = abs(delta) <= sampling_tolerance_pct
        out["converged_at_3pct"] = bool(converged)
        if converged:
            reading = ("resolution difference at the finest pair is inside the %g %% "
                       "sampling tolerance: quote P95 at the finest grid with its "
                       "sampling interval" % sampling_tolerance_pct)
        elif law["continuum_identifiable"]:
            reading = ("resolution difference (%.2f %%) is %.1fx the sampling "
                       "half-width (%.2f %%) and the cell-average law extrapolates to "
                       "%.4g m (beta = %.2f): the statistic is NOT converged at %.3g m "
                       "and must be quoted as a cell-size-of-record quantity together "
                       "with the scaling exponent"
                       % (abs(delta), abs(delta) / half if half else float("nan"), half,
                          law["continuum_value_m"], law["beta"], levels[0]))
        else:
            reading = ("resolution difference (%.2f %%) is %.1fx the sampling "
                       "half-width (%.2f %%). The cell-average quantile follows a "
                       "power law with exponent %.3f over the whole tested range "
                       "(continuum value not identifiable there), so P95 has no "
                       "certified continuum value at %.3g m: report it as a "
                       "cell-size-of-record quantity with the power-law exponent "
                       "(+/- %.1f %% per halving of the cell size)"
                       % (abs(delta), abs(delta) / half if half else float("nan"), half,
                          law["power_law_exponent"], levels[0],
                          (math.pow(0.5, -law["power_law_exponent"]) - 1.0) * 100.0))
        out["reading"] = reading
    return out


# ---------------------------------------------------------------------------
# 7. Why the quantile moves: cell averaging of the contact-activity profile
# ---------------------------------------------------------------------------


def waterline_activity_p95(activity: Any, hull: Any, *, dx_m: float, dz_m: float,
                           below_m: float = 1.2, above_m: float = 1.0,
                           n_sub: int = 64, q: float = 0.95) -> dict[str, Any]:
    """P95 of the *expected* depth field of the waterline band, cell averaged.

    The contact activity of :class:`tensordem.ice_abrasion.ActivityDistribution`
    is separable, ``p(x, z) = w_x(x) w_z(z)`` (a forebody/shoulder weight along the
    hull times a Gaussian across the waterline), so the expected depth of a cell is
    the product of the two *cell averages* and the cell-averaged field needs no
    simulation at all.  Comparing this prediction with the DEM fields separates the
    two candidate causes of a resolution-dependent P95: cell averaging of a smooth
    deterministic activity profile (this function, reproducible with zero
    parameters) versus shot noise of the discrete pit placement (small here, see
    :func:`shot_noise_quantile`).

    ``activity`` is duck-typed: it must expose the ``x_*``/``z``/``waterline_*`` and
    ``sigma_*`` attributes of ``ActivityDistribution``; ``hull`` needs
    ``length_m`` and ``draught_m``.
    """
    length = float(hull.length_m)
    draught = float(hull.draught_m)
    band_lo, band_hi = draught - float(below_m), draught + float(above_m)
    n_x = max(4, int(round(length / float(dx_m))))
    n_z = max(2, int(round((band_hi - band_lo) / float(dz_m))))
    x_edges = np.linspace(0.0, length, n_x + 1)
    z_edges = np.linspace(band_lo, band_hi, n_z + 1)
    sub = np.linspace(-0.5, 0.5, int(n_sub))
    off = (sub[1] - sub[0]) if n_sub > 1 else 1.0

    def wx(u: np.ndarray) -> np.ndarray:
        return (float(activity.x_background)
                + (1.0 - float(activity.x_background))
                * np.exp(-u / float(activity.x_decay_fraction))
                + float(activity.x_shoulder_boost)
                * np.exp(-0.5 * ((u - float(activity.x_shoulder_center))
                                 / float(activity.x_shoulder_width)) ** 2))

    def wz(z: np.ndarray) -> np.ndarray:
        zc = draught + float(activity.waterline_offset_m)
        sigma = np.where(z < draught, float(activity.sigma_below_m),
                         float(activity.sigma_above_m))
        return np.exp(-0.5 * ((z - zc) / sigma) ** 2)

    # cell averages by sub-sampling the cell (uniform sub-grid, offsets in (-0.5, 0.5))
    x_c = 0.5 * (x_edges[1:] + x_edges[:-1])
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])
    dxs = (x_edges[1:] - x_edges[:-1])
    dzs = (z_edges[1:] - z_edges[:-1])
    wx_cell = np.array([float(np.mean(wx((x + sub * dx) / length))) for x, dx in zip(x_c, dxs)])
    wz_cell = np.array([float(np.mean(wz(z + sub * dz))) for z, dz in zip(z_c, dzs)])
    field = np.outer(wz_cell, wx_cell)
    area = float(np.mean(np.outer(dzs, dxs)))      # areas are uniform by construction
    mean = float((field * area).sum() / (area * field.size))
    field = field / mean                            # unit-mean activity profile
    return {"dx_m": float(dx_m), "dz_m": float(dz_m), "n_x": n_x, "n_z": n_z,
            "dz_actual_m": float(dzs.mean()), "dx_actual_m": float(dxs.mean()),
            "p95_relative": float(np.quantile(field, q)),
            "mean_relative": float(field.mean()), "n_cells": int(field.size),
            "q": q, "cell_uniform_area": bool(np.ptp(np.outer(dzs, dxs)) <= 1e-12 * area),
            "field": field, "x_centers": x_c, "z_centers": z_c}


def shot_rate_from_zero_fraction(profile: Any, zero_fraction: float) -> float:
    """Mean Poisson hit count per cell implied by the empty-cell fraction.

    Cells receive ``K ~ Poisson(c * a)`` pits, ``a`` the normalised activity
    (unit mean), so the empty fraction is ``mean(exp(-c a))`` -- not
    ``exp(-c)``, because the activity is far from uniform.  Solving this
    monotone relation fixes the *only* free parameter of the two-mechanism
    resolution model (:func:`band_field_p95`) from the field itself.
    """
    from scipy.optimize import brentq

    a = np.asarray(profile, dtype=float).ravel()
    a = a / a.mean()
    target = float(zero_fraction)
    if not 0.0 < target < 1.0:
        raise ValueError("zero_fraction must lie in (0, 1)")

    def f(c: float) -> float:
        return float(np.mean(np.exp(-c * a))) - target

    return float(brentq(f, 1e-8, 1e5, xtol=1e-12))


def band_field_p95(profile: Any, mean_depth_m: float, hit_rate: float,
                   shape_sample: Any, *, q: float = 0.95, n_realisations: int = 200,
                   level: float = 0.95, seed: int = 20261106) -> dict[str, Any]:
    """Two-mechanism model of a cell-averaged CTMC/spot wear field.

    Cell depth is ``d_ij = (mbar / c) * sum_{k <= K_ij} X_k`` with
    ``K_ij ~ Poisson(c * a_ij)``, ``X = V/<V>`` the normalised pit-volume shape and
    ``a`` the normalised contact-activity profile.  It reproduces both mechanisms
    that move a cell-averaged quantile: the **deterministic smoothing** of ``a`` by
    the cell average (given by :func:`waterline_activity_p95`) and the **shot noise**
    of a finite number of pits per cell.  Averaging over realisations also returns
    the model's own replicate scatter, which is directly comparable with the DEM
    replicate CV.
    """
    a = np.asarray(profile, dtype=float)
    mean_profile = float(a.mean())
    a = a / mean_profile
    x = np.asarray(shape_sample, dtype=float).ravel()
    if x.size < 10 or np.any(x <= 0.0):
        raise ValueError("shape_sample must be >= 10 positive values")
    x = x / x.mean()
    if not hit_rate > 0.0:
        raise ValueError("hit_rate must be positive")
    rng = np.random.default_rng(seed)
    lam = (float(hit_rate) * a).ravel()
    p95 = np.empty(int(n_realisations))
    zeros = np.empty(int(n_realisations))
    for r in range(int(n_realisations)):
        k = rng.poisson(lam)
        total = int(k.sum())
        draws = x[rng.integers(0, x.size, total)]
        csum = np.concatenate(([0.0], np.cumsum(draws)))
        ends = np.cumsum(k)
        starts = ends - k
        depth = (csum[ends] - csum[starts]) * (float(mean_depth_m) / float(hit_rate))
        p95[r] = float(np.quantile(depth, q))
        zeros[r] = float(np.mean(k == 0))
    rep = replicate_quantile_ci(p95, level=level)
    return {"q": q, "hit_rate": float(hit_rate), "n_realisations": int(n_realisations),
            "p95_mean_m": rep["mean"], "p95_sd_m": rep["sd"],
            "p95_cv_pct": rep["cv_pct"],
            "p95_ci_normal_m": rep["ci_normal"],
            "zero_fraction_model": float(zeros.mean()),
            "implied_mean_depth_m": float(mean_depth_m),
            "n_cells": int(a.size)}
