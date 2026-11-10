"""Breaking-force ensemble statistics and the width (size) effect.

Why this module exists
----------------------
Criterion (c) of campaign node ``P295W1-N07`` is a statement about an *ensemble*:
with ``N = 50`` realisations of the random strength field, the mean breaking
force ``mu_F`` must be converged to 5 %, the coefficient of variation
``sigma_F / mu_F`` must land in ``[0.1, 0.3]``, the 5 % quantile must sit below
the mean, and the normalised 5 % quantile ``F_5% / B`` must decrease monotonically
as the contact-line width grows from 10 m to 100 m.  All four are properties of
the *estimators*, not only of the model, so this module implements the
estimators explicitly and reports their uncertainty:

* :func:`ensemble_statistics` -- mean/std/CV/quantiles plus the relative standard
  error of the mean, which is the honest statement of "how converged is
  ``mu_F`` at this ``N``".
* :func:`running_mean_convergence` -- the path of the running mean, and the worst
  relative drift between the second half of the ensemble and the full ensemble.
  A converged mean has *both* a small standard error and a flat running mean;
  quoting only one of them would hide a trend.
* :func:`bootstrap_quantile_ci` -- a percentile bootstrap interval for ``F_5%``,
  because the 5 % quantile of 50 samples is the 2.5-th order statistic and its
  sampling error is the dominant uncertainty in the width sweep.
* :func:`size_effect_sweep` -- the sweep itself, run with **nested
  realisations**: one exact field decomposition on the widest domain, sliced to
  every smaller width (see :mod:`tensordem.random_ice_strength.field`).  Nested
  realisations are positively correlated across widths, which is what makes the
  small differences between adjacent widths resolvable at N = 50; the
  correlation of the field itself is *exactly* that of the smaller domain, so
  this is variance reduction, not a change of model.

Fit of the width trend
----------------------
:func:`fit_power_law` fits ``p(B) = p_ref (B / B_ref) ** -n`` to the normalised
quantile curve.  ``n`` is the model's own pressure-area exponent, and the report
compares it with the ISO 19906 global-pressure band rather than asserting it --
it is a *consequence* of the strength scatter here, not an input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .beam import IceSheetStrengthModel
from .field import KlField1D, correlated_weibull_field, gaussian_field_1d
from .distributions import WeibullMarginal

__all__ = [
    "ensemble_statistics",
    "running_mean_convergence",
    "bootstrap_quantile_ci",
    "breaking_force_ensemble",
    "size_effect_sweep",
    "fit_power_law",
    "monotone_decreasing",
]


def _as_positive(values, name: str = "values") -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError(f"{name} needs at least two finite entries")
    return x


def ensemble_statistics(forces: Sequence[float], *, quantiles: Sequence[float] = (0.05, 0.5, 0.95)
                        ) -> Dict[str, float]:
    """Mean/std/CV/quantiles of a force ensemble and the standard error of the mean."""
    f = _as_positive(forces, "forces")
    mu = float(f.mean())
    sd = float(f.std(ddof=1))
    out: Dict[str, float] = {
        "n": float(f.size),
        "mean": mu,
        "std": sd,
        "cov": sd / mu if mu > 0.0 else float("nan"),
        "se": sd / math.sqrt(f.size),
        "se_relative": (sd / math.sqrt(f.size)) / mu if mu > 0.0 else float("nan"),
        "min": float(f.min()),
        "max": float(f.max()),
        "skewness": float(np.mean((f - mu) ** 3) / sd ** 3) if sd > 0.0 else float("nan"),
    }
    for q in quantiles:
        tag = f"q{int(round(100 * q)):02d}"
        out[tag] = float(np.quantile(f, q))
        if mu > 0.0:
            out[f"{tag}_over_mean"] = float(np.quantile(f, q)) / mu
    return out


def running_mean_convergence(forces: Sequence[float], *, tail_fraction: float = 0.5
                             ) -> Dict[str, float]:
    """Running-mean path and its worst relative drift over the tail of the ensemble.

    ``max_relative_drift`` is ``max_{k >= tail_fraction*n} |mu_k - mu_n| / mu_n``:
    the largest excursion of the running mean inside the last ``1 - tail_fraction``
    of the ensemble.  A mean that is still drifting is not converged no matter
    how small its final standard error is.
    """
    f = _as_positive(forces, "forces")
    n = f.size
    if not (0.0 < tail_fraction < 1.0):
        raise ValueError("tail_fraction must be in (0, 1)")
    running = np.cumsum(f) / np.arange(1, n + 1)
    final = float(running[-1])
    start = max(int(math.ceil(tail_fraction * n)), 1) - 1
    tail = running[start:]
    drift = float(np.max(np.abs(tail - final))) / final if final > 0.0 else float("nan")
    # also report the drift of the last estimate before doubling the ensemble when
    # n is large enough for a half-vs-full comparison
    half = n // 2
    half_vs_full = (abs(float(running[half - 1]) - final) / final) if half >= 2 else float("nan")
    return {
        "n": float(n),
        "final_mean": final,
        "max_relative_drift": drift,
        "half_vs_full_relative": half_vs_full,
        "running_mean": [float(v) for v in running],
    }


def bootstrap_quantile_ci(forces: Sequence[float], q: float = 0.05, *,
                          n_bootstrap: int = 4000, confidence: float = 0.95,
                          seed: int = 20260101) -> Dict[str, float]:
    """Percentile-bootstrap interval for the ``q`` quantile of the ensemble."""
    f = _as_positive(forces, "forces")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, f.size, size=(n_bootstrap, f.size))
    stats = np.quantile(f[idx], q, axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return {
        "q": float(q),
        "estimate": float(np.quantile(f, q)),
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1.0 - alpha)),
        "bootstrap_std": float(stats.std(ddof=1)),
        "confidence": float(confidence),
        "n_bootstrap": float(n_bootstrap),
    }


def breaking_force_ensemble(sheet: IceSheetStrengthModel,
                            marginal: WeibullMarginal,
                            kl: KlField1D,
                            n_real: int, *,
                            db_m: float = 1.0,
                            n_strip: Optional[int] = None,
                            failure_model: str = "weakest_link",
                            seed: int = 20260101,
                            rng: Optional[np.random.Generator] = None
                            ) -> Dict[str, object]:
    """Run ``n_real`` realisations on the first ``n_strip`` grid points of ``kl``.

    Returns the per-realisation forces, ``F / B``, the strength field, and the
    deterministic reference value, so that every downstream number in the
    report can be recomputed from this dictionary alone.
    """
    if n_strip is None:
        n_strip = kl.n_points
    if not (2 <= n_strip <= kl.n_points):
        raise ValueError("n_strip must be within the field grid")
    fields = correlated_weibull_field(marginal, kl, n_real, rng=rng, seed=seed)
    fields = fields[:, :n_strip]
    forces = sheet.strip_forces(fields, db_m=db_m, failure_model=failure_model)
    width = db_m * n_strip
    det = sheet.deterministic_force(width)
    return {
        "fields": fields,
        "forces": forces,
        "per_width": forces / width,
        "width_m": float(width),
        "db_m": float(db_m),
        "n_strip": int(n_strip),
        "deterministic_force": float(det),
        "deterministic_per_width": float(det / width),
        "failure_model": str(failure_model),
        "weakest_link_strength_mean": float(fields.min(axis=1).mean()),
        "strip_strength_mean": float(fields.mean()),
        "strip_strength_cov": float(fields.std(ddof=1) / fields.mean()),
    }


def size_effect_sweep(sheet: IceSheetStrengthModel,
                      marginal: WeibullMarginal,
                      kl: KlField1D,
                      widths_m: Iterable[float], *,
                      n_real: int,
                      db_m: float = 1.0,
                      failure_model: str = "weakest_link",
                      nested: bool = True,
                      seed: int = 20260101) -> List[Dict[str, object]]:
    """Monte-Carlo width sweep with nested (common-random-number) realisations.

    One draw of the field on the widest domain is reused for every narrower
    width by taking its leading columns.  As a Gaussian process restricted to a
    sub-interval has exactly the restricted covariance, each slice *is* a
    realisation of the narrower sheet -- the nesting only correlates the
    realisations across widths, it does not alter any marginal distribution.
    """
    widths = [float(w) for w in widths_m]
    if not widths:
        raise ValueError("widths_m must be non-empty")
    if any(w <= 0.0 for w in widths):
        raise ValueError("widths must be positive")
    rows: List[Dict[str, object]] = []
    nudged = 0
    for width in sorted(widths):
        n_strip = int(round(width / db_m))
        if n_strip < 1 or n_strip > kl.n_points:
            raise ValueError(f"width {width} m is outside the field grid")
        if nested:
            # common random numbers: one field on the widest domain, sliced.  The
            # restriction of a Gaussian process to a sub-interval has exactly the
            # restricted covariance, so each slice is a *bona fide* realisation of
            # the narrower sheet -- and because min over a superset is <= min over
            # its subset, the weakest-link curves are pointwise monotone in B.
            basis = kl
            field_seed = seed
        else:
            # independent seeds per width: the noise reference for the sweep
            basis = gaussian_field_1d(kl.x[:n_strip], kl.length, kl.model)
            field_seed = seed + 1000 * nudged
            nudged += 1
        fields = correlated_weibull_field(marginal, basis, n_real, seed=field_seed)
        sub = fields[:, :n_strip]
        forces = sheet.strip_forces(sub, db_m=db_m, failure_model=failure_model)
        actual_width = db_m * n_strip
        stats = ensemble_statistics(forces)
        stats.pop("running_mean", None)
        rows.append({
            "width_m": float(actual_width),
            "n_strip": n_strip,
            "B_over_L": float(actual_width / kl.length),
            "per_width_mean": float(forces.mean() / actual_width),
            "per_width_q05": float(np.quantile(forces, 0.05) / actual_width),
            "per_width_q50": float(np.quantile(forces, 0.50) / actual_width),
            "per_width_std": float(forces.std(ddof=1) / actual_width),
            "per_width_cov": float(forces.std(ddof=1) / forces.mean()),
            "failure_model": str(failure_model),
            "nested": bool(nested),
            "weakest_link_strength_mean_pa": float(sub.min(axis=1).mean()),
            "deterministic_per_width": float(sheet.deterministic_force(actual_width) / actual_width),
            "q05_over_deterministic": float(np.quantile(forces, 0.05)
                                            / sheet.deterministic_force(actual_width)),
            "q05_over_mean": float(np.quantile(forces, 0.05) / forces.mean()),
            "ensemble": stats,
            "n_real": int(n_real),
        })
    return rows


def fit_power_law(widths: Sequence[float], values: Sequence[float]) -> Dict[str, float]:
    """Fit ``v(B) = v_ref (B / B_ref) ** -n`` in log-log space; report ``n`` and ``R^2``."""
    b = np.asarray(widths, dtype=float)
    v = np.asarray(values, dtype=float)
    if b.size != v.size or b.size < 2:
        raise ValueError("widths and values must be equally sized, with >= 2 entries")
    if np.any(b <= 0.0) or np.any(v <= 0.0):
        raise ValueError("both series must be strictly positive")
    x = np.log(b)
    y = np.log(v)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent": float(-slope),
        "log_intercept": float(intercept),
        "reference_width_m": float(b[0]),
        "reference_value": float(np.exp(intercept + slope * x[0])),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0,
        "rmse_log": float(math.sqrt(ss_res / x.size)),
    }


def monotone_decreasing(values: Sequence[float], *, rtol: float = 0.0) -> Dict[str, object]:
    """Test strict (or tolerant) monotone decrease of a sequence.

    ``rtol > 0`` accepts a decrease or an increase smaller than
    ``rtol * |previous|`` as non-violating, which is what "monotone" means in
    practice for Monte-Carlo estimates with a stated resolution.  Every
    violation is listed with its index and relative size so the report can show
    the worst case instead of a bare boolean.
    """
    v = np.asarray(values, dtype=float)
    violations = []
    for i in range(1, v.size):
        if v[i] > v[i - 1] + rtol * abs(v[i - 1]):
            violations.append({
                "index": int(i),
                "previous": float(v[i - 1]),
                "current": float(v[i]),
                "relative_increase": float(v[i] / v[i - 1] - 1.0),
            })
    return {
        "monotone_decreasing": len(violations) == 0,
        "n_violations": len(violations),
        "violations": violations,
        "rtol": float(rtol),
        "values": [float(x) for x in v],
    }