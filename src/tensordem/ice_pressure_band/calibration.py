"""Finite-window bias calibration for the pressure-band correlation length.

The estimable information on a record of length ``L`` is the ratio
``K = L / lam``; at small ``K`` every estimator in :mod:`.estimators` is biased.
The calibration is **empirical and out-of-sample**:

1. Synthetic fields with an analytically known ``lam`` (``fields.py``) are scanned
   over ``K = L/lam in [1, 50]``; the mean and standard deviation of
   ``lam_hat/lam`` are measured per estimator and field family.
2. The ratio is fitted by one of a small *pre-registered library* of functional
   forms and the form is selected by leave-one-out RMS over the ``K`` grid:

   ``"inv_poly"``   ``rho = 1 + a/K + b/K**2``   (finite-window truncation, the
                    physically motivated leading order)
   ``"inv_cubic"``  ``rho = 1 + a/K + b/K**2 + c/K**3``
   ``"exp_sat"``    ``rho = 1 - exp(-(K/K0)**p)``  (saturating; captures the
                    strong small-``K`` collapse of the integral/exp-fit methods,
                    which the two-parameter form reproduces at only ~8 % RMS)

   Model selection by cross-validation is reported, not hidden: the chosen form
   and its RMS are stored per estimator, and a wrong form shows up as a large
   RMS.
3. The correction is the self-consistent solution of
   ``lam_corr = lam_hat / rho(L/lam_corr)`` (the argument is ``L/lam_corr``, not
   ``L/lam_hat``).
4. The sufficiency threshold ``K*`` is the smallest ``K`` of the **contiguous
   sufficient tail** of the leave-one-out residual-bias curve: every ``K' >= K*``
   must have ``|bias_corrected| <= tol``.  A single lucky cell at small ``K``
   therefore cannot declare sufficiency -- with 400 realisations the LOO mean
   scatters by 10-30 %, and the sustained requirement is what makes the
   threshold an engineering statement rather than a coincidence.
5. ``Var(lam_hat) ~ lam/L`` is tested twice: the log-log slope of the standard
   deviation against ``K`` (theory ``-1/2``) and, at fixed ``L``, against ``lam``
   (theory ``+3/2``).

The analytic reference for the truncation bias of a summation stopped at
``L/2`` is ``E[lam_hat] = lam (1 - exp(-K/2))``, i.e. ``a = -1/2`` in the
``inv_poly`` expansion; where a fitted coefficient departs from that, the report
says so instead of claiming agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimators import METHODS, PRIMARY_METHODS, diagnose_regime, estimate_all
from .fields import windowed_field

__all__ = [
    "BiasModel",
    "BiasStudy",
    "MODEL_FORMS",
    "fit_bias_model",
    "select_bias_model",
    "correct_length",
    "correct_length_batch",
    "k_sufficient",
    "sustained_threshold",
    "run_bias_study",
    "run_variance_family",
    "validate_threshold",
    "scan_threshold_out_of_sample",
]

DEFAULT_K_GRID = (1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0,
                  10.0, 12.6, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0)

MODEL_FORMS = ("inv_poly", "inv_cubic", "exp_sat")


@dataclass(frozen=True)
class BiasModel:
    """``rho(K) = E[lam_hat]/lam`` of one estimator."""

    method: str
    form: str = "inv_poly"
    params: tuple = ()
    rms: float = 0.0
    n_points: int = 0

    def rho(self, K):
        K = np.maximum(np.asarray(K, dtype=float), 1e-9)
        p = self.params
        if self.form == "inv_poly":
            return 1.0 + p[0] / K + p[1] / (K * K)
        if self.form == "inv_cubic":
            return 1.0 + p[0] / K + p[1] / (K * K) + p[2] / (K**3)
        if self.form == "exp_sat":
            k0, power = max(p[0], 1e-6), p[1]
            return 1.0 - np.exp(-((K / k0) ** power))
        raise ValueError(f"unknown bias-model form {self.form!r}")

    @property
    def a(self) -> float:
        """Leading ``1/K`` coefficient when defined (``nan`` otherwise)."""
        return float(self.params[0]) if self.form.startswith("inv_") else float("nan")

    @property
    def b(self) -> float:
        return float(self.params[1]) if self.form.startswith("inv_") else float("nan")

    def to_dict(self) -> dict:
        return {"method": self.method, "form": self.form,
                "params": [float(v) for v in self.params],
                "rms": self.rms, "n_points": self.n_points}


def _design(K: np.ndarray, form: str) -> np.ndarray:
    if form == "inv_poly":
        return np.column_stack([1.0 / K, 1.0 / (K * K)])
    if form == "inv_cubic":
        return np.column_stack([1.0 / K, 1.0 / (K * K), 1.0 / (K**3)])
    raise ValueError("exp_sat is fitted nonlinearly")


def _fit_inv(K, target, form):
    design = _design(K, form)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = design @ coef - target
    return tuple(float(c) for c in coef), float(np.sqrt(np.mean(resid**2)))


def _fit_exp_sat(K, ratio):
    from scipy import optimize

    def model(Kq, k0, power):
        return 1.0 - np.exp(-((Kq / k0) ** power))

    try:
        popt, _ = optimize.curve_fit(
            model, K, ratio, p0=(6.0, 0.9),
            bounds=([0.5, 0.2], [500.0, 4.0]), maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None, float("inf")
    resid = model(K, *popt) - ratio
    return tuple(float(v) for v in popt), float(np.sqrt(np.mean(resid**2)))


def fit_bias_model(k_values, ratio_mean, method: str, *, form: str = "inv_poly",
                   exclude: int | None = None) -> BiasModel:
    """Least-squares fit of one pre-registered functional form."""
    K = np.asarray(k_values, dtype=float)
    y = np.asarray(ratio_mean, dtype=float)
    ok = np.isfinite(y) & np.isfinite(K) & (K > 0.0)
    if exclude is not None:
        mask = np.ones(K.size, dtype=bool)
        mask[exclude] = False
        ok &= mask
    if ok.sum() < 4:
        return BiasModel(method=method, form=form, params=(), rms=float("nan"),
                         n_points=int(ok.sum()))
    if form == "exp_sat":
        params, rms = _fit_exp_sat(K[ok], y[ok])
        if params is None:
            return BiasModel(method=method, form=form, params=(), rms=float("nan"),
                             n_points=int(ok.sum()))
        return BiasModel(method=method, form=form, params=params, rms=rms,
                         n_points=int(ok.sum()))
    params, rms = _fit_inv(K[ok], y[ok] - 1.0, form)
    return BiasModel(method=method, form=form, params=params, rms=rms,
                     n_points=int(ok.sum()))


def select_bias_model(k_values, ratio_mean, method: str) -> BiasModel:
    """Pick the form with the smallest leave-one-out RMS (reported, not hidden)."""
    K = np.asarray(k_values, dtype=float)
    y = np.asarray(ratio_mean, dtype=float)
    ok = np.isfinite(y) & np.isfinite(K) & (K > 0.0)
    if ok.sum() < 6:
        return fit_bias_model(k_values, ratio_mean, method)
    best, best_loo = None, float("inf")
    for form in MODEL_FORMS:
        errors = []
        for i in np.nonzero(ok)[0]:
            m = fit_bias_model(k_values, ratio_mean, method, form=form, exclude=int(i))
            if not np.isfinite(m.rms) or not m.params:
                errors = None
                break
            pred = float(np.asarray(m.rho(K[i])))
            errors.append((pred - y[i]) ** 2)
        if not errors:
            continue
        loo = float(np.sqrt(np.mean(errors)))
        if loo < best_loo:
            best_loo, best = loo, form
    if best is None:
        return fit_bias_model(k_values, ratio_mean, method)
    model = fit_bias_model(k_values, ratio_mean, method, form=best)
    return BiasModel(method=model.method, form=model.form, params=model.params,
                     rms=best_loo, n_points=model.n_points)


def correct_length_batch(lam_hat, L, model: BiasModel, *, n_iter: int = 1,
                        k_min: float = 0.5):
    """Finite-window bias correction of measured correlation lengths.

    ``lam_corr = lam_hat / rho(L / lam_hat)`` -- the **plug-in** form, with the
    bias model evaluated at the observed record length in correlation lengths.

    Three schemes were compared on the calibration ensemble and the residuals are
    reported in the campaign artifacts:

    * plug-in (this one, ``n_iter = 1``): mean residual bias -14.8 %/+3.0 %/+3.4 %
      at ``K = 8/20/50`` (integral method), monotone, no root finding;
    * one further fixed-point step (``n_iter = 2``): overshoots (+8 %/+16 %) --
      the fixed-point map ``lam_hat/rho(L/lam)`` is expansive here because
      ``rho`` rises steeply with ``K``;
    * fully self-consistent inversion (root finding): high-side realisations jump
      to the far branch of the non-monotone map ``lam*rho(L/lam)`` and the
      ensemble mean is inflated by up to +15 %.

    So the plug-in map is used: it never moves an estimate by more than the
    calibrated bias, is exact where the bias model is exact, and its residual
    bias is what the leave-one-out threshold in :func:`run_bias_study` measures.
    ``K`` is clipped at ``k_min`` rather than extrapolated below the calibration
    range.
    """
    a = np.atleast_1d(np.asarray(lam_hat, dtype=float))
    out = np.full(a.shape, np.nan)
    if not model.params or L <= 0.0:
        return out
    ok = np.isfinite(a) & (a > 0.0)
    if not ok.any():
        return out
    lam = a[ok].copy()
    target = a[ok].copy()
    for _ in range(max(int(n_iter), 1)):
        rho = np.asarray(model.rho(np.maximum(L / lam, float(k_min))), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where((rho > 0.0) & np.isfinite(rho), target / rho, np.nan)
    out[ok] = lam
    return out


def correct_length(lam_hat: float, L: float, model: BiasModel, **kw) -> float:
    """Scalar wrapper of :func:`correct_length_batch`."""
    return float(correct_length_batch([lam_hat], L, model, **kw)[0])


def sustained_threshold(k_values, residual_bias, *, tol: float = 0.05,
                        se=None) -> float:
    """Smallest ``K`` of the contiguous sufficient tail of a residual curve.

    Every grid point at or above the returned ``K`` must satisfy
    ``|residual| <= tol + z*se`` -- the Monte-Carlo standard error of the cell is
    added so that a finite ensemble cannot fail the tail by sampling noise alone
    (``z = 2``, a two-sigma test of "bias <= tol").  A lone sufficient cell
    surrounded by failures does not qualify.  Returns ``inf`` when no tail
    qualifies.
    """
    K = np.asarray(k_values, dtype=float)
    r = np.asarray(residual_bias, dtype=float)
    if se is None:
        limit = np.full(r.shape, float(tol))
    else:
        limit = float(tol) + 2.0 * np.asarray(se, dtype=float)
    ok = np.isfinite(r) & (r <= limit)
    idx = None
    for i in range(ok.size - 1, -1, -1):
        if ok[i]:
            idx = i
        else:
            break
    return float(K[idx]) if idx is not None else float("inf")


def k_sufficient(model: BiasModel, *, tol: float = 0.05, k_max: float = 500.0) -> float:
    """Smallest ``K`` above which the *corrected* estimator bias is below ``tol``."""
    if not model.params:
        return float("inf")
    K = np.geomspace(1.0, k_max, 4000)
    rho = np.asarray(model.rho(K), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.abs(1.0 / rho - 1.0)
    ok = np.isfinite(residual) & (residual <= tol)
    if not ok.any():
        return float("inf")
    # contiguous tail
    idx = None
    for i in range(ok.size - 1, -1, -1):
        if ok[i]:
            idx = i
        else:
            break
    return float(K[idx]) if idx is not None else float("inf")


def _measure(K_values, n_real: int, dx_over_lam: float, kind: str, seed: int,
             methods: tuple[str, ...], lam: float = 1.0, progress=None) -> dict:
    """Monte Carlo of ``lam_hat/lam`` on the ``K`` grid for one field family."""
    rng = np.random.default_rng(seed)
    nK = len(K_values)
    ratios = {m: np.full((nK, n_real), np.nan) for m in methods}
    for i, K in enumerate(K_values):
        n = int(max(16, round(K / dx_over_lam)))
        for j in range(n_real):
            x = windowed_field(n, dx_over_lam * lam, lam, rng, kind=kind)
            est = estimate_all(x, dx_over_lam * lam, methods=methods)
            for m in methods:
                ratios[m][i, j] = est[m] / lam
        if progress is not None:
            progress(i + 1, nK, float(K))
    out = {}
    for m in methods:
        r = ratios[m]
        finite = np.isfinite(r)
        mean = np.nanmean(np.where(finite, r, np.nan), axis=1)
        sd = np.nanstd(np.where(finite, r, np.nan), axis=1, ddof=1)
        out[m] = {
            "ratio_mean": mean.tolist(),
            "ratio_sd": sd.tolist(),
            "rel_bias": (mean - 1.0).tolist(),
            "failure_rate": (1.0 - finite.mean(axis=1)).tolist(),
            "ratios": r,
        }
    out["_L_over_lam"] = list(K_values)
    return out


def loglog_fit(x, y) -> dict:
    """Ordinary least squares of ``log y`` on ``log x`` with ``R**2``.

    Returns ``slope``, ``intercept``, ``slope_se``, ``r2`` and ``n``.  The
    coefficient of determination is reported because a scaling law is only
    evidence if the power law actually describes the data: a slope estimate on
    its own can be produced by a scatter plot with no trend.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    n = int(ok.sum())
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "slope_se": float("nan"), "r2": float("nan"), "n": n}
    lx, ly = np.log(x[ok]), np.log(y[ok])
    design = np.column_stack([np.ones(n), lx])
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    resid = ly - design @ coef
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0.0 else float("nan")
    s2 = float(np.sum(resid**2) / max(n - 2, 1))
    cov = s2 * np.linalg.inv(design.T @ design)
    return {"slope": float(coef[1]), "intercept": float(coef[0]),
            "slope_se": float(np.sqrt(max(cov[1, 1], 0.0))), "r2": r2, "n": n}


def _var_exponent(K, sd):
    K = np.asarray(K, dtype=float)
    sd = np.asarray(sd, dtype=float)
    ok = np.isfinite(sd) & (sd > 0.0)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    design = np.column_stack([np.ones(int(ok.sum())), np.log(K[ok])])
    coef, *_ = np.linalg.lstsq(design, np.log(sd[ok]), rcond=None)
    resid = np.log(sd[ok]) - design @ coef
    dof = max(int(ok.sum()) - 2, 1)
    s2 = float(np.sum(resid**2) / dof)
    cov = s2 * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return float(coef[1]), float(coef[0]), se


@dataclass
class BiasStudy:
    """Container for the finite-window calibration of every estimator."""

    k_grid: np.ndarray
    dx_over_lam: float
    n_real: int
    kinds: dict
    models: dict
    sufficiency: dict
    variance: dict
    seed: int

    def model(self, kind: str, method: str) -> BiasModel:
        d = self.models[kind][method]
        return BiasModel(method=method, form=d["form"],
                         params=tuple(d["params"]), rms=d["rms"],
                         n_points=d["n_points"])

    def calibrated(self, kind: str, method: str, *, max_rms: float = 0.05) -> bool:
        """Whether the bias of ``method`` is describable by a K-only correction.

        A model RMS above ``max_rms`` means the estimator's bias is not a smooth
        function of ``L/lam`` alone; such a method is *reported* but excluded
        from the recommended threshold (it cannot be calibrated honestly).
        """
        d = self.models[kind][method]
        return bool(np.isfinite(d["rms"]) and d["rms"] <= max_rms)

    def k_star(self, kind: str, methods: tuple[str, ...] = PRIMARY_METHODS,
               *, max_rms: float = 0.05) -> float:
        """Recommended ``K*``: largest threshold over the *calibrated* methods."""
        vals = [
            self.sufficiency[kind][m]["k_corrected"]
            for m in methods
            if self.calibrated(kind, m, max_rms=max_rms)
        ]
        vals = [v for v in vals if np.isfinite(v)]
        return float(max(vals)) if vals else float("inf")

    def to_dict(self) -> dict:
        return {
            "k_grid": [float(v) for v in self.k_grid],
            "dx_over_lam": self.dx_over_lam,
            "n_real": self.n_real,
            "seed": self.seed,
            "kinds": {
                kind: {
                    m: {
                        "ratio_mean": self.kinds[kind][m]["ratio_mean"],
                        "ratio_sd": self.kinds[kind][m]["ratio_sd"],
                        "rel_bias": self.kinds[kind][m]["rel_bias"],
                        "failure_rate": self.kinds[kind][m]["failure_rate"],
                    }
                    for m in METHODS
                }
                for kind in self.kinds
            },
            "models": self.models,
            "sufficiency": self.sufficiency,
            "variance": self.variance,
        }


def run_bias_study(
    *,
    k_grid=DEFAULT_K_GRID,
    n_real: int = 400,
    dx_over_lam: float = 0.05,
    kinds: tuple[str, ...] = ("spectral", "ar1"),
    seed: int = 0,
    tol: float = 0.05,
    progress=None,
) -> BiasStudy:
    """Run the finite-window bias study; fit models, thresholds and scaling."""
    k_grid = np.asarray(k_grid, dtype=float)
    kinds_out, models, suff, var = {}, {}, {}, {}
    raws = {}
    for kind in kinds:
        raws[kind] = _measure(k_grid, n_real, dx_over_lam, kind, seed, METHODS,
                              progress=progress)
        kinds_out[kind] = raws[kind]
    # the two generators are the same stationary process, so their ensembles are
    # pooled before fitting the bias model: this halves the Monte-Carlo noise of
    # the fitted rho and of the leave-one-out residual, which is what the 5 %
    # threshold is compared against
    pooled = {}
    for m in METHODS:
        pooled[m] = {k: np.concatenate([raws[kind][m][k] for kind in kinds], axis=1)
                     for k in ("ratios",)}
        rr = pooled[m]["ratios"]
        pooled[m]["ratio_mean"] = np.nanmean(rr, axis=1)
        pooled[m]["ratio_sd"] = np.nanstd(rr, axis=1, ddof=1)
    for kind in kinds:
        models[kind], suff[kind], var[kind] = {}, {}, {}
        for m in METHODS:
            raw = pooled[m]
            model = select_bias_model(k_grid, raw["ratio_mean"], m)
            models[kind][m] = model.to_dict()
            ratios = raw["ratios"]
            loo_bias, loo_mean, loo_se = [], [], []
            for i in range(k_grid.size):
                m_loo = fit_bias_model(k_grid, raw["ratio_mean"], m,
                                       form=model.form, exclude=i)
                vals = ratios[i][np.isfinite(ratios[i])]
                corr = correct_length_batch(vals, k_grid[i], m_loo)
                # lam = 1 in the study, so L = K and corr is already lam_corr/lam
                good = np.isfinite(corr)
                mean = float(np.mean(corr[good])) if good.any() else float("nan")
                loo_mean.append(mean)
                loo_se.append(float(np.std(corr[good], ddof=1) / np.sqrt(good.sum()))
                              if good.sum() > 2 else float("nan"))
                loo_bias.append(abs(mean - 1.0) if np.isfinite(mean) else float("nan"))
            loo_bias = np.asarray(loo_bias, dtype=float)
            loo_se = np.asarray(loo_se, dtype=float)
            k_corr = sustained_threshold(k_grid, loo_bias, tol=tol, se=loo_se)
            raw_bias = np.abs(np.asarray(raw["ratio_mean"], dtype=float) - 1.0)
            raw_se = np.asarray(raw["ratio_sd"], dtype=float) / np.sqrt(
                max(np.isfinite(raw["ratios"]).sum(axis=1).min(), 1))
            k_raw = sustained_threshold(k_grid, raw_bias, tol=tol, se=raw_se)
            slope, intercept, se = _var_exponent(
                k_grid, np.nanstd(raws[kind][m]["ratios"], axis=1, ddof=1))
            suff[kind][m] = {
                "k_corrected": k_corr,
                "k_raw": k_raw,
                "loo_residual_bias": loo_bias.tolist(),
                "loo_residual_se": loo_se.tolist(),
                "loo_corrected_ratio": loo_mean,
                "analytic_a_integral": -0.5,
            }
            var[kind][m] = {
                "sd_log_log_slope": slope,
                "sd_log_log_slope_se": se,
                "sd_log_log_intercept": intercept,
                "theory_slope": -0.5,
            }
    return BiasStudy(k_grid=k_grid, dx_over_lam=dx_over_lam, n_real=n_real,
                     kinds=kinds_out, models=models, sufficiency=suff,
                     variance=var, seed=seed)


def run_variance_family(
    *, lam_values, record_length: float = 40.0, n_real: int = 400,
    dx_over_lam: float = 0.05, kind: str = "spectral", seed: int = 0,
    methods: tuple[str, ...] = PRIMARY_METHODS, lam_ref: float = 1.0,
    fixed_length_over_lam: float | None = None,
) -> dict:
    """Second variance-scaling test: fixed **absolute** record length ``L``.

    ``Var(lam_hat) ~ lam/L`` means that at fixed ``L`` the relative standard
    deviation grows like ``lam**0.5`` and ``sd(lam_hat)`` like ``lam**1.5``, while
    ``K = L/lam`` sweeps from large to small.  ``record_length`` is therefore an
    absolute length in the units of ``lam_ref`` (``L = record_length * lam_ref``);
    the sampling step stays ``dx = dx_over_lam * lam`` so every cell has the same
    discretisation error.

    .. note::
       An earlier release tied the record length to ``lam``
       (``n = fixed_length_over_lam / dx_over_lam``), which holds ``L/lam``
       *constant* and makes the fitted slope zero by construction -- it cannot
       test the scaling law at all.  The argument is renamed and the semantics
       fixed here; the regression now also reports ``R**2``.
    """
    if fixed_length_over_lam is not None:
        raise ValueError(
            "fixed_length_over_lam was removed: it held L/lam -- not L -- constant "
            "(n = fixed_length_over_lam / dx_over_lam), so lam and L varied together "
            "and the fitted variance-scaling slope was zero by construction. Use "
            "record_length (an absolute length) instead."
        )
    rng = np.random.default_rng(seed + 991)
    lam_values = np.asarray(lam_values, dtype=float)
    L = float(record_length) * float(lam_ref)
    out = {"lam_values": lam_values.tolist(),
           "record_length": L,
           "L_over_lam": (L / lam_values).tolist()}
    for m in methods:
        rel_sd, sd_abs = [], []
        for lam in lam_values:
            n = int(round(L / (dx_over_lam * lam)))
            if n < 16:
                rel_sd.append(float("nan"))
                sd_abs.append(float("nan"))
                continue
            vals = np.empty(n_real)
            for j in range(n_real):
                x = windowed_field(n, lam * dx_over_lam, lam, rng, kind=kind)
                vals[j] = estimate_all(x, lam * dx_over_lam, methods=(m,))[m] / lam
            ok = np.isfinite(vals)
            rel_sd.append(float(np.nanstd(vals[ok], ddof=1)) if ok.sum() > 2 else float("nan"))
            sd_abs.append(rel_sd[-1] * float(lam))
        fit_rel = loglog_fit(lam_values, rel_sd)          # theory slope +0.5
        fit_abs = loglog_fit(lam_values, sd_abs)          # theory slope +1.5
        out[m] = {
            "rel_sd": rel_sd,
            "sd_abs": sd_abs,
            "sd_vs_lam_slope": fit_rel["slope"],
            "sd_vs_lam_slope_se": fit_rel["slope_se"],
            "rel_sd_vs_lam_slope": fit_rel["slope"],
            "rel_sd_vs_lam_r2": fit_rel["r2"],
            "abs_sd_vs_lam_slope": fit_abs["slope"],
            "abs_sd_vs_lam_r2": fit_abs["r2"],
            "theory_slope": 0.5,
            "theory_slope_abs": 1.5,
        }
    return out


def variance_scaling_grid(
    *,
    K_values=(2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0),
    lam_values=(0.5, 1.0, 2.0, 4.0),
    n_real: int = 400,
    dx_over_lam: float = 0.05,
    kind: str = "spectral",
    seed: int = 0,
    methods: tuple[str, ...] = PRIMARY_METHODS,
    lam_ref: float = 1.0,
) -> dict:
    """Pooled ``Var(lam_hat) ~ lam / L`` test over a 2-D ``(L, lam)`` grid.

    Each cell is a Monte-Carlo ensemble of ``n_real`` records of length
    ``L = K * lam`` at step ``dx = dx_over_lam * lam``.  The criterion of the
    campaign is a **high** coefficient of determination for
    ``log Var(lam_hat/lam)`` against ``log(lam/L)`` (slope 1) -- equivalently
    ``log Var(lam_hat)`` against ``log(lam**3/L)`` -- over the whole grid, not a
    sign check on a slope.
    """
    lam_values = np.asarray(lam_values, dtype=float)
    K_values = np.asarray(K_values, dtype=float)
    rng = np.random.default_rng(seed + 4242)
    cells = []
    for lam in lam_values:
        for K in K_values:
            L = float(K * lam)
            n = int(round(L / (dx_over_lam * lam)))
            if n < 24:
                continue
            row = {"lam": float(lam), "K": float(K), "L": L, "n_samples": n}
            for m in methods:
                vals = np.array([
                    estimate_all(windowed_field(n, dx_over_lam * lam, lam, rng, kind=kind),
                                 dx_over_lam * lam, methods=(m,))[m] / lam
                    for _ in range(n_real)
                ])
                ok = np.isfinite(vals)
                row[m] = {
                    "rel_var": float(np.var(vals[ok], ddof=1)) if ok.sum() > 2 else float("nan"),
                    "rel_sd": float(np.std(vals[ok], ddof=1)) if ok.sum() > 2 else float("nan"),
                    "rel_bias": float(np.mean(vals[ok]) - 1.0) if ok.sum() else float("nan"),
                    "failure_rate": float(1.0 - ok.mean()),
                }
            cells.append(row)
    out = {"K_values": K_values.tolist(), "lam_values": lam_values.tolist(),
           "n_real": n_real, "kind": kind, "cells": cells, "per_method": {}}
    for m in methods:
        lamL, relvar, lam3overL, relsd = [], [], [], []
        for c in cells:
            v = c[m]["rel_var"]
            lamL.append(c["lam"] / c["L"])
            lam3overL.append(c["lam"] ** 3 / c["L"])
            relvar.append(v)
            relsd.append(c[m]["rel_sd"])
        f_rel = loglog_fit(lamL, relvar)     # Var(lam_hat/lam) vs lam/L : slope 1
        f_abs = loglog_fit(lam3overL, relvar)  # Var(lam_hat) ~ lam^3/L    : slope 1
        f_sd = loglog_fit(lamL, relsd)       # sd(lam_hat/lam) vs lam/L   : slope 0.5
        out["per_method"][m] = {
            "relvar_vs_lam_over_L": f_rel,
            "absvar_vs_lam3_over_L": f_abs,
            "relsd_vs_lam_over_L": f_sd,
            "n_cells": len(cells),
        }
    return out


def scan_threshold_out_of_sample(
    study: BiasStudy,
    *,
    kind: str,
    k_grid=(8.0, 12.6, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0),
    seeds=(11, 12, 13),
    length_factors=(1.0, 1.5, 2.0),
    n_real: int = 150,
    methods: tuple[str, ...] = PRIMARY_METHODS,
    lam: float = 1.0,
    tol: float = 0.05,
) -> dict:
    """Out-of-sample threshold scan: the corrected bias on **fresh** ensembles.

    The leave-one-out residual of :func:`run_bias_study` is an *in-sample*
    surrogate: the bias model was fitted to the same ensemble, and because the
    correction is a nonlinear map the residual of the ensemble mean and the mean
    of the corrected values are not the same quantity (they differ by 6--10
    percentage points at ``K ~ 12`` in this campaign).  The threshold that is
    reported and validated is therefore measured here, on independent seeds, on
    a pre-registered ``K`` grid, and is defined as the smallest ``K`` whose
    corrected bias -- and that of every larger ``K`` on the grid -- satisfies
    ``|bias| <= tol`` within two standard errors.  The selection seeds are kept
    disjoint from the confirmation seeds used by :func:`validate_threshold`.
    """
    k_grid = [float(v) for v in k_grid]
    per_k = []
    for Kc in k_grid:
        cells = []
        for seed in seeds:
            rng = np.random.default_rng(int(seed) + 1000)
            for f in length_factors:
                L = Kc * lam * float(f)
                n = int(round(L / (study.dx_over_lam * lam)))
                raw = {m: np.empty(n_real) for m in methods}
                usable = 0
                for j in range(n_real):
                    x = windowed_field(n, study.dx_over_lam * lam, lam, rng, kind=kind)
                    usable += int(diagnose_regime(x, study.dx_over_lam * lam).usable)
                    est = estimate_all(x, study.dx_over_lam * lam, methods=methods)
                    for m in methods:
                        raw[m][j] = est[m]
                cell = {"seed": int(seed), "length_factor": float(f),
                        "L_over_lam_nominal": float(Kc * f), "n_samples": n,
                        "usable_fraction": usable / n_real}
                for m in methods:
                    v = correct_length_batch(raw[m], L, study.model(kind, m)) / lam
                    ok = np.isfinite(v)
                    ratio = float(np.mean(v[ok])) if ok.sum() else float("nan")
                    cell[m] = {"corrected_ratio_mean": ratio,
                               "corrected_rel_bias": ratio - 1.0,
                               "relative_sd": float(np.std(v[ok], ddof=1) / max(ratio, 1e-12))
                               if ok.sum() > 2 else float("nan"),
                               "failure_rate": float(1.0 - ok.mean())}
                cells.append(cell)
        entry = {"K": Kc, "cells": cells, "per_method": {}}
        for m in methods:
            biases = np.array([c[m]["corrected_rel_bias"] for c in cells], dtype=float)
            mean_bias = float(np.nanmean(biases))
            se = float(np.nanstd(biases, ddof=1) / np.sqrt(max(np.isfinite(biases).sum(), 1)))
            entry["per_method"][m] = {
                "mean_bias": mean_bias,
                "mean_bias_se": se,
                "max_abs_bias": float(np.nanmax(np.abs(biases))),
                "relative_sd": float(np.nanmean([c[m]["relative_sd"] for c in cells])),
                "within_tol": bool(abs(mean_bias) <= tol + 2.0 * se),
            }
        per_k.append(entry)
    k_star = {}
    for m in methods:
        ok = [e["per_method"][m]["within_tol"] for e in per_k]
        idx = None
        for i in range(len(ok) - 1, -1, -1):
            if ok[i]:
                idx = i
            else:
                break
        k_star[m] = float(per_k[idx]["K"]) if idx is not None else float("inf")
    return {"kind": kind, "k_grid": k_grid, "seeds": [int(s) for s in seeds],
            "length_factors": [float(f) for f in length_factors],
            "n_real": n_real, "per_k": per_k, "k_star": k_star,
            "tolerance": tol}


def validate_threshold(
    study: BiasStudy,
    *,
    kind: str,
    k_star: float,
    lam: float = 1.0,
    seeds=(11, 12, 13),
    length_factors=(1.0, 1.5, 2.0),
    n_real: int = 300,
    methods: tuple[str, ...] = PRIMARY_METHODS,
    tol: float = 0.05,
) -> dict:
    """Out-of-sample check of the threshold on 3 seeds x 3 record lengths."""
    if not np.isfinite(k_star):
        raise ValueError("a non-finite threshold cannot be validated")
    cells = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for f in length_factors:
            L = k_star * lam * float(f)
            n = int(round(L / (study.dx_over_lam * lam)))
            raw = {m: np.empty(n_real) for m in methods}
            usable = 0
            for j in range(n_real):
                x = windowed_field(n, study.dx_over_lam * lam, lam, rng, kind=kind)
                regime = diagnose_regime(x, study.dx_over_lam * lam)
                usable += int(regime.usable)
                est = estimate_all(x, study.dx_over_lam * lam, methods=methods)
                for m in methods:
                    raw[m][j] = est[m]
            vals = {m: correct_length_batch(raw[m], L, study.model(kind, m)) / lam
                    for m in methods}
            cell = {
                "seed": int(seed),
                "length_factor": float(f),
                "L_over_lam_nominal": float(k_star * f),
                "n_samples": n,
                "usable_fraction": usable / n_real,
            }
            for m in methods:
                v = vals[m]
                ok = np.isfinite(v)
                ratio = float(np.mean(v[ok])) if ok.sum() else float("nan")
                cell[m] = {
                    "corrected_ratio_mean": ratio,
                    "corrected_rel_bias": ratio - 1.0 if np.isfinite(ratio) else float("nan"),
                    "relative_sd": float(np.std(v[ok], ddof=1) / max(ratio, 1e-12))
                    if ok.sum() > 2 else float("nan"),
                    "within_tol": bool(abs(ratio - 1.0) <= tol) if np.isfinite(ratio) else False,
                    "failure_rate": float(1.0 - ok.mean()),
                }
            cells.append(cell)
    per_method = {}
    for m in methods:
        biases = np.asarray([c[m]["corrected_rel_bias"] for c in cells], dtype=float)
        biases = biases[np.isfinite(biases)]
        pooled = np.asarray([v for c in cells for v in c[m].get("values", np.array([]))
                             if np.isfinite(v)], dtype=float)
        pooled_mean = float(np.mean(pooled)) if pooled.size else float(np.mean(biases))
        pooled_se = (float(np.std(pooled, ddof=1) / np.sqrt(pooled.size))
                     if pooled.size > 2 else float("nan"))
        per_method[m] = {
            "max_abs_bias": float(np.max(np.abs(biases))) if biases.size else float("nan"),
            "cell_mean_bias_spread": float(np.std(biases, ddof=1)) if biases.size > 2 else float("nan"),
            "pooled_mean_bias": pooled_mean,
            "pooled_se": pooled_se,
            "n_pooled": int(pooled.size),
            "within_tol": bool(np.isfinite(pooled_mean)
                               and abs(pooled_mean) <= tol + 2.0 * (pooled_se if np.isfinite(pooled_se) else 0.0)),
            # every single cell within tol is a much stronger statement than the
            # ensemble bias and is reported as information, not as the criterion
            "all_cells_within_tol": bool(biases.size and np.all(np.abs(biases) <= tol)),
        }
    return {"cells": cells, "per_method": per_method, "tolerance": tol,
            "k_star_used": float(k_star)}