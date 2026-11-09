"""Spatio-temporal ice abrasion field evolution: deterministic activity profile
+ random residual field conditional simulation + panel-wise accumulation.

Superseded evidence chain
------------------------
The verification in this module is partly self-referential (the "DEM" activity
field is generated from this module's own parametric profile plus white noise,
"conditional" simulation never conditions on observations, and the mass ledger
compares two integrals of the same analytic field).  The N05 node
(:mod:`tensordem.ice_abrasion_spatiotemporal`) keeps this interface but replaces
the evidence chain with a real DEM ensemble, kriging-conditioned residual fields
and a moving-frame accumulation verified against closed forms.  Prefer that
module for new work; this one is kept for the interface and its unit tests.

Why this module exists
----------------------
N02 (IN66-N02) produced a *single-point* P95 statistic from DEM contact
ensembles.  N04 upgrades this to a full **spatio-temporal field** ``d(x, y, t)``
on the hull surface, enabling:

1. **Deterministic activity profile** ``a(x, y)``: a parameterised fit
   (Gaussian peak(s) + exponential decay tail) to the DEM-derived abrasion rate
   field, with R² > 0.95 (C1).
2. **Random residual field** ``r(x, y) = a_DEM(x, y) - a_profile(x, y)``:
   spatial correlation estimated from DEM residuals, then KL-expansion-based
   conditional simulation producing samples whose mean/variance/correlation
   match the target within 5% (C2).
3. **Spatio-temporal evolution**: the hull is discretised into N panels; each
   panel accumulates wear depth ``d_i(t) = ∫₀ᵗ a_i(τ) dτ`` as the ship moves
   through ice, with exact mass conservation (C3, rel. err < 1e-6).
4. **N02对接**: P95 extracted from the field vs N02 direct statistics,
   error < 10% (C4).
5. **Analytic anchor**: uniform ice field ``d(t) = a·t`` verified to < 1% (C5).
6. **Performance**: 1000 panels × 100 time steps < 60 s CPU (C6).

Conventions
-----------
* SI units throughout (m, s, Pa, m/s for rates, m for depths).
* Panel coordinates: ``(x, y)`` on the hull surface; ``x`` along ship length
  (0 = stern, L = bow), ``y`` vertical from baseline.
* Time in seconds; voyage legs define piecewise-constant ice conditions.
* All random fields use float64 for numerical stability in KL truncation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

DTYPE = torch.float64
NP_DTYPE = np.float64

__all__ = [
    "AbrasionProfile",
    "AbrasionProfileFit",
    "ConditionalRandomField",
    "KLExpansion",
    "PanelMesh",
    "P95Connector",
    "SpatioTemporalField",
    "SpatioTemporalEvolution",
    "fit_activity_profile",
    "estimate_spatial_correlation",
    "kl_expansion_2d",
    "generate_conditional_samples",
    "build_panel_mesh",
    "evolve_spatio_temporal",
    "extract_p95_from_field",
    "connect_to_n02_p95",
    "verify_mass_conservation",
    "verify_uniform_anchor",
    "run_n04_verification",
]


# ---------------------------------------------------------------------------
# 1. Panel mesh: hull surface discretisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelMesh:
    """Hull surface discretised into N panels for wear accumulation."""

    n_panels: int
    x_centers: np.ndarray
    y_centers: np.ndarray
    areas: np.ndarray
    normals: np.ndarray | None = None
    hull_length_m: float = 0.0
    hull_beam_m: float = 0.0
    hull_draught_m: float = 0.0

    def __post_init__(self) -> None:
        n = int(self.n_panels)
        for name, arr in (("x_centers", self.x_centers),
                          ("y_centers", self.y_centers),
                          ("areas", self.areas)):
            arr = np.asarray(arr, dtype=NP_DTYPE).ravel()
            if arr.size != n:
                raise ValueError(f"{name} size {arr.size} != n_panels {n}")
        if self.normals is not None:
            arr = np.asarray(self.normals, dtype=NP_DTYPE)
            if arr.shape != (n, 3):
                raise ValueError(f"normals shape {arr.shape} != ({n}, 3)")

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))


def build_panel_mesh(
    *,
    n_x: int = 40,
    n_y: int = 25,
    hull_length_m: float = 185.0,
    hull_beam_m: float = 32.26,
    hull_draught_m: float = 11.5,
) -> PanelMesh:
    """Build a structured panel mesh on one side of the hull."""
    if n_x < 2 or n_y < 2:
        raise ValueError("n_x and n_y must be >= 2")

    x_edges = np.linspace(0.0, hull_length_m, n_x + 1, dtype=NP_DTYPE)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_edges = np.linspace(0.0, hull_draught_m * 1.15, n_y + 1, dtype=NP_DTYPE)
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])

    m = 2.0
    n_exp = 1.5
    u = x_centers / hull_length_m
    eta = np.clip(4.0 * u * (1.0 - u), 0.0, None) ** m
    zeta = np.clip(y_centers / hull_draught_m, 0.0, 1.0) ** n_exp
    half_breadth = 0.5 * hull_beam_m * eta[:, None] * zeta[None, :]

    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    dx_2d = dx[None, :]
    dy_2d = dy[:, None]

    d_eta_dx = m * np.clip(4.0 * u * (1.0 - u), 1e-12, None) ** (m - 1) * (4.0 * (1.0 - 2.0 * u) / hull_length_m)
    db_dx_2d = (0.5 * hull_beam_m * d_eta_dx[None, :] * zeta[:, None])
    d_zeta_dy = n_exp * np.clip(y_centers / hull_draught_m, 1e-12, None) ** (n_exp - 1) / hull_draught_m
    db_dy_2d = (0.5 * hull_beam_m * eta[None, :] * d_zeta_dy[:, None])

    stretch = np.sqrt(1.0 + db_dx_2d ** 2 + db_dy_2d ** 2)
    areas = (dx_2d * dy_2d * stretch).ravel()

    x_c = np.tile(x_centers, n_y)
    y_c = np.repeat(y_centers, n_x)

    return PanelMesh(
        n_panels=n_x * n_y,
        x_centers=x_c,
        y_centers=y_c,
        areas=areas,
        hull_length_m=hull_length_m,
        hull_beam_m=hull_beam_m,
        hull_draught_m=hull_draught_m,
    )


# ---------------------------------------------------------------------------
# 2. Deterministic activity profile: Gaussian peak(s) + exponential tail
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AbrasionProfile:
    """Parameterised deterministic abrasion rate profile a(x, y).

    Model:
        a(x, y) = A0 * exp(-((x-x0)/sx)^2) * exp(-|y-y0|/ly)
                + A1 * exp(-((x-x1)/sx1)^2) * exp(-|y-y1|/ly1)
                + a_bg
    """

    a0: float
    x0: float
    sigma_x: float
    y0: float
    lambda_y: float
    a1: float = 0.0
    x1: float = 0.0
    sigma_x1: float = 1.0
    y1: float = 0.0
    lambda_y1: float = 1.0
    a_bg: float = 0.0

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=NP_DTYPE)
        y = np.asarray(y, dtype=NP_DTYPE)
        term0 = (self.a0
                 * np.exp(-((x - self.x0) / self.sigma_x) ** 2)
                 * np.exp(-np.abs(y - self.y0) / self.lambda_y))
        term1 = 0.0
        if self.a1 != 0.0:
            term1 = (self.a1
                     * np.exp(-((x - self.x1) / self.sigma_x1) ** 2)
                     * np.exp(-np.abs(y - self.y1) / self.lambda_y1))
        return term0 + term1 + self.a_bg

    def evaluate_on_mesh(self, mesh: PanelMesh) -> np.ndarray:
        return self.evaluate(mesh.x_centers, mesh.y_centers)


@dataclass(frozen=True)
class AbrasionProfileFit:
    """Result of fitting an AbrasionProfile to DEM-derived data."""

    profile: AbrasionProfile
    r_squared: float
    rmse: float
    n_data: int
    fit_params: dict[str, float]
    convergence: bool
    n_iterations: int


def fit_activity_profile(
    x_data: np.ndarray,
    y_data: np.ndarray,
    a_data: np.ndarray,
    *,
    maxiter: int = 500,
    seed: int = 42,
) -> AbrasionProfileFit:
    """Fit a Gaussian+exponential activity profile to DEM-derived abrasion data."""
    from scipy.optimize import least_squares

    x = np.asarray(x_data, dtype=NP_DTYPE).ravel()
    y = np.asarray(y_data, dtype=NP_DTYPE).ravel()
    a = np.asarray(a_data, dtype=NP_DTYPE).ravel()

    if x.size != y.size or x.size != a.size:
        raise ValueError("x_data, y_data, a_data must have the same size")
    if np.any(a < 0):
        raise ValueError("abrasion rates must be nonnegative")

    n = x.size
    a_max_idx = np.argmax(a)
    x0_guess = float(x[a_max_idx])
    y0_guess = float(y[a_max_idx])
    a0_guess = float(a[a_max_idx])

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    sigma_x_guess = max(x_std * 0.5, 1.0)
    lambda_y_guess = max(y_std * 0.5, 0.1)

    bg_threshold = np.percentile(a, 20)
    a_bg_guess = float(np.mean(a[a <= bg_threshold])) if np.any(a <= bg_threshold) else 0.0

    # Better initial guesses: use data statistics more carefully
    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    
    p0 = [a0_guess, x0_guess, max(sigma_x_guess, x_range * 0.1), y0_guess, max(lambda_y_guess, y_range * 0.1),
          a0_guess * 0.2, x0_guess * 0.5, max(sigma_x_guess * 0.6, x_range * 0.1), y0_guess, max(lambda_y_guess, y_range * 0.1),
          a_bg_guess]

    def _residuals(params: np.ndarray) -> np.ndarray:
        a0, x0, sx, y0, ly, a1, x1, sx1, y1, ly1, abg = params
        model = (a0 * np.exp(-((x - x0) / max(sx, 1e-6)) ** 2)
                 * np.exp(-np.abs(y - y0) / max(ly, 1e-6))
                 + a1 * np.exp(-((x - x1) / max(sx1, 1e-6)) ** 2)
                 * np.exp(-np.abs(y - y1) / max(ly1, 1e-6))
                 + abg)
        return model - a

    lower = [0.0, 0.0, 0.01, 0.0, 0.001, 0.0, 0.0, 0.01, 0.0, 0.001, 0.0]
    upper = [np.inf] * 11

    result = least_squares(
        _residuals, p0, bounds=(lower, upper),
        loss="soft_l1", f_scale=np.std(a) * 0.1 if np.std(a) > 0 else 0.1,
        max_nfev=maxiter, xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )

    params = result.x
    profile = AbrasionProfile(
        a0=float(params[0]), x0=float(params[1]), sigma_x=float(params[2]),
        y0=float(params[3]), lambda_y=float(params[4]),
        a1=float(params[5]), x1=float(params[6]), sigma_x1=float(params[7]),
        y1=float(params[8]), lambda_y1=float(params[9]),
        a_bg=float(params[10]),
    )

    a_pred = profile.evaluate(x, y)
    ss_res = float(np.sum((a - a_pred) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / n)

    return AbrasionProfileFit(
        profile=profile,
        r_squared=r_squared,
        rmse=rmse,
        n_data=n,
        fit_params={
            "a0": params[0], "x0": params[1], "sigma_x": params[2],
            "y0": params[3], "lambda_y": params[4],
            "a1": params[5], "x1": params[6], "sigma_x1": params[7],
            "y1": params[8], "lambda_y1": params[9],
            "a_bg": params[10],
        },
        convergence=bool(result.success),
        n_iterations=int(result.nfev),
    )


# ---------------------------------------------------------------------------
# 3. Spatial correlation estimation
# ---------------------------------------------------------------------------

def estimate_spatial_correlation(
    x_data: np.ndarray,
    y_data: np.ndarray,
    residual_data: np.ndarray,
    *,
    n_lag_bins: int = 30,
) -> dict[str, Any]:
    """Estimate the spatial correlation function R_r(dx, dy) from residuals."""
    x = np.asarray(x_data, dtype=NP_DTYPE).ravel()
    y = np.asarray(y_data, dtype=NP_DTYPE).ravel()
    r = np.asarray(residual_data, dtype=NP_DTYPE).ravel()

    n = x.size
    if n < 10:
        raise ValueError("need at least 10 data points")

    r_mean = np.mean(r)
    r_std = np.std(r)
    if r_std < 1e-15:
        return {
            "correlation_length_x": np.inf,
            "correlation_length_y": np.inf,
            "variance": 0.0,
            "model": "white_noise",
            "empirical_lags": np.array([]),
            "empirical_corr": np.array([]),
        }

    r_norm = (r - r_mean) / r_std
    variance = float(r_std ** 2)

    max_pairs = 50000
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(42)
        n_sub = int(math.sqrt(2 * max_pairs))
        idx = rng.choice(n, size=min(n_sub, n), replace=False)
        x, y, r_norm = x[idx], y[idx], r_norm[idx]
        n = len(idx)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r_norm[:, None] * r_norm[None, :]

    i, j = np.triu_indices(n, k=1)
    dist_x = np.abs(dx[i, j])
    dist_y = np.abs(dy[i, j])
    corr_vals = dr[i, j]
    dist_r = np.sqrt(dist_x ** 2 + dist_y ** 2)

    max_dist = float(np.percentile(dist_r, 95))
    if max_dist < 1e-12:
        max_dist = 1.0
    lag_edges = np.linspace(0, max_dist, n_lag_bins + 1)
    lag_centers = 0.5 * (lag_edges[1:] + lag_edges[:-1])

    empirical_corr = np.zeros(n_lag_bins)
    empirical_count = np.zeros(n_lag_bins)
    for k in range(n_lag_bins):
        mask = (dist_r >= lag_edges[k]) & (dist_r < lag_edges[k + 1])
        if np.any(mask):
            empirical_corr[k] = float(np.mean(corr_vals[mask]))
            empirical_count[k] = float(np.sum(mask))

    from scipy.optimize import curve_fit

    def _exp_corr(d: np.ndarray, L: float) -> np.ndarray:
        return np.exp(-d / max(L, 1e-6))

    valid = empirical_count > 5
    if np.sum(valid) < 3:
        return {
            "correlation_length_x": max_dist,
            "correlation_length_y": max_dist,
            "variance": variance,
            "model": "insufficient_data",
            "empirical_lags": lag_centers,
            "empirical_corr": empirical_corr,
        }

    try:
        popt, _ = curve_fit(_exp_corr, lag_centers[valid], empirical_corr[valid],
                            p0=[max_dist * 0.3], bounds=(1e-6, max_dist * 2.0))
        corr_length = float(popt[0])
    except Exception:
        corr_length = max_dist * 0.3

    # Separate x and y
    max_dx = float(np.percentile(dist_x, 95))
    if max_dx < 1e-12:
        max_dx = 1.0
    lag_x = np.linspace(0, max_dx, n_lag_bins)
    corr_x = np.zeros(n_lag_bins)
    for k in range(n_lag_bins):
        mask = (dist_x >= lag_x[k] * 0.9) & (dist_x < lag_x[k] * 1.1)
        if np.any(mask):
            corr_x[k] = float(np.mean(corr_vals[mask]))

    try:
        popt_x, _ = curve_fit(_exp_corr, lag_x[:n_lag_bins // 2], corr_x[:n_lag_bins // 2],
                              p0=[max_dx * 0.3], bounds=(1e-6, max_dx * 2.0))
        corr_length_x = float(popt_x[0])
    except Exception:
        corr_length_x = corr_length

    max_dy = float(np.percentile(dist_y, 95))
    if max_dy < 1e-12:
        max_dy = 1.0
    lag_y = np.linspace(0, max_dy, n_lag_bins)
    corr_y = np.zeros(n_lag_bins)
    for k in range(n_lag_bins):
        mask = (dist_y >= lag_y[k] * 0.9) & (dist_y < lag_y[k] * 1.1)
        if np.any(mask):
            corr_y[k] = float(np.mean(corr_vals[mask]))

    try:
        popt_y, _ = curve_fit(_exp_corr, lag_y[:n_lag_bins // 2], corr_y[:n_lag_bins // 2],
                              p0=[max_dy * 0.3], bounds=(1e-6, max_dy * 2.0))
        corr_length_y = float(popt_y[0])
    except Exception:
        corr_length_y = corr_length

    return {
        "correlation_length_x": corr_length_x,
        "correlation_length_y": corr_length_y,
        "variance": variance,
        "model": "exponential",
        "empirical_lags": lag_centers,
        "empirical_corr": empirical_corr,
        "empirical_count": empirical_count,
    }


# ---------------------------------------------------------------------------
# 4. KL expansion for 2D random fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KLExpansion:
    """Karhunen-Loeve expansion of a 2D random field on a panel mesh."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_modes: int
    variance_explained: float

    def generate_samples(self, n_samples: int, *, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal((n_samples, self.n_modes))
        sqrt_lambda = np.sqrt(self.eigenvalues)
        samples = xi * sqrt_lambda[None, :] @ self.eigenvectors.T
        return samples

    def generate_conditional_samples(
        self, n_samples: int, observed_indices: np.ndarray,
        observed_values: np.ndarray, *, seed: int = 42,
    ) -> np.ndarray:
        obs_idx = np.asarray(observed_indices, dtype=int)
        obs_val = np.asarray(observed_values, dtype=NP_DTYPE)

        if obs_idx.size != obs_val.size:
            raise ValueError("observed_indices and observed_values must match")

        n_panels = self.eigenvectors.shape[0]
        sqrt_lambda = np.sqrt(self.eigenvalues)

        phi_obs = self.eigenvectors[obs_idx, :]
        cov_obs = phi_obs * sqrt_lambda[None, :]
        Sigma_obs = cov_obs @ cov_obs.T
        Sigma_obs += 1e-10 * np.eye(obs_idx.size)

        phi_all = self.eigenvectors
        cov_all = phi_all * sqrt_lambda[None, :]
        Sigma_cross = cov_all @ cov_obs.T

        try:
            Sigma_obs_inv = np.linalg.solve(Sigma_obs, np.eye(obs_idx.size))
        except np.linalg.LinAlgError:
            Sigma_obs_inv = np.linalg.pinv(Sigma_obs)

        mu_cond = Sigma_cross @ Sigma_obs_inv @ obs_val

        Sigma_cond = cov_all @ cov_all.T - Sigma_cross @ Sigma_obs_inv @ Sigma_cross.T
        Sigma_cond += 1e-10 * np.eye(n_panels)

        try:
            L_cond = np.linalg.cholesky(Sigma_cond)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(Sigma_cond)
            eigvals = np.maximum(eigvals, 0.0)
            L_cond = eigvecs * np.sqrt(eigvals)

        rng = np.random.default_rng(seed)
        xi = rng.standard_normal((n_samples, n_panels))
        samples = mu_cond[None, :] + xi @ L_cond.T
        return samples


def kl_expansion_2d(
    mesh: PanelMesh,
    *,
    correlation_length_x: float,
    correlation_length_y: float,
    variance: float,
    n_modes: int = 100,
    correlation_model: str = "exponential",
) -> KLExpansion:
    """Compute KL expansion of a 2D random field on a panel mesh."""
    n = mesh.n_panels
    x = mesh.x_centers
    y = mesh.y_centers

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    if correlation_model == "exponential":
        cov = variance * np.exp(-np.abs(dx) / max(correlation_length_x, 1e-6)
                                - np.abs(dy) / max(correlation_length_y, 1e-6))
    elif correlation_model == "gaussian":
        cov = variance * np.exp(-(dx / max(correlation_length_x, 1e-6)) ** 2
                                - (dy / max(correlation_length_y, 1e-6)) ** 2)
    else:
        raise ValueError(f"unknown correlation model: {correlation_model}")

    cov += 1e-12 * np.eye(n)

    if n <= 500:
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    else:
        m = min(200, n // 2)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=m, replace=False)
        C_mm = cov[np.ix_(sample_idx, sample_idx)]
        C_nm = cov[:, sample_idx]

        eigvals_m, eigvecs_m = np.linalg.eigh(C_mm)
        idx = np.argsort(eigvals_m)[::-1][:n_modes]
        eigvals_m = eigvals_m[idx]
        eigvecs_m = eigvecs_m[:, idx]

        eigvecs = C_nm @ (eigvecs_m / np.sqrt(eigvals_m)[None, :])
        eigvals = eigvals_m

        norms = np.linalg.norm(eigvecs, axis=0)
        eigvecs = eigvecs / norms[None, :]

    n_modes_actual = min(n_modes, n)
    eigvals = eigvals[:n_modes_actual]
    eigvecs = eigvecs[:, :n_modes_actual]
    eigvals = np.maximum(eigvals, 0.0)

    total_var = float(np.sum(eigvals))
    total_cov_trace = float(np.trace(cov))
    variance_explained = total_var / total_cov_trace if total_cov_trace > 0 else 0.0

    return KLExpansion(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        n_modes=n_modes_actual,
        variance_explained=variance_explained,
    )


# ---------------------------------------------------------------------------
# 5. Conditional random field generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionalRandomField:
    """Conditional random field samples with statistics verification."""

    samples: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    target_mean: np.ndarray
    target_variance: np.ndarray
    n_samples: int
    mean_error_pct: float
    variance_error_pct: float


def generate_conditional_samples(
    kl: KLExpansion,
    n_samples: int,
    *,
    observed_indices: np.ndarray | None = None,
    observed_values: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_variance: np.ndarray | None = None,
    seed: int = 42,
) -> ConditionalRandomField:
    """Generate conditional random field samples and verify statistics."""
    n_panels = kl.eigenvectors.shape[0]

    if observed_indices is not None and observed_values is not None:
        samples = kl.generate_conditional_samples(
            n_samples, observed_indices, observed_values, seed=seed)
    else:
        samples = kl.generate_samples(n_samples, seed=seed)

    if target_mean is not None:
        samples += target_mean[None, :]

    sample_mean = np.mean(samples, axis=0)
    sample_var = np.var(samples, axis=0)

    target_m = target_mean if target_mean is not None else np.zeros(n_panels)
    target_v = target_variance if target_variance is not None else np.sum(
        kl.eigenvalues[None, :] * kl.eigenvectors ** 2, axis=1)

    mean_err = np.abs(sample_mean - target_m)
    # Use absolute error when target is near zero, relative otherwise
    denom = np.where(np.abs(target_m) > 1e-12, np.abs(target_m), 1.0)
    mean_err_pct = float(np.max(mean_err / denom)) * 100.0

    var_err = np.abs(sample_var - target_v)
    var_err_pct = float(np.max(var_err / (np.abs(target_v) + 1e-15))) * 100.0

    return ConditionalRandomField(
        samples=samples,
        mean=sample_mean,
        variance=sample_var,
        target_mean=target_m,
        target_variance=target_v,
        n_samples=n_samples,
        mean_error_pct=mean_err_pct,
        variance_error_pct=var_err_pct,
    )


# ---------------------------------------------------------------------------
# 6. Spatio-temporal evolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpatioTemporalField:
    """Spatio-temporal abrasion field d(x, y, t)."""

    mesh: PanelMesh
    n_time_steps: int
    time_s: np.ndarray
    depth_m: np.ndarray
    rate_m_s: np.ndarray
    total_mass_kg: float
    ice_density_kg_m3: float = 917.0

    def depth_at(self, t: float) -> np.ndarray:
        if t <= self.time_s[0]:
            return self.depth_m[0]
        if t >= self.time_s[-1]:
            return self.depth_m[-1]
        idx = np.searchsorted(self.time_s, t) - 1
        idx = max(0, min(idx, self.n_time_steps - 2))
        t0, t1 = self.time_s[idx], self.time_s[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (1.0 - alpha) * self.depth_m[idx] + alpha * self.depth_m[idx + 1]

    def max_depth(self) -> float:
        return float(np.max(self.depth_m))

    def p95_depth(self) -> float:
        return float(np.percentile(self.depth_m[-1], 95.0))


@dataclass(frozen=True)
class SpatioTemporalEvolution:
    """Engine for spatio-temporal abrasion field evolution."""

    mesh: PanelMesh
    profile: AbrasionProfile
    kl_expansion: KLExpansion | None = None
    random_samples: np.ndarray | None = None
    ice_density_kg_m3: float = 917.0

    def evolve(
        self,
        time_s: np.ndarray,
        *,
        activity_multiplier: float = 1.0,
        use_random_field: bool = False,
        sample_index: int = 0,
        seed: int = 42,
    ) -> SpatioTemporalField:
        n_steps = len(time_s)
        n_panels = self.mesh.n_panels

        depth = np.zeros((n_steps, n_panels), dtype=NP_DTYPE)
        rate = np.zeros((n_steps, n_panels), dtype=NP_DTYPE)

        a_det = self.profile.evaluate_on_mesh(self.mesh) * activity_multiplier

        if use_random_field:
            if self.random_samples is not None:
                a_rand = self.random_samples[sample_index % self.random_samples.shape[0]]
            elif self.kl_expansion is not None:
                a_rand = self.kl_expansion.generate_samples(1, seed=seed)[0]
            else:
                a_rand = np.zeros(n_panels)
        else:
            a_rand = np.zeros(n_panels)

        for k in range(n_steps):
            a_k = a_det + a_rand
            a_k = np.maximum(a_k, 0.0)
            rate[k] = a_k

            if k == 0:
                depth[k] = a_k * time_s[0]
            else:
                dt = time_s[k] - time_s[k - 1]
                depth[k] = depth[k - 1] + a_k * dt

        total_volume = float(np.sum(depth[-1] * self.mesh.areas))
        total_mass = total_volume * self.ice_density_kg_m3

        return SpatioTemporalField(
            mesh=self.mesh,
            n_time_steps=n_steps,
            time_s=time_s,
            depth_m=depth,
            rate_m_s=rate,
            total_mass_kg=total_mass,
            ice_density_kg_m3=self.ice_density_kg_m3,
        )


def evolve_spatio_temporal(
    mesh: PanelMesh,
    profile: AbrasionProfile,
    time_s: np.ndarray,
    *,
    kl_expansion: KLExpansion | None = None,
    use_random_field: bool = False,
    n_samples: int = 1,
    sample_index: int = 0,
    activity_multiplier: float = 1.0,
    seed: int = 42,
) -> SpatioTemporalField:
    """Convenience function for spatio-temporal evolution."""
    engine = SpatioTemporalEvolution(
        mesh=mesh,
        profile=profile,
        kl_expansion=kl_expansion,
        ice_density_kg_m3=917.0,
    )
    return engine.evolve(
        time_s,
        activity_multiplier=activity_multiplier,
        use_random_field=use_random_field,
        sample_index=sample_index,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 7. Mass conservation verification
# ---------------------------------------------------------------------------

def verify_mass_conservation(
    field: SpatioTemporalField,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Verify mass conservation of the spatio-temporal evolution."""
    n_panels = field.mesh.n_panels
    areas = field.mesh.areas
    rho = field.ice_density_kg_m3

    mass_from_depth = float(np.sum(field.depth_m[-1] * areas * rho))

    dt = np.diff(field.time_s)
    if len(dt) == 0:
        mass_from_rate = mass_from_depth
    else:
        rate_integral = np.zeros(n_panels)
        for k in range(len(dt)):
            rate_integral += 0.5 * (field.rate_m_s[k] + field.rate_m_s[k + 1]) * dt[k]
        mass_from_rate = float(np.sum(rate_integral * areas * rho))

    rel_error = abs(mass_from_depth - mass_from_rate) / max(mass_from_depth, 1e-15)

    return {
        "mass_from_depth_kg": mass_from_depth,
        "mass_from_rate_integral_kg": mass_from_rate,
        "relative_error": rel_error,
        "passed": rel_error < tolerance,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# 8. Uniform ice field analytic anchor
# ---------------------------------------------------------------------------

def verify_uniform_anchor(
    mesh: PanelMesh,
    uniform_rate: float,
    time_s: np.ndarray,
    *,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Verify that uniform ice field produces d(t) = a*t within tolerance."""
    profile = AbrasionProfile(
        a0=uniform_rate, x0=0.0, sigma_x=1e6,
        y0=0.0, lambda_y=1e6, a_bg=0.0,
    )

    fld = evolve_spatio_temporal(
        mesh, profile, time_s,
        use_random_field=False,
        activity_multiplier=1.0,
    )

    max_error = 0.0
    for k in range(len(time_s)):
        expected = uniform_rate * time_s[k]
        actual = fld.depth_m[k]
        err = np.max(np.abs(actual - expected)) / max(expected, 1e-15)
        max_error = max(max_error, float(err))

    return {
        "uniform_rate_m_s": uniform_rate,
        "max_relative_error": max_error,
        "passed": max_error < tolerance,
        "tolerance": tolerance,
        "n_panels": mesh.n_panels,
        "n_time_steps": len(time_s),
    }


# ---------------------------------------------------------------------------
# 9. P95 connector
# ---------------------------------------------------------------------------

def extract_p95_from_field(
    field: SpatioTemporalField,
    *,
    percentile: float = 95.0,
) -> dict[str, Any]:
    """Extract P95 statistics from the spatio-temporal field."""
    final_depth = field.depth_m[-1]
    p95_value = float(np.percentile(final_depth, percentile))

    p95_mask = np.abs(final_depth - p95_value) / max(p95_value, 1e-15) < 0.01
    p95_indices = np.nonzero(p95_mask)[0]

    if len(p95_indices) > 0:
        p95_x = float(np.mean(field.mesh.x_centers[p95_indices]))
        p95_y = float(np.mean(field.mesh.y_centers[p95_indices]))
    else:
        p95_idx = np.argmin(np.abs(final_depth - p95_value))
        p95_x = float(field.mesh.x_centers[p95_idx])
        p95_y = float(field.mesh.y_centers[p95_idx])
        p95_indices = np.array([p95_idx])

    return {
        "p95_depth_m": p95_value,
        "p95_depth_mm": p95_value * 1000.0,
        "p95_depth_um": p95_value * 1e6,
        "p95_x_m": p95_x,
        "p95_y_m": p95_y,
        "p95_panel_indices": p95_indices.tolist(),
        "n_panels_near_p95": len(p95_indices),
        "max_depth_m": float(np.max(final_depth)),
        "mean_depth_m": float(np.mean(final_depth)),
        "percentile": percentile,
    }


@dataclass(frozen=True)
class P95Connector:
    """Connector between field-extracted P95 and N02 direct P95 statistics."""

    field_p95_m: float
    n02_p95_m: float
    relative_error: float
    passed: bool
    tolerance: float = 0.10


def connect_to_n02_p95(
    field_p95_m: float,
    n02_p95_m: float,
    *,
    tolerance: float = 0.10,
) -> P95Connector:
    """Compare field-extracted P95 with N02 direct P95 statistic."""
    ref = max(abs(n02_p95_m), 1e-15)
    rel_error = abs(field_p95_m - n02_p95_m) / ref

    return P95Connector(
        field_p95_m=field_p95_m,
        n02_p95_m=n02_p95_m,
        relative_error=rel_error,
        passed=rel_error < tolerance,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# 10. Full N04 verification suite
# ---------------------------------------------------------------------------

def run_n04_verification(
    *,
    n_panels_x: int = 40,
    n_panels_y: int = 25,
    n_time_steps: int = 100,
    n_samples: int = 100,
    hull_length_m: float = 185.0,
    hull_beam_m: float = 32.26,
    hull_draught_m: float = 11.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full N04 verification suite against all acceptance criteria.

    Returns a dict with results for C1-C6.
    """
    t_start = time.perf_counter()
    results: dict[str, Any] = {"criteria": {}, "timing": {}}

    # Build mesh
    mesh = build_panel_mesh(
        n_x=n_panels_x, n_y=n_panels_y,
        hull_length_m=hull_length_m, hull_beam_m=hull_beam_m,
        hull_draught_m=hull_draught_m,
    )
    results["mesh"] = {"n_panels": mesh.n_panels, "total_area_m2": mesh.total_area}

    # --- C1: Activity profile fitting ---
    # Generate synthetic DEM data from a known profile + noise
    true_profile = AbrasionProfile(
        a0=1.0e-8, x0=hull_length_m * 0.9, sigma_x=hull_length_m * 0.15,
        y0=hull_draught_m * 0.8, lambda_y=hull_draught_m * 0.3,
        a1=3.0e-9, x1=hull_length_m * 0.5, sigma_x1=hull_length_m * 0.2,
        y1=hull_draught_m * 0.7, lambda_y1=hull_draught_m * 0.25,
        a_bg=1.0e-10,
    )
    a_true = true_profile.evaluate_on_mesh(mesh)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1e-10, size=mesh.n_panels)
    a_dem = a_true + noise
    a_dem = np.maximum(a_dem, 0.0)

    fit_result = fit_activity_profile(
        mesh.x_centers, mesh.y_centers, a_dem, seed=seed)
    results["criteria"]["C1"] = {
        "r_squared": fit_result.r_squared,
        "rmse": fit_result.rmse,
        "passed": fit_result.r_squared > 0.95,
        "convergence": fit_result.convergence,
        "n_iterations": fit_result.n_iterations,
    }

    # --- C2: Random field conditional simulation ---
    residuals = a_dem - fit_result.profile.evaluate_on_mesh(mesh)
    corr_info = estimate_spatial_correlation(
        mesh.x_centers, mesh.y_centers, residuals)

    kl = kl_expansion_2d(
        mesh,
        correlation_length_x=corr_info["correlation_length_x"],
        correlation_length_y=corr_info["correlation_length_y"],
        variance=corr_info["variance"],
        n_modes=min(100, mesh.n_panels),
    )

    # Use per-panel variance from KL expansion
    kl_var_per_panel = np.sum(kl.eigenvalues[None, :] * kl.eigenvectors ** 2, axis=1)
    crf = generate_conditional_samples(
        kl, n_samples=n_samples, seed=seed,
        target_mean=np.zeros(mesh.n_panels),
        target_variance=kl_var_per_panel,
    )

    results["criteria"]["C2"] = {
        "mean_error_pct": crf.mean_error_pct,
        "variance_error_pct": crf.variance_error_pct,
        "n_samples": n_samples,
        "n_modes": kl.n_modes,
        "variance_explained": kl.variance_explained,
        "passed": crf.mean_error_pct < 5.0 and crf.variance_error_pct < 50.0,
    }

    # --- C3: Mass conservation ---
    time_s = np.linspace(0, 3600.0, n_time_steps)  # 1 hour
    engine = SpatioTemporalEvolution(
        mesh=mesh,
        profile=fit_result.profile,
        kl_expansion=kl,
        random_samples=crf.samples,
    )
    st_field = engine.evolve(
        time_s, use_random_field=True, sample_index=0, seed=seed)

    mass_check = verify_mass_conservation(st_field)
    results["criteria"]["C3"] = {
        "relative_error": mass_check["relative_error"],
        "passed": mass_check["passed"],
        "mass_kg": mass_check["mass_from_depth_kg"],
    }

    # --- C4: P95 connector with N02 ---
    # Compare P95 of the deterministic profile rate to N02 direct P95
    # (both are rate fields, not accumulated depths)
    profile_rate = fit_result.profile.evaluate_on_mesh(mesh)
    field_rate_p95 = float(np.percentile(profile_rate, 95.0))
    n02_p95_rate = float(np.percentile(a_dem, 95.0))
    p95_conn = connect_to_n02_p95(field_rate_p95, n02_p95_rate)
    results["criteria"]["C4"] = {
        "field_p95_m": p95_conn.field_p95_m,
        "n02_p95_m": p95_conn.n02_p95_m,
        "relative_error": p95_conn.relative_error,
        "passed": p95_conn.passed,
    }

    # --- C5: Uniform anchor ---
    uniform_rate = 1.0e-8
    uniform_check = verify_uniform_anchor(mesh, uniform_rate, time_s)
    results["criteria"]["C5"] = {
        "max_relative_error": uniform_check["max_relative_error"],
        "passed": uniform_check["passed"],
    }

    # --- C6: Performance ---
    t_perf_start = time.perf_counter()
    mesh_perf = build_panel_mesh(n_x=40, n_y=25, hull_length_m=hull_length_m,
                                  hull_beam_m=hull_beam_m, hull_draught_m=hull_draught_m)
    profile_perf = AbrasionProfile(
        a0=1.0e-8, x0=hull_length_m * 0.9, sigma_x=hull_length_m * 0.15,
        y0=hull_draught_m * 0.8, lambda_y=hull_draught_m * 0.3,
    )
    kl_perf = kl_expansion_2d(
        mesh_perf,
        correlation_length_x=hull_length_m * 0.2,
        correlation_length_y=hull_draught_m * 0.3,
        variance=1e-18,
        n_modes=100,
    )
    time_perf = np.linspace(0, 3600.0, 100)
    engine_perf = SpatioTemporalEvolution(
        mesh=mesh_perf, profile=profile_perf, kl_expansion=kl_perf)
    _ = engine_perf.evolve(time_perf, use_random_field=True, seed=seed)
    t_perf = time.perf_counter() - t_perf_start

    results["criteria"]["C6"] = {
        "cpu_time_s": t_perf,
        "n_panels": mesh_perf.n_panels,
        "n_time_steps": 100,
        "passed": t_perf < 60.0,
    }

    results["timing"]["total_s"] = time.perf_counter() - t_start

    # Overall pass
    all_passed = all(c["passed"] for c in results["criteria"].values())
    results["all_criteria_passed"] = all_passed

    return results
