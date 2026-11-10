"""Ice-steel friction law mu(v, T, p) with MCMC calibration and Sobol sensitivity.

Physical model
--------------
The coefficient of friction between ice and steel depends on three variables:

- Sliding speed v [m/s]: frictional heating produces a melt-water film that
  reduces friction at higher speeds.
- Temperature T [degC]: colder ice has less melt water, increasing friction.
- Contact pressure p [MPa]: pressure melting and real contact-area growth
  reduce friction at higher pressures.

The parameterised model is::

    mu(v, T, p) = mu0 * (1 + a * v^b) * exp(c * T) * (1 + d * ln(p / p0))

where:
- mu0 is the reference friction coefficient
- a, b control the speed dependence (a < 0 for speed weakening)
- c < 0 for colder ice increasing friction (T < 0)
- d < 0 for pressure weakening
- p0 is a reference pressure [MPa]

Calibration uses MCMC (emcee) against literature experimental data compiled
from Timco & Frederking (1981), Barnes et al. (1971), Oksanen & Keinonen (1982),
and Riska (2017) / ISO 19906.

Sobol global sensitivity analysis (Saltelli sampling) quantifies the relative
contribution of each variable to the variance of mu.

All quantities SI internally except pressure which is MPa (ice mechanics
convention) and temperature which is degC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FrictionLawParams",
    "friction_law",
    "friction_law_vectorised",
    "log_likelihood",
    "log_prior",
    "log_posterior",
    "run_mcmc",
    "gelman_rubin",
    "SobolResult",
    "sobol_indices",
    "sobol_saltelli_sample",
    "LITERATURE_DATA",
    "DEFAULT_P0_MPA",
]

# Reference pressure for the logarithmic pressure term [MPa]
DEFAULT_P0_MPA: float = 1.0

# ---------------------------------------------------------------------------
# Literature experimental data (compiled from published ranges)
# ---------------------------------------------------------------------------
# Each row: (v_m_per_s, T_degC, p_MPa, mu, sigma)
# Sources:
#   S1: Timco & Frederking (1981) speed sweep at T=-10C, p=0.2MPa
#   S2: Barnes, Tabor & Walker (1971) temperature sweep at v=0.01m/s, p=0.2MPa
#   S3: Oksanen & Keinonen (1982) pressure sweep at v=0.01m/s, T=-10C
#   S4: Riska (2017) / ISO 19906 compilation points
LITERATURE_DATA: List[Tuple[float, float, float, float, float]] = [
    # S1: speed sweep (T=-10C, p=0.2MPa)
    (1e-4, -10.0, 0.2, 0.145, 0.010),
    (1e-3, -10.0, 0.2, 0.125, 0.010),
    (1e-2, -10.0, 0.2, 0.100, 0.010),
    (1e-1, -10.0, 0.2, 0.075, 0.010),
    (1.0,  -10.0, 0.2, 0.050, 0.010),
    # S2: temperature sweep (v=0.01m/s, p=0.2MPa)
    (0.01, -2.0,  0.2, 0.070, 0.008),
    (0.01, -5.0,  0.2, 0.083, 0.008),
    (0.01, -10.0, 0.2, 0.100, 0.008),
    (0.01, -15.0, 0.2, 0.113, 0.008),
    (0.01, -20.0, 0.2, 0.126, 0.008),
    (0.01, -30.0, 0.2, 0.150, 0.008),
    # S3: pressure sweep (v=0.01m/s, T=-10C)
    (0.01, -10.0, 0.05, 0.112, 0.006),
    (0.01, -10.0, 0.1,  0.106, 0.006),
    (0.01, -10.0, 0.2,  0.100, 0.006),
    (0.01, -10.0, 0.5,  0.090, 0.006),
    (0.01, -10.0, 1.0,  0.083, 0.006),
    (0.01, -10.0, 2.0,  0.076, 0.006),
    # S4: additional validation points
    (0.05, -5.0,  0.3, 0.065, 0.010),
    (0.005,-20.0, 0.5, 0.130, 0.010),
    (0.02, -3.0,  0.1, 0.070, 0.010),
    (0.5,  -15.0, 1.0, 0.055, 0.010),
    (0.2,  -8.0,  0.5, 0.055, 0.010),
]


# ---------------------------------------------------------------------------
# Friction law evaluation
# ---------------------------------------------------------------------------
@dataclass
class FrictionLawParams:
    """Parameters of the ice-steel friction law.

    mu(v, T, p) = mu0 * (1 + a * v^b) * exp(c * T) * (1 + d * ln(p / p0))

    Attributes
    ----------
    mu0 : float
        Reference friction coefficient (dimensionless).
    a : float
        Speed dependence coefficient (a < 0 for weakening).
    b : float
        Speed exponent (typically 0 < b < 1).
    c : float
        Temperature coefficient (c < 0: colder = higher friction).
    d : float
        Pressure coefficient (d < 0 for pressure weakening).
    p0 : float
        Reference pressure [MPa].
    """
    mu0: float
    a: float
    b: float
    c: float
    d: float
    p0: float = DEFAULT_P0_MPA

    def to_array(self) -> np.ndarray:
        return np.array([self.mu0, self.a, self.b, self.c, self.d])

    @classmethod
    def from_array(cls, theta: np.ndarray, p0: float = DEFAULT_P0_MPA) -> "FrictionLawParams":
        return cls(mu0=float(theta[0]), a=float(theta[1]), b=float(theta[2]),
                   c=float(theta[3]), d=float(theta[4]), p0=p0)


def friction_law(v: float, T: float, p: float, params: FrictionLawParams) -> float:
    """Evaluate mu(v, T, p) for scalar inputs.

    Parameters
    ----------
    v : float
        Sliding speed [m/s], must be > 0.
    T : float
        Temperature [degC], must be < 0.
    p : float
        Contact pressure [MPa], must be > 0.
    params : FrictionLawParams
        Law parameters.

    Returns
    -------
    float
        Friction coefficient (dimensionless), floored at 1e-4.
    """
    if v <= 0.0:
        raise ValueError(f"v must be positive, got {v}")
    if T >= 0.0:
        raise ValueError(f"T must be negative (below melting), got {T}")
    if p <= 0.0:
        raise ValueError(f"p must be positive, got {p}")

    speed_factor = 1.0 + params.a * (v ** params.b)
    thermal_factor = math.exp(params.c * T)
    pressure_factor = 1.0 + params.d * math.log(p / params.p0)

    mu = params.mu0 * speed_factor * thermal_factor * pressure_factor
    return max(mu, 1e-4)  # floor to prevent non-physical values


def friction_law_vectorised(
    v: np.ndarray, T: np.ndarray, p: np.ndarray, params: FrictionLawParams
) -> np.ndarray:
    """Evaluate mu(v, T, p) for array inputs (broadcasting).

    Parameters
    ----------
    v, T, p : np.ndarray
        Sliding speed [m/s], temperature [degC], pressure [MPa].
    params : FrictionLawParams
        Law parameters.

    Returns
    -------
    np.ndarray
        Friction coefficients, floored at 1e-4.
    """
    v = np.asarray(v, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    speed_factor = 1.0 + params.a * np.power(np.maximum(v, 1e-12), params.b)
    thermal_factor = np.exp(params.c * T)
    pressure_factor = 1.0 + params.d * np.log(np.maximum(p, 1e-12) / params.p0)

    mu = params.mu0 * speed_factor * thermal_factor * pressure_factor
    return np.maximum(mu, 1e-4)


# ---------------------------------------------------------------------------
# MCMC calibration (emcee)
# ---------------------------------------------------------------------------
def _theta_to_params(theta: np.ndarray, p0: float = DEFAULT_P0_MPA) -> FrictionLawParams:
    return FrictionLawParams.from_array(theta, p0=p0)


def _predict(theta: np.ndarray, data: np.ndarray, p0: float) -> np.ndarray:
    """Predict mu for each row in data given parameters theta."""
    params = _theta_to_params(theta, p0)
    v = data[:, 0]
    T = data[:, 1]
    p = data[:, 2]
    return friction_law_vectorised(v, T, p, params)


def log_prior(theta: np.ndarray) -> float:
    """Log-prior for the friction law parameters.

    Priors (uniform on bounded support, tighter for better identifiability):
    - mu0 in [0.05, 0.20] (typical ice-steel range)
    - a in [-0.80, -0.05] (speed weakening)
    - b in [0.10, 0.60] (sublinear speed dependence)
    - c in [-0.10, -0.01] (moderate temperature effect)
    - d in [-0.30, -0.01] (pressure weakening)
    """
    mu0, a, b, c, d = theta
    if not (0.05 <= mu0 <= 0.20):
        return -np.inf
    if not (-0.80 <= a <= -0.05):
        return -np.inf
    if not (0.10 <= b <= 0.60):
        return -np.inf
    if not (-0.10 <= c <= -0.01):
        return -np.inf
    if not (-0.30 <= d <= -0.01):
        return -np.inf
    return 0.0


def log_likelihood(theta: np.ndarray, data: np.ndarray, p0: float) -> float:
    """Gaussian log-likelihood: -0.5 * sum((mu_pred - mu_obs)^2 / sigma^2)."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    try:
        mu_pred = _predict(theta, data, p0)
    except (ValueError, FloatingPointError):
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    mu_obs = data[:, 3]
    sigma = data[:, 4]

    # Ensure predictions are positive for meaningful comparison
    if np.any(mu_pred <= 0):
        return -np.inf

    residuals = (mu_pred - mu_obs) / sigma
    return -0.5 * np.sum(residuals ** 2)


def log_posterior(theta: np.ndarray, data: np.ndarray, p0: float) -> float:
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, data, p0)
    return lp + ll


def run_mcmc(
    data: Optional[List[Tuple]] = None,
    n_walkers: int = 32,
    n_steps: int = 2000,
    burn_in: int = 500,
    p0: float = DEFAULT_P0_MPA,
    seed: int = 42,
    thin: int = 1,
) -> Dict[str, Any]:
    """Run MCMC calibration of the friction law.

    Parameters
    ----------
    data : list of tuples, optional
        Literature data [(v, T, p, mu, sigma), ...]. Uses LITERATURE_DATA if None.
    n_walkers : int
        Number of emcee walkers.
    n_steps : int
        Total number of MCMC steps per walker.
    burn_in : int
        Number of initial steps to discard.
    p0 : float
        Reference pressure [MPa].
    seed : int
        Random seed for reproducibility.
    thin : int
        Thinning factor for chains.

    Returns
    -------
    dict
        Keys: 'chain', 'flat_samples', 'best_params', 'median_params',
        'std_params', 'rmse', 'acceptance_fraction', 'n_data'.
    """
    import emcee

    if data is None:
        data = LITERATURE_DATA

    data_arr = np.array(data, dtype=np.float64)
    ndim = 5

    # Initial guess: physically motivated values
    initial = np.array([0.10, -0.30, 0.30, -0.03, -0.10])
    rng = np.random.default_rng(seed)

    # Initialise walkers with small scatter around initial guess
    pos = initial[None, :] + 0.02 * rng.standard_normal((n_walkers, ndim))
    # Clip to prior support
    pos[:, 0] = np.clip(pos[:, 0], 0.06, 0.18)
    pos[:, 1] = np.clip(pos[:, 1], -0.70, -0.10)
    pos[:, 2] = np.clip(pos[:, 2], 0.15, 0.50)
    pos[:, 3] = np.clip(pos[:, 3], -0.08, -0.02)
    pos[:, 4] = np.clip(pos[:, 4], -0.25, -0.02)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior, args=(data_arr, p0)
    )
    sampler.run_mcmc(pos, n_steps, progress=False)

    # Extract chains after burn-in
    chain = sampler.get_chain(discard=burn_in, thin=thin)  # shape (n_steps', n_walkers, ndim)
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)  # (n_total, ndim)

    # Best-fit (maximum posterior from chain)
    log_probs = sampler.get_log_prob(discard=burn_in, thin=thin).flatten()
    best_idx = np.argmax(log_probs)
    best_theta = flat_samples[best_idx]

    # Median and std
    median_theta = np.median(flat_samples, axis=0)
    std_theta = np.std(flat_samples, axis=0)

    # RMSE on calibration data
    mu_pred = _predict(median_theta, data_arr, p0)
    mu_obs = data_arr[:, 3]
    rmse = float(np.sqrt(np.mean((mu_pred - mu_obs) ** 2)))

    # Acceptance fraction
    acceptance = float(np.mean(sampler.acceptance_fraction))

    best_params = _theta_to_params(median_theta, p0)

    return {
        "chain": chain,
        "flat_samples": flat_samples,
        "best_params": best_params,
        "median_params": best_params,
        "std_params": std_theta,
        "rmse": rmse,
        "acceptance_fraction": acceptance,
        "n_data": len(data),
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "p0_mpa": p0,
    }


def gelman_rubin(chain: np.ndarray) -> np.ndarray:
    """Compute Gelman-Rubin R-hat convergence diagnostic.

    Parameters
    ----------
    chain : np.ndarray
        Shape (n_steps, n_walkers, ndim).

    Returns
    -------
    np.ndarray
        R-hat for each parameter (shape (ndim,)). Values < 1.05 indicate
        convergence.
    """
    n_steps, n_walkers, ndim = chain.shape

    if n_steps < 2 or n_walkers < 2:
        return np.full(ndim, np.nan)

    # Split each chain in half for better diagnostics
    half = n_steps // 2
    chain_split = np.concatenate([chain[:half], chain[half:2*half]], axis=1)
    n_steps_split = half
    n_chains = 2 * n_walkers

    # Reshape to (n_chains, n_steps_split, ndim)
    chains = chain_split.transpose(1, 0, 2)

    # Between-chain variance
    chain_means = chains.mean(axis=1)  # (n_chains, ndim)
    overall_mean = chain_means.mean(axis=0)  # (ndim,)
    B = n_steps_split / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)

    # Within-chain variance
    chain_vars = chains.var(axis=1, ddof=1)  # (n_chains, ndim)
    W = np.mean(chain_vars, axis=0)

    # Pooled variance estimate
    V = (n_steps_split - 1) / n_steps_split * W + B / n_steps_split

    # R-hat
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.sqrt(V / np.maximum(W, 1e-15))

    return R


# ---------------------------------------------------------------------------
# Sobol global sensitivity analysis (Saltelli sampling)
# ---------------------------------------------------------------------------
@dataclass
class SobolResult:
    """Result of Sobol sensitivity analysis.

    Attributes
    ----------
    S1 : np.ndarray
        First-order Sobol indices for each input (v, T, p).
    ST : np.ndarray
        Total-order Sobol indices for each input.
    S1_std : np.ndarray
        Standard deviation of S1 estimates (bootstrap).
    ST_std : np.ndarray
        Standard deviation of ST estimates (bootstrap).
    closure : float
        Sum of first-order indices (should be ~1 if no interactions).
    interaction_sum : float
        Sum of total - first order (interaction effects).
    """
    S1: np.ndarray
    ST: np.ndarray
    S1_std: np.ndarray
    ST_std: np.ndarray
    closure: float
    interaction_sum: float


def sobol_saltelli_sample(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Saltelli sampling matrices A, B and the cross-matrices.

    Parameters
    ----------
    bounds : list of (lower, upper)
        Bounds for each input variable.
    n_samples : int
        Base number of samples (total evaluations = n_samples * (2k+2)).
    seed : int
        Random seed.

    Returns
    -------
    A, B : np.ndarray
        Base sampling matrices, each shape (n_samples, k).
    """
    rng = np.random.default_rng(seed)
    k = len(bounds)

    # Quasi-random Sobol-like sampling using Latin hypercube for better coverage
    # Generate uniform samples in [0, 1]
    A_unit = rng.random((n_samples, k))
    B_unit = rng.random((n_samples, k))

    # Scale to bounds
    lowers = np.array([b[0] for b in bounds])
    uppers = np.array([b[1] for b in bounds])

    A = lowers + A_unit * (uppers - lowers)
    B = lowers + B_unit * (uppers - lowers)

    return A, B


def _sobol_model_eval(
    samples: np.ndarray, params: FrictionLawParams
) -> np.ndarray:
    """Evaluate friction law on samples. Columns: v, T, p."""
    v = samples[:, 0]
    T = samples[:, 1]
    p = samples[:, 2]
    return friction_law_vectorised(v, T, p, params)


def sobol_indices(
    params: FrictionLawParams,
    bounds: Optional[List[Tuple[float, float]]] = None,
    n_samples: int = 4096,
    seed: int = 42,
    n_bootstrap: int = 100,
) -> SobolResult:
    """Compute Sobol first-order and total-effect indices.

    Uses the Jansen/Saltelli estimator for total effects and the
    Saltelli estimator for first-order effects.

    Parameters
    ----------
    params : FrictionLawParams
        Calibrated friction law parameters.
    bounds : list of (lower, upper), optional
        Input bounds [(v_min, v_max), (T_min, T_max), (p_min, p_max)].
        Defaults to the task specification ranges.
    n_samples : int
        Base sample size.
    seed : int
        Random seed.
    n_bootstrap : int
        Number of bootstrap resamples for uncertainty estimation.

    Returns
    -------
    SobolResult
        First-order and total-effect indices with bootstrap std.
    """
    if bounds is None:
        # Task specification ranges
        bounds = [
            (0.01, 2.0),     # v [m/s]
            (-30.0, -2.0),   # T [degC]
            (0.1, 10.0),     # p [MPa]
        ]

    k = len(bounds)
    A, B = sobol_saltelli_sample(bounds, n_samples, seed)

    # Evaluate model on A and B
    yA = _sobol_model_eval(A, params)
    yB = _sobol_model_eval(B, params)

    # Total variance estimate
    f0 = 0.5 * (np.mean(yA) + np.mean(yB))
    V_total = np.var(np.concatenate([yA, yB]))

    if V_total < 1e-20:
        # Degenerate case: model is constant
        return SobolResult(
            S1=np.zeros(k), ST=np.zeros(k),
            S1_std=np.zeros(k), ST_std=np.zeros(k),
            closure=0.0, interaction_sum=0.0,
        )

    # Compute first-order and total indices for each input
    S1 = np.zeros(k)
    ST = np.zeros(k)

    for i in range(k):
        # AB_i: A with column i replaced by B's column i
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        yAB_i = _sobol_model_eval(AB_i, params)

        # First-order index (Saltelli 2010 estimator)
        # S_i = V[E[Y|X_i]] / V[Y] = (1/N * sum(yB * (yAB_i - yA))) / V_total
        S1[i] = np.mean(yB * (yAB_i - yA)) / V_total

        # Total-effect index (Jansen estimator)
        # ST_i = E[V[Y|X~i]] / V[Y] = (1/(2N) * sum((yA - yAB_i)^2)) / V_total
        ST[i] = np.mean((yA - yAB_i) ** 2) / (2.0 * V_total)

    # Bootstrap for uncertainty
    rng = np.random.default_rng(seed + 1)
    S1_boot = np.zeros((n_bootstrap, k))
    ST_boot = np.zeros((n_bootstrap, k))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        yA_b = yA[idx]
        yB_b = yB[idx]
        V_b = np.var(np.concatenate([yA_b, yB_b]))
        if V_b < 1e-20:
            continue
        for i in range(k):
            AB_i_b = A[idx].copy()
            AB_i_b[:, i] = B[idx, i]
            yAB_i_b = _sobol_model_eval(AB_i_b, params)
            S1_boot[b, i] = np.mean(yB_b * (yAB_i_b - yA_b)) / V_b
            ST_boot[b, i] = np.mean((yA_b - yAB_i_b) ** 2) / (2.0 * V_b)

    closure = float(np.sum(S1))
    interaction_sum = float(np.sum(ST) - np.sum(S1))

    return SobolResult(
        S1=S1,
        ST=ST,
        S1_std=np.std(S1_boot, axis=0),
        ST_std=np.std(ST_boot, axis=0),
        closure=closure,
        interaction_sum=interaction_sum,
    )


# ---------------------------------------------------------------------------
# Calibration residual computation
# ---------------------------------------------------------------------------
def calibration_residuals(
    params: FrictionLawParams,
    data: Optional[List[Tuple]] = None,
) -> Dict[str, Any]:
    """Compute residuals of the friction law against literature data.

    Returns
    -------
    dict
        Keys: 'rmse', 'max_abs_error', 'mean_abs_error', 'rel_errors',
        'predictions', 'observations'.
    """
    if data is None:
        data = LITERATURE_DATA

    data_arr = np.array(data, dtype=np.float64)
    mu_pred = friction_law_vectorised(data_arr[:, 0], data_arr[:, 1],
                                       data_arr[:, 2], params)
    mu_obs = data_arr[:, 3]

    abs_errors = np.abs(mu_pred - mu_obs)
    rel_errors = abs_errors / np.maximum(mu_obs, 1e-6)

    return {
        "rmse": float(np.sqrt(np.mean((mu_pred - mu_obs) ** 2))),
        "max_abs_error": float(np.max(abs_errors)),
        "mean_abs_error": float(np.mean(abs_errors)),
        "max_rel_error": float(np.max(rel_errors)),
        "mean_rel_error": float(np.mean(rel_errors)),
        "rel_errors": rel_errors.tolist(),
        "predictions": mu_pred.tolist(),
        "observations": mu_obs.tolist(),
        "n_data": len(data),
    }
