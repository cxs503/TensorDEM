"""Sensitivity analysis for ice resistance uncertainty.

Sobol indices decompose the variance of the output (resistance R) into
contributions from each input parameter (mu, sigma_c) and their interactions.

For a model R = f(mu, sigma_c):
    Var(R) = V_mu + V_sigma + V_mu_sigma
    
where:
- V_mu = variance due to mu alone (first-order Sobol index S_mu = V_mu / Var(R))
- V_sigma = variance due to sigma_c alone (S_sigma = V_sigma / Var(R))
- V_mu_sigma = variance due to interaction (S_mu_sigma = V_mu_sigma / Var(R))

For the linear model R = sigma_c * (a + b * mu):
- If mu and sigma_c are independent:
    Var(R) = (a + b*E[mu])^2 * Var(sigma_c) + b^2 * E[sigma_c]^2 * Var(mu)
           + b^2 * Var(sigma_c) * Var(mu)
- The interaction term is b^2 * Var(sigma_c) * Var(mu)

Methods:
- Analytical: for linear models, exact Sobol indices can be computed
- Monte Carlo estimation: Saltelli's method for general models
- Morris screening: elementary effects for screening important parameters
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
from .distributions import FrictionDistribution
from .surrogate import ResistanceSurrogate


@dataclass
class SobolAnalysis:
    """Sobol sensitivity analysis results.
    
    Attributes
    ----------
    s_mu : float
        First-order Sobol index for mu (fraction of variance due to mu)
    s_sigma : float
        First-order Sobol index for sigma_c
    s_mu_sigma : float
        Interaction index (mu × sigma_c)
    st_mu : float
        Total Sobol index for mu (includes interactions)
    st_sigma : float
        Total Sobol index for sigma_c
    var_total : float
        Total variance of resistance [kN^2]
    var_mu : float
        Variance contribution from mu [kN^2]
    var_sigma : float
        Variance contribution from sigma_c [kN^2]
    var_interaction : float
        Variance contribution from interaction [kN^2]
    """
    s_mu: float
    s_sigma: float
    s_mu_sigma: float
    st_mu: float
    st_sigma: float
    var_total: float
    var_mu: float
    var_sigma: float
    var_interaction: float
    
    def ranking(self) -> list[tuple[str, float]]:
        """Return parameter ranking by first-order Sobol index.
        
        Returns
        -------
        list of (str, float)
            Sorted list of (parameter_name, sobol_index)
        """
        indices = [
            ('mu', self.s_mu),
            ('sigma_c', self.s_sigma),
        ]
        return sorted(indices, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'first_order': {
                'mu': self.s_mu,
                'sigma_c': self.s_sigma,
            },
            'interaction': {
                'mu_sigma': self.s_mu_sigma,
            },
            'total': {
                'mu': self.st_mu,
                'sigma_c': self.st_sigma,
            },
            'variance_decomposition': {
                'total_kN2': self.var_total,
                'mu_kN2': self.var_mu,
                'sigma_kN2': self.var_sigma,
                'interaction_kN2': self.var_interaction,
            },
            'ranking': self.ranking(),
        }


def compute_sobol_indices(
    surrogate: ResistanceSurrogate,
    mu_dist: FrictionDistribution,
    sigma_c_mean: float,
    sigma_c_std: float,
) -> SobolAnalysis:
    """Compute Sobol indices analytically for the linear surrogate model.
    
    For R = sigma_c * (a + b * mu) with independent mu and sigma_c:
    
    E[R] = E[sigma_c] * (a + b * E[mu])
    Var[R] = (a + b*E[mu])^2 * Var(sigma_c) + b^2 * E[sigma_c]^2 * Var(mu)
             + b^2 * Var(sigma_c) * Var(mu)
    
    The three terms are:
    1. Variance due to sigma_c alone: (a + b*E[mu])^2 * Var(sigma_c)
    2. Variance due to mu alone: b^2 * E[sigma_c]^2 * Var(mu)
    3. Interaction variance: b^2 * Var(sigma_c) * Var(mu)
    
    Parameters
    ----------
    surrogate : ResistanceSurrogate
        Surrogate model with coefficients a, b
    mu_dist : FrictionDistribution
        Distribution of friction coefficient
    sigma_c_mean : float
        Mean compressive strength [Pa]
    sigma_c_std : float
        Standard deviation of compressive strength [Pa]
        
    Returns
    -------
    SobolAnalysis
        Sobol indices and variance decomposition
    """
    a = surrogate.a
    b = surrogate.b
    
    # Moments of mu
    mu_mean = mu_dist.mean
    mu_var = mu_dist.std ** 2
    
    # Moments of sigma_c
    sigma_mean = sigma_c_mean
    sigma_var = sigma_c_std ** 2
    
    # Variance decomposition
    # Term 1: variance due to sigma_c alone
    var_sigma = (a + b * mu_mean)**2 * sigma_var
    
    # Term 2: variance due to mu alone
    var_mu = b**2 * sigma_mean**2 * mu_var
    
    # Term 3: interaction variance
    var_interaction = b**2 * sigma_var * mu_var
    
    # Total variance
    var_total = var_sigma + var_mu + var_interaction
    
    # Sobol indices
    if var_total > 0:
        s_sigma = var_sigma / var_total
        s_mu = var_mu / var_total
        s_mu_sigma = var_interaction / var_total
        
        # Total indices (first-order + interactions)
        st_sigma = s_sigma + s_mu_sigma
        st_mu = s_mu + s_mu_sigma
    else:
        s_sigma = s_mu = s_mu_sigma = 0.0
        st_sigma = st_mu = 0.0
    
    return SobolAnalysis(
        s_mu=float(s_mu),
        s_sigma=float(s_sigma),
        s_mu_sigma=float(s_mu_sigma),
        st_mu=float(st_mu),
        st_sigma=float(st_sigma),
        var_total=float(var_total),
        var_mu=float(var_mu),
        var_sigma=float(var_sigma),
        var_interaction=float(var_interaction),
    )


def compute_sobol_mc(
    surrogate: ResistanceSurrogate,
    mu_dist: FrictionDistribution,
    sigma_c_mean: float,
    sigma_c_std: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> SobolAnalysis:
    """Compute Sobol indices via Monte Carlo estimation.
    
    Uses Saltelli's method to estimate first-order and total indices.
    
    Parameters
    ----------
    surrogate : ResistanceSurrogate
        Surrogate model
    mu_dist : FrictionDistribution
        Distribution of mu
    sigma_c_mean : float
        Mean compressive strength [Pa]
    sigma_c_std : float
        Standard deviation of compressive strength [Pa]
    n_samples : int
        Number of samples for estimation
    seed : int
        Random seed
        
    Returns
    -------
    SobolAnalysis
        Estimated Sobol indices
    """
    rng = np.random.RandomState(seed)
    
    # Sample mu and sigma_c
    mu_a = mu_dist.sample(n_samples, seed=seed)
    mu_a = np.maximum(mu_a, 0.0)
    
    sigma_a = rng.normal(sigma_c_mean, sigma_c_std, n_samples)
    sigma_a = np.maximum(sigma_a, 0.0)
    
    mu_b = mu_dist.sample(n_samples, seed=seed + 1)
    mu_b = np.maximum(mu_b, 0.0)
    
    sigma_b = rng.normal(sigma_c_mean, sigma_c_std, n_samples)
    sigma_b = np.maximum(sigma_b, 0.0)
    
    # Evaluate model on different sample combinations
    # R_a = f(mu_a, sigma_a)
    r_a = surrogate.predict(mu_a, sigma_a)
    
    # R_b = f(mu_b, sigma_b)
    r_b = surrogate.predict(mu_b, sigma_b)
    
    # R_ab_mu = f(mu_a, sigma_b) - swap sigma
    r_ab_sigma = surrogate.predict(mu_a, sigma_b)
    
    # R_ab_sigma = f(mu_b, sigma_a) - swap mu
    r_ba_mu = surrogate.predict(mu_b, sigma_a)
    
    # Total variance
    var_total = np.var(np.concatenate([r_a, r_b]))
    
    # First-order index for mu: V[E[R|mu]] / V[R]
    # Estimate E[R|mu] by averaging over sigma for fixed mu
    # Using Jansen's estimator for total index
    var_e_r_given_mu = np.mean((r_a - r_b) ** 2) / 2
    s_mu = 1.0 - var_e_r_given_mu / var_total if var_total > 0 else 0.0
    
    # First-order index for sigma_c
    var_e_r_given_sigma = np.mean((r_a - r_ba_mu) ** 2) / 2
    s_sigma = 1.0 - var_e_r_given_sigma / var_total if var_total > 0 else 0.0
    
    # Total indices
    var_total_mu = np.mean((r_a - r_ab_sigma) ** 2) / 2
    st_mu = var_total_mu / var_total if var_total > 0 else 0.0
    
    var_total_sigma = np.mean((r_a - r_ba_mu) ** 2) / 2
    st_sigma = var_total_sigma / var_total if var_total > 0 else 0.0
    
    # Interaction
    s_mu_sigma = st_mu - s_mu
    
    # Variance decomposition
    var_mu = s_mu * var_total
    var_sigma_comp = s_sigma * var_total
    var_interaction = s_mu_sigma * var_total
    
    return SobolAnalysis(
        s_mu=float(np.clip(s_mu, 0, 1)),
        s_sigma=float(np.clip(s_sigma, 0, 1)),
        s_mu_sigma=float(np.clip(s_mu_sigma, 0, 1)),
        st_mu=float(np.clip(st_mu, 0, 1)),
        st_sigma=float(np.clip(st_sigma, 0, 1)),
        var_total=float(var_total),
        var_mu=float(var_mu),
        var_sigma=float(var_sigma_comp),
        var_interaction=float(var_interaction),
    )
