"""Monte Carlo propagation of friction coefficient uncertainty to ice resistance.

Given a surrogate model R(mu, sigma_c) and a probability distribution for mu,
we propagate the uncertainty through the model to obtain the distribution of
ice resistance R.

Methods:
- Direct Monte Carlo: sample mu from its distribution, compute R for each sample
- Latin Hypercube Sampling: stratified sampling for better coverage
- Quasi-Monte Carlo: low-discrepancy sequences (Sobol, Halton)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from .distributions import FrictionDistribution
from .surrogate import ResistanceSurrogate


@dataclass
class MonteCarloPropagation:
    """Monte Carlo uncertainty propagation results.
    
    Attributes
    ----------
    mu_samples : np.ndarray
        Sampled friction coefficient values, shape (n_samples,)
    r_samples : np.ndarray
        Corresponding resistance values [kN], shape (n_samples,)
    mu_dist : FrictionDistribution
        Input distribution for mu
    surrogate : ResistanceSurrogate
        Surrogate model used for propagation
    n_samples : int
        Number of Monte Carlo samples
    sigma_c_pa : float
        Compressive strength used [Pa]
    """
    mu_samples: np.ndarray
    r_samples: np.ndarray
    mu_dist: FrictionDistribution
    surrogate: ResistanceSurrogate
    n_samples: int
    sigma_c_pa: float
    
    def compute_statistics(self) -> dict[str, float]:
        """Compute statistics of the resistance distribution.
        
        Returns
        -------
        dict
            Statistics including mean, std, quantiles, CI width
        """
        mean_r = float(np.mean(self.r_samples))
        std_r = float(np.std(self.r_samples, ddof=1))
        median_r = float(np.median(self.r_samples))
        q05 = float(np.percentile(self.r_samples, 5))
        q95 = float(np.percentile(self.r_samples, 95))
        ci_width = q95 - q05
        
        # Coefficient of variation
        cv = std_r / mean_r if mean_r > 0 else 0.0
        
        return {
            'mean_kN': mean_r,
            'std_kN': std_r,
            'median_kN': median_r,
            'q05_kN': q05,
            'q25_kN': float(np.percentile(self.r_samples, 25)),
            'q75_kN': float(np.percentile(self.r_samples, 75)),
            'q95_kN': q95,
            'ci_90_width_kN': float(np.percentile(self.r_samples, 95) - 
                                    np.percentile(self.r_samples, 5)),
            'ci_95_width_kN': ci_width,
            'cv': cv,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        stats = self.compute_statistics()
        return {
            'n_samples': self.n_samples,
            'sigma_c_pa': self.sigma_c_pa,
            'mu_distribution': {
                'type': self.mu_dist.dist_type,
                'mean': self.mu_dist.mean,
                'std': self.mu_dist.std,
            },
            'surrogate': self.surrogate.to_dict(),
            'statistics': stats,
            'samples': {
                'mu': self.mu_samples.tolist(),
                'resistance_kN': self.r_samples.tolist(),
            },
        }


def propagate_uncertainty(
    surrogate: ResistanceSurrogate,
    mu_dist: FrictionDistribution,
    sigma_c_pa: float = 2.477e6,
    n_samples: int = 10000,
    seed: int = 42,
    method: str = 'mc',
) -> MonteCarloPropagation:
    """Propagate mu uncertainty through the surrogate model.
    
    Parameters
    ----------
    surrogate : ResistanceSurrogate
        Surrogate model R(mu, sigma_c)
    mu_dist : FrictionDistribution
        Probability distribution for mu
    sigma_c_pa : float
        Compressive strength [Pa]
    n_samples : int
        Number of Monte Carlo samples
    seed : int
        Random seed
    method : str
        Sampling method: 'mc' (Monte Carlo), 'lhs' (Latin Hypercube)
        
    Returns
    -------
    MonteCarloPropagation
        Propagation results with samples and statistics
    """
    # Sample mu from its distribution
    mu_samples = mu_dist.sample(n_samples, seed=seed)
    
    # Ensure mu is non-negative (physical constraint)
    mu_samples = np.maximum(mu_samples, 0.0)
    
    # Compute resistance for each sample
    sigma_c_array = np.full_like(mu_samples, sigma_c_pa)
    r_samples = surrogate.predict(mu_samples, sigma_c_array)
    
    return MonteCarloPropagation(
        mu_samples=mu_samples,
        r_samples=r_samples,
        mu_dist=mu_dist,
        surrogate=surrogate,
        n_samples=n_samples,
        sigma_c_pa=sigma_c_pa,
    )


def propagate_mu_sweep(
    surrogate: ResistanceSurrogate,
    mu_values: np.ndarray,
    sigma_c_pa: float = 2.477e6,
) -> dict[str, np.ndarray]:
    """Compute resistance for a sweep of mu values (deterministic).
    
    Parameters
    ----------
    surrogate : ResistanceSurrogate
        Surrogate model
    mu_values : np.ndarray
        Array of mu values to test
    sigma_c_pa : float
        Compressive strength [Pa]
        
    Returns
    -------
    dict
        mu_values, r_values arrays
    """
    mu_values = np.asarray(mu_values)
    sigma_c_array = np.full_like(mu_values, sigma_c_pa)
    r_values = surrogate.predict(mu_values, sigma_c_array)
    
    return {
        'mu_values': mu_values,
        'r_values': r_values,
    }
