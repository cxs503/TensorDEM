"""Probability distributions for friction coefficient uncertainty.

Based on experimental data from ice-steel friction tests (Løset et al., 
Frederking & Timco, etc.), the friction coefficient mu for ice-structure
contact exhibits variability due to:
- Surface roughness variations
- Temperature fluctuations
- Ice salinity and brine content
- Contact pressure dependence
- Sliding velocity effects

Common distributions used in literature:
- Normal: mu ~ N(mu_mean, mu_std) for small variability
- Lognormal: for positive-valued parameters with multiplicative uncertainty
- Uniform: when only bounds are known
- Beta: bounded distributions with shape control
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy import stats


@dataclass
class FrictionDistribution:
    """Probability distribution for friction coefficient mu.
    
    Parameters
    ----------
    dist_type : str
        Distribution type: 'normal', 'lognormal', 'uniform', 'beta'
    mean : float
        Mean friction coefficient (for normal/lognormal) or lower bound (for uniform)
    std : float, optional
        Standard deviation (for normal/lognormal). For uniform, computed from bounds.
    lower : float, optional
        Lower bound (for uniform/beta)
    upper : float, optional
        Upper bound (for uniform/beta)
    a : float, optional
        First shape parameter (for beta distribution)
    b : float, optional
        Second shape parameter (for beta distribution)
    
    Notes
    -----
    Literature values for ice-steel friction:
    - Cold ice (-10°C): mu ≈ 0.05-0.15 (Frederking & Timco, 1985)
    - Warm ice (-1°C): mu ≈ 0.10-0.30 
    - Typical engineering default: mu = 0.10 (Løset et al., 2005)
    - Coefficient of variation: 15-25% based on experimental scatter
    """
    dist_type: Literal['normal', 'lognormal', 'uniform', 'beta']
    mean: float = 0.10
    std: float = 0.02
    lower: float | None = None
    upper: float | None = None
    a: float | None = None
    b: float | None = None
    
    def __post_init__(self):
        """Validate and setup distribution."""
        if self.dist_type == 'normal':
            if self.mean <= 0:
                raise ValueError("Normal distribution mean must be positive for friction")
            if self.std <= 0:
                raise ValueError("Normal distribution std must be positive")
            self._dist = stats.norm(loc=self.mean, scale=self.std)
            
        elif self.dist_type == 'lognormal':
            # Parameterize lognormal by mean and std of the distribution
            # (not the underlying normal)
            if self.mean <= 0 or self.std <= 0:
                raise ValueError("Lognormal mean and std must be positive")
            # Convert to log-space parameters
            phi = np.sqrt(self.std**2 + self.mean**2)
            zeta = np.sqrt(np.log(phi**2 / self.mean**2))
            sigma_ln = zeta
            mu_ln = np.log(self.mean) - 0.5 * zeta**2
            self._dist = stats.lognorm(s=sigma_ln, scale=np.exp(mu_ln))
            
        elif self.dist_type == 'uniform':
            if self.lower is None or self.upper is None:
                # Default: mean ± 2*std covers 95% of normal, use as bounds
                self.lower = max(0.0, self.mean - 2 * self.std)
                self.upper = self.mean + 2 * self.std
            if self.lower >= self.upper:
                raise ValueError("Uniform lower must be < upper")
            self._dist = stats.uniform(loc=self.lower, scale=self.upper - self.lower)
            # Compute mean and std for uniform
            self.mean = (self.lower + self.upper) / 2
            self.std = (self.upper - self.lower) / np.sqrt(12)
            
        elif self.dist_type == 'beta':
            if self.a is None or self.b is None:
                self.a = 2.0
                self.b = 5.0
            if self.lower is None or self.upper is None:
                self.lower = 0.0
                self.upper = 0.3
            self._dist = stats.beta(a=self.a, b=self.b, 
                                   loc=self.lower, 
                                   scale=self.upper - self.lower)
            # Compute mean and std
            self.mean = self._dist.mean()
            self.std = self._dist.std()
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")
    
    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        """Sample n values from the distribution.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with friction coefficient values
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        return self._dist.rvs(size=n_samples, random_state=rng)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        return self._dist.pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return self._dist.cdf(x)
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(q)
    
    def stats(self) -> dict[str, float]:
        """Return distribution statistics."""
        return {
            'mean': float(self._dist.mean()),
            'std': float(self._dist.std()),
            'median': float(self._dist.median()),
            'q05': float(self._dist.ppf(0.05)),
            'q95': float(self._dist.ppf(0.95)),
            'cv': float(self._dist.std() / self._dist.mean()),
        }


def sample_mu(
    n_samples: int,
    mu_mean: float = 0.10,
    mu_std: float = 0.02,
    dist_type: str = 'normal',
    seed: int | None = None,
) -> np.ndarray:
    """Convenience function to sample friction coefficients.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    mu_mean : float
        Mean friction coefficient
    mu_std : float
        Standard deviation
    dist_type : str
        Distribution type
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Array of friction coefficient samples
    """
    dist = FrictionDistribution(
        dist_type=dist_type,
        mean=mu_mean,
        std=mu_std,
    )
    return dist.sample(n_samples, seed=seed)
