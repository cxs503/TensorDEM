"""Joint uncertainty propagation for mu and sigma_c.

This module couples the friction coefficient uncertainty with the material
strength uncertainty to obtain the combined effect on ice resistance.

Under the independence assumption:
    Var(R) = Var_mu(R) + Var_sigma(R) + Var_interaction(R)

where the interaction term arises from the product structure R = sigma_c * (a + b * mu).

This module provides:
- Joint Monte Carlo sampling
- Analytical variance decomposition under independence
- 3D mapping closure: combining with other uncertainty sources
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from .distributions import FrictionDistribution
from .surrogate import ResistanceSurrogate


@dataclass
class JointUncertainty:
    """Joint uncertainty propagation results.
    
    Attributes
    ----------
    mu_samples : np.ndarray
        Sampled friction coefficients, shape (n_samples,)
    sigma_samples : np.ndarray
        Sampled compressive strengths [Pa], shape (n_samples,)
    r_samples : np.ndarray
        Corresponding resistance values [kN], shape (n_samples,)
    surrogate : ResistanceSurrogate
        Surrogate model used
    mu_dist : FrictionDistribution
        Distribution of mu
    sigma_c_mean : float
        Mean compressive strength [Pa]
    sigma_c_std : float
        Standard deviation of compressive strength [Pa]
    n_samples : int
        Number of samples
    """
    mu_samples: np.ndarray
    sigma_samples: np.ndarray
    r_samples: np.ndarray
    surrogate: ResistanceSurrogate
    mu_dist: FrictionDistribution
    sigma_c_mean: float
    sigma_c_std: float
    n_samples: int
    
    def compute_statistics(self) -> dict[str, float]:
        """Compute joint statistics."""
        return {
            'mean_kN': float(np.mean(self.r_samples)),
            'std_kN': float(np.std(self.r_samples, ddof=1)),
            'median_kN': float(np.median(self.r_samples)),
            'q05_kN': float(np.percentile(self.r_samples, 5)),
            'q95_kN': float(np.percentile(self.r_samples, 95)),
            'ci_95_width_kN': float(np.percentile(self.r_samples, 95) - 
                                    np.percentile(self.r_samples, 5)),
        }
    
    def verify_variance_decomposition(self) -> dict[str, float]:
        """Verify variance decomposition under independence assumption.
        
        For R = sigma_c * (a + b * mu) with independent mu and sigma_c:
        
        Var[R] = E[sigma_c]^2 * b^2 * Var[mu]           (mu contribution)
               + (a + b*E[mu])^2 * Var[sigma_c]         (sigma contribution)
               + b^2 * Var[sigma_c] * Var[mu]           (interaction)
        
        This method verifies that the Monte Carlo estimate matches the
        analytical decomposition within statistical error.
        
        Returns
        -------
        dict
            Verification metrics:
            - var_total_mc: total variance from MC sampling
            - var_total_analytical: analytical total variance
            - var_mu: variance contribution from mu
            - var_sigma: variance contribution from sigma_c
            - var_interaction: interaction variance
            - relative_error: |var_mc - var_analytical| / var_analytical
        """
        # Analytical decomposition
        a = self.surrogate.a
        b = self.surrogate.b
        mu_mean = self.mu_dist.mean
        mu_var = self.mu_dist.std ** 2
        sigma_mean = self.sigma_c_mean
        sigma_var = self.sigma_c_std ** 2
        
        # Variance contributions
        var_mu = sigma_mean**2 * b**2 * mu_var
        var_sigma = (a + b * mu_mean)**2 * sigma_var
        var_interaction = b**2 * sigma_var * mu_var
        var_total_analytical = var_mu + var_sigma + var_interaction
        
        # Monte Carlo estimate
        var_total_mc = np.var(self.r_samples, ddof=1)
        
        # Relative error
        relative_error = abs(var_total_mc - var_total_analytical) / var_total_analytical \
            if var_total_analytical > 0 else 0.0
        
        return {
            'var_total_mc_kN2': float(var_total_mc),
            'var_total_analytical_kN2': float(var_total_analytical),
            'var_mu_kN2': float(var_mu),
            'var_sigma_kN2': float(var_sigma),
            'var_interaction_kN2': float(var_interaction),
            'relative_error': float(relative_error),
            'mu_fraction': float(var_mu / var_total_analytical) if var_total_analytical > 0 else 0.0,
            'sigma_fraction': float(var_sigma / var_total_analytical) if var_total_analytical > 0 else 0.0,
            'interaction_fraction': float(var_interaction / var_total_analytical) if var_total_analytical > 0 else 0.0,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        stats = self.compute_statistics()
        verification = self.verify_variance_decomposition()
        return {
            'n_samples': self.n_samples,
            'mu_distribution': {
                'type': self.mu_dist.dist_type,
                'mean': self.mu_dist.mean,
                'std': self.mu_dist.std,
            },
            'sigma_distribution': {
                'mean_pa': self.sigma_c_mean,
                'std_pa': self.sigma_c_std,
                'cv': self.sigma_c_std / self.sigma_c_mean if self.sigma_c_mean > 0 else 0.0,
            },
            'surrogate': self.surrogate.to_dict(),
            'statistics': stats,
            'variance_decomposition': verification,
        }


def propagate_joint_uncertainty(
    surrogate: ResistanceSurrogate,
    mu_dist: FrictionDistribution,
    sigma_c_mean: float,
    sigma_c_std: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> JointUncertainty:
    """Propagate joint uncertainty in mu and sigma_c.
    
    Parameters
    ----------
    surrogate : ResistanceSurrogate
        Surrogate model R(mu, sigma_c)
    mu_dist : FrictionDistribution
        Distribution of friction coefficient
    sigma_c_mean : float
        Mean compressive strength [Pa]
    sigma_c_std : float
        Standard deviation of compressive strength [Pa]
    n_samples : int
        Number of Monte Carlo samples
    seed : int
        Random seed
        
    Returns
    -------
    JointUncertainty
        Joint propagation results
    """
    # Use different seeds for mu and sigma to ensure independence
    mu_samples = mu_dist.sample(n_samples, seed=seed)
    mu_samples = np.maximum(mu_samples, 0.0)
    
    rng = np.random.RandomState(seed + 1000)  # Different seed for sigma
    sigma_samples = rng.normal(sigma_c_mean, sigma_c_std, n_samples)
    sigma_samples = np.maximum(sigma_samples, 0.0)
    
    # Compute resistance
    r_samples = surrogate.predict(mu_samples, sigma_samples)
    
    return JointUncertainty(
        mu_samples=mu_samples,
        sigma_samples=sigma_samples,
        r_samples=r_samples,
        surrogate=surrogate,
        mu_dist=mu_dist,
        sigma_c_mean=sigma_c_mean,
        sigma_c_std=sigma_c_std,
        n_samples=n_samples,
    )


def three_d_mapping_closure(
    surrogate: ResistanceSurrogate,
    mu_dist: FrictionDistribution,
    sigma_c_mean: float,
    sigma_c_std: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Perform 3D mapping closure analysis.
    
    Combines mu and sigma_c uncertainties and verifies the variance
    decomposition under the independence assumption. This is the "3D mapping"
    mentioned in the task: coupling friction uncertainty with material
    strength uncertainty.
    
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
        Number of samples
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Complete 3D mapping results including:
        - Joint statistics
        - Variance decomposition verification
        - Sensitivity indices
        - Confidence intervals
    """
    from .sensitivity import compute_sobol_indices
    
    # Joint propagation
    joint = propagate_joint_uncertainty(
        surrogate, mu_dist, sigma_c_mean, sigma_c_std, n_samples, seed
    )
    
    # Sobol analysis
    sobol = compute_sobol_indices(surrogate, mu_dist, sigma_c_mean, sigma_c_std)
    
    # Compile results
    results = {
        'joint_propagation': joint.to_dict(),
        'sobol_analysis': sobol.to_dict(),
        'confidence_intervals': {
            '90_percent': {
                'lower_kN': float(np.percentile(joint.r_samples, 5)),
                'upper_kN': float(np.percentile(joint.r_samples, 95)),
            },
            '95_percent': {
                'lower_kN': float(np.percentile(joint.r_samples, 2.5)),
                'upper_kN': float(np.percentile(joint.r_samples, 97.5)),
            },
        },
    }
    
    return results
