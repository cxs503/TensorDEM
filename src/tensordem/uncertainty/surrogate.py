"""Physics-based surrogate model for ice resistance.

The DEM contact law gives:
- Normal force: f_n = sigma_c * t * w (crushing-limited)
- Tangential force: f_t = clamp(c_t * v_t, -mu * f_n, mu * f_n)

In steady sliding (most contacts), f_t ≈ mu * f_n, so:
    R(mu, sigma_c) ≈ R_0(sigma_c) + mu * F_friction(sigma_c)

For a linear surrogate:
    R(mu, sigma_c) = sigma_c * (a + b * mu)

where a, b are calibrated from DEM simulations at different (mu, sigma_c) points.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ResistanceSurrogate:
    """Linear response surface for ice resistance.
    
    Model: R(mu, sigma_c) = sigma_c * (a + b * mu) [kN]
    
    where:
    - a: normal-force coefficient [kN/Pa]
    - b: friction-force coefficient [kN/Pa]
    - mu: friction coefficient [-]
    - sigma_c: compressive strength [Pa]
    
    The model captures:
    - R increases linearly with mu (friction contribution)
    - R increases linearly with sigma_c (crushing strength)
    - Cross-term: friction force scales with normal force (which scales with sigma_c)
    """
    a: float  # Normal force coefficient [kN/Pa]
    b: float  # Friction coefficient [kN/Pa]
    r_squared: float = 0.0  # Goodness of fit
    n_calibration: int = 0  # Number of calibration points
    
    def predict(self, mu: np.ndarray, sigma_c: np.ndarray) -> np.ndarray:
        """Predict resistance for given mu and sigma_c.
        
        Parameters
        ----------
        mu : np.ndarray
            Friction coefficient values, shape (n,)
        sigma_c : np.ndarray
            Compressive strength values [Pa], shape (n,)
            
        Returns
        -------
        np.ndarray
            Predicted resistance [kN], shape (n,)
        """
        mu = np.asarray(mu)
        sigma_c = np.asarray(sigma_c)
        return sigma_c * (self.a + self.b * mu)
    
    def predict_scalar(self, mu: float, sigma_c: float) -> float:
        """Predict resistance for scalar inputs."""
        return float(sigma_c * (self.a + self.b * mu))
    
    def predict_mean(self, mu_mean: float, sigma_c_mean: float) -> float:
        """Predict mean resistance for given mean mu and sigma_c."""
        return float(self.predict_scalar(mu_mean, sigma_c_mean))
    
    def predict_stats(self, mu_mean: float, mu_std: float, sigma_c_mean: float) -> dict[str, float]:
        """Predict resistance statistics for a mu distribution at fixed sigma_c.
        
        Parameters
        ----------
        mu_mean : float
            Mean friction coefficient
        mu_std : float
            Standard deviation of friction coefficient
        sigma_c_mean : float
            Mean compressive strength [Pa]
            
        Returns
        -------
        dict
            Statistics: mean, std, q05, q95 [kN]
        """
        # R = sigma_c * (a + b * mu)
        # E[R] = sigma_c * (a + b * E[mu])
        # Var[R] = sigma_c^2 * b^2 * Var[mu]
        mean_r = sigma_c_mean * (self.a + self.b * mu_mean)
        std_r = abs(sigma_c_mean * self.b * mu_std)
        
        # Assuming normal distribution for R (linear transform of mu)
        q05 = mean_r - 1.645 * std_r
        q95 = mean_r + 1.645 * std_r
        
        return {
            'mean_kN': float(mean_r),
            'std_kN': float(std_r),
            'q05_kN': float(q05),
            'q95_kN': float(q95),
        }
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'a': self.a,
            'b': self.b,
            'r_squared': self.r_squared,
            'n_calibration': self.n_calibration,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ResistanceSurrogate':
        """Deserialize from dictionary."""
        return cls(
            a=d['a'],
            b=d['b'],
            r_squared=d.get('r_squared', 0.0),
            n_calibration=d.get('n_calibration', 0),
        )


def calibrate_surrogate(
    mu_values: Sequence[float],
    sigma_c_values: Sequence[float],
    resistance_values: Sequence[float],
) -> ResistanceSurrogate:
    """Calibrate surrogate model from DEM simulation results.
    
    Fits: R = sigma_c * (a + b * mu)
    
    Parameters
    ----------
    mu_values : sequence of float
        Friction coefficient values [-]
    sigma_c_values : sequence of float
        Compressive strength values [Pa]
    resistance_values : sequence of float
        Measured resistance values [kN]
        
    Returns
    -------
    ResistanceSurrogate
        Calibrated surrogate model
    """
    mu = np.asarray(mu_values, dtype=np.float64)
    sigma_c = np.asarray(sigma_c_values, dtype=np.float64)
    R = np.asarray(resistance_values, dtype=np.float64)
    
    # Linear regression: R / sigma_c = a + b * mu
    y = R / sigma_c
    X = np.vstack([np.ones_like(mu), mu]).T
    
    # Least squares: [a, b] = (X^T X)^{-1} X^T y
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs
    
    # Compute R-squared
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return ResistanceSurrogate(
        a=float(a),
        b=float(b),
        r_squared=float(r_squared),
        n_calibration=len(mu_values),
    )


def calibrate_from_dem(
    n_mu_levels: int = 7,
    mu_range: tuple[float, float] = (0.0, 0.3),
    sigma_c_pa: float = 2.477e6,
    max_steps: int = 10000,
    seed: int = 7,
) -> tuple[ResistanceSurrogate, list[dict]]:
    """Calibrate surrogate by running DEM simulations at different mu values.
    
    Parameters
    ----------
    n_mu_levels : int
        Number of mu values to test
    mu_range : tuple
        (min, max) friction coefficient values
    sigma_c_pa : float
        Compressive strength [Pa] (fixed for calibration)
    max_steps : int
        DEM simulation steps per case
    seed : int
        Random seed for floe field generation
        
    Returns
    -------
    tuple of (ResistanceSurrogate, list of dict)
        Calibrated surrogate and raw calibration data
    """
    from tensordem.broken_ice_resistance.ledger import CaseSpec
    from tensordem.broken_ice_resistance.simulate import run_case
    
    mu_vals = list(np.linspace(mu_range[0], mu_range[1], n_mu_levels))
    sigma_c_vals = []
    r_vals = []
    raw_data = []
    
    for mu in mu_vals:
        spec = CaseSpec(
            friction_coefficient=float(mu),
            compressive_strength_pa=sigma_c_pa,
            seed=seed,
            name=f"calibration_mu={mu:.3f}",
        )
        result = run_case(spec, max_steps=max_steps)
        
        sigma_c_vals.append(sigma_c_pa)
        r_vals.append(result.mean_kN)
        raw_data.append({
            'mu': float(mu),
            'sigma_c_pa': sigma_c_pa,
            'mean_kN': result.mean_kN,
            'std_kN': result.std_kN,
        })
    
    surrogate = calibrate_surrogate(mu_vals, sigma_c_vals, r_vals)
    return surrogate, raw_data
