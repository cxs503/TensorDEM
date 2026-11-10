"""Ice-steel adhesion/friction cyclic degradation: parameter identifiability analysis.

This module provides:
1. Fisher Information Matrix (FIM) via numerical Jacobian
2. Cramér-Rao Lower Bound (CRLB) for parameter uncertainty
3. Parameter correlation matrix analysis
4. D-optimal experimental design (greedy exchange algorithm)
5. Synthetic twin inversion (MAP + MCMC)
6. Uncertainty propagation to DEM single-bond pull-off and ship-ice resistance

Physical model
--------------
The degradation law follows Weibull-type cyclic damage accumulation::

    D(N) = 1 - exp(-(N/N₀)^β)

where the characteristic cycle number follows Basquin's S-N law::

    N₀(τ/τ_c) = N_ref · (τ/τ_c)^(-m)

Parameters θ = (τ_c, m, β, N_ref, k_t/k_n):
    τ_c   : cohesive shear strength (adfreeze bond strength) [Pa]
    m     : Basquin exponent (S-N slope) [-]
    β     : Weibull shape parameter [-]
    N_ref : reference cycles at τ/τ_c = 1.0 [-]
    k_t/k_n: tangential-to-normal stiffness ratio [-]

Observables:
    - N_f(r): cycles to failure (D ≥ D_crit) at stress ratio r
    - D(N; r): damage evolution curve sampled at selected cycles
    - F_pull: pull-off force = τ_c × A_bond (identifies τ_c)
    - F_t(δ): tangential force (identifies k_t/k_n)

References
----------
- Timco & Frederking (1996): adfreeze strength measurements
- Suyuthi et al. (2013): ice adhesion degradation under cyclic loading
- m7757: parent mission establishing baseline m=8, β=1.5 hypothesis
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Callable
import json

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv, det

__all__ = [
    "IceAdhesionParams",
    "ForwardModel",
    "FisherInformation",
    "DOptimalDesign",
    "SyntheticTwin",
    "UncertaintyPropagation",
    "run_identifiability_analysis",
]


# =============================================================================
# Parameter container
# =============================================================================

@dataclass
class IceAdhesionParams:
    """Ice-steel adhesion degradation parameters.
    
    Default values based on m7757 hypothesis and Timco & Frederking (1996).
    """
    tau_c: float = 1.0e6        # Cohesive shear strength [Pa]
    m: float = 8.0              # Basquin exponent [-]
    beta: float = 1.5           # Weibull shape parameter [-]
    N_ref: float = 1000.0       # Reference cycles at r=1.0 [-]
    kt_kn: float = 0.5          # Tangential/normal stiffness ratio [-]
    D_crit: float = 0.95        # Critical damage for failure [-]
    r_fatigue: float = 0.3      # Fatigue limit ratio [-]
    k_adh: float = 1.0e6        # Bond stiffness [N/m]
    A_bond: float = 1.0e-4      # Tributary area [m²]
    
    def to_vector(self) -> np.ndarray:
        """Return identifiable parameters as array [τ_c, m, β, N_ref, k_t/k_n]."""
        return np.array([self.tau_c, self.m, self.beta, self.N_ref, self.kt_kn])
    
    @classmethod
    def from_vector(cls, x: np.ndarray, **kwargs) -> "IceAdhesionParams":
        """Create from parameter vector."""
        return cls(
            tau_c=float(x[0]),
            m=float(x[1]),
            beta=float(x[2]),
            N_ref=float(x[3]),
            kt_kn=float(x[4]),
            **kwargs
        )
    
    @property
    def param_names(self) -> List[str]:
        return ["τ_c", "m", "β", "N_ref", "k_t/k_n"]
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Physical bounds for each parameter."""
        return [
            (1e5, 5e6),      # τ_c: 0.1 - 5 MPa
            (2.0, 15.0),     # m: Basquin exponent
            (0.5, 5.0),      # β: Weibull shape
            (100, 1e5),      # N_ref: reference cycles
            (0.1, 2.0),      # k_t/k_n: stiffness ratio
        ]


# =============================================================================
# Forward model
# =============================================================================

class ForwardModel:
    """Forward model for ice-steel adhesion degradation experiments.
    
    Predicts observables given parameters θ and experimental conditions.
    """
    
    def __init__(self, params: Optional[IceAdhesionParams] = None):
        self.params = params or IceAdhesionParams()
    
    def N0_at_stress_ratio(self, r: float, params: Optional[IceAdhesionParams] = None) -> float:
        """Characteristic cycle number N₀ at stress ratio r = τ/τ_c."""
        p = params or self.params
        if r <= p.r_fatigue:
            return float('inf')
        N_f = p.N_ref * r ** (-p.m)
        N0 = N_f / (-math.log(1.0 - p.D_crit)) ** (1.0 / p.beta)
        return N0
    
    def cycles_to_failure(self, r: float, params: Optional[IceAdhesionParams] = None) -> Optional[float]:
        """Cycles to failure at stress ratio r."""
        p = params or self.params
        if r <= p.r_fatigue:
            return None
        N0 = self.N0_at_stress_ratio(r, params)
        N_f = N0 * (-math.log(1.0 - p.D_crit)) ** (1.0 / p.beta)
        return N_f
    
    def damage_at_cycle(self, N: int, r: float, params: Optional[IceAdhesionParams] = None) -> float:
        """Damage D at cycle N for stress ratio r."""
        p = params or self.params
        if r <= p.r_fatigue:
            return 0.0
        N0 = self.N0_at_stress_ratio(r, params)
        if N0 == float('inf') or N0 <= 0:
            return 0.0
        return 1.0 - math.exp(-(N / N0) ** p.beta)
    
    def damage_curve(self, cycles: np.ndarray, r: float, 
                     params: Optional[IceAdhesionParams] = None) -> np.ndarray:
        """Damage evolution D(N) for stress ratio r."""
        return np.array([self.damage_at_cycle(int(N), r, params) for N in cycles])
    
    def pull_off_force(self, params: Optional[IceAdhesionParams] = None) -> float:
        """Pull-off force F_pull = τ_c × A_bond [N].
        
        This observable directly identifies τ_c.
        """
        p = params or self.params
        return p.tau_c * p.A_bond
    
    def tangential_force(self, delta_t: float, params: Optional[IceAdhesionParams] = None) -> float:
        """Tangential force at displacement δ_t [N].
        
        F_t = k_t × δ_t = (k_t/k_n) × k_n × δ_t
        This observable identifies k_t/k_n.
        """
        p = params or self.params
        k_t = p.kt_kn * p.k_adh
        return k_t * delta_t
    
    def force_displacement(self, delta: float, N: int, r: float,
                           params: Optional[IceAdhesionParams] = None) -> float:
        """Force at displacement δ after N cycles at stress ratio r.
        
        Includes stiffness degradation and tangential coupling.
        """
        p = params or self.params
        D = self.damage_at_cycle(N, r, params)
        # Effective stiffness with damage
        eta = 0.01  # residual stiffness ratio
        k_eff = p.k_adh * ((1.0 - D) + eta * D)
        # Tangential coupling
        k_t = p.kt_kn * p.k_adh
        F_normal = k_eff * delta
        F_tangential = k_t * delta * 0.1  # simplified coupling
        return F_normal + F_tangential
    
    def predict_experiment(self, conditions: Dict, 
                          params: Optional[IceAdhesionParams] = None) -> np.ndarray:
        """Predict observables for an experiment.
        
        conditions: dict with keys:
            - 'measure_type': str, one of:
                'N_f': cycles to failure at stress_ratio
                'damage_curve': damage evolution at stress_ratio
                'pull_off': pull-off force (identifies τ_c)
                'tangential': tangential force (identifies k_t/k_n)
                'force_disp': force-displacement at given N, r
        """
        p = params or self.params
        measure = conditions.get('measure_type', 'N_f')
        
        if measure == 'N_f':
            r = conditions['stress_ratio']
            N_f = self.cycles_to_failure(r, params)
            return np.array([N_f if N_f is not None else 1e10])
        
        elif measure == 'damage_curve':
            r = conditions['stress_ratio']
            sample_cycles = conditions.get('sample_cycles', np.linspace(1, 1000, 20))
            return self.damage_curve(sample_cycles, r, params)
        
        elif measure == 'pull_off':
            # Pull-off force directly measures τ_c
            return np.array([self.pull_off_force(params)])
        
        elif measure == 'tangential':
            # Tangential force at specified displacement
            delta_t = conditions.get('delta_t', 1e-4)
            return np.array([self.tangential_force(delta_t, params)])
        
        elif measure == 'force_disp':
            delta_range = conditions.get('delta_range', np.linspace(-1e-4, 1e-4, 11))
            N = conditions.get('n_cycles', 0)
            r = conditions.get('stress_ratio', 0.7)
            return np.array([self.force_displacement(d, N, r, params) for d in delta_range])
        
        else:
            raise ValueError(f"Unknown measure_type: {measure}")
    
    def jacobian(self, conditions: Dict, params: Optional[IceAdhesionParams] = None,
                 eps: float = 1e-4) -> np.ndarray:
        """Numerical Jacobian ∂y/∂θ at given conditions.
        
        Uses central differences with relative perturbation eps.
        """
        p = params or self.params
        theta = p.to_vector()
        y0 = self.predict_experiment(conditions, p)
        n_obs = len(y0)
        n_param = len(theta)
        J = np.zeros((n_obs, n_param))
        
        for j in range(n_param):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            h = max(eps * abs(theta[j]), 1e-10)
            theta_plus[j] += h
            theta_minus[j] -= h
            
            p_plus = IceAdhesionParams.from_vector(theta_plus)
            p_minus = IceAdhesionParams.from_vector(theta_minus)
            
            y_plus = self.predict_experiment(conditions, p_plus)
            y_minus = self.predict_experiment(conditions, p_minus)
            
            J[:, j] = (y_plus - y_minus) / (2.0 * h)
        
        return J


# =============================================================================
# Fisher Information & CRLB
# =============================================================================

class FisherInformation:
    """Fisher Information Matrix and Cramér-Rao Lower Bound analysis."""
    
    def __init__(self, model: ForwardModel):
        self.model = model
    
    def fim_single_experiment(self, conditions: Dict, sigma: float = 0.01,
                              params: Optional[IceAdhesionParams] = None) -> np.ndarray:
        """FIM for a single experiment.
        
        F = (1/σ²) J^T J  (Gaussian noise model)
        """
        J = self.model.jacobian(conditions, params)
        return J.T @ J / (sigma ** 2)
    
    def fim_design_matrix(self, design_matrix: List[Dict], sigmas: Optional[List[float]] = None,
                          params: Optional[IceAdhesionParams] = None) -> np.ndarray:
        """FIM for a design matrix (collection of experiments).
        
        F = Σ_i (1/σ_i²) J_i^T J_i
        """
        n_param = 5
        F = np.zeros((n_param, n_param))
        n_exp = len(design_matrix)
        if sigmas is None:
            sigmas = [0.01] * n_exp
        
        for i, cond in enumerate(design_matrix):
            F += self.fim_single_experiment(cond, sigmas[i], params)
        
        return F
    
    def crlb(self, F: np.ndarray) -> np.ndarray:
        """Cramér-Rao Lower Bound: diagonal of F^{-1}.
        
        Returns standard deviations (sqrt of diagonal).
        """
        try:
            C = inv(F)
            return np.sqrt(np.abs(np.diag(C)))
        except np.linalg.LinAlgError:
            return np.full(5, np.inf)
    
    def relative_crlb(self, F: np.ndarray, params: Optional[IceAdhesionParams] = None) -> np.ndarray:
        """Relative CRLB: σ_θ / |θ|."""
        p = params or self.model.params
        theta = p.to_vector()
        crlb_vals = self.crlb(F)
        return crlb_vals / np.abs(theta)
    
    def correlation_matrix(self, F: np.ndarray) -> np.ndarray:
        """Parameter correlation matrix from FIM."""
        try:
            C = inv(F)
            d = np.sqrt(np.abs(np.diag(C)))
            corr = C / np.outer(d, d)
            return corr
        except np.linalg.LinAlgError:
            return np.full((5, 5), np.nan)
    
    def d_optimality(self, F: np.ndarray) -> float:
        """D-optimality criterion: det(F)^{1/p}."""
        d = det(F)
        if d <= 0:
            return 0.0
        return d ** (1.0 / 5)
    
    def a_optimality(self, F: np.ndarray) -> float:
        """A-optimality criterion: tr(F^{-1})."""
        try:
            return np.trace(inv(F))
        except np.linalg.LinAlgError:
            return np.inf


# =============================================================================
# D-optimal design
# =============================================================================

class DOptimalDesign:
    """D-optimal experimental design via greedy exchange algorithm."""
    
    def __init__(self, fim_calc: FisherInformation, candidate_pool: List[Dict]):
        self.fim = fim_calc
        self.candidates = candidate_pool
    
    def uniform_grid_design(self, n_stress: int = 5, n_cycles: int = 4) -> List[Dict]:
        """Generate uniform grid design as baseline.
        
        Includes multiple measurement types for full identifiability.
        """
        design = []
        
        # Damage curves at various stress ratios
        for r in np.linspace(0.4, 0.9, n_stress):
            for nc in [100, 1000, 5000]:
                design.append({
                    'stress_ratio': float(r),
                    'n_cycles': int(nc),
                    'measure_type': 'damage_curve',
                    'sample_cycles': np.linspace(1, nc, 10).astype(int).tolist()
                })
        
        # Pull-off tests (identify τ_c)
        design.append({'measure_type': 'pull_off'})
        
        # Tangential tests (identify k_t/k_n)
        for delta_t in [1e-5, 1e-4, 1e-3]:
            design.append({'measure_type': 'tangential', 'delta_t': float(delta_t)})
        
        return design
    
    def greedy_exchange(self, n_initial: int = 10, max_iter: int = 30,
                        params: Optional[IceAdhesionParams] = None) -> Tuple[List[Dict], float]:
        """Greedy exchange algorithm for D-optimal design.
        
        Start with random subset, iteratively swap points to maximize det(F).
        """
        rng = np.random.default_rng(42)
        n_cand = len(self.candidates)
        
        # Initial random selection
        indices = rng.choice(n_cand, size=min(n_initial, n_cand), replace=False).tolist()
        current_design = [self.candidates[i] for i in indices]
        
        F = self.fim.fim_design_matrix(current_design, params=params)
        current_det = det(F) if det(F) > 0 else 1e-30
        
        for iteration in range(max_iter):
            improved = False
            
            # Try swapping each point
            for i in range(len(current_design)):
                best_swap_det = current_det
                best_swap_idx = -1
                
                # Sample candidates to speed up
                candidate_indices = rng.choice(n_cand, size=min(20, n_cand), replace=False)
                
                for j in candidate_indices:
                    if j in indices:
                        continue
                    
                    # Try swap
                    old_point = current_design[i]
                    current_design[i] = self.candidates[j]
                    
                    F_trial = self.fim.fim_design_matrix(current_design, params=params)
                    trial_det = det(F_trial) if det(F_trial) > 0 else 1e-30
                    
                    if trial_det > best_swap_det * 1.001:  # 0.1% improvement threshold
                        best_swap_det = trial_det
                        best_swap_idx = j
                    
                    current_design[i] = old_point
                
                if best_swap_idx >= 0:
                    indices[i] = best_swap_idx
                    current_design[i] = self.candidates[best_swap_idx]
                    current_det = best_swap_det
                    improved = True
            
            if not improved:
                break
        
        return current_design, current_det
    
    def compare_designs(self, d_opt_design: List[Dict], uniform_design: List[Dict],
                        params: Optional[IceAdhesionParams] = None) -> Dict:
        """Compare D-optimal vs uniform design."""
        F_dopt = self.fim.fim_design_matrix(d_opt_design, params=params)
        F_unif = self.fim.fim_design_matrix(uniform_design, params=params)
        
        det_dopt = det(F_dopt) if det(F_dopt) > 0 else 1e-30
        det_unif = det(F_unif) if det(F_unif) > 0 else 1e-30
        
        return {
            'det_ratio': det_dopt / det_unif,
            'd_opt_dopt': self.fim.d_optimality(F_dopt),
            'd_opt_unif': self.fim.d_optimality(F_unif),
            'rel_crlb_dopt': self.fim.relative_crlb(F_dopt, params).tolist(),
            'rel_crlb_unif': self.fim.relative_crlb(F_unif, params).tolist(),
            'corr_dopt': self.fim.correlation_matrix(F_dopt).tolist(),
            'corr_unif': self.fim.correlation_matrix(F_unif).tolist(),
        }


# =============================================================================
# Synthetic twin inversion
# =============================================================================

class SyntheticTwin:
    """Synthetic twin experiment: generate data with truth, invert with noise."""
    
    def __init__(self, true_params: IceAdhesionParams, design: List[Dict]):
        self.true_params = true_params
        self.design = design
        self.model = ForwardModel(true_params)
    
    def generate_data(self, noise_level: float = 0.0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic experimental data.
        
        Args:
            noise_level: relative noise standard deviation (0 = no noise)
            seed: random seed
        
        Returns:
            y_obs: concatenated observations
            y_std: observation standard deviations
        """
        rng = np.random.default_rng(seed)
        y_list = []
        std_list = []
        
        for cond in self.design:
            y_true = self.model.predict_experiment(cond, self.true_params)
            y_std = noise_level * np.abs(y_true) + 1e-10
            y_obs = y_true + rng.normal(0, y_std)
            y_list.append(y_obs)
            std_list.append(y_std)
        
        return np.concatenate(y_list), np.concatenate(std_list)
    
    def log_likelihood(self, theta: np.ndarray, y_obs: np.ndarray, 
                       y_std: np.ndarray) -> float:
        """Gaussian log-likelihood."""
        params = IceAdhesionParams.from_vector(theta)
        model = ForwardModel(params)
        
        y_pred_list = []
        for cond in self.design:
            y_pred_list.append(model.predict_experiment(cond, params))
        y_pred = np.concatenate(y_pred_list)
        
        # Gaussian log-likelihood
        ll = -0.5 * np.sum(((y_obs - y_pred) / y_std) ** 2 + np.log(2 * np.pi * y_std ** 2))
        return ll
    
    def log_prior(self, theta: np.ndarray) -> float:
        """Uniform prior within bounds."""
        bounds = IceAdhesionParams().bounds
        for i, (lo, hi) in enumerate(bounds):
            if theta[i] < lo or theta[i] > hi:
                return -np.inf
        return 0.0
    
    def log_posterior(self, theta: np.ndarray, y_obs: np.ndarray, 
                      y_std: np.ndarray) -> float:
        """Log posterior = log prior + log likelihood."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, y_obs, y_std)
    
    def map_estimate(self, y_obs: np.ndarray, y_std: np.ndarray,
                     x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Maximum a posteriori (MAP) estimate via L-BFGS-B."""
        if x0 is None:
            x0 = self.true_params.to_vector() * 1.1  # perturbed start
        
        bounds = IceAdhesionParams().bounds
        
        def neg_log_post(theta):
            return -self.log_posterior(theta, y_obs, y_std)
        
        result = minimize(neg_log_post, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, -result.fun
    
    def laplace_covariance(self, theta_map: np.ndarray, y_obs: np.ndarray,
                           y_std: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Laplace approximation: posterior covariance from Hessian at MAP."""
        n = len(theta_map)
        H = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                theta_pp = theta_map.copy()
                theta_pm = theta_map.copy()
                theta_mp = theta_map.copy()
                theta_mm = theta_map.copy()
                
                hi = max(eps * abs(theta_map[i]), 1e-8)
                hj = max(eps * abs(theta_map[j]), 1e-8)
                
                theta_pp[i] += hi; theta_pp[j] += hj
                theta_pm[i] += hi; theta_pm[j] -= hj
                theta_mp[i] -= hi; theta_mp[j] += hj
                theta_mm[i] -= hi; theta_mm[j] -= hj
                
                fpp = self.log_posterior(theta_pp, y_obs, y_std)
                fpm = self.log_posterior(theta_pm, y_obs, y_std)
                fmp = self.log_posterior(theta_mp, y_obs, y_std)
                fmm = self.log_posterior(theta_mm, y_obs, y_std)
                
                H[i, j] = -(fpp - fpm - fmp + fmm) / (4 * hi * hj)
                H[j, i] = H[i, j]
        
        try:
            cov = inv(H)
            return cov
        except np.linalg.LinAlgError:
            return np.full((n, n), np.nan)
    
    def mcmc_sample(self, y_obs: np.ndarray, y_std: np.ndarray,
                    n_samples: int = 1000, n_burn: int = 200,
                    theta_init: Optional[np.ndarray] = None,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Metropolis-Hastings MCMC sampling."""
        rng = np.random.default_rng(seed)
        n_param = 5
        
        if theta_init is None:
            theta_init = self.true_params.to_vector()
        
        theta_current = theta_init.copy()
        lp_current = self.log_posterior(theta_current, y_obs, y_std)
        
        # Proposal scale (adaptive)
        proposal_scale = 0.02 * np.abs(theta_init)
        
        samples = np.zeros((n_samples, n_param))
        log_probs = np.zeros(n_samples)
        n_accept = 0
        
        for i in range(n_samples + n_burn):
            # Propose
            theta_prop = theta_current + rng.normal(0, proposal_scale, n_param)
            lp_prop = self.log_posterior(theta_prop, y_obs, y_std)
            
            # Accept/reject
            log_alpha = lp_prop - lp_current
            if log_alpha > 0 or rng.random() < math.exp(min(log_alpha, 0)):
                theta_current = theta_prop
                lp_current = lp_prop
                if i >= n_burn:
                    n_accept += 1
            
            if i >= n_burn:
                samples[i - n_burn] = theta_current
                log_probs[i - n_burn] = lp_current
        
        return samples, log_probs


# =============================================================================
# Uncertainty propagation
# =============================================================================

class UncertaintyPropagation:
    """Propagate parameter posterior to QoI: single-bond pull-off and ship-ice resistance."""
    
    def __init__(self, model: ForwardModel):
        self.model = model
    
    def single_bond_pulloff(self, params: IceAdhesionParams) -> float:
        """Single bond pull-off force [N]."""
        return params.tau_c * params.A_bond
    
    def debond_lifetime(self, params: IceAdhesionParams, stress_ratio: float = 0.7) -> float:
        """Cycles to debond at given stress ratio."""
        N_f = self.model.cycles_to_failure(stress_ratio, params)
        return N_f if N_f is not None else 1e10
    
    def ship_ice_friction_resistance(self, params: IceAdhesionParams,
                                     ice_thickness: float = 0.5,
                                     ship_speed: float = 3.0) -> float:
        """Simplified ship-ice friction resistance estimate [N]."""
        mu_base = 0.15
        mu_adh = 0.05 * params.kt_kn
        mu_total = mu_base + mu_adh
        
        rho_ice = 900.0
        g = 9.81
        p = rho_ice * g * ice_thickness
        
        L = 100.0
        B = 20.0
        A_wl = L * B * 0.1
        
        return mu_total * p * A_wl
    
    def propagate_samples(self, samples: np.ndarray, 
                          qoi_func: Callable[[IceAdhesionParams], float]) -> Dict:
        """Propagate posterior samples through QoI function."""
        qoi_values = np.array([qoi_func(IceAdhesionParams.from_vector(s)) for s in samples])
        
        return {
            'mean': float(np.mean(qoi_values)),
            'std': float(np.std(qoi_values)),
            'P5': float(np.percentile(qoi_values, 5)),
            'P50': float(np.percentile(qoi_values, 50)),
            'P95': float(np.percentile(qoi_values, 95)),
            'P2.5': float(np.percentile(qoi_values, 2.5)),
            'P97.5': float(np.percentile(qoi_values, 97.5)),
        }
    
    def bandwidth_contraction(self, prior_samples: np.ndarray, 
                              posterior_samples: np.ndarray) -> Dict:
        """Compute bandwidth contraction from prior to posterior."""
        prior_pulloff = self.propagate_samples(prior_samples, self.single_bond_pulloff)
        post_pulloff = self.propagate_samples(posterior_samples, self.single_bond_pulloff)
        
        prior_bw = prior_pulloff['P95'] - prior_pulloff['P5']
        post_bw = post_pulloff['P95'] - post_pulloff['P5']
        
        contraction = 1.0 - post_bw / prior_bw if prior_bw > 0 else 0.0
        
        return {
            'prior_bandwidth': prior_bw,
            'posterior_bandwidth': post_bw,
            'contraction_ratio': contraction,
        }


# =============================================================================
# Main analysis runner
# =============================================================================

def run_identifiability_analysis(
    true_params: Optional[IceAdhesionParams] = None,
    noise_levels: List[float] = None,
    n_mcmc_samples: int = 1000,
    n_twin_repeats: int = 20,
    seed: int = 42
) -> Dict:
    """Run complete identifiability analysis."""
    if true_params is None:
        true_params = IceAdhesionParams()
    
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.03]
    
    results = {
        'true_params': true_params.to_vector().tolist(),
        'param_names': true_params.param_names,
    }
    
    # Generate candidate pool
    candidates = []
    
    # Damage curves
    for r in np.linspace(0.35, 0.95, 10):
        for nc in [100, 500, 2000, 8000]:
            candidates.append({
                'stress_ratio': float(r),
                'n_cycles': int(nc),
                'measure_type': 'damage_curve',
                'sample_cycles': np.linspace(1, nc, 8).astype(int).tolist()
            })
    
    # Pull-off tests
    candidates.append({'measure_type': 'pull_off'})
    
    # Tangential tests
    for delta_t in [1e-5, 1e-4, 1e-3]:
        candidates.append({'measure_type': 'tangential', 'delta_t': float(delta_t)})
    
    # Fisher Information Analysis
    model = ForwardModel(true_params)
    fim_calc = FisherInformation(model)
    
    # Uniform grid baseline
    d_opt = DOptimalDesign(fim_calc, candidates)
    uniform_design = d_opt.uniform_grid_design()
    
    F_unif = fim_calc.fim_design_matrix(uniform_design)
    
    # D-optimal design
    d_opt_design, d_opt_det = d_opt.greedy_exchange(n_initial=10, max_iter=20)
    F_dopt = fim_calc.fim_design_matrix(d_opt_design)
    
    design_comparison = d_opt.compare_designs(d_opt_design, uniform_design)
    
    results['design'] = {
        'uniform': {
            'n_experiments': len(uniform_design),
            'det_F': float(det(F_unif)) if det(F_unif) > 0 else 0.0,
            'rel_crlb': fim_calc.relative_crlb(F_unif).tolist(),
        },
        'd_optimal': {
            'n_experiments': len(d_opt_design),
            'det_F': float(det(F_dopt)) if det(F_dopt) > 0 else 0.0,
            'rel_crlb': fim_calc.relative_crlb(F_dopt).tolist(),
        },
        'det_ratio': design_comparison['det_ratio'],
    }
    
    # Synthetic twin inversion
    twin_results = {}
    for noise in noise_levels:
        twin = SyntheticTwin(true_params, d_opt_design)
        y_obs, y_std = twin.generate_data(noise_level=noise, seed=seed)
        
        theta_map, _ = twin.map_estimate(y_obs, y_std)
        cov_laplace = twin.laplace_covariance(theta_map, y_obs, y_std)
        
        recovery_error = np.abs(theta_map - true_params.to_vector()) / np.abs(true_params.to_vector())
        
        twin_results[f'noise_{noise}'] = {
            'theta_map': theta_map.tolist(),
            'recovery_error': recovery_error.tolist(),
            'laplace_std': np.sqrt(np.abs(np.diag(cov_laplace))).tolist(),
        }
    
    results['twin_inversion'] = twin_results
    
    # Coverage analysis
    coverage_results = {}
    for noise in [0.01]:
        n_repeats = min(n_twin_repeats, 20)
        n_covered = np.zeros(5)
        
        for rep in range(n_repeats):
            twin = SyntheticTwin(true_params, d_opt_design)
            y_obs, y_std = twin.generate_data(noise_level=noise, seed=seed + rep)
            theta_map, _ = twin.map_estimate(y_obs, y_std)
            cov = twin.laplace_covariance(theta_map, y_obs, y_std)
            std = np.sqrt(np.abs(np.diag(cov)))
            
            for j in range(5):
                lo = theta_map[j] - 1.96 * std[j]
                hi = theta_map[j] + 1.96 * std[j]
                if lo <= true_params.to_vector()[j] <= hi:
                    n_covered[j] += 1
        
        coverage_results[f'noise_{noise}'] = {
            'n_repeats': n_repeats,
            'coverage': (n_covered / n_repeats).tolist(),
            'mean_coverage': float(np.mean(n_covered / n_repeats)),
        }
    
    results['coverage'] = coverage_results
    
    # Uncertainty propagation
    up = UncertaintyPropagation(model)
    
    rng = np.random.default_rng(seed)
    bounds = true_params.bounds
    prior_samples = np.zeros((200, 5))
    for j in range(5):
        prior_samples[:, j] = rng.uniform(bounds[j][0], bounds[j][1], 200)
    
    twin = SyntheticTwin(true_params, d_opt_design)
    y_obs, y_std = twin.generate_data(noise_level=0.01, seed=seed)
    post_samples, _ = twin.mcmc_sample(y_obs, y_std, n_samples=500, n_burn=100, seed=seed)
    
    contraction = up.bandwidth_contraction(prior_samples, post_samples)
    
    results['propagation'] = {
        'pulloff': up.propagate_samples(post_samples, up.single_bond_pulloff),
        'bandwidth_contraction': {
            'prior_bw': contraction['prior_bandwidth'],
            'posterior_bw': contraction['posterior_bandwidth'],
            'contraction_ratio': contraction['contraction_ratio'],
        }
    }
    
    # Data halving test
    half_design = d_opt_design[:len(d_opt_design)//2]
    twin_half = SyntheticTwin(true_params, half_design)
    y_obs_half, y_std_half = twin_half.generate_data(noise_level=0.01, seed=seed)
    theta_map_half, _ = twin_half.map_estimate(y_obs_half, y_std_half)
    cov_half = twin_half.laplace_covariance(theta_map_half, y_obs_half, y_std_half)
    std_half = np.sqrt(np.abs(np.diag(cov_half)))
    
    twin_full = SyntheticTwin(true_params, d_opt_design)
    y_obs_full, y_std_full = twin_full.generate_data(noise_level=0.01, seed=seed)
    theta_map_full, _ = twin_full.map_estimate(y_obs_full, y_std_full)
    cov_full = twin_full.laplace_covariance(theta_map_full, y_obs_full, y_std_full)
    std_full = np.sqrt(np.abs(np.diag(cov_full)))
    
    bandwidth_ratio = std_half / (std_full + 1e-30)
    
    results['data_halving'] = {
        'full_std': std_full.tolist(),
        'half_std': std_half.tolist(),
        'bandwidth_ratio': bandwidth_ratio.tolist(),
        'mean_ratio': float(np.mean(bandwidth_ratio)),
    }
    
    return results
