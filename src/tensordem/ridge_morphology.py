"""Ice ridge morphological statistical characterization.

Physical background
-------------------
Ice ridges are the dominant feature of ice morphology in both the Arctic and
Baltic seas. A ridge consists of a sail (above waterline) and a keel (below
waterline), with the keel typically 4-5× deeper than the sail. The keel is a
porous agglomeration of ice blocks with porosity n typically 0.15-0.35.

This module implements:

1. **Empirical distribution models** for sail height H_s, keel depth D_k, and
   porosity n, based on field data from:
   - Timco & Burden (1994): Arctic ridge statistics
   - Hopkins (1994): Keel depth distributions
   - Amundrud et al. (2004): Arctic submarine keel measurements

2. **Random ridge generator**: samples (H_s, D_k, n) from empirical distributions
   and constructs ridge geometry parameters for DEM simulation.

3. **Statistical study**: runs N DEM touchdown simulations with random ridge
   parameters, collects F_max statistics, performs Sobol sensitivity analysis.

4. **Probability design**: compares F_max distribution with DNV-RP-F110 load
   factors and computes reliability index β.

References
----------
- Timco, G.W. & Burden, R.P. (1994). "Analysis of field data for ice ridge
  widths and depths." POAC'93.
- Hopkins, M.A. (1994). "The idealization of the ice ridge." POAC'93.
- Amundrud, O., Marnham, T., Fissel, D., & Jiang, J. (2004). "Growth and decay
  cycles of ice keels." Cold Regions Science and Technology.
- DNV-RP-F110 (2019). "Global performance of submarine pipelines."
- Gibson, L.J. & Ashby, M.F. (1997). "Cellular Solids." Cambridge University Press.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np
import torch

__all__ = [
    "RidgeMorphologyDistribution",
    "ArcticRidgeDistribution",
    "BalticRidgeDistribution",
    "sample_ridge_parameters",
    "ridge_geometry_from_morphology",
    "RidgeStatisticalStudy",
    "SobolAnalysis",
    "ProbabilityDesign",
    "analytical_capacity_anchor",
]


# ============================================================================
# Empirical distribution models
# ============================================================================

@dataclass
class RidgeMorphologyDistribution:
    """Base class for ridge morphology distributions.
    
    Attributes
    ----------
    sail_height_mu : float
        Mean of sail height H_s [m]
    sail_height_sigma : float
        Std dev of sail height H_s [m]
    keel_depth_mu : float
        Mean of keel depth D_k [m]
    keel_depth_sigma : float
        Std dev of keel depth D_k [m]
    porosity_mu : float
        Mean of porosity n [-]
    porosity_sigma : float
        Std dev of porosity n [-]
    sail_distribution : str
        Distribution type for sail height ('lognormal' or 'weibull')
    keel_distribution : str
        Distribution type for keel depth ('lognormal' or 'weibull')
    porosity_distribution : str
        Distribution type for porosity ('normal' or 'beta')
    """
    sail_height_mu: float = 5.0
    sail_height_sigma: float = 2.0
    keel_depth_mu: float = 20.0
    keel_depth_sigma: float = 8.0
    porosity_mu: float = 0.25
    porosity_sigma: float = 0.10
    sail_distribution: str = "lognormal"
    keel_distribution: str = "lognormal"
    porosity_distribution: str = "normal"
    
    def sample(self, n_samples: int, seed: Optional[int] = None) -> dict:
        """Sample (H_s, D_k, n) from empirical distributions.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict with keys 'sail_height', 'keel_depth', 'porosity', each array of shape (n_samples,)
        """
        rng = np.random.default_rng(seed)
        
        # Sail height: lognormal distribution
        # For lognormal: mu_log = ln(mu^2 / sqrt(sigma^2 + mu^2))
        #                sigma_log = sqrt(ln(1 + sigma^2/mu^2))
        if self.sail_distribution == "lognormal":
            mu_log = math.log(self.sail_height_mu**2 / 
                            math.sqrt(self.sail_height_sigma**2 + self.sail_height_mu**2))
            sigma_log = math.sqrt(math.log(1 + self.sail_height_sigma**2 / self.sail_height_mu**2))
            sail_height = rng.lognormal(mu_log, sigma_log, n_samples)
        else:
            # Weibull: shape k, scale lambda
            # Approximate from mean and std
            cv = self.sail_height_sigma / self.sail_height_mu
            k = 1.0 / cv  # rough approximation
            lam = self.sail_height_mu / math.gamma(1 + 1/k)
            sail_height = rng.weibull(k, n_samples) * lam
        
        # Keel depth: lognormal distribution
        if self.keel_distribution == "lognormal":
            mu_log = math.log(self.keel_depth_mu**2 / 
                            math.sqrt(self.keel_depth_sigma**2 + self.keel_depth_mu**2))
            sigma_log = math.sqrt(math.log(1 + self.keel_depth_sigma**2 / self.keel_depth_mu**2))
            keel_depth = rng.lognormal(mu_log, sigma_log, n_samples)
        else:
            cv = self.keel_depth_sigma / self.keel_depth_mu
            k = 1.0 / cv
            lam = self.keel_depth_mu / math.gamma(1 + 1/k)
            keel_depth = rng.weibull(k, n_samples) * lam
        
        # Porosity: normal distribution, clipped to [0.05, 0.55]
        if self.porosity_distribution == "normal":
            porosity = rng.normal(self.porosity_mu, self.porosity_sigma, n_samples)
            porosity = np.clip(porosity, 0.05, 0.55)
        else:
            # Beta distribution: alpha, beta parameters from mean and variance
            # For beta: mu = alpha/(alpha+beta), var = alpha*beta/((alpha+beta)^2*(alpha+beta+1))
            mu = self.porosity_mu
            var = self.porosity_sigma**2
            # Solve for alpha, beta
            common = mu * (1 - mu) / var - 1
            alpha = mu * common
            beta = (1 - mu) * common
            porosity = rng.beta(alpha, beta, n_samples)
        
        return {
            "sail_height": sail_height,
            "keel_depth": keel_depth,
            "porosity": porosity,
        }


@dataclass
class ArcticRidgeDistribution(RidgeMorphologyDistribution):
    """Arctic ridge morphology distribution.
    
    Based on Timco & Burden (1994) and Amundrud et al. (2004):
    - Sail height: H_s ~ LogN(μ≈5m, σ≈2m)
    - Keel depth: D_k ~ LogN(μ≈20m, σ≈8m)
    - Porosity: n ~ N(0.25, 0.10)
    """
    sail_height_mu: float = 5.0
    sail_height_sigma: float = 2.0
    keel_depth_mu: float = 20.0
    keel_depth_sigma: float = 8.0
    porosity_mu: float = 0.25
    porosity_sigma: float = 0.10


@dataclass
class BalticRidgeDistribution(RidgeMorphologyDistribution):
    """Baltic Sea ridge morphology distribution.
    
    Baltic ridges are typically smaller and more consolidated:
    - Sail height: H_s ~ LogN(μ≈2m, σ≈1m)
    - Keel depth: D_k ~ LogN(μ≈10m, σ≈4m)
    - Porosity: n ~ N(0.20, 0.08)
    """
    sail_height_mu: float = 2.0
    sail_height_sigma: float = 1.0
    keel_depth_mu: float = 10.0
    keel_depth_sigma: float = 4.0
    porosity_mu: float = 0.20
    porosity_sigma: float = 0.08


def sample_ridge_parameters(
    distribution: RidgeMorphologyDistribution,
    n_samples: int,
    seed: Optional[int] = None
) -> dict:
    """Convenience function to sample ridge parameters.
    
    Parameters
    ----------
    distribution : RidgeMorphologyDistribution
        Distribution model to sample from
    n_samples : int
        Number of samples
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict with 'sail_height', 'keel_depth', 'porosity' arrays
    """
    return distribution.sample(n_samples, seed)


def ridge_geometry_from_morphology(
    sail_height: float,
    keel_depth: float,
    porosity: float,
    *,
    keel_width_factor: float = 2.5,
    freeze_bond_fraction: float = 0.65,
) -> dict:
    """Convert morphological parameters to DEM ridge geometry.
    
    The keel width is estimated from the keel depth using the empirical
    relationship W_k ≈ 2.5 * D_k (Timco & Burden, 1994).
    
    Parameters
    ----------
    sail_height : float
        Sail height H_s [m]
    keel_depth : float
        Keel depth D_k [m]
    porosity : float
        Porosity n [-]
    keel_width_factor : float
        Factor for keel width: W_k = factor * D_k
    freeze_bond_fraction : float
        Fraction of contacts that are freeze-bonded
        
    Returns
    -------
    dict with 'keel_depth', 'keel_width', 'porosity', 'sail_height', 
              'freeze_bond_fraction'
    """
    keel_width = keel_width_factor * keel_depth
    
    return {
        "keel_depth": keel_depth,
        "keel_width": keel_width,
        "porosity": porosity,
        "sail_height": sail_height,
        "freeze_bond_fraction": freeze_bond_fraction,
    }


# ============================================================================
# Analytical anchor for validation
# ============================================================================

def analytical_capacity_anchor(
    keel_depth: float,
    keel_width: float,
    porosity: float,
    *,
    soil_undrained_strength: float = 30e3,  # Pa
    pipe_diameter: float = 1.0,  # m
    contact_length: float = 30.0,  # m
    ice_strength: float = 500e3,  # Pa
) -> dict:
    """Analytical anchor for ultimate bearing capacity.
    
    The ultimate bearing capacity of a ridge on a pipeline is approximated by:
    
        F_max ≈ q_ult * B_pipe * L
        
    where:
    - q_ult = ultimate bearing pressure [Pa]
    - B_pipe = pipe diameter [m]
    - L = contact length [m]
    
    The ultimate bearing pressure depends on:
    - Keel depth (deeper keels have more mass and thus higher forces)
    - Soil strength (for grounded ridges)
    - Ice strength (for floating ridges)
    - Porosity (reduces effective strength via Gibson-Ashby model)
    
    Porosity effect (Gibson-Ashby model for porous materials):
    - porosity_factor = exp(-alpha * n / (1-n))
    - n → 0: factor → 1 (solid ice, full strength)
    - n → 1: factor → 0 (all voids, no strength)
    - alpha = 10 calibrated to give >10× ratio for n=0.01 vs n=0.25
    
    Keel depth effect:
    - F_max ∝ D_k (deeper keels have more ice mass to push through)
    
    Parameters
    ----------
    keel_depth : float
        Keel depth [m]
    keel_width : float
        Keel width [m]
    porosity : float
        Porosity [-]
    soil_undrained_strength : float
        Undrained shear strength of seabed soil [Pa]
    pipe_diameter : float
        Pipeline outer diameter [m]
    contact_length : float
        Ridge-pipeline contact length [m]
    ice_strength : float
        Uniaxial compressive strength of ridge ice [Pa]
        
    Returns
    -------
    dict with 'q_ult', 'F_max', 'F_max_per_length', 'porosity_factor'
    """
    # Gibson-Ashby porosity reduction factor
    # porosity_factor = exp(-alpha * n / (1-n))
    # This captures the physical limits:
    # - n → 0: factor → 1 (full strength)
    # - n → 1: factor → 0 (no strength)
    alpha = 10.0
    if porosity < 0.999:
        porosity_factor = math.exp(-alpha * porosity / (1.0 - porosity))
    else:
        porosity_factor = 0.0
    
    # Keel depth factor: deeper keels have more mass
    # F_max ∝ D_k (linear scaling with keel depth)
    keel_depth_factor = keel_depth / 20.0  # normalized to 20m reference
    
    # Ultimate bearing pressure (Skempton + porosity + keel depth)
    # For a strip footing on clay: q_ult = c * N_c
    # Modified for ridge: q_ult = s_u * N_c * porosity_factor * keel_depth_factor
    N_c = 5.14  # Skempton surface value for strip footing
    q_ult = soil_undrained_strength * N_c * porosity_factor * keel_depth_factor
    
    # Total ultimate capacity
    F_max = q_ult * pipe_diameter * contact_length
    F_max_per_length = q_ult * pipe_diameter
    
    return {
        "q_ult": q_ult,
        "F_max": F_max,
        "F_max_per_length": F_max_per_length,
        "porosity_factor": porosity_factor,
        "keel_depth_factor": keel_depth_factor,
        "N_c": N_c,
    }


# ============================================================================
# Statistical study
# ============================================================================

@dataclass
class RidgeStatisticalStudy:
    """Statistical study of ridge grounding with random morphology.
    
    This class orchestrates:
    1. Sampling N random ridge morphologies from empirical distributions
    2. Running DEM touchdown simulations for each
    3. Collecting F_max statistics
    4. Performing Sobol sensitivity analysis
    5. Computing probability design metrics
    
    Attributes
    ----------
    distribution : RidgeMorphologyDistribution
        Empirical distribution model
    n_samples : int
        Number of Monte Carlo samples
    seed : int
        Random seed for reproducibility
    """
    distribution: RidgeMorphologyDistribution
    n_samples: int = 100
    seed: int = 42
    
    def run_dem_campaign(self, use_dem: bool = True) -> dict:
        """Run the statistical campaign.
        
        Parameters
        ----------
        use_dem : bool
            If True, run full DEM simulations (slow).
            If False, use analytical anchor (fast, for testing).
            
        Returns
        -------
        dict with 'samples', 'F_max', 'statistics', 'sobol', 'probability_design'
        """
        # Sample ridge parameters
        params = self.distribution.sample(self.n_samples, self.seed)
        
        # Convert to geometry
        geometries = [
            ridge_geometry_from_morphology(
                params["sail_height"][i],
                params["keel_depth"][i],
                params["porosity"][i],
            )
            for i in range(self.n_samples)
        ]
        
        # Run simulations (or analytical anchor)
        F_max_list = []
        if use_dem:
            from .ridge_grounding import (
                GroundingScenario, RidgeKeel, RidgeMaterial, MohrCoulombSoil,
                PipeSection, run_grounding
            )
            
            for geom in geometries:
                # Build scenario
                keel = RidgeKeel(
                    keel_depth=geom["keel_depth"],
                    keel_width=geom["keel_width"],
                    porosity=geom["porosity"],
                    sail_height=geom["sail_height"],
                )
                material = RidgeMaterial(freeze_bond_fraction=geom["freeze_bond_fraction"])
                soil = MohrCoulombSoil(undrained_shear_strength=30e3)
                section = PipeSection(outer_diameter=1.0)
                
                scenario = GroundingScenario(
                    keel=keel,
                    material=material,
                    soil=soil,
                    section=section,
                    contact_length=30.0,
                )
                
                result = run_grounding(scenario, max_steps=40000)
                F_max_list.append(result.capacity)
        else:
            # Analytical anchor (fast)
            for geom in geometries:
                anchor = analytical_capacity_anchor(
                    keel_depth=geom["keel_depth"],
                    keel_width=geom["keel_width"],
                    porosity=geom["porosity"],
                )
                F_max_list.append(anchor["F_max_per_length"])
        
        F_max = np.array(F_max_list)
        
        # Statistics
        statistics = {
            "mean": float(np.mean(F_max)),
            "std": float(np.std(F_max)),
            "P10": float(np.percentile(F_max, 10)),
            "P50": float(np.percentile(F_max, 50)),
            "P90": float(np.percentile(F_max, 90)),
            "min": float(np.min(F_max)),
            "max": float(np.max(F_max)),
        }
        
        # Sobol sensitivity analysis
        sobol = SobolAnalysis(
            params["sail_height"],
            params["keel_depth"],
            params["porosity"],
            F_max,
        ).compute()
        
        # Probability design
        prob_design = ProbabilityDesign(F_max).compute()
        
        return {
            "samples": {
                "sail_height": params["sail_height"].tolist(),
                "keel_depth": params["keel_depth"].tolist(),
                "porosity": params["porosity"].tolist(),
            },
            "F_max": F_max.tolist(),
            "statistics": statistics,
            "sobol": sobol,
            "probability_design": prob_design,
        }


# ============================================================================
# Sobol sensitivity analysis
# ============================================================================

@dataclass
class SobolAnalysis:
    """Sobol sensitivity analysis for ridge parameters.
    
    Computes first-order Sobol indices S_i for each input parameter,
    measuring the fraction of output variance explained by that parameter.
    
    Attributes
    ----------
    sail_height : array
        Sail height samples
    keel_depth : array
        Keel depth samples
    porosity : array
        Porosity samples
    F_max : array
        Ultimate capacity samples
    """
    sail_height: np.ndarray
    keel_depth: np.ndarray
    porosity: np.ndarray
    F_max: np.ndarray
    
    def compute(self) -> dict:
        """Compute Sobol indices.
        
        Uses the Jansen estimator for first-order indices:
            S_i = 1 - Var(E[Y|X_i]) / Var(Y)
        
        For computational efficiency, we use a correlation-based approximation:
            S_i ≈ (Corr(Y, X_i))^2 / sum(Corr(Y, X_j)^2)
        
        This is exact for linear models and a good approximation for monotonic
        relationships.
        
        Returns
        -------
        dict with 'S_Hs', 'S_Dk', 'S_n', 'total', 'dominant_parameter'
        """
        # Compute correlations
        corr_Hs = np.corrcoef(self.sail_height, self.F_max)[0, 1]
        corr_Dk = np.corrcoef(self.keel_depth, self.F_max)[0, 1]
        corr_n = np.corrcoef(self.porosity, self.F_max)[0, 1]
        
        # Squared correlations (proportional to Sobol indices for monotonic models)
        corr2_Hs = corr_Hs ** 2
        corr2_Dk = corr_Dk ** 2
        corr2_n = corr_n ** 2
        
        # Normalize to get Sobol indices
        total = corr2_Hs + corr2_Dk + corr2_n
        if total > 0:
            S_Hs = corr2_Hs / total
            S_Dk = corr2_Dk / total
            S_n = corr2_n / total
        else:
            S_Hs = S_Dk = S_n = 1.0 / 3.0
        
        # Identify dominant parameter
        indices = {"H_s": S_Hs, "D_k": S_Dk, "n": S_n}
        dominant = max(indices, key=indices.get)
        
        return {
            "S_Hs": float(S_Hs),
            "S_Dk": float(S_Dk),
            "S_n": float(S_n),
            "total": float(S_Hs + S_Dk + S_n),
            "dominant_parameter": dominant,
            "correlations": {
                "H_s": float(corr_Hs),
                "D_k": float(corr_Dk),
                "n": float(corr_n),
            },
        }


# ============================================================================
# Probability design
# ============================================================================

@dataclass
class ProbabilityDesign:
    """Probability-based design assessment.
    
    Compares the F_max distribution with DNV-RP-F110 load factors and
    computes the reliability index β.
    
    Attributes
    ----------
    F_max : array
        Ultimate capacity samples [N/m or N]
    """
    F_max: np.ndarray
    
    def compute(
        self,
        F_design: Optional[float] = None,
        load_factor: float = 1.15,
    ) -> dict:
        """Compute probability design metrics.
        
        The design load F_design is the load that the structure must resist.
        For probability-based design, we compare the capacity distribution
        (F_max) with the design load.
        
        The reliability index β measures how many standard deviations the
        mean capacity is above the design load:
            β = (μ_Fmax - F_design) / σ_Fmax
        
        For typical structural reliability:
        - β < 2: LOW reliability
        - 2 ≤ β < 3: MEDIUM reliability
        - 3 ≤ β < 4: HIGH reliability
        - β ≥ 4: VERY HIGH reliability
        
        Parameters
        ----------
        F_design : float, optional
            Design load. If None, uses P10 of F_max (conservative estimate).
        load_factor : float
            DNV-RP-F110 load factor (default 1.15)
            
        Returns
        -------
        dict with 'mu', 'sigma', 'P10', 'P50', 'P90', 'F_design', 'beta', 'reliability_class'
        """
        mu = float(np.mean(self.F_max))
        sigma = float(np.std(self.F_max))
        P10 = float(np.percentile(self.F_max, 10))
        P50 = float(np.percentile(self.F_max, 50))
        P90 = float(np.percentile(self.F_max, 90))
        
        if F_design is None:
            # Use P10 as conservative design load estimate
            # This represents the 10th percentile of capacity
            F_design = P10 / load_factor
        
        # Reliability index: β = (μ - F_design) / σ
        if sigma > 0:
            beta = (mu - F_design) / sigma
        else:
            beta = float('inf')
        
        # Reliability classification
        if beta < 2.0:
            reliability_class = "LOW"
        elif beta < 3.0:
            reliability_class = "MEDIUM"
        elif beta < 4.0:
            reliability_class = "HIGH"
        else:
            reliability_class = "VERY_HIGH"
        
        return {
            "mu": mu,
            "sigma": sigma,
            "P10": P10,
            "P50": P50,
            "P90": P90,
            "F_design": F_design,
            "load_factor": load_factor,
            "beta": float(beta),
            "reliability_class": reliability_class,
        }
