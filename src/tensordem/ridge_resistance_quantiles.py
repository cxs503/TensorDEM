"""Ridge resistance and grounding depth quantile analysis.

Physical background
-------------------
Ice ridge resistance is inherently variable due to the stochastic nature of
ridge geometry (sail height, keel depth, porosity) and material properties
(ice strength, friction angle).  This module computes statistical distributions
(quantiles) of ridge resistance and grounding depth for engineering design.

The analysis combines:
1. **Sail resistance** (Kujala 1994 / Riska 1997) — analytical closed-form
2. **Keel resistance** — from DEM simulation with pore fluid coupling
3. **Grounding depth** — the depth at which the keel contacts the seabed

Quantile computation
--------------------
For a given set of input parameter distributions, the module computes:
- P10, P50, P90 quantiles of total ridge resistance
- P10, P50, P90 quantiles of grounding depth
- Sensitivity indices (which parameters dominate the variance)

This is essential for design codes (e.g. ISO 19906) which require resistance
values at specific exceedance probabilities.

References
----------
- ISO 19906:2019. "Petroleum and natural gas industries — Arctic offshore
  structures."
- Kujala, K. (1994). "Ice ridge loads on ship hulls."
- Riska, K. (1997). "A model for ice ridge keel resistance."
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ridge_sail_resistance import (
    SailResistanceConfig,
    SailResistanceResult,
    compute_sail_resistance,
)
from .ridge_pore_fluid import (
    PoreFluidConfig,
    KeelPoreFluidModel,
    effective_stress,
    shear_strength_mohr_coulomb,
)

__all__ = [
    "RidgeResistanceQuantileConfig",
    "RidgeResistanceQuantileResult",
    "compute_ridge_resistance_quantiles",
    "compute_grounding_depth_quantiles",
    "sensitivity_analysis",
]


@dataclass(frozen=True)
class RidgeResistanceQuantileConfig:
    """Configuration for ridge resistance quantile analysis.

    Attributes
    ----------
    n_samples : int
        Number of Monte Carlo samples.
    sail_height_mean : float
        Mean sail height [m].
    sail_height_std : float
        Standard deviation of sail height [m].
    keel_depth_mean : float
        Mean keel depth [m].
    keel_depth_std : float
        Standard deviation of keel depth [m].
    porosity_mean : float
        Mean porosity.
    porosity_std : float
        Standard deviation of porosity.
    sigma_f_mean : float
        Mean ice flexural strength [Pa].
    sigma_f_std : float
        Standard deviation of ice flexural strength [Pa].
    B : float
        Ship beam [m].
    alpha : float
        Hull waterline angle [deg].
    phi_mean : float
        Mean friction angle [deg].
    phi_std : float
        Standard deviation of friction angle [deg].
    seed : int
        Random seed for reproducibility.
    """

    n_samples: int = 1000
    sail_height_mean: float = 2.0
    sail_height_std: float = 0.5
    keel_depth_mean: float = 10.0
    keel_depth_std: float = 2.5
    porosity_mean: float = 0.35
    porosity_std: float = 0.08
    sigma_f_mean: float = 500e3
    sigma_f_std: float = 150e3
    B: float = 20.0
    alpha: float = 30.0
    phi_mean: float = 40.0
    phi_std: float = 5.0
    seed: int = 20261111

    def __post_init__(self) -> None:
        if self.n_samples < 10:
            raise ValueError("n_samples must be >= 10")
        for name in ("sail_height_mean", "keel_depth_mean", "B"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("sail_height_std", "keel_depth_std", "porosity_std",
                     "sigma_f_std", "phi_std"):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass
class RidgeResistanceQuantileResult:
    """Result of ridge resistance quantile analysis.

    Attributes
    ----------
    F_sail_samples : np.ndarray
        Sail resistance samples [N].
    F_keel_samples : np.ndarray
        Keel resistance samples [N].
    F_total_samples : np.ndarray
        Total resistance samples [N].
    grounding_depth_samples : np.ndarray
        Grounding depth samples [N].
    quantiles : dict
        Dictionary with keys 'P10', 'P50', 'P90' for each quantity.
    statistics : dict
        Mean, std, min, max for each quantity.
    """

    F_sail_samples: np.ndarray
    F_keel_samples: np.ndarray
    F_total_samples: np.ndarray
    grounding_depth_samples: np.ndarray
    quantiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_quantiles(self) -> None:
        """Compute P10, P50, P90 quantiles."""
        for name, arr in [
            ("F_sail", self.F_sail_samples),
            ("F_keel", self.F_keel_samples),
            ("F_total", self.F_total_samples),
            ("grounding_depth", self.grounding_depth_samples),
        ]:
            self.quantiles[name] = {
                "P10": float(np.percentile(arr, 10)),
                "P50": float(np.percentile(arr, 50)),
                "P90": float(np.percentile(arr, 90)),
            }
            self.statistics[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }


def _sample_truncated_normal(mean: float, std: float, n: int,
                              low: float = 0.0, high: Optional[float] = None,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample from a truncated normal distribution."""
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(mean, std, n * 2)  # oversample
    samples = samples[(samples >= low) & (high is None or samples <= high)]
    if len(samples) < n:
        # Fallback: clip
        samples = rng.normal(mean, std, n)
        samples = np.clip(samples, low, high if high is not None else np.inf)
    return samples[:n]


def compute_ridge_resistance_quantiles(
    config: RidgeResistanceQuantileConfig
) -> RidgeResistanceQuantileResult:
    """Compute quantiles of ridge resistance via Monte Carlo.

    Parameters
    ----------
    config : RidgeResistanceQuantileConfig
        Analysis configuration.

    Returns
    -------
    RidgeResistanceQuantileResult
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    # Sample input parameters
    h_s = _sample_truncated_normal(
        config.sail_height_mean, config.sail_height_std, n, low=0.1, rng=rng
    )
    h_k = _sample_truncated_normal(
        config.keel_depth_mean, config.keel_depth_std, n, low=1.0, rng=rng
    )
    porosity = _sample_truncated_normal(
        config.porosity_mean, config.porosity_std, n, low=0.1, high=0.6, rng=rng
    )
    sigma_f = _sample_truncated_normal(
        config.sigma_f_mean, config.sigma_f_std, n, low=50e3, rng=rng
    )
    phi = _sample_truncated_normal(
        config.phi_mean, config.phi_std, n, low=20, high=60, rng=rng
    )

    F_sail_samples = np.zeros(n)
    F_keel_samples = np.zeros(n)
    grounding_depth_samples = np.zeros(n)

    for i in range(n):
        # Sail resistance (analytical)
        sail_cfg = SailResistanceConfig(
            h_s=float(h_s[i]),
            B=config.B,
            alpha=config.alpha,
            sigma_f=float(sigma_f[i]),
            n=float(porosity[i]),
            phi=float(phi[i]),
        )
        sail_result = compute_sail_resistance(sail_cfg)
        F_sail_samples[i] = sail_result.F_sail

        # Keel resistance (simplified analytical model with pore fluid effect)
        # F_keel = buoyancy + breaking + pore fluid reduction
        rho_w = 1025.0
        rho_i = 917.0
        g = 9.81

        # Buoyancy component
        V_keel = 0.5 * h_k[i] * config.B * config.B * 0.3  # simplified volume
        F_buoyancy = rho_w * g * V_keel

        # Breaking component (similar to sail but with keel parameters)
        F_keel_breaking = 0.5 * sigma_f[i] * h_k[i] ** 2 * config.B / math.sin(math.radians(config.alpha))

        # Pore fluid strength reduction factor
        pore_cfg = PoreFluidConfig(
            n_layers=5,
            keel_depth=float(h_k[i]),
            porosity=float(porosity[i]),
            d_mean=1.0,
            cohesion=5e3,
            friction_angle=float(phi[i]),
        )
        pore_model = KeelPoreFluidModel(config=pore_cfg)
        # Apply a representative load
        applied_load = F_buoyancy / (config.B * config.B * 0.3)
        pore_model.apply_surface_load(applied_load)
        # Consolidate for a representative time
        pore_model.consolidate(dt=3600.0)  # 1 hour
        strength_factors = pore_model.get_strength_reduction_factor()
        mean_reduction = float(np.mean(strength_factors))

        # Keel resistance with pore fluid effect
        F_keel_samples[i] = (F_buoyancy + F_keel_breaking) * mean_reduction

        # Grounding depth: keel depth minus consolidation settlement
        # Settlement ≈ initial porosity change × keel depth
        settlement = porosity[i] * 0.1 * h_k[i]  # ~10% porosity reduction
        grounding_depth_samples[i] = h_k[i] - settlement

    result = RidgeResistanceQuantileResult(
        F_sail_samples=F_sail_samples,
        F_keel_samples=F_keel_samples,
        F_total_samples=F_sail_samples + F_keel_samples,
        grounding_depth_samples=grounding_depth_samples,
    )
    result.compute_quantiles()
    return result


def compute_grounding_depth_quantiles(
    keel_depth_mean: float = 10.0,
    keel_depth_std: float = 2.5,
    porosity_mean: float = 0.35,
    porosity_std: float = 0.08,
    n_samples: int = 1000,
    seed: int = 20261111,
) -> Dict[str, float]:
    """Compute quantiles of grounding depth.

    Parameters
    ----------
    keel_depth_mean : float
        Mean keel depth [m].
    keel_depth_std : float
        Standard deviation of keel depth [m].
    porosity_mean : float
        Mean porosity.
    porosity_std : float
        Standard deviation of porosity.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.

    Returns
    -------
    quantiles : dict
        P10, P50, P90 of grounding depth [m].
    """
    rng = np.random.default_rng(seed)
    h_k = _sample_truncated_normal(keel_depth_mean, keel_depth_std, n_samples,
                                    low=1.0, rng=rng)
    porosity = _sample_truncated_normal(porosity_mean, porosity_std, n_samples,
                                         low=0.1, high=0.6, rng=rng)

    grounding_depths = h_k - porosity * 0.1 * h_k  # settlement

    return {
        "P10": float(np.percentile(grounding_depths, 10)),
        "P50": float(np.percentile(grounding_depths, 50)),
        "P90": float(np.percentile(grounding_depths, 90)),
        "mean": float(np.mean(grounding_depths)),
        "std": float(np.std(grounding_depths)),
    }


def sensitivity_analysis(
    config: RidgeResistanceQuantileConfig,
    n_samples: int = 500,
) -> Dict[str, Dict[str, float]]:
    """One-at-a-time sensitivity analysis of ridge resistance.

    For each input parameter, compute the change in P50 total resistance
    when the parameter is varied by ±10%.

    Parameters
    ----------
    config : RidgeResistanceQuantileConfig
        Base configuration.
    n_samples : int
        Number of samples per sensitivity run.

    Returns
    -------
    sensitivity : dict
        For each parameter: {'plus_10pct': ..., 'minus_10pct': ...,
        'sensitivity_index': ...}
    """
    # Base case
    base_result = compute_ridge_resistance_quantiles(config)
    base_P50 = base_result.quantiles["F_total"]["P50"]

    sensitivity = {}
    params = [
        ("sail_height_mean", "sail_height_mean"),
        ("keel_depth_mean", "keel_depth_mean"),
        ("porosity_mean", "porosity_mean"),
        ("sigma_f_mean", "sigma_f_mean"),
        ("phi_mean", "phi_mean"),
    ]

    for attr, name in params:
        base_val = getattr(config, attr)
        delta = 0.1 * base_val

        # +10%
        cfg_plus = config.__class__(**{**config.__dict__, attr: base_val + delta})
        result_plus = compute_ridge_resistance_quantiles(cfg_plus)
        plus_P50 = result_plus.quantiles["F_total"]["P50"]

        # -10%
        cfg_minus = config.__class__(**{**config.__dict__, attr: base_val - delta})
        result_minus = compute_ridge_resistance_quantiles(cfg_minus)
        minus_P50 = result_minus.quantiles["F_total"]["P50"]

        sensitivity[name] = {
            "base_P50": base_P50,
            "plus_10pct_P50": plus_P50,
            "minus_10pct_P50": minus_P50,
            "sensitivity_index": (plus_P50 - minus_P50) / (2 * base_P50),
        }

    return sensitivity
