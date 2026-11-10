"""Parameter sensitivity: τ vs wave amplitude and grain size distribution.

The consolidation time scale τ depends on:
1. **Wave amplitude A**: larger waves → more vigorous shaking → faster
   consolidation → smaller τ
2. **Grain size distribution**: broader distributions allow smaller grains
   to fill voids between larger ones → faster consolidation → smaller τ
3. **Gravity g**: stronger gravity → faster settling → smaller τ
4. **Friction μ**: higher friction → slower rearrangement → larger τ

This module performs parametric studies to quantify these dependencies
and derive a parameterized formula for τ.

Expected scaling:
    τ ∝ 1 / A^α   (α ≈ 0.5–1.0 for vibratory compaction)
    τ ∝ σ_g^β     (β ≈ -0.3 to -0.5, where σ_g is geometric std dev)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple, Dict

import numpy as np

from .geometry import KeelGeometry
from .packing import KeelPackingConfig
from .consolidation import ConsolidationConfig
from .evolution import fit_exponential_decay, run_evolution_analysis

__all__ = [
    "SensitivityConfig",
    "SensitivityResult",
    "run_sensitivity_study",
    "tau_parameterization",
]


@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for parameter sensitivity study.

    Attributes
    ----------
    wave_amplitudes : list of float
        Wave amplitudes to test (m/s²). Default [0.1, 0.3, 0.5, 1.0, 2.0].
    d_min_values : list of float
        Minimum grain diameters to test (m). Default [0.1, 0.2, 0.3].
    d_max_values : list of float
        Maximum grain diameters to test (m). Default [0.3, 0.5, 0.8].
    n_particles : int
        Number of particles per run. Default 300.
    total_time : float
        Simulation time per run (s). Default 300.0.
    seed_base : int
        Base random seed. Default 42.
    """
    wave_amplitudes: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0, 2.0)
    d_min_values: Tuple[float, ...] = (0.1, 0.2, 0.3)
    d_max_values: Tuple[float, ...] = (0.3, 0.5, 0.8)
    n_particles: int = 300
    total_time: float = 300.0
    seed_base: int = 42


@dataclass
class SensitivityResult:
    """Result of parameter sensitivity study.

    Attributes
    ----------
    wave_sweep : list of dict
        Results for wave amplitude sweep.
    gradation_sweep : list of dict
        Results for grain size distribution sweep.
    tau_wave_params : dict
        Fitted parameters for τ(A) = c / A^α.
    tau_gradation_params : dict
        Fitted parameters for τ(gradation).
    """
    wave_sweep: List[Dict]
    gradation_sweep: List[Dict]
    tau_wave_params: Dict[str, float]
    tau_gradation_params: Dict[str, float]


def _run_single(
    geom: KeelGeometry,
    wave_amplitude: float,
    d_min: float,
    d_max: float,
    n_particles: int,
    total_time: float,
    seed: int,
) -> Dict:
    """Run a single consolidation simulation and extract τ."""
    packing_config = KeelPackingConfig(
        n_particles=n_particles,
        d_min=d_min,
        d_max=d_max,
        seed=seed,
    )
    consolidation_config = ConsolidationConfig(
        total_time=total_time,
        record_interval=total_time / 50,
        wave_amplitude=wave_amplitude,
        wave_period=10.0,
        seed=seed,
    )

    result = run_evolution_analysis(geom, packing_config, consolidation_config)

    return {
        "wave_amplitude": wave_amplitude,
        "d_min": d_min,
        "d_max": d_max,
        "n_0": result.n_0,
        "n_inf": result.n_inf,
        "tau": result.tau,
        "r_squared": result.r_squared,
        "residual_rms": result.residual_rms,
    }


def _fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit y = c * x^α in log-log space.

    Returns (c, α, r_squared).
    """
    # Filter positive values
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        return 1.0, 0.0, 0.0

    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    # Linear fit in log space: log(y) = log(c) + α * log(x)
    coeffs = np.polyfit(log_x, log_y, 1)
    α = coeffs[0]
    log_c = coeffs[1]
    c = np.exp(log_c)

    # R²
    y_fit = c * x[mask] ** α
    ss_res = np.sum((y[mask] - y_fit) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return c, α, r_squared


def run_sensitivity_study(
    geom: KeelGeometry = None,
    config: SensitivityConfig = None,
) -> SensitivityResult:
    """Run parameter sensitivity study.

    Performs two sweeps:
    1. Wave amplitude sweep (fixed gradation)
    2. Grain size distribution sweep (fixed wave amplitude)

    Parameters
    ----------
    geom : KeelGeometry, optional
        Keel geometry.
    config : SensitivityConfig, optional
        Sensitivity study configuration.

    Returns
    -------
    SensitivityResult
        Complete sensitivity analysis result.
    """
    if geom is None:
        geom = KeelGeometry(L_top=10.0, W_top=10.0, L_bot=2.0, W_bot=2.0, depth=5.0)

    if config is None:
        config = SensitivityConfig()

    # Wave amplitude sweep
    wave_sweep = []
    d_min_fixed = 0.2
    d_max_fixed = 0.5
    for i, A in enumerate(config.wave_amplitudes):
        result = _run_single(
            geom, A, d_min_fixed, d_max_fixed,
            config.n_particles, config.total_time,
            config.seed_base + i,
        )
        wave_sweep.append(result)

    # Fit τ(A) = c / A^α
    A_values = np.array([r["wave_amplitude"] for r in wave_sweep])
    tau_values = np.array([r["tau"] for r in wave_sweep])
    c_wave, alpha_wave, r2_wave = _fit_power_law(A_values, tau_values)
    # Convert to τ = c * A^α (α should be negative)
    tau_wave_params = {
        "c": c_wave,
        "alpha": alpha_wave,
        "r_squared": r2_wave,
        "formula": f"tau = {c_wave:.2f} * A^({alpha_wave:.2f})",
    }

    # Gradation sweep (fixed wave amplitude)
    gradation_sweep = []
    A_fixed = 0.5
    idx = 0
    for d_min in config.d_min_values:
        for d_max in config.d_max_values:
            if d_max <= d_min:
                continue
            result = _run_single(
                geom, A_fixed, d_min, d_max,
                config.n_particles, config.total_time,
                config.seed_base + 100 + idx,
            )
            result["gradation_ratio"] = d_max / d_min
            gradation_sweep.append(result)
            idx += 1

    # Fit τ vs gradation ratio
    ratios = np.array([r["gradation_ratio"] for r in gradation_sweep])
    tau_grad = np.array([r["tau"] for r in gradation_sweep])
    c_grad, alpha_grad, r2_grad = _fit_power_law(ratios, tau_grad)
    tau_gradation_params = {
        "c": c_grad,
        "alpha": alpha_grad,
        "r_squared": r2_grad,
        "formula": f"tau = {c_grad:.2f} * (d_max/d_min)^({alpha_grad:.2f})",
    }

    return SensitivityResult(
        wave_sweep=wave_sweep,
        gradation_sweep=gradation_sweep,
        tau_wave_params=tau_wave_params,
        tau_gradation_params=tau_gradation_params,
    )


def tau_parameterization(
    wave_amplitude: float,
    gradation_ratio: float,
    tau_wave_params: Dict[str, float],
    tau_gradation_params: Dict[str, float],
) -> float:
    """Compute τ from parameterized formula.

    τ(A, d_max/d_min) = τ_0 * (A/A_0)^α_A * (d_max/d_min)^α_g

    This is a simplified multiplicative model. For more accuracy,
    a full 2D response surface should be fitted.

    Parameters
    ----------
    wave_amplitude : float
        Wave acceleration amplitude (m/s²).
    gradation_ratio : float
        Ratio d_max / d_min.
    tau_wave_params : dict
        Parameters from wave sweep fit.
    tau_gradation_params : dict
        Parameters from gradation sweep fit.

    Returns
    -------
    tau : float
        Predicted consolidation time scale (s).
    """
    # Reference values
    A_ref = 0.5  # m/s²
    ratio_ref = 2.5  # d_max/d_min

    # Base τ (from reference conditions)
    tau_0 = 200.0  # s (typical value)

    # Wave amplitude effect
    tau_A = tau_wave_params["c"] * wave_amplitude ** tau_wave_params["alpha"]

    # Gradation effect
    tau_g = tau_gradation_params["c"] * gradation_ratio ** tau_gradation_params["alpha"]

    # Combined (geometric mean as simple combination)
    tau = math.sqrt(tau_A * tau_g)

    return tau
