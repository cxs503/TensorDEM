"""Visualization of ice strength surfaces and MYI/FYI comparison.

This module generates publication-quality figures for:
1. 3D strength-temperature-salinity surface
2. MYI vs FYI strength comparison
3. Uncertainty confidence bands
4. Temperature and salinity trend curves

All figures are generated using matplotlib and saved as PNG files.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple
import os

from .compressive_strength import (
    compressive_strength_fyi,
    compressive_strength_myi,
    strength_temperature_curve,
    strength_salinity_curve,
)
from .uncertainty import strength_confidence_band


def plot_strength_surface_3d(
    T_range: Tuple[float, float] = (-20, -1),
    S_range: Tuple[float, float] = (0, 15),
    n_T: int = 30,
    n_S: int = 30,
    ice_type: str = "FYI",
    f_desal: float = 0.3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 3D strength-temperature-salinity surface.

    Parameters
    ----------
    T_range : tuple
        Temperature range (min, max) in °C.
    S_range : tuple
        Salinity range (min, max) in ppt.
    n_T : int
        Number of temperature points.
    n_S : int
        Number of salinity points.
    ice_type : str
        "FYI" or "MYI".
    f_desal : float
        Desalination factor (only for MYI).
    save_path : str, optional
        Path to save figure. If None, figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    T_arr = np.linspace(T_range[0], T_range[1], n_T)
    S_arr = np.linspace(S_range[0], S_range[1], n_S)
    T_grid, S_grid = np.meshgrid(T_arr, S_arr)

    # Compute strength on grid
    sigma_grid = np.zeros_like(T_grid)
    for i in range(n_S):
        for j in range(n_T):
            if ice_type.upper() == "FYI":
                sigma_grid[i, j] = compressive_strength_fyi(
                    T_grid[i, j], S_grid[i, j], extrapolate=True
                )
            else:
                sigma_grid[i, j] = compressive_strength_myi(
                    T_grid[i, j], S_grid[i, j], f_desal=f_desal, extrapolate=True
                )

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        T_grid, S_grid, sigma_grid,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Salinity (ppt)")
    ax.set_zlabel("Compressive Strength (MPa)")
    ax.set_title(f"{ice_type} Compressive Strength Surface")

    fig.colorbar(surf, shrink=0.5, aspect=5, label="Strength (MPa)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_myi_fyi_comparison(
    temperature: float = -10.0,
    S_range: Tuple[float, float] = (0, 15),
    n_S: int = 30,
    f_desal: float = 0.3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot MYI vs FYI strength comparison at fixed temperature.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    S_range : tuple
        Salinity range (min, max) in ppt.
    n_S : int
        Number of salinity points.
    f_desal : float
        Desalination factor for MYI.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    S_arr = np.linspace(S_range[0], S_range[1], n_S)

    sigma_fyi = compressive_strength_fyi(temperature, S_arr, extrapolate=True)
    sigma_myi = compressive_strength_myi(
        temperature, S_arr, f_desal=f_desal, extrapolate=True
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(S_arr, sigma_fyi, "b-o", label="FYI", linewidth=2, markersize=4)
    ax.plot(S_arr, sigma_myi, "r-s", label=f"MYI (f_desal={f_desal})", linewidth=2, markersize=4)

    ax.set_xlabel("Salinity (ppt)", fontsize=12)
    ax.set_ylabel("Compressive Strength (MPa)", fontsize=12)
    ax.set_title(f"MYI vs FYI Strength at T = {temperature}°C", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S_range)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_uncertainty_band(
    T_range: Tuple[float, float] = (-20, -1),
    n_T: int = 20,
    salinity: float = 5.0,
    ice_type: str = "FYI",
    T_std: float = 0.5,
    S_std: float = 0.5,
    n_samples: int = 500,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot strength vs temperature with uncertainty confidence band.

    Parameters
    ----------
    T_range : tuple
        Temperature range (min, max) in °C.
    n_T : int
        Number of temperature points.
    salinity : float
        Mean salinity in ppt.
    ice_type : str
        "FYI" or "MYI".
    T_std : float
        Temperature uncertainty std (°C).
    S_std : float
        Salinity uncertainty std (ppt).
    n_samples : int
        Number of Monte Carlo samples per temperature.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    T_arr = np.linspace(T_range[0], T_range[1], n_T)

    band = strength_confidence_band(
        T_arr, salinity, ice_type=ice_type,
        T_std=T_std, S_std=S_std,
        n_samples_per_T=n_samples,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot mean
    ax.plot(band["temperatures"], band["mean"], "b-", linewidth=2, label="Mean")

    # Plot 90% confidence band (5th-95th percentile)
    ax.fill_between(
        band["temperatures"],
        band["p05"],
        band["p95"],
        alpha=0.3,
        color="blue",
        label="90% CI (5th-95th)",
    )

    # Plot median
    ax.plot(
        band["temperatures"], band["p50"],
        "b--", linewidth=1, alpha=0.7, label="Median"
    )

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Compressive Strength (MPa)", fontsize=12)
    ax.set_title(f"{ice_type} Strength with Uncertainty (S={salinity} ppt)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_temperature_trend(
    salinity: float = 5.0,
    T_range: Tuple[float, float] = (-40, -1),
    n_T: int = 40,
    ice_type: str = "FYI",
    f_desal: float = 0.3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot strength vs temperature at fixed salinity.

    Parameters
    ----------
    salinity : float
        Salinity in ppt.
    T_range : tuple
        Temperature range (min, max) in °C.
    n_T : int
        Number of temperature points.
    ice_type : str
        "FYI" or "MYI".
    f_desal : float
        Desalination factor (only for MYI).
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    T_arr, sigma_arr = strength_temperature_curve(
        salinity, temperatures=np.linspace(T_range[0], T_range[1], n_T),
        ice_type=ice_type, f_desal=f_desal,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_arr, sigma_arr, "b-o", linewidth=2, markersize=4)

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Compressive Strength (MPa)", fontsize=12)
    ax.set_title(f"{ice_type} Strength vs Temperature (S={salinity} ppt)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T_range)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_salinity_trend(
    temperature: float = -10.0,
    S_range: Tuple[float, float] = (0, 15),
    n_S: int = 30,
    ice_type: str = "FYI",
    f_desal: float = 0.3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot strength vs salinity at fixed temperature.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    S_range : tuple
        Salinity range (min, max) in ppt.
    n_S : int
        Number of salinity points.
    ice_type : str
        "FYI" or "MYI".
    f_desal : float
        Desalination factor (only for MYI).
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    S_arr, sigma_arr = strength_salinity_curve(
        temperature, salinities=np.linspace(S_range[0], S_range[1], n_S),
        ice_type=ice_type, f_desal=f_desal,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(S_arr, sigma_arr, "r-s", linewidth=2, markersize=4)

    ax.set_xlabel("Salinity (ppt)", fontsize=12)
    ax.set_ylabel("Compressive Strength (MPa)", fontsize=12)
    ax.set_title(f"{ice_type} Strength vs Salinity (T={temperature}°C)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S_range)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_all_figures(output_dir: str = "artifacts") -> dict:
    """Generate all standard figures for ice material constitutive model.

    Parameters
    ----------
    output_dir : str
        Directory to save figures.

    Returns
    -------
    paths : dict
        Dictionary mapping figure names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # 1. 3D surface for FYI
    fig1 = plot_strength_surface_3d(
        ice_type="FYI",
        save_path=os.path.join(output_dir, "strength_surface_fyi_3d.png"),
    )
    paths["strength_surface_fyi_3d"] = os.path.join(output_dir, "strength_surface_fyi_3d.png")
    plt.close(fig1)

    # 2. 3D surface for MYI
    fig2 = plot_strength_surface_3d(
        ice_type="MYI", f_desal=0.3,
        save_path=os.path.join(output_dir, "strength_surface_myi_3d.png"),
    )
    paths["strength_surface_myi_3d"] = os.path.join(output_dir, "strength_surface_myi_3d.png")
    plt.close(fig2)

    # 3. MYI vs FYI comparison
    fig3 = plot_myi_fyi_comparison(
        temperature=-10.0, f_desal=0.3,
        save_path=os.path.join(output_dir, "myi_fyi_comparison.png"),
    )
    paths["myi_fyi_comparison"] = os.path.join(output_dir, "myi_fyi_comparison.png")
    plt.close(fig3)

    # 4. Uncertainty band for FYI
    fig4 = plot_uncertainty_band(
        salinity=5.0, ice_type="FYI",
        save_path=os.path.join(output_dir, "uncertainty_band_fyi.png"),
    )
    paths["uncertainty_band_fyi"] = os.path.join(output_dir, "uncertainty_band_fyi.png")
    plt.close(fig4)

    # 5. Temperature trend for FYI
    fig5 = plot_temperature_trend(
        salinity=5.0, ice_type="FYI", T_range=(-30, -1),
        save_path=os.path.join(output_dir, "temperature_trend_fyi.png"),
    )
    paths["temperature_trend_fyi"] = os.path.join(output_dir, "temperature_trend_fyi.png")
    plt.close(fig5)

    # 6. Salinity trend for FYI
    fig6 = plot_salinity_trend(
        temperature=-10.0, ice_type="FYI",
        save_path=os.path.join(output_dir, "salinity_trend_fyi.png"),
    )
    paths["salinity_trend_fyi"] = os.path.join(output_dir, "salinity_trend_fyi.png")
    plt.close(fig6)

    return paths
