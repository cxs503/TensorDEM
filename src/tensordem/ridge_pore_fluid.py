"""Pore fluid module for ice ridge keel DEM — Terzaghi 1D consolidation coupling.

Physical background
-------------------
An ice ridge keel is a submerged granular assembly of broken ice blocks.  The
pore space between blocks is filled with seawater.  When the keel is loaded
(e.g. by a ship hull pushing against the ridge, or by self-weight consolidation),
the pore water pressure rises above the hydrostatic value — this is the **excess
pore water pressure** Δu.

According to Terzaghi's effective stress principle:

    σ' = σ - u = σ - (u_0 + Δu)

where:
    σ   = total stress
    σ'  = effective stress (carried by the particle skeleton)
    u_0 = hydrostatic pore pressure
    Δu  = excess pore pressure

The shear strength of the granular skeleton follows Mohr-Coulomb:

    τ = c' + σ' · tan(φ')

As Δu accumulates, σ' decreases, and so does the shear strength τ.  This is
the mechanism of **liquefaction** or **strength degradation** in saturated
granular media.

Terzaghi 1D consolidation theory
--------------------------------
The dissipation of excess pore pressure is governed by:

    ∂(Δu)/∂t = c_v · ∂²(Δu)/∂z²

where c_v is the coefficient of consolidation:

    c_v = k / (m_v · γ_w)

with:
    k   = hydraulic conductivity [m/s]
    m_v = coefficient of volume compressibility [m²/N]
    γ_w = unit weight of water [N/m³]

For a layer of thickness H with double drainage (drainage at top and bottom),
the characteristic drainage path is H_dr = H/2, and the time factor is:

    T_v = c_v · t / H_dr²

The degree of consolidation U (fraction of excess pore pressure dissipated) is:

    U = 1 - (8/π²) · Σ exp(-(2m+1)²π²T_v/4) / (2m+1)²    for m = 0, 1, 2, ...

For T_v > 0.2, the approximation U ≈ 1 - (8/π²)·exp(-π²T_v/4) is accurate.

Coupling with DEM
-----------------
In the DEM keel model, each particle (or layer of particles) carries:
- A position and velocity (mechanical DOF)
- An excess pore pressure Δu (hydraulic DOF)
- An effective stress σ' derived from the total stress and Δu

The coupling works as follows:
1. At each DEM step, the total stress σ is computed from contact forces.
2. The excess pore pressure Δu is updated by the consolidation equation.
3. The effective stress σ' = σ - Δu is used to scale the shear strength.
4. If σ' → 0 (full liquefaction), the shear strength → 0 and the skeleton
   loses its load-bearing capacity.

This module provides:
- :class:`PoreFluidState` — per-layer pore pressure and effective stress state
- :class:`TerzaghiConsolidation` — 1D consolidation solver
- :class:`KeelPoreFluidModel` — full coupling model for the keel DEM
- :func:`compute_excess_pore_pressure` — analytical solution for simple loading

References
----------
- Terzaghi, K. (1925). "Erdbaumechanik auf bodenphysikalischer Grundlage."
  Franz Deuticke, Wien.
- Terzaghi, K., Peck, R.B., & Mesri, G. (1996). "Soil Mechanics in
  Engineering Practice." 3rd ed., Wiley.
- Seed, H.B. & Idriss, I.M. (1971). "Simplified procedures for evaluating
  soil liquefaction potential." *J. Soil Mech. Found. Div.*, 97(SM9), 1249–1273.
- Hopkins, M.A. (1994). "On the ridging of intact lead ice." *J. Geophys.
  Res.*, 99(C11), 22383–22393.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "PoreFluidState",
    "TerzaghiConsolidation",
    "KeelPoreFluidModel",
    "PoreFluidConfig",
    "compute_excess_pore_pressure",
    "degree_of_consolidation",
    "hydraulic_conductivity_kozeny_carman",
    "effective_stress",
    "shear_strength_mohr_coulomb",
]

# Default physical constants
WATER_DENSITY = 1025.0     # kg/m³
GRAVITY = 9.81             # m/s²
WATER_UNIT_WEIGHT = WATER_DENSITY * GRAVITY  # N/m³


@dataclass
class PoreFluidState:
    """Pore fluid state for a single layer of the keel.

    Attributes
    ----------
    depth : float
        Depth below waterline [m].
    thickness : float
        Layer thickness [m].
    porosity : float
        Current porosity (void fraction).
    excess_pore_pressure : float
        Excess pore water pressure Δu [Pa], relative to hydrostatic.
    total_stress : float
        Total vertical stress σ [Pa].
    effective_stress : float
        Effective vertical stress σ' = σ - u_0 - Δu [Pa].
    shear_strength : float
        Current shear strength τ = c' + σ'·tan(φ') [Pa].
    """

    depth: float
    thickness: float
    porosity: float
    excess_pore_pressure: float = 0.0
    total_stress: float = 0.0
    effective_stress: float = 0.0
    shear_strength: float = 0.0

    @property
    def hydrostatic_pressure(self) -> float:
        """Hydrostatic pore pressure u_0 = ρ_w · g · depth."""
        return WATER_DENSITY * GRAVITY * self.depth

    @property
    def total_pore_pressure(self) -> float:
        """Total pore pressure u = u_0 + Δu."""
        return self.hydrostatic_pressure + self.excess_pore_pressure

    def update_effective_stress(self, cohesion: float = 0.0,
                                 friction_angle_deg: float = 35.0) -> None:
        """Update effective stress and shear strength from current state."""
        self.effective_stress = max(0.0, self.total_stress - self.total_pore_pressure)
        phi_rad = math.radians(friction_angle_deg)
        self.shear_strength = cohesion + self.effective_stress * math.tan(phi_rad)


def hydraulic_conductivity_kozeny_carman(porosity: float, d_mean: float,
                                          shape_factor: float = 0.2,
                                          mu_water: float = 1.79e-3) -> float:
    """Hydraulic conductivity from Kozeny-Carman equation.

    k = (g/μ) · (n³ / (1-n)²) · (d² / C)

    where:
        g   = gravitational acceleration
        μ   = dynamic viscosity of water
        n   = porosity
        d   = mean particle diameter
        C   = Kozeny constant (≈ 5 for spheres, shape_factor × 180)

    Parameters
    ----------
    porosity : float
        Porosity (void fraction).
    d_mean : float
        Mean particle diameter [m].
    shape_factor : float
        Shape factor (0.2 for spheres, higher for angular blocks).
    mu_water : float
        Dynamic viscosity of water [Pa·s].  Default: 1.79e-3 at 0°C.

    Returns
    -------
    k : float
        Hydraulic conductivity [m/s].
    """
    if not (0 < porosity < 1):
        raise ValueError("porosity must be in (0, 1)")
    if d_mean <= 0:
        raise ValueError("d_mean must be positive")

    kozeny_constant = shape_factor * 180.0  # ≈ 36 for spheres
    k = (GRAVITY / mu_water) * (porosity ** 3 / (1 - porosity) ** 2) * (d_mean ** 2 / kozeny_constant)
    return k


def degree_of_consolidation(T_v: float, n_terms: int = 20) -> float:
    """Degree of consolidation U from Terzaghi's series solution.

    U = 1 - (8/π²) · Σ exp(-(2m+1)²π²T_v/4) / (2m+1)²

    Parameters
    ----------
    T_v : float
        Time factor T_v = c_v · t / H_dr².
    n_terms : int
        Number of terms in the series (default 20).

    Returns
    -------
    U : float
        Degree of consolidation (0 ≤ U ≤ 1).
    """
    if T_v <= 0:
        return 0.0
    if T_v > 10:
        return 1.0  # Fully consolidated

    U = 0.0
    for m in range(n_terms):
        term = (2 * m + 1)
        U += math.exp(-(term ** 2) * math.pi ** 2 * T_v / 4) / (term ** 2)
    U = 1.0 - (8.0 / math.pi ** 2) * U
    return max(0.0, min(1.0, U))


def compute_excess_pore_pressure(surf_load: float, H: float, z: float,
                                  c_v: float, t: float,
                                  drainage: str = "double") -> float:
    """Analytical excess pore pressure from Terzaghi 1D consolidation.

    For a suddenly applied surface load q on a layer of thickness H:

    Δu(z, t) = q · Σ (2/M) · sin(M·z/H_dr) · exp(-M²·T_v)

    where M = (2m+1)·π/2, H_dr = drainage path length.

    Parameters
    ----------
    surf_load : float
        Applied surface load q [Pa].
    H : float
        Layer thickness [m].
    z : float
        Depth from top of layer [m].
    c_v : float
        Coefficient of consolidation [m²/s].
    t : float
        Time since load application [s].
    drainage : str
        Drainage condition: "single" (top only) or "double" (top and bottom).

    Returns
    -------
    Δu : float
        Excess pore pressure [Pa].
    """
    if t <= 0:
        return surf_load  # Initial condition: all load carried by pore water

    if drainage == "double":
        H_dr = H / 2.0
    else:
        H_dr = H

    T_v = c_v * t / (H_dr ** 2)

    if T_v > 10:
        return 0.0  # Fully dissipated

    delta_u = 0.0
    n_terms = 50
    for m in range(n_terms):
        M = (2 * m + 1) * math.pi / 2
        sin_term = math.sin(M * z / H_dr)
        exp_term = math.exp(-M ** 2 * T_v)
        delta_u += (2.0 / M) * sin_term * exp_term

    return surf_load * delta_u


@dataclass(frozen=True)
class PoreFluidConfig:
    """Configuration for the pore fluid model.

    Attributes
    ----------
    n_layers : int
        Number of vertical layers in the keel.
    keel_depth : float
        Total keel depth below waterline [m].
    porosity : float
        Initial porosity of the keel rubble.
    d_mean : float
        Mean ice block diameter [m].
    cohesion : float
        Effective cohesion c' [Pa].  For ice rubble, typically 0–10 kPa.
    friction_angle : float
        Effective friction angle φ' [deg].  Typical: 30–45°.
    c_v : float
        Coefficient of consolidation [m²/s].  If None, computed from
        Kozeny-Carman.
    drainage : str
        Drainage condition: "single" or "double".
    """

    n_layers: int = 10
    keel_depth: float = 5.0
    porosity: float = 0.35
    d_mean: float = 1.0
    cohesion: float = 5e3
    friction_angle: float = 35.0
    c_v: Optional[float] = None
    drainage: str = "double"

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.keel_depth <= 0:
            raise ValueError("keel_depth must be positive")
        if not (0 < self.porosity < 1):
            raise ValueError("porosity must be in (0, 1)")
        if self.d_mean <= 0:
            raise ValueError("d_mean must be positive")
        if self.cohesion < 0:
            raise ValueError("cohesion must be nonnegative")
        if not (0 < self.friction_angle < 90):
            raise ValueError("friction_angle must be in (0, 90)")
        if self.c_v is not None and self.c_v <= 0:
            raise ValueError("c_v must be positive")
        if self.drainage not in ("single", "double"):
            raise ValueError("drainage must be 'single' or 'double'")

    @property
    def layer_thickness(self) -> float:
        """Thickness of each layer [m]."""
        return self.keel_depth / self.n_layers

    @property
    def computed_c_v(self) -> float:
        """Coefficient of consolidation from Kozeny-Carman [m²/s]."""
        if self.c_v is not None:
            return self.c_v
        k = hydraulic_conductivity_kozeny_carman(self.porosity, self.d_mean)
        # m_v ≈ n / (E_s) where E_s is the constrained modulus
        # For ice rubble, E_s ≈ 10–100 MPa; use a typical value
        E_s = 50e6  # Pa
        m_v = self.porosity / E_s
        return k / (m_v * WATER_UNIT_WEIGHT)


class TerzaghiConsolidation:
    """1D Terzaghi consolidation solver for a single layer.

    Solves ∂(Δu)/∂t = c_v · ∂²(Δu)/∂z² with finite differences.
    """

    def __init__(self, c_v: float, H: float, n_nodes: int = 20,
                 drainage: str = "double"):
        self.c_v = c_v
        self.H = H
        self.n_nodes = n_nodes
        self.drainage = drainage
        self.dz = H / (n_nodes - 1) if n_nodes > 1 else H

        if drainage == "double":
            self.H_dr = H / 2.0
        else:
            self.H_dr = H

    def solve(self, initial_excess: np.ndarray, dt: float,
              n_steps: int) -> np.ndarray:
        """Solve consolidation for given initial excess pore pressure.

        Parameters
        ----------
        initial_excess : np.ndarray
            Initial excess pore pressure at each node [Pa].
        dt : float
            Time step [s].
        n_steps : int
            Number of time steps.

        Returns
        -------
        excess_history : np.ndarray
            Shape (n_steps+1, n_nodes), excess pore pressure history.
        """
        n = len(initial_excess)
        u = initial_excess.copy()
        history = np.zeros((n_steps + 1, n))
        history[0] = u.copy()

        r = self.c_v * dt / (self.dz ** 2)
        if r > 0.5:
            raise ValueError(f"FTCS stability violated: r={r:.3f} > 0.5")

        for step in range(1, n_steps + 1):
            u_new = u.copy()
            for i in range(1, n - 1):
                u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])

            # Boundary conditions: u = 0 at drainage boundaries
            if self.drainage == "double":
                u_new[0] = 0.0
                u_new[-1] = 0.0
            else:
                u_new[0] = 0.0  # Top drainage

            u = u_new
            history[step] = u.copy()

        return history


@dataclass
class KeelPoreFluidModel:
    """Full pore fluid model for the ice ridge keel.

    Manages the coupling between the DEM particle skeleton and the pore fluid.

    Attributes
    ----------
    config : PoreFluidConfig
        Configuration.
    layers : list[PoreFluidState]
        Per-layer pore fluid states.
    time : float
        Current simulation time [s].
    """

    config: PoreFluidConfig
    layers: List[PoreFluidState] = field(default_factory=list)
    time: float = 0.0

    def __post_init__(self) -> None:
        if not self.layers:
            self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers with hydrostatic conditions."""
        self.layers = []
        h = self.config.layer_thickness
        for i in range(self.config.n_layers):
            depth = (i + 0.5) * h  # depth at layer centre
            state = PoreFluidState(
                depth=depth,
                thickness=h,
                porosity=self.config.porosity,
                excess_pore_pressure=0.0,
                total_stress=0.0,
                effective_stress=0.0,
                shear_strength=0.0,
            )
            state.update_effective_stress(
                self.config.cohesion, self.config.friction_angle
            )
            self.layers.append(state)

    def apply_surface_load(self, load: float) -> None:
        """Apply a surface load and distribute excess pore pressure.

        Initially, the entire load is carried by the pore water (undrained
        response).  The excess pore pressure is distributed linearly with depth.

        Parameters
        ----------
        load : float
            Applied surface load [Pa].
        """
        for layer in self.layers:
            # Undrained response: excess pore pressure proportional to depth
            fraction = 1.0 - layer.depth / self.config.keel_depth
            layer.excess_pore_pressure = load * fraction
            layer.update_effective_stress(
                self.config.cohesion, self.config.friction_angle
            )

    def consolidate(self, dt: float) -> None:
        """Advance consolidation by one time step.

        Parameters
        ----------
        dt : float
            Time step [s].
        """
        c_v = self.config.computed_c_v
        H = self.config.keel_depth
        H_dr = H / 2.0 if self.config.drainage == "double" else H

        for layer in self.layers:
            if layer.excess_pore_pressure <= 0:
                continue

            T_v = c_v * dt / (H_dr ** 2)
            U = degree_of_consolidation(T_v)

            # Dissipate excess pore pressure
            layer.excess_pore_pressure *= (1 - U)
            layer.update_effective_stress(
                self.config.cohesion, self.config.friction_angle
            )

        self.time += dt

    def update_stresses_from_dem(self, contact_forces: np.ndarray,
                                  layer_areas: np.ndarray) -> None:
        """Update total stresses from DEM contact forces.

        Parameters
        ----------
        contact_forces : np.ndarray
            Vertical contact force per layer [N].
        layer_areas : np.ndarray
            Cross-sectional area per layer [m²].
        """
        for i, layer in enumerate(self.layers):
            if i < len(contact_forces) and i < len(layer_areas):
                layer.total_stress = contact_forces[i] / layer_areas[i]
            layer.update_effective_stress(
                self.config.cohesion, self.config.friction_angle
            )

    def get_strength_reduction_factor(self) -> np.ndarray:
        """Get shear strength reduction factor for each layer.

        Returns
        -------
        factors : np.ndarray
            Ratio of current shear strength to initial (drained) strength.
            Values < 1 indicate strength degradation due to excess pore pressure.
        """
        factors = np.zeros(self.config.n_layers)
        for i, layer in enumerate(self.layers):
            # Initial drained strength (no excess pore pressure)
            sigma_initial = layer.total_stress - layer.hydrostatic_pressure
            tau_initial = (self.config.cohesion +
                          max(0, sigma_initial) * math.tan(math.radians(self.config.friction_angle)))
            if tau_initial > 0:
                factors[i] = layer.shear_strength / tau_initial
            else:
                factors[i] = 1.0
        return factors

    def summary(self) -> dict:
        """Return summary statistics."""
        excess = [l.excess_pore_pressure for l in self.layers]
        eff_stress = [l.effective_stress for l in self.layers]
        strength = [l.shear_strength for l in self.layers]

        return {
            "time_s": self.time,
            "n_layers": self.config.n_layers,
            "keel_depth_m": self.config.keel_depth,
            "porosity": self.config.porosity,
            "c_v_m2_s": self.config.computed_c_v,
            "max_excess_pore_pressure_Pa": max(excess) if excess else 0.0,
            "mean_excess_pore_pressure_Pa": float(np.mean(excess)) if excess else 0.0,
            "min_effective_stress_Pa": min(eff_stress) if eff_stress else 0.0,
            "mean_effective_stress_Pa": float(np.mean(eff_stress)) if eff_stress else 0.0,
            "min_shear_strength_Pa": min(strength) if strength else 0.0,
            "mean_shear_strength_Pa": float(np.mean(strength)) if strength else 0.0,
            "drainage": self.config.drainage,
        }


def shear_strength_mohr_coulomb(effective_stress: float, cohesion: float = 0.0,
                                 friction_angle_deg: float = 35.0) -> float:
    """Mohr-Coulomb shear strength.

    τ = c' + σ' · tan(φ')

    Parameters
    ----------
    effective_stress : float
        Effective normal stress σ' [Pa].
    cohesion : float
        Effective cohesion c' [Pa].
    friction_angle_deg : float
        Effective friction angle φ' [deg].

    Returns
    -------
    tau : float
        Shear strength [Pa].
    """
    phi_rad = math.radians(friction_angle_deg)
    return cohesion + max(0.0, effective_stress) * math.tan(phi_rad)


def effective_stress(total_stress: float, pore_pressure: float) -> float:
    """Effective stress from Terzaghi's principle.

    σ' = σ - u

    Parameters
    ----------
    total_stress : float
        Total stress σ [Pa].
    pore_pressure : float
        Pore pressure u [Pa].

    Returns
    -------
    sigma_eff : float
        Effective stress σ' [Pa], clamped to ≥ 0.
    """
    return max(0.0, total_stress - pore_pressure)
