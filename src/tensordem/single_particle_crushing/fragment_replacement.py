"""Fragment replacement with mass and momentum conservation ledger.

When a parent particle fractures in DEM, it is replaced by N child fragments.
Physical consistency requires:

1. **Mass conservation**: Σ m_i = m_parent
2. **Momentum conservation**: Σ m_i v_i = m_parent v_parent + Δp_fracture

where Δp_fracture accounts for any impulse imparted during fracture (e.g.,
from elastic energy release). In the simplest case of no external impulse,
Δp_fracture = 0 and momentum is exactly conserved.

This module provides:
  - Fragment generation with exact mass conservation
  - Velocity assignment with exact momentum conservation
  - A conservation ledger that tracks cumulative errors over many events
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


@dataclass
class FragmentReplacement:
    """Result of replacing a parent particle with fragments.

    Attributes
    ----------
    parent_mass : float
        Mass of the parent particle (kg).
    parent_velocity : np.ndarray
        Velocity of the parent particle (m/s), shape (3,).
    fragment_masses : np.ndarray
        Masses of the child fragments (kg), shape (N,).
    fragment_velocities : np.ndarray
        Velocities of the child fragments (m/s), shape (N, 3).
    fragment_radii : np.ndarray
        Radii of the child fragments (m), shape (N,).
    mass_error : float
        Absolute mass conservation error: |Σ m_i − m_parent| (kg).
    momentum_error : np.ndarray
        Momentum conservation error vector: Σ m_i v_i − m_parent v_parent (kg·m/s).
    """
    parent_mass: float
    parent_velocity: np.ndarray
    fragment_masses: np.ndarray
    fragment_velocities: np.ndarray
    fragment_radii: np.ndarray
    mass_error: float = 0.0
    momentum_error: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def n_fragments(self) -> int:
        return len(self.fragment_masses)

    @property
    def mass_error_relative(self) -> float:
        """Relative mass error: |Σ m_i − m_parent| / m_parent."""
        if self.parent_mass == 0:
            return 0.0
        return abs(self.mass_error) / self.parent_mass

    @property
    def momentum_error_norm(self) -> float:
        """Norm of momentum error vector."""
        return float(np.linalg.norm(self.momentum_error))

    @property
    def momentum_error_relative(self) -> float:
        """Relative momentum error: |Δp| / |m_parent v_parent|."""
        p_parent = self.parent_mass * np.linalg.norm(self.parent_velocity)
        if p_parent == 0:
            return self.momentum_error_norm
        return self.momentum_error_norm / p_parent


class ConservationLedger:
    """Tracks mass and momentum conservation over many fracture events.

    This ledger accumulates the conservation errors from each fragment
    replacement event and provides summary statistics.
    """

    def __init__(self) -> None:
        self.n_events: int = 0
        self.total_parent_mass: float = 0.0
        self.total_mass_error: float = 0.0
        self.total_momentum_error: np.ndarray = np.zeros(3)
        self.max_mass_error_relative: float = 0.0
        self.max_momentum_error_relative: float = 0.0
        self._mass_errors: list[float] = []
        self._momentum_errors: list[float] = []

    def record(self, replacement: FragmentReplacement) -> None:
        """Record a fragment replacement event."""
        self.n_events += 1
        self.total_parent_mass += replacement.parent_mass
        self.total_mass_error += replacement.mass_error
        self.total_momentum_error += replacement.momentum_error

        rel_mass = replacement.mass_error_relative
        rel_mom = replacement.momentum_error_relative
        self.max_mass_error_relative = max(self.max_mass_error_relative, rel_mass)
        self.max_momentum_error_relative = max(self.max_momentum_error_relative, rel_mom)
        self._mass_errors.append(rel_mass)
        self._momentum_errors.append(rel_mom)

    @property
    def mean_mass_error_relative(self) -> float:
        if self.n_events == 0:
            return 0.0
        return float(np.mean(self._mass_errors))

    @property
    def mean_momentum_error_relative(self) -> float:
        if self.n_events == 0:
            return 0.0
        return float(np.mean(self._momentum_errors))

    def summary(self) -> dict:
        """Return a summary dictionary of the ledger statistics."""
        return {
            "n_events": self.n_events,
            "total_parent_mass_kg": self.total_parent_mass,
            "total_mass_error_kg": self.total_mass_error,
            "mean_mass_error_relative": self.mean_mass_error_relative,
            "max_mass_error_relative": self.max_mass_error_relative,
            "total_momentum_error_kg_m_s": self.total_momentum_error.tolist(),
            "mean_momentum_error_relative": self.mean_momentum_error_relative,
            "max_momentum_error_relative": self.max_momentum_error_relative,
        }


def generate_fragment_masses(
    parent_mass: float,
    n_fragments: int,
    size_distribution: str = "uniform",
    alpha: float = 2.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate fragment masses that exactly sum to parent_mass.

    Parameters
    ----------
    parent_mass : float
        Mass of the parent particle (kg).
    n_fragments : int
        Number of child fragments to generate.
    size_distribution : str
        Type of size distribution: 'uniform', 'power_law', or 'gaudin'.
    alpha : float
        Exponent for power-law/Gaudin distribution.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Fragment masses (kg), shape (n_fragments,), summing exactly to parent_mass.
    """
    if parent_mass <= 0:
        raise ValueError("parent_mass must be positive")
    if n_fragments < 1:
        raise ValueError("n_fragments must be at least 1")
    if rng is None:
        rng = np.random.default_rng(42)

    if n_fragments == 1:
        return np.array([parent_mass])

    if size_distribution == "uniform":
        weights = rng.random(n_fragments)
    elif size_distribution == "power_law":
        # Power-law distributed fragment sizes
        u = rng.random(n_fragments)
        weights = u ** (-1.0 / alpha)
    elif size_distribution == "gaudin":
        # Gaudin-Schuhmann distributed fragment sizes
        u = rng.random(n_fragments)
        # For Gaudin CDF: P(d) = (d/d_max)^α, inverse: d = d_max * P^(1/α)
        weights = u ** (1.0 / alpha)
    else:
        raise ValueError(f"Unknown size_distribution: {size_distribution}")

    # Normalize to sum to parent_mass
    weights = weights / weights.sum()
    masses = parent_mass * weights

    # Ensure exact conservation by adjusting the largest fragment
    mass_sum = masses.sum()
    correction = parent_mass - mass_sum
    max_idx = np.argmax(masses)
    masses[max_idx] += correction

    return masses


def masses_to_radii(masses: np.ndarray, density: float) -> np.ndarray:
    """Convert fragment masses to equivalent spherical radii.

    Parameters
    ----------
    masses : np.ndarray
        Fragment masses (kg).
    density : float
        Material density (kg/m^3).

    Returns
    -------
    np.ndarray
        Equivalent spherical radii (m).
    """
    volumes = masses / density
    radii = (0.75 * volumes / math.pi) ** (1.0 / 3.0)
    return radii


def generate_fragment_velocities(
    parent_velocity: np.ndarray,
    fragment_masses: np.ndarray,
    parent_mass: float,
    fracture_impulse: np.ndarray | None = None,
    velocity_perturbation: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate fragment velocities conserving momentum.

    The fragment velocities are assigned such that:
        Σ m_i v_i = m_parent v_parent + Δp_fracture

    Parameters
    ----------
    parent_velocity : np.ndarray
        Parent particle velocity (m/s), shape (3,).
    fragment_masses : np.ndarray
        Fragment masses (kg), shape (N,).
    parent_mass : float
        Parent particle mass (kg).
    fracture_impulse : np.ndarray, optional
        Impulse imparted during fracture (kg·m/s), shape (3,).
        If None, no impulse (pure momentum conservation).
    velocity_perturbation : float
        Magnitude of random velocity perturbation (m/s).
        Fragments get random velocities around the parent velocity,
        adjusted to conserve momentum.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Fragment velocities (m/s), shape (N, 3).
    """
    n = len(fragment_masses)
    if rng is None:
        rng = np.random.default_rng(42)

    parent_velocity = np.asarray(parent_velocity, dtype=np.float64)
    if fracture_impulse is None:
        fracture_impulse = np.zeros(3)
    else:
        fracture_impulse = np.asarray(fracture_impulse, dtype=np.float64)

    # Target total momentum
    target_momentum = parent_mass * parent_velocity + fracture_impulse

    if velocity_perturbation > 0 and n > 1:
        # Assign random velocities around parent velocity
        velocities = np.tile(parent_velocity, (n, 1))
        perturbations = rng.standard_normal((n, 3)) * velocity_perturbation
        velocities += perturbations

        # Adjust to conserve momentum exactly
        current_momentum = np.sum(fragment_masses[:, np.newaxis] * velocities, axis=0)
        momentum_deficit = target_momentum - current_momentum

        # Distribute deficit proportionally to mass
        mass_fractions = fragment_masses / fragment_masses.sum()
        correction = momentum_deficit / fragment_masses.sum()
        velocities += correction[np.newaxis, :] * fragment_masses.sum()
        # Re-adjust: each fragment gets correction proportional to 1/mass
        # Actually, we need Σ m_i Δv_i = momentum_deficit
        # Simplest: distribute equally in velocity space weighted by 1/m_i
        # But that can give huge velocities to small fragments.
        # Better: distribute proportionally to mass fraction
        velocities = np.tile(parent_velocity, (n, 1)) + perturbations
        current_momentum = np.sum(fragment_masses[:, np.newaxis] * velocities, axis=0)
        momentum_deficit = target_momentum - current_momentum
        # Apply correction: Δv_i = momentum_deficit / (N * m_i) would work
        # but let's use mass-weighted: Δv_i = momentum_deficit * (m_i / Σm²)
        # Actually simplest: equal velocity correction
        velocity_correction = momentum_deficit / fragment_masses.sum()
        velocities += velocity_correction[np.newaxis, :]
    else:
        # All fragments inherit parent velocity
        velocities = np.tile(parent_velocity, (n, 1))

        # Apply fracture impulse as velocity correction
        if np.linalg.norm(fracture_impulse) > 0:
            velocity_correction = fracture_impulse / fragment_masses.sum()
            velocities += velocity_correction[np.newaxis, :]

    # Verify momentum conservation
    final_momentum = np.sum(fragment_masses[:, np.newaxis] * velocities, axis=0)
    # Small numerical correction if needed
    residual = target_momentum - final_momentum
    if np.linalg.norm(residual) > 1e-15 * np.linalg.norm(target_momentum):
        correction = residual / fragment_masses.sum()
        velocities += correction[np.newaxis, :]

    return velocities


def replace_parent_with_fragments(
    parent_mass: float,
    parent_velocity: np.ndarray,
    parent_radius: float,
    density: float,
    n_fragments: int,
    size_distribution: str = "uniform",
    alpha: float = 2.5,
    fracture_impulse: np.ndarray | None = None,
    velocity_perturbation: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FragmentReplacement:
    """Replace a parent particle with fragments, conserving mass and momentum.

    Parameters
    ----------
    parent_mass : float
        Parent particle mass (kg).
    parent_velocity : np.ndarray
        Parent particle velocity (m/s), shape (3,).
    parent_radius : float
        Parent particle radius (m).
    density : float
        Material density (kg/m^3).
    n_fragments : int
        Number of child fragments.
    size_distribution : str
        Fragment size distribution type.
    alpha : float
        Distribution exponent.
    fracture_impulse : np.ndarray, optional
        Impulse from fracture (kg·m/s).
    velocity_perturbation : float
        Random velocity perturbation magnitude (m/s).
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    FragmentReplacement
        Result containing fragment properties and conservation errors.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    parent_velocity = np.asarray(parent_velocity, dtype=np.float64)

    # Generate fragment masses (exact conservation by construction)
    frag_masses = generate_fragment_masses(
        parent_mass, n_fragments, size_distribution, alpha, rng
    )

    # Convert to radii
    frag_radii = masses_to_radii(frag_masses, density)

    # Generate fragment velocities (momentum-conserving)
    frag_velocities = generate_fragment_velocities(
        parent_velocity, frag_masses, parent_mass,
        fracture_impulse, velocity_perturbation, rng
    )

    # Compute conservation errors
    mass_error = float(frag_masses.sum() - parent_mass)
    final_momentum = np.sum(frag_masses[:, np.newaxis] * frag_velocities, axis=0)
    target_momentum = parent_mass * parent_velocity
    if fracture_impulse is not None:
        target_momentum = target_momentum + np.asarray(fracture_impulse)
    momentum_error = final_momentum - target_momentum

    return FragmentReplacement(
        parent_mass=parent_mass,
        parent_velocity=parent_velocity,
        fragment_masses=frag_masses,
        fragment_velocities=frag_velocities,
        fragment_radii=frag_radii,
        mass_error=mass_error,
        momentum_error=momentum_error,
    )
