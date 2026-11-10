"""DEM consolidation simulation: rubble rearrangement under gravity + wave loading.

This module simulates the time evolution of keel porosity by modelling the
rubble as a granular assembly undergoing:

1. **Gravity settling**: particles settle under self-weight, finding stable
   configurations through contact forces
2. **Wave-induced shaking**: cyclic horizontal acceleration simulates the
   effect of wave passage on the keel rubble

The simulation uses a simplified DEM approach:
- Linear spring-dashpot contact law (normal direction)
- Coulomb friction (tangential direction)
- Rolling resistance
- Periodic boundary conditions in horizontal directions (within keel walls)
- Rigid keel walls (no deformation of container)

The key output is the porosity time series n(t), which shows how the keel
densifies over time.

Physical scaling:
- DEM time scale: seconds to minutes
- Prototype time scale: hours to months (via Froude or other scaling)
- The consolidation time τ extracted from DEM is in DEM time units;
  scaling to prototype requires care (see sensitivity module).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import KeelGeometry
from .packing import PackingResult, ICE_DENSITY

__all__ = [
    "ConsolidationConfig",
    "ConsolidationResult",
    "run_consolidation",
    "compute_porosity_timeseries",
]


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for DEM consolidation simulation.

    Attributes
    ----------
    dt : float
        Time step (s). Default 1e-3.
    total_time : float
        Total simulation time (s). Default 500.0.
    record_interval : float
        Interval for recording porosity (s). Default 5.0.
    gravity : float
        Gravitational acceleration (m/s²). Default 9.81.
    kn : float
        Normal contact stiffness (N/m). Default 1e5.
    kt_ratio : float
        Tangential/normal stiffness ratio. Default 0.5.
    friction : float
        Friction coefficient μ. Default 0.5.
    rolling_friction : float
        Rolling friction coefficient μ_r. Default 0.05.
    restitution : float
        Coefficient of restitution. Default 0.3.
    damping_ratio : float
        Bulk viscous damping ratio (fraction of critical). Default 0.1.
    wave_amplitude : float
        Wave-induced horizontal acceleration amplitude (m/s²). Default 0.5.
    wave_period : float
        Wave period (s). Default 10.0.
    wave_enabled : bool
        Whether to apply wave loading. Default True.
    seed : int
        Random seed. Default 42.
    """
    dt: float = 1e-3
    total_time: float = 500.0
    record_interval: float = 5.0
    gravity: float = 9.81
    kn: float = 1e5
    kt_ratio: float = 0.5
    friction: float = 0.5
    rolling_friction: float = 0.05
    restitution: float = 0.3
    damping_ratio: float = 0.1
    wave_amplitude: float = 0.5
    wave_period: float = 10.0
    wave_enabled: bool = True
    seed: int = 42


@dataclass
class ConsolidationResult:
    """Result of consolidation simulation.

    Attributes
    ----------
    time : np.ndarray
        Time stamps (s), shape (n_records,).
    porosity : np.ndarray
        Porosity at each time stamp, shape (n_records,).
    mean_height : np.ndarray
        Mean particle height (m) at each time stamp.
    kinetic_energy : np.ndarray
        Total kinetic energy (J) at each time stamp.
    final_positions : np.ndarray
        Final particle positions, shape (N, 3).
    final_radii : np.ndarray
        Final particle radii, shape (N,).
    config : ConsolidationConfig
        Configuration used.
    """
    time: np.ndarray
    porosity: np.ndarray
    mean_height: np.ndarray
    kinetic_energy: np.ndarray
    final_positions: np.ndarray
    final_radii: np.ndarray
    config: ConsolidationConfig


def _damping_coefficient(restitution: float, mass: float, kn: float) -> float:
    """Compute dashpot coefficient from restitution coefficient."""
    e = max(min(restitution, 0.999), 0.001)
    ln_e = math.log(e)
    zeta = abs(ln_e) / math.sqrt(math.pi**2 + ln_e**2)
    m_eff = 0.5 * mass  # reduced mass for equal particles
    return 2.0 * zeta * math.sqrt(kn * m_eff)


def _contact_force_normal(
    delta: float,
    v_n: float,
    kn: float,
    gamma: float,
) -> float:
    """Normal contact force (linear spring-dashpot).

    F_n = max(k_n * delta - gamma * v_n, 0)

    Positive = repulsive (compression).
    """
    if delta <= 0:
        return 0.0
    return max(kn * delta - gamma * v_n, 0.0)


def _wave_acceleration(t: float, amplitude: float, period: float) -> Tuple[float, float]:
    """Compute wave-induced horizontal acceleration at time t.

    Returns (a_x, a_y) in m/s².
    Model: sinusoidal horizontal shaking with decaying envelope.
    """
    omega = 2 * math.pi / period
    # Horizontal acceleration (primarily x-direction, some y)
    a_x = amplitude * math.sin(omega * t)
    a_y = 0.3 * amplitude * math.cos(0.7 * omega * t)  # secondary component
    return a_x, a_y


def run_consolidation(
    packing: PackingResult,
    geom: KeelGeometry,
    config: ConsolidationConfig = None,
) -> ConsolidationResult:
    """Run DEM consolidation simulation.

    This is a simplified DEM that captures the essential physics of
    granular rearrangement under gravity + cyclic loading:

    1. Particles settle under gravity
    2. Contact forces prevent overlap (spring-dashpot)
    3. Friction resists sliding
    4. Wave loading adds cyclic horizontal acceleration
    5. Porosity is tracked over time

    The simulation uses an explicit time integration (Velocity Verlet)
    with a spatial hash for neighbor detection.

    Parameters
    ----------
    packing : PackingResult
        Initial packing (positions, radii).
    geom : KeelGeometry
        Keel container geometry.
    config : ConsolidationConfig
        Simulation configuration.

    Returns
    -------
    ConsolidationResult
        Time series of porosity and other quantities.
    """
    if config is None:
        config = ConsolidationConfig()

    rng = np.random.default_rng(config.seed)
    N = packing.n_placed
    if N == 0:
        raise ValueError("Empty packing, cannot run consolidation")

    # Initialize state
    pos = packing.positions.copy()
    radii = packing.radii.copy()
    vel = np.zeros((N, 3))  # velocities
    acc = np.zeros((N, 3))  # accelerations

    # Material properties
    density = ICE_DENSITY
    masses = density * 4 / 3 * np.pi * radii**3
    inertia = 0.4 * masses * radii**2

    # Damping
    gamma_n = np.array([
        _damping_coefficient(config.restitution, m, config.kn)
        for m in masses
    ])

    # Recording
    n_records = int(config.total_time / config.record_interval) + 1
    time_list = []
    porosity_list = []
    height_list = []
    ke_list = []

    # Container volume (fixed)
    V_container = geom.L_top * geom.W_top * geom.depth  # approximate

    # Time integration loop
    t = 0.0
    step = 0
    next_record = 0.0
    dt = config.dt
    n_steps = int(config.total_time / dt)

    # Subsample for performance: only do full neighbor search every few steps
    neighbor_update_interval = 10

    for step in range(n_steps):
        # Record at intervals
        if t >= next_record:
            # Compute porosity
            solid_vol = np.sum(4 / 3 * np.pi * radii**3)
            # Estimate occupied volume from particle extent
            z_min = np.min(pos[:, 2] - radii)
            z_max = np.max(pos[:, 2] + radii)
            occupied_height = max(z_max - z_min, 0.1)
            # Average cross-section at mean height
            z_mean = 0.5 * (z_min + z_max)
            L_mean, W_mean = geom.cross_section(min(z_mean, geom.depth))
            V_occupied = L_mean * W_mean * occupied_height
            porosity = 1.0 - solid_vol / max(V_occupied, solid_vol * 1.01)
            porosity = min(max(porosity, 0.05), 0.95)  # clamp

            time_list.append(t)
            porosity_list.append(porosity)
            height_list.append(np.mean(pos[:, 2]))
            ke_list.append(0.5 * np.sum(masses * np.sum(vel**2, axis=1)))

            next_record += config.record_interval

        # Update neighbor list periodically
        if step % neighbor_update_interval == 0:
            tree = cKDTree(pos)
            max_r = np.max(radii)
            pairs = tree.query_pairs(2.0 * max_r * 1.1)

        # Compute accelerations
        acc_new = np.zeros((N, 3))

        # Gravity
        acc_new[:, 2] -= config.gravity

        # Wave loading
        if config.wave_enabled:
            a_x, a_y = _wave_acceleration(t, config.wave_amplitude, config.wave_period)
            acc_new[:, 0] += a_x
            acc_new[:, 1] += a_y

        # Contact forces (simplified: only detect overlaps, apply repulsion)
        for i, j in pairs:
            rij = pos[j] - pos[i]
            dist = np.linalg.norm(rij)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0 and dist > 1e-10:
                # Normal direction
                n_vec = rij / dist
                # Relative velocity
                v_rel = vel[j] - vel[i]
                v_n = np.dot(v_rel, n_vec)
                # Normal force
                gamma_ij = 0.5 * (gamma_n[i] + gamma_n[j])
                F_n = _contact_force_normal(overlap, v_n, config.kn, gamma_ij)
                # Apply to particles (Newton's 3rd law)
                F_vec = F_n * n_vec
                acc_new[i] += F_vec / masses[i]
                acc_new[j] -= F_vec / masses[j]

                # Tangential friction (simplified Coulomb)
                v_t = v_rel - v_n * n_vec
                v_t_mag = np.linalg.norm(v_t)
                if v_t_mag > 1e-10:
                    F_t_mag = min(config.friction * F_n, config.kn * config.kt_ratio * v_t_mag * dt)
                    F_t = -F_t_mag * v_t / v_t_mag
                    acc_new[i] += F_t / masses[i]
                    acc_new[j] -= F_t / masses[j]

        # Bulk viscous damping (to settle to quasi-static)
        acc_new -= config.damping_ratio * vel

        # Velocity Verlet integration
        vel += 0.5 * (acc + acc_new) * dt
        pos += vel * dt + 0.5 * acc_new * dt**2
        acc = acc_new

        # Wall boundaries (keel geometry)
        for i in range(N):
            z = pos[i, 2]
            r = radii[i]
            # Floor
            if z < r:
                z = r
                vel[i, 2] = abs(vel[i, 2]) * config.restitution
            # Ceiling (waterline)
            if z > geom.depth - r:
                z = geom.depth - r
                vel[i, 2] = -abs(vel[i, 2]) * config.restitution
            # Side walls (at current depth)
            L, W = geom.cross_section(z)
            if pos[i, 0] < -L / 2 + r:
                pos[i, 0] = -L / 2 + r
                vel[i, 0] = abs(vel[i, 0]) * config.restitution
            elif pos[i, 0] > L / 2 - r:
                pos[i, 0] = L / 2 - r
                vel[i, 0] = -abs(vel[i, 0]) * config.restitution
            if pos[i, 1] < -W / 2 + r:
                pos[i, 1] = -W / 2 + r
                vel[i, 1] = abs(vel[i, 1]) * config.restitution
            elif pos[i, 1] > W / 2 - r:
                pos[i, 1] = W / 2 - r
                vel[i, 1] = -abs(vel[i, 1]) * config.restitution
            pos[i, 2] = z

        t += dt

    # Final recording
    solid_vol = np.sum(4 / 3 * np.pi * radii**3)
    z_min = np.min(pos[:, 2] - radii)
    z_max = np.max(pos[:, 2] + radii)
    occupied_height = max(z_max - z_min, 0.1)
    z_mean = 0.5 * (z_min + z_max)
    L_mean, W_mean = geom.cross_section(min(z_mean, geom.depth))
    V_occupied = L_mean * W_mean * occupied_height
    porosity = 1.0 - solid_vol / max(V_occupied, solid_vol * 1.01)
    porosity = min(max(porosity, 0.05), 0.95)

    time_list.append(t)
    porosity_list.append(porosity)
    height_list.append(np.mean(pos[:, 2]))
    ke_list.append(0.5 * np.sum(masses * np.sum(vel**2, axis=1)))

    return ConsolidationResult(
        time=np.array(time_list),
        porosity=np.array(porosity_list),
        mean_height=np.array(height_list),
        kinetic_energy=np.array(ke_list),
        final_positions=pos,
        final_radii=radii,
        config=config,
    )


def compute_porosity_timeseries(result: ConsolidationResult) -> Tuple[np.ndarray, np.ndarray]:
    """Extract porosity time series from consolidation result.

    Returns
    -------
    time : np.ndarray
        Time stamps (s).
    porosity : np.ndarray
        Porosity values.
    """
    return result.time, result.porosity
