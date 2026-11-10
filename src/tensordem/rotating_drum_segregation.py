"""Rotating drum bidisperse segregation verification (Brazil-nut effect).

Physical background
-------------------
When a partially filled rotating drum contains a bidisperse granular bed,
the two species segregate. The canonical Brazil-nut effect (BNE) predicts
that large particles rise to the top (periphery) due to percolation through
voids opened in the flowing layer. The reverse Brazil-nut effect (RBNE)
occurs when density differences dominate over size differences.

This module implements:
1. **Rotating drum simulation** -- 2D disks in a circular drum rotating at
   prescribed angular velocity omega.
2. **Segregation metric** -- radial concentration profile Phi(r) of large
   species, with segregation degree S = |<Phi> - Phi_0| / Phi_0.
3. **Direction criterion** -- large particles at periphery (r > r_mid) means
   BNE (direction = +1); at center means RBNE (direction = -1).
4. **Scaling law** -- S vs diameter ratio d_l/d_s and volume fraction phi.

References
----------
- Hill & Tan (2013), Phys. Fluids 25, 083302 -- BNE in rotating drums.
- Jain et al. (2013), Powder Technology 246, 305 -- size ratio effects.
- Bridgwater et al. (1978), Powder Technology 15, 207 -- early experiments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


__all__ = [
    "DrumSegregationConfig",
    "DrumSegregationResult",
    "run_rotating_drum_segregation",
    "compute_radial_concentration",
    "compute_segregation_degree_radial",
    "brazil_nut_direction",
    "fit_segregation_scaling",
    "segregation_scaling_sweep",
]


@dataclass(frozen=True)
class DrumSegregationConfig:
    """Configuration for rotating drum segregation simulation."""
    
    # Geometry
    drum_radius: float = 0.10          # m
    fill_fraction: float = 0.30        # fraction of drum cross-section
    
    # Particles
    d_small: float = 0.004             # m (4 mm)
    d_large: float = 0.008             # m (8 mm), ratio = 2.0
    density: float = 2500.0            # kg/m^3
    n_small: int = 200
    n_large: int = 50
    
    # Contact
    kn: float = 5.0e4                  # N/m normal stiffness
    kt_ratio: float = 0.4              # tangential/normal
    restitution: float = 0.3
    mu: float = 0.5                    # friction
    mu_rolling: float = 0.05           # rolling resistance
    
    # Drum rotation
    omega: float = 1.0                 # rad/s (about 10 rpm)
    
    # Simulation
    dt: float = 1.0e-5                 # s
    n_steps_settle: int = 5000         # settling with static drum
    n_steps_rotate: int = 20000        # rotation steps
    
    # Measurement
    n_radial_bins: int = 8
    sample_every: int = 500
    
    seed: int = 42
    
    @property
    def size_ratio(self) -> float:
        return self.d_large / self.d_small
    
    @property
    def r_small(self) -> float:
        return 0.5 * self.d_small
    
    @property
    def r_large(self) -> float:
        return 0.5 * self.d_large
    
    @property
    def r_threshold(self) -> float:
        return math.sqrt(self.r_small * self.r_large)
    
    @property
    def phi_large(self) -> float:
        """Initial large-species area fraction."""
        a_l = math.pi * self.r_large**2
        a_s = math.pi * self.r_small**2
        total_area = self.n_large * a_l + self.n_small * a_s
        return self.n_large * a_l / total_area
    
    def __post_init__(self) -> None:
        if self.drum_radius <= 0:
            raise ValueError("drum_radius must be positive")
        if not 0.0 < self.fill_fraction < 0.7:
            raise ValueError("fill_fraction must be in (0, 0.7)")
        if self.d_large < self.d_small:
            raise ValueError("d_large must be >= d_small")


@dataclass
class DrumSegregationResult:
    """Results from rotating drum segregation simulation."""
    
    # Concentration profile
    r_bins: np.ndarray          # bin centers (m)
    phi_radial: np.ndarray      # large-species area fraction per bin
    phi_0: float                # initial (nominal) large-species fraction
    
    # Segregation metrics
    S: float                    # segregation degree
    direction: int              # +1 = BNE (large at periphery), -1 = RBNE
    direction_significant: bool # whether direction is statistically significant
    
    # Raw data
    final_positions: np.ndarray
    final_radii: np.ndarray
    is_large: np.ndarray
    
    # Config
    config: DrumSegregationConfig


def _damping_ratio(e: float) -> float:
    """Convert restitution coefficient to damping ratio."""
    e = min(max(e, 1e-6), 1.0 - 1e-12)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi**2 + ln_e**2)


def _init_drum_packing(cfg: DrumSegregationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize bidisperse packing in bottom of drum."""
    rng = np.random.default_rng(cfg.seed)
    
    n_total = cfg.n_small + cfg.n_large
    radii = np.concatenate([
        np.full(cfg.n_large, cfg.r_large),
        np.full(cfg.n_small, cfg.r_small)
    ])
    is_large = np.concatenate([
        np.ones(cfg.n_large, dtype=bool),
        np.zeros(cfg.n_small, dtype=bool)
    ])
    
    # Target area for fill
    target_area = cfg.fill_fraction * math.pi * cfg.drum_radius**2
    particle_area = np.sum(math.pi * radii**2)
    
    # Place particles in bottom half of drum using jittered lattice
    positions = []
    r_max = cfg.r_large
    spacing = 2.0 * r_max * 1.02  # slight expansion to avoid overlaps
    
    y = -cfg.drum_radius + 1.5 * r_max
    row = 0
    while len(positions) < n_total and y < 0:
        x_offset = (row % 2) * 0.5 * spacing
        x = -cfg.drum_radius + 1.5 * r_max + x_offset
        while x < cfg.drum_radius - 1.5 * r_max and len(positions) < n_total:
            r_test = radii[len(positions)] if len(positions) < n_total else r_max
            if math.hypot(x, y) < cfg.drum_radius - 1.2 * r_test:
                positions.append([x, y])
            x += spacing
        y += spacing * math.sqrt(3) / 2
        row += 1
    
    positions = np.array(positions[:n_total])
    
    # Shuffle particle types
    perm = rng.permutation(n_total)
    radii = radii[perm]
    is_large = is_large[perm]
    
    # Add jitter
    positions += rng.normal(0, 0.05 * cfg.r_small, positions.shape)
    
    # Project back into drum
    r_pos = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    outside = r_pos > cfg.drum_radius - 1.2 * radii
    if outside.any():
        scale = (cfg.drum_radius - 1.2 * radii[outside]) / r_pos[outside]
        positions[outside, 0] *= scale
        positions[outside, 1] *= scale
    
    # Keep in bottom half
    positions[:, 1] = np.minimum(positions[:, 1], -0.1 * cfg.drum_radius)
    
    return positions, radii, is_large


def _overlap_relaxation(pos: np.ndarray, radii: np.ndarray, drum_r: float, n_iter: int = 50) -> np.ndarray:
    """Remove overlaps by pushing particles apart."""
    n = len(pos)
    for _ in range(n_iter):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(2.0 * radii.max(), output_type="ndarray")
        if len(pairs) == 0:
            break
        
        moved = False
        for i, j in pairs:
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0 and dist > 1e-10:
                n_vec = d / dist
                push = 0.5 * overlap * n_vec
                pos[i] -= push
                pos[j] += push
                moved = True
        
        # Keep inside drum
        r_pos = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        outside = r_pos > drum_r - 1.05 * radii
        if outside.any():
            scale = (drum_r - 1.05 * radii[outside]) / r_pos[outside]
            pos[outside, 0] *= scale
            pos[outside, 1] *= scale
        
        if not moved:
            break
    
    return pos


def _run_drum_dynamics(
    pos: np.ndarray,
    radii: np.ndarray,
    is_large: np.ndarray,
    cfg: DrumSegregationConfig,
    n_steps: int,
    omega: float,
) -> np.ndarray:
    """Run drum dynamics for n_steps, return final positions."""
    n = len(pos)
    vel = np.zeros((n, 2))
    
    dt = cfg.dt
    kn = cfg.kn
    kt = cfg.kt_ratio * kn
    mu = cfg.mu
    zeta = _damping_ratio(cfg.restitution)
    gravity = np.array([0.0, -9.81 * cfg.density])
    
    # Mass per unit thickness
    thickness = 0.01
    masses = cfg.density * math.pi * radii**2 * thickness
    
    for step in range(n_steps):
        # Neighbor list
        if step % 10 == 0:
            tree = cKDTree(pos)
            pairs = np.array(list(tree.query_pairs(2.0 * radii.max() + 0.001)), dtype=int)
        
        force = np.zeros((n, 2))
        
        # Gravity
        force[:, 1] -= masses * 9.81
        
        # Particle-particle contacts
        if len(pairs) > 0:
            for i, j in pairs:
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                overlap = radii[i] + radii[j] - dist
                if overlap > 0 and dist > 1e-10:
                    n_vec = d / dist
                    v_rel = vel[j] - vel[i]
                    v_n = np.dot(v_rel, n_vec)
                    
                    m_eff = masses[i] * masses[j] / (masses[i] + masses[j])
                    gamma = 2 * zeta * math.sqrt(kn * m_eff)
                    
                    f_n = max(kn * overlap - gamma * v_n, 0.0)
                    force[i] -= f_n * n_vec
                    force[j] += f_n * n_vec
        
        # Wall contacts (rotating drum)
        for i in range(n):
            r_pos = math.hypot(pos[i, 0], pos[i, 1])
            overlap = r_pos + radii[i] - cfg.drum_radius
            if overlap > 0 and r_pos > 1e-10:
                n_vec = pos[i] / r_pos
                v_n = np.dot(vel[i], n_vec)
                
                gamma = 2 * zeta * math.sqrt(kn * masses[i])
                f_n = max(kn * overlap - gamma * v_n, 0.0)
                
                # Wall velocity (rotating)
                v_wall = omega * np.array([-pos[i, 1], pos[i, 0]])
                v_t = vel[i] - v_wall - np.dot(vel[i] - v_wall, n_vec) * n_vec
                
                force[i] -= f_n * n_vec
                if mu > 0:
                    force[i] -= mu * f_n * v_t / (np.linalg.norm(v_t) + 1e-10)
        
        # Integrate
        acc = force / masses[:, None]
        vel += acc * dt
        vel *= 0.999  # small damping
        pos += vel * dt
        
        # Keep inside drum
        r_pos = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        outside = r_pos > cfg.drum_radius - radii
        if outside.any():
            scale = (cfg.drum_radius - radii[outside]) / r_pos[outside]
            pos[outside] *= scale[:, None]
    
    return pos


def run_rotating_drum_segregation(cfg: DrumSegregationConfig) -> DrumSegregationResult:
    """Run full rotating drum segregation simulation."""
    # Initialize
    pos, radii, is_large = _init_drum_packing(cfg)
    pos = _overlap_relaxation(pos, radii, cfg.drum_radius)
    
    phi_0 = float(np.sum(math.pi * radii[is_large]**2) / np.sum(math.pi * radii**2))
    
    # Settling phase (static drum)
    pos = _run_drum_dynamics(pos, radii, is_large, cfg, cfg.n_steps_settle, omega=0.0)
    
    # Rotation phase
    pos = _run_drum_dynamics(pos, radii, is_large, cfg, cfg.n_steps_rotate, omega=cfg.omega)
    
    # Compute radial concentration profile
    r_bins, phi_radial = compute_radial_concentration(pos, radii, is_large, cfg.drum_radius, cfg.n_radial_bins)
    
    # Segregation metrics
    S = compute_segregation_degree_radial(phi_radial, phi_0, r_bins, cfg.drum_radius)
    direction, direction_significant = brazil_nut_direction(phi_radial, r_bins, cfg.drum_radius)
    
    return DrumSegregationResult(
        r_bins=r_bins,
        phi_radial=phi_radial,
        phi_0=phi_0,
        S=S,
        direction=direction,
        direction_significant=direction_significant,
        final_positions=pos,
        final_radii=radii,
        is_large=is_large,
        config=cfg,
    )


def compute_radial_concentration(
    pos: np.ndarray,
    radii: np.ndarray,
    is_large: np.ndarray,
    drum_r: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial concentration profile of large species.
    
    Returns
    -------
    r_bins : array of bin centers (normalized by drum radius)
    phi_radial : large-species area fraction per bin
    """
    r_pos = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2) / drum_r
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    phi_radial = np.zeros(n_bins)
    areas = math.pi * radii**2
    
    for b in range(n_bins):
        mask = (r_pos >= bin_edges[b]) & (r_pos < bin_edges[b + 1])
        if mask.any():
            total_area = areas[mask].sum()
            large_area = areas[mask & is_large].sum()
            phi_radial[b] = large_area / total_area if total_area > 0 else np.nan
        else:
            phi_radial[b] = np.nan
    
    return bin_centers, phi_radial


def compute_segregation_degree_radial(
    phi_radial: np.ndarray,
    phi_0: float,
    r_bins: np.ndarray,
    drum_r: float,
) -> float:
    """Compute segregation degree S = <|Phi - Phi_0|> / Phi_0.
    
    Weighted by radial position (outer bins count more for BNE detection).
    """
    valid = np.isfinite(phi_radial)
    if not valid.any():
        return 0.0
    
    # Weight by r (outer bins have more area)
    weights = r_bins[valid]
    weights = weights / weights.sum()
    
    deviation = np.abs(phi_radial[valid] - phi_0)
    S = np.sum(weights * deviation) / phi_0
    
    return float(S)


def brazil_nut_direction(
    phi_radial: np.ndarray,
    r_bins: np.ndarray,
    drum_r: float,
) -> tuple[int, bool]:
    """Determine Brazil-nut direction from radial profile.
    
    Returns
    -------
    direction : +1 if large at periphery (BNE), -1 if large at center (RBNE), 0 if unresolved
    significant : whether the direction is statistically meaningful
    """
    valid = np.isfinite(phi_radial)
    if valid.sum() < 2:
        return 0, False
    
    r_mid = 0.5
    inner = (r_bins < r_mid) & valid
    outer = (r_bins >= r_mid) & valid
    
    if not inner.any() or not outer.any():
        return 0, False
    
    phi_inner = np.mean(phi_radial[inner])
    phi_outer = np.mean(phi_radial[outer])
    
    diff = phi_outer - phi_inner
    
    # Simple significance: difference > 10% of mean
    phi_mean = 0.5 * (phi_inner + phi_outer)
    significant = abs(diff) > 0.1 * phi_mean if phi_mean > 0 else False
    
    if diff > 0:
        direction = 1  # BNE: large at periphery
    elif diff < 0:
        direction = -1  # RBNE: large at center
    else:
        direction = 0
    
    return direction, significant


def fit_segregation_scaling(
    size_ratios: Sequence[float],
    S_values: Sequence[float],
) -> dict:
    """Fit scaling law S = A * (d_l/d_s - 1)^alpha.
    
    Returns dict with keys: A, alpha, R2
    """
    ratios = np.array(size_ratios)
    S = np.array(S_values)
    
    # Filter valid
    valid = (ratios > 1.0) & np.isfinite(S) & (S > 0)
    if valid.sum() < 2:
        return {"A": 0.0, "alpha": 0.0, "R2": 0.0}
    
    x = np.log(ratios[valid] - 1.0)
    y = np.log(S[valid])
    
    # Linear fit in log-log space
    coeffs = np.polyfit(x, y, 1)
    alpha = coeffs[0]
    A = np.exp(coeffs[1])
    
    # R^2
    y_pred = alpha * x + coeffs[1]
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {"A": float(A), "alpha": float(alpha), "R2": float(R2)}


def segregation_scaling_sweep(
    size_ratios: Sequence[float] = (1.2, 1.5, 2.0, 3.0),
    base_config: DrumSegregationConfig | None = None,
) -> dict:
    """Run segregation for multiple size ratios and fit scaling law.
    
    Returns dict with:
    - results: list of DrumSegregationResult
    - scaling: fit parameters
    """
    if base_config is None:
        base_config = DrumSegregationConfig()
    
    results = []
    S_values = []
    
    for ratio in size_ratios:
        cfg = DrumSegregationConfig(
            drum_radius=base_config.drum_radius,
            fill_fraction=base_config.fill_fraction,
            d_small=base_config.d_small,
            d_large=base_config.d_small * ratio,
            density=base_config.density,
            n_small=base_config.n_small,
            n_large=base_config.n_large,
            kn=base_config.kn,
            kt_ratio=base_config.kt_ratio,
            restitution=base_config.restitution,
            mu=base_config.mu,
            mu_rolling=base_config.mu_rolling,
            omega=base_config.omega,
            dt=base_config.dt,
            n_steps_settle=base_config.n_steps_settle,
            n_steps_rotate=base_config.n_steps_rotate,
            n_radial_bins=base_config.n_radial_bins,
            sample_every=base_config.sample_every,
            seed=base_config.seed,
        )
        result = run_rotating_drum_segregation(cfg)
        results.append(result)
        S_values.append(result.S)
    
    scaling = fit_segregation_scaling(size_ratios, S_values)
    
    return {
        "results": results,
        "size_ratios": list(size_ratios),
        "S_values": S_values,
        "scaling": scaling,
    }
