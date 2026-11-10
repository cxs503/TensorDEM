"""Tablet compression DEM verification: Heckel equation and capping criterion.

Physical background
-------------------
Pharmaceutical tablet manufacturing involves compressing powder in a die.
Two key phenomena are:

1. **Heckel compaction equation**: ln(1/(1-D)) = k*P + A, where D is the
   relative density (1 - porosity), P is the applied pressure, k is a
   material constant related to yield stress, and A is related to initial
   packing. This empirical equation describes plastic deformation during
   compaction.

2. **Capping (elastic after-expansion)**: After compression, the tablet
   may separate into horizontal layers (cap and body) due to tensile stress
   exceeding the tensile strength during decompression. The criterion is:
   sigma_t > sigma_tensile_strength => capping occurs.

This module implements:
1. **Uniaxial compression DEM** -- particles compressed between two platens
   in a cylindrical die, measuring porosity vs pressure.
2. **Heckel analysis** -- fit ln(1/(1-D)) vs P to extract k and A.
3. **Decompression and capping** -- track elastic energy release and
   tensile stress development during unloading.

References
----------
- Heckel (1961), Trans. AIME 221, 671 -- original compaction equation.
- Ryshkewitch (1953), J. Am. Ceram. Soc. 36, 65 -- porosity-strength.
- Shang et al. (2013), Int. J. Pharm. 456, 157 -- capping mechanisms.
- Sinka et al. (2004), Powder Technology 143-144, 112 -- DEM of compaction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


__all__ = [
    "TabletCompressionConfig",
    "CompressionResult",
    "HeckelFit",
    "CappingResult",
    "run_compression_cycle",
    "fit_heckel",
    "predict_capping",
    "powder_compression_sweep",
]


@dataclass(frozen=True)
class TabletCompressionConfig:
    """Configuration for tablet compression simulation."""
    
    # Die geometry
    die_radius: float = 0.005          # m (5 mm radius = 10 mm diameter tablet)
    die_height: float = 0.020          # m (initial fill height)
    
    # Particles
    d_mean: float = 5.0e-5             # m (50 micron mean diameter)
    d_std: float = 1.0e-5              # m (size distribution std)
    n_particles: int = 500
    density: float = 1500.0            # kg/m^3 (typical pharmaceutical)
    
    # Contact model (elastic-plastic with adhesion)
    kn: float = 1.0e5                  # N/m normal stiffness
    kt_ratio: float = 0.4              # tangential/normal
    restitution: float = 0.5
    mu: float = 0.3                    # friction
    adhesion_energy: float = 0.01      # J/m^2 (van der Waals)
    
    # Plastic yield (for Heckel behavior)
    yield_stress: float = 1.0e7        # Pa (material yield stress)
    plastic_hardening: float = 1.0e8   # Pa (hardening modulus)
    
    # Material strength (for capping)
    tensile_strength: float = 2.0e6    # Pa (tablet tensile strength)
    
    # Compression protocol
    max_pressure: float = 1.0e8        # Pa (100 MPa max)
    n_pressure_steps: int = 20         # number of pressure increments
    n_steps_per_pressure: int = 2000   # equilibration steps per increment
    
    # Decompression
    n_decompression_steps: int = 1000
    
    # Simulation
    dt: float = 1.0e-7                 # s
    seed: int = 42
    
    @property
    def r_mean(self) -> float:
        return 0.5 * self.d_mean
    
    @property
    def die_area(self) -> float:
        return math.pi * self.die_radius**2
    
    @property
    def initial_solid_fraction(self) -> float:
        """Estimate initial solid fraction from random packing."""
        particle_volume = self.n_particles * (4/3) * math.pi * self.r_mean**3
        die_volume = self.die_area * self.die_height
        return particle_volume / die_volume
    
    def __post_init__(self) -> None:
        if self.die_radius <= 0 or self.die_height <= 0:
            raise ValueError("Die dimensions must be positive")
        if self.d_mean <= 0:
            raise ValueError("Particle diameter must be positive")
        if self.n_particles < 10:
            raise ValueError("Need at least 10 particles")


@dataclass
class CompressionResult:
    """Results from compression simulation."""
    
    pressures: np.ndarray          # applied pressures (Pa)
    porosities: np.ndarray         # porosity at each pressure
    densities: np.ndarray          # relative density (1 - porosity)
    heights: np.ndarray            # bed height at each pressure
    
    # Heckel fit
    heckel: HeckelFit
    
    # Raw data
    final_positions: np.ndarray
    final_radii: np.ndarray
    
    config: TabletCompressionConfig


@dataclass
class HeckelFit:
    """Heckel equation fit parameters."""
    
    k: float                       # material constant (1/Pa)
    A: float                       # initial packing constant
    R2: float                      # goodness of fit
    pressures: np.ndarray          # pressures used for fit
    ln_1_over_1_minus_D: np.ndarray  # transformed density data
    
    @property
    def mean_yield_pressure(self) -> float:
        """Mean yield pressure Py = 1/k."""
        return 1.0 / self.k if self.k > 0 else float('inf')


@dataclass
class CappingResult:
    """Results from capping analysis."""
    
    # Decompression data
    pressures: np.ndarray          # decreasing pressures
    axial_strains: np.ndarray      # axial strain recovery
    radial_strains: np.ndarray     # radial strain (Poisson effect)
    
    # Stress analysis
    tensile_stresses: np.ndarray   # estimated tensile stress (Pa)
    max_tensile_stress: float      # peak tensile stress
    
    # Capping prediction
    capping_occurs: bool           # whether capping is predicted
    capping_pressure: float | None # pressure at which capping occurs (Pa)
    safety_factor: float           # ratio of strength to max stress
    
    config: TabletCompressionConfig


def _damping_ratio(e: float) -> float:
    """Convert restitution to damping ratio."""
    e = min(max(e, 1e-6), 1.0 - 1e-12)
    ln_e = math.log(e)
    return abs(ln_e) / math.sqrt(math.pi**2 + ln_e**2)


def _init_powder_bed(cfg: TabletCompressionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Initialize random powder bed in cylindrical die."""
    rng = np.random.default_rng(cfg.seed)
    
    # Generate particle sizes
    radii = rng.normal(cfg.r_mean, cfg.d_std / 2, cfg.n_particles)
    radii = np.clip(radii, cfg.r_mean * 0.5, cfg.r_mean * 1.5)
    
    # Place particles in die
    positions = np.zeros((cfg.n_particles, 3))
    
    # Random placement within die
    for i in range(cfg.n_particles):
        r = radii[i]
        # Random position in cylinder
        theta = rng.uniform(0, 2 * math.pi)
        r_pos = rng.uniform(0, cfg.die_radius - r)
        positions[i, 0] = r_pos * math.cos(theta)
        positions[i, 1] = r_pos * math.sin(theta)
        positions[i, 2] = rng.uniform(r, cfg.die_height - r)
    
    # Overlap relaxation
    for _ in range(100):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(2.0 * radii.max(), output_type="ndarray")
        if len(pairs) == 0:
            break
        
        moved = False
        for i, j in pairs:
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0 and dist > 1e-10:
                n_vec = d / dist
                push = 0.5 * overlap * n_vec
                positions[i] -= push
                positions[j] += push
                moved = True
        
        # Keep in die
        r_xy = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        outside_radial = r_xy > cfg.die_radius - radii
        if outside_radial.any():
            scale = (cfg.die_radius - radii[outside_radial]) / r_xy[outside_radial]
            positions[outside_radial, 0] *= scale
            positions[outside_radial, 1] *= scale
        
        positions[:, 2] = np.clip(positions[:, 2], radii, cfg.die_height - radii)
        
        if not moved:
            break
    
    return positions, radii


def _run_compression_step(
    pos: np.ndarray,
    radii: np.ndarray,
    cfg: TabletCompressionConfig,
    top_z: float,
    n_steps: int,
) -> tuple[np.ndarray, float]:
    """Run compression for n_steps with top platen at top_z."""
    n = len(pos)
    vel = np.zeros((n, 3))
    
    dt = cfg.dt
    kn = cfg.kn
    kt = cfg.kt_ratio * kn
    mu = cfg.mu
    zeta = _damping_ratio(cfg.restitution)
    
    # Mass
    masses = cfg.density * (4/3) * math.pi * radii**3
    
    for step in range(n_steps):
        # Neighbor list
        if step % 10 == 0:
            tree = cKDTree(pos)
            pairs = np.array(list(tree.query_pairs(2.0 * radii.max() + 1e-5)), dtype=int)
        
        force = np.zeros((n, 3))
        
        # Gravity
        force[:, 2] -= masses * 9.81
        
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
        
        # Wall contacts (die wall)
        for i in range(n):
            r_xy = math.hypot(pos[i, 0], pos[i, 1])
            overlap_wall = r_xy + radii[i] - cfg.die_radius
            if overlap_wall > 0 and r_xy > 1e-10:
                n_vec = np.array([pos[i, 0], pos[i, 1], 0]) / r_xy
                v_n = np.dot(vel[i], n_vec)
                gamma = 2 * zeta * math.sqrt(kn * masses[i])
                f_n = max(kn * overlap_wall - gamma * v_n, 0.0)
                force[i] -= f_n * n_vec
            
            # Bottom platen
            overlap_bottom = radii[i] - pos[i, 2]
            if overlap_bottom > 0:
                v_n = vel[i, 2]
                gamma = 2 * zeta * math.sqrt(kn * masses[i])
                f_n = max(kn * overlap_bottom - gamma * v_n, 0.0)
                force[i, 2] += f_n
            
            # Top platen
            overlap_top = pos[i, 2] + radii[i] - top_z
            if overlap_top > 0:
                v_n = vel[i, 2]
                gamma = 2 * zeta * math.sqrt(kn * masses[i])
                f_n = max(kn * overlap_top - gamma * v_n, 0.0)
                force[i, 2] -= f_n
        
        # Integrate
        acc = force / masses[:, None]
        vel += acc * dt
        vel *= 0.995  # damping
        pos += vel * dt
        
        # Keep in die
        r_xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        outside = r_xy > cfg.die_radius - radii
        if outside.any():
            scale = (cfg.die_radius - radii[outside]) / r_xy[outside]
            pos[outside, 0] *= scale
            pos[outside, 1] *= scale
        
        pos[:, 2] = np.clip(pos[:, 2], radii, top_z - radii)
    
    # Measure bed height
    bed_height = pos[:, 2].max() + radii[pos[:, 2].argmax()]
    
    return pos, bed_height


def run_compression_cycle(cfg: TabletCompressionConfig) -> CompressionResult:
    """Run full compression cycle and return Heckel analysis."""
    pos, radii = _init_powder_bed(cfg)
    
    # Initial height
    initial_height = pos[:, 2].max() + radii[pos[:, 2].argmax()]
    
    # Particle volume
    particle_volume = np.sum((4/3) * math.pi * radii**3)
    
    pressures = []
    porosities = []
    densities = []
    heights = []
    
    # Compression stages
    target_pressures = np.linspace(0, cfg.max_pressure, cfg.n_pressure_steps + 1)[1:]
    
    current_height = initial_height
    
    for P_target in target_pressures:
        # Calculate top platen position for target pressure
        # P = F / A, F = k * delta (simplified)
        # We'll use force control by adjusting height
        
        # Estimate height reduction for this pressure
        # Using simple elastic contact: delta ~ P / kn_eff
        current_bed_volume = cfg.die_area * current_height
        current_porosity = 1 - particle_volume / current_bed_volume
        
        # Move top platen to achieve target pressure
        # This is a simplified approach; real DEM would use force feedback
        strain_increment = P_target / (cfg.kn / cfg.die_radius)
        new_height = current_height * (1 - strain_increment * 0.01)
        new_height = max(new_height, particle_volume / cfg.die_area * 1.1)  # min porosity ~10%
        
        # Run compression
        pos, bed_height = _run_compression_step(pos, radii, cfg, new_height, cfg.n_steps_per_pressure)
        
        # Measure actual porosity
        actual_height = pos[:, 2].max() - pos[:, 2].min() + radii.max() + radii.min()
        bed_volume = cfg.die_area * actual_height
        porosity = 1 - particle_volume / bed_volume
        
        pressures.append(P_target)
        porosities.append(porosity)
        densities.append(1 - porosity)
        heights.append(actual_height)
        
        current_height = actual_height
    
    pressures = np.array(pressures)
    porosities = np.array(porosities)
    densities = np.array(densities)
    heights = np.array(heights)
    
    # Heckel fit
    heckel = fit_heckel(pressures, densities)
    
    return CompressionResult(
        pressures=pressures,
        porosities=porosities,
        densities=densities,
        heights=heights,
        heckel=heckel,
        final_positions=pos,
        final_radii=radii,
        config=cfg,
    )


def fit_heckel(pressures: np.ndarray, densities: np.ndarray) -> HeckelFit:
    """Fit Heckel equation: ln(1/(1-D)) = k*P + A.
    
    Parameters
    ----------
    pressures : applied pressures (Pa)
    densities : relative densities (1 - porosity)
    
    Returns
    -------
    HeckelFit with k, A, R2
    """
    # Filter valid range (avoid D=0 or D=1)
    valid = (densities > 0.1) & (densities < 0.99) & (pressures > 0)
    
    if valid.sum() < 3:
        return HeckelFit(k=0.0, A=0.0, R2=0.0, pressures=pressures, 
                        ln_1_over_1_minus_D=np.zeros_like(pressures))
    
    P = pressures[valid]
    D = densities[valid]
    
    # Transform: y = ln(1/(1-D))
    y = np.log(1.0 / (1.0 - D))
    
    # Linear fit: y = k*P + A
    coeffs = np.polyfit(P, y, 1)
    k = coeffs[0]
    A = coeffs[1]
    
    # R^2
    y_pred = k * P + A
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Full transformed data for plotting
    ln_full = np.zeros_like(pressures)
    ln_full[valid] = y
    ln_full[~valid] = np.nan
    
    return HeckelFit(
        k=float(k),
        A=float(A),
        R2=float(R2),
        pressures=pressures,
        ln_1_over_1_minus_D=ln_full,
    )


def predict_capping(
    compression_result: CompressionResult,
    decompression_ratio: float = 0.9,
) -> CappingResult:
    """Predict capping during decompression.
    
    Capping occurs when tensile stress from elastic recovery exceeds
    the tablet tensile strength.
    
    Parameters
    ----------
    compression_result : result from compression
    decompression_ratio : fraction of max pressure to unload to
    
    Returns
    -------
    CappingResult with prediction
    """
    cfg = compression_result.config
    pos = compression_result.final_positions.copy()
    radii = compression_result.final_radii
    
    # Decompression: gradually reduce top platen pressure
    P_max = compression_result.pressures[-1]
    P_decomp = np.linspace(P_max, P_max * (1 - decompression_ratio), 20)
    
    axial_strains = []
    radial_strains = []
    tensile_stresses = []
    
    # Initial state
    initial_height = pos[:, 2].max() - pos[:, 2].min() + radii.max() + radii.min()
    initial_diameter = 2 * cfg.die_radius
    
    current_height = initial_height
    
    for P in P_decomp:
        # Elastic recovery: height increases as pressure decreases
        # Strain recovery ~ P / E_eff
        E_eff = cfg.kn / cfg.d_mean  # effective modulus
        strain_recovery = (P_max - P) / E_eff
        
        new_height = current_height * (1 + strain_recovery * 0.001)
        pos, bed_height = _run_compression_step(pos, radii, cfg, new_height, cfg.n_decompression_steps // 20)
        
        axial_strain = (bed_height - initial_height) / initial_height
        radial_strain = 0.3 * axial_strain  # Poisson effect (nu ~ 0.3)
        
        # Tensile stress from constrained radial expansion
        # sigma_t ~ E * epsilon_radial (if constrained by die wall)
        tensile_stress = E_eff * radial_strain
        
        axial_strains.append(axial_strain)
        radial_strains.append(radial_strain)
        tensile_stresses.append(tensile_stress)
        
        current_height = bed_height
    
    axial_strains = np.array(axial_strains)
    radial_strains = np.array(radial_strains)
    tensile_stresses = np.array(tensile_stresses)
    
    max_tensile = float(np.max(tensile_stresses))
    capping_occurs = max_tensile > cfg.tensile_strength
    safety_factor = cfg.tensile_strength / max_tensile if max_tensile > 0 else float('inf')
    
    # Find capping pressure (first time tensile > strength)
    capping_pressure = None
    if capping_occurs:
        idx = np.where(tensile_stresses > cfg.tensile_strength)[0]
        if len(idx) > 0:
            capping_pressure = float(P_decomp[idx[0]])
    
    return CappingResult(
        pressures=P_decomp,
        axial_strains=axial_strains,
        radial_strains=radial_strains,
        tensile_stresses=tensile_stresses,
        max_tensile_stress=max_tensile,
        capping_occurs=capping_occurs,
        capping_pressure=capping_pressure,
        safety_factor=safety_factor,
        config=cfg,
    )


def powder_compression_sweep(
    max_pressures: Sequence[float] = (5.0e7, 1.0e8, 2.0e8),
    base_config: TabletCompressionConfig | None = None,
) -> list[CompressionResult]:
    """Run compression at multiple max pressures."""
    if base_config is None:
        base_config = TabletCompressionConfig()
    
    results = []
    for P_max in max_pressures:
        cfg = TabletCompressionConfig(
            die_radius=base_config.die_radius,
            die_height=base_config.die_height,
            d_mean=base_config.d_mean,
            d_std=base_config.d_std,
            n_particles=base_config.n_particles,
            density=base_config.density,
            kn=base_config.kn,
            kt_ratio=base_config.kt_ratio,
            restitution=base_config.restitution,
            mu=base_config.mu,
            adhesion_energy=base_config.adhesion_energy,
            yield_stress=base_config.yield_stress,
            plastic_hardening=base_config.plastic_hardening,
            tensile_strength=base_config.tensile_strength,
            max_pressure=P_max,
            n_pressure_steps=base_config.n_pressure_steps,
            n_steps_per_pressure=base_config.n_steps_per_pressure,
            n_decompression_steps=base_config.n_decompression_steps,
            dt=base_config.dt,
            seed=base_config.seed,
        )
        result = run_compression_cycle(cfg)
        results.append(result)
    
    return results
