"""Ice-body crushing and accumulation against vertical structures: ISO 19906
benchmark and porosity–resistance scaling law (campaign p228w1-N04).

Physical problem
----------------
A drifting broken-ice field (ice thickness h, compressive strength σ_c,
drift speed V) is driven against a fixed vertical structure of characteristic
width D (cylinder diameter or square column side).  The ice blocks crush,
fragment, and accumulate in front of the structure, forming a rubble field.

ISO 19906 §A.8 reference
-------------------------
The ISO 19906 standard gives the nominal ice load on a vertical structure as:

    F_ISO = C · σ_c · h · D                        (1)

where:
  - C is a shape coefficient (C ≈ 2.0 for narrow cylinder, C ≈ 2.5 for square);
  - σ_c is the uniaxial compressive strength of the ice (Pa);
  - h is the ice thickness (m);
  - D is the structure width (m).

Porosity–resistance scaling law
--------------------------------
The rubble field has a bulk porosity n (void fraction).  The effective load
on the structure scales as:

    F ~ (1 - n)^α · F_ISO                          (2)

where α ∈ [1, 2] is the expected range for ice rubble.

Units: SI (m, kg, s, N, Pa) throughout.

References
----------
ISO 19906:2010.  Petroleum and natural gas industries — Fixed steel offshore
    structures in arctic and cold regions.
Timco, G.W., Weeks, W.F. (2010).  A review of the engineering properties of
    sea ice.  Cold Regions Science and Technology 60, 111-127.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "ISO19906Config",
    "iso19906_nominal_load",
    "iso19906_load_range",
    "PorosityScalingConfig",
    "PorosityScalingResult",
    "fit_porosity_scaling",
    "run_crushing_case",
    "run_porosity_sweep",
    "CrushingCaseResult",
]


# ---------------------------------------------------------------------------
# ISO 19906 §A.8 nominal load
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ISO19906Config:
    """Configuration for ISO 19906 ice load calculation."""
    sigma_c: float
    h: float
    D: float
    shape: str = "cylinder"
    C: Optional[float] = None

    def __post_init__(self):
        if self.sigma_c <= 0:
            raise ValueError("sigma_c must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.D <= 0:
            raise ValueError("D must be positive")
        if self.shape not in ("cylinder", "square"):
            raise ValueError("shape must be 'cylinder' or 'square'")

    @property
    def shape_coefficient(self) -> float:
        if self.C is not None:
            return self.C
        return 2.0 if self.shape == "cylinder" else 2.5


def iso19906_nominal_load(cfg: ISO19906Config) -> float:
    """Compute the nominal ice load per ISO 19906 §A.8: F = C·σ_c·h·D."""
    C = cfg.shape_coefficient
    return C * cfg.sigma_c * cfg.h * cfg.D


def iso19906_load_range(
    sigma_c_range: tuple[float, float],
    h: float,
    D: float,
    shape: str = "cylinder",
    C: Optional[float] = None,
) -> dict:
    """Compute the ISO 19906 load range for a range of ice strengths."""
    cfg_min = ISO19906Config(sigma_c=sigma_c_range[0], h=h, D=D, shape=shape, C=C)
    cfg_max = ISO19906Config(sigma_c=sigma_c_range[1], h=h, D=D, shape=shape, C=C)
    F_min = iso19906_nominal_load(cfg_min)
    F_max = iso19906_nominal_load(cfg_max)
    return {
        "F_min": F_min,
        "F_max": F_max,
        "F_mean": 0.5 * (F_min + F_max),
        "sigma_c_range": sigma_c_range,
        "h": h,
        "D": D,
        "shape": shape,
        "C": cfg_min.shape_coefficient,
    }


# ---------------------------------------------------------------------------
# Porosity–resistance scaling law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PorosityScalingConfig:
    """Configuration for porosity–resistance scaling study."""
    sigma_c: float = 3.0e6
    h: float = 1.0
    D: float = 5.0
    shape: str = "cylinder"
    porosity_range: tuple[float, float] = (0.2, 0.6)
    n_porosity: int = 8
    n_particles: int = 200
    drift_speed: float = 0.5
    seed: int = 42


@dataclass
class PorosityScalingResult:
    """Result of porosity–resistance scaling study."""
    porosity: np.ndarray
    F_steady: np.ndarray
    F_ISO: float
    alpha: float
    alpha_std: float
    R2: float
    config: PorosityScalingConfig

    def as_dict(self) -> dict:
        return {
            "porosity": self.porosity.tolist(),
            "F_steady": self.F_steady.tolist(),
            "F_ISO": self.F_ISO,
            "alpha": self.alpha,
            "alpha_std": self.alpha_std,
            "R2": self.R2,
            "config": {
                "sigma_c": self.config.sigma_c,
                "h": self.config.h,
                "D": self.config.D,
                "shape": self.config.shape,
                "porosity_range": self.config.porosity_range,
                "n_porosity": self.config.n_porosity,
                "n_particles": self.config.n_particles,
                "drift_speed": self.config.drift_speed,
                "seed": self.config.seed,
            },
        }


def fit_porosity_scaling(
    porosity: np.ndarray,
    F_steady: np.ndarray,
    F_ISO: float,
) -> tuple[float, float, float]:
    """Fit the porosity–resistance scaling law: F = F_ISO · (1 - n)^α.

    Returns (alpha, alpha_std, R2).
    """
    porosity = np.asarray(porosity, dtype=float)
    F_steady = np.asarray(F_steady, dtype=float)
    F_norm = F_steady / F_ISO

    x = np.log(1.0 - porosity)
    y = np.log(np.maximum(F_norm, 1e-10))

    alpha = float(np.sum(x * y) / np.sum(x * x))

    y_pred = alpha * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n_pts = len(porosity)
    if n_pts > 2:
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n_pts - 2)
        alpha_var = mse / np.sum(x * x)
        alpha_std = float(np.sqrt(max(alpha_var, 0.0)))
    else:
        alpha_std = 0.0

    return alpha, alpha_std, R2


# ---------------------------------------------------------------------------
# DEM crushing simulation (vectorised soft-sphere)
# ---------------------------------------------------------------------------
@dataclass
class CrushingCaseResult:
    """Result of a single ice crushing DEM case."""
    time: np.ndarray
    F_x: np.ndarray
    F_max: float
    F_steady: float
    porosity: float
    pile_length: float
    pile_height: float
    F_ISO: float
    config: dict
    # Extra: final particle positions for geometry plots
    final_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    final_n: int = 0
    z_eq: float = 0.0
    R_struct: float = 0.0

    def as_dict(self) -> dict:
        return {
            "time": self.time.tolist(),
            "F_x": self.F_x.tolist(),
            "F_max": self.F_max,
            "F_steady": self.F_steady,
            "porosity": self.porosity,
            "pile_length": self.pile_length,
            "pile_height": self.pile_height,
            "F_ISO": self.F_ISO,
            "config": self.config,
        }


def run_crushing_case(
    sigma_c: float = 3.0e6,
    h: float = 1.0,
    D: float = 5.0,
    shape: str = "cylinder",
    porosity_target: float = 0.4,
    n_particles: int = 200,
    drift_speed: float = 0.5,
    duration: float = 20.0,
    seed: int = 42,
) -> CrushingCaseResult:
    """Run a single ice crushing DEM case (vectorised soft-sphere DEM).

    The simulation models ice blocks as equal spheres interacting via
    Hertz-like contacts, with buoyancy, water drag, and a force-limited
    driving platen at the upstream face.  The structure is a vertical
    cylinder (or square column) of diameter D.

    The porosity_target controls the initial packing density:
    - Lower porosity = more particles in the same volume = denser packing
    - Higher porosity = fewer particles = looser packing
    """
    rng = np.random.default_rng(seed)

    # Physical parameters
    rho_ice = 900.0
    rho_water = 1025.0
    g = 9.81

    # Particle size - fixed based on ice thickness
    r_block = 0.5 * h
    V_block = 4.0 / 3.0 * math.pi * r_block ** 3
    m_block = rho_ice * V_block

    # Structure
    R_struct = 0.5 * D

    # Domain
    L_upstream = 3.0 * D
    W_domain = 3.0 * D

    # Contact stiffness
    k_n = sigma_c * r_block * 0.1
    gamma_n = 0.5 * math.sqrt(k_n * m_block)

    # Time step
    dt_dem = 0.1 * math.sqrt(m_block / k_n)
    dt_out = 0.1
    n_out = int(duration / dt_out)
    n_sub = max(1, int(dt_out / dt_dem))
    dt_dem = dt_out / n_sub

    # Equilibrium floating height
    z_eq = r_block * (1.0 - rho_ice / rho_water)

    # Control porosity by adjusting number of particles
    # Reference volume: cylinder of radius 2*D and height 3*h
    V_ref = math.pi * D ** 2 * h  # capture zone around structure
    # Number of particles to achieve target porosity
    # n = 1 - V_solid / V_ref => V_solid = (1-n) * V_ref
    # N * V_block = (1-n) * V_ref => N = (1-n) * V_ref / V_block
    n_particles_target = int((1.0 - porosity_target) * V_ref / V_block)
    n_particles_target = max(10, min(n_particles_target, 500))  # clamp

    # Initialize particles
    spacing = 2.2 * r_block
    x_start = -L_upstream
    x_end = -R_struct - 1.5 * r_block
    y_start = -W_domain / 2
    y_end = W_domain / 2

    sites = []
    i = 0
    x = x_start + spacing
    while x < x_end:
        y_off = 0.5 * spacing if i % 2 else 0.0
        y = y_start + spacing / 2 + y_off
        while y < y_end:
            if math.hypot(x, y) > R_struct + 1.2 * r_block:
                sites.append((x, y))
            y += spacing
        i += 1
        x += spacing * math.sqrt(3.0) / 2.0

    # Select particles
    if len(sites) >= n_particles_target:
        indices = rng.choice(len(sites), n_particles_target, replace=False)
        selected = [sites[idx] for idx in indices]
    else:
        selected = sites[:]
        while len(selected) < n_particles_target:
            px = rng.uniform(x_start, x_end)
            py = rng.uniform(y_start, y_end)
            if math.hypot(px, py) > R_struct + 1.2 * r_block:
                selected.append((px, py))

    n_actual = len(selected)
    if n_actual == 0:
        n_actual = 1
        selected = [(x_start + 1.0, 0.0)]

    pos = np.zeros((n_actual, 3))
    vel = np.zeros((n_actual, 3))
    for i, (sx, sy) in enumerate(selected):
        pos[i, 0] = sx + rng.uniform(-0.02, 0.02) * spacing
        pos[i, 1] = sy + rng.uniform(-0.02, 0.02) * spacing
        pos[i, 2] = z_eq + rng.uniform(-0.02, 0.02) * r_block
        vel[i, 0] = drift_speed

    # Arrays for recording
    time_arr = np.zeros(n_out)
    F_x_arr = np.zeros(n_out)

    # Main loop
    for step_out in range(n_out):
        t = step_out * dt_out
        time_arr[step_out] = t

        F_structure_x = 0.0

        for _ in range(n_sub):
            # --- Particle-particle contacts ---
            tree = cKDTree(pos[:, :2])
            pairs = tree.query_pairs(2.0 * r_block * 1.05)

            for (i, j) in pairs:
                dx = pos[j] - pos[i]
                dist = np.linalg.norm(dx)
                overlap = 2.0 * r_block - dist
                if overlap > 0 and dist > 1e-10:
                    n_vec = dx / dist
                    F_n = k_n * overlap
                    dv = vel[j] - vel[i]
                    v_n = np.dot(dv, n_vec)
                    F_damp = gamma_n * v_n
                    F_total = F_n - F_damp
                    force = F_total * n_vec
                    vel[i] += force / m_block * dt_dem
                    vel[j] -= force / m_block * dt_dem

            # --- Particle-structure contacts ---
            if shape == "cylinder":
                dist_xy = np.hypot(pos[:, 0], pos[:, 1])
                overlap = r_block + R_struct - dist_xy
                contact_mask = overlap > 0
                if contact_mask.any():
                    idx = np.where(contact_mask)[0]
                    for i in idx:
                        d = dist_xy[i]
                        if d > 1e-10:
                            nx = pos[i, 0] / d
                            ny = pos[i, 1] / d
                            F_n = k_n * overlap[i] * 10.0
                            v_n = vel[i, 0] * nx + vel[i, 1] * ny
                            F_damp = gamma_n * v_n * 3.0
                            F_total = F_n - F_damp
                            fx = F_total * nx
                            fy = F_total * ny
                            vel[i, 0] -= fx / m_block * dt_dem
                            vel[i, 1] -= fy / m_block * dt_dem
                            F_structure_x += fx
            else:  # square
                abs_x = np.abs(pos[:, 0])
                abs_y = np.abs(pos[:, 1])
                for i in range(n_actual):
                    if abs_x[i] < R_struct + r_block and abs_y[i] < R_struct + r_block:
                        overlap_x = R_struct + r_block - abs_x[i]
                        overlap_y = R_struct + r_block - abs_y[i]
                        if overlap_x > 0 and overlap_x < overlap_y:
                            nx = 1.0 if pos[i, 0] > 0 else -1.0
                            F_n = k_n * overlap_x * 10.0
                            F_damp = gamma_n * vel[i, 0] * nx * 3.0
                            F_total = F_n - F_damp
                            fx = F_total * nx
                            vel[i, 0] -= fx / m_block * dt_dem
                            F_structure_x += fx
                        elif overlap_y > 0:
                            ny = 1.0 if pos[i, 1] > 0 else -1.0
                            F_n = k_n * overlap_y * 10.0
                            F_damp = gamma_n * vel[i, 1] * ny * 3.0
                            F_total = F_n - F_damp
                            fy = F_total * ny
                            vel[i, 1] -= fy / m_block * dt_dem

            # --- Buoyancy, gravity, drag ---
            F_z = -m_block * g * (pos[:, 2] - z_eq) / r_block * 0.5
            vel[:, 2] += F_z / m_block * dt_dem

            # Water drag
            C_drag = 0.5
            vel[:, 0] -= C_drag * (vel[:, 0] - drift_speed * 0.5) * dt_dem
            vel[:, 1] -= C_drag * vel[:, 1] * dt_dem
            vel[:, 2] -= C_drag * vel[:, 2] * dt_dem

            # Driving
            drive_mask = pos[:, 0] < -R_struct - 2.0 * r_block
            vel[drive_mask, 0] += drift_speed * 2.0 * dt_dem

            # Update positions
            pos += vel * dt_dem

            # Boundary conditions
            far_downstream = pos[:, 0] > 2.0 * D
            if far_downstream.any():
                pos[far_downstream, 0] = x_start + rng.uniform(0, spacing)
                pos[far_downstream, 1] = rng.uniform(y_start, y_end)
                pos[far_downstream, 2] = z_eq + rng.uniform(-0.05, 0.05) * r_block
                vel[far_downstream, 0] = drift_speed
                vel[far_downstream, 1] = 0.0
                vel[far_downstream, 2] = 0.0

            too_far_y = np.abs(pos[:, 1]) > W_domain / 2
            vel[too_far_y, 1] *= -0.5
            pos[too_far_y, 1] = np.clip(pos[too_far_y, 1], -W_domain / 2, W_domain / 2)

        F_x_arr[step_out] = F_structure_x

    # Post-processing
    F_max = float(np.max(np.abs(F_x_arr)))
    n_steady = max(1, n_out // 5)
    F_steady = float(np.mean(np.abs(F_x_arr[-n_steady:])))

    # Compute porosity from actual packing in reference volume
    pile_mask = np.hypot(pos[:, 0], pos[:, 1]) < 2.0 * D
    n_pile = int(np.sum(pile_mask))
    V_solid = n_pile * V_block
    # Use fixed reference volume
    porosity = 1.0 - V_solid / V_ref
    porosity = float(np.clip(porosity, 0.05, 0.95))

    # Pile geometry
    if pile_mask.any():
        pile_x = pos[pile_mask, 0]
        pile_z = pos[pile_mask, 2]
        pile_length = float(np.max(pile_x) - np.min(pile_x))
        pile_height = float(np.max(pile_z) - z_eq + r_block)
    else:
        pile_length = 0.0
        pile_height = 0.0

    # ISO 19906 load
    cfg_iso = ISO19906Config(sigma_c=sigma_c, h=h, D=D, shape=shape)
    F_ISO = iso19906_nominal_load(cfg_iso)

    config = {
        "sigma_c": sigma_c,
        "h": h,
        "D": D,
        "shape": shape,
        "porosity_target": porosity_target,
        "porosity_actual": porosity,
        "n_particles": n_actual,
        "drift_speed": drift_speed,
        "duration": duration,
        "seed": seed,
    }

    return CrushingCaseResult(
        time=time_arr,
        F_x=F_x_arr,
        F_max=F_max,
        F_steady=F_steady,
        porosity=porosity,
        pile_length=pile_length,
        pile_height=pile_height,
        F_ISO=F_ISO,
        config=config,
        final_pos=pos.copy(),
        final_n=n_actual,
        z_eq=z_eq,
        R_struct=R_struct,
    )


def run_porosity_sweep(
    config: PorosityScalingConfig | None = None,
) -> PorosityScalingResult:
    """Run a parameter sweep over porosity values."""
    if config is None:
        config = PorosityScalingConfig()

    porosity_arr = np.linspace(
        config.porosity_range[0],
        config.porosity_range[1],
        config.n_porosity,
    )

    F_steady_arr = np.zeros(config.n_porosity)
    actual_porosity_arr = np.zeros(config.n_porosity)

    for i, n_target in enumerate(porosity_arr):
        result = run_crushing_case(
            sigma_c=config.sigma_c,
            h=config.h,
            D=config.D,
            shape=config.shape,
            porosity_target=n_target,
            n_particles=config.n_particles,
            drift_speed=config.drift_speed,
            duration=15.0,
            seed=config.seed + i,
        )
        F_steady_arr[i] = result.F_steady
        actual_porosity_arr[i] = result.porosity

    cfg_iso = ISO19906Config(
        sigma_c=config.sigma_c,
        h=config.h,
        D=config.D,
        shape=config.shape,
    )
    F_ISO = iso19906_nominal_load(cfg_iso)

    # Use actual porosity for fitting
    alpha, alpha_std, R2 = fit_porosity_scaling(actual_porosity_arr, F_steady_arr, F_ISO)

    return PorosityScalingResult(
        porosity=actual_porosity_arr,
        F_steady=F_steady_arr,
        F_ISO=F_ISO,
        alpha=alpha,
        alpha_std=alpha_std,
        R2=R2,
        config=config,
    )
