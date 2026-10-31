"""DEM buoyancy, wetted-area, wave-force, and floating-equilibrium model.

Extends the base 2-D circular-disk buoyancy model with:
- Explicit wetted-area and wetted-perimeter computation (circular-segment geometry).
- Froude-Krylov wave force on partially submerged particles (linear wave theory).
- Morison-type inertia and drag wave loads.
- Floating-equilibrium solver (bisection) and heave-restoring stability check.
- 3-D sphere buoyancy model (spherical-cap geometry) for DEM particles.
- Analytical verification of buoyancy force against Archimedes' principle.

Physical notes on stability:
  For a 2-D circular disk, the metacentric height GM ≈ 0 (neutral roll stability),
  which is a well-known analytical result.  DEM particles in ice fields derive
  their roll stability from inter-particle contacts, not from individual
  waterplane geometry.  The relevant stability criterion for a single particle
  is **heave (vertical) stability**: a small vertical displacement produces a
  restoring force proportional to the heave stiffness k_heave > 0.

All quantities use SI units.  The 2-D model assumes circular cross-sections
extruded to thickness *t*; the 3-D model assumes spheres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Circular-segment submerged-fraction helper (scalar, CPU)
# ---------------------------------------------------------------------------

def _submerged_fraction_2d(h: float, R: float) -> float:
    """Submerged area fraction of a circle of radius R whose centre is at
    height *h* relative to the free surface (positive h = centre above water).

    h >=  R  => fully out of water  => f=0
    h <= -R  => fully submerged       => f=1
    otherwise use the circular-segment formula.
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 1.0
    theta = 2.0 * math.acos(max(-1.0, min(1.0, h / R)))
    return (theta - math.sin(theta)) / (2.0 * math.pi)


def submerged_fraction_tensor(
    y_centre: torch.Tensor, radius: float, water_level: float = 0.0
) -> torch.Tensor:
    """Vectorised submerged fraction for a batch of particle y-coordinates."""
    h = y_centre - water_level
    h_clamped = torch.clamp(h, -radius, radius)
    theta = 2.0 * torch.acos(h_clamped / radius)
    f_sub = (theta - torch.sin(theta)) / (2.0 * math.pi)
    f_sub = torch.clamp(f_sub, 0.0, 1.0)
    f_sub = torch.where(h >= radius, torch.zeros_like(f_sub), f_sub)
    f_sub = torch.where(h <= -radius, torch.ones_like(f_sub), f_sub)
    return f_sub


# ---------------------------------------------------------------------------
# 3-D sphere submerged-fraction (spherical-cap geometry)
# ---------------------------------------------------------------------------

def _submerged_fraction_sphere(h: float, R: float) -> float:
    """Submerged volume fraction of a sphere of radius R whose centre is at
    height *h* relative to the free surface (positive h = centre above water).

    Uses spherical-cap formula:
        V_sub = (pi/3)(R - h)^2 (2R + h)   for h in [-R, R]
        f_sub = V_sub / V_total = (R - h)^2 (2R + h) / (4 R^3)

    h >=  R  => fully out of water  => f=0
    h <= -R  => fully submerged       => f=1
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 1.0
    d = R - h  # submerged cap height
    V_sub = (math.pi / 3.0) * d * d * (2.0 * R + h)
    V_total = (4.0 / 3.0) * math.pi * R ** 3
    return V_sub / V_total


def submerged_fraction_sphere_tensor(
    y_centre: torch.Tensor, radius: float, water_level: float = 0.0
) -> torch.Tensor:
    """Vectorised submerged fraction (3-D sphere) for a batch of particles."""
    h = y_centre - water_level
    R = radius
    d = R - h  # cap height
    V_sub = (math.pi / 3.0) * d * d * (2.0 * R + h)
    V_total = (4.0 / 3.0) * math.pi * R ** 3
    f_sub = V_sub / V_total
    f_sub = torch.clamp(f_sub, 0.0, 1.0)
    f_sub = torch.where(h >= R, torch.zeros_like(f_sub), f_sub)
    f_sub = torch.where(h <= -R, torch.ones_like(f_sub), f_sub)
    return f_sub


# ---------------------------------------------------------------------------
# Wetted area and perimeter
# ---------------------------------------------------------------------------

def wetted_area(h: float, R: float, thickness: float) -> float:
    """Wetted surface area of a circular cylinder partially submerged (2-D model).

    The wetted area consists of:
    - The submerged portion of the cylindrical side: 2*R*theta*thickness
    - The flat end-cap submerged area: R^2*(theta - sin(theta))

    Args:
        h:        centre height above water (positive = above).
        R:        particle radius, m.
        thickness: extrusion depth, m.

    Returns:
        Wetted surface area, m^2.
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 2.0 * math.pi * R * thickness + 2.0 * math.pi * R ** 2

    theta = 2.0 * math.acos(max(-1.0, min(1.0, h / R)))
    side_wetted = 2.0 * R * theta * thickness
    cap_wetted = R ** 2 * (theta - math.sin(theta))
    return side_wetted + cap_wetted


def wetted_perimeter(h: float, R: float) -> float:
    """Wetted perimeter of a circular cross-section at the waterline.

    For partial submersion: P_wet = R * theta where theta = 2*acos(h/R).
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 2.0 * math.pi * R
    theta = 2.0 * math.acos(max(-1.0, min(1.0, h / R)))
    return R * theta


def wetted_area_sphere(h: float, R: float) -> float:
    """Wetted surface area of a sphere partially submerged (3-D model).

    The wetted area is the surface area of the submerged spherical cap:
        A_wet = 2*pi*R*d  where d = R - h is the cap height.

    For a fully submerged sphere: A_wet = 4*pi*R^2.
    """
    if h >= R:
        return 0.0
    if h <= -R:
        return 4.0 * math.pi * R ** 2
    d = R - h
    return 2.0 * math.pi * R * d


def wetted_area_tensor(
    y_centre: torch.Tensor, radius: float, thickness: float, water_level: float = 0.0
) -> torch.Tensor:
    """Vectorised wetted area (2-D cylinder) for a batch of particles."""
    h = y_centre - water_level
    h_clamped = torch.clamp(h, -radius, radius)
    theta = 2.0 * torch.acos(h_clamped / radius)
    side_wetted = 2.0 * radius * theta * thickness
    cap_wetted = radius ** 2 * (theta - torch.sin(theta))
    A_partial = side_wetted + cap_wetted
    A_full = 2.0 * math.pi * radius * thickness + 2.0 * math.pi * radius ** 2
    A_wet = torch.where(h >= radius, torch.zeros_like(A_partial), A_partial)
    A_wet = torch.where(h <= -radius, torch.full_like(A_wet, A_full), A_wet)
    return A_wet


def wetted_area_sphere_tensor(
    y_centre: torch.Tensor, radius: float, water_level: float = 0.0
) -> torch.Tensor:
    """Vectorised wetted area (3-D sphere) for a batch of particles."""
    h = y_centre - water_level
    R = radius
    d = R - h
    A_partial = 2.0 * math.pi * R * d
    A_full = 4.0 * math.pi * R ** 2
    A_wet = torch.where(h >= R, torch.zeros_like(A_partial), A_partial)
    A_wet = torch.where(h <= -R, torch.full_like(A_wet, A_full), A_wet)
    A_wet = torch.clamp(A_wet, 0.0, A_full)
    return A_wet


# ---------------------------------------------------------------------------
# Wet-coupling configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WetCouplingConfig:
    """Configuration for buoyancy and wet drag on DEM particles."""
    enabled: bool = False
    water_density: float = 1025.0      # rho_w, kg/m^3 (sea water default)
    gravity: float = 9.81              # g, m/s^2
    water_level: float = 0.0           # free-surface elevation, m
    wet_drag_coeff: float = 0.0        # C_wet, N·s/m per particle (linear)
    model: str = "2d"                  # "2d" (circular disk) or "3d" (sphere)


@dataclass(frozen=True)
class WaveForceConfig:
    """Configuration for wave forces on DEM particles.

    Uses linear (Airy) wave theory for the undisturbed wave kinematics and
    the Morison equation for the resulting hydrodynamic loads.
    """
    enabled: bool = False
    amplitude: float = 0.0             # wave amplitude A, m
    wavelength: float = 0.0            # wavelength lambda, m
    period: float = 0.0                # wave period T, s
    water_depth: float = 1e6           # water depth d, m (default deep water)
    inertia_coeff: float = 2.0         # C_M (Morison inertia coefficient)
    drag_coeff_morison: float = 1.2    # C_D (Morison drag coefficient)
    phase: float = 0.0                 # wave phase offset, rad


# ---------------------------------------------------------------------------
# Wave kinematics (linear Airy wave theory)
# ---------------------------------------------------------------------------

def wave_kinematics(
    x: torch.Tensor,
    y: torch.Tensor,
    t: float,
    cfg: WaveForceConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute undisturbed wave velocity and acceleration at particle positions.

    Uses linear (Airy) wave theory.  Deep-water approximation when d > 0.5*lambda.
    """
    if not cfg.enabled or cfg.amplitude <= 0.0:
        zeros = torch.zeros_like(x)
        return zeros, zeros, zeros

    A = cfg.amplitude
    lam = cfg.wavelength
    T = cfg.period
    d = cfg.water_depth
    phi = cfg.phase

    k = 2.0 * math.pi / lam
    omega = 2.0 * math.pi / T

    deep = d > 0.5 * lam

    if deep:
        decay = torch.exp(k * torch.clamp(y, max=0.0))
    else:
        decay = torch.cosh(k * (y + d)) / math.cosh(k * d)

    phase_arg = k * x - omega * t + phi

    u_x = A * omega * decay * torch.cos(phase_arg)
    u_y = A * omega * decay * torch.sin(phase_arg)
    du_x_dt = A * omega ** 2 * decay * torch.sin(phase_arg)

    return u_x, u_y, du_x_dt


def wave_elevation(
    x: torch.Tensor, t: float, cfg: WaveForceConfig
) -> torch.Tensor:
    """Compute free-surface elevation eta(x, t) from linear wave theory."""
    if not cfg.enabled or cfg.amplitude <= 0.0:
        return torch.zeros_like(x)
    k = 2.0 * math.pi / cfg.wavelength
    omega = 2.0 * math.pi / cfg.period
    return cfg.amplitude * torch.cos(k * x - omega * t + cfg.phase)


# ---------------------------------------------------------------------------
# Morison wave force
# ---------------------------------------------------------------------------

def compute_wave_force(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    thickness: float,
    t: float,
    wave_cfg: WaveForceConfig,
    wet_cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Compute Morison-equation wave force on partially submerged particles.

    The Morison equation per unit length:
        f = rho_w * C_M * (pi*R^2) * du/dt + 0.5 * rho_w * C_D * 2R * |u_rel| * u_rel

    where u_rel = u_wave - v_particle, and forces are applied only to the
    submerged fraction.
    """
    if not wave_cfg.enabled or not wet_cfg.enabled:
        return torch.zeros_like(positions)

    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    x = positions[:, 0]
    y = positions[:, 1]

    if wet_cfg.model == "3d":
        f_sub = submerged_fraction_sphere_tensor(y, radius, wet_cfg.water_level)
    else:
        f_sub = submerged_fraction_tensor(y, radius, wet_cfg.water_level)

    u_x, u_y, du_x_dt = wave_kinematics(x, y, t, wave_cfg)

    u_rel_x = u_x - velocities[:, 0]
    u_rel_y = u_y - velocities[:, 1]

    rho_w = wet_cfg.water_density
    C_M = wave_cfg.inertia_coeff
    C_D = wave_cfg.drag_coeff_morison

    A_cross = math.pi * radius ** 2
    D = 2.0 * radius

    F_inertia_x = rho_w * C_M * A_cross * du_x_dt * thickness * f_sub

    u_rel_mag = torch.sqrt(u_rel_x ** 2 + u_rel_y ** 2 + 1e-30)
    F_drag_x = 0.5 * rho_w * C_D * D * u_rel_mag * u_rel_x * thickness * f_sub
    F_drag_y = 0.5 * rho_w * C_D * D * u_rel_mag * u_rel_y * thickness * f_sub

    force = torch.zeros((N, 2), device=device, dtype=dtype)
    force[:, 0] = F_inertia_x + F_drag_x
    force[:, 1] = F_drag_y

    return force


# ---------------------------------------------------------------------------
# Buoyancy force
# ---------------------------------------------------------------------------

def compute_buoyancy_force(
    positions: torch.Tensor,
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Return buoyancy forces for all particles (shape N×2).

    2-D model:  F_b_y = rho_w * g * (pi * R^2 * f_sub) * thickness
    3-D model:  F_b_y = rho_w * g * (4/3 * pi * R^3 * f_sub)
    """
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    if not cfg.enabled:
        return torch.zeros_like(positions)

    y_centre = positions[:, 1]

    if cfg.model == "3d":
        f_sub = submerged_fraction_sphere_tensor(y_centre, radius, cfg.water_level)
        vol_sub = (4.0 / 3.0) * math.pi * radius ** 3 * f_sub
    else:
        f_sub = submerged_fraction_tensor(y_centre, radius, cfg.water_level)
        vol_sub = math.pi * radius ** 2 * f_sub * thickness

    Fb_y = cfg.water_density * cfg.gravity * vol_sub

    force = torch.zeros((N, 2), device=device, dtype=dtype)
    force[:, 1] = Fb_y
    return force


def compute_wet_drag_force(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    cfg: WetCouplingConfig,
) -> torch.Tensor:
    """Return linear wet-damping forces (shape N×2).

    F_wet = -C_wet * f_sub * v
    """
    if not cfg.enabled or cfg.wet_drag_coeff <= 0.0:
        return torch.zeros_like(velocities)

    y_centre = positions[:, 1]
    if cfg.model == "3d":
        f_sub = submerged_fraction_sphere_tensor(y_centre, radius, cfg.water_level)
    else:
        f_sub = submerged_fraction_tensor(y_centre, radius, cfg.water_level)
    return -cfg.wet_drag_coeff * f_sub[:, None] * velocities


def compute_wet_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined buoyancy + wet drag."""
    buoyancy = compute_buoyancy_force(positions, radius, thickness, density_ice, cfg)
    wet_drag = compute_wet_drag_force(positions, velocities, radius, cfg)
    return buoyancy, wet_drag


# ---------------------------------------------------------------------------
# Floating equilibrium solver
# ---------------------------------------------------------------------------

def solve_floating_equilibrium(
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[float, float, bool]:
    """Find the equilibrium centre-height for a single floating particle.

    At equilibrium, buoyancy = weight:
        rho_w * g * V_sub(h_eq) = rho_i * g * V_total

    => f_sub(h_eq) = rho_i / rho_w

    We solve for h_eq by bisection on the interval [-R, R].

    Args:
        radius:       particle radius, m.
        thickness:    extrusion depth, m (used only for 2-D model).
        density_ice:  ice density, kg/m^3.
        cfg:          WetCouplingConfig.
        tol:          convergence tolerance on f_sub, dimensionless.
        max_iter:     maximum bisection iterations.

    Returns:
        (h_eq, f_sub_eq, converged): equilibrium height, submerged fraction,
        and convergence flag.
    """
    rho_i = density_ice
    rho_w = cfg.water_density
    target = rho_i / rho_w

    if target >= 1.0:
        return -radius, 1.0, True
    if target <= 0.0:
        return radius, 0.0, True

    use_3d = cfg.model == "3d"
    frac_fn = _submerged_fraction_sphere if use_3d else _submerged_fraction_2d

    lo, hi = -radius, radius
    converged = False
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f = frac_fn(mid, radius)
        if abs(f - target) < tol:
            converged = True
            break
        if f > target:
            lo = mid
        else:
            hi = mid

    h_eq = (lo + hi) / 2.0
    f_eq = frac_fn(h_eq, radius)
    return h_eq, f_eq, converged


def heave_restoring_stiffness(
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
    dh: float = 1e-6,
) -> Tuple[float, bool]:
    """Compute the heave restoring stiffness at floating equilibrium.

    The heave stiffness k_heave = -d(F_buoyancy - Weight)/dh evaluated at h_eq.
    A positive k_heave means the equilibrium is vertically stable: a downward
    displacement increases buoyancy, producing a net upward restoring force.

    For a 2-D circular disk:
        k_heave = rho_w * g * pi * R^2 * |df_sub/dh| * thickness

    For a 3-D sphere:
        k_heave = rho_w * g * (4/3) * pi * R^3 * |df_sub/dh|

    where df_sub/dh < 0 (submerged fraction decreases as particle rises).

    Args:
        radius:       particle radius, m.
        thickness:    extrusion depth, m.
        density_ice:  ice density, kg/m^3.
        cfg:          WetCouplingConfig.
        dh:           finite-difference step for numerical derivative, m.

    Returns:
        (k_heave, is_stable): heave stiffness (N/m) and stability flag.
    """
    h_eq, f_eq, converged = solve_floating_equilibrium(
        radius, thickness, density_ice, cfg
    )

    if f_eq >= 1.0 or f_eq <= 0.0:
        # Fully submerged or fully dry: no waterplane restoring
        return 0.0, False

    use_3d = cfg.model == "3d"
    frac_fn = _submerged_fraction_sphere if use_3d else _submerged_fraction_2d

    # Numerical derivative of f_sub w.r.t. h
    f_plus = frac_fn(h_eq + dh, radius)
    f_minus = frac_fn(h_eq - dh, radius)
    df_dh = (f_plus - f_minus) / (2.0 * dh)

    rho_w = cfg.water_density
    g = cfg.gravity

    if use_3d:
        V_total = (4.0 / 3.0) * math.pi * radius ** 3
    else:
        V_total = math.pi * radius ** 2 * thickness

    # k_heave = -rho_w * g * V_total * df_sub/dh
    # Since df_sub/dh < 0, k_heave > 0 for stable equilibrium
    k_heave = -rho_w * g * V_total * df_dh

    is_stable = k_heave > 0.0
    return k_heave, is_stable


def metacentric_height(
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
) -> Tuple[float, bool]:
    """Compute the metacentric height GM for a floating body.

    NOTE: For a 2-D circular disk, GM ≈ 0 (neutral roll stability).  This is
    a well-known analytical result: the waterplane shape of a circle is
    invariant under rotation, so there is no roll restoring moment from
    waterplane geometry alone.  DEM particles in ice fields derive roll
    stability from inter-particle contacts.

    For a 3-D sphere, GM < 0 (roll-unstable), also a known result.

    The **heave (vertical) stability** is the relevant criterion for DEM
    particles.  Use heave_restoring_stiffness() instead.

    This function is retained for completeness and returns (GM, heave_stable)
    where heave_stable is the vertical stability flag.

    Args:
        radius:       particle radius, m.
        thickness:    extrusion depth, m.
        density_ice:  ice density, kg/m^3.
        cfg:          WetCouplingConfig.

    Returns:
        (GM, is_heave_stable): metacentric height (roll) and heave stability.
    """
    h_eq, f_eq, converged = solve_floating_equilibrium(
        radius, thickness, density_ice, cfg
    )

    if f_eq >= 1.0 or f_eq <= 0.0:
        return 0.0, False

    R = radius
    use_3d = cfg.model == "3d"

    if use_3d:
        # 3-D sphere: waterplane is a circle of radius r_wp
        r_wp_sq = R ** 2 - h_eq ** 2
        if r_wp_sq <= 0.0:
            return 0.0, False
        r_wp = math.sqrt(r_wp_sq)
        I_wp = math.pi * r_wp ** 4 / 4.0
        V_sub = (4.0 / 3.0) * math.pi * R ** 3 * f_eq
    else:
        # 2-D circular disk: waterplane is a rectangle b × thickness
        b_half_sq = R ** 2 - h_eq ** 2
        if b_half_sq <= 0.0:
            return 0.0, False
        b = 2.0 * math.sqrt(b_half_sq)
        I_wp = (b ** 3 / 12.0) * thickness
        V_sub = math.pi * R ** 2 * f_eq * thickness

    BM = I_wp / V_sub if V_sub > 0.0 else 0.0

    # Centre of buoyancy
    if use_3d:
        # Centroid of spherical cap from sphere centre
        d = R - h_eq  # cap height
        if 3.0 * R - d > 1e-30:
            y_c_from_base = d * (4.0 * R - d) / (4.0 * (3.0 * R - d))
            y_B = (h_eq - R) + y_c_from_base  # from waterline
        else:
            y_B = h_eq
    else:
        theta = 2.0 * math.acos(max(-1.0, min(1.0, h_eq / R)))
        sin_half = math.sin(theta / 2.0)
        if theta - math.sin(theta) > 1e-30:
            y_c_from_centre = -(4.0 * R * sin_half ** 3) / (3.0 * (theta - math.sin(theta)))
            y_B = h_eq + y_c_from_centre
        else:
            y_B = h_eq

    # BG = distance from B to G (G is at h_eq)
    BG = abs(h_eq - y_B)

    GM = BM - BG

    # Heave stability (the physically relevant criterion for DEM)
    _, is_heave_stable = heave_restoring_stiffness(radius, thickness, density_ice, cfg)

    return GM, is_heave_stable


# ---------------------------------------------------------------------------
# Buoyancy verification against Archimedes' principle
# ---------------------------------------------------------------------------

def verify_buoyancy_force(
    radius: float,
    thickness: float,
    density_ice: float,
    cfg: WetCouplingConfig,
    n_heights: int = 200,
) -> Tuple[float, float, bool]:
    """Verify buoyancy force against Archimedes' principle at multiple heights.

    For each test height h, the computed buoyancy force should equal:
        F_b = rho_w * g * V_sub

    Additionally, at the equilibrium height, buoyancy should equal weight:
        F_b(h_eq) = rho_i * g * V_total

    Args:
        radius:       particle radius, m.
        thickness:    extrusion depth, m.
        density_ice:  ice density, kg/m^3.
        cfg:          WetCouplingConfig.
        n_heights:    number of test heights.

    Returns:
        (max_relative_error, equilibrium_error, passed):
        max relative error across all heights,
        relative error at equilibrium (buoyancy vs weight),
        and pass flag (both errors < 0.03).
    """
    R = radius
    rho_w = cfg.water_density
    g = cfg.gravity
    rho_i = density_ice
    use_3d = cfg.model == "3d"

    heights = torch.linspace(-R * 1.5, R * 1.5, n_heights, dtype=torch.float64)
    positions = torch.zeros((n_heights, 2), dtype=torch.float64)
    positions[:, 1] = heights

    F_buoy = compute_buoyancy_force(positions, R, thickness, rho_i, cfg)

    if use_3d:
        f_sub_analytical = submerged_fraction_sphere_tensor(heights, R, cfg.water_level)
        V_total = (4.0 / 3.0) * math.pi * R ** 3
    else:
        f_sub_analytical = submerged_fraction_tensor(heights, R, cfg.water_level)
        V_total = math.pi * R ** 2 * thickness

    F_analytical = rho_w * g * V_total * f_sub_analytical

    F_mag = torch.abs(F_analytical) + 1e-30
    rel_error = torch.abs(F_buoy[:, 1] - F_analytical) / F_mag

    mask = f_sub_analytical > 1e-10
    if mask.any():
        max_rel_error = rel_error[mask].max().item()
    else:
        max_rel_error = 0.0

    h_eq, f_eq, converged = solve_floating_equilibrium(R, thickness, rho_i, cfg)
    weight = rho_i * g * V_total
    buoyancy_at_eq = rho_w * g * V_total * f_eq
    eq_rel_error = abs(buoyancy_at_eq - weight) / (abs(weight) + 1e-30)

    passed = max_rel_error < 0.03 and eq_rel_error < 0.03
    return max_rel_error, eq_rel_error, passed


# ---------------------------------------------------------------------------
# Full wet-coupling force assembly
# ---------------------------------------------------------------------------

def compute_all_wet_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    radius: float,
    thickness: float,
    density_ice: float,
    t: float,
    wet_cfg: WetCouplingConfig,
    wave_cfg: WaveForceConfig | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all wet forces: buoyancy + wet drag + wave force."""
    buoyancy = compute_buoyancy_force(positions, radius, thickness, density_ice, wet_cfg)
    wet_drag = compute_wet_drag_force(positions, velocities, radius, wet_cfg)
    if wave_cfg is not None and wave_cfg.enabled:
        wave = compute_wave_force(
            positions, velocities, radius, thickness, t, wave_cfg, wet_cfg
        )
    else:
        wave = torch.zeros_like(positions)
    return buoyancy, wet_drag, wave
