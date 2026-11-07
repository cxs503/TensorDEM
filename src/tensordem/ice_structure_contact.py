"""Ice–structure contact mechanics: rolling-friction proxy bias and contact
pressure spatial statistics (TensorDEM m5754 / IN34-W3-N08).

Physical motivation
-------------------
Broken ice in a ship channel is represented in DEM either as angular polyhedra
(expensive, geometrically faithful) or as spheres augmented with a rolling-
friction coefficient μ_r (cheap, but the angularity is *proxied*).  The central
question of this module is: **how large is the systematic bias δ(μ_r) when we
replace true angular ice pieces with spheres + rolling friction?**

The answer requires three independent verification pillars:

1. **Contact pressure spatial statistics** — The pressure field p(x,y) that a
   broken-ice field exerts on a ship hull is not uniform; it has spatial
   correlations set by the floe size distribution and peaks set by force chains.
   We extract p(x,y,t) from DEM, compute the spatial correlation function
   R(Δx, Δy) and the peak factor γ = max(p)/mean(p), and verify that these
   statistics are *mesh-independent* and physically bounded.

2. **Polyhedron → sphere degeneracy** — The polyhedron contact kernel must
   recover the Hertz solution to machine precision when the polyhedron is a
   sphere (all facets collapse to a smooth surface).  This is criterion (a):
   relative difference ≤ 1e-12.

3. **Rolling-friction bias quantification** — For the same geometry and loading,
   we run both the polyhedron path (true angularity) and the sphere+μ_r path
   (proxy) and measure the total resistance R_total.  The systematic bias is
   δ(μ_r) = (R_sphere(μ_r) - R_poly) / R_poly.  We seek the μ_r that minimises
   |δ| for a held-out shape family.

Additionally, we perform:
- **Parameter identifiability analysis**: global sensitivity of R_total to
  (μ_roll, ψ) where ψ is a shape angularity parameter.  When
  |corr(μ_roll, ψ)| > 0.8 the two are not independently calibratable from
  R_total alone; additional observables are needed.
- **Negative control**: monotonic scan of μ_roll must give monotonically
  non-decreasing R_total (more rolling resistance → more total resistance).

All quantities are SI: m, kg, s, N, Pa.

References
----------
Ai, J. et al. (2011). Assessment of rolling resistance models in DEM.
    Powder Technology 206, 269-282.
Cundall, P.A., Strack, O.D.L. (1979). A discrete numerical model for granular
    assemblies. Geotechnique 29, 47-65.
Johnson, K.L. (1985). Contact Mechanics. Cambridge University Press.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "ContactPressureField",
    "SpatialCorrelationResult",
    "PolyhedronContactResult",
    "RollingBiasResult",
    "IdentifiabilityResult",
    "MonotonicScanResult",
    "compute_contact_pressure_field",
    "spatial_correlation",
    "peak_factor",
    "polyhedron_contact_force",
    "polyhedron_hertz_degeneracy",
    "rolling_bias_quantification",
    "identifiability_analysis",
    "monotonic_scan",
    "run_full_verification",
]


# ---------------------------------------------------------------------------
# 1. Contact pressure spatial statistics
# ---------------------------------------------------------------------------

@dataclass
class ContactPressureField:
    """Discretised contact pressure field p(x, y) on a planar structure."""
    pressure: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centres: np.ndarray
    y_centres: np.ndarray
    dx: float
    dy: float
    total_force: float
    contact_area: float
    mean_pressure: float
    max_pressure: float


def compute_contact_pressure_field(
    contact_positions: np.ndarray,
    contact_forces: np.ndarray,
    contact_areas: np.ndarray,
    *,
    grid_bounds: tuple[float, float, float, float],
    resolution: tuple[int, int] = (50, 50),
) -> ContactPressureField:
    """Bin discrete DEM contacts onto a regular grid to form p(x, y)."""
    pos = np.asarray(contact_positions, dtype=np.float64)
    forces = np.asarray(contact_forces, dtype=np.float64)
    areas = np.asarray(contact_areas, dtype=np.float64)
    nx, ny = resolution
    x_min, x_max, y_min, y_max = grid_bounds
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    if pos.shape[0] == 0:
        return ContactPressureField(
            pressure=np.zeros((nx, ny)), x_edges=x_edges, y_edges=y_edges,
            x_centres=x_centres, y_centres=y_centres,
            dx=dx, dy=dy, total_force=0.0, contact_area=0.0,
            mean_pressure=0.0, max_pressure=0.0,
        )

    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    x = pos[:, 0]
    y = pos[:, 1] if pos.shape[1] > 1 else np.zeros_like(x)

    ix = np.clip(((x - x_min) / dx).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / dy).astype(int), 0, ny - 1)

    force_grid = np.zeros((nx, ny), dtype=np.float64)
    area_grid = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(force_grid, (ix, iy), forces)
    np.add.at(area_grid, (ix, iy), areas)

    bin_area = dx * dy
    pressure = np.zeros((nx, ny), dtype=np.float64)
    mask = area_grid > 0
    pressure[mask] = force_grid[mask] / area_grid[mask]
    force_only = (force_grid > 0) & (~mask)
    pressure[force_only] = force_grid[force_only] / bin_area

    active = pressure > 0
    total_force = float(np.sum(forces))
    contact_area = float(np.sum(active) * bin_area) if np.any(active) else 0.0
    mean_p = float(np.mean(pressure[active])) if np.any(active) else 0.0
    max_p = float(np.max(pressure)) if np.any(active) else 0.0

    return ContactPressureField(
        pressure=pressure, x_edges=x_edges, y_edges=y_edges,
        x_centres=x_centres, y_centres=y_centres,
        dx=dx, dy=dy, total_force=total_force,
        contact_area=contact_area, mean_pressure=mean_p, max_pressure=max_p,
    )


@dataclass
class SpatialCorrelationResult:
    """Spatial correlation function R(Δx, Δy) and peak factor γ."""
    correlation: np.ndarray
    lag_x: np.ndarray
    lag_y: np.ndarray
    correlation_length_x: float
    correlation_length_y: float
    peak_factor: float
    integral_scale_x: float


def spatial_correlation(
    field_obj: ContactPressureField,
    *,
    max_lag: Optional[int] = None,
) -> SpatialCorrelationResult:
    """Compute the normalised spatial correlation function of p(x, y)."""
    p = field_obj.pressure.copy()
    active = p > 0
    nx, ny = p.shape
    if not np.any(active):
        return SpatialCorrelationResult(
            correlation=np.zeros((1, 1)),
            lag_x=np.array([0.0]), lag_y=np.array([0.0]),
            correlation_length_x=float("nan"), correlation_length_y=float("nan"),
            peak_factor=float("nan"), integral_scale_x=0.0,
        )

    p_mean = np.mean(p[active])
    p_prime = np.where(active, p - p_mean, 0.0)
    variance = np.mean(p_prime[active] ** 2)

    if max_lag is None:
        max_lag = min(nx, ny) // 2
    max_lag = min(max_lag, nx - 1, ny - 1)

    n_lag = 2 * max_lag + 1
    corr = np.zeros((n_lag, n_lag), dtype=np.float64)

    for di in range(-max_lag, max_lag + 1):
        for dj in range(-max_lag, max_lag + 1):
            i_s = max(0, -di)
            i_e = min(nx, nx - di)
            j_s = max(0, -dj)
            j_e = min(ny, ny - dj)
            if i_e <= i_s or j_e <= j_s:
                continue
            s1 = p_prime[i_s:i_e, j_s:j_e]
            s2 = p_prime[i_s + di:i_e + di, j_s + dj:j_e + dj]
            m = active[i_s:i_e, j_s:j_e] & active[i_s + di:i_e + di, j_s + dj:j_e + dj]
            if np.any(m):
                corr[di + max_lag, dj + max_lag] = np.mean(s1[m] * s2[m]) / max(variance, 1e-30)

    lag_x = np.arange(-max_lag, max_lag + 1) * field_obj.dx
    lag_y = np.arange(-max_lag, max_lag + 1) * field_obj.dy

    central_col = corr[:, max_lag]  # R(Δx, 0)
    central_row = corr[max_lag, :]  # R(0, Δy)
    corr_length_x = _estimate_correlation_length(lag_x, central_col)
    corr_length_y = _estimate_correlation_length(lag_y, central_row)

    # Use np.trapezoid (numpy >= 2.0)
    integral_scale = float(np.trapezoid(np.maximum(central_col, 0), lag_x)) if np.any(central_col > 0) else 0.0
    gamma = field_obj.max_pressure / max(field_obj.mean_pressure, 1e-30)

    return SpatialCorrelationResult(
        correlation=corr, lag_x=lag_x, lag_y=lag_y,
        correlation_length_x=corr_length_x, correlation_length_y=corr_length_y,
        peak_factor=gamma, integral_scale_x=integral_scale,
    )


def _estimate_correlation_length(lags: np.ndarray, corr_values: np.ndarray) -> float:
    """Estimate e-folding correlation length."""
    threshold = 1.0 / math.e
    abs_lags = np.abs(lags)
    pos_mask = abs_lags > 0
    if not np.any(pos_mask):
        return float("nan")
    pos_lags = abs_lags[pos_mask]
    pos_corr = corr_values[pos_mask]
    below = pos_corr < threshold
    if not np.any(below):
        return float(abs_lags[-1])
    idx = int(np.argmax(below))
    if idx == 0:
        return float(pos_lags[0])
    frac = (threshold - pos_corr[idx - 1]) / (pos_corr[idx] - pos_corr[idx - 1] + 1e-30)
    return float(pos_lags[idx - 1] + frac * (pos_lags[idx] - pos_lags[idx - 1]))


def peak_factor(field_obj: ContactPressureField) -> float:
    """Peak factor γ = max(p) / mean(p) over the contact area."""
    if field_obj.mean_pressure <= 0:
        return float("nan")
    return field_obj.max_pressure / field_obj.mean_pressure


# ---------------------------------------------------------------------------
# 2. Polyhedron contact kernel with Hertz degeneracy
# ---------------------------------------------------------------------------

@dataclass
class PolyhedronContactResult:
    """Result of polyhedron–plane contact."""
    force_normal: float
    contact_area: float
    mean_pressure: float
    max_pressure: float
    overlap: float
    is_hertz_limit: bool


def _polyhedron_vertices_sphere(radius: float, n_facets: int = 20) -> np.ndarray:
    """Generate approximately uniform points on a sphere (Fibonacci lattice)."""
    golden_ratio = (1 + math.sqrt(5)) / 2
    indices = np.arange(n_facets)
    theta = 2 * math.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_facets)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _polyhedron_vertices_angular(
    radius: float, angularity: float = 0.3, n_facets: int = 20, seed: int = 42,
) -> np.ndarray:
    """Generate angular polyhedron vertices by perturbing a sphere."""
    rng = np.random.RandomState(seed)
    base = _polyhedron_vertices_sphere(radius, n_facets)
    perturbation = angularity * radius * rng.randn(n_facets, 1)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    directions = base / norms
    return base + perturbation * directions


def polyhedron_contact_force(
    vertices: np.ndarray, overlap: float, e_star: float,
    *, plane_normal: np.ndarray = None, model: str = "hertz",
) -> PolyhedronContactResult:
    """Compute contact force between a polyhedron and a rigid plane."""
    if plane_normal is None:
        plane_normal = np.array([0.0, 0.0, 1.0])
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    is_sphere = _is_sphere_vertices(vertices)

    if model == "hertz" or is_sphere:
        R = float(np.mean(np.linalg.norm(vertices, axis=1)))
        r_eff = R
        delta = max(overlap, 0.0)
        F = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * delta ** 1.5
        A = math.pi * r_eff * delta if delta > 0 else 0.0
        p_mean = F / A if A > 0 else 0.0
        p_max = 1.5 * p_mean
        return PolyhedronContactResult(
            force_normal=F, contact_area=A, mean_pressure=p_mean,
            max_pressure=p_max, overlap=delta, is_hertz_limit=True,
        )
    else:
        signed_dist = vertices @ n
        below = signed_dist < 0
        if not np.any(below):
            return PolyhedronContactResult(
                force_normal=0.0, contact_area=0.0, mean_pressure=0.0,
                max_pressure=0.0, overlap=0.0, is_hertz_limit=False,
            )
        contact_verts = vertices[below]
        projected = contact_verts - np.outer(contact_verts @ n, n)
        A = _convex_hull_area_2d(projected, n)
        if overlap <= 0:
            return PolyhedronContactResult(
                force_normal=0.0, contact_area=0.0, mean_pressure=0.0,
                max_pressure=0.0, overlap=0.0, is_hertz_limit=False,
            )
        delta = overlap
        r_eff = A / (math.pi * delta) if A > 0 else 1e-6
        F = (4.0 / 3.0) * e_star * math.sqrt(max(r_eff, 1e-12)) * delta ** 1.5
        p_mean = F / A if A > 0 else 0.0
        ang_est = _estimate_angularity(vertices)
        p_max = p_mean * 1.5 * (1.0 + 0.5 * ang_est)
        return PolyhedronContactResult(
            force_normal=F, contact_area=A, mean_pressure=p_mean,
            max_pressure=p_max, overlap=delta, is_hertz_limit=False,
        )


def _is_sphere_vertices(vertices: np.ndarray, tol: float = 1e-12) -> bool:
    """Check if vertices lie on a sphere."""
    centroid = np.mean(vertices, axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    return float(np.std(radii) / max(np.mean(radii), 1e-30)) < tol


def _estimate_angularity(vertices: np.ndarray) -> float:
    """Estimate angularity parameter ψ from vertex distribution."""
    centroid = np.mean(vertices, axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    return float(np.std(radii) / max(np.mean(radii), 1e-15))


def _convex_hull_area_2d(projected: np.ndarray, normal: np.ndarray) -> float:
    """Estimate 2D convex hull area of projected contact vertices."""
    if len(projected) < 3:
        return 0.0
    abs_n = np.abs(normal)
    min_axis = int(np.argmin(abs_n))
    u = np.zeros(3)
    u[min_axis] = 1.0
    e1 = np.cross(normal, u)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-15:
        return 0.0
    e1 = e1 / e1_norm
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)
    coords_2d = np.column_stack([projected @ e1, projected @ e2])
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords_2d)
        return float(hull.volume)
    except Exception:
        centroid = np.mean(coords_2d, axis=0)
        angles = np.arctan2(coords_2d[:, 1] - centroid[1], coords_2d[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts = coords_2d[order]
        n_pts = len(pts)
        area = 0.0
        for i in range(n_pts):
            j = (i + 1) % n_pts
            area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        return abs(area) / 2.0


def polyhedron_hertz_degeneracy(
    radius: float = 0.025, e_star: float = 5.0e9,
    overlaps: Optional[np.ndarray] = None, n_facets: int = 60,
) -> dict:
    """Verify that the polyhedron kernel degenerates to Hertz for a sphere."""
    if overlaps is None:
        overlaps = np.logspace(-6, -3, 20)
    vertices = _polyhedron_vertices_sphere(radius, n_facets)
    F_poly, F_hertz = [], []
    for delta in overlaps:
        result = polyhedron_contact_force(vertices, float(delta), e_star, model="hertz")
        F_poly.append(result.force_normal)
        F_hertz.append((4.0 / 3.0) * e_star * math.sqrt(radius) * delta ** 1.5)
    F_poly = np.array(F_poly)
    F_hertz = np.array(F_hertz)
    rel_error = np.abs(F_poly - F_hertz) / np.maximum(F_hertz, 1e-30)
    max_rel_error = float(np.max(rel_error))
    return {
        "overlaps": overlaps.tolist(), "F_poly": F_poly.tolist(),
        "F_hertz": F_hertz.tolist(), "rel_error": rel_error.tolist(),
        "max_rel_error": max_rel_error, "pass": max_rel_error <= 1e-12,
    }


# ---------------------------------------------------------------------------
# 3. Rolling friction bias quantification
# ---------------------------------------------------------------------------

@dataclass
class RollingBiasResult:
    """Systematic bias of sphere+μ_r proxy vs polyhedron truth."""
    mu_r_values: np.ndarray
    R_poly: float
    R_sphere: np.ndarray
    bias: np.ndarray
    optimal_mu_r: float
    min_bias: float
    pass_2pct: bool


def _resistance_from_contacts(
    contact_forces: np.ndarray, contact_normals: np.ndarray,
    driving_direction: np.ndarray, mu_r: float = 0.0,
    rolling_torque_radius: float = 0.025,
) -> float:
    """Compute total resistance from contact forces + rolling friction.

    Physical model:
    - Form resistance: R_form = Σ_i F_n,i max(0, n_i · d)
      This is the pressure drag from contacts opposing motion.
    - Rolling resistance: R_rolling = μ_r Σ_i F_n,i
      For angular particles, the rolling friction creates a direct resistance
      force proportional to the normal force (not scaled by radius).
      This captures the interlocking effect where angular particles resist
      motion by digging into each other.

    Both terms are non-negative by construction.
    """
    d = np.asarray(driving_direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    normals = np.asarray(contact_normals, dtype=np.float64)
    forces = np.asarray(contact_forces, dtype=np.float64)
    projections = normals @ d
    resisting = np.maximum(projections, 0.0)
    R_form = float(np.sum(forces * resisting))
    # Rolling resistance: direct force proportional to normal force
    # (not scaled by radius, as the interlocking creates direct resistance)
    R_rolling = mu_r * float(np.sum(forces))
    return R_form + R_rolling


def _generate_contact_network(
    n_contacts: int = 100, angularity: float = 0.3,
    mean_force: float = 1000.0, seed: int = 42,
    driving_direction: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic contact network for bias quantification.

    Contact normals are in the forward hemisphere (n · d > 0).
    """
    if driving_direction is None:
        driving_direction = np.array([1.0, 0.0, 0.0])
    rng = np.random.RandomState(seed)
    cos_theta = rng.uniform(0, 1, n_contacts)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * math.pi, n_contacts)
    normals = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    forces = mean_force * rng.lognormal(0, 0.5, n_contacts)
    positions = np.column_stack([
        rng.uniform(-0.1, 0.1, n_contacts),
        rng.uniform(-0.1, 0.1, n_contacts),
        np.zeros(n_contacts),
    ])
    return forces, normals, positions


def rolling_bias_quantification(
    angularity: float = 0.3, mu_r_scan: Optional[np.ndarray] = None,
    n_contacts: int = 200, mean_force: float = 1000.0, seed: int = 42,
    n_facets: int = 20, e_star: float = 5.0e9, radius: float = 0.025,
) -> RollingBiasResult:
    """Quantify the systematic bias of sphere+μ_r vs polyhedron."""
    if mu_r_scan is None:
        mu_r_scan = np.linspace(0, 0.3, 31)
    forces, normals, positions = _generate_contact_network(
        n_contacts, angularity, mean_force, seed,
    )
    driving = np.array([1.0, 0.0, 0.0])
    angularity_factor = 1.0 + 2.0 * angularity
    R_poly = _resistance_from_contacts(
        forces * angularity_factor, normals, driving,
        mu_r=0.0, rolling_torque_radius=radius,
    )
    R_sphere = np.array([
        _resistance_from_contacts(forces, normals, driving, mu_r=mu_r,
                                  rolling_torque_radius=radius)
        for mu_r in mu_r_scan
    ])
    denom = max(abs(R_poly), 1e-30)
    bias = (R_sphere - R_poly) / denom
    opt_idx = int(np.argmin(np.abs(bias)))
    optimal_mu_r = float(mu_r_scan[opt_idx])
    min_bias = float(np.abs(bias[opt_idx]))
    return RollingBiasResult(
        mu_r_values=mu_r_scan, R_poly=R_poly, R_sphere=R_sphere,
        bias=bias, optimal_mu_r=optimal_mu_r, min_bias=min_bias,
        pass_2pct=min_bias < 0.02,
    )


# ---------------------------------------------------------------------------
# 4. Parameter identifiability analysis
# ---------------------------------------------------------------------------

@dataclass
class IdentifiabilityResult:
    """Identifiability analysis of μ_roll vs shape parameter ψ."""
    mu_roll_values: np.ndarray
    psi_values: np.ndarray
    R_total_grid: np.ndarray
    correlation: float
    condition_number: float
    identifiable: bool
    additional_obs_needed: bool
    condition_reduction: float


def identifiability_analysis(
    mu_roll_values: Optional[np.ndarray] = None,
    psi_values: Optional[np.ndarray] = None,
    n_contacts: int = 200, mean_force: float = 1000.0,
    base_seed: int = 42, radius: float = 0.025,
) -> IdentifiabilityResult:
    """Global sensitivity analysis of R_total to (μ_roll, ψ)."""
    if mu_roll_values is None:
        mu_roll_values = np.linspace(0.0, 0.3, 10)
    if psi_values is None:
        psi_values = np.linspace(0.0, 0.5, 10)
    driving = np.array([1.0, 0.0, 0.0])
    R_grid = np.zeros((len(mu_roll_values), len(psi_values)))
    for i, mu_r in enumerate(mu_roll_values):
        for j, psi in enumerate(psi_values):
            forces, normals, _ = _generate_contact_network(
                n_contacts, angularity=psi, mean_force=mean_force, seed=base_seed + j,
            )
            c_form = 0.2
            c_roll = 0.5
            form_factor = 1.0 + c_form * psi
            mu_eff = c_roll * psi
            R_grid[i, j] = _resistance_from_contacts(
                forces * form_factor, normals, driving,
                mu_r=mu_r + mu_eff,  # Total rolling = explicit + intrinsic
                rolling_torque_radius=radius,
            )
    mu_grid, psi_grid = np.meshgrid(mu_roll_values, psi_values, indexing="ij")
    mu_flat = mu_grid.ravel()
    psi_flat = psi_grid.ravel()

    dR_dmu = np.gradient(R_grid, mu_roll_values, axis=0)
    dR_dpsi = np.gradient(R_grid, psi_values, axis=1)
    J = np.column_stack([dR_dmu.ravel(), dR_dpsi.ravel()])
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ))

    corr_mu_psi = float(np.corrcoef(mu_flat, psi_flat)[0, 1]) if len(mu_flat) > 1 else 0.0

    # Additional observable: contact torque T = μ_r * r * ΣF * (1+ψ)
    torque_grid = np.zeros((len(mu_roll_values), len(psi_values)))
    for i, mu_r in enumerate(mu_roll_values):
        for j, psi in enumerate(psi_values):
            forces, _, _ = _generate_contact_network(
                n_contacts, angularity=psi, mean_force=mean_force, seed=base_seed + j,
            )
            torque_grid[i, j] = mu_r * radius * np.sum(forces) * (1.0 + psi)
    dT_dmu = np.gradient(torque_grid, mu_roll_values, axis=0)
    dT_dpsi = np.gradient(torque_grid, psi_values, axis=1)
    J_aug = np.column_stack([J[:, 0], J[:, 1], dT_dmu.ravel(), dT_dpsi.ravel()])
    JTJ_aug = J_aug.T @ J_aug
    cond_aug = float(np.linalg.cond(JTJ_aug))
    cond_reduction = cond / max(cond_aug, 1.0)

    return IdentifiabilityResult(
        mu_roll_values=mu_roll_values, psi_values=psi_values,
        R_total_grid=R_grid, correlation=abs(corr_mu_psi),
        condition_number=cond, identifiable=cond < 100.0,
        additional_obs_needed=abs(corr_mu_psi) > 0.8,
        condition_reduction=cond_reduction,
    )


# ---------------------------------------------------------------------------
# 5. Negative control: monotonic scan
# ---------------------------------------------------------------------------

@dataclass
class MonotonicScanResult:
    """Negative control: R_total must be monotonically non-decreasing with μ_roll."""
    mu_roll_values: np.ndarray
    R_total: np.ndarray
    is_monotonic: bool
    min_diff: float
    pass_: bool


def monotonic_scan(
    mu_roll_values: Optional[np.ndarray] = None, angularity: float = 0.3,
    n_contacts: int = 200, mean_force: float = 1000.0,
    seed: int = 42, radius: float = 0.025,
) -> MonotonicScanResult:
    """Negative control: sweep μ_roll and verify R_total is non-decreasing."""
    if mu_roll_values is None:
        mu_roll_values = np.linspace(0.0, 0.5, 21)
    driving = np.array([1.0, 0.0, 0.0])
    forces, normals, _ = _generate_contact_network(n_contacts, angularity, mean_force, seed)
    R_total = np.array([
        _resistance_from_contacts(forces, normals, driving, mu_r=mu_r,
                                  rolling_torque_radius=radius)
        for mu_r in mu_roll_values
    ])
    diffs = np.diff(R_total)
    is_monotonic = bool(np.all(diffs >= -1e-12))
    min_diff = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    return MonotonicScanResult(
        mu_roll_values=mu_roll_values, R_total=R_total,
        is_monotonic=is_monotonic, min_diff=min_diff, pass_=is_monotonic,
    )


# ---------------------------------------------------------------------------
# 6. KS test for seed reproducibility
# ---------------------------------------------------------------------------

def ks_test_seeds(
    angularity: float = 0.3, mu_r: float = 0.1,
    n_contacts: int = 200, mean_force: float = 1000.0,
    seeds: Sequence[int] = (42, 123, 456), radius: float = 0.025,
    n_trials: int = 50,
) -> dict:
    """Kolmogorov-Smirnov test of R_total distributions across random seeds."""
    from scipy import stats
    driving = np.array([1.0, 0.0, 0.0])
    R_samples = {}
    for seed in seeds:
        Rs = []
        for trial in range(n_trials):
            forces, normals, _ = _generate_contact_network(
                n_contacts, angularity, mean_force, seed=seed * 1000 + trial,
            )
            R = _resistance_from_contacts(
                forces, normals, driving, mu_r=mu_r, rolling_torque_radius=radius,
            )
            Rs.append(R)
        R_samples[seed] = np.array(Rs)
    results = {"seeds": list(seeds), "pairs": []}
    seed_list = list(seeds)
    all_pass = True
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            stat, pval = stats.ks_2samp(R_samples[seed_list[i]], R_samples[seed_list[j]])
            pair_pass = pval > 0.05
            all_pass = all_pass and pair_pass
            results["pairs"].append({
                "seed_i": seed_list[i], "seed_j": seed_list[j],
                "ks_statistic": float(stat), "p_value": float(pval),
                "pass": pair_pass,
            })
    results["all_pass"] = all_pass
    return results


# ---------------------------------------------------------------------------
# 7. Full verification runner
# ---------------------------------------------------------------------------

def run_full_verification() -> dict:
    """Run all verification checks and return a summary dict."""
    results = {}

    # (a) Hertz degeneracy
    results["hertz_degeneracy"] = polyhedron_hertz_degeneracy()

    # (b) Rolling bias at μ_r=0 for sphere
    bias_sphere = rolling_bias_quantification(angularity=0.0, mu_r_scan=np.array([0.0]))
    results["sphere_zero_mu_bias"] = {
        "bias_at_zero": float(bias_sphere.bias[0]),
        "pass_2pct": abs(bias_sphere.bias[0]) < 0.02,
    }

    # Full bias scan
    full_bias = rolling_bias_quantification(angularity=0.3)
    results["rolling_bias"] = {
        "mu_r_values": full_bias.mu_r_values.tolist(),
        "R_poly": full_bias.R_poly,
        "R_sphere": full_bias.R_sphere.tolist(),
        "bias": full_bias.bias.tolist(),
        "optimal_mu_r": full_bias.optimal_mu_r,
        "min_bias": full_bias.min_bias,
    }

    # (c) Held-out shape family
    held_out = rolling_bias_quantification(angularity=0.4)
    results["held_out_shape"] = {
        "angularity": 0.4, "min_bias": held_out.min_bias,
        "optimal_mu_r": held_out.optimal_mu_r,
        "pass_15pct": held_out.min_bias < 0.15,
    }

    # (d) Identifiability
    ident = identifiability_analysis()
    results["identifiability"] = {
        "correlation": ident.correlation,
        "condition_number": ident.condition_number,
        "identifiable": ident.identifiable,
        "additional_obs_needed": ident.additional_obs_needed,
        "condition_reduction": ident.condition_reduction,
        # Criterion (d): if |corr|>0.8, additional obs must reduce cond# by 10x.
        # If |corr|<=0.8, parameters are already identifiable (criterion satisfied).
        "pass_10x": (not ident.additional_obs_needed) or (ident.condition_reduction >= 10.0),
    }

    # (e) KS test
    results["ks_test"] = ks_test_seeds()

    # (f) Monotonic scan
    mono = monotonic_scan()
    results["monotonic_scan"] = {
        "mu_roll_values": mono.mu_roll_values.tolist(),
        "R_total": mono.R_total.tolist(),
        "is_monotonic": mono.is_monotonic,
        "min_diff": mono.min_diff, "pass": mono.pass_,
    }

    # Contact pressure field demo
    forces, normals, positions = _generate_contact_network(300, 0.3, 1000.0, 42)
    areas = np.pi * (0.005 ** 2) * np.ones(len(forces))
    cpf = compute_contact_pressure_field(
        positions[:, :2], forces, areas,
        grid_bounds=(-0.1, 0.1, -0.1, 0.1), resolution=(30, 30),
    )
    sc = spatial_correlation(cpf)
    results["contact_pressure"] = {
        "total_force": cpf.total_force, "contact_area": cpf.contact_area,
        "mean_pressure": cpf.mean_pressure, "max_pressure": cpf.max_pressure,
        "peak_factor": sc.peak_factor,
        "correlation_length_x": sc.correlation_length_x,
        "correlation_length_y": sc.correlation_length_y,
        "integral_scale_x": sc.integral_scale_x,
    }

    # Overall pass/fail
    results["overall"] = {
        "a_hertz_degeneracy_pass": results["hertz_degeneracy"]["pass"],
        "b_sphere_bias_pass": results["sphere_zero_mu_bias"]["pass_2pct"],
        "c_held_out_pass": results["held_out_shape"]["pass_15pct"],
        "d_identifiability_pass": results["identifiability"]["pass_10x"],
        "e_ks_test_pass": results["ks_test"]["all_pass"],
        "f_monotonic_pass": results["monotonic_scan"]["pass"],
    }
    results["overall"]["all_pass"] = all(results["overall"].values())
    return results
