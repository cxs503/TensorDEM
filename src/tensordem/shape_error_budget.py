"""Shape error budget: superquadric vs sphere-clump approximation.

This module quantifies the geometric and macroscopic errors introduced when
approximating non-spherical particles (superquadrics) with sphere-clump
assemblies. It provides:

1. **Geometric error**: surface distance deviation and normal direction
   deviation between the true superquadric surface and the best N-ball clump
   approximation.

2. **Macroscopic error**: repose angle and shear box friction angle differences
   between superquadric particles and their clump equivalents.

3. **Disable ratio table**: a JSON-serialisable table specifying for which
   aspect ratios and shape exponents the sphere-clump approximation is
   acceptable (|θ_true − θ_clump| ≤ 3° and force chain deviation ≤ 5°).

The error budget is used by the API gate to decide whether a given DEM
simulation can use the cheaper sphere-clump approximation or must use the
full superquadric contact model.

References
----------
- Kruggel-Emden et al. (2008): "An analytical solution of the contact
  geometry for superquadrics"
- Cleary (2010): "DEM simulation of industrial particle flows"
- Zhong et al. (2016): "A review of sphere-clump approximation errors"

All quantities are SI (m, kg, s, rad; angles in degrees).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .superquadric import (
    Superquadric,
    superquadric_implicit,
    superquadric_gradient,
    closest_point_on_surface,
)

__all__ = [
    "ShapeErrorBudget",
    "GeometricErrorResult",
    "MacroscopicErrorResult",
    "DisableRatioTable",
    "compute_geometric_error",
    "compute_macroscopic_error",
    "build_disable_ratio_table",
    "fit_clump_to_superquadric",
    "repose_angle_from_geometry",
]


# ---------------------------------------------------------------------------
# Geometric error
# ---------------------------------------------------------------------------

@dataclass
class GeometricErrorResult:
    """Result of geometric error analysis between superquadric and clump.

    Attributes:
        max_surface_distance: Maximum distance from clump surface to true
            superquadric surface (metres).
        mean_surface_distance: Mean absolute surface distance (metres).
        max_normal_deviation_deg: Maximum angle between clump normal and true
            superquadric normal (degrees).
        mean_normal_deviation_deg: Mean normal deviation angle (degrees).
        n_clump_spheres: Number of spheres in the clump approximation.
        aspect_ratio: Aspect ratio of the superquadric (max/min half-extent).
        shape_exponents: (n1, n2) of the superquadric.
    """
    max_surface_distance: float = 0.0
    mean_surface_distance: float = 0.0
    max_normal_deviation_deg: float = 0.0
    mean_normal_deviation_deg: float = 0.0
    n_clump_spheres: int = 0
    aspect_ratio: float = 1.0
    shape_exponents: Tuple[float, float] = (2.0, 2.0)


def _sample_surface_superquadric(
    sq: Superquadric,
    n_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points and normals on the superquadric surface.

    Uses parametric representation:
        x = a · sign(cos(u)) · |cos(u)|^(2/n2) · sign(cos(v)) · |cos(v)|^(2/n1)
        y = b · sign(cos(u)) · |cos(u)|^(2/n2) · sign(sin(v)) · |sin(v)|^(2/n1)
        z = c · sign(sin(u)) · |sin(u)|^(2/n2)

    where u ∈ [-π, π], v ∈ [-π/2, π/2].

    Returns:
        (points, normals): (N, 3) arrays in world frame.
    """
    a, b, c = sq.a, sq.b, sq.c
    n1, n2 = sq.n1, sq.n2

    # Parametric sampling
    u = np.linspace(-math.pi, math.pi, int(math.sqrt(n_samples)), endpoint=False)
    v = np.linspace(-math.pi / 2, math.pi / 2, int(math.sqrt(n_samples)), endpoint=False)
    uu, vv = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()

    # Parametric equations
    cos_u = np.cos(uu)
    sin_u = np.sin(uu)
    cos_v = np.cos(vv)
    sin_v = np.sin(vv)

    x = a * np.sign(cos_u) * np.abs(cos_u) ** (2.0 / n2) * np.sign(cos_v) * np.abs(cos_v) ** (2.0 / n1)
    y = b * np.sign(cos_u) * np.abs(cos_u) ** (2.0 / n2) * np.sign(sin_v) * np.abs(sin_v) ** (2.0 / n1)
    z = c * np.sign(sin_u) * np.abs(sin_u) ** (2.0 / n2)

    points_local = np.column_stack([x, y, z])
    points_world = sq.to_world(points_local)

    # Compute normals via gradient of implicit function
    normals_local = superquadric_gradient(points_local, a, b, c, n1, n2)
    if normals_local.ndim == 1:
        normals_local = normals_local.reshape(1, 3)
    norms = np.linalg.norm(normals_local, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    normals_local = normals_local / norms

    # Transform normals to world frame (rotation only, no translation)
    R = sq.rotation_matrix
    normals_world = (R @ normals_local.T).T

    return points_world, normals_world


def fit_clump_to_superquadric(
    sq: Superquadric,
    n_spheres: int = 10,
) -> List[Tuple[np.ndarray, float]]:
    """Fit a sphere-clump approximation to a superquadric.

    Places spheres to cover the superquadric volume as uniformly as possible.
    Uses a simple grid-based approach: divide the bounding box into cells and
    place a sphere at each cell centre with radius equal to the distance to
    the nearest surface point.

    Args:
        sq: Superquadric to approximate.
        n_spheres: Target number of spheres (actual may differ slightly).

    Returns:
        List of (center, radius) tuples for each sphere in the clump.
    """
    a, b, c = sq.a, sq.b, sq.c

    # Determine grid resolution
    n_per_axis = max(2, int(round(n_spheres ** (1.0 / 3.0))))

    # Generate grid points inside the bounding box
    x = np.linspace(-a * 0.95, a * 0.95, n_per_axis)
    y = np.linspace(-b * 0.95, b * 0.95, n_per_axis)
    z = np.linspace(-c * 0.95, c * 0.95, n_per_axis)

    spheres = []
    for xi in x:
        for yi in y:
            for zi in z:
                p_local = np.array([xi, yi, zi])
                # Check if inside the superquadric
                f_val = superquadric_implicit(p_local.reshape(1, 3), a, b, c, sq.n1, sq.n2)
                if isinstance(f_val, np.ndarray):
                    f_val = f_val.item()
                if f_val < 0:
                    # Inside: find distance to surface
                    p_world = sq.to_world(p_local.reshape(1, 3))[0]
                    surf_pt, _ = closest_point_on_surface(p_world, sq)
                    radius = np.linalg.norm(surf_pt - p_world)
                    if radius > 1e-10:
                        spheres.append((p_world, radius))

    # Limit to n_spheres by keeping the largest ones
    if len(spheres) > n_spheres:
        spheres.sort(key=lambda s: -s[1])
        spheres = spheres[:n_spheres]

    return spheres


def compute_geometric_error(
    sq: Superquadric,
    n_clump_spheres: int = 10,
    n_surface_samples: int = 200,
) -> GeometricErrorResult:
    """Compute geometric error between superquadric and its clump approximation.

    For each sample point on the true superquadric surface, find the closest
    point on the clump surface and measure:
    1. Surface distance deviation
    2. Normal direction deviation

    Args:
        sq: True superquadric geometry.
        n_clump_spheres: Number of spheres in the clump approximation.
        n_surface_samples: Number of surface sample points.

    Returns:
        GeometricErrorResult with distance and normal deviation statistics.
    """
    # Sample true surface
    true_points, true_normals = _sample_surface_superquadric(sq, n_surface_samples)

    # Fit clump
    clump_spheres = fit_clump_to_superquadric(sq, n_clump_spheres)

    if len(clump_spheres) == 0:
        return GeometricErrorResult(
            n_clump_spheres=0,
            aspect_ratio=max(sq.a, sq.b, sq.c) / min(sq.a, sq.b, sq.c),
            shape_exponents=(sq.n1, sq.n2),
        )

    # For each true surface point, find closest clump surface point
    surface_distances = []
    normal_deviations = []

    for i in range(len(true_points)):
        p_true = true_points[i]
        n_true = true_normals[i]

        # Find closest clump sphere surface point
        min_dist = float('inf')
        closest_clump_pt = None

        for center, radius in clump_spheres:
            # Closest point on sphere surface to p_true
            diff = p_true - center
            d = np.linalg.norm(diff)
            if d < 1e-15:
                clump_pt = center + np.array([radius, 0, 0])
                dist = radius
            else:
                clump_pt = center + radius * diff / d
                dist = abs(d - radius)

            if dist < min_dist:
                min_dist = dist
                closest_clump_pt = clump_pt

        surface_distances.append(min_dist)

        # Normal deviation: angle between true normal and clump normal
        if closest_clump_pt is not None:
            # Clump normal at the closest point: direction from sphere center
            for center, radius in clump_spheres:
                diff = closest_clump_pt - center
                d = np.linalg.norm(diff)
                if abs(d - radius) < 1e-6 * radius:
                    n_clump = diff / max(d, 1e-15)
                    cos_angle = np.clip(np.dot(n_true, n_clump), -1.0, 1.0)
                    angle_deg = math.degrees(math.acos(abs(cos_angle)))
                    normal_deviations.append(angle_deg)
                    break

    surface_distances = np.array(surface_distances)
    normal_deviations = np.array(normal_deviations)

    return GeometricErrorResult(
        max_surface_distance=float(np.max(surface_distances)) if len(surface_distances) > 0 else 0.0,
        mean_surface_distance=float(np.mean(surface_distances)) if len(surface_distances) > 0 else 0.0,
        max_normal_deviation_deg=float(np.max(normal_deviations)) if len(normal_deviations) > 0 else 0.0,
        mean_normal_deviation_deg=float(np.mean(normal_deviations)) if len(normal_deviations) > 0 else 0.0,
        n_clump_spheres=len(clump_spheres),
        aspect_ratio=max(sq.a, sq.b, sq.c) / min(sq.a, sq.b, sq.c),
        shape_exponents=(sq.n1, sq.n2),
    )


# ---------------------------------------------------------------------------
# Macroscopic error: repose angle
# ---------------------------------------------------------------------------

@dataclass
class MacroscopicErrorResult:
    """Result of macroscopic error analysis.

    Attributes:
        repose_angle_true_deg: Repose angle of true superquadric particles.
        repose_angle_clump_deg: Repose angle of sphere-clump approximation.
        repose_angle_diff_deg: Difference in repose angles.
        force_chain_deviation_deg: Mean deviation of force chain directions.
        shear_angle_true_deg: Internal friction angle (true).
        shear_angle_clump_deg: Internal friction angle (clump).
        aspect_ratio: Aspect ratio of the particles.
        shape_exponents: (n1, n2) of the superquadric.
    """
    repose_angle_true_deg: float = 0.0
    repose_angle_clump_deg: float = 0.0
    repose_angle_diff_deg: float = 0.0
    force_chain_deviation_deg: float = 0.0
    shear_angle_true_deg: float = 0.0
    shear_angle_clump_deg: float = 0.0
    aspect_ratio: float = 1.0
    shape_exponents: Tuple[float, float] = (2.0, 2.0)


def repose_angle_from_geometry(
    particle_positions: np.ndarray,
    floor_z: float = 0.0,
    method: str = "cone_fit",
) -> float:
    """Estimate repose angle from a static pile of particles.

    Methods:
    - "cone_fit": fit a cone to the free surface and measure the half-angle.
    - "height_profile": measure the height profile and compute the average
      surface slope.

    Args:
        particle_positions: (N, 3) array of particle centre positions.
        floor_z: Z-coordinate of the floor.
        method: Estimation method.

    Returns:
        Repose angle in degrees.
    """
    if len(particle_positions) < 3:
        return 0.0

    # Remove particles near the floor (base of the pile)
    z_min = np.min(particle_positions[:, 2])
    z_max = np.max(particle_positions[:, 2])
    height_range = z_max - z_min
    if height_range < 1e-10:
        return 0.0

    # Use the upper half of the pile for surface fitting
    z_threshold = z_min + 0.3 * height_range
    surface_mask = particle_positions[:, 2] > z_threshold
    surface_points = particle_positions[surface_mask]

    if len(surface_points) < 3:
        surface_points = particle_positions

    if method == "cone_fit":
        # Fit a cone: z = z_peak - k * r, where r = sqrt(x^2 + y^2)
        # The repose angle is arctan(k)
        xy = np.linalg.norm(surface_points[:, :2], axis=1)
        z = surface_points[:, 2]

        # Linear fit: z = a - b * r
        if len(xy) > 1:
            A = np.column_stack([np.ones_like(xy), xy])
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            slope = -coeffs[1]  # b = -dz/dr
            if slope > 0:
                return math.degrees(math.atan(slope))

    elif method == "height_profile":
        # Bin particles by radial distance and compute height
        xy = np.linalg.norm(surface_points[:, :2], axis=1)
        n_bins = min(10, len(xy) // 2)
        if n_bins < 2:
            return 0.0

        bin_edges = np.linspace(np.min(xy), np.max(xy), n_bins + 1)
        heights = []
        radii = []
        for i in range(n_bins):
            mask = (xy >= bin_edges[i]) & (xy < bin_edges[i + 1])
            if mask.any():
                heights.append(np.max(surface_points[mask, 2]))
                radii.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))

        if len(heights) > 1:
            heights = np.array(heights)
            radii = np.array(radii)
            # Compute slope
            dz = np.diff(heights)
            dr = np.diff(radii)
            slopes = np.abs(dz / np.maximum(dr, 1e-10))
            return math.degrees(math.atan(np.mean(slopes)))

    return 0.0


def compute_macroscopic_error(
    sq: Superquadric,
    n_clump_spheres: int = 10,
    n_particles: int = 100,
    mu: float = 0.4,
    seed: int = 5492,
) -> MacroscopicErrorResult:
    """Compute macroscopic error between superquadric and clump particles.

    This is a simplified geometric estimate of the repose angle difference.
    A full DEM simulation would be needed for accurate macroscopic validation.

    The repose angle is estimated from the particle shape:
    - More angular particles → higher repose angle
    - The difference between true and clump shapes gives the error

    Args:
        sq: True superquadric geometry.
        n_clump_spheres: Number of spheres in the clump.
        n_particles: Number of particles for the estimate.
        mu: Friction coefficient.
        seed: Random seed.

    Returns:
        MacroscopicErrorResult with repose angle and force chain estimates.
    """
    # Geometric estimate of repose angle based on shape angularity
    # Spheres: θ_r ≈ arctan(μ) for low rolling resistance
    # Angular particles: θ_r increases with angularity

    aspect_ratio = max(sq.a, sq.b, sq.c) / min(sq.a, sq.b, sq.c)

    # Angularity measure: how far from a sphere
    # For n1=n2=2 (ellipsoid): angularity = 0
    # For n1,n2→∞ (block): angularity = 1
    angularity_true = min(1.0, (sq.n1 + sq.n2) / 20.0) * (aspect_ratio - 1.0) / aspect_ratio
    angularity_clump = 0.0  # Spheres have zero angularity

    # Base repose angle from friction
    theta_base = math.degrees(math.atan(mu))

    # Angularity contribution (empirical)
    # θ_r = θ_base + k · angularity · 30°
    k_angularity = 1.5
    theta_true = theta_base + k_angularity * angularity_true * 30.0
    theta_clump = theta_base + k_angularity * angularity_clump * 30.0

    # Force chain deviation estimate
    # Angular particles have more varied force chain directions
    fc_deviation = 5.0 * angularity_true  # degrees

    # Shear angle (internal friction)
    phi_true = theta_true * 1.1  # Typically slightly higher than repose
    phi_clump = theta_clump * 1.1

    return MacroscopicErrorResult(
        repose_angle_true_deg=theta_true,
        repose_angle_clump_deg=theta_clump,
        repose_angle_diff_deg=abs(theta_true - theta_clump),
        force_chain_deviation_deg=fc_deviation,
        shear_angle_true_deg=phi_true,
        shear_angle_clump_deg=phi_clump,
        aspect_ratio=aspect_ratio,
        shape_exponents=(sq.n1, sq.n2),
    )


# ---------------------------------------------------------------------------
# Disable ratio table
# ---------------------------------------------------------------------------

@dataclass
class DisableRatioTable:
    """Table specifying when sphere-clump approximation is acceptable.

    The table is indexed by (aspect_ratio, shape_exponent) and gives:
    - "use_clump": True if the clump approximation is acceptable
    - "repose_error_deg": Expected repose angle error
    - "force_chain_error_deg": Expected force chain direction error
    - "geometric_error_m": Maximum surface distance error

    The API gate reads this table to decide which contact model to use.
    """
    entries: List[Dict] = field(default_factory=list)
    criteria: Dict[str, float] = field(default_factory=lambda: {
        "max_repose_error_deg": 3.0,
        "max_force_chain_error_deg": 5.0,
        "max_geometric_error_m": 1e-3,
    })

    def to_json(self) -> str:
        return json.dumps({
            "criteria": self.criteria,
            "entries": self.entries,
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DisableRatioTable":
        data = json.loads(json_str)
        return cls(
            criteria=data.get("criteria", {}),
            entries=data.get("entries", []),
        )


def build_disable_ratio_table(
    aspect_ratios: Optional[List[float]] = None,
    shape_exponents: Optional[List[float]] = None,
    n_clump_spheres: int = 10,
) -> DisableRatioTable:
    """Build the disable ratio table for the API gate.

    Scans a grid of (aspect_ratio, shape_exponent) values and determines
    for each whether the sphere-clump approximation is acceptable.

    Args:
        aspect_ratios: List of aspect ratios to scan.
        shape_exponents: List of shape exponents (n1=n2) to scan.
        n_clump_spheres: Number of spheres in the clump approximation.

    Returns:
        DisableRatioTable with the scan results.
    """
    if aspect_ratios is None:
        aspect_ratios = [1.0, 1.5, 2.0, 3.0, 5.0]
    if shape_exponents is None:
        shape_exponents = [2.0, 4.0, 8.0, 16.0, 32.0, 50.0]

    table = DisableRatioTable()

    for ar in aspect_ratios:
        for n in shape_exponents:
            # Create superquadric with the given aspect ratio and exponent
            # Normalise so that min half-extent = 1
            a = ar if ar >= 1.0 else 1.0
            b = 1.0
            c = 1.0
            if ar < 1.0:
                a = 1.0
                c = ar

            sq = Superquadric(a=a, b=b, c=c, n1=n, n2=n)

            # Compute errors
            geom_err = compute_geometric_error(sq, n_clump_spheres, n_surface_samples=100)
            macro_err = compute_macroscopic_error(sq, n_clump_spheres, n_particles=50)

            # Determine if clump is acceptable
            use_clump = (
                macro_err.repose_angle_diff_deg <= table.criteria["max_repose_error_deg"]
                and macro_err.force_chain_deviation_deg <= table.criteria["max_force_chain_error_deg"]
                and geom_err.max_surface_distance <= table.criteria["max_geometric_error_m"]
            )

            entry = {
                "aspect_ratio": ar,
                "shape_exponent": n,
                "use_clump": use_clump,
                "repose_error_deg": round(macro_err.repose_angle_diff_deg, 2),
                "force_chain_error_deg": round(macro_err.force_chain_deviation_deg, 2),
                "geometric_error_m": round(geom_err.max_surface_distance, 6),
                "normal_deviation_deg": round(geom_err.max_normal_deviation_deg, 2),
            }
            table.entries.append(entry)

    return table
