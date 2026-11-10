"""Named polyhedra for DEM particle shape studies.

Provides canonical polyhedra (dodecahedron, cube, tetrahedron, octahedron,
icosahedron, plate-like, columnar) as vertex clouds, plus shape descriptors
(sphericity, angularity, aspect ratio) for parametric studies.

All shapes are centred at the origin with unit characteristic size (edge length
or diameter), and can be scaled to any target volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "PolyhedronSpec",
    "ShapeDescriptor",
    "make_dodecahedron",
    "make_cube",
    "make_tetrahedron",
    "make_octahedron",
    "make_icosahedron",
    "make_plate",
    "make_column",
    "make_prism",
    "make_polyhedron_by_name",
    "scale_to_volume",
    "compute_shape_descriptor",
    "SHAPE_NAMES",
]


@dataclass(frozen=True)
class PolyhedronSpec:
    """A named polyhedron specification."""
    name: str
    vertices: np.ndarray  # (V, 3) body-frame vertices
    volume: float  # raw volume of the unscaled shape
    surface_area: float  # raw surface area of the unscaled shape


@dataclass(frozen=True)
class ShapeDescriptor:
    """Geometric shape descriptors for a polyhedron."""
    name: str
    volume: float
    surface_area: float
    sphericity: float  # Ψ = (π^(1/3) (6V)^(2/3)) / A  (1.0 for sphere)
    angularity: float  # 1 - sphericity
    aspect_ratio: float  # max extent / min extent
    circumradius: float  # bounding sphere radius
    equivalent_radius: float  # radius of equal-volume sphere
    n_vertices: int
    n_faces: int
    n_edges: int


def _hull_properties(vertices: np.ndarray) -> tuple[float, float, int, int, int]:
    """Compute volume, surface area, and counts from convex hull."""
    hull = ConvexHull(vertices)
    vol = float(hull.volume)
    area = float(hull.area)
    n_v = len(hull.vertices)
    n_f = len(hull.simplices)
    # Euler: V - E + F = 2  =>  E = V + F - 2
    n_e = n_v + n_f - 2
    return vol, area, n_v, n_f, n_e


def make_dodecahedron() -> PolyhedronSpec:
    """Regular dodecahedron (12 pentagonal faces, 20 vertices)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    verts = []
    # (±1, ±1, ±1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append([sx, sy, sz])
    # (0, ±1/φ, ±φ)
    for sy in (-1, 1):
        for sz in (-1, 1):
            verts.append([0.0, sy / phi, sz * phi])
    # (±1/φ, ±φ, 0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            verts.append([sx / phi, sy * phi, 0.0])
    # (±φ, 0, ±1/φ)
    for sx in (-1, 1):
        for sz in (-1, 1):
            verts.append([sx * phi, 0.0, sz / phi])
    verts = np.array(verts, dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec("dodecahedron", verts, vol, area)


def make_cube() -> PolyhedronSpec:
    """Regular cube (6 faces, 8 vertices)."""
    verts = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
    ], dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec("cube", verts, vol, area)


def make_tetrahedron() -> PolyhedronSpec:
    """Regular tetrahedron (4 faces, 4 vertices)."""
    verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec("tetrahedron", verts, vol, area)


def make_octahedron() -> PolyhedronSpec:
    """Regular octahedron (8 faces, 6 vertices)."""
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec("octahedron", verts, vol, area)


def make_icosahedron() -> PolyhedronSpec:
    """Regular icosahedron (20 faces, 12 vertices)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sy in (-1, 1):
        for sz in (-1, 1):
            verts.append([0.0, sy, sz * phi])
    for sx in (-1, 1):
        for sz in (-1, 1):
            verts.append([sx, sy * phi, 0.0])
    for sx in (-1, 1):
        for sy in (-1, 1):
            verts.append([sx * phi, 0.0, sy])
    verts = np.array(verts, dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec("icosahedron", verts, vol, area)


def make_plate(aspect: float = 0.25) -> PolyhedronSpec:
    """Plate-like (oblate) polyhedron: flattened icosahedron.

    Parameters
    ----------
    aspect : float
        Ratio of thickness to diameter (z/x). Default 0.25 gives a thin plate.
    """
    base = make_icosahedron()
    verts = base.vertices.copy()
    verts[:, 2] *= aspect
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec(f"plate_{aspect:.2f}", verts, vol, area)


def make_column(aspect: float = 3.0) -> PolyhedronSpec:
    """Columnar (prolate) polyhedron: elongated hexagonal prism.

    Parameters
    ----------
    aspect : float
        Ratio of height to diameter (z/x). Default 3.0 gives a tall column.
    """
    n_sides = 6
    verts = []
    for k in range(n_sides):
        theta = 2.0 * math.pi * k / n_sides
        x, y = math.cos(theta), math.sin(theta)
        verts.append([x, y, -aspect / 2.0])
        verts.append([x, y, aspect / 2.0])
    # Add cap centres for convexity
    verts.append([0.0, 0.0, -aspect / 2.0])
    verts.append([0.0, 0.0, aspect / 2.0])
    verts = np.array(verts, dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec(f"column_{aspect:.1f}", verts, vol, area)


def make_prism(n_sides: int = 6, aspect: float = 1.5) -> PolyhedronSpec:
    """Regular n-gonal prism.

    Parameters
    ----------
    n_sides : int
        Number of sides of the prism cross-section.
    aspect : float
        Height / diameter ratio.
    """
    verts = []
    for k in range(n_sides):
        theta = 2.0 * math.pi * k / n_sides
        x, y = math.cos(theta), math.sin(theta)
        verts.append([x, y, -aspect / 2.0])
        verts.append([x, y, aspect / 2.0])
    verts.append([0.0, 0.0, -aspect / 2.0])
    verts.append([0.0, 0.0, aspect / 2.0])
    verts = np.array(verts, dtype=np.float64)
    vol, area, _, _, _ = _hull_properties(verts)
    return PolyhedronSpec(f"prism{n_sides}_{aspect:.1f}", verts, vol, area)


def scale_to_volume(vertices: np.ndarray, target_volume: float) -> np.ndarray:
    """Scale vertices to achieve exact target volume."""
    hull = ConvexHull(vertices)
    raw_vol = float(hull.volume)
    if raw_vol <= 0:
        raise ValueError("degenerate polyhedron (zero or negative volume)")
    scale = (target_volume / raw_vol) ** (1.0 / 3.0)
    return vertices * scale


def compute_shape_descriptor(
    name: str,
    vertices: np.ndarray,
    volume: Optional[float] = None,
    surface_area: Optional[float] = None,
) -> ShapeDescriptor:
    """Compute full shape descriptor for a polyhedron."""
    hull = ConvexHull(vertices)
    if volume is None:
        volume = float(hull.volume)
    if surface_area is None:
        surface_area = float(hull.area)

    # Sphericity: Ψ = π^(1/3) * (6V)^(2/3) / A
    sphericity = (math.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / surface_area
    angularity = 1.0 - sphericity

    # Aspect ratio from principal extents
    centroid = vertices.mean(axis=0)
    centred = vertices - centroid
    extents = np.ptp(centred, axis=0)
    aspect_ratio = float(extents.max() / max(extents.min(), 1e-30))

    circumradius = float(np.max(np.linalg.norm(centred, axis=1)))
    equivalent_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    n_v = len(hull.vertices)
    n_f = len(hull.simplices)
    n_e = n_v + n_f - 2

    return ShapeDescriptor(
        name=name,
        volume=volume,
        surface_area=surface_area,
        sphericity=sphericity,
        angularity=angularity,
        aspect_ratio=aspect_ratio,
        circumradius=circumradius,
        equivalent_radius=equivalent_radius,
        n_vertices=n_v,
        n_faces=n_f,
        n_edges=n_e,
    )


# Registry of named shapes
SHAPE_NAMES = {
    "dodecahedron": make_dodecahedron,
    "cube": make_cube,
    "tetrahedron": make_tetrahedron,
    "octahedron": make_octahedron,
    "icosahedron": make_icosahedron,
    "plate": lambda: make_plate(0.25),
    "column": lambda: make_column(3.0),
}


def make_polyhedron_by_name(name: str) -> PolyhedronSpec:
    """Create a named polyhedron from the registry."""
    if name not in SHAPE_NAMES:
        raise ValueError(f"Unknown shape '{name}'. Available: {list(SHAPE_NAMES.keys())}")
    return SHAPE_NAMES[name]()
