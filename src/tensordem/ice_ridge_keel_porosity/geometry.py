"""Keel geometry: inverted truncated pyramid container for rubble ice.

The keel of an ice ridge is modelled as an inverted truncated pyramid
(frustum) with:
- Top (waterline): rectangular opening L_top × W_top
- Bottom (deepest point): rectangular opening L_bot × W_bot
- Depth: D_keel

Typical values (Hopky 1983, Timco & Burden 1994):
- L_top ≈ 10–30 m, W_top ≈ 5–15 m
- D_keel ≈ 5–20 m
- L_bot ≈ 1–5 m (pointed or flat bottom)

The cross-section at depth z (measured downward from waterline) is:

    L(z) = L_top - (L_top - L_bot) * z / D_keel
    W(z) = W_top - (W_top - W_bot) * z / D_keel

This module provides the geometric description and volume computation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

__all__ = [
    "KeelGeometry",
    "keel_volume",
    "keel_containing_box",
    "cross_section_at_depth",
]


@dataclass(frozen=True)
class KeelGeometry:
    """Inverted truncated pyramid keel geometry.

    Attributes
    ----------
    L_top : float
        Top width in x-direction (m). Default 10 m.
    W_top : float
        Top width in y-direction (m). Default 10 m.
    L_bot : float
        Bottom width in x-direction (m). Default 2 m.
    W_bot : float
        Bottom width in y-direction (m). Default 2 m.
    depth : float
        Keel depth (m). Default 5 m.
    """
    L_top: float = 10.0
    W_top: float = 10.0
    L_bot: float = 2.0
    W_bot: float = 2.0
    depth: float = 5.0

    def __post_init__(self):
        for name in ("L_top", "W_top", "L_bot", "W_bot", "depth"):
            val = getattr(self, name)
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be finite and positive, got {val}")
        if self.L_bot > self.L_top:
            raise ValueError("L_bot must be <= L_top")
        if self.W_bot > self.W_top:
            raise ValueError("W_bot must be <= W_top")

    def cross_section(self, z: float) -> Tuple[float, float]:
        """Return (L, W) at depth z (0 = top, depth = bottom)."""
        if z < 0 or z > self.depth:
            raise ValueError(f"z={z} out of range [0, {self.depth}]")
        frac = z / self.depth
        L = self.L_top - (self.L_top - self.L_bot) * frac
        W = self.W_top - (self.W_top - self.W_bot) * frac
        return L, W

    def is_inside(self, x: float, y: float, z: float) -> bool:
        """Check if point (x, y, z) is inside the keel.

        Coordinates: origin at center of top face, z positive downward.
        """
        if z < 0 or z > self.depth:
            return False
        L, W = self.cross_section(z)
        return abs(x) <= L / 2 and abs(y) <= W / 2


def cross_section_at_depth(geom: KeelGeometry, z: float) -> Tuple[float, float]:
    """Functional form of KeelGeometry.cross_section."""
    return geom.cross_section(z)


def keel_volume(geom: KeelGeometry) -> float:
    """Volume of the inverted truncated pyramid.

    V = (depth / 3) * (A_top + A_bot + sqrt(A_top * A_bot))

    where A_top = L_top * W_top, A_bot = L_bot * W_bot.
    """
    A_top = geom.L_top * geom.W_top
    A_bot = geom.L_bot * geom.W_bot
    return (geom.depth / 3.0) * (A_top + A_bot + math.sqrt(A_top * A_bot))


def keel_containing_box(geom: KeelGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) bounding box of the keel.

    Origin at center of top face, z positive downward.
    """
    lo = np.array([-geom.L_top / 2, -geom.W_top / 2, 0.0])
    hi = np.array([geom.L_top / 2, geom.W_top / 2, geom.depth])
    return lo, hi


def random_point_in_keel(geom: KeelGeometry, rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly random point inside the keel.

    Uses rejection sampling from the bounding box.
    """
    lo, hi = keel_containing_box(geom)
    while True:
        pt = lo + rng.random(3) * (hi - lo)
        if geom.is_inside(pt[0], pt[1], pt[2]):
            return pt
