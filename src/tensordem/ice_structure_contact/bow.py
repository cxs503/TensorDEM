"""Rigid wedge-bow idealisation for the longitudinal ice-structure slice.

Geometry (ice reference frame, waterline at ``z = 0``, ship advancing in
``+x``)::

        z
        ^    free ice edge            ship hull (solid) above the rake surface
        |    |<---- broken ice ---->|
    0   |----+====================+-------------------  waterline / ice top
        |     \                   /
   -h   |      \                 /   rake surface  z = -min((x_s - x) tan b, T)
        |       \_______________/
        +--------------------------------------------> x
              x_s = stem position (advances with speed V)

The bow underside descends aft of the stem with rake angle ``b`` down to the
draught ``T``; the hull solid is *above* that surface, so an ice particle that
the rake sweeps over is pushed **down and forward** — exactly the icebreaking
mechanism (the ship rides up, the sheet is bent down and fails).

Only the local rake (length ``h / tan b``) ever touches the sheet, which is why
Lindqvist's crushing term is stem-localised.

The hull is rigid: contact geometry is analytic (one plane + one horizontal
bottom), so no hull mesh is needed and the contact is exactly reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["WedgeBow"]


@dataclass(frozen=True)
class WedgeBow:
    """Straight-line wedge bow of a longitudinal slice (rigid, prescribed motion)."""

    rake_angle_deg: float = 22.0
    draught_m: float = 7.0
    speed_m_s: float = 1.0289  # 2 knots
    initial_stem_x_m: float = 0.0

    @property
    def rake_angle_rad(self) -> float:
        return math.radians(self.rake_angle_deg)

    @property
    def normal(self) -> tuple[float, float]:
        """Outward unit normal of the rake surface pointing into the ice.

        The surface is ``z = -(x_s - x) tan b`` (hull above), so the outward
        normal on the ice side is ``(sin b, -cos b)``: forward and downward.
        """
        b = self.rake_angle_rad
        return (math.sin(b), -math.cos(b))

    @property
    def tangent(self) -> tuple[float, float]:
        """Unit tangent along the rake face pointing forward and upward."""
        b = self.rake_angle_rad
        return (math.cos(b), math.sin(b))

    def hull_surface_z(self, x: np.ndarray | float, stem_x: float) -> np.ndarray:
        """Hull surface height at ``x`` for a stem at ``stem_x`` (hull above)."""
        b = self.rake_angle_rad
        x_arr = np.asarray(x, dtype=float)
        drop = np.minimum((stem_x - x_arr) * math.tan(b), self.draught_m)
        return -drop

    def penetration(
        self, x: np.ndarray, z: np.ndarray, radius: float, stem_x: float
    ) -> np.ndarray:
        """Perpendicular penetration of a particle circle into the rake surface.

        The rake plane passes through the stem at the waterline:
        ``z = -(stem_x - x) tan b``.  Its outward normal (into the ice) is
        ``(sin b, -cos b)``, so the signed perpendicular distance of a centre
        *inside* the hull is ``z cos b + (stem_x - x) sin b`` and the
        penetration is that plus the particle radius.  Contact exists only on
        the rake (able to reach the ice, ``(stem_x - x) tan b < draught``) and
        only aft of the stem; ahead of the stem the ice is free.
        """
        b = self.rake_angle_rad
        x_arr = np.asarray(x, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        if isinstance(radius, np.ndarray):
            r = radius
        else:
            r = np.full_like(x_arr, float(radius))
        drop = (stem_x - x_arr) * math.tan(b)
        d_perp = z_arr * math.cos(b) + (stem_x - x_arr) * math.sin(b)
        delta = d_perp + r
        on_rake = (drop < self.draught_m) & (x_arr <= stem_x)
        delta = np.where(on_rake, delta, 0.0)
        return np.maximum(delta, 0.0)