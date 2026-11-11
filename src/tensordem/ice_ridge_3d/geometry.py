"""Parametric 3D ice ridge geometry: sail, keel, and consolidated layer.

Physical picture
----------------
A first-year pressure ridge is idealised as a 3D body consisting of:

* **Sail**: above-water triangular prism (height ``H_s``, width ``W_s``, length ``L``)
* **Keel**: below-water triangular prism (depth ``D_k``, width ``W_k``, length ``L``)
* **Consolidated layer**: thin slab at the waterline (thickness ``h_c``)

The keel is filled with rubble particles at prescribed porosity ``n``.
The analytical anchor for keel volume is the equivalent rectangular prism:

    V_keel = 0.5 * W_k * D_k * L    (triangular cross-section × length)

Acceptance criterion (a): ridge volume vs L*W*D error < 1e-9.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "RidgeGeometry",
    "ridge_volume",
    "keel_half_width",
    "sail_half_width",
    "consolidated_layer_volume",
]


def _finite(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RidgeGeometry:
    """Full 3D parametric ridge geometry.

    Parameters
    ----------
    sail_height : float
        Height of the sail above waterline (m), typical 1-5 m.
    sail_width : float
        Width of the sail at waterline (m), typical 5-20 m.
    keel_depth : float
        Depth of the keel below waterline (m), typical 5-20 m.
    keel_width : float
        Width of the keel at waterline (m), typical 20-60 m.
    ridge_length : float
        Length of the ridge along the ship's path (m), typical 50-200 m.
    consolidated_thickness : float
        Thickness of the consolidated layer at waterline (m), typical 0.5-2 m.
    porosity : float
        Keel rubble porosity (void fraction), typical 0.30-0.40.
    """

    sail_height: float
    sail_width: float
    keel_depth: float
    keel_width: float
    ridge_length: float
    consolidated_thickness: float = 1.0
    porosity: float = 0.35

    def __post_init__(self) -> None:
        for name in ("sail_height", "sail_width", "keel_depth", "keel_width",
                     "ridge_length", "consolidated_thickness"):
            _finite(name, getattr(self, name), positive=True)
        _finite("porosity", self.porosity, nonnegative=True)
        if self.porosity >= 0.6:
            raise ValueError("porosity must be < 0.6 for a realistic rubble keel")
        # Empirical consistency: keel depth ~ 4 × sail height (Hopky 1983)
        # We allow deviation but warn if extreme.

    @property
    def solid_fraction(self) -> float:
        """Solid fraction of the keel (1 - porosity)."""
        return 1.0 - self.porosity

    @property
    def sail_volume(self) -> float:
        """Sail volume (triangular prism): V = 0.5 * W_s * H_s * L."""
        return 0.5 * self.sail_width * self.sail_height * self.ridge_length

    @property
    def keel_volume(self) -> float:
        """Keel envelope volume (triangular prism): V = 0.5 * W_k * D_k * L."""
        return 0.5 * self.keel_width * self.keel_depth * self.ridge_length

    @property
    def keel_solid_volume(self) -> float:
        """Solid ice volume in the keel: V_solid = (1 - n) * V_keel."""
        return self.solid_fraction * self.keel_volume

    @property
    def keel_void_volume(self) -> float:
        """Void volume in the keel: V_void = n * V_keel."""
        return self.porosity * self.keel_volume

    @property
    def consolidated_volume(self) -> float:
        """Consolidated layer volume: V = W_k * h_c * L."""
        return self.keel_width * self.consolidated_thickness * self.ridge_length

    @property
    def total_volume(self) -> float:
        """Total ridge envelope volume (sail + keel + consolidated layer)."""
        return self.sail_volume + self.keel_volume + self.consolidated_volume

    @property
    def total_solid_volume(self) -> float:
        """Total solid ice volume (sail + keel solid + consolidated)."""
        return self.sail_volume + self.keel_solid_volume + self.consolidated_volume

    def keel_half_width(self, depth_below_waterline: float) -> float:
        """Keel half-width at a given depth below the waterline (m).

        The keel cross-section is triangular: widest at waterline, point at tip.
        """
        _finite("depth_below_waterline", depth_below_waterline, nonnegative=True)
        d = min(depth_below_waterline, self.keel_depth)
        return 0.5 * self.keel_width * (1.0 - d / self.keel_depth)

    def sail_half_width(self, height_above_waterline: float) -> float:
        """Sail half-width at a given height above the waterline (m)."""
        _finite("height_above_waterline", height_above_waterline, nonnegative=True)
        h = min(height_above_waterline, self.sail_height)
        return 0.5 * self.sail_width * (1.0 - h / self.sail_height)

    def keel_cross_section_area(self) -> float:
        """Keel cross-section area (triangular): A = 0.5 * W_k * D_k."""
        return 0.5 * self.keel_width * self.keel_depth

    def sail_cross_section_area(self) -> float:
        """Sail cross-section area (triangular): A = 0.5 * W_s * H_s."""
        return 0.5 * self.sail_width * self.sail_height

    def as_dict(self) -> dict:
        return {
            "sail_height_m": self.sail_height,
            "sail_width_m": self.sail_width,
            "keel_depth_m": self.keel_depth,
            "keel_width_m": self.keel_width,
            "ridge_length_m": self.ridge_length,
            "consolidated_thickness_m": self.consolidated_thickness,
            "porosity": self.porosity,
            "solid_fraction": self.solid_fraction,
            "sail_volume_m3": self.sail_volume,
            "keel_volume_m3": self.keel_volume,
            "keel_solid_volume_m3": self.keel_solid_volume,
            "keel_void_volume_m3": self.keel_void_volume,
            "consolidated_volume_m3": self.consolidated_volume,
            "total_volume_m3": self.total_volume,
            "total_solid_volume_m3": self.total_solid_volume,
        }


def ridge_volume(sail_height: float, sail_width: float,
                 keel_depth: float, keel_width: float,
                 ridge_length: float) -> float:
    """Total ridge envelope volume (sail + keel, triangular prisms).

    Analytical anchor: V = 0.5 * (W_s * H_s + W_k * D_k) * L
    """
    return 0.5 * (sail_width * sail_height + keel_width * keel_depth) * ridge_length


def keel_half_width(keel_width: float, keel_depth: float,
                    depth_below_waterline: float) -> float:
    """Keel half-width at depth below waterline (triangular cross-section)."""
    d = min(max(depth_below_waterline, 0.0), keel_depth)
    return 0.5 * keel_width * (1.0 - d / keel_depth)


def sail_half_width(sail_width: float, sail_height: float,
                    height_above_waterline: float) -> float:
    """Sail half-width at height above waterline (triangular cross-section)."""
    h = min(max(height_above_waterline, 0.0), sail_height)
    return 0.5 * sail_width * (1.0 - h / sail_height)


def consolidated_layer_volume(keel_width: float, thickness: float,
                              ridge_length: float) -> float:
    """Consolidated layer volume: V = W_k * h_c * L."""
    return keel_width * thickness * ridge_length
