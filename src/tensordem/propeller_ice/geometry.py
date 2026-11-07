"""Parametric propeller and ice-block geometry for propeller-ice milling.

The milling model in :mod:`tensordem.propeller_ice.milling` consumes two
geometry objects:

``PropellerGeometry``
    A radially parametrised propeller: chord, pitch, skew, thickness and the
    Prandtl hub/tip loss factors on an arbitrary number of radial strips.  No
    proprietary section tables are used; the radial laws are the documented
    quadratics below, anchored so that the three anchor values (hub, 0.7R, tip)
    are reproduced exactly.

``IceBlockGeometry``
    The ice body the propeller mills: an annular sector of a level ice sheet of
    thickness ``thickness``, spanning radii
    ``[r_inner_frac, r_outer_frac] * R`` and a fraction ``block_fraction`` of the
    azimuth circle.  ``block_fraction`` is the *blocked-azimuth* occupancy of the
    disc used in propeller-ice interaction work (the "ice-blocked sector"); the
    blocked disc area follows from :meth:`IceBlockGeometry.blocked_area`.

Both classes are frozen and side-effect free: the same object can be handed to
the geometry verifier, the open-water solver and the milling simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["PropellerGeometry", "IceBlockGeometry"]


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class PropellerGeometry:
    """Radially parametrised propeller geometry (model-scale ice-class baseline).

    Parameters
    ----------
    diameter
        Tip-to-tip diameter ``D`` [m].
    n_blades
        Blade number ``Z``.
    hub_ratio
        Hub diameter over propeller diameter, ``d_h / D``.
    pitch_ratio
        Constant pitch ratio ``P / D``; the pitch angle is ``atan(P / (2 pi r))``.
    chord_ratio
        Chord at ``0.7 R`` divided by ``D``.
    thickness_ratio
        Section thickness/chord at ``0.7 R``.
    skew_tip_deg
        Total skew at the tip [deg]; the skew law is linear in ``r``.
    chord_root_factor, chord_tip_factor
        ``c(r_hub) / c(0.7R)`` and ``c(R) / c(0.7R)``; with ``chord_ratio`` these
        fix the quadratic chord law.
    """

    diameter: float = 0.60
    n_blades: int = 4
    hub_ratio: float = 0.35
    pitch_ratio: float = 1.00
    chord_ratio: float = 0.40
    thickness_ratio: float = 0.12
    skew_tip_deg: float = 25.0
    chord_root_factor: float = 0.75
    chord_tip_factor: float = 0.35

    def __post_init__(self) -> None:
        for name in ("diameter", "hub_ratio", "pitch_ratio", "chord_ratio",
                     "thickness_ratio", "skew_tip_deg", "chord_root_factor",
                     "chord_tip_factor"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.diameter <= 0.0:
            raise ValueError("diameter must be positive")
        if self.n_blades < 2:
            raise ValueError("n_blades must be >= 2")
        if not 0.0 < self.hub_ratio < 1.0:
            raise ValueError("hub_ratio must lie in (0, 1)")
        if min(self.pitch_ratio, self.chord_ratio, self.thickness_ratio) <= 0.0:
            raise ValueError("pitch_ratio, chord_ratio and thickness_ratio must be positive")
        if self.chord_ratio * (1.6 if self.chord_root_factor > 1.5 else 1.0) >= 1.0:
            raise ValueError("chord_ratio implies an unphysically large chord")

    # -- basic dimensions ---------------------------------------------------
    @property
    def radius(self) -> float:
        """Tip radius ``R`` [m]."""
        return 0.5 * self.diameter

    @property
    def hub_radius(self) -> float:
        """Hub radius ``r_h`` [m]."""
        return 0.5 * self.hub_ratio * self.diameter

    @property
    def pitch(self) -> float:
        """Constant pitch ``P`` [m]."""
        return self.pitch_ratio * self.diameter

    @property
    def chord_at_07(self) -> float:
        """Chord at ``0.7 R`` [m]."""
        return self.chord_ratio * self.diameter

    # -- radial laws --------------------------------------------------------
    def _chord_coefficients(self) -> tuple[float, float, float]:
        """Quadratic ``q(xi) = a0 + a1 xi + a2 xi**2`` with ``c = c(0.7R) q``."""
        anchors_x = np.array([self.hub_ratio, 0.7, 1.0])
        anchors_y = np.array([self.chord_root_factor, 1.0, self.chord_tip_factor])
        design = np.stack([np.ones_like(anchors_x), anchors_x, anchors_x ** 2], axis=1)
        coefficients = np.linalg.solve(design, anchors_y)
        return tuple(float(c) for c in coefficients)

    def chord(self, r: Any) -> Any:
        """Chord ``c(r)`` [m] at radius ``r`` [m] (span-clipped)."""
        r = np.asarray(r, dtype=float)
        xi = np.clip(r / self.radius, self.hub_ratio, 1.0)
        a0, a1, a2 = self._chord_coefficients()
        chord = self.chord_at_07 * (a0 + a1 * xi + a2 * xi ** 2)
        return float(chord) if chord.ndim == 0 else chord

    def pitch_angle(self, r: Any) -> Any:
        """Section pitch angle ``phi(r) = atan(P / (2 pi r))`` [rad]."""
        r = np.asarray(r, dtype=float)
        phi = np.arctan(self.pitch / (2.0 * np.pi * np.maximum(r, 1e-12)))
        return float(phi) if phi.ndim == 0 else phi

    def skew_angle(self, r: Any) -> Any:
        """Skew angle [rad]: linear from zero at the hub to ``skew_tip`` at the tip."""
        r = np.asarray(r, dtype=float)
        span = max(self.radius - self.hub_radius, 1e-12)
        s = np.clip((r - self.hub_radius) / span, 0.0, 1.0)
        skew = math.radians(self.skew_tip_deg) * s
        return float(skew) if skew.ndim == 0 else skew

    def thickness_ratio_at(self, r: Any) -> Any:
        """Section thickness/chord: ``1.6 t/c`` at the hub linearly to ``0.45 t/c`` at the tip."""
        r = np.asarray(r, dtype=float)
        span = max(self.radius - self.hub_radius, 1e-12)
        s = np.clip((r - self.hub_radius) / span, 0.0, 1.0)
        ratio = self.thickness_ratio * (1.6 - 1.15 * s)
        return float(ratio) if ratio.ndim == 0 else ratio

    def thickness(self, r: Any) -> Any:
        """Maximum section thickness ``t(r)`` [m]."""
        r = np.asarray(r, dtype=float)
        return self.thickness_ratio_at(r) * self.chord(r)

    # -- loss factors -------------------------------------------------------
    def tip_loss(self, r: Any, n_blades: int | None = None) -> Any:
        """Prandtl tip-loss factor at radius ``r`` (``F -> 0`` at the tip)."""
        r = np.asarray(r, dtype=float)
        z = self.n_blades if n_blades is None else int(n_blades)
        phi = np.asarray(self.pitch_angle(r), dtype=float)
        span = np.maximum(self.radius - r, 1e-9)
        f = 0.5 * z * span / np.maximum(r * np.sin(np.maximum(phi, 1e-6)), 1e-9)
        loss = (2.0 / math.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
        return float(loss) if loss.ndim == 0 else loss

    # -- strip discretisation ----------------------------------------------
    def radial_strips(self, n_strips: int) -> dict[str, np.ndarray]:
        """Uniform-in-radius strip discretisation between the hub and the tip.

        Returns arrays of length ``n_strips``: ``r`` (strip centre), ``dr``
        (width), ``chord``, ``pitch_angle``, ``skew``, ``thickness``,
        ``tip_loss`` and ``chord_angle`` (``c / r``, the azimuth subtended by the
        chord, which is the milling model's entry-ramp duration).
        """
        if n_strips < 2:
            raise ValueError("n_strips must be >= 2")
        edges = np.linspace(self.hub_radius, self.radius, n_strips + 1)
        r = 0.5 * (edges[:-1] + edges[1:])
        dr = np.diff(edges)
        chord = np.asarray(self.chord(r), dtype=float)
        phi = np.asarray(self.pitch_angle(r), dtype=float)
        return {
            "r": r,
            "dr": dr,
            "chord": chord,
            "pitch_angle": phi,
            "skew": np.asarray(self.skew_angle(r), dtype=float),
            "thickness": np.asarray(self.thickness(r), dtype=float),
            "tip_loss": np.asarray(self.tip_loss(r), dtype=float),
            "chord_angle": chord / np.maximum(r, 1e-9),
        }

    # -- integral properties ----------------------------------------------
    def expanded_area_ratio(self, n_strips: int = 400) -> float:
        """Expanded blade area ratio ``A_E / A_0``."""
        strips = self.radial_strips(n_strips)
        area = self.n_blades * float(np.sum(strips["chord"] * strips["dr"]))
        return area / (math.pi * self.radius ** 2)

    def verifying_note(self) -> str:
        """Human-readable geometry summary used in the report/audit."""
        strips = self.radial_strips(200)
        return (
            f"Z={self.n_blades} D={self.diameter:.3f} m r_h/R={self.hub_ratio:.2f} "
            f"P/D={self.pitch_ratio:.2f} A_E/A_0={self.expanded_area_ratio():.3f} "
            f"c(0.7R)={self.chord_at_07:.3f} m skew_tip={self.skew_tip_deg:.1f} deg "
            f"phi(0.7R)={math.degrees(strips['pitch_angle'][len(strips['r']) // 2]):.1f} deg"
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready geometry record."""
        return {
            "diameter": self.diameter,
            "n_blades": self.n_blades,
            "hub_ratio": self.hub_ratio,
            "pitch_ratio": self.pitch_ratio,
            "chord_ratio": self.chord_ratio,
            "thickness_ratio": self.thickness_ratio,
            "skew_tip_deg": self.skew_tip_deg,
            "radius": self.radius,
            "hub_radius": self.hub_radius,
            "chord_at_0.7R": self.chord_at_07,
            "expanded_area_ratio": self.expanded_area_ratio(),
            "pitch_angle_07R_deg": math.degrees(self.pitch_angle(0.7 * self.radius)),
        }


@dataclass(frozen=True)
class IceBlockGeometry:
    """Annular ice-block sector milled by the propeller.

    Parameters
    ----------
    thickness
        Ice thickness ``h_ice`` [m].
    r_inner_frac, r_outer_frac
        Radial extent of the ice block as fractions of the propeller radius.
    block_fraction
        Azimuthal occupancy of the ice block, in ``(0, 1]`` of the full circle.
    centre_angle_rad
        Azimuth of the sector centre [rad]; azimuth zero is the +x axis and the
        propeller turns in the +theta direction.
    """

    thickness: float = 0.40
    r_inner_frac: float = 0.35
    r_outer_frac: float = 0.95
    block_fraction: float = 0.65
    centre_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in ("thickness", "r_inner_frac", "r_outer_frac",
                     "block_fraction", "centre_angle_rad"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.thickness <= 0.0:
            raise ValueError("ice thickness must be positive")
        if not 0.0 <= self.r_inner_frac < self.r_outer_frac <= 1.0:
            raise ValueError("require 0 <= r_inner_frac < r_outer_frac <= 1")
        if not 0.0 < self.block_fraction <= 1.0:
            raise ValueError("block_fraction must lie in (0, 1]")

    # -- derived quantities -------------------------------------------------
    def r_inner(self, radius: float) -> float:
        return self.r_inner_frac * radius

    def r_outer(self, radius: float) -> float:
        return self.r_outer_frac * radius

    @property
    def sector_width_rad(self) -> float:
        """Angular width of the ice sector [rad]."""
        return 2.0 * math.pi * self.block_fraction

    def entry_angle(self) -> float:
        """Azimuth of the sector's leading edge [rad] (blades enter here)."""
        return self.centre_angle_rad - 0.5 * self.sector_width_rad

    def blocked_area(self, radius: float) -> float:
        """Blocked ice area in the propeller plane [m^2]."""
        annulus = math.pi * (self.r_outer(radius) ** 2 - self.r_inner(radius) ** 2)
        return self.block_fraction * annulus

    def radial_mask(self, r: np.ndarray, radius: float) -> np.ndarray:
        """Boolean mask of strips whose centre lies inside the ice radial extent."""
        return (r >= self.r_inner(radius)) & (r <= self.r_outer(radius))

    def as_dict(self, radius: float) -> dict[str, Any]:
        """JSON-ready ice-geometry record."""
        return {
            "thickness": self.thickness,
            "r_inner_frac": self.r_inner_frac,
            "r_outer_frac": self.r_outer_frac,
            "block_fraction": self.block_fraction,
            "sector_width_deg": math.degrees(self.sector_width_rad),
            "blocked_area_m2": self.blocked_area(radius),
            "blocked_area_ratio": self.blocked_area(radius) / (math.pi * radius ** 2),
        }