"""Ship-formed hull outline for the plan-view floe DEM.

The baseline ``tensordem.ice_channel.HullGeometry`` is an equal-width block, so
every ice contact is broadside: the ice is shoved sideways by a vertical wall
rather than deflected by an inclined bow.  A ship form is a *convex* polygon in
plan view --

    stem apex  ->  shoulder  ->  stern quarter  ->  stern quarter  ->  shoulder

with a wedge bow of the stated entry angle and a parallel body behind it.  For a
convex outline the signed distance is

    s(p) = max_i n_i . (p - v_i)

(``n_i`` outward face normals, ``v_i`` a point on face ``i``), exact inside and
at the faces; at a corner it is the face-plane distance, which over-estimates
the true distance by at most ``(sqrt(2) - 1) * R`` for a floe of radius ``R``
touching a corner.  The approximation is declared: bow shoulders of a real hull
are rounded, and the corner region is a small fraction of the waterline.

All quantities SI: metres, m/s, radians internally, degrees at the boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class ShipForm:
    """Plan-view hull outline of a ship advancing along +x.

    ``x`` is the *stem apex* abscissa (``x_stem(t) = x_start + speed * t``); the
    hull extends backwards to ``x_stem - length``.  ``y`` is the transverse axis
    with the centreline at ``0``.
    """

    length: float = 105.0
    beam: float = 26.0
    speed: float = 3.0
    entry_angle_deg: float = 18.0
    draught: float = 8.0
    y_position: float = 0.0
    x_start: float = 5.0
    form: str = "ship"          # "ship" (wedge bow) | "block" (baseline)
    name: str = "MT Uikku equivalent transport"

    def __post_init__(self) -> None:
        for name in ("length", "beam", "speed", "draught"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.length <= 0.0 or self.beam <= 0.0:
            raise ValueError("length and beam must be positive")
        if not 0.0 < self.entry_angle_deg < 89.0:
            raise ValueError("entry_angle_deg must lie in (0, 89)")
        if self.form not in ("ship", "block"):
            raise ValueError("form must be 'ship' or 'block'")

    # -- outline ------------------------------------------------------------
    @property
    def bow_length(self) -> float:
        """Plan-view length of the wedge bow [m] (zero for the block form)."""
        if self.form == "block":
            return 0.0
        return 0.5 * self.beam / math.tan(math.radians(self.entry_angle_deg))

    @property
    def half_beam(self) -> float:
        return 0.5 * self.beam

    @property
    def half_length(self) -> float:
        return 0.5 * self.length

    def x_centre(self, time: float) -> float:
        """Stem-apex abscissa at ``time`` (the base class' hull convention)."""
        return float(self.x_start) + float(self.speed) * float(time)

    def velocity(self) -> torch.Tensor:
        return torch.tensor([float(self.speed), 0.0], dtype=torch.float64)

    def outline(self, x_stem: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Outward face normals and a point on each face, counter-clockwise."""
        yc = float(self.y_position)
        hb = self.half_beam
        bow = self.bow_length
        stern = x_stem - self.length
        if bow > 0.0:
            vertices = [
                (x_stem, yc),                  # stem apex
                (x_stem - bow, yc + hb),       # starboard shoulder
                (stern, yc + hb),              # starboard quarter
                (stern, yc - hb),              # port quarter
                (x_stem - bow, yc - hb),       # port shoulder
            ]
        else:
            vertices = [
                (x_stem, yc + hb),
                (stern, yc + hb),
                (stern, yc - hb),
                (x_stem, yc - hb),
            ]
        normals, points = [], []
        n = len(vertices)
        for i in range(n):
            x0, y0 = vertices[i]
            x1, y1 = vertices[(i + 1) % n]
            ex, ey = x1 - x0, y1 - y0
            length = math.hypot(ex, ey)
            if length <= 0.0:
                continue
            # outward normal of a counter-clockwise polygon: (ey, -ex)
            normals.append((ey / length, -ex / length))
            points.append((x0, y0))
        return (torch.tensor(normals, dtype=torch.float64),
                torch.tensor(points, dtype=torch.float64))

    # -- contact query ------------------------------------------------------
    def signed_distance(self, pos: torch.Tensor, x_stem: float
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Face-plane signed distance and the outward normal of the active face.

        ``s > 0`` outside, ``s < 0`` inside, ``|s|`` the (face-plane) distance to
        the outline.  ``pos`` is ``(N, 2)``; both returns are ``(N,)`` / ``(N, 2)``.
        """
        normals, points = self.outline(x_stem)
        # d[i, k] = n_k . (p_i - v_k)
        delta = pos[:, None, :] - points[None, :, :]
        d = (delta * normals[None, :, :]).sum(dim=-1)          # (N, K)
        best = d.argmax(dim=1)
        signed = d.gather(1, best[:, None]).squeeze(1)
        normal = normals.gather(0, best[:, None].expand(-1, 2))
        return signed, normal

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "length_m": self.length,
            "beam_m": self.beam,
            "draught_m": self.draught,
            "entry_angle_deg": self.entry_angle_deg,
            "bow_length_m": self.bow_length,
            "speed_m_s": self.speed,
            "form": self.form,
        }