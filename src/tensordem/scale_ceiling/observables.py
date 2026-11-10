"""Scale-invariant observables: ice resistance, coordination number, repose angle.

All three observables are *intensive*: they are normalised so that a system with
more particles (grown along the periodic ``y`` direction, identical local
geometry) must give the same number.  That is exactly criterion C4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .kernel import FloeBed, FloeBedConfig


def slab_config(
    n_target: int,
    *,
    n_x: int = 24,
    height_layers: int = 8,
    **kwargs,
) -> FloeBedConfig:
    """Build a config whose particle count is ``n_x * layers_y * height_layers``.

    Growing ``layers_y`` grows ``N`` while every other length (and therefore the
    local geometry) stays fixed — the clean way to test scale independence.
    """
    layers_y = max(2, int(round(n_target / float(n_x * height_layers))))
    n = n_x * layers_y * height_layers
    return FloeBedConfig(n=n, layers_y=layers_y, height_layers=height_layers, **kwargs)


def settle(bed: FloeBed, steps: int) -> dict[str, float]:
    """Deposit the bed under gravity with every wall fixed."""
    bed.set_wall_velocity("right", 0.0)
    bed.step(steps)
    return {
        "steps": float(steps),
        "kinetic": bed.kinetic_energy(),
        "coordination": bed.coordination_number(),
        "packing": bed.pack_fraction(),
    }


@dataclass
class CrushResult:
    steps: int
    mean_force: float
    sd_force: float
    mean_force_per_width: float
    indenter_travel: float
    coordination: float
    kinetic: float
    force_series: list[float]

    def as_dict(self) -> dict[str, float]:
        return {
            "steps": self.steps,
            "mean_force_N": self.mean_force,
            "sd_force_N": self.sd_force,
            "mean_force_per_width_N_per_m": self.mean_force_per_width,
            "indenter_travel_m": self.indenter_travel,
            "coordination": self.coordination,
            "kinetic_J": self.kinetic,
        }


def crush_resistance(
    bed: FloeBed,
    *,
    steps: int,
    speed_ratio: float = 0.25,
    settle_steps: int = 800,
    sample_every: int = 5,
) -> CrushResult:
    """Drive the right indenter into the settled bed and average the reaction.

    ``speed_ratio`` is the indenter speed in units of ``sqrt(g d)``; the mean
    normal force on the indenter divided by the box width in ``y`` is the
    reported ice resistance (N/m of indenter width).
    """
    settle(bed, settle_steps)
    speed = -float(speed_ratio) * math.sqrt(bed.config.gravity * bed.config.diameter)
    bed.set_wall_velocity("right", speed)
    start = bed.wall_pos["right"]
    series: list[float] = []
    for k in range(steps):
        bed.advance_indenter()
        bed.step(1)
        if k % sample_every == 0:
            series.append(bed.wall_force("right"))
    arr = np.asarray(series, dtype=np.float64)
    width = float(bed.box[1])
    # drop the first 20 % of the window (transient before the bed is loaded)
    keep = arr[int(0.2 * arr.size):] if arr.size > 5 else arr
    return CrushResult(
        steps=steps,
        mean_force=float(keep.mean()),
        sd_force=float(keep.std(ddof=1)) if keep.size > 1 else 0.0,
        mean_force_per_width=float(keep.mean()) / width,
        indenter_travel=float(start - bed.wall_pos["right"]),
        coordination=bed.coordination_number(),
        kinetic=bed.kinetic_energy(),
        force_series=[float(x) for x in arr],
    )


def _surface_profile(bed: FloeBed, *, n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    pos = bed.pos.numpy()
    x = pos[:, 0]
    z = pos[:, 2]
    lo, hi = float(x.min()), float(x.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    centres, top = [], []
    for k in range(n_bins):
        m = (x >= edges[k]) & (x < edges[k + 1])
        if m.sum() < max(2, bed.config.n_y // 4):
            continue
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        top.append(float(np.percentile(z[m], 90.0)))
    return np.asarray(centres), np.asarray(top)


def slump_repose_angle(
    bed: FloeBed,
    *,
    settle_steps: int = 600,
    collapse_steps: int = 2500,
    spread_factor: float = 3.0,
    tail_fraction: float = 0.55,
) -> dict[str, float]:
    """Angle of repose from a slumping column (free-surface linear fit).

    The right wall is opened and moved ``spread_factor`` box widths away; the
    pile then relaxes to its angle of repose, which is read off the free
    surface over the downhill ``tail_fraction`` of the profile.
    """
    settle(bed, settle_steps)
    bed.box = bed.box.copy()
    bed.box[0] *= spread_factor
    bed.wall_pos["right"] = float(bed.box[0])
    cfg = bed.config
    saved = cfg.right_wall
    cfg.right_wall = False
    try:
        bed.build_neighbours(keep_state=False)
        bed.step(collapse_steps)
    finally:
        cfg.right_wall = saved
    centres, top = _surface_profile(bed)
    if centres.size < 4:
        return {"angle_deg": float("nan"), "n_bins": float(centres.size)}
    # fit the downhill (right) part of the surface
    order = np.argsort(centres)
    centres, top = centres[order], top[order]
    n_tail = max(4, int(tail_fraction * centres.size))
    xs = centres[-n_tail:]
    zs = top[-n_tail:]
    slope, intercept = np.polyfit(xs, zs, 1)
    pred = slope * xs + intercept
    ss_tot = float(((zs - zs.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - float(((zs - pred) ** 2).sum()) / ss_tot
    return {
        "angle_deg": float(math.degrees(math.atan(-slope))) if slope < 0 else float(
            math.degrees(math.atan(abs(slope)))),
        "slope": float(slope),
        "surface_r2": float(r2),
        "n_bins": float(centres.size),
        "profile_x": [float(v) for v in centres],
        "profile_z": [float(v) for v in top],
    }


def cohesion_free_length_scale(bed: FloeBed) -> float:
    """Diagnostic: implied hydrostatic length scale ``sigma/(rho g)`` of the bed."""
    cfg = bed.config
    pressure = bed.mean_contact_pressure()
    return pressure / (cfg.density * cfg.gravity) if cfg.gravity else float("nan")