"""Initial rubble bed for the keel DEM: geometry-consistent dilute placement.

The blocks of a ridge keel cannot be placed by Random Sequential Adsorption at
the target porosity: RSA saturates at a volume fraction of ~0.38 (porosity
~0.62) for equal spheres, far looser than the ``n_0 ~ 0.4`` of a freshly formed
keel.  The physical way to obtain a loose packing is to let the blocks
*deposit* under gravity, which is what the DEM does.  This module therefore
builds only the *dilute* starting configuration -- blocks scattered without
overlap inside the keel volume -- and the deposition stage of
:class:`..rubble_dem.RubbleDEM` produces the loose bed.

Volume budget: ``n_blocks * <V_block>`` solid volume of ice must fit in the
keel frustum at the target porosity; :func:`slab_bounds` returns the depth
interval of the frustum that holds it, so the parcel thickness is set by mass
balance instead of being assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from .geometry import KeelGeometry, keel_volume
from .rubble_dem import container_volume
from scipy.spatial import cKDTree

__all__ = [
    "solid_volume_of",
    "slab_bounds",
    "dilute_positions",
    "DiluteBed",
]


def solid_volume_of(radii: np.ndarray) -> float:
    """Total solid volume of the blocks (m3)."""
    return float(np.sum(4.0 / 3.0 * math.pi * np.asarray(radii, dtype=float) ** 3))


def slab_bounds(geom: KeelGeometry, solid_volume: float, porosity: float,
                from_bottom: bool = True) -> Tuple[float, float]:
    """Depth interval of the frustum that holds ``solid_volume`` at ``porosity``.

    Returns ``(z_lo, z_hi)`` with ``z_hi = geom.depth`` when ``from_bottom``:
    the rubble parcel sits on the deepest part of the keel (the tip), which is
    where rubble accumulates.  Bisection on the exact analytic frustum volume.
    """
    if not 0.0 < porosity < 1.0:
        raise ValueError("porosity must be in (0, 1)")
    bulk = solid_volume / (1.0 - porosity)
    total = keel_volume(geom)
    if bulk > total:
        raise ValueError(
            f"parcel bulk volume {bulk:.2f} m3 exceeds the keel volume {total:.2f} m3")
    lo, hi = 0.0, geom.depth
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if container_volume(geom, mid, geom.depth) > bulk:
            lo = mid          # too much volume below mid -> move the base deeper
        else:
            hi = mid
    return 0.5 * (lo + hi), geom.depth


@dataclass
class DiluteBed:
    """Dilute starting configuration produced by :func:`dilute_positions`."""

    positions: np.ndarray
    radii: np.ndarray
    placement_porosity: float
    z_lo: float
    z_hi: float
    attempts_used: int
    accepted: int

    @property
    def solid_volume(self) -> float:
        return solid_volume_of(self.radii)


def dilute_positions(geom: KeelGeometry, radii: np.ndarray, rng: np.random.Generator,
                     placement_porosity: float = 0.85,
                     z_lo: float | None = None,
                     min_gap: float = 0.0,
                     max_attempts_per_block: int = 4000) -> DiluteBed:
    """Scatter the blocks inside the keel without overlap (dilute state).

    ``placement_porosity`` sets how much free room the blocks get: the parcel
    occupies the depth interval in which the container volume equals
    ``V_solid/(1-placement_porosity)``.  A dilute state is cheap to generate and
    is then densified by the deposition stage under gravity.
    """
    radii = np.asarray(radii, dtype=float)
    solid = solid_volume_of(radii)
    if z_lo is None:
        z_lo, z_hi = slab_bounds(geom, solid, placement_porosity)
    else:
        z_hi = geom.depth
    n = len(radii)
    pos = np.empty((n, 3))
    placed = 0
    attempts = 0
    tree_pos = np.empty((0, 3))
    while placed < n and attempts < n * max_attempts_per_block:
        attempts += 1
        r = radii[placed]
        x = rng.uniform(-geom.L_top / 2, geom.L_top / 2)
        y = rng.uniform(-geom.W_top / 2, geom.W_top / 2)
        z = rng.uniform(max(z_lo, r), geom.depth - r)
        frac = z / geom.depth
        half_l = 0.5 * (geom.L_top - (geom.L_top - geom.L_bot) * frac)
        half_w = 0.5 * (geom.W_top - (geom.W_top - geom.W_bot) * frac)
        if abs(x) > half_l - r or abs(y) > half_w - r:
            continue
        if placed:
            d = np.linalg.norm(tree_pos - np.array([x, y, z]), axis=1)
            need = radii[:placed] + r + min_gap * np.minimum(radii[:placed], r)
            if np.any(d < need):
                continue
        pos[placed] = (x, y, z)
        tree_pos = pos[:placed + 1]
        placed += 1
    if placed < n:
        raise RuntimeError(
            f"only {placed}/{n} blocks could be placed at porosity "
            f"{placement_porosity}; increase the placement porosity")
    volume = container_volume(geom, z_lo, geom.depth)
    return DiluteBed(positions=pos, radii=radii.copy(),
                     placement_porosity=1.0 - solid / volume,
                     z_lo=z_lo, z_hi=z_hi, attempts_used=attempts, accepted=placed)


@dataclass
class CellParcel:
    """Periodic-cell abstraction of a keel rubble parcel.

    ``lx``/``ly`` are the periodic footprint, taken as the keel cross-section
    width at the mid-depth of the parcel; ``target_height`` is the thickness
    that holds the blocks at the target porosity, and ``floor`` is the z of the
    rigid floor (the dilute placement needs headroom, so the cell is taller than
    the settled bed).  Representing the converging frustum walls by periodicity
    keeps the measured porosity a *bulk* property: a rigid funnel arches and
    jams the rubble (see docs) and its wall friction would bias the void ratio.
    """

    lx: float
    ly: float
    floor: float
    target_height: float
    z_lo: float            # shallowest keel depth spanned by the settled parcel
    z_mid: float
    volume: float

    @property
    def box_size(self) -> Tuple[float, float]:
        return (self.lx, self.ly)

    @property
    def footprint(self) -> float:
        return self.lx * self.ly


def parcel_from_keel(geom: KeelGeometry, solid_volume: float, porosity: float,
                     placement_porosity: float = 0.75) -> CellParcel:
    """Map the keel frustum onto an equivalent periodic cell (mass balance)."""
    z_lo, z_hi = slab_bounds(geom, solid_volume, porosity, from_bottom=True)
    z_mid = 0.5 * (z_lo + z_hi)
    frac = z_mid / geom.depth
    lx = geom.L_top - (geom.L_top - geom.L_bot) * frac
    ly = geom.W_top - (geom.W_top - geom.W_bot) * frac
    volume = container_volume(geom, z_lo, z_hi)
    target_height = volume / (lx * ly)
    dilute_height = solid_volume / (1.0 - placement_porosity) / (lx * ly)
    return CellParcel(lx=lx, ly=ly, floor=max(dilute_height, target_height),
                      target_height=target_height, z_lo=z_lo, z_mid=z_mid,
                      volume=volume)


def dilute_cell_positions(parcel: CellParcel, radii: np.ndarray,
                          rng: np.random.Generator,
                          placement_porosity: float = 0.75,
                          min_gap: float = 0.0,
                          max_attempts_per_block: int = 3000) -> DiluteBed:
    """Scatter the blocks in the periodic cell (dilute starting state).

    Uses an incrementally rebuilt KD-tree so the O(N) syntax check stays cheap
    for N ~ 1000.  Neighbour distances use the minimum image, so blocks straddling
    the periodic seam do not start out overlapping.
    """
    radii = np.asarray(radii, dtype=float)
    solid = solid_volume_of(radii)
    lx, ly = parcel.lx, parcel.ly
    dilute_height = solid / (1.0 - placement_porosity) / (lx * ly)
    z_hi = parcel.floor
    z_lo = max(z_hi - dilute_height, 0.0)
    n = len(radii)
    pos = np.empty((n, 3))
    placed = 0
    attempts = 0
    tree = None
    box = np.array([lx, ly, 1e12])
    r_max = float(radii.max())
    while placed < n and attempts < n * max_attempts_per_block:
        attempts += 1
        r = radii[placed]
        x = rng.uniform(0.0, lx)
        y = rng.uniform(0.0, ly)
        z = rng.uniform(max(z_lo, r), z_hi - r)
        cand = np.array([x, y, z])
        ok = True
        if placed:
            if tree is not None:
                for k in tree.query_ball_point(cand, r + r_max + min_gap * r):
                    d = pos[k] - cand
                    d = d - box * np.round(d / box)
                    if np.linalg.norm(d) < radii[k] + r + min_gap * min(radii[k], r):
                        ok = False
                        break
            if ok and placed % 64:
                tail = np.arange(placed - placed % 64, placed)
                d = pos[tail] - cand
                d = d - box * np.round(d / box)
                dist = np.linalg.norm(d, axis=1)
                if np.any(dist < radii[tail] + r + min_gap * np.minimum(radii[tail], r)):
                    ok = False
        if not ok:
            continue
        pos[placed] = cand
        placed += 1
        if placed % 64 == 0:
            tree = cKDTree(pos[:placed], boxsize=[lx, ly, 1e12])
    if placed < n:
        raise RuntimeError(
            f"only {placed}/{n} blocks placed at porosity {placement_porosity}")
    return DiluteBed(positions=pos, radii=radii.copy(),
                     placement_porosity=1.0 - solid / (lx * ly * dilute_height),
                     z_lo=z_lo, z_hi=z_hi, attempts_used=attempts, accepted=placed)
