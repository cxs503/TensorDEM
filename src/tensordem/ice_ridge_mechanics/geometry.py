"""Ridge sail/keel geometry statistics and the Hopky (1983) empirical anchor.

The empirical anchor used by the campaign node is

    H_sail = 0.4 * sqrt(h * d_squeeze)          (Hopky & Lawrence 1983 form)
    D_keel = 4 * H_sail

with ``h`` the parent ice-sheet thickness (m) and ``d_squeeze`` the convergence
of the two ice sheets that feed the ridge (m).  ``d_squeeze = 0`` is a physical
negative control: no convergence, no ridge, ``H_sail = 0``.

The geometry statistics are measured on the *rubble* particles only (the intact
parent sheets are removed from the sample using the connected components of the
freeze-bond graph), so the numbers are not contaminated by the flat sheets that
still straddle the waterline.
"""
from __future__ import annotations

import math

import torch
from scipy.spatial import ConvexHull, QhullError, cKDTree

__all__ = [
    "hopky_sail_height",
    "hopky_keel_depth",
    "ridge_geometry",
    "waterline_width",
]

DEFAULT_RATIO = 4.0


def _finite(name: str, value, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")


def hopky_sail_height(ice_thickness: float, squeeze: float) -> float:
    """``H_sail = 0.4 sqrt(h d)``; exactly zero when ``d = 0`` (negative control)."""
    _finite("ice_thickness", ice_thickness, positive=True)
    _finite("squeeze", squeeze)
    if squeeze < 0.0:
        raise ValueError("squeeze must be nonnegative")
    if squeeze == 0.0:
        return 0.0
    return 0.4 * math.sqrt(ice_thickness * squeeze)


def hopky_keel_depth(ice_thickness: float, squeeze: float,
                     ratio: float = DEFAULT_RATIO) -> float:
    """``D_keel = ratio * H_sail`` (``ratio = 4`` in the campaign node)."""
    _finite("ratio", ratio, positive=True)
    return ratio * hopky_sail_height(ice_thickness, squeeze)


def waterline_width(positions: torch.Tensor, radii: torch.Tensor,
                    waterline: float = 0.0) -> float:
    """Horizontal extent of the disks that straddle the waterline, metres."""
    if positions.numel() == 0:
        return 0.0
    band = (positions[:, 1] - waterline).abs() < radii
    if not bool(band.any()):
        return 0.0
    xs = positions[band, 0]
    return float(xs.max() - xs.min())


def ridge_geometry(positions, radii, *, waterline: float = 0.0,
                   column_width: float | None = None,
                   peak_count: int = 3) -> dict:
    """Measure sail height, keel depth, width and porosity of a rubble sample.

    Parameters
    ----------
    positions, radii : array-like, shape (N, 2) and (N,)
        Centres and radii of the *rubble* disks, metres.
    waterline : float
        Waterline elevation (y = 0 in the simulations).
    column_width : float, optional
        Width of the vertical strips used to integrate the section area.
        Defaults to the mean diameter.
    peak_count : int
        Number of extreme particles averaged for the sail peak / keel tip, which
        removes single-disk spikes without smoothing away the ridge.

    Returns
    -------
    dict with ``sail_height``, ``keel_depth``, ``width``, ``porosity``,
    ``solid_area``, ``envelope_area``, ``sail_area``, ``keel_area`` and
    ``column_count``.
    """
    pos = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 2)
    rad = torch.as_tensor(radii, dtype=torch.float64).reshape(-1)
    if pos.shape[0] != rad.shape[0]:
        raise ValueError("positions and radii must have the same length")
    count = int(pos.shape[0])
    empty = {"sail_height": 0.0, "keel_depth": 0.0, "width": 0.0, "porosity": 0.0,
             "solid_area": 0.0, "envelope_area": 0.0, "sail_area": 0.0,
             "keel_area": 0.0, "column_count": 0, "particle_count": count}
    if count == 0:
        return empty
    if peak_count < 1:
        raise ValueError("peak_count must be at least one")
    width = waterline_width(pos, rad, waterline)
    if width <= 0.0:
        # no disk straddles the waterline: fall back to the full x extent
        width = float(pos[:, 0].max() - pos[:, 0].min())
    centre = 0.5 * float(pos[:, 0].max() + pos[:, 0].min())
    half = 0.5 * width
    zone = (pos[:, 0] - centre).abs() <= half
    pos_zone, rad_zone = pos[zone], rad[zone]
    if pos_zone.shape[0] == 0:
        return empty
    tops = (pos_zone[:, 1] + rad_zone)
    bottoms = (pos_zone[:, 1] - rad_zone)
    above = tops[tops > waterline]
    below = bottoms[bottoms < waterline]
    if above.numel():
        k = min(peak_count, int(above.numel()))
        sail_height = float(above.sort(descending=True).values[:k].mean()) - waterline
    else:
        sail_height = 0.0
    if below.numel():
        k = min(peak_count, int(below.numel()))
        keel_depth = waterline - float(below.sort().values[:k].mean())
    else:
        keel_depth = 0.0
    # column integration of the section envelope
    if column_width is None:
        column_width = float(2.0 * rad.mean())
    _finite("column_width", float(column_width), positive=True)
    x_min = float(pos_zone[:, 0].min() - rad_zone.max())
    x_max = float(pos_zone[:, 0].max() + rad_zone.max())
    columns = max(int(round((x_max - x_min) / column_width)), 1)
    solid = float((math.pi * rad_zone ** 2).sum())
    envelope = 0.0
    sail_area = 0.0
    keel_area = 0.0
    used = 0
    for index in range(columns):
        lo = x_min + index * column_width
        hi = lo + column_width
        in_col = (pos_zone[:, 0] + rad_zone > lo) & (pos_zone[:, 0] - rad_zone < hi)
        if not bool(in_col.any()):
            continue
        top = float((pos_zone[in_col, 1] + rad_zone[in_col]).max())
        bottom = float((pos_zone[in_col, 1] - rad_zone[in_col]).min())
        used += 1
        envelope += (top - bottom) * column_width
        sail_area += max(top - waterline, 0.0) * column_width
        keel_area += max(waterline - bottom, 0.0) * column_width
    porosity = (min(max(1.0 - solid / envelope, 0.0), 1.0)
                if envelope > 0.0 else 0.0)
    return {
        "sail_height": max(sail_height, 0.0),
        "keel_depth": max(keel_depth, 0.0),
        "width": width,
        "porosity": porosity,
        "solid_area": solid,
        "envelope_area": envelope,
        "sail_area": sail_area,
        "keel_area": keel_area,
        "column_count": used,
        "particle_count": count,
    }

def union_area(positions, radii, cells: int = 160) -> float:
    """Area of the *union* of the disks, m^2 (grid quadrature, 1 % accuracy).

    ``N pi r^2`` over-counts because touching disks overlap, which is how a
    column-integrated porosity could fall to zero in the first acceptance run.
    """
    pos = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 2)
    rad = torch.as_tensor(radii, dtype=torch.float64).reshape(-1)
    if pos.shape[0] == 0:
        return 0.0
    r_max = float(rad.max())
    lo = pos.min(dim=0).values - r_max
    hi = pos.max(dim=0).values + r_max
    span = hi - lo
    nx = max(int(cells), 8)
    ny = max(int(cells * float(span[1] / max(float(span[0]), 1e-9))), 8)
    xs = torch.linspace(float(lo[0]), float(hi[0]), nx)
    ys = torch.linspace(float(lo[1]), float(hi[1]), ny)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    tree = cKDTree(pos.numpy())
    distance, index = tree.query(grid.numpy())
    inside = distance <= rad.numpy()[index]
    cell = float(span[0] / (nx - 1)) * float(span[1] / (ny - 1))
    return float(inside.sum()) * cell


def envelope_area(positions, radii) -> float:
    """Area of the union-envelope of the assembly, m^2 per metre of ridge.

    The envelope is the convex hull of the disk centres offset by the particle
    radius (Minkowski sum with a disk), ``A = A_hull + P_hull r + pi r^2``.  It is
    the standard "r-offset" envelope and it is always >= the solid area, so the
    porosity cannot become negative, unlike a column integration which
    double-counts disks that straddle a column boundary.
    """
    pos = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 2)
    rad = torch.as_tensor(radii, dtype=torch.float64).reshape(-1)
    if pos.shape[0] == 0:
        return 0.0
    r = float(rad.mean())
    if pos.shape[0] < 3:
        span = float(torch.linalg.vector_norm(pos.max(dim=0).values - pos.min(dim=0).values))
        return span * 2.0 * r + math.pi * r * r
    points = pos.numpy()
    try:
        hull = ConvexHull(points)
        return float(hull.volume + hull.area * r + math.pi * r * r)
    except QhullError:
        return float(pos.shape[0]) * math.pi * r * r


def ridge_profile_geometry(positions, radii, *, waterline: float = 0.0,
                           ice_thickness: float, thickness_factor: float = 1.6,
                           column_width: float | None = None,
                           peak_count: int = 3) -> dict:
    """Ridge geometry discriminated from the intact parent ice sheets.

    The two sheets that feed a ridge are slabs of thickness ``h`` whose columns
    straddle the waterline over the whole domain, so ``waterline_width`` is
    useless for them.  A ridge column, in contrast, is much thicker than the
    parent sheet.  Columns are therefore binned over the full x range and the
    ridge is the contiguous block of columns whose ice thickness exceeds
    ``thickness_factor * h`` around the thickest column.

    Returns the same keys as :func:`ridge_geometry` plus ``x_min``/``x_max`` of
    the ridge and ``thickness`` per column.
    """
    pos = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 2)
    rad = torch.as_tensor(radii, dtype=torch.float64).reshape(-1)
    if pos.shape[0] != rad.shape[0]:
        raise ValueError("positions and radii must have the same length")
    _finite("ice_thickness", ice_thickness, positive=True)
    _finite("thickness_factor", thickness_factor)
    if pos.shape[0] == 0:
        return {"sail_height": 0.0, "keel_depth": 0.0, "width": 0.0, "porosity": 0.0,
                "solid_area": 0.0, "envelope_area": 0.0, "sail_area": 0.0,
                "keel_area": 0.0, "column_count": 0, "particle_count": 0,
                "x_min": 0.0, "x_max": 0.0, "thickness": []}
    if column_width is None:
        column_width = float(2.0 * rad.mean())
    x_min_all = float(pos[:, 0].min() - rad.max())
    x_max_all = float(pos[:, 0].max() + rad.max())
    columns = max(int(round((x_max_all - x_min_all) / column_width)), 1)
    edges = x_min_all + torch.arange(columns + 1, dtype=torch.float64) * column_width
    tops = torch.full((columns,), -float("inf"), dtype=torch.float64)
    bottoms = torch.full((columns,), float("inf"), dtype=torch.float64)
    bin_index = ((pos[:, 0] - x_min_all) / column_width).long().clamp(0, columns - 1)
    top_val = pos[:, 1] + rad
    bottom_val = pos[:, 1] - rad
    for k in range(columns):
        in_col = bin_index == k
        if bool(in_col.any()):
            tops[k] = top_val[in_col].max()
            bottoms[k] = bottom_val[in_col].min()
    thickness = tops - bottoms
    filled = torch.isfinite(thickness)
    thick = torch.where(filled, thickness, torch.zeros_like(thickness))
    threshold = thickness_factor * ice_thickness
    ridge = filled & (thick > threshold)
    stats = {"sail_height": 0.0, "keel_depth": 0.0, "width": 0.0, "porosity": 0.0,
             "solid_area": 0.0, "envelope_area": 0.0, "sail_area": 0.0,
             "keel_area": 0.0, "column_count": 0, "particle_count": int(pos.shape[0]),
             "x_min": 0.0, "x_max": 0.0, "thickness": thick.tolist()}
    if not bool(ridge.any()):
        return stats
    anchor = int(torch.argmax(thick).item())
    lo = anchor
    while lo - 1 >= 0 and bool(ridge[lo - 1]):
        lo -= 1
    hi = anchor
    while hi + 1 < columns and bool(ridge[hi + 1]):
        hi += 1
    block = slice(lo, hi + 1)
    col_tops = tops[block]
    col_bottoms = bottoms[block]
    above = (col_tops[col_tops > waterline] - waterline)
    below = (waterline - col_bottoms[col_bottoms < waterline])
    sail_height = 0.0 if above.numel() == 0 else float(above.sort(descending=True).values[:peak_count].mean())
    keel_depth = 0.0 if below.numel() == 0 else float(below.sort(descending=True).values[:peak_count].mean())
    in_block = (bin_index >= lo) & (bin_index <= hi)
    solid = union_area(pos[in_block], rad[in_block])
    envelope = envelope_area(pos[in_block], rad[in_block])
    sail_area = float((col_tops - waterline).clamp_min(0.0).sum()) * column_width
    keel_area = float((waterline - col_bottoms).clamp_min(0.0).sum()) * column_width
    stats.update({
        "sail_height": sail_height,
        "keel_depth": keel_depth,
        "width": float(hi - lo + 1) * column_width,
        "porosity": (min(max(1.0 - solid / envelope, 0.0), 1.0)
                     if envelope > 0.0 else 0.0),
        "solid_area": solid,
        "envelope_area": envelope,
        "sail_area": sail_area,
        "keel_area": keel_area,
        "column_count": int(hi - lo + 1),
        "x_min": float(edges[lo]),
        "x_max": float(edges[hi + 1]),
        "thickness": thick.tolist(),
    })
    return stats
