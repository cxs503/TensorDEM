"""Bidisperse specimen generation for the size-segregation validation surface.

Two primitive operations:

``bidisperse_radii``
    radii of a binary sphere/disk mixture defined by the *volume* (3-D) or
    *area* (quasi-2-D slice) fraction of the large species, together with the
    diameter ratio ``R = d_large / d_small``.

``random_sequential_pack``
    a dense, overlap-free initial configuration inside a rectangle.  The
    insertion order is size-descending (large particles first, small ones fill
    the remaining voids), which is the standard way of reaching a high volume
    fraction with a random-sequential-adsorption (RSA) algorithm.

Both functions are deterministic for a given seed, which is what the
three-seed reproducibility check of the validation campaign relies on.

Units are SI throughout (m, kg, s).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "bidisperse_radii",
    "large_volume_fraction",
    "random_sequential_pack",
    "mixture_area",
]


def bidisperse_radii(
    n: int,
    ratio: float,
    large_volume_fraction: float,
    *,
    small_radius: float = 1.0e-3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(radii, is_large)`` for a binary mixture.

    ``large_volume_fraction`` is the fraction of the *solid volume* (3-D) or
    solid area (2-D slice, uniform thickness) carried by the large species, so
    it is the quantity that enters the nominal fine/large fraction used by the
    segregation index.  ``ratio == 1`` gives a monodisperse suspension with a
    random colouring, i.e. the negative control: the two species are
    geometrically identical and no segregation can be physical.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("ratio must be finite and >= 1")
    if not 0.0 < large_volume_fraction < 1.0:
        raise ValueError("large_volume_fraction must lie in (0, 1)")
    if not math.isfinite(small_radius) or small_radius <= 0.0:
        raise ValueError("small_radius must be finite and positive")

    r_large = ratio * small_radius
    # area (2-D) or volume/area equivalence: the solver is 2-D, so the conserved
    # measure is the disk area; the *ratio* of areas is what fixes the counts.
    area_ratio = ratio**2
    n_large = n / (1.0 + area_ratio * (1.0 - large_volume_fraction) / large_volume_fraction)
    n_large_int = int(round(n_large))
    n_large_int = min(max(n_large_int, 1), n - 1)
    is_large = np.zeros(n, dtype=bool)
    is_large[:n_large_int] = True
    rng = np.random.default_rng(seed)
    is_large = is_large[rng.permutation(n)]
    radii = np.where(is_large, r_large, small_radius)
    return radii, is_large


def mixture_area(radii: np.ndarray) -> float:
    """Solid area (2-D) of the specimen, i.e. ``sum(pi r^2)``."""
    return float(np.sum(math.pi * np.asarray(radii, dtype=np.float64) ** 2))


def large_volume_fraction(radii: np.ndarray, is_large: np.ndarray) -> float:
    """Realised large-species solid fraction of a mixture."""
    area = math.pi * np.asarray(radii, dtype=np.float64) ** 2
    total = float(area.sum())
    if total <= 0.0:
        raise ValueError("the specimen has no solid area")
    return float(area[is_large].sum() / total)


def random_sequential_pack(
    radii: np.ndarray,
    width: float,
    height: float,
    *,
    seed: int = 0,
    target_fraction: float = 0.78,
    margin: float = 0.0,
    max_attempts_per_particle: int = 4000,
) -> np.ndarray:
    """Overlap-free random packing of ``radii`` in ``[0, width] x [0, height]``.

    Parameters
    ----------
    target_fraction
        Solid area fraction at which insertion stops; the caller sizes
        ``width``/``height`` so that the whole specimen fits.  Reaching a high
        fraction is faster when the largest particles are placed first, hence
        the size-descending insertion order.
    margin
        Extra horizontal room: particles are allowed to sit in
        ``[-margin, width + margin]``.  Used to pre-load a piling box that is
        wider than the gate.

    Raises
    ------
    RuntimeError
        If the packing stalls (``target_fraction`` unreachable inside the
        attempt budget); the caller is expected to enlarge the box or lower the
        target.  Failing loudly is deliberate: a silently under-filled specimen
        would change the nominal mixture fraction of the segregation index.
    """
    rad = np.asarray(radii, dtype=np.float64).reshape(-1)
    if rad.size < 1:
        raise ValueError("at least one particle is required")
    if not np.isfinite(rad).all() or (rad <= 0.0).any():
        raise ValueError("radii must be finite and positive")
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must lie in (0, 1)")
    if not math.isfinite(width) or width <= 0.0 or not math.isfinite(height) or height <= 0.0:
        raise ValueError("width and height must be finite and positive")
    area = float(np.sum(math.pi * rad**2))
    box = (width + 2.0 * margin) * height
    if area / box > target_fraction * (1.0 + 1e-9):
        raise ValueError(
            f"specimen needs area fraction {area / box:.3f} > target {target_fraction:.3f}"
        )

    rng = np.random.default_rng(seed)
    n = rad.size
    pos = np.full((n, 2), np.nan)
    r_max = float(rad.max())
    cell = 2.0 * r_max
    n_x = int(math.ceil((width + 2.0 * margin) / cell)) + 1
    n_y = int(math.ceil(height / cell)) + 1
    grid: list[list[int]] = [[] for _ in range(n_x * n_y)]
    x_min = -margin
    # plain Python state for the trial loop: numpy scalars cost ~1 us per op,
    # which dominates an insertion loop that runs ~10^6 times
    r_list = [float(v) for v in rad]
    px = [0.0] * n
    py = [0.0] * n
    rnd = rng.random
    span_max = int(math.ceil((2.0 * r_max + margin) / cell)) + 1
    placed = 0
    for idx in np.argsort(-rad, kind="stable"):
        idx = int(idx)
        r = r_list[idx]
        x_lo = x_min + r
        x_span = width + 2.0 * margin - 2.0 * r
        y_span = height - 2.0 * r
        span = int(math.ceil((r + r_max) / cell)) + 1
        for _ in range(max_attempts_per_particle):
            cx = x_lo + rnd() * x_span
            cy = r + rnd() * y_span
            ix = int((cx - x_min) / cell)
            if ix < 0:
                ix = 0
            elif ix >= n_x:
                ix = n_x - 1
            iy = int(cy / cell)
            if iy < 0:
                iy = 0
            elif iy >= n_y:
                iy = n_y - 1
            hit = False
            for gx in range(ix - span, ix + span + 1):
                if gx < 0 or gx >= n_x:
                    continue
                for gy in range(iy - span, iy + span + 1):
                    if gy < 0 or gy >= n_y:
                        continue
                    for other in grid[gy * n_x + gx]:
                        dx = cx - px[other]
                        dy = cy - py[other]
                        rr = r + r_list[other]
                        if dx * dx + dy * dy < rr * rr:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if not hit:
                px[idx] = cx
                py[idx] = cy
                pos[idx, 0] = cx
                pos[idx, 1] = cy
                grid[iy * n_x + ix].append(idx)
                placed += 1
                break
        else:
            done = 0.0
            if placed:
                done = float(np.sum(math.pi * rad[np.flatnonzero(~np.isnan(pos[:, 0]))] ** 2)) / box
            raise RuntimeError(
                f"packing stalled after placing {placed}/{n} particles "
                f"(area fraction reached {done:.3f}, target {target_fraction:.3f}); "
                "enlarge the box or lower the target"
            )
    return pos
