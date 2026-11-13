"""Deposit-side audit of archived axisymmetric column-collapse runs.

Why this module exists
----------------------
The ``CASE-COLUMN-COLLAPSE`` verdict (Lube et al. 2004, *JFM* 508, 175-199)
is an **experimental** comparison, so the only way to re-check a verdict after
the fact is to go back to the archived raw grain coordinates
(``runout_radius_p995``, ``runout_ratio_p995``, ... are *scalars*: they cannot
reveal whether the deposit the metric was taken on was the deposit the
solver produced).  This module supplies the three diagnostics that can be
computed from raw final positions alone and that the metric needs in order to
be trustworthy:

``horizontal_com_drift``
    Horizontal centre-of-mass displacement.  The initial HCP cylinder is
    coaxial with the floor, so for an axisymmetric collapse the horizontal COM
    must not move; drift is a direct measure of symmetry breaking in the
    integrator / pair build order.
``azimuthal_moments`` + ``azimuthal_shuffle_floor``
    Mass-weighted azimuthal moments ``A_m = |sum_i r_i exp(i m theta_i)| /
    sum_i r_i`` of the deposit footprint.  A radial metric (``r_p995``,
    ``r_max``) is only meaningful for a statistically axisymmetric deposit;
    the moments measure the anisotropy and ``azimuthal_shuffle_floor`` supplies
    the *sampling* floor of the same moments under the null hypothesis that the
    azimuths are i.i.d. uniform (azimuths shuffled, radial distribution kept),
    so a measured ``A_m`` can be quoted as a multiple of its noise floor
    instead of against an arbitrary number.
``decimation_bias``
    Reproducible-from-archive check: what happens to ``r_p995`` when only every
    ``stride``-th archived grain is available.  Motivated by archives that store
    a subsample of the particles (``n_positions < n_particles``); the bias is
    measured on the fully archived cases so that the affected cases can be
    quoted with an uncertainty instead of a bare number.

All functions are pure ``numpy`` on ``(n, 3)`` position arrays in SI units and
do not touch the solver: an audit of archived output must not be able to change
the output it audits.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "horizontal_com_drift",
    "azimuthal_moments",
    "azimuthal_shuffle_floor",
    "azimuthal_uniform_floor",
    "floor_clearance",
    "shape_factor",
    "decimated",
    "decimation_bias",
    "metric_bundle",
    "edge_effective_count",
    "subsample_noise",
]


def _as_pos(pos) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("positions must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions contain non-finite entries")
    return arr


def horizontal_com_drift(pos, radius: float) -> dict:
    """Horizontal centre-of-mass displacement of the deposit.

    Returns the drift components, its magnitude and the magnitude normalised by
    the grain radius and by the deposit's radial extent (``r_p995`` + ``radius``,
    the same quantity :func:`tensordem.granular_column.axisym.radial_extent`
    uses, recomputed here so that this module has no solver dependency).
    """
    p = _as_pos(pos)
    radius = float(radius)
    if not (radius > 0.0):
        raise ValueError("radius must be positive")
    cx = float(p[:, 0].mean())
    cy = float(p[:, 1].mean())
    drift = math.hypot(cx, cy)
    rr = np.hypot(p[:, 0], p[:, 1])
    r_outer = float(np.quantile(rr, 0.995)) + radius
    return {
        "com_x": cx,
        "com_y": cy,
        "drift": drift,
        "drift_over_radius": drift / radius,
        "drift_over_outer": drift / r_outer,
    }


def azimuthal_moments(pos, orders=(1, 2, 6)) -> dict:
    """Mass-weighted azimuthal moments of the deposit footprint.

    ``A_m = |sum_i r_i exp(i m theta_i)| / sum_i r_i`` with unit grain mass, so
    ``A_0 = 1`` by construction and ``A_m in [0, 1]``.  ``A_1`` measures
    lopsidedness, ``A_2`` ellipticity, ``A_6`` the hexagonal fingerprint of the
    initial HCP lattice.
    """
    p = _as_pos(pos)
    if any(int(m) < 1 for m in orders):
        raise ValueError("orders must be >= 1")
    r = np.hypot(p[:, 0], p[:, 1])
    theta = np.arctan2(p[:, 1], p[:, 0])
    total = float(r.sum())
    if total <= 0.0:
        raise ValueError("degenerate deposit: all grains on the axis")
    out = {}
    for m in orders:
        m = int(m)
        vec = np.sum(r * np.exp(1j * m * theta))
        out[f"a{m}"] = float(abs(vec) / total)
    out["r_mean"] = float(r.mean())
    out["r_p995"] = float(np.quantile(r, 0.995))
    return out


def azimuthal_shuffle_floor(pos, orders=(1, 2, 6), n_trials: int = 64,
                            seed: int = 20240101) -> dict:
    """Sampling floor of :func:`azimuthal_moments` under statistically axisymmetric azimuths.

    The null hypothesis is ``theta_i i.i.d. uniform`` with the *measured* radial
    distribution kept fixed, i.e. exactly the permutation test for "the deposit
    footprint carries no preferred direction".  ``mean``/``std``/``q95`` over
    ``n_trials`` shuffles are returned per order, together with the largest
    trial value (a conservative floor).
    """
    p = _as_pos(pos)
    if int(n_trials) < 2:
        raise ValueError("n_trials must be >= 2")
    r = np.hypot(p[:, 0], p[:, 1])
    total = float(r.sum())
    if total <= 0.0:
        raise ValueError("degenerate deposit: all grains on the axis")
    rng = np.random.default_rng(int(seed))
    theta0 = np.arctan2(p[:, 1], p[:, 0])
    samples = {int(m): np.empty(int(n_trials)) for m in orders}
    for k in range(int(n_trials)):
        theta = rng.permutation(theta0)
        for m in samples:
            samples[m][k] = abs(np.sum(r * np.exp(1j * m * theta))) / total
    floor = {}
    for m, vals in samples.items():
        floor[f"a{m}"] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "q95": float(np.quantile(vals, 0.95)),
            "max": float(vals.max()),
        }
    return floor


def azimuthal_uniform_floor(pos, orders=(1, 2, 6), n_trials: int = 64,
                           seed: int = 20240101) -> dict:
    """Sampling floor of :func:`azimuthal_moments` for a statistically axisymmetric footprint.

    The null hypothesis is ``theta_i i.i.d. uniform`` with the measured *radii*
    kept fixed (drawing fresh angles, **not** permuting the observed ones — a
    permutation keeps the angle multiset and therefore keeps any genuine
    anisotropy, which is why :func:`azimuthal_shuffle_floor` answers a different
    question: whether radius and angle are correlated).  ``mean``/``std``/``q95``/
    ``max`` over ``n_trials`` draws are returned per order.
    """
    p = _as_pos(pos)
    if int(n_trials) < 2:
        raise ValueError("n_trials must be >= 2")
    r = np.hypot(p[:, 0], p[:, 1])
    total = float(r.sum())
    if total <= 0.0:
        raise ValueError("degenerate deposit: all grains on the axis")
    rng = np.random.default_rng(int(seed))
    n = r.size
    samples = {int(m): np.empty(int(n_trials)) for m in orders}
    for k in range(int(n_trials)):
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        for m in samples:
            samples[m][k] = abs(np.sum(r * np.exp(1j * m * theta))) / total
    floor = {}
    for m, vals in samples.items():
        floor[f"a{m}"] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "q95": float(np.quantile(vals, 0.95)),
            "max": float(vals.max()),
        }
    return floor


def floor_clearance(pos, radius: float) -> dict:
    """Floor-contact sanity from the archived coordinates.

    A grain resting on the floor has ``z = radius`` up to the contact overlap
    ``delta``; ``min(z) - radius`` is therefore ``-max(delta)`` over the grains
    in floor contact.  A large negative clearance means grains have been pushed
    through the floor (solver defect), a positive one means nobody touches it.
    """
    p = _as_pos(pos)
    radius = float(radius)
    if not (radius > 0.0):
        raise ValueError("radius must be positive")
    clearance = p[:, 2] - radius
    return {
        "min_clearance_over_radius": float(clearance.min() / radius),
        "max_clearance_over_radius": float(clearance.max() / radius),
        "frac_below_minus_5pct": float(np.mean(clearance < -0.05 * radius)),
        "frac_below_minus_20pct": float(np.mean(clearance < -0.20 * radius)),
    }


def shape_factor(pos, radius: float, n_total: int | None = None) -> dict:
    """How close is the deposit to a cone of the same footprint?

    ``h_eq`` is the height the archived grains would occupy on a flat cylinder
    of radius ``r_outer = r_p995 + radius``; a cone of apex height ``H`` and base
    radius ``R`` has ``H = 3 h_eq``, a flat disc ``H = h_eq``.  Reported as a
    dimensionless shape indicator only (no reference value is claimed).
    """
    p = _as_pos(pos)
    radius = float(radius)
    if not (radius > 0.0):
        raise ValueError("radius must be positive")
    n = p.shape[0] if n_total is None else int(n_total)
    if n < p.shape[0]:
        raise ValueError("n_total must be >= the number of archived grains")
    rr = np.hypot(p[:, 0], p[:, 1])
    r_outer = float(np.quantile(rr, 0.995)) + radius
    solid = n * 4.0 / 3.0 * math.pi * radius ** 3
    h_eq = solid / (math.pi * r_outer ** 2)
    h_max = float(p[:, 2].max() + radius)
    return {
        "n_archived": int(p.shape[0]),
        "n_used_for_volume": int(n),
        "r_outer": r_outer,
        "h_max": h_max,
        "h_eq": h_eq,
        "cone_factor": h_max / h_eq,
    }


def decimated(pos, stride: int, offset: int = 0) -> np.ndarray:
    """Deterministic subsample ``pos[offset::stride]`` (archive-truncation model)."""
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    p = _as_pos(pos)
    if p.shape[0] < stride:
        raise ValueError("stride larger than the number of grains")
    sub = p[int(offset) % stride::stride]
    if sub.shape[0] == 0:                     # pragma: no cover - guarded above
        raise ValueError("stride larger than the number of grains")
    return sub


def decimation_bias(pos, radius: float, strides=(2, 4), orders=(1, 2)) -> dict:
    """Spread of the radial metrics induced by archiving only every ``stride``-th grain.

    For each stride every phase ``offset`` is used, so the reported spread is
    the full range of what an archive that keeps a regular subsample can show.
    ``delta_p995_over_outer`` is the metric error a subsampled archive would
    induce, in units of the deposit outer radius (i.e. the same units as the
    relative error quoted in the verdicts).
    """
    p = _as_pos(pos)
    radius = float(radius)
    rr = np.hypot(p[:, 0], p[:, 1])
    r_outer_full = float(np.quantile(rr, 0.995)) + radius
    full = azimuthal_moments(p, orders=orders)
    out = {}
    for stride in strides:
        stride = int(stride)
        if stride < 2:
            raise ValueError("strides must be >= 2")
        if p.shape[0] < 8 * stride:
            continue
        p995, mx, mom = [], [], {f"a{int(m)}": [] for m in orders}
        for offset in range(stride):
            sub = decimated(p, stride, offset)
            rs = np.hypot(sub[:, 0], sub[:, 1])
            p995.append(float(np.quantile(rs, 0.995)) + radius)
            mx.append(float(rs.max()) + radius)
            m = azimuthal_moments(sub, orders=orders)
            for key in mom:
                mom[key].append(m[key])
        p995 = np.asarray(p995)
        mx = np.asarray(mx)
        out[stride] = {
            "n_kept": int(p.shape[0] // stride),
            "p995_min": float(p995.min()),
            "p995_max": float(p995.max()),
            "delta_p995_over_outer": float((p995.max() - p995.min()) / r_outer_full),
            "max_delta_p995_over_outer": float(np.abs(p995 - r_outer_full).max() / r_outer_full),
            "delta_max_over_outer": float((mx.max() - mx.min()) / r_outer_full),
            "moment_delta": {k: float(max(v) - min(v)) for k, v in mom.items()},
            "moment_full": {k: float(full[k]) for k in mom},
        }
    return out


def metric_bundle(pos, radius: float, orders=(1, 2, 6),
                  n_total: int | None = None) -> dict:
    """All deposit diagnostics for one archived configuration."""
    out = {"com": horizontal_com_drift(pos, radius),
           "moments": azimuthal_moments(pos, orders=orders),
           "floor": floor_clearance(pos, radius),
           "shape": shape_factor(pos, radius, n_total=n_total)}
    return out

def edge_effective_count(pos, radius: float, quantile: float = 0.995) -> dict:
    """How many grains determine the radial runout metric.

    ``r_p995 = quantile(r, 0.995) + radius`` is an extreme order statistic; the
    grains within one grain radius of it are the ones that decide its value.
    ``n_eff`` (that count, normalised by the total) is the quantity that
    explains the metric's sampling noise: a metric set by a handful of loose
    edge grains cannot be certified to a percent.
    """
    p = _as_pos(pos)
    radius = float(radius)
    q = float(quantile)
    if not (0.5 <= q <= 1.0):
        raise ValueError("quantile must be in [0.5, 1]")
    rr = np.hypot(p[:, 0], p[:, 1])
    edge = float(np.quantile(rr, q))
    n_eff = int(np.count_nonzero(rr >= edge - radius))
    return {"r_quantile": edge, "n_eff": n_eff,
            "n_eff_fraction": n_eff / float(rr.size),
            "r_max": float(rr.max())}


def subsample_noise(pos, radius: float, fractions=(0.5, 0.25, 0.1),
                    n_trials: int = 200, seed: int = 9091,
                    quantile: float = 0.995) -> dict:
    """Sampling noise of ``r_p995`` under random archive sub-sampling.

    For each fraction the metric is recomputed on ``n_trials`` random subsets
    without replacement; ``std_over_outer`` is the 1-sigma of the metric in
    units of the deposit outer radius, i.e. the smallest error that a runout
    comparison made from an archive of that size can resolve.
    """
    p = _as_pos(pos)
    radius = float(radius)
    rr = np.hypot(p[:, 0], p[:, 1])
    outer = float(np.quantile(rr, quantile)) + radius
    rng = np.random.default_rng(int(seed))
    out = {"outer": outer}
    for frac in fractions:
        frac = float(frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError("fractions must be in (0, 1]")
        k = max(int(round(frac * rr.size)), 8)
        vals = np.empty(int(n_trials))
        for i in range(int(n_trials)):
            sub = rng.choice(rr, size=k, replace=False)
            vals[i] = float(np.quantile(sub, quantile)) + radius
        out[str(frac)] = {
            "n_kept": int(k),
            "mean": float(vals.mean()),
            "std_over_outer": float(vals.std(ddof=1) / outer),
            "p16_over_outer": float((np.quantile(vals, 0.16) - outer) / outer),
            "p84_over_outer": float((np.quantile(vals, 0.84) - outer) / outer),
        }
    return out
