"""Cross-section convergence of the DEM3D direct-shear apparent modulus.

Why this module exists
----------------------
``docs/dem3d_shear_static_reference.md`` §5 closed the "apparent modulus"
WITHHELD with a rate-free static solve, but left one debt explicitly open: a
*shape-independent* material shear modulus could not be extracted, because the
trend ``G_eff/C44 = 0.844 (9x5) -> 0.939 (21x21)`` was too short a series to
decide whether the deficit is a lateral free-surface **boundary layer**
(``O(1/n)``, removable by extrapolation) or a residual bias of the measurement
itself.  The page named the remedy: "a cross-section convergence study with a
fixed ``h/a = 2`` and no fracture is the natural next step".

This module runs exactly that study.  For square cross-sections ``n x n`` at
``nz = 3`` (``h/a = 2``, no fracture, linear elastic) it records

* ``G_eff/C44`` from :func:`tensordem.shear_static.static_direct_shear`, plus the
  exact bond-census area ratio ``A_eff/A_nom`` of the same lattice, and
* the deficit ``d(n) = 1 - G_eff/C44``,

then fits the boundary-layer series

    d(n) = c1/n + c2/n**2 + ...            (deficit vanishes as n -> infinity)

with three nested models (``1/n``; ``1/n + 1/n^2``; plus an ``O(1)`` term whose
coefficient *is* the asymptotic bias).  The reported quantities are the observed
convergence order, the Richardson-extrapolated limit ``G_eff(inf)/C44`` (which
must equal 1 for ``C44 = 7k/(3a)`` to be recoverable), and the cross-section at
which the *direct* measurement enters the 3 % band.

:func:`surface_lag_profile` explains the mechanism: the displacement lag
``1 - u_x/(gamma h)`` of the mid layer, binned by distance from the lateral
faces, and its measured decay length in lattice spacings.

All quantities SI.  No fracture, no rate, no contact: this is the geometrically
linear elastic reference problem only.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from .dem3d import DEM3DConfig
from .shear_specimen import lattice_spacing, shear_modulus_closed_form
from .shear_static import static_direct_shear

__all__ = [
    "cross_section_series",
    "boundary_layer_fit",
    "surface_lag_profile",
]

#: Relative tolerance under which a measurement counts as converged (platform gate).
ACCURACY_GATE = 0.03


def _square_config(n: int, nz: int, config: DEM3DConfig | None) -> DEM3DConfig:
    if n < 3:
        raise ValueError("cross-section size n must be >= 3 (needs an interior node)")
    if nz < 3:
        raise ValueError("nz must be >= 3 (bottom and top layers plus an interior layer)")
    base = config or DEM3DConfig()
    return replace(base, nx=int(n), ny=int(n), nz=int(nz), drag=0.0, fix_x_edges=False)


def cross_section_series(
    sizes: Sequence[int],
    *,
    nz: int = 3,
    config: DEM3DConfig | None = None,
    gamma: float = 1.0e-3,
    gate: float = ACCURACY_GATE,
) -> dict[str, Any]:
    """Run the square cross-section series and fit the boundary-layer deficit.

    ``sizes`` must be increasing and contain at least four values, otherwise the
    three-point order estimate and the three-parameter bias model are unconstrained.
    ``nz`` is held fixed (``h/a = nz - 1``), so the only varied quantity is the
    lateral free-surface fraction ``~4/n``.
    """
    sizes = [int(n) for n in sizes]
    if len(sizes) < 4 or any(b <= a for a, b in zip(sizes, sizes[1:])):
        raise ValueError("sizes must be strictly increasing and contain >= 4 values")
    rows = []
    for n in sizes:
        cfg = _square_config(n, nz, config)
        result = static_direct_shear(cfg, gamma=gamma)
        c44 = shear_modulus_closed_form(cfg) if config is None else result["C44_Pa"]
        rows.append({
            "n": n,
            "nz": int(nz),
            "height_over_spacing": float(nz - 1),
            "particle_count": int(result["particle_count"]),
            "lateral_boundary_fraction": float(1.0 - (n - 2) ** 2 / n ** 2),
            "C44_Pa": float(c44),
            "G_nominal_over_C44": float(result["G_nominal_over_C44"]),
            "G_effective_over_C44": float(result["G_effective_over_C44"]),
            "deficit": float(1.0 - result["G_effective_over_C44"]),
            "area_ratio": float(result["area_ratio"]),
            "reaction_relative_error": float(result["reaction_relative_error"]),
            "free_residual_over_bond_force": float(result["free_residual_over_bond_force"]),
        })
    fit = boundary_layer_fit(sizes, [row["deficit"] for row in rows], gate=gate)
    return {
        "sizes": sizes,
        "nz": int(nz),
        "gamma": float(gamma),
        "lattice_spacing_m": lattice_spacing(config or DEM3DConfig()),
        "rows": rows,
        "fit": fit,
    }


def boundary_layer_fit(
    sizes: Sequence[int],
    deficits: Sequence[float],
    *,
    gate: float = ACCURACY_GATE,
) -> dict[str, Any]:
    """Least-squares boundary-layer fit of ``d(n) = 1 - G_eff/C44``.

    Nested models on ``x = 1/n``:

    * ``A``: ``d = c1 x``                    -- pure boundary layer, limit 0;
    * ``B``: ``d = c1 x + c2 x**2``          -- boundary layer + curvature, limit 0;
    * ``C``: ``d = c0 + c1 x + c2 x**2``     -- ``c0`` is the asymptotic bias.

    Model ``C`` is the falsification test: if the deficit were a measurement bias
    rather than a boundary layer, ``c0`` would be non-zero and no amount of extra
    cross-section would recover ``C44``.

    Also reported: the observed order from the three smallest deficits (finest
    three sizes) and a two-point Richardson limit with that order.
    """
    n = np.asarray(sizes, dtype=float)
    d = np.asarray(deficits, dtype=float)
    if n.shape != d.shape or n.size < 4:
        raise ValueError("need >= 4 (size, deficit) pairs")
    if not np.all(np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("deficits must be finite and non-negative")
    if np.any(np.diff(n) <= 0.0) or np.any(np.diff(d) >= 0.0):
        raise ValueError("sizes must increase and deficits must decrease")

    x = 1.0 / n
    models: dict[str, Any] = {}
    for name, powers in (("A", (1,)), ("B", (1, 2)), ("C", (0, 1, 2))):
        design = np.vstack([x ** p for p in powers]).T
        coef, *_ = np.linalg.lstsq(design, d, rcond=None)
        residual = d - design @ coef
        models[name] = {
            "powers": list(powers),
            "coefficients": {f"c{p}": float(c) for p, c in zip(powers, coef)},
            "rms_residual": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "dof": int(d.size - len(powers)),
        }

    # observed order from the three finest sizes: d = A n**-p
    d3, d2, d1 = d[-1], d[-2], d[-3]
    n3, n2, n1 = n[-1], n[-2], n[-3]
    ratio = (d1 - d2) / (d2 - d3) if d2 != d3 else float("nan")
    order = float(np.log(ratio) / np.log(n2 / n1)) if ratio > 0.0 else float("nan")

    # two-point Richardson limit of the deficit with the observed order
    r = n3 / n2
    asymptotic_deficit = float(d3 + (d3 - d2) / (r ** order - 1.0)) if np.isfinite(order) else float("nan")

    c1 = models["B"]["coefficients"]["c1"]
    c2 = models["B"]["coefficients"]["c2"]
    if c1 > 0.0:
        # positive root of c1/n + c2/n**2 = gate
        n_gate = (c1 + np.sqrt(c1 ** 2 + 4.0 * max(c2, 0.0) * gate)) / (2.0 * gate)
    else:
        n_gate = float("nan")

    measured = [(float(nn), float(dd)) for nn, dd in zip(n, d) if dd < gate]
    return {
        "gate": float(gate),
        "models": models,
        "observed_order": order,
        "richardson_asymptotic_deficit": asymptotic_deficit,
        "richardson_extrapolated_G_over_C44": (
            float(1.0 - asymptotic_deficit) if np.isfinite(asymptotic_deficit) else float("nan")),
        "asymptotic_deficit_model_C": models["C"]["coefficients"]["c0"],
        "extrapolated_G_over_C44_model_C": float(1.0 - models["C"]["coefficients"]["c0"]),
        "predicted_n_for_gate": float(n_gate),
        "largest_n_measured": int(n[-1]),
        "deficit_at_largest_n": float(d[-1]),
        "gate_met_by_measurement": bool(d[-1] < gate),
    }


def surface_lag_profile(
    config: DEM3DConfig | None = None,
    *,
    gamma: float = 1.0e-3,
    n: int | None = None,
) -> dict[str, Any]:
    """Near-surface displacement lag of the mid layer, in lattice spacings.

    The affine reference is ``u_x = gamma * z``.  The solved field lags behind it
    next to the lateral faces; the lag ``1 - u_x/(gamma h)`` at ``z = h/2`` is
    binned by distance from the nearest lateral face (in lattice spacings ``a``,
    measured as ``delta = u_x/(gamma z) - 1`` at the mid layer)
    and an exponential decay length ``lambda/a`` is fitted to the first few bins.

    ``n`` overrides ``config.nx`` (square cross-section).  ``nz`` must be odd so
    that a mid layer exists.
    """
    base = config or DEM3DConfig()
    if n is not None:
        base = replace(base, nx=int(n), ny=int(n), drag=0.0)
    cfg = replace(base, drag=0.0, fix_x_edges=False)
    if cfg.nx != cfg.ny:
        raise ValueError("surface_lag_profile needs a square cross-section (nx == ny)")
    if cfg.nz % 2 == 0:
        raise ValueError("surface_lag_profile needs an odd nz to have a mid layer")
    result = static_direct_shear(cfg, gamma=gamma, return_displacement=True)
    a = lattice_spacing(cfg)
    positions = np.asarray(result["initial_positions_m"])
    u = np.asarray(result["displacement_m"])
    height = float(cfg.nz - 1) * a
    gamma_h = gamma * height

    x = positions[:, 0]
    z = positions[:, 2]
    z_mid = float(np.median(np.unique(np.round(z, 12))))
    mid = np.isclose(z, z_mid)
    ix = np.rint((x - x.min()) / a).astype(int)
    # deviation from the affine field *at the same height*: delta = u_x/(gamma z) - 1
    z_safe = np.where(z > 0.0, z, np.nan)
    lag = u[:, 0] / (gamma * z_safe) - 1.0

    dist = np.minimum(np.minimum(ix, cfg.nx - 1 - ix), np.minimum(
        np.rint((positions[:, 1] - positions[:, 1].min()) / a).astype(int),
        cfg.ny - 1 - np.rint((positions[:, 1] - positions[:, 1].min()) / a).astype(int)))
    bins = []
    for k in sorted(set(dist[mid].tolist())):
        mask = mid & (dist == k)
        bins.append({
            "distance_over_spacing": float(k),
            "node_count": int(mask.sum()),
            "mean_lag": float(np.nanmean(lag[mask])),
            "max_abs_lag": float(np.nanmax(np.abs(lag[mask]))),
        })

    # exponential decay length from consecutive bins (log-linear regression)
    decay_length: float | None = None
    positive = [b for b in bins if b["distance_over_spacing"] >= 1.0 and b["mean_lag"] > 0.0][:3]
    if len(positive) >= 2:
        kk = np.array([b["distance_over_spacing"] for b in positive])
        ll = np.log(np.array([b["mean_lag"] for b in positive]))
        slope, _ = np.polyfit(kk, ll, 1)
        decay_length = float(-1.0 / slope) if slope < 0.0 else float("inf")

    return {
        "n": int(cfg.nx),
        "nz": int(cfg.nz),
        "gamma": float(gamma),
        "G_effective_over_C44": float(result["G_effective_over_C44"]),
        "mean_lag_overall": float(np.nanmean(lag[mid])),
        "lag_at_center": float(np.nanmean(lag[mid & (dist == dist.max())])),
        "bins": bins,
        "decay_length_over_spacing": decay_length,
    }