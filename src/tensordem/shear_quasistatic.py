"""Quasi-static limit of the driven direct-shear specimen (rate convergence).

Why this module exists
----------------------
``docs/dem3d_shear_specimen.md`` reports the driven-specimen apparent shear
modulus as **WITHHELD** and states the reason: *"the ramp is 20 % from rate
convergence ... Reaching the quasi-static limit needs roughly one decade slower
ramps"*.  ``docs/dem3d_material_calibration_roadmap.md`` repeats the same item
under *"Still open: quasi-static apparent modulus of a driven block"*.  The
previous rate series only covered ``v = 0.4 ... 0.1 m/s`` (three points, all
faster than the specimen's own relaxation time), so it could not say what the
limit is, only that it had not been reached.

This module closes the measurement, without touching the solver:

* the two time scales that make the driven test rate dependent are written in
  closed form -- the lattice shear-wave transit ``t_w = h / c`` with
  ``c = a sqrt(k/m)`` and the viscous displacement-diffusion time
  ``t_d = drag * h^2 / (k a^2)`` that the earlier document identified.  A ramp
  is *quasi-static* only when ``t_ramp = gamma_max h / v`` exceeds both by a
  large margin, which is a machine-checkable criterion on (v, gamma_max, h, k).
* ``apparent_modulus_at_rate`` drives ``shear_specimen.run_direct_shear`` at a
  requested rate and fits the elastic slope on a **fixed engineering-strain
  window** (identical gamma range at every rate), so the rate series is
  apples-to-apples.
* ``fit_zero_rate_limit`` extrapolates ``G_app(v) = G0 + c v`` to ``v -> 0``.
  The linear term is the expected v dependence: the per-particle drag force
  ``-drag * v_p`` produces a viscous shear stress proportional to the applied
  shear rate ``gamma_dot = v / h``, so the measured modulus carries a
  ``O(v)`` bias whose size is set by ``drag``.
* ``drag_series`` and ``width_series`` test the two claims that remain after the
  extrapolation: that the rate bias is of viscous origin (intercept scales with
  ``drag``), and that the geometry-limited ``G_app`` relaxes onto the affine
  constant ``C44 = 7k/(3a)`` as the specimen gets wider (``a/L_x -> 0``).

Scope limits (do not read more into the numbers)
------------------------------------------------
* The zero-rate limit of *this* specimen is a boundary-value statement about a
  26-neighbour lattice block with a clamped base, a rigidly driven top and free
  lateral faces.  It is compared with the affine constant ``C44`` because the
  exact continuum solution of the infinite-width version of that problem *is*
  homogeneous simple shear, ``tau = C44 gamma``; finite width is corrected by
  the ``a/L_x -> 0`` extrapolation, not by assumption.
* Nothing here is a sea-ice material claim.  The specimen is 225 particles at
  ``k = 2000 N/m``; it audits the solver's own elastic response and the
  protocol, not ice.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Iterable, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D
from .shear_specimen import (
    apparent_shear_modulus_from_history,
    direct_shear_specimen_area,
    direct_shear_specimen_height,
    lattice_spacing,
    run_direct_shear,
    shear_modulus_closed_form,
)

__all__ = [
    "apparent_modulus_at_rate",
    "quasi_static_config",
    "drag_series",
    "fit_two_way_finite_size",
    "fit_zero_rate_limit",
    "height_series",
    "quasi_static_velocity",
    "lattice_wave_speed",
    "rate_sweep",
    "quasi_static_margins",
    "shear_diffusion_time",
    "shear_wave_transit",
    "width_series",
]


#: Reference specimen protocol of ``docs/dem3d_shear_specimen.md``: 9x5x5 simple
#: cubic block, per-particle viscous drag 10 N s/m, tensile threshold parked at
#: 1e3 so that the elastic ramp cannot trip the axial criterion (mode isolation).
_REFERENCE: dict[str, Any] = {
    "nx": 9, "ny": 5, "nz": 5, "drag": 10.0, "tool_gap": 20.0, "tool_speed": 1.0,
    "breaking_strain": 1.0e3, "shear_breaking_strain": 0.03,
}


def quasi_static_config(**overrides: Any) -> DEM3DConfig:
    """Reference direct-shear specimen with optional overrides."""
    values = dict(_REFERENCE)
    values.update(overrides)
    return DEM3DConfig(**values)


def lattice_wave_speed(config: DEM3DConfig) -> float:
    """Shear-wave speed of the simple-cubic spring lattice, ``c = a sqrt(k/m)``.

    A central-force bond network with one spring per bond transmits a transverse
    disturbance at ``a sqrt(k/m)`` (single-particle-per-cell chain result, exact
    in the long-wavelength limit).  With ``k = 2000 N/m``, ``m = 0.060018 kg``,
    ``a = 0.05 m`` this is ``9.13 m/s``.
    """
    a = lattice_spacing(config)
    return a * math.sqrt(config.bond_stiffness / config.mass)


def shear_wave_transit(config: DEM3DConfig) -> float:
    """Time for a shear wave to cross the specimen height, ``h / c``."""
    return direct_shear_specimen_height(config) / lattice_wave_speed(config)


def shear_diffusion_time(config: DEM3DConfig, *, drag: float | None = None) -> float:
    """Viscous displacement-diffusion time ``drag * h^2 / (k a^2)``.

    Obtained by balancing the drag force ``drag * u / t`` on a particle against
    the bond restoring force ``k a^2 * d^2u/dy^2 ~ k a^2 u / h^2``; it is the
    same estimate as in ``docs/dem3d_shear_specimen.md`` (0.08 s for the shipped
    parameters).  With ``drag -> 0`` the diffusion time vanishes and the response
    is limited by the wave transit instead.
    """
    a = lattice_spacing(config)
    height = direct_shear_specimen_height(config)
    damping = config.drag if drag is None else float(drag)
    return damping * height * height / (config.bond_stiffness * a * a)


def quasi_static_margins(
    config: DEM3DConfig, *, velocity: float, gamma_max: float
) -> dict[str, float]:
    """Dimensionless margins that decide whether a ramp is quasi-static.

    ``ramp_time`` must be large compared with *both* the wave transit (the load
    has to reach the bottom of the block) and the diffusion time (the viscous
    drag has to let the strain field settle).  ``transits`` and
    ``diffusion_times`` are those two ratios; ``quasi_static`` is the honest
    conjunction ``transits >= 20 and diffusion_times >= 5``, which is the
    criterion used to select the converged end of the rate series.
    """
    if velocity <= 0.0 or gamma_max <= 0.0:
        raise ValueError("velocity and gamma_max must be positive")
    height = direct_shear_specimen_height(config)
    t_ramp = gamma_max * height / velocity
    t_wave = shear_wave_transit(config)
    t_diff = shear_diffusion_time(config)
    transits = t_ramp / t_wave
    diffusion_times = t_ramp / t_diff if t_diff > 0.0 else math.inf
    return {
        "ramp_time_s": t_ramp,
        "wave_transit_s": t_wave,
        "diffusion_time_s": t_diff,
        "transits": transits,
        "diffusion_times": diffusion_times,
        "shear_rate_per_s": velocity / height,
        "quasi_static": bool(transits >= 20.0 and diffusion_times >= 5.0),
    }


def quasi_static_velocity(
    config: DEM3DConfig,
    gamma_max: float,
    *,
    min_transits: float = 20.0,
    min_diffusion_times: float = 5.0,
) -> float:
    """Fastest ramp that still satisfies the quasi-static criterion.

    The ramp time must exceed ``min_transits * t_w`` (the load has to traverse the
    block many times) *and* ``min_diffusion_times * t_d`` (the viscous
    displacement diffusion has to keep up).  Since ``t_ramp = gamma_max h / v``,

        v <= gamma_max h / max(min_transits * t_w, min_diffusion_times * t_d),

    and both bounds shrink with height, so a taller specimen must be driven
    slower.  This helper is what makes a height series affordable: it picks the
    *cheapest* admissible rate instead of a fixed one.
    """
    if gamma_max <= 0.0:
        raise ValueError("gamma_max must be positive")
    height = direct_shear_specimen_height(config)
    budget = max(min_transits * shear_wave_transit(config),
                 min_diffusion_times * shear_diffusion_time(config))
    return gamma_max * height / budget


def apparent_modulus_at_rate(
    config: DEM3DConfig | None = None,
    *,
    velocity: float,
    gamma_max: float = 0.025,
    window_fraction: tuple[float, float] = (0.80, 0.96),
    sample_target: int = 200,
    keep_history: bool = False,
) -> dict[str, Any]:
    """Drive the specimen at one rate and fit ``G_app`` on a fixed strain window.

    The window is specified as a *fraction of* ``gamma_max`` (default
    ``[0.020, 0.024]`` for ``gamma_max = 0.025``), so every member of a rate
    series is fitted over the same engineering-strain range: differences between
    the runs are then rate effects, not fit-window effects.
    """
    base = config or quasi_static_config()
    config = replace(base, fix_bottom=True, fix_x_edges=False)
    a = lattice_spacing(config)
    height = direct_shear_specimen_height(config)
    steps_estimate = max(int(round(gamma_max * height / (velocity * _dt_of(config)))),
                         1)
    result = run_direct_shear(
        config,
        velocity=velocity,
        gamma_max=gamma_max,
        sample_every=max(1, steps_estimate // sample_target),
        record_histories=True,
    )
    window = (window_fraction[0] * gamma_max, window_fraction[1] * gamma_max)
    fit = apparent_shear_modulus_from_history(result["history"], *window)
    # Fit-window sensitivity from the *same* history: a converged ramp must not
    # care where in the intact range the slope is taken.
    window_fits: dict[str, dict[str, float]] = {}
    for tag, fractions in (("early", (0.40, 0.60)), ("mid", (0.60, 0.80)),
                           ("late", (0.80, 0.96))):
        lo, hi = fractions[0] * gamma_max, fractions[1] * gamma_max
        try:
            alt = apparent_shear_modulus_from_history(result["history"], lo, hi)
        except ValueError:
            continue
        window_fits[tag] = {
            "window": [lo, hi],
            "apparent_shear_modulus_Pa": alt["apparent_shear_modulus_Pa"],
            "relative_residual": alt["relative_residual"],
        }
    slopes = [w["apparent_shear_modulus_Pa"] for w in window_fits.values()]
    window_spread = (max(slopes) - min(slopes)) / abs(fit["apparent_shear_modulus_Pa"]) \
        if slopes and fit["apparent_shear_modulus_Pa"] else math.inf
    c44 = shear_modulus_closed_form(config)
    margins = quasi_static_margins(config, velocity=velocity, gamma_max=gamma_max)
    broken = int(result["final"]["broken_bonds"])
    return {
        "velocity_m_per_s": velocity,
        "gamma_max": gamma_max,
        "window": list(window),
        "steps": int(result["protocol"]["steps"]),
        "dt_s": result["protocol"]["dt_s"],
        "particle_count": int(result["protocol"]["particle_count"]),
        "specimen_height_m": height,
        "specimen_area_m2": direct_shear_specimen_area(config),
        "apparent_shear_modulus_Pa": fit["apparent_shear_modulus_Pa"],
        "intercept_Pa": fit["intercept_Pa"],
        "intercept_per_drag_Pa": fit["intercept_Pa"] / config.drag if config.drag else math.inf,
        "fit_samples": fit["sample_count"],
        "fit_relative_residual": fit["relative_residual"],
        "C44_Pa": c44,
        "ratio_to_C44": fit["apparent_shear_modulus_Pa"] / c44,
        "broken_bonds": broken,
        "drag_Ns_per_m": config.drag,
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "lattice_spacing_m": a,
        "inverse_width_a_over_Lx": 1.0 / (config.nx * a),
        "inverse_height_a_over_h": a / height,
        "margins": margins,
        "window_fits": window_fits,
        "window_spread_relative": window_spread,
        "history": result["history"] if keep_history else None,
    }


def _dt_of(config: DEM3DConfig) -> float:
    """Solver time step without building a simulation (config guard rule)."""
    if config.dt is not None:
        return float(config.dt)
    return float(IceDEM3D(config).dt)


def rate_sweep(
    config: DEM3DConfig | None = None,
    *,
    velocities: Sequence[float],
    gamma_max: float = 0.025,
    window_fraction: tuple[float, float] = (0.80, 0.96),
) -> list[dict[str, Any]]:
    """Fitted apparent modulus for each ramp speed, slowest last."""
    base = config or quasi_static_config()
    rows = [
        apparent_modulus_at_rate(base, velocity=v, gamma_max=gamma_max,
                                 window_fraction=window_fraction)
        for v in sorted(velocities, reverse=True)
    ]
    for row in rows:
        if row["broken_bonds"]:
            raise ValueError("rate series ran into bond failure; lower gamma_max")
    return rows


def fit_zero_rate_limit(
    rows: Iterable[dict[str, Any]], *, points: int = 3
) -> dict[str, Any]:
    """Least-squares extrapolation of ``G_app(v) = G0 + c v`` to ``v -> 0``.

    ``points`` selects how many of the *slowest* rates enter the fit; the
    returned record also carries the fit for one fewer point, so the stability of
    the intercept under the choice of window is visible instead of asserted.
    """
    ordered = sorted(rows, key=lambda r: r["velocity_m_per_s"])
    if len(ordered) < 2:
        raise ValueError("need at least two rates to extrapolate")
    use = ordered[: max(2, min(points, len(ordered)))]
    v = [r["velocity_m_per_s"] for r in use]
    g = [r["apparent_shear_modulus_Pa"] for r in use]
    n = float(len(use))
    mean_v = sum(v) / n
    mean_g = sum(g) / n
    sxx = sum((vi - mean_v) ** 2 for vi in v)
    sxy = sum((vi - mean_v) * (gi - mean_g) for vi, gi in zip(v, g))
    slope = sxy / sxx
    intercept = mean_g - slope * mean_v
    residuals = [gi - (intercept + slope * vi) for vi, gi in zip(v, g)]
    worst = max(abs(r) for r in residuals)
    out: dict[str, Any] = {
        "points_used": len(use),
        "velocities_m_per_s": v,
        "G0_Pa": intercept,
        "slope_Pa_per_m_per_s": slope,
        "max_abs_residual_Pa": worst,
        "relative_residual": worst / abs(intercept) if intercept else math.inf,
        "slowest_ratio_to_G0": ordered[0]["apparent_shear_modulus_Pa"] / intercept
        if intercept else math.inf,
        "C44_Pa": ordered[0]["C44_Pa"],
        "G0_over_C44": intercept / ordered[0]["C44_Pa"],
    }
    out["G0_over_C44_relative_error"] = abs(out["G0_over_C44"] - 1.0)
    if len(use) > 2:
        out["stability"] = fit_zero_rate_limit(use, points=len(use) - 1)["G0_Pa"]
        out["stability_relative_change"] = abs(out["stability"] - intercept) / abs(intercept)
    return out


def drag_series(
    config: DEM3DConfig | None = None,
    *,
    drags: Sequence[float],
    velocity: float,
    gamma_max: float = 0.025,
    window_fraction: tuple[float, float] = (0.80, 0.96),
    keep_history: bool = False,
) -> list[dict[str, Any]]:
    """Same ramp at fixed speed for several drag coefficients.

    A viscous mechanism makes the *intercept* (the part of the traction that is
    proportional to the shear rate) scale with ``drag`` while the fitted slope is
    left alone once the ramp is slow enough; a mechanism that lives in the
    lattice inertia instead would move the slope.
    """
    base = config or quasi_static_config()
    rows = []
    for drag in sorted(drags, reverse=True):
        per_drag = replace(base, drag=float(drag))
        row = apparent_modulus_at_rate(per_drag, velocity=velocity,
                                       gamma_max=gamma_max,
                                       window_fraction=window_fraction,
                                       keep_history=keep_history)
        rows.append(row)
    return rows


def width_series(
    config: DEM3DConfig | None = None,
    *,
    nx_values: Sequence[int],
    velocity: float,
    gamma_max: float = 0.025,
    window_fraction: tuple[float, float] = (0.80, 0.96),
    keep_history: bool = False,
) -> list[dict[str, Any]]:
    """Same ramp at fixed (slow) speed for several specimen widths ``nx``."""
    base = config or quasi_static_config()
    rows = []
    for nx in sorted(int(n) for n in nx_values):
        per_width = replace(base, nx=nx)
        row = apparent_modulus_at_rate(per_width, velocity=velocity,
                                       gamma_max=gamma_max,
                                       window_fraction=window_fraction,
                                       keep_history=keep_history)
        rows.append(row)
    return rows


def height_series(
    config: DEM3DConfig | None = None,
    *,
    nz_values: Sequence[int],
    gamma_max: float = 0.025,
    velocities: Sequence[float] | None = None,
    keep_history: bool = False,
) -> list[dict[str, Any]]:
    """Converged modulus for several specimen heights ``nz`` at fixed width.

    The ramp speed is chosen per height by ``quasi_static_velocity`` unless
    explicit ``velocities`` are supplied: ``t_d`` grows with ``h^2`` and the wave
    transit with ``h``, so a single speed cannot be admissible for all heights
    (this is exactly why the earlier document's height series was rate biased).
    """
    base = config or quasi_static_config()
    heights = sorted(int(n) for n in nz_values)
    if velocities is None:
        velocities = [quasi_static_velocity(replace(base, nz=nz), gamma_max)
                      for nz in heights]
    if len(velocities) != len(heights):
        raise ValueError("velocities must match nz_values")
    rows = []
    for nz, velocity in zip(heights, velocities):
        per_height = replace(base, nz=nz)
        row = apparent_modulus_at_rate(per_height, velocity=float(velocity),
                                       gamma_max=gamma_max,
                                       keep_history=keep_history)
        rows.append(row)
    return rows


def fit_two_way_finite_size(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Two-stage finite-size extrapolation ``c0 - alpha (a/Lx) - beta (a/h)``.

    A single specimen has two free surfaces: the lateral faces unload a fraction
    ``~ a/Lx`` of the shear traction and the clamped base / driven platen soften a
    fraction ``~ a/h``.  The two effects are separated sequentially, because a
    single three-parameter least-squares fit on a small grid is ill conditioned
    (the ``a/Lx`` and ``a/h`` columns are nearly collinear for two heights):

    1. for every height that has at least two widths, extrapolate linearly in
       ``a/Lx`` to infinite width (``fit_infinite_width``);
    2. extrapolate those width-free values linearly in ``a/h`` to infinite
       height.  The intercept is the doubly infinite-size modulus.

    Returns the coefficients, the residual of the *combined* model over all
    supplied points, and the grid actually used.
    """
    points = list(rows)
    if len(points) < 3:
        raise ValueError("need at least three (width, height) points")
    by_height: dict[int, list[dict[str, Any]]] = {}
    for row in points:
        by_height.setdefault(int(row["nz"]), []).append(row)
    levels = []
    for nz, group in sorted(by_height.items()):
        if len(group) < 2:
            continue
        fit = fit_infinite_width(group)
        a = group[0]["lattice_spacing_m"]
        height = group[0]["specimen_height_m"]
        levels.append({"nz": nz, "widths": sorted(int(r["nx"]) for r in group),
                       "inverse_height_a_over_h": a / height,
                       "G_infinite_width_Pa": fit["G_infinite_width_Pa"],
                       "width_slope_Pa_m": fit["slope_Pa_m"],
                       "width_residual": fit["relative_residual"]})
    if len(levels) < 2:
        raise ValueError("need widths at two heights for the two-stage fit")
    x = [level["inverse_height_a_over_h"] for level in levels]
    y = [level["G_infinite_width_Pa"] for level in levels]
    n = float(len(levels))
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    beta_pa = sxy / sxx
    c0_pa = mean_y - beta_pa * mean_x
    c44 = points[0]["C44_Pa"]
    alpha_pa = sum(level["width_slope_Pa_m"] for level in levels) / n
    residuals = []
    for row in points:
        a = row["lattice_spacing_m"]
        predicted = (c0_pa - alpha_pa * (a / (row["nx"] * a))
                     - beta_pa * (a / row["specimen_height_m"]))
        residuals.append(row["apparent_shear_modulus_Pa"] - predicted)
    worst = max(abs(r) for r in residuals) / c44
    return {
        "point_count": len(points),
        "c0_G_over_C44_at_infinite_size": c0_pa / c44,
        "alpha_lateral": alpha_pa / c44,
        "beta_height": beta_pa / c44,
        "C44_Pa": c44,
        "G_infinite_Pa": c0_pa,
        "max_abs_residual_over_C44": worst,
        "relative_error_vs_C44": abs(c0_pa / c44 - 1.0),
        "height_levels": levels,
        "grid": sorted({(int(r["nx"]), int(r["nz"])) for r in points}),
    }


def fit_infinite_width(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extrapolate the converged ``G_app`` of a width series to ``a/L_x -> 0``.

    The exact continuum solution of the infinite-width version of the driven
    block problem is homogeneous simple shear (``tau = C44 gamma``), so the
    width extrapolation is the honest route to the affine constant: side-free
    relaxation decays with the inverse aspect ratio.
    """
    ordered = sorted(rows, key=lambda r: r["inverse_width_a_over_Lx"])
    if len(ordered) < 2:
        raise ValueError("need at least two widths")
    x = [r["inverse_width_a_over_Lx"] for r in ordered]
    y = [r["apparent_shear_modulus_Pa"] for r in ordered]
    n = float(len(ordered))
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    worst = max(abs(r) for r in residuals)
    c44 = ordered[0]["C44_Pa"]
    return {
        "widths_nx": [r["nx"] for r in ordered],
        "G_infinite_width_Pa": intercept,
        "slope_Pa_m": slope,
        "max_abs_residual_Pa": worst,
        "relative_residual": worst / abs(intercept) if intercept else math.inf,
        "C44_Pa": c44,
        "G_infinite_width_over_C44": intercept / c44,
        "relative_error_vs_C44": abs(intercept / c44 - 1.0),
    }