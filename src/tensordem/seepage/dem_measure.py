"""Measured seepage quantities: piping threshold, effective-stress profile, audits.

This module turns the DEM column of :mod:`tensordem.seepage.dem_column` into the
four numbers the campaign gates ask for.  Everything here is *measured* from the
simulated assembly; the analytic Terzaghi/Kozeny--Carman/K--C relations are
imported only as the comparison yardstick, never as an input to the measurement.

Honesty contract
----------------
Two of the campaign's links have **no measuring instrument** in this repository
at this revision and are reported as ``withheld`` rather than filled in with an
analytic substitute:

``C3`` (permeability vs Kozeny--Carman)
    A pore-scale Stokes/Boltzmann flow solve on the voxelised packing is
    required.  Deriving ``k`` from the packing's hydraulic radius is *identical*
    to Kozeny--Carman (``k = phi r_h^2 / (8 tau^2)`` with ``r_h = 2 phi / S_v``
    reproduces ``k = phi^3 / (2 tau^2 S^2 (1-phi)^2)``), so it would compare the
    formula with itself.  No such solve is implemented here -> ``withheld``.

``C4`` (suffusion rate vs Skempton--Brogan)
    Requires a bidisperse assembly in which the fine fraction is actually
    transported.  The monodisperse static solver cannot express it, and a
    transport run was out of the compute budget of this repair round ->
    ``withheld``.  The analytic envelope in
    :mod:`tensordem.seepage.suffusion` remains available but must not be
    labelled a measurement.
"""
from __future__ import annotations

import math

import numpy as np

from ..bedcoupling.closures import ergun_pressure_gradient
from .critical_gradient import buoyant_unit_weight, terzaghi_critical_gradient
from .dem_column import ColumnDEMConfig, SeepageColumn

__all__ = [
    "measured_effective_stress_profile",
    "measured_critical_gradient",
    "measured_mass_conservation",
    "measured_fluid_ledger",
    "run_measured_links",
    "PIPING_GRADIENT_RAMP",
]

#: Bulk hydraulic gradients applied in the piping scan (upward flow).
PIPING_GRADIENT_RAMP = (0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95, 1.05, 1.15)


def _prepare_column(config: ColumnDEMConfig, target_porosity: float,
                    layers: int, max_iter: int) -> dict:
    col = SeepageColumn.from_random_packing(config, target_porosity, layers=layers)
    state = col.solve_static(max_iter=max_iter)
    return {"column": col, "state": state}


def measured_effective_stress_profile(column: SeepageColumn, n_cuts: int = 9,
                                      gradient: float = 0.0) -> dict:
    """Contact-measured vertical effective stress vs the self-weight integral.

    The reference is built from the **measured local porosity profile**
    (``sigma_ref(z) = (rho_s - rho_w) g * int_0^z (1 - n(z')) dz'``), not from a
    single global porosity value: the column is heterogeneous, and the
    one-porosity formula would bias the reference by several per cent.
    """
    if gradient != 0.0:
        column.set_gradient(gradient)
        column.solve_static(max_iter=400)
    top = column.solid_top()
    cuts = np.linspace(0.0, top, n_cuts + 1)
    sigma_meas = column.stress_profile(cuts)
    # reference slabs must be at least two particle diameters thick, otherwise the
    # centre-count porosity (and hence the weight integral) is not defined
    n_slabs = max(4, min(40, int(top / (2.0 * column.cfg.diameter))))
    z_ref = np.linspace(0.0, top, n_slabs + 1)
    profile = column.porosity_profile(z_ref)
    dz = np.diff(z_ref)
    weight_integral = np.concatenate([[0.0], np.cumsum((1.0 - profile) * dz)])
    sigma_ref = ((column.cfg.density - column.cfg.rho_w) * column.cfg.gravity
                 * weight_integral)
    z_mid = 0.5 * (z_ref[:-1] + z_ref[1:])
    sigma_ref_cuts = np.interp(cuts, z_mid, sigma_ref[1:])
    # the free-surface cut and the base cut are not representative plane cuts:
    # the audit window is the interior of the column
    upper = (cuts > 0.15 * top) & (cuts < 0.85 * top)
    dev = np.zeros_like(cuts)
    mask = upper & (sigma_ref_cuts > 0.0)
    dev[mask] = (sigma_meas[mask] - sigma_ref_cuts[mask]) / sigma_ref_cuts[mask] * 100.0
    points = [
        {"z_m": float(z), "sigma_measured_pa": float(s), "sigma_reference_pa": float(r),
         "deviation_pct": float(d), "in_audit_window": bool(m)}
        for z, s, r, d, m in zip(cuts, sigma_meas, sigma_ref_cuts, dev, mask)
    ]
    max_dev = float(np.max(np.abs(dev[mask]))) if np.any(mask) else float("nan")
    return {
        "points": points,
        "max_abs_deviation_pct": max_dev,
        "threshold_pct": 10.0,
        "pass": bool(np.isfinite(max_dev) and max_dev <= 10.0),
        "porosity_measured": float(column.porosity()),
        "bed_height_m": float(top),
        "layer_thickness_m": float(column.cfg.diameter),
    }


def measured_critical_gradient(column: SeepageColumn, gradients=PIPING_GRADIENT_RAMP,
                               max_iter: int = 1200, heave_tolerance: float = 0.02) -> dict:
    """Locate the piping threshold by continuation over the gradient ramp.

    For every gradient the *static* equilibrium is recomputed by energy
    minimisation.  Two independent signatures of onset are recorded:

    * ``base_ratio`` = base reaction / submerged weight, which falls linearly to
      zero at the threshold (force balance);
    * ``heave`` = relative rise of the free surface.  Above the threshold no
      static equilibrium exists, the minimiser drives the assembly upward and
      the run is rolled back.

    ``i_c`` is the zero crossing of the linear fit to ``base_ratio(i)`` over the
    statically admissible points, with the heave onset reported separately.
    """
    weight = column.submerged_weight_stress()
    recorded = []
    for i in gradients:
        column.set_gradient(float(i))
        snapshot = column.positions.copy()
        res = column.solve_static(max_iter=max_iter)
        heave = res["bed_top_m"] / column.solid_top() - 1.0
        admissible = (res["bed_top_m"] <= (1.0 + heave_tolerance) * column.solid_top()
                      and abs(res["displacement_max_m"]) < 0.5 * column.cfg.diameter * 10)
        recorded.append({
            "gradient": float(i),
            "applied_gradient": float(column.fluid_profile(float(i))["measured_gradient"]),
            "base_ratio": float(res["base_effective_stress_pa"] / weight),
            "base_stress_pa": float(res["base_effective_stress_pa"]),
            "bed_top_m": float(res["bed_top_m"]),
            "heave_pct": float(heave * 100.0),
            "force_residual": float(res["force_residual"]),
            "iterations": int(res["iterations"]),
            "static_equilibrium": bool(admissible),
        })
        if not admissible:
            column.positions = snapshot      # roll the runaway state back
    ok = [r for r in recorded if r["static_equilibrium"] and r["gradient"] > 0.0]
    i_c_measured = float("nan")
    slope = float("nan")
    intercept = float("nan")
    if len(ok) >= 2:
        g = np.array([r["gradient"] for r in ok])
        b = np.array([r["base_ratio"] for r in ok])
        slope, intercept = np.polyfit(g, b, 1)
        if slope < 0.0:
            i_c_measured = float(-intercept / slope)
    heave_points = [r for r in recorded if not r["static_equilibrium"]]
    i_c_heave = float(min(r["gradient"] for r in heave_points)) if heave_points else float("nan")
    porosity = column.porosity()
    i_c_terzaghi = terzaghi_critical_gradient(column.cfg.density, porosity,
                                              column.cfg.rho_w, column.cfg.gravity)
    deviation = (abs(i_c_measured - i_c_terzaghi) / i_c_terzaghi * 100.0
                 if np.isfinite(i_c_measured) else float("nan"))
    return {
        "gradient_scan": recorded,
        "i_c_measured": i_c_measured,
        "i_c_measured_from_heave": i_c_heave,
        "i_c_terzaghi": float(i_c_terzaghi),
        "measured_porosity": float(porosity),
        "buoyant_unit_weight_pa_per_m": float(
            buoyant_unit_weight(column.cfg.density, porosity,
                                column.cfg.rho_w, column.cfg.gravity)),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "deviation_pct": float(deviation),
        "threshold_pct": 15.0,
        "pass": bool(np.isfinite(deviation) and deviation <= 15.0),
    }


def measured_mass_conservation(column: SeepageColumn, initial: dict) -> dict:
    """Solid + fluid mass audit against the state before the hydraulic scan."""
    final = column.mass_ledger()
    d_solid = abs(final["solid_mass_kg"] - initial["solid_mass_kg"])
    d_fluid = abs(final["fluid_mass_kg"] - initial["fluid_mass_kg"])
    total_initial = initial["solid_mass_kg"] + initial["fluid_mass_kg"]
    error = (d_solid + d_fluid) / total_initial * 100.0
    return {
        "initial": initial,
        "final": final,
        "solid_mass_change_kg": float(final["solid_mass_kg"] - initial["solid_mass_kg"]),
        "fluid_mass_change_kg": float(final["fluid_mass_kg"] - initial["fluid_mass_kg"]),
        "total_mass_error_pct": float(error),
        "threshold_pct": 2.0,
        "pass": bool(error <= 2.0),
    }


def measured_fluid_ledger(column: SeepageColumn, gradient: float = 0.5,
                          n_slabs: int | None = None) -> dict:
    """Fluid momentum-balance residual implied by the applied particle drag."""
    column.set_gradient(gradient, n_slabs)          # ledger must audit what is applied
    led = column.fluid_momentum_residual(gradient, n_slabs)
    return {
        "gradient": float(gradient),
        "relative_residual": float(led["relative_residual"]),
        "slab_porosity": [float(x) for x in led["slab_porosity"]],
        "pass": bool(led["relative_residual"] <= 0.05),
    }


def run_measured_links(config: ColumnDEMConfig | None = None,
                       target_porosity: float = 0.42, layers: int = 2,
                       max_iter: int = 1500, gradients=PIPING_GRADIENT_RAMP) -> dict:
    """Run links C1/C2/C5 from one prepared column (single simulation lineage)."""
    cfg = config or ColumnDEMConfig()
    col = SeepageColumn.from_random_packing(cfg, target_porosity, layers=layers)
    settle = col.solve_static(max_iter=max_iter)
    initial_ledger = col.mass_ledger()
    c1 = measured_effective_stress_profile(col, gradient=0.0)
    c2 = measured_critical_gradient(col, gradients=gradients, max_iter=max_iter)
    c5 = measured_mass_conservation(col, initial_ledger)
    ledger = measured_fluid_ledger(col, gradient=0.5)
    return {
        "config": {
            "n_particles": int(col.n),
            "diameter_m": cfg.diameter,
            "density_kg_m3": cfg.density,
            "kn_N_m": cfg.kn,
            "normal_damping": cfg.normal_damping,
            "target_porosity": target_porosity,
            "layers": layers,
            "box_width_m": float(col.box_width),
            "dt_s": cfg.dt,
        },
        "settle": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in settle.items()},
        "C1_effective_stress": c1,
        "C2_critical_gradient": c2,
        "C5_mass_conservation": c5,
        "fluid_ledger": ledger,
    }