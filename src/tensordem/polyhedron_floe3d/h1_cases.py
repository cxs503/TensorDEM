"""H1-BROKEN-ICE-DEM round-2 cases: floating equilibrium, rubble repose, clearing.

Why this module exists
----------------------
Anchor ``H1-BROKEN-ICE-DEM`` asks for three machine-checkable observables of a
broken-ice DEM:

1. **hydrostatic floating equilibrium** -- the relaxed draft of a freely floating
   floe as a fraction of its thickness must reproduce Archimedes'
   ``rho_ice / rho_water`` (tolerance 5 % here);
2. **rubble angle of repose** -- the settled heap of polyhedral ice rubble must be
   compared against the ice-rubble literature band (see
   ``docs/H1_BROKEN_ICE_DEM_ROUND2.md`` for the anchor and its provenance);
3. **channel clearing** -- a time series of the ice concentration in the corridor
   the hull has just cleared (the corpus of the "航道清空" demonstration).

The numerical machinery lives in :mod:`tensordem.polyhedron_floe3d`
(``dynamics`` / ``study``); this module only builds the cases, measures the
observables and returns JSON-serialisable dicts.  Nothing here is tuned to make a
target pass: every comparison value is either analytic (Archimedes) or external
(literature band), and the measured value is reported next to it.

Declared modelling choices (not measured constants)
---------------------------------------------------
* ``rho_ice = 917 kg/m3`` (typical first-year sea-ice bulk density),
  ``rho_water = 1025 kg/m3`` (sea water) -- the analytic target is their ratio.
* hydrodynamic load is the linear (Stokes-like) closure ``F = -c v`` with
  ``c = fraction * 2 m omega_0``; ``fraction`` sets how fast the floe relaxes to
  its hydrostatic draft (damping), *not* where that draft is.  The equilibrium
  draft is buoyancy-driven and independent of ``c``.
* ice-ice friction ``mu`` is a swept variable (0.2 / 0.4 / 0.6) and the reported
  repose angle is reported per ``mu``; no single ``mu`` is chosen post hoc.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .contact import ContactLaw
from .dynamics import DEMSystem3D, FloeBody, submerged_tet_moments
from .field import build_brash_ice_field
from .geometry import equivalent_disc_prism
from .study import case3_repose, run_channel

__all__ = [
    "SEA_ICE_DENSITY",
    "SEA_WATER_DENSITY",
    "buoyancy_equilibrium_case",
    "repose_case",
    "channel_clearing_case",
    "validate",
]

SEA_ICE_DENSITY = 917.0
SEA_WATER_DENSITY = 1025.0


# ---------------------------------------------------------------------------
# 1. hydrostatic floating equilibrium
# ---------------------------------------------------------------------------
def _submerged_fraction(system: DEMSystem3D, body: FloeBody) -> float:
    """Exact submerged volume fraction of one body (tetrahedral clipping)."""
    volume, _moment, _centroid = submerged_tet_moments(body.world_tets(),
                                                       system.water_level)
    return float(np.sum(volume)) / body.shape.volume


def buoyancy_equilibrium_case(*, footprint_area: float = 0.36,
                              thickness: float = 0.12, n_sides: int = 16,
                              rho_ice: float = SEA_ICE_DENSITY,
                              rho_water: float = SEA_WATER_DENSITY,
                              drag_fraction: float = 0.7,
                              dt: float = 5.0e-4,
                              max_steps: int = 6000,
                              settle_tol: float = 1.0e-4,
                              settle_window: int = 400,
                              windows: int = 40,
                              gravity: float = 9.81,
                              tolerance: float = 0.05,
                              label: str = "floe") -> Dict[str, Any]:
    """Release one prism floe at the waterline and let it relax to its draft.

    The floe is a regular ``n_sides``-gon prism of the requested footprint area and
    thickness with its centroid initially at ``z = 0`` (the waterline), i.e. 50 %
    submerged for this geometry; it then settles to the Archimedes draft

    .. math:: d = (rho_ice / rho_water) * thickness .

    Returns the relaxed submerged **volume** fraction (the shape-independent
    Archimedes quantity), the geometric draft and their relative errors, plus the
    settle history so the relaxation can be audited.
    """
    volume = footprint_area * thickness
    shape = equivalent_disc_prism(volume, thickness, n_sides)
    body = FloeBody(shape, rho_ice, position=[0.0, 0.0, 0.0], label=label)
    mass = body.mass
    # heave natural frequency of the waterplane: omega0 = sqrt(rho_w g A / m)
    omega0 = math.sqrt(rho_water * gravity * footprint_area / mass)
    linear_drag = drag_fraction * 2.0 * mass * omega0
    system = DEMSystem3D([body], law=ContactLaw(), gravity=gravity,
                         water_level=0.0, water_density=rho_water, dt=dt,
                         linear_drag=linear_drag, angular_drag=0.3 * linear_drag,
                         record=False)
    system.record_initial_energy()

    history: List[Dict[str, float]] = []
    quiet = 0
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        system.step()
        if steps % max(1, settle_window // 20) == 0:
            history.append({
                "t": float(system.time),
                "z": float(body.position[2]),
                "vz": float(body.velocity[2]),
                "submerged_fraction": _submerged_fraction(system, body),
            })
        if abs(float(body.velocity[2])) < settle_tol:
            quiet += 1
            if quiet >= settle_window:
                break
        else:
            quiet = 0

    # average over the last ``windows`` sampled states (a settled rest state, not a
    # single lucky step)
    tail = history[-int(windows):] if len(history) >= windows else history
    fraction = float(np.mean([h["submerged_fraction"] for h in tail]))
    z_settled = float(np.mean([h["z"] for h in tail]))
    v_drag = float(np.std([h["submerged_fraction"] for h in tail]))

    analytic_fraction = rho_ice / rho_water
    analytic_draft = analytic_fraction * thickness
    # geometric draft measured from the body: distance from the bottom face to z = 0
    bottom = float(body.world_vertices()[:, 2].min())
    draft = -bottom
    rel_err_fraction = abs(fraction - analytic_fraction) / analytic_fraction
    rel_err_draft = abs(draft - analytic_draft) / analytic_draft

    return {
        "case": "buoyancy_equilibrium",
        "setup": {
            "footprint_area_m2": footprint_area, "thickness_m": thickness,
            "n_sides": n_sides, "rho_ice": rho_ice, "rho_water": rho_water,
            "volume_m3": volume, "mass_kg": mass, "water_level_m": 0.0,
            "dt_s": dt, "linear_drag_N_s_per_m": linear_drag,
            "drag_fraction_of_critical": drag_fraction,
            "heave_omega0_rad_per_s": omega0,
            "gravity": gravity,
        },
        "ref": {
            "submerged_fraction_archimedes": analytic_fraction,
            "draft_archimedes_m": analytic_draft,
            "source": "Archimedes: rho_water g V_sub = rho_ice g V -> V_sub/V = "
                      "rho_ice/rho_water (independent of plan form)",
        },
        "got": {
            "submerged_fraction": fraction,
            "submerged_fraction_std_settled": v_drag,
            "draft_m": draft,
            "z_centroid_settled_m": z_settled,
            "steps": steps,
            "sim_time_s": float(system.time),
            "energy_residual": float(system.energy_residual()),
        },
        "error": {
            "rel_err_submerged_fraction": rel_err_fraction,
            "rel_err_draft": rel_err_draft,
        },
        "tolerance": tolerance,
        "passed": bool(rel_err_fraction <= tolerance and rel_err_draft <= tolerance),
        "history": history,
    }


# ---------------------------------------------------------------------------
# 2. rubble angle of repose
# ---------------------------------------------------------------------------
# literature anchor (see docs/H1_BROKEN_ICE_DEM_ROUND2.md section 2 for provenance)
REPOSE_BAND_DEG: tuple = (25.0, 40.0)


def repose_case(*, n_bodies: int = 45,
                friction: Sequence[float] = (0.2, 0.4, 0.6),
                kn: float = 2.0e5, zeta: float = 0.2, dt: float = 1.0e-4,
                n_steps: int = 6000, box_half: float = 0.5,
                cell_half: float = 0.13, density: float = SEA_ICE_DENSITY,
                seed: int = 5, band: Sequence[float] = REPOSE_BAND_DEG,
                settle_tol: Optional[float] = 0.01,
                settle_window: int = 400,
                shard_of: Optional[Sequence[float]] = None,
                shard_index: Optional[int] = None) -> Dict[str, Any]:
    """Rubble heap of polyhedral ice cells: repose angle per friction coefficient.

    ``n_steps`` is an upper bound: the run stops as soon as the heap is at rest
    (see :func:`~tensordem.polyhedron_floe3d.study.case3_repose`).  A case is only
    ``passed`` when every friction run settled *and* its angle lies inside the
    literature band; an unsettled run reports ``settled = False`` instead of an
    angle, which is how the round-1 failure is kept visible.
    """
    t0 = time.time()
    raw = case3_repose(n_bodies=n_bodies, friction=friction, kn=kn, zeta=zeta, dt=dt,
                       n_steps=n_steps, box_half=box_half, cell_half=cell_half,
                       density=density, seed=seed, settle_tol=settle_tol,
                       settle_window=settle_window)
    angles = [float(r["repose_angle_deg"]) for r in raw["runs"]]
    inside = [bool(band[0] <= a <= band[1]) for a in angles]
    settled = [bool(r.get("settled", False)) for r in raw["runs"]]
    return {
        "case": "repose",
        "setup": raw["setup"],
        "shard": {"of": list(shard_of) if shard_of else list(friction),
                  "index": shard_index},
        "ref": {
            "band_deg": [float(band[0]), float(band[1])],
            "source": "ice-rubble angle of repose literature band; provenance in "
                      "docs/H1_BROKEN_ICE_DEM_ROUND2.md section 2",
        },
        "runs": raw["runs"],
        "angles_deg": angles,
        "settled": settled,
        "inside_literature_band": inside,
        "all_runs_valid": raw["all_runs_valid"],
        "repose_angle_monotone_in_friction": raw["repose_angle_monotone_in_friction"],
        "runtime_s": time.time() - t0,
        "passed": bool(raw["all_runs_valid"] and all(inside)),
    }


# ---------------------------------------------------------------------------
# 3. channel clearing
# ---------------------------------------------------------------------------
def channel_clearing_case(*, n_floes: int = 24, median_thickness: float = 0.12,
                          speed: float = 0.4, mu: float = 0.35,
                          kn: float = 5.0e4, dt: float = 5.0e-4,
                          n_steps: int = 900, width: float = 1.0,
                          length: float = 2.4, corridor: float = 1.0,
                          seed: int = 0, ice_density: float = 900.0,
                          water_density: float = SEA_WATER_DENSITY) -> Dict[str, Any]:
    """Advance a hull through a brash-ice field and record the corridor concentration.

    The reported observable is the **ice concentration** (solid fraction) in the
    corridor of length ``corridor`` immediately ahead of the stem, sampled every
    step: it starts at the field concentration and drops as the hull pushes the
    floes out of the corridor.  ``concentration_half_time_s`` is the first time the
    concentration falls to the midpoint between its initial and final value.

    ``mu`` (ice-ice / ice-hull sliding friction, one value for both) is **declared**
    and not calibrated: it is the value used by the shipped channel study.
    """
    t0 = time.time()
    field = build_brash_ice_field(n_floes, box=(0.02, -0.5 * width, length, 0.5 * width),
                                  median_thickness=median_thickness, seed=seed,
                                  ice_density=ice_density, water_density=water_density)
    # the fixed swath starts at the ice-field leading edge and is one corridor long;
    # the hull starts ~1 cm behind it, so the window is "cleared" as the hull sweeps
    x0 = min(float(b.world_vertices()[:, 0].min()) for b in field.bodies)
    record = run_channel(field, kn=kn, mu=mu, dt=dt, speed=speed, n_steps=n_steps,
                         width=width, length=length, corridor=corridor, record_every=1,
                         fixed_window=(x0, x0 + corridor))
    times = np.asarray(record["time"], dtype=float)
    porosity = np.asarray(record["porosity"], dtype=float)
    concentration = 1.0 - porosity
    # the lane metric: solid fraction of the *fixed* swath the hull sweeps through
    conc_lane = 1.0 - np.asarray(record["porosity_fixed"], dtype=float)
    c0 = float(concentration[0])
    c1 = float(concentration[-1])
    half = 0.5 * (c0 + c1)
    below = np.flatnonzero(concentration <= half)
    half_time = float(times[below[0]]) if below.size else float("nan")
    resistance = np.asarray(record["resistance"], dtype=float)
    steady = resistance[len(resistance) // 3:] if resistance.size >= 3 else resistance
    return {
        "case": "channel_clearing",
        "setup": {
            "n_floes": int(len(field.bodies)), "median_thickness_m": median_thickness,
            "hull_speed_m_s": speed, "mu": mu, "kn": kn, "dt_s": dt,
            "n_steps": n_steps, "width_m": width, "length_m": length,
            "corridor_m": corridor, "seed": seed,
            "ice_density": ice_density, "water_density": water_density,
            "initial_field_porosity": float(field.porosity()),
            "friction_note": "mu = ice-ice = ice-hull sliding friction, declared "
                             "value of the shipped channel study (not calibrated here)",
        },
        "series": {
            "time_s": times.tolist(),
            "porosity": porosity.tolist(),
            "concentration": concentration.tolist(),
            "lane_concentration": conc_lane.tolist(),
            "resistance_N": resistance.tolist(),
            "hull_advance_m": float(record["hull_advance"][0]),
        },
        "summary": {
            "lane_concentration_initial": float(conc_lane[0]) if conc_lane.size else float("nan"),
            "lane_concentration_final": float(conc_lane[-1]) if conc_lane.size else float("nan"),
            "concentration_initial": c0,
            "concentration_final": c1,
            "concentration_half_time_s": half_time,
            "resistance_mean_steady_N": float(steady.mean()) if steady.size else float("nan"),
            "resistance_std_steady_N": float(steady.std(ddof=1)) if steady.size > 1 else float("nan"),
            "mass_residual": float(record["mass_residual"][0]),
            "energy_residual": float(record["energy_residual"][0]),
            "safety_clamps": int(record["clamps"][0]),
        },
        "runtime_s": time.time() - t0,
        "passed": bool(conc_lane.size > 1
                       and float(conc_lane[-1]) < float(conc_lane[0]) * 0.5
                       and float(record["mass_residual"][0]) < 1e-12
                       and float(record["energy_residual"][0]) < 0.05),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def validate(*, cases: Iterable[str] = ("buoyancy", "repose", "channel"),
             quick: bool = False, **kwargs: Any) -> Dict[str, Any]:
    """Run the requested H1 cases and return the JSON-serialisable result tree."""
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    out: Dict[str, Any] = {"started": started, "quick": bool(quick), "cases": {}}
    if "buoyancy" in cases:
        out["cases"]["buoyancy"] = buoyancy_equilibrium_case(**(kwargs.get("buoyancy") or {}))
    if "repose" in cases:
        rk = dict(kwargs.get("repose") or {})
        if quick:
            rk.setdefault("n_bodies", 27)
            rk.setdefault("n_steps", 1200)
            rk.setdefault("friction", (0.4,))
        out["cases"]["repose"] = repose_case(**rk)
    if "channel" in cases:
        ck = dict(kwargs.get("channel") or {})
        if quick:
            ck.setdefault("n_floes", 12)
            ck.setdefault("n_steps", 300)
        out["cases"]["channel"] = channel_clearing_case(**ck)
    out["runtime_s"] = time.time() - t0
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["passed"] = {name: bool(case.get("passed")) for name, case in out["cases"].items()}
    return out