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
        "got": {
            # the measured side of the L4 ref/got pair (ref = literature band)
            "angles_deg": angles,
            "angle_mean_deg": float(np.mean(angles)) if angles else float("nan"),
            "angle_min_deg": float(np.min(angles)) if angles else float("nan"),
            "angle_max_deg": float(np.max(angles)) if angles else float("nan"),
            "n_runs": len(angles),
            "n_settled": int(sum(settled)),
            "n_inside_band": int(sum(inside)),
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
                          n_steps: int = 6000, width: float = 1.0,
                          length: float = 2.4, corridor: float = 1.0,
                          seed: int = 0, ice_density: float = 900.0,
                          water_density: float = SEA_WATER_DENSITY,
                          outlet_margin: float = 3.0) -> Dict[str, Any]:
    """Advance a hull through a brash-ice field and record the corridor concentration.

    The reported observable is the **ice concentration** (solid fraction) of the
    *fixed* swath of length ``corridor`` that the hull sweeps through, sampled every
    step.  The lane is the physical question ("is the channel it passed through
    clear?"); a window that follows the stem only measures the ice being pushed.

    The concentration is the **exact clipped solid volume** of the floes inside the
    swath (:meth:`~tensordem.polyhedron_floe3d.dynamics.DEMSystem3D.solid_volume_slab`)
    divided by ``corridor * width * median_thickness`` -- the layer volume the swath
    would hold if it were full.  The earlier centroid-count proxy is kept in the
    series for continuity but is not used for the verdict: it can only change in
    steps, as a floe's centroid crosses a face.

    The run is only ``passed`` when the sweep is **complete** (``hull_advance >=
    corridor``), the lane concentration has dropped by more than 50 %, and the mass
    / energy ledgers stay closed.  A short run therefore reports an honest
    ``passed = False`` with ``sweep_complete = False`` instead of a partial drop
    being read as clearing.

    ``mu`` (ice-ice / ice-hull sliding friction, one value for both) is **declared**
    and not calibrated: it is the value used by the shipped channel study.
    """
    t0 = time.time()
    # how many steps the declared corridor actually needs: a step budget below
    # this cannot sweep the corridor, so ``sweep_complete`` would be a budget
    # artefact instead of a measurement.  Reported (not silently fixed) so a
    # caller that overrides ``n_steps`` sees the mismatch.
    steps_for_corridor = (int(math.ceil(corridor / (speed * dt)))
                          if speed > 0.0 and dt > 0.0 else 0)
    budget_sufficient = bool(steps_for_corridor > 0 and n_steps >= steps_for_corridor)
    field = build_brash_ice_field(n_floes, box=(0.02, -0.5 * width, length, 0.5 * width),
                                  median_thickness=median_thickness, seed=seed,
                                  ice_density=ice_density, water_density=water_density)
    # the fixed swath starts at the ice-field leading edge and is one corridor long;
    # the hull starts ~1 cm behind it, so the window is "cleared" as the hull sweeps
    x0 = min(float(b.world_vertices()[:, 0].min()) for b in field.bodies)
    record = run_channel(field, kn=kn, mu=mu, dt=dt, speed=speed, n_steps=n_steps,
                         width=width, length=length, corridor=corridor, record_every=1,
                         fixed_window=(x0, x0 + corridor),
                         slab_thickness=median_thickness, outlet_margin=outlet_margin)
    times = np.asarray(record["time"], dtype=float)
    porosity = np.asarray(record["porosity"], dtype=float)
    concentration = 1.0 - porosity
    # the lane metric: exact clipped solid fraction of the *fixed* swath
    slab_volume = corridor * width * median_thickness
    lane_solid = np.asarray(record["lane_solid_volume"], dtype=float)
    conc_lane = lane_solid / slab_volume if slab_volume > 0 else lane_solid
    conc_lane_proxy = 1.0 - np.asarray(record["porosity_fixed"], dtype=float)
    c0 = float(concentration[0])
    c1 = float(concentration[-1])
    half = 0.5 * (c0 + c1)
    below = np.flatnonzero(concentration <= half)
    half_time = float(times[below[0]]) if below.size else float("nan")
    # lane half-time: first time the swept swath is half as full as it started
    l0 = float(conc_lane[0]) if conc_lane.size else float("nan")
    lane_half = 0.5 * l0
    lane_below = np.flatnonzero(conc_lane <= lane_half)
    lane_half_time = float(times[lane_below[0]]) if lane_below.size else float("nan")
    advance = float(record["hull_advance"][0])
    resistance = np.asarray(record["resistance"], dtype=float)
    steady = resistance[len(resistance) // 3:] if resistance.size >= 3 else resistance
    criteria = {
        "sweep_budget_sufficient": budget_sufficient,
        "sweep_complete": bool(advance >= corridor),
        "lane_halved": bool(conc_lane.size > 1
                            and float(conc_lane[-1]) < float(conc_lane[0]) * 0.5),
        "mass_conserved": bool(float(record["mass_residual"][0]) < 1e-12),
        "energy_closed": bool(float(record["energy_residual"][0]) < 0.05),
    }
    return {
        "case": "channel_clearing",
        "setup": {
            "n_floes": int(len(field.bodies)), "median_thickness_m": median_thickness,
            "hull_speed_m_s": speed, "mu": mu, "kn": kn, "dt_s": dt,
            "n_steps": n_steps, "width_m": width, "length_m": length,
            "corridor_m": corridor, "outlet_margin_m": outlet_margin, "seed": seed,
            "steps_for_corridor": steps_for_corridor,
            "sweep_budget_ratio": (float(n_steps) / steps_for_corridor
                                   if steps_for_corridor else float("nan")),
            "lane_metric": "exact clipped solid volume / (corridor*width*median_thickness)",
            "ice_density": ice_density, "water_density": water_density,
            "initial_field_porosity": float(field.porosity()),
            "friction_note": "mu = ice-ice = ice-hull sliding friction, declared "
                             "value of the shipped channel study (not calibrated here)",
        },
        "ref": {
            "lane_concentration_target": 0.0,
            "lane_drop_threshold_fraction": 0.5,
            "mass_residual_tolerance": 1e-12,
            "energy_residual_tolerance": 0.05,
            "sweep_requirement": "hull_advance >= corridor",
            "source": "acceptance criteria of anchor H1-BROKEN-ICE-DEM (a swept lane must "
                      "end empty within the declared ledger tolerances); no experimental "
                      "resistance benchmark is claimed",
        },
        "series": {
            "time_s": times.tolist(),
            "porosity": porosity.tolist(),
            "concentration": concentration.tolist(),
            "lane_concentration": conc_lane.tolist(),
            "lane_concentration_centroid_proxy": conc_lane_proxy.tolist(),
            "lane_solid_volume_m3": lane_solid.tolist(),
            "resistance_N": resistance.tolist(),
            "hull_advance_m": advance,
        },
        "summary": {
            "lane_concentration_initial": l0,
            "lane_concentration_final": float(conc_lane[-1]) if conc_lane.size else float("nan"),
            "lane_concentration_ratio": (float(conc_lane[-1]) / l0) if l0 else float("nan"),
            "lane_concentration_half_time_s": lane_half_time,
            "sweep_complete": bool(advance >= corridor),
            "hull_advance_m": advance,
            "concentration_initial": c0,
            "concentration_final": c1,
            "concentration_half_time_s": half_time,
            "resistance_mean_steady_N": float(steady.mean()) if steady.size else float("nan"),
            "resistance_std_steady_N": float(steady.std(ddof=1)) if steady.size > 1 else float("nan"),
            "mass_residual": float(record["mass_residual"][0]),
            "energy_residual": float(record["energy_residual"][0]),
            "safety_clamps": int(record["clamps"][0]),
        },
        "got": {
            # the measured side of the L4 ref/got pair (ref = acceptance criteria)
            "lane_concentration_initial": l0,
            "lane_concentration_final": (float(conc_lane[-1]) if conc_lane.size
                                        else float("nan")),
            "lane_concentration_ratio": (float(conc_lane[-1]) / l0) if l0 else float("nan"),
            "lane_concentration_half_time_s": lane_half_time,
            "hull_advance_m": advance,
            "resistance_mean_steady_N": (float(steady.mean()) if steady.size
                                         else float("nan")),
            "mass_residual": float(record["mass_residual"][0]),
            "energy_residual": float(record["energy_residual"][0]),
        },
        "runtime_s": time.time() - t0,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
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
            # A smoke run must still *sweep* the swath, otherwise
            # ``sweep_complete`` is a step-budget artefact and the lane criterion
            # is never exercised (the pre-fix quick run advanced 0.06 m through a
            # 1.0 m corridor, i.e. 6 % of the swath, and could never pass).
            # Scale the corridor to 80 % of the distance the budget covers.
            reach = (float(ck["n_steps"]) * float(ck.get("dt", 5.0e-4))
                     * float(ck.get("speed", 0.4)))
            ck.setdefault("corridor", min(1.0, 0.8 * reach))
        out["cases"]["channel"] = channel_clearing_case(**ck)
    out["runtime_s"] = time.time() - t0
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["passed"] = {name: bool(case.get("passed")) for name, case in out["cases"].items()}
    return out