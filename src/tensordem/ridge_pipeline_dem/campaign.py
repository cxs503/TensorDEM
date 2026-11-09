"""Campaign driver: ridge-keel grounding on a buried pipeline, DEM vs theory.

Node ``p54w3-n06`` asks for one quantitative answer to four questions and five
machine-checkable acceptance criteria:

(a) does the DEM vertical ice load on a buried pipeline reproduce the ISO
    19906-form bearing capacity within 20 %?
(b) does the DEM lateral (break-out) load reproduce the ``N_p``-form lateral
    capacity within 25 %?
(c) is the DEM cover-instability verdict (pipe settlement beyond ``D/10``)
    consistent with the theoretical demand-vs-capacity prediction?
(d) does the coupled energy ledger close within 5 %?
(e) is the peak load grid independent within 5 % under particle refinement?

The physics lives in the existing worktree modules
(:mod:`tensordem.ridge_dem`, :mod:`tensordem.ridge_geotech`,
:mod:`tensordem.ridge_grounding`, :mod:`tensordem.ridge_iso19906`); this module
only builds the cases, runs them, and adjudicates the criteria.  Nothing is
hand-copied: every number in the campaign payload is produced by the functions
below and serialised by ``scripts/p54w3_n06_campaign.py``.

Honest scope of the comparison (stated here as well as in the report): the
compliant-pipe peak load is reached while the pipe is pressing on the hyperbolic
subgrade, whose *ultimate* pressure is the same bearing-capacity theory the ISO
form uses.  Agreement within 20 % therefore shows that the coupled chain
mobilises the theoretical capacity (and that the ice can supply it), not that
the bearing-capacity theory itself has been independently validated by DEM.
The `rigid` reference run, where the pipe cannot yield, is reported next to it:
that load is purely DEM-generated and shows whether the ice supply or the
soil capacity governs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Iterable, Mapping, Sequence

import torch

from ..ridge_dem import RidgeKeel, RidgeMaterial
from ..ridge_geotech import MohrCoulombSoil, PipeSection
from ..ridge_grounding import (GroundingScenario, run_grounding,
                               load_displacement_metrics)
from ..ridge_iso19906 import (iso19906_vertical_load, iso19906_lateral_load,
                              compare_with_dem, cover_envelope, instability_limits)
from .energy import closure_from_track, closure_scan

__all__ = [
    "PIPE_DIAMETER",
    "PIPE_WALL",
    "PIPE_YIELD",
    "PIPE_MODULUS",
    "PipelineCase",
    "undrained_clay",
    "rubble_material",
    "pipe_section",
    "case_scenario",
    "run_case",
    "vertical_study",
    "lateral_scenario_gap",
    "lateral_study",
    "grid_study",
    "cover_study",
    "dt_refinement_cases",
    "energy_study",
    "evaluate_acceptance",
    "ACCEPTANCE_THRESHOLDS",
]

PIPE_DIAMETER = 0.914
"""Outer diameter of a 36-inch export pipeline, m."""
PIPE_WALL = 0.0159
PIPE_YIELD = 448.0e6
PIPE_MODULUS = 207.0e9

ACCEPTANCE_THRESHOLDS = {
    "vertical_deviation": 0.20,
    "lateral_deviation": 0.25,
    "energy_closure": 0.05,
    "grid_change": 0.05,
}
"""Machine thresholds of the campaign node (dimensionless)."""


def undrained_clay(su: float = 30.0e3) -> MohrCoulombSoil:
    """Saturated undrained clay used for every campaign case."""
    return MohrCoulombSoil(friction_angle_deg=0.0, cohesion=su,
                           saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                           poisson_ratio=0.3, water_unit_weight=10.06e3,
                           undrained=True)


def rubble_material(particle_radius: float = 0.5, *,
                    freeze_bond_fraction: float = 1.0) -> RidgeMaterial:
    """Freeze-bonded rubble micro-parameters (same set as the W2 verification)."""
    return RidgeMaterial(particle_radius=particle_radius, drag=2.0e4,
                         freeze_bond_fraction=freeze_bond_fraction,
                         bond_tensile_strength=100.0e3,
                         bond_shear_strength=200.0e3)


def pipe_section() -> PipeSection:
    return PipeSection(PIPE_DIAMETER, PIPE_WALL, PIPE_MODULUS, PIPE_YIELD)


@dataclass(frozen=True)
class PipelineCase:
    """One grounding case of the campaign.

    ``stroke`` is generous enough for the keel tip to travel from its start
    clearance through the soil cover and onto the pipe crown, so every case
    reaches the same physical stage regardless of the time step.
    """

    name: str
    cover_depth: float = 0.5
    su: float = 30.0e3
    keel_depth: float = 6.0
    keel_width: float = 12.0
    keel_porosity: float = 0.10
    particle_radius: float = 0.5
    platen_velocity: float = 0.40
    tip_clearance: float = 0.15
    initial_penetration: float = 0.0
    keel_offset_x: float = 0.0
    stroke: float | None = None
    loading: str = "vertical"
    pipe_response: str = "compliant"
    pipe_lateral_compliance: bool = False
    steps_factor: float = 1.0
    dt: float | None = None
    freeze_bond_fraction: float = 1.0
    max_steps: int = 60000

    def __post_init__(self) -> None:
        if self.cover_depth < 0.0 or not math.isfinite(self.cover_depth):
            raise ValueError("cover_depth must be finite and nonnegative")
        if self.su <= 0.0:
            raise ValueError("su must be positive")
        if self.loading not in ("vertical", "lateral"):
            raise ValueError("loading must be 'vertical' or 'lateral'")
        if self.pipe_response not in ("compliant", "rigid"):
            raise ValueError("pipe_response must be 'compliant' or 'rigid'")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.initial_penetration < 0.0 or not math.isfinite(self.initial_penetration):
            raise ValueError("initial_penetration must be finite and nonnegative")

    @property
    def stroke_used(self) -> float:
        if self.stroke is not None:
            return self.stroke
        # tip clearance + cover depth + 0.6 D of travel past the pipe crown: the
        # keel must reach the crown and load it well past the peak, while the
        # post-peak branch stays well inside the record.
        return self.tip_clearance + self.cover_depth + 0.6 * PIPE_DIAMETER


def case_scenario(case: PipelineCase) -> GroundingScenario:
    """Build the :class:`GroundingScenario` of one campaign case."""
    return GroundingScenario(
        keel=RidgeKeel(case.keel_depth, case.keel_width, case.keel_porosity),
        material=rubble_material(case.particle_radius,
                                 freeze_bond_fraction=case.freeze_bond_fraction),
        soil=undrained_clay(case.su),
        section=pipe_section(),
        cover_depth=case.cover_depth,
        tip_clearance=case.tip_clearance,
        initial_penetration=case.initial_penetration,
        keel_offset_x=case.keel_offset_x,
        stroke=case.stroke_used,
        platen_velocity=case.platen_velocity,
        loading=case.loading,
        pipe_response=case.pipe_response,
        pipe_lateral_compliance=case.pipe_lateral_compliance,
        contact_length=max(2.0 * case.keel_depth, 12.0),
        steps_factor=case.steps_factor,
        track_energy=True,
        dt=case.dt,
    )


def run_case(case: PipelineCase) -> dict:
    """Run one case and return its load record, ISO comparison and energy audit."""
    scenario = case_scenario(case)
    result = run_grounding(scenario, max_steps=case.max_steps)
    metrics = load_displacement_metrics(result)
    section = scenario.section
    soil = scenario.soil
    vertical = iso19906_vertical_load(soil, section.outer_diameter,
                                      cover_depth=case.cover_depth)
    lateral = iso19906_lateral_load(soil, section.outer_diameter,
                                    cover_depth=case.cover_depth)
    track = result.history.get("energy_track", [])
    scan = closure_scan(track) if track else []
    loading_window = closure_from_track(track, start=len(track) // 4) \
        if len(track) > 4 else None
    whole = closure_from_track(track) if len(track) > 4 else None
    lateral_info = result.info.get("lateral", {})
    payload = {
        "case": {
            "name": case.name, "cover_depth": case.cover_depth, "su": case.su,
            "keel_depth": case.keel_depth, "keel_width": case.keel_width,
            "particle_radius": case.particle_radius, "loading": case.loading,
            "pipe_response": case.pipe_response, "dt": result.dt,
            "steps": result.steps, "time": result.history["time"][-1],
            "particles": result.info["particles"],
            "broken_bonds": result.info["broken_bonds"],
        },
        "dem": {
            "peak_load_per_length": metrics["peak_load_per_length"],
            "residual_load_per_length": metrics["residual_load_per_length"],
            "softening_ratio": metrics["softening_ratio"],
            "displacement_at_peak": metrics["displacement_at_peak"],
            "keel_penetration_final": result.history["keel_penetration"][-1],
            "pipe_displacement_final": result.final_pipe_displacement,
            "pipe_displacement_max": metrics["pipe_displacement_max"],
            "metrics_passed": metrics["passed"],
            "metrics_checks": metrics["checks"],
            "lateral": {"peak": lateral_info.get("peak_load", 0.0),
                        "sustained": lateral_info.get("sustained_load", 0.0),
                        "capacity": lateral_info.get("capacity", 0.0),
                        "displacement": lateral_info.get("final_displacement", 0.0),
                        "regime": lateral_info.get("regime", "rigid")},
        },
        "iso": {
            "vertical": vertical.as_dict(),
            "lateral": dict(lateral),
        },
        "comparison": {
            "vertical": compare_with_dem(metrics["peak_load_per_length"],
                                         vertical.total),
            "lateral": (compare_with_dem(lateral_info.get("sustained_load", 0.0),
                                         lateral["total_n_per_m"])
                        if case.loading == "lateral" else None),
            "lateral_peak": (compare_with_dem(lateral_info.get("peak_load", 0.0),
                                              lateral["total_n_per_m"])
                             if case.loading == "lateral" else None),
        },
        "energy_track": track,
        "snapshot": result.final_snapshot,
        "energy": {
            "whole_record": whole,
            "loading_phase": loading_window,
            "suffix_scan": scan,
        },
        "history": {
            "time": result.history["time"],
            "stroke": result.history["stroke"],
            "keel_penetration": result.history["keel_penetration"],
            "pipe_load": result.history["pipe_load"],
            "pipe_load_horizontal": result.history["pipe_load_horizontal"],
            "pipe_displacement": result.history["pipe_displacement"],
            "drive_force": result.history["drive_force"],
            "soil_force": result.history["soil_force"],
            "broken_bonds": result.history["broken_bonds"],
        },
    }
    return payload


def PIPELINE_CASES() -> dict:
    """Case lists of the five studies (single source of truth for the driver)."""
    return {
        "vertical": [
            PipelineCase("v_cover0.30_su30", cover_depth=0.30),
            PipelineCase("v_cover0.50_su30", cover_depth=0.50),
            PipelineCase("v_cover0.50_su15", cover_depth=0.50, su=15.0e3),
            PipelineCase("v_cover1.00_su45", cover_depth=1.00, su=45.0e3),
        ],
        "lateral": [
            PipelineCase("l_cover0.50_su30", cover_depth=0.50, su=30.0e3,
                         loading="lateral", pipe_lateral_compliance=True,
                         initial_penetration=1.0, keel_offset_x=-1.4, stroke=1.2),
            PipelineCase("l_cover0.50_su15", cover_depth=0.50, su=15.0e3,
                         loading="lateral", pipe_lateral_compliance=True,
                         initial_penetration=1.0, keel_offset_x=-1.4, stroke=1.2),
        ],
        "lateral_default_mode": [
            PipelineCase("l_default_mode", cover_depth=0.50, su=30.0e3,
                         loading="lateral", pipe_lateral_compliance=True,
                         stroke=1.2),
        ],
        "cover": [
            PipelineCase("c_cover0.30", cover_depth=0.30),
            PipelineCase("c_cover0.60", cover_depth=0.60),
            PipelineCase("c_cover1.00", cover_depth=1.00),
            PipelineCase("c_cover1.50", cover_depth=1.50),
        ],
        "grid": [
            PipelineCase("g_r0.50", cover_depth=0.50, particle_radius=0.50),
            PipelineCase("g_r0.35", cover_depth=0.50, particle_radius=0.35,
                         max_steps=60000),
            PipelineCase("g_r0.28", cover_depth=0.50, particle_radius=0.28,
                         max_steps=90000),
        ],
        "rigid_reference": [
            PipelineCase("rigid_ref", cover_depth=0.50, pipe_response="rigid"),
        ],
    }


def dt_refinement_cases(base: PipelineCase,
                        fractions: Sequence[float] = (1.0, 0.5, 0.25)
                        ) -> list[PipelineCase]:
    """Same case at ``fraction * recommended_dt`` - the energy refinement ladder.

    ``recommended_dt`` comes from the explicit-stability estimate of the micro
    parameters, so ``fraction = 1`` is the *coarsest* admissible step, not the
    accurate one; the ladder makes the ledger's dt sensitivity measurable
    instead of assumed.
    """
    dt0 = rubble_material(base.particle_radius).recommended_dt
    return [replace(base, name=f"e_dt{fraction:.2f}", dt=fraction * dt0)
            for fraction in fractions]


def _verdict(rows: Sequence[Mapping], deviation_key: str, threshold: float) -> dict:
    deviations = [abs(float(row[deviation_key])) for row in rows]
    worst = max(rows, key=lambda row: abs(float(row[deviation_key])))
    return {
        "rows": list(rows),
        "cases": len(rows),
        "max_abs_deviation": max(deviations) if deviations else None,
        "worst_case": worst.get("case") if deviations else None,
        "threshold": threshold,
        "satisfied": bool(deviations) and max(deviations) < threshold,
    }


def vertical_study(cases: Iterable[PipelineCase]) -> dict:
    """Criterion (a): DEM vertical peak load vs the ISO 19906-form capacity."""
    rows = []
    payloads = []
    for case in cases:
        payload = run_case(case)
        payloads.append(payload)
        rows.append({
            "case": case.name,
            "cover_depth": case.cover_depth,
            "su": case.su,
            "dem_n_per_m": payload["comparison"]["vertical"]["dem_n_per_m"],
            "iso_n_per_m": payload["comparison"]["vertical"]["iso_n_per_m"],
            "deviation": payload["comparison"]["vertical"]["deviation"],
            "abs_deviation": payload["comparison"]["vertical"]["abs_deviation"],
            "n_q": payload["iso"]["vertical"]["n_q"],
            "n_c": payload["iso"]["vertical"]["n_c"],
            "q_term": payload["iso"]["vertical"]["q_term_n_per_m"],
            "c_term": payload["iso"]["vertical"]["c_term_n_per_m"],
        })
    verdict = _verdict(rows, "deviation", ACCEPTANCE_THRESHOLDS["vertical_deviation"])
    verdict["payloads"] = payloads
    return verdict


def lateral_study(cases: Iterable[PipelineCase]) -> dict:
    """Criterion (b): DEM lateral load vs the ``N_p``-form break-out capacity."""
    rows = []
    payloads = []
    for case in cases:
        payload = run_case(case)
        payloads.append(payload)
        sustained = payload["comparison"]["lateral"]
        peak = payload["comparison"]["lateral_peak"]
        rows.append({
            "case": case.name,
            "cover_depth": case.cover_depth,
            "su": case.su,
            "dem_sustained_n_per_m": sustained["dem_n_per_m"],
            "dem_peak_n_per_m": peak["dem_n_per_m"],
            "iso_n_per_m": sustained["iso_n_per_m"],
            "deviation_sustained": sustained["deviation"],
            "deviation_peak": peak["deviation"],
            "n_factor": payload["iso"]["lateral"]["n_factor"],
            "regime": payload["dem"]["lateral"]["regime"],
            "pipe_lateral_displacement": payload["dem"]["lateral"]["displacement"],
        })
    verdict = _verdict(rows, "deviation_sustained",
                       ACCEPTANCE_THRESHOLDS["lateral_deviation"])
    verdict["payloads"] = payloads
    return verdict


def grid_study(radii: Iterable[float], *, base: PipelineCase,
               cover_depth: float | None = None) -> dict:
    """Criterion (e): peak-load change under particle refinement.

    Every case is run to the same physical stroke, so the comparison is not
    contaminated by runs that stop before the keel reaches the pipe crown.
    """
    rows = []
    payloads = []
    for radius in radii:
        case = replace(base, name=f"grid_r{radius:.2f}", particle_radius=radius,
                       cover_depth=base.cover_depth if cover_depth is None
                       else cover_depth)
        payload = run_case(case)
        payloads.append(payload)
        rows.append({
            "case": case.name,
            "particle_radius": radius,
            "particles": payload["case"]["particles"],
            "dt": payload["case"]["dt"],
            "steps": payload["case"]["steps"],
            "peak_n_per_m": payload["dem"]["peak_load_per_length"],
            "iso_n_per_m": payload["comparison"]["vertical"]["iso_n_per_m"],
        })
    rows.sort(key=lambda row: -row["particle_radius"])
    changes = []
    for coarse, fine in zip(rows, rows[1:]):
        if coarse["peak_n_per_m"] <= 0.0:
            continue
        change = (fine["peak_n_per_m"] - coarse["peak_n_per_m"]) / coarse["peak_n_per_m"]
        coarse["change_to_next"] = change
        changes.append(change)
    threshold = ACCEPTANCE_THRESHOLDS["grid_change"]
    return {
        "rows": rows,
        "payloads": payloads,
        "changes": changes,
        "particle_ratio": (rows[-1]["particles"] / rows[0]["particles"]
                           if len(rows) > 1 else 1.0),
        "max_abs_change": max((abs(c) for c in changes), default=None),
        "threshold": threshold,
        "satisfied": bool(changes) and max(abs(c) for c in changes) < threshold,
    }


def lateral_scenario_gap(case: PipelineCase) -> dict:
    """Reference run with the *unshifted, unburied* lateral setup.

    This is the configuration the worktree's lateral mode produces by default
    (keel above the mudline, section centred on the pipe, drifting sideways).
    The buoyant keel slides over the seabed without reaching the buried pipe, so
    the horizontal load is exactly zero: the node's lateral criterion cannot be
    evaluated in that setup at all.  The run is kept in the artifact as the
    evidence for *why* the campaign lateral cases bury the keel and offset it
    sideways.
    """
    payload = run_case(case)
    lateral = payload["dem"]["lateral"]
    return {
        "case": case.name,
        "contact_established": bool(lateral["peak"] != 0.0),
        "dem_peak_h": lateral["peak"],
        "dem_sustained_h": lateral["sustained"],
        "dem_displacement_h": lateral["displacement"],
        "iso_n_per_m": payload["comparison"]["vertical"]["iso_n_per_m"],
        "iso_lateral_n_per_m": payload["iso"]["lateral"]["total_n_per_m"],
        "note": ("buoyant keel drifts at the mouldline and never touches the "
                 "buried pipe: zero lateral load, criterion not evaluable"),
        "payload": payload,
    }


def cover_study(cases: Iterable[PipelineCase]) -> dict:
    """Criterion (c): DEM cover-instability verdict vs the theoretical prediction."""
    limits = instability_limits(pipe_section())
    rows = []
    payloads = []
    for case in cases:
        payload = run_case(case)
        payloads.append(payload)
        rows.append({
            "cover_depth": case.cover_depth,
            "dem_capacity": payload["dem"]["peak_load_per_length"],
            "iso_vertical": payload["comparison"]["vertical"]["iso_n_per_m"],
            "dem_displacement": payload["dem"]["pipe_displacement_max"],
            "dem_demand": payload["comparison"]["vertical"]["dem_n_per_m"],
            "iso_lateral": payload["comparison"]["lateral"]["iso_n_per_m"]
            if payload["comparison"]["lateral"] else
            payload["iso"]["lateral"]["total_n_per_m"],
            "dem_lateral": payload["dem"]["lateral"]["sustained"],
            "dem_lateral_displacement": payload["dem"]["lateral"]["displacement"],
        })
    envelope = cover_envelope(rows, diameter=PIPE_DIAMETER,
                              vertical_limit=limits["vertical"],
                              lateral_limit=limits["lateral"])
    envelope["limits"] = limits
    envelope["payloads"] = payloads
    envelope["satisfied"] = envelope["consistent_predictions"] == envelope["cases"]
    return envelope


def energy_study(entries: Iterable[Mapping]) -> dict:
    """Criterion (d): closure of the coupled ledger, whole record and loading phase."""
    rows = []
    for entry in entries:
        track = entry["track"]
        payload = entry.get("payload")
        rows.append({
            "case": entry.get("name"),
            "dt": float(entry["dt"]),
            "steps": entry.get("steps"),
            "whole_record": closure_from_track(track),
            "loading_phase": closure_from_track(track, start=len(track) // 4),
            "legacy_whole_record": closure_from_track(track, partition="legacy"),
        })
    finest = min(rows, key=lambda row: row["dt"]) if rows else None
    threshold = ACCEPTANCE_THRESHOLDS["energy_closure"]
    return {
        "rows": rows,
        "threshold": threshold,
        "finest_dt": finest["dt"] if finest else None,
        "finest_whole_record": finest["whole_record"]["closure_error"] if finest else None,
        "finest_loading_phase": finest["loading_phase"]["closure_error"] if finest else None,
        "satisfied": bool(finest and finest["loading_phase"]["closure_error"] < threshold),
    }


def evaluate_acceptance(*, vertical: Mapping, lateral: Mapping, cover: Mapping,
                        energy: Mapping, grid: Mapping) -> dict:
    """Combine the five criteria into the campaign verdict.

    All five must hold for the node to pass; the individual numbers are kept so
    a partial pass can be reported honestly instead of collapsed into a bool.
    """
    criteria = {
        "a_vertical_vs_iso": {
            "satisfied": bool(vertical["satisfied"]),
            "max_abs_deviation": vertical["max_abs_deviation"],
            "threshold": vertical["threshold"],
            "worst_case": vertical.get("worst_case"),
            "cases": vertical["cases"],
        },
        "b_lateral_vs_nc": {
            "satisfied": bool(lateral["satisfied"]),
            "max_abs_deviation": lateral["max_abs_deviation"],
            "threshold": lateral["threshold"],
            "worst_case": lateral.get("worst_case"),
            "cases": lateral["cases"],
        },
        "c_cover_instability": {
            "satisfied": bool(cover["satisfied"]),
            "consistent": cover["consistent_predictions"],
            "cases": cover["cases"],
            "max_abs_capacity_deviation": cover["max_abs_capacity_deviation"],
        },
        "d_energy_closure": {
            "satisfied": bool(energy["satisfied"]),
            "finest_closure_loading_phase": energy["finest_loading_phase"],
            "finest_closure_whole_record": energy["finest_whole_record"],
            "threshold": energy["threshold"],
            "finest_dt": energy["finest_dt"],
        },
        "e_grid_independence": {
            "satisfied": bool(grid["satisfied"]),
            "max_abs_change": grid["max_abs_change"],
            "threshold": grid["threshold"],
            "particle_ratio": grid["particle_ratio"],
        },
    }
    satisfied = {key: value["satisfied"] for key, value in criteria.items()}
    return {
        "criteria": criteria,
        "satisfied": satisfied,
        "passed": all(satisfied.values()),
        "failed": [key for key, value in satisfied.items() if not value],
        "thresholds": dict(ACCEPTANCE_THRESHOLDS),
    }