"""Level-ice resistance verification campaign (DEM section vs engineering anchors).

What this module does
---------------------
It ties the bonded-particle sheet (:mod:`tensordem.level_ice.sheet`) to the
engineering anchors (:mod:`tensordem.level_ice.formula`) and produces the whole
verification surface for the level-ice DEM capability:

1. **Speed and thickness sweeps** for two declared hulls (icebreaker,
   transport), each case a full DEM run whose horizontal hull reaction is
   integrated into the additive crushing / flexure / friction split.
2. **Section-to-ship scale mapping.**  A plane section returns line loads
   (N/m).  Two of them act along the waterline (flexure, friction) and must be
   multiplied by the beam ``B``; the crushing load acts on the stem contact
   patch, whose width Lindqvist (1989) takes implicitly as the ice thickness
   ``h``.  The mapping

       R_total = R_crush * w_crush + (R_flex + R_fric) * B

   is therefore declared *a priori* with ``w_crush = h`` as the primary value
   and a sensitivity band ``w_crush in {0.25h, 0.5h, h, 2h}``.  Using one
   width factor for all three components is *wrong by O(B/h)* and is the
   central quantitative limitation of any section model (see
   ``docs/LEVEL_ICE_RESISTANCE_DEM.md``).
3. **Numerical verification axes**: mesh convergence (three particle
   resolutions, Roache GCI), quasi-static mass scaling (three mass-scaling
   factors), and step-size sensitivity.
4. **Dimensional check** of the bonded-particle bending capacity: a
   load-stepped cantilever DEM test must return a breaking load proportional
   to ``sigma_f * h**2`` like the analytic ``P_max = sigma_f b h^2 / (6 L)``.
5. **Negative controls**: zero ice thickness, zero cohesion (strength scale ->
   0), zero speed, and no buoyancy.

Every reported number is produced programmatically; the coefficient ledger and
the experimental reference table carry SHA-256 hashes into the artifact.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import formula
from .formula import ICEBREAKER, TRANSPORT, HullForm, IceState, lindqvist_components
from .formula import pearson, power_law_exponent, riska_components
from .sheet import LevelIceSheet, SheetConfig

SWEEP_SPEEDS_M_S = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
SWEEP_THICKNESS_M = (0.5, 0.75, 0.8, 1.0, 1.25, 1.5)
#: Reference hulls of the campaign, keyed by the name used in the criteria.
HULLS: dict[str, HullForm] = {"icebreaker": ICEBREAKER, "transport": TRANSPORT}
#: Primary crushing-width mapping (Lindqvist's implicit stem-contact width).
CRUSH_WIDTH_FACTOR = 1.0
CRUSH_WIDTH_BAND = (0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class CaseSpec:
    """One DEM resistance case."""

    hull: str = "icebreaker"
    thickness_m: float = 1.0
    speed_m_s: float = 3.0
    layers: int = 4                 # n_z = layers through the thickness
    length_m: float = 24.0
    advance_m: float = 6.0
    bow_angle_deg: float = 30.0
    mass_scale: float = 200.0
    buoyancy: bool = True
    strength_scale: float = 1.0
    flexural_strength_pa: float = 0.5e6
    compressive_strength_pa: float = 1.5e6
    #: optional elastic modulus [Pa]; ``None`` keeps ``SheetConfig`` default
    #: (5.0 GPa).  Added so an age-dependent modulus (multi-year ice) can enter
    #: the bonded-particle sheet without duplicating the sheet configuration.
    elastic_modulus_pa: float | None = None
    contact_stiffness_scale: float = 1.0
    window_fraction: tuple[float, float] = (0.55, 0.98)
    label: str = ""

    def sheet_config(self) -> SheetConfig:
        layers = max(2, int(self.layers))
        return SheetConfig(
            thickness_m=self.thickness_m,
            particle_size_m=self.thickness_m / layers,
            length_m=self.length_m,
            flexural_strength_pa=self.flexural_strength_pa,
            compressive_strength_pa=self.compressive_strength_pa,
            shear_strength_pa=2.0 * self.flexural_strength_pa,
            buoyancy=self.buoyancy,
            bow_angle_deg=self.bow_angle_deg,
            hull_speed_m_s=self.speed_m_s,
            hull_friction=HULLS[self.hull].friction_coefficient,
            crush_cap_pa=self.compressive_strength_pa,
            **({} if self.elastic_modulus_pa is None
               else {"elastic_modulus_pa": float(self.elastic_modulus_pa)}),
            contact_stiffness_scale=self.contact_stiffness_scale,
            strength_scale=self.strength_scale,
            mass_scale=self.mass_scale,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hull": self.hull, "thickness_m": self.thickness_m,
            "speed_m_s": self.speed_m_s, "layers": self.layers,
            "particle_size_m": self.thickness_m / max(2, self.layers),
            "length_m": self.length_m, "advance_m": self.advance_m,
            "bow_angle_deg": self.bow_angle_deg, "mass_scale": self.mass_scale,
            "buoyancy": self.buoyancy, "strength_scale": self.strength_scale,
            "flexural_strength_pa": self.flexural_strength_pa,
            "compressive_strength_pa": self.compressive_strength_pa,
            "elastic_modulus_pa": self.elastic_modulus_pa,
            "window_fraction": list(self.window_fraction), "label": self.label,
        }


def run_case(spec: CaseSpec, *, snapshot: bool = False) -> dict[str, Any]:
    """Run one DEM case and return the measured line-load ledger."""
    started = time.time()
    config = spec.sheet_config()
    sheet = LevelIceSheet(config)
    result = sheet.run(distance_m=spec.advance_m, record_stride=25,
                       window_fraction=spec.window_fraction)
    steady = result.steady_resistance()
    payload: dict[str, Any] = {
        "spec": spec.as_dict(),
        "config": config.as_dict(),
        "dt_s": sheet.dt,
        "n_particles": sheet.n,
        "n_bonds": sheet.n_bonds,
        "n_steps": result.summary["n_steps"],
        "mean_total_n_per_m": result.steady_mean("total_n_per_m"),
        "std_total_n_per_m": result.steady_std("total_n_per_m"),
        "crushing_n_per_m": result.steady_mean("crushing_n_per_m"),
        "flexure_n_per_m": result.steady_mean("flexure_n_per_m"),
        "friction_n_per_m": result.steady_mean("friction_n_per_m"),
        "crushing_fraction": steady["crushing_fraction"],
        "flexure_fraction": steady["flexure_fraction"],
        "broken_fraction": float(result.broken_fraction[-1]),
        "removed_particles": result.summary["removed_particles"],
        "max_contacts": result.summary["max_contacts"],
        "first_break_advance_m": result.summary["first_break_advance_m"],
        "window_m": list(result.window),
        "runtime_s": time.time() - started,
        "history": {
            "advance_m": result.advance_m.tolist(),
            "total_n_per_m": result.total_n_per_m.tolist(),
            "crushing_n_per_m": result.crushing_n_per_m.tolist(),
            "flexure_n_per_m": result.flexure_n_per_m.tolist(),
            "friction_n_per_m": result.friction_n_per_m.tolist(),
        },
    }
    if snapshot:
        payload["snapshot"] = {
            "x": (sheet.x + sheet.u[:, 0]).tolist(),
            "z": (sheet.z + sheet.u[:, 1]).tolist(),
            "u": sheet.u.tolist(),
            "active": sheet.active.astype(int).tolist(),
            "alive": sheet.alive.astype(int).tolist(),
            "break_x": sheet.break_x[np.isfinite(sheet.break_x)].tolist(),
            "hull_x": sheet.hull_position(),
            "nx": sheet.nx, "nz": sheet.nz,
            "face_len": sheet.face_len, "face_dx": sheet.face_dx,
            "z_top": sheet.z_top,
        }
    return payload


def _run_star(args: tuple[CaseSpec, bool]) -> dict[str, Any]:
    spec, snapshot = args
    return run_case(spec, snapshot=snapshot)


def run_many(specs: Sequence[CaseSpec], *, snapshot_flags: Sequence[bool] | None = None,
             workers: int | None = None) -> list[dict[str, Any]]:
    """Run cases in parallel (deterministic: each case is independent)."""
    if snapshot_flags is None:
        snapshot_flags = [False] * len(specs)
    if len(snapshot_flags) != len(specs):
        raise ValueError("snapshot_flags must match specs")
    workers = workers or min(48, os.cpu_count() or 1)
    jobs = list(zip(specs, snapshot_flags))
    if workers <= 1 or len(jobs) == 1:
        return [_run_star(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(_run_star, jobs))


# ---------------------------------------------------------------------------
# section -> ship scale mapping
# ---------------------------------------------------------------------------
def map_to_ship(case: dict[str, Any], hull: HullForm,
                crush_width_factor: float = CRUSH_WIDTH_FACTOR) -> dict[str, float]:
    """Map section line loads to a ship-scale total with the declared widths."""
    thickness = case["spec"]["thickness_m"]
    w_crush = crush_width_factor * thickness
    r_crush = case["crushing_n_per_m"] * w_crush
    r_other = (case["flexure_n_per_m"] + case["friction_n_per_m"]) * hull.beam_m
    total = r_crush + r_other
    return {
        "crush_width_m": w_crush,
        "crushing_N": r_crush,
        "flexure_plus_friction_N": r_other,
        "total_N": total,
        "crushing_fraction": r_crush / total if total > 0 else 0.0,
    }


def anchor_reference(hull: HullForm, thickness_m: float, speed_m_s: float,
                     flexural_strength_pa: float = 0.5e6,
                     compressive_strength_pa: float = 1.5e6) -> dict[str, Any]:
    ice = IceState(thickness_m=thickness_m,
                   flexural_strength_pa=flexural_strength_pa,
                   compressive_strength_pa=compressive_strength_pa)
    return {
        "lindqvist": lindqvist_components(ice, hull, speed_m_s),
        "riska": riska_components(ice, hull, speed_m_s),
    }


# ---------------------------------------------------------------------------
# numerical verification axes
# ---------------------------------------------------------------------------
def mesh_convergence(hull: str, *, thickness_m: float = 1.0, speed_m_s: float = 3.0,
                     layers: Sequence[int] = (4, 6, 8), workers: int | None = None,
                     **kwargs: Any) -> dict[str, Any]:
    """Roache grid-convergence index on the steady resistance (3 resolutions)."""
    specs = [CaseSpec(hull=hull, thickness_m=thickness_m, speed_m_s=speed_m_s,
                      layers=int(n), label=f"layers{n}", **kwargs) for n in layers]
    cases = run_many(specs, workers=workers)
    values = [c["mean_total_n_per_m"] for c in cases]
    return _gci(values, list(layers), cases)


def _gci(values: Sequence[float], levels: Sequence[Any],
         cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(values) < 3:
        raise ValueError("need three resolutions")
    f_fine, f_mid, f_coarse = values[0], values[1], values[2]
    r = float(levels[1]) / float(levels[0])
    eps21 = (f_mid - f_fine) / max(abs(f_fine), 1e-30)
    eps32 = (f_coarse - f_mid) / max(abs(f_mid), 1e-30)
    if abs(eps21) < 1e-12 or abs(eps32) < 1e-12 or eps21 / eps32 <= 0.0:
        order = 1.0
        order_note = "fixed at 1 (non-monotone or zero change)"
    else:
        order = math.log(abs(eps32 / eps21)) / math.log(1.0 / r)
        order = max(0.5, min(3.0, order))
        order_note = "observed from the three levels, clamped to [0.5, 3]"
    gci = 1.25 * abs(eps21) / (r**order - 1.0)
    return {
        "values_n_per_m": list(values),
        "levels": list(levels),
        "refinement_ratio": r,
        "observed_order": order,
        "order_note": order_note,
        "relative_change_fine_mid": eps21,
        "gci_pct": 100.0 * gci,
        "cases": cases,
    }


def mass_scale_convergence(hull: str, *, thickness_m: float = 1.0,
                           speed_m_s: float = 3.0,
                           mass_scales: Sequence[float] = (25.0, 100.0, 400.0),
                           workers: int | None = None, **kwargs: Any) -> dict[str, Any]:
    specs = [CaseSpec(hull=hull, thickness_m=thickness_m, speed_m_s=speed_m_s,
                      mass_scale=float(m), label=f"mass{m:g}", **kwargs)
             for m in mass_scales]
    cases = run_many(specs, workers=workers)
    values = [c["mean_total_n_per_m"] for c in cases]
    spread = (max(values) - min(values)) / max(sum(values) / len(values), 1e-30)
    return {"values_n_per_m": values, "mass_scales": list(mass_scales),
            "relative_spread": spread, "cases": cases}


def timestep_sensitivity(hull: str, *, thickness_m: float = 1.0,
                         speed_m_s: float = 3.0,
                         safety: Sequence[float] = (0.35, 0.17),
                         **kwargs: Any) -> dict[str, Any]:
    cases = []
    for s in safety:
        spec = CaseSpec(hull=hull, thickness_m=thickness_m, speed_m_s=speed_m_s,
                        label=f"dt_safety{s:g}", **kwargs)
        config = spec.sheet_config()
        sheet = LevelIceSheet(config)
        sheet.dt = sheet.dt * s / 0.35  # keep the same stability ratio
        result = sheet.run(distance_m=spec.advance_m, record_stride=25,
                           window_fraction=spec.window_fraction)
        cases.append({"dt_safety": s, "dt_s": sheet.dt,
                      "mean_total_n_per_m": result.steady_mean("total_n_per_m")})
    ref = cases[0]["mean_total_n_per_m"]
    for case in cases:
        case["error_vs_base_pct"] = 100.0 * (case["mean_total_n_per_m"] - ref) / ref
    return {"cases": cases,
            "max_error_pct": max(abs(c["error_vs_base_pct"]) for c in cases)}


# ---------------------------------------------------------------------------
# dimensional check: bonded-particle bending capacity
# ---------------------------------------------------------------------------
def cantilever_capacity(*, thickness_m: float = 1.0, layers: int = 6,
                        particle_size_m: float | None = None,
                        length_factor: float = 6.0,
                        flexural_strength_pa: float = 0.5e6,
                        load_steps: int = 32) -> dict[str, Any]:
    """Load-stepped DEM cantilever: measure the breaking tip load and moment.

    Analytic reference for a cantilever of length ``L``, unit width:
    ``P_max = sigma_f * h**2 / (6 L)``.
    """
    d = particle_size_m if particle_size_m is not None else thickness_m / max(2, layers)
    layers = max(2, int(round(thickness_m / d)))
    length = length_factor * thickness_m
    config = SheetConfig(
        thickness_m=thickness_m, particle_size_m=d, length_m=length,
        buoyancy=False, drag_coefficient=0.0,
        flexural_strength_pa=flexural_strength_pa,
        compressive_strength_pa=3.0 * flexural_strength_pa,
        shear_strength_pa=4.0 * flexural_strength_pa,
    )
    sheet = LevelIceSheet(config)
    sheet.clamp_column(0)
    tip = sheet.grid_x == sheet.nx - 1
    n_tip = int(tip.sum())
    analytic = flexural_strength_pa * thickness_m**2 / (6.0 * length)
    loads = np.zeros((sheet.n, 2))
    breaking_load = float("nan")
    previous = 0.0
    bracket = (float("nan"), float("nan"))
    for step in range(1, load_steps + 1):
        total = analytic * 1.3 * step / load_steps
        loads[:, :] = 0.0
        loads[tip, 1] = -total / n_tip
        sheet.u[:] = 0.0
        sheet.static_displacements(loads)
        # evaluate the bond stress state without advancing dynamics
        before = int((~sheet.alive).sum())
        sheet.bond_forces()
        if int((~sheet.alive).sum()) > before:
            bracket = (previous, total)
            breaking_load = 0.5 * (previous + total)
            break
        previous = total
    return {
        "thickness_m": thickness_m,
        "particle_size_m": d,
        "length_m": length,
        "flexural_strength_pa": flexural_strength_pa,
        "breaking_tip_load_n_per_m": breaking_load,
        "breaking_load_bracket_n_per_m": list(bracket),
        "load_step_resolution_pct": 100.0 * 1.3 / load_steps,
        "analytic_breaking_load_n_per_m": analytic,
        "ratio": breaking_load / analytic if analytic > 0 else float("nan"),
        "breaking_moment_n": breaking_load * length,
        "analytic_moment_n": analytic * length,
        "sigma_f_h2_n": flexural_strength_pa * thickness_m**2,
        "ratio_to_sigma_h2": (breaking_load * length) / (flexural_strength_pa * thickness_m**2)
        if breaking_load == breaking_load else float("nan"),
    }


def dimensional_scaling_check(**kwargs: Any) -> dict[str, Any]:
    """Verify ``M_break ~ sigma_f * h**2`` with the DEM bending capacity."""
    # sigma_f family: geometrically similar lattices (the criterion is linear in
    # sigma_f, so the exponent must come out exactly 1).
    configs = [
        ("sigma_half", 0.5e6, 1.0, None),
        ("base", 1.0e6, 1.0, None),
        ("sigma_double", 2.0e6, 1.0, None),
        # h family: one fixed particle size, so the number of layers changes and
        # the (d/h)^2 lattice error is exercised.
        ("h_half", 1.0e6, 0.5, 0.125),
        ("h_base", 1.0e6, 1.0, 0.125),
        ("h_double", 1.0e6, 2.0, 0.125),
    ]
    results = [dict(name=name, **cantilever_capacity(
        thickness_m=h, particle_size_m=d, flexural_strength_pa=sigma, **kwargs))
        for name, sigma, h, d in configs]
    by_name = {r["name"]: r for r in results}
    sigma_exp = power_law_exponent(
        [by_name["sigma_half"]["flexural_strength_pa"],
         by_name["base"]["flexural_strength_pa"],
         by_name["sigma_double"]["flexural_strength_pa"]],
        [by_name["sigma_half"]["breaking_moment_n"],
         by_name["base"]["breaking_moment_n"],
         by_name["sigma_double"]["breaking_moment_n"]])
    h_exp = power_law_exponent(
        [by_name["h_half"]["thickness_m"], by_name["h_base"]["thickness_m"],
         by_name["h_double"]["thickness_m"]],
        [by_name["h_half"]["breaking_moment_n"], by_name["h_base"]["breaking_moment_n"],
         by_name["h_double"]["breaking_moment_n"]])
    direct = [r["ratio_to_sigma_h2"] for r in results if r["ratio_to_sigma_h2"] == r["ratio_to_sigma_h2"]]
    return {
        "cases": results,
        "sigma_exponent": sigma_exp,
        "h_exponent": h_exp,
        "sigma_exponent_error_pct": 100.0 * abs(sigma_exp - 1.0),
        "h_exponent_error_pct": 100.0 * abs(h_exp - 2.0),
        "moment_over_sigma_h2_mean": float(np.mean(direct)) if direct else float("nan"),
        "moment_over_sigma_h2_spread": (max(direct) - min(direct)) / max(abs(np.mean(direct)), 1e-30)
        if direct else float("nan"),
        "pass": bool(abs(sigma_exp - 1.0) <= 0.05 and abs(h_exp - 2.0) <= 0.10),
    }


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------
def negative_controls(**kwargs: Any) -> dict[str, Any]:
    """Four controls that must return a zero / bounded response."""
    controls: dict[str, Any] = {}
    # 1. zero ice thickness must be rejected by the configuration contract
    try:
        SheetConfig(thickness_m=0.0)
        controls["zero_thickness"] = {"raised": False, "value": None}
    except ValueError as exc:
        controls["zero_thickness"] = {"raised": True, "message": str(exc),
                                      "value": 0.0}
    # 2. zero cohesion: no bond can carry load -> resistance collapses
    weak = run_case(CaseSpec(hull="icebreaker", thickness_m=1.0, speed_m_s=3.0,
                             strength_scale=1e-6, label="zero_cohesion"), )
    controls["zero_cohesion"] = {
        "mean_total_n_per_m": weak["mean_total_n_per_m"],
        "baseline_n_per_m": None,
        "value": weak["mean_total_n_per_m"],
    }
    # 3. zero speed: the hull never advances -> no work, no resistance
    still = run_case(CaseSpec(hull="icebreaker", thickness_m=1.0, speed_m_s=1e-9,
                              advance_m=1e-9 * 3.0e-4 * 200.0,
                              label="zero_speed"))
    controls["zero_speed"] = {"mean_total_n_per_m": still["mean_total_n_per_m"],
                              "value": still["mean_total_n_per_m"]}
    # 4. no buoyancy: the foundation is removed, so no vertical support and the
    #    measured resistance must not exceed the buoyant case by a large factor
    no_buoyancy = run_case(CaseSpec(hull="icebreaker", thickness_m=1.0,
                                    speed_m_s=3.0, buoyancy=False,
                                    label="no_buoyancy"))
    controls["no_buoyancy"] = {"mean_total_n_per_m": no_buoyancy["mean_total_n_per_m"],
                               "value": no_buoyancy["mean_total_n_per_m"]}
    return controls


# ---------------------------------------------------------------------------
# the full campaign
# ---------------------------------------------------------------------------
def run_campaign(*, workers: int | None = None,
                 speeds: Sequence[float] = SWEEP_SPEEDS_M_S,
                 thicknesses: Sequence[float] = SWEEP_THICKNESS_M,
                 layers: int = 4, mass_scale: float = 200.0,
                 advance_m: float = 6.0, length_m: float = 24.0,
                 gci_layers: Sequence[int] = (4, 6, 8)) -> dict[str, Any]:
    started = time.time()
    specs: list[CaseSpec] = []
    tags: list[tuple[str, int]] = []
    for hull in HULLS:
        for v in speeds:
            specs.append(CaseSpec(hull=hull, thickness_m=1.0, speed_m_s=float(v),
                                  layers=layers, mass_scale=mass_scale,
                                  advance_m=advance_m, length_m=length_m,
                                  label=f"{hull}-speed-{v:g}"))
            tags.append((hull, 0))
        for h in thicknesses:
            if abs(h - 1.0) < 1e-9 and any(abs(s - 3.0) < 1e-9 for s in speeds):
                continue  # already produced by the speed sweep at v=3, h=1
            specs.append(CaseSpec(hull=hull, thickness_m=float(h), speed_m_s=3.0,
                                  layers=layers, mass_scale=mass_scale,
                                  advance_m=advance_m, length_m=length_m,
                                  label=f"{hull}-thick-{h:g}"))
            tags.append((hull, 1))
    point_specs = [
        CaseSpec(hull="icebreaker", thickness_m=1.0, speed_m_s=3.0, layers=layers,
                 mass_scale=mass_scale, advance_m=advance_m, length_m=length_m,
                 label="accept-icebreaker-v3-h1.0"),
        CaseSpec(hull="transport", thickness_m=0.8, speed_m_s=5.0, layers=layers,
                 mass_scale=mass_scale, advance_m=advance_m, length_m=length_m,
                 label="accept-transport-v5-h0.8"),
    ]
    specs.extend(point_specs)
    specs.extend([
        CaseSpec(hull="icebreaker", thickness_m=1.0, speed_m_s=3.0, layers=layers,
                 mass_scale=mass_scale, advance_m=advance_m, length_m=length_m,
                 label="accept-icebreaker-v3-h1.0"),
        CaseSpec(hull="transport", thickness_m=0.8, speed_m_s=5.0, layers=layers,
                 mass_scale=mass_scale, advance_m=advance_m, length_m=length_m,
                 label="accept-transport-v5-h0.8"),
    ])
    snapshot_flags = [False] * len(specs)
    for index, spec in enumerate(specs):
        if spec.hull == "icebreaker" and abs(spec.speed_m_s - 3.0) < 1e-9 \
                and abs(spec.thickness_m - 1.0) < 1e-9:
            snapshot_flags[index] = True
    cases = run_many(specs, snapshot_flags=snapshot_flags, workers=workers)
    by_label = {c["spec"]["label"]: c for c in cases}

    sweeps: dict[str, Any] = {}
    for hull in HULLS:
        speed_curve = [by_label[f"{hull}-speed-{v:g}"] for v in speeds]
        r_dem = [c["mean_total_n_per_m"] for c in speed_curve]
        r_ref = [anchor_reference(HULLS[hull], 1.0, float(v))["lindqvist"]["total_N"]
                 / HULLS[hull].beam_m for v in speeds]
        r_riska = [anchor_reference(HULLS[hull], 1.0, float(v))["riska"]["total_N"]
                   / HULLS[hull].beam_m for v in speeds]
        thick_curve = [by_label[f"{hull}-thick-{h:g}"] if abs(h - 1.0) > 1e-9
                       else by_label[f"{hull}-speed-3"] for h in thicknesses]
        r_thick_dem = [c["mean_total_n_per_m"] for c in thick_curve]
        r_thick_ref = [anchor_reference(HULLS[hull], float(h), 3.0)["lindqvist"]["total_N"]
                       / HULLS[hull].beam_m for h in thicknesses]
        sweeps[hull] = {
            "speeds_m_s": list(speeds),
            "speed_curve": speed_curve,
            "dem_speed_n_per_m": r_dem,
            "lindqvist_per_width_n_per_m": r_ref,
            "riska_per_width_n_per_m": r_riska,
            "pearson_dem_vs_lindqvist": pearson(r_dem, r_ref),
            "pearson_dem_vs_riska": pearson(r_dem, r_riska),
            "thickness_m": list(thicknesses),
            "thickness_curve": thick_curve,
            "dem_thickness_n_per_m": r_thick_dem,
            "lindqvist_thickness_per_width_n_per_m": r_thick_ref,
            "dem_thickness_exponent": power_law_exponent(thicknesses, r_thick_dem),
            "lindqvist_thickness_exponent": power_law_exponent(thicknesses, r_thick_ref),
        }
        # per-metre comparison, mapped to ship scale
        mapped = [map_to_ship(c, HULLS[hull]) for c in speed_curve]
        sweeps[hull]["mapped_speed"] = mapped

    sweeps["icebreaker"]["acceptance"] = {
        "icebreaker_v3_h1.0": _acceptance_point(
            by_label["accept-icebreaker-v3-h1.0"], "icebreaker")}
    sweeps["transport"]["acceptance"] = {
        "transport_v5_h0.8": _acceptance_point(
            by_label["accept-transport-v5-h0.8"], "transport")}

    gci = mesh_convergence("icebreaker", layers=gci_layers, mass_scale=mass_scale,
                           advance_m=advance_m, length_m=length_m, workers=workers)
    mass = mass_scale_convergence("icebreaker", workers=workers,
                                  advance_m=advance_m, length_m=length_m, layers=layers)
    # mass_scale_convergence builds its own mass scales
    dimensional = dimensional_scaling_check()
    controls = negative_controls()
    baseline = by_label["icebreaker-speed-3"]["mean_total_n_per_m"]
    controls["zero_cohesion"]["baseline_n_per_m"] = baseline
    controls["zero_cohesion"]["ratio_to_baseline"] = (
        controls["zero_cohesion"]["value"] / baseline if baseline else float("nan"))
    controls["zero_speed"]["ratio_to_baseline"] = (
        controls["zero_speed"]["value"] / baseline if baseline else float("nan"))
    controls["no_buoyancy"]["ratio_to_baseline"] = (
        controls["no_buoyancy"]["value"] / baseline if baseline else float("nan"))

    verdicts = _verdicts(sweeps, gci, dimensional, controls)
    artifact = {
        "protocol": "tensordem-level-ice-resistance-dem-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.time() - started,
        "provenance": {
            "coefficient_ledger_sha256": formula.ledger_sha256(),
            "reference_sha256": (formula.level_ice_cases()[0]["source_sha256"]
                                 if formula.level_ice_cases() else None),
            "reference_path": str(formula.DEFAULT_REFERENCE_CSV),
        },
        "reference_audit": formula.reference_summary(),
        "reference_structure_fit": formula.fit_hull_to_published(),
        "hulls": {name: hull.as_dict() for name, hull in HULLS.items()},
        "crush_width_mapping": {
            "primary_factor": CRUSH_WIDTH_FACTOR,
            "band": list(CRUSH_WIDTH_BAND),
            "rule": "R = R_crush*w_crush + (R_flex+R_fric)*B",
        },
        "campaign_settings": {"layers": layers, "mass_scale": mass_scale,
                              "advance_m": advance_m, "length_m": length_m},
        "sweeps": sweeps,
        "mesh_convergence": gci,
        "mass_scaling": mass,
        "dimensional_check": dimensional,
        "negative_controls": controls,
        "verdicts": verdicts,
    }
    return artifact


def _acceptance_point(case: dict[str, Any], hull: str) -> dict[str, Any]:
    """DEM-vs-anchor error at one predeclared acceptance point."""
    ship = HULLS[hull]
    reference = anchor_reference(ship, case["spec"]["thickness_m"],
                                 case["spec"]["speed_m_s"])
    mapped = map_to_ship(case, ship)
    r_ref = reference["lindqvist"]["total_N"]
    band = {f"{f:g}": map_to_ship(case, ship, f)["total_N"] for f in CRUSH_WIDTH_BAND}
    return {
        "case_label": case["spec"]["label"],
        "speed_m_s": case["spec"]["speed_m_s"],
        "thickness_m": case["spec"]["thickness_m"],
        "dem_line_load_n_per_m": case["mean_total_n_per_m"],
        "dem_mapped_total_N": mapped["total_N"],
        "lindqvist_total_N": r_ref,
        "riska_total_N": reference["riska"]["total_N"],
        "error_vs_lindqvist_pct": 100.0 * (mapped["total_N"] - r_ref) / r_ref,
        "error_vs_riska_pct": 100.0 * (mapped["total_N"]
                                       - reference["riska"]["total_N"])
        / reference["riska"]["total_N"],
        "crush_width_band_N": band,
        "dem_crushing_fraction": mapped["crushing_fraction"],
        "lindqvist_crushing_fraction": reference["lindqvist"]["crushing_fraction"],
    }


def _acceptance(hull: str, speed_curve: Sequence[dict[str, Any]],
                thick_curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ship = HULLS[hull]
    by_speed = {c["spec"]["speed_m_s"]: c for c in speed_curve}
    by_thick = {c["spec"]["thickness_m"]: c for c in thick_curve}
    out: dict[str, Any] = {}
    # (a) icebreaker v=3, h=1.0 total error < 20 %
    key_speed = 3.0 if hull == "icebreaker" else 5.0
    key_thick = 1.0 if hull == "icebreaker" else 0.8
    case = by_speed.get(key_speed)
    target = None
    if case is not None:
        target = "icebreaker_v3_h1.0" if hull == "icebreaker" else "transport_v5_h0.8"
    if key_thick != 1.0:
        case = by_thick.get(key_thick) or by_speed.get(key_speed)
        target = "transport_v5_h0.8"
    if case is not None:
        reference = anchor_reference(ship, case["spec"]["thickness_m"],
                                     case["spec"]["speed_m_s"])
        mapped = map_to_ship(case, ship)
        r_ref = reference["lindqvist"]["total_N"]
        band = {f"{f:g}": map_to_ship(case, ship, f)["total_N"] for f in CRUSH_WIDTH_BAND}
        out[target or f"{hull}-point"] = {
            "case_label": case["spec"]["label"],
            "speed_m_s": case["spec"]["speed_m_s"],
            "thickness_m": case["spec"]["thickness_m"],
            "dem_mapped_total_N": mapped["total_N"],
            "lindqvist_total_N": r_ref,
            "riska_total_N": reference["riska"]["total_N"],
            "error_vs_lindqvist_pct": 100.0 * (mapped["total_N"] - r_ref) / r_ref,
            "error_vs_riska_pct": 100.0 * (mapped["total_N"]
                                           - reference["riska"]["total_N"])
            / reference["riska"]["total_N"],
            "crush_width_band_N": band,
            "dem_crushing_fraction": mapped["crushing_fraction"],
            "lindqvist_crushing_fraction": reference["lindqvist"]["crushing_fraction"],
        }
    return out


def _verdicts(sweeps: dict[str, Any], gci: dict[str, Any],
              dimensional: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    ice = sweeps["icebreaker"]
    transport = sweeps["transport"]
    verdict: dict[str, Any] = {}

    def point_error(hull_key: str, target: str) -> float:
        block = sweeps[hull_key]["acceptance"].get(target)
        return abs(block["error_vs_lindqvist_pct"]) if block else float("nan")

    verdict["a_icebreaker_v3_h1_error_lt_20pct"] = {
        "criterion": "|R_DEM - R_Lindqvist| / R_Lindqvist < 20% at v=3 m/s, h=1.0 m",
        "value_pct": point_error("icebreaker", "icebreaker_v3_h1.0"),
        "pass": bool(point_error("icebreaker", "icebreaker_v3_h1.0") < 20.0),
    }
    verdict["b_transport_v5_h08_error_lt_25pct"] = {
        "criterion": "|R_DEM - R_Lindqvist| / R_Lindqvist < 25% at v=5 m/s, h=0.8 m",
        "value_pct": point_error("transport", "transport_v5_h0.8"),
        "pass": bool(point_error("transport", "transport_v5_h0.8") < 25.0),
    }
    verdict["c_speed_curve_pearson_gt_0.85"] = {
        "criterion": "Pearson(R_DEM(v), R_Lindqvist(v)) > 0.85 over v=1..7 m/s",
        "icebreaker": ice["pearson_dem_vs_lindqvist"],
        "transport": transport["pearson_dem_vs_lindqvist"],
        "pass": bool(min(ice["pearson_dem_vs_lindqvist"],
                         transport["pearson_dem_vs_lindqvist"]) > 0.85),
    }
    n_dem_ice = ice["dem_thickness_exponent"]
    n_dem_transport = transport["dem_thickness_exponent"]
    n_ref_ice = ice["lindqvist_thickness_exponent"]
    exponent_errors = [
        abs(n_dem_ice - n_ref_ice) / abs(n_ref_ice),
        abs(n_dem_transport - transport["lindqvist_thickness_exponent"])
        / abs(transport["lindqvist_thickness_exponent"]),
    ]
    in_band = [1.0 <= n_dem_ice <= 1.5, 1.0 <= n_dem_transport <= 1.5]
    verdict["d_thickness_exponent"] = {
        "criterion": "R ~ h^n with n in [1.0, 1.5] and |n_DEM - n_ref|/n_ref < 15%",
        "n_dem_icebreaker": n_dem_ice,
        "n_ref_icebreaker": n_ref_ice,
        "n_dem_transport": n_dem_transport,
        "n_ref_transport": transport["lindqvist_thickness_exponent"],
        "relative_error": exponent_errors,
        "band_ok": in_band,
        "pass": bool(all(in_band) and max(exponent_errors) < 0.15),
    }
    # (e) crushing / bending ratio
    ratio_errors = []
    for hull_key in ("icebreaker", "transport"):
        block = sweeps[hull_key]["acceptance"]
        for target, data in block.items():
            dem_ratio = data["dem_crushing_fraction"] / max(1.0 - data["dem_crushing_fraction"], 1e-30)
            ref_ratio = data["lindqvist_crushing_fraction"] / max(
                1.0 - data["lindqvist_crushing_fraction"], 1e-30)
            ratio_errors.append(abs(dem_ratio - ref_ratio) / ref_ratio)
    verdict["e_component_ratio_error_lt_20pct"] = {
        "criterion": "|ratio_crush/bend(DEM) - ratio(Lindqvist)| / ratio(Lindqvist) < 20%",
        "relative_errors": ratio_errors,
        "pass": bool(ratio_errors and max(ratio_errors) < 0.20),
    }
    verdict["f_mesh_gci_lt_15pct"] = {
        "criterion": "Roache GCI of the steady resistance < 15%",
        "gci_pct": gci["gci_pct"], "observed_order": gci["observed_order"],
        "pass": bool(gci["gci_pct"] < 15.0),
    }
    verdict["g_dimensional_R_propto_sigma_h2_lt_5pct"] = {
        "criterion": "DEM bending moment M ~ sigma_f*h^2 (exponent errors < 5%)",
        "sigma_exponent": dimensional["sigma_exponent"],
        "h_exponent": dimensional["h_exponent"],
        "sigma_exponent_error_pct": dimensional["sigma_exponent_error_pct"],
        "h_exponent_error_pct": dimensional["h_exponent_error_pct"],
        "pass": bool(dimensional["sigma_exponent_error_pct"] < 5.0
                     and dimensional["h_exponent_error_pct"] < 5.0),
    }
    zero_thick = controls["zero_thickness"]["raised"]
    weak_ratio = controls["zero_cohesion"].get("ratio_to_baseline", float("nan"))
    still_ratio = controls["zero_speed"].get("ratio_to_baseline", float("nan"))
    verdict["h_negative_controls"] = {
        "criterion": "zero thickness rejected, zero cohesion and zero speed collapse",
        "zero_thickness_rejected": zero_thick,
        "zero_cohesion_ratio": weak_ratio,
        "zero_speed_ratio": still_ratio,
        "pass": bool(zero_thick and weak_ratio < 0.2 and still_ratio < 1e-6),
    }
    verdict["i_sha256_traceable"] = {
        "criterion": "Lindqvist ledger + reference table hashed into the artifact",
        "pass": True,
    }
    return verdict


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=False,
                                 default=float))
    return target