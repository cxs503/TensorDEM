"""Pre-registered campaign for the steady-state broken-ice transit surface.

The design is fixed *before* any case is run (``prereg/p156w2_n01_prereg.json``):
case set, thresholds, stopping rule and the analysis.  Nothing here is tuned
after a resistance number has been seen; the graded parameter ledger
(:data:`COEFFICIENT_LEDGER`) carries every constant with its grade and is hashed
into every artefact.

Decision this campaign supports
-------------------------------
Does the plan-view floe DEM -- with a zero-offset crushing law, a recycling
inlet (true steady state) and a compliant pack boundary -- reproduce the
Lindqvist (1989) level-ice resistance for a *consolidated* (C -> hexagonal
limit) broken-ice field within the predeclared 15 % band, and does its energy
ledger close?

Alternatives that are distinguished
-----------------------------------
``H1``  the DEM is limited by *measurement* (finite domain, growing plug, free
        lateral boundary): with those repaired it should reach Lindqvist in the
        consolidated limit.
``H2``  the DEM is limited by *model structure*: a plan-view disk pack has no
        out-of-plane bending/submersion mechanics, so it must stay below the
        level-ice anchor at every concentration.  H1 and H2 are separated by the
        concentration sweep (H1 predicts R/R_L -> 1 as C -> 0.907; H2 predicts a
        bounded ratio) and by the mechanism ledger (where the thrust power goes).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import torch

from ..broken_ice_resistance.hull import ShipForm
from ..broken_ice_resistance.ledger import MATERIAL
from ..ice_channel import WaterEnvironment
from ..ice_resistance_scaling import ShipCase, broken_ice_anchor
from ..level_ice.formula import TRANSPORT, IceState, lindqvist_components
from .laws import CrushingLaw
from .transit import SteadyTransit

__all__ = [
    "COEFFICIENT_LEDGER",
    "THRESHOLDS",
    "TransitSpec",
    "STANDARD_CASE",
    "CASE_SPECS",
    "ledger_sha256",
    "run_case",
    "evaluate_criteria",
    "run_campaign",
    "reference_anchors",
]

HEXAGONAL_LIMIT = math.pi / (2.0 * math.sqrt(3.0))   # 0.9069, max disk coverage

#: Constants that enter every reported number, with grade and justification.
COEFFICIENT_LEDGER: dict[str, tuple[float, str, str]] = {
    "compressive_strength_pa": (
        MATERIAL["compressive_strength_pa"], "CITED",
        "MT Uikku ice sheet III (tests 301-303), "
        "benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv"),
    "flexural_strength_pa": (
        MATERIAL["flexural_strength_pa"], "CITED", "same sheet as above"),
    "elastic_modulus_pa": (
        MATERIAL["elastic_modulus_pa"], "CITED",
        "same sheet; enters only through the contact regularisation "
        "eps = sigma_c R / E (the elastic penetration at crushing onset)"),
    "friction_coefficient": (
        0.10, "CITED",
        "cold, slow steel-on-sea-ice value pinned by docs/ICE_STEEL_FRICTION_LAW.md"),
    "water_drag_coefficient": (
        1.0, "CONVENTION",
        "form drag of a thin floe in its own plane, referenced to 2 R h "
        "(order-one bluff body; same value as tensordem.ice_channel)"),
    "normal_damping_ratio": (0.2, "CONVENTION",
                             "critical-damping fraction of the local contact oscillator"),
    "tangential_damping_ratio": (0.2, "CONVENTION", "same"),
    "heave_damping_ratio": (0.2, "CONVENTION",
                            "heave radiation damping of a flat disk"),
    "wall_damping_ratio": (0.2, "CONVENTION", "pack-boundary dashpot"),
    "pack_boundary_stiffness": (
        1.0, "DERIVED",
        "the compliant pack boundary uses the floe's own hydrostatic stiffness "
        "rho_w g A_wp: the stiffness a floe feels when it is pushed into a pack"),
    "contact_cap_diameters": (
        1.0, "DERIVED",
        "the crushing chord cannot exceed the floe diameter (full engagement)"),
    "hexagonal_coverage_limit": (
        HEXAGONAL_LIMIT, "DERIVED",
        "max coverage of equal disks in a plane; a requested concentration above "
        "it is refused instead of being silently clamped"),
    "recycling_inlet": (
        1.0, "CONVENTION",
        "floes abaft the stern re-enter ahead of the stem with zero velocity and "
        "equilibrium heave; their kinetic/potential energy is booked on the "
        "'removed' side of the ledger"),
    "ice_elastic_modulus_use": (
        float("nan"), "DERIVED",
        "E appears only in eps; the contact law itself is strength-limited, so "
        "the resistance is not sensitive to E through a bulk stiffness"),
    "literature_agreement": (
        float("nan"), "WITHHELD",
        "the MT Uikku broken-ice channel tests (304-306, 404-406) are not "
        "transcribed in this repository, so no experimental broken-ice agreement "
        "is claimed"),
}


def ledger_sha256() -> str:
    payload = json.dumps(
        {"material": {k: (v if not isinstance(v, float) or math.isfinite(v) else "nan")
                      for k, v in MATERIAL.items()},
         "ledger": {k: [(v if math.isfinite(v) else "nan"), g, t]
                    for k, (v, g, t) in sorted(COEFFICIENT_LEDGER.items())}},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


#: Predeclared thresholds.  Written before the campaign was run.
THRESHOLDS: dict[str, float] = {
    "E1_rel_error_vs_lindqvist": 0.15,      # headline gate, consolidated case
    "E2_ledger_residual_rel": 0.10,         # energy balance closure
    "E3_null_field_kN": 1.0,                # field-free negative control
    "E4_concentration_pearson_r": 0.90,     # R increases with ice concentration
    "E5_thickness_pearson_r": 0.95,         # R increases with thickness
    "E5_thickness_exponent_min": 1.0,
    "E5_thickness_exponent_max": 3.0,
    "E6_strength_pearson_r": 0.90,
    "E7_stationarity_drift": 0.15,          # |drift| over the steady window
    "E8_seed_relative_spread": 0.30,
    "E9_min_tests": 12,
    "E10_min_route_ship_lengths": 10.0,
}


@dataclass(frozen=True)
class TransitSpec:
    """One steady-state transit measurement."""

    name: str = "standard"
    concentration: float = HEXAGONAL_LIMIT - 0.007     # 0.900
    thickness_m: float = 1.0
    speed_m_s: float = 3.0
    seed: int = 7
    mean_floe_radius_m: float = 5.0                    # L_ice = 10 m
    floe_radius_range: tuple[float, float] = (2.5, 10.0)
    floe_radius_jitter: float = 0.35
    pack_width_m: float = 100.0                        # full width of the ice pack
    window_upstream_m: float = 80.0
    window_downstream_m: float = 110.0
    time_step_s: float = 5.0e-3
    route_ship_lengths: float = 10.0
    steady_start_s: float = 120.0
    record_every: int = 25
    friction_coefficient: float = 0.10
    compressive_strength_pa: float = MATERIAL["compressive_strength_pa"]
    flexural_strength_pa: float = MATERIAL["flexural_strength_pa"]
    elastic_modulus_pa: float = MATERIAL["elastic_modulus_pa"]
    hull_form: str = "ship"
    pack_boundary: bool = True
    null_field: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.concentration <= HEXAGONAL_LIMIT + 1e-12:
            raise ValueError(
                f"concentration must lie in [0, hexagonal limit {HEXAGONAL_LIMIT:.4f}]; "
                f"got {self.concentration!r} (over-unity coverage is not realisable "
                "with non-overlapping disks and is refused, not clamped)")
        for name in ("thickness_m", "speed_m_s", "mean_floe_radius_m", "pack_width_m",
                     "window_upstream_m", "window_downstream_m", "time_step_s",
                     "route_ship_lengths", "compressive_strength_pa",
                     "elastic_modulus_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive (got {value!r})")
        if self.hull_form not in ("ship", "block"):
            raise ValueError("hull_form must be 'ship' or 'block'")
        if self.pack_width_m <= 26.0:
            raise ValueError("the pack must be wider than the hull beam/2")

    # -- derived ------------------------------------------------------------
    @property
    def route_length_m(self) -> float:
        return self.route_ship_lengths * 105.0

    @property
    def simulated_time_s(self) -> float:
        return self.route_length_m / self.speed_m_s

    def as_dict(self) -> dict[str, Any]:
        body = dict(self.__dict__)
        body["route_length_m"] = self.route_length_m
        body["simulated_time_s"] = self.simulated_time_s
        body["floe_radius_range"] = list(self.floe_radius_range)
        return body

    def with_(self, **changes: Any) -> "TransitSpec":
        return replace(self, **changes)


STANDARD_CASE = TransitSpec()

CASE_SPECS: dict[str, TransitSpec] = {STANDARD_CASE.name: STANDARD_CASE}

# concentration sweep (sensitivity map axis 1)
for _c in (0.50, 0.60, 0.70, 0.80, 0.90):
    CASE_SPECS[f"conc_{_c:.2f}"] = STANDARD_CASE.with_(
        name=f"conc_{_c:.2f}", concentration=_c)
# thickness sweep (axis 2)
for _h in (0.50, 0.75, 1.25, 1.50):
    CASE_SPECS[f"thick_{_h:.2f}"] = STANDARD_CASE.with_(
        name=f"thick_{_h:.2f}", thickness_m=_h)
# strength sweep (axis 3): 1.0 MPa (warm/first-year low) and 1.5x the Uikku sheet
for _s in (1.0e6, 1.5e6, 3.7e6):
    CASE_SPECS[f"strength_{_s / 1e6:.2f}"] = STANDARD_CASE.with_(
        name=f"strength_{_s / 1e6:.2f}", compressive_strength_pa=_s)
# seed ensemble / controls / sensitivity of the declared model choices
CASE_SPECS.update({
    "seed_11": STANDARD_CASE.with_(name="seed_11", seed=11),
    "seed_23": STANDARD_CASE.with_(name="seed_23", seed=23),
    "no_ice": STANDARD_CASE.with_(name="no_ice", null_field=True,
                                  concentration=0.0),
    "no_pack_boundary": STANDARD_CASE.with_(name="no_pack_boundary",
                                            pack_boundary=False),
    "pack_narrow": STANDARD_CASE.with_(name="pack_narrow", pack_width_m=60.0),
    "pack_wide": STANDARD_CASE.with_(name="pack_wide", pack_width_m=160.0),
    "dt_fine": STANDARD_CASE.with_(name="dt_fine", time_step_s=2.5e-3),
    "block_hull": STANDARD_CASE.with_(name="block_hull", hull_form="block"),
    "floe_small": STANDARD_CASE.with_(name="floe_small", mean_floe_radius_m=2.5,
                                      floe_radius_range=(1.25, 5.0)),
    "floe_large": STANDARD_CASE.with_(name="floe_large", mean_floe_radius_m=10.0,
                                      floe_radius_range=(5.0, 20.0)),
})


def case_spec(name: str) -> TransitSpec:
    try:
        return CASE_SPECS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown case {name!r}; known: {sorted(CASE_SPECS)}") from exc


# -- references -------------------------------------------------------------
def reference_anchors(spec: TransitSpec) -> dict[str, Any]:
    """Lindqvist level-ice anchor and the in-repository broken-ice anchor."""
    hull = TRANSPORT
    ice = IceState(thickness_m=spec.thickness_m,
                   compressive_strength_pa=spec.compressive_strength_pa,
                   flexural_strength_pa=spec.flexural_strength_pa,
                   elastic_modulus_pa=spec.elastic_modulus_pa)
    lq = lindqvist_components(ice, hull, spec.speed_m_s)
    ship = ShipCase(length_m=105.0, beam_m=26.0,
                    speed_m_s=spec.speed_m_s,
                    thickness_m=spec.thickness_m,
                    strength_pa=spec.compressive_strength_pa,
                    flexural_strength_pa=spec.flexural_strength_pa,
                    entry_angle_deg=18.0,
                    friction_coefficient=spec.friction_coefficient,
                    concentration=spec.concentration,
                    floe_diameter_m=2.0 * spec.mean_floe_radius_m)
    broken = broken_ice_anchor(ship)
    momentum = 917.0 * spec.concentration * spec.thickness_m * 26.0 \
        * spec.speed_m_s ** 2
    return {
        "lindqvist": lq,
        "lindqvist_total_kN": lq["total_N"] / 1e3,
        "broken_ice_anchor": broken,
        "broken_ice_anchor_kN": broken["total_N"] / 1e3,
        "momentum_flux_kN": momentum / 1e3,
        "hexagonal_limit": HEXAGONAL_LIMIT,
    }


# -- one case ---------------------------------------------------------------
def _build_sim(spec: TransitSpec) -> SteadyTransit:
    water = WaterEnvironment(wind_velocity=(0.0, 0.0))
    law = CrushingLaw(sigma_c=spec.compressive_strength_pa,
                      modulus=spec.elastic_modulus_pa)
    hull = ShipForm(length=105.0, beam=26.0, speed=spec.speed_m_s,
                    entry_angle_deg=18.0, draught=8.0,
                    y_position=0.5 * spec.pack_width_m, x_start=0.0,
                    form=spec.hull_form)
    if spec.null_field:
        sim = SteadyTransit([0.05], [spec.thickness_m], hull=hull,
                            speed=spec.speed_m_s, law=law,
                            friction=spec.friction_coefficient, water=water,
                            window_upstream=spec.window_upstream_m,
                            window_downstream=spec.window_downstream_m,
                            pack_width=spec.pack_width_m, dt=spec.time_step_s)
        sim.pos = torch.tensor([[spec.window_upstream_m - 0.5,
                                 0.5 * spec.pack_width_m]],
                               dtype=torch.float64)
        sim.z = sim.equilibrium_heave(sim.thickness)
        sim.null_field = True
        return sim
    sim = SteadyTransit.build(
        concentration=spec.concentration,
        mean_radius=spec.mean_floe_radius_m,
        radius_jitter=spec.floe_radius_jitter,
        radius_range=spec.floe_radius_range,
        thickness=spec.thickness_m,
        hull=hull, speed=spec.speed_m_s, law=law,
        friction=spec.friction_coefficient, water=water,
        pack_width=spec.pack_width_m,
        window_upstream=spec.window_upstream_m,
        window_downstream=spec.window_downstream_m,
        dt=spec.time_step_s, seed=spec.seed,
        ice_modulus=spec.elastic_modulus_pa)
    return sim


def run_case(spec: TransitSpec, *, max_steps: int = 400_000) -> dict[str, Any]:
    """Run one transit and return the result payload (JSON-ready)."""
    torch.set_num_threads(1)
    started = time.time()
    sim = _build_sim(spec)
    if not spec.pack_boundary:
        original = sim.wall_stiffness
        sim.wall_stiffness = torch.zeros_like(original)
    steps = int(round(spec.simulated_time_s / spec.time_step_s))
    steps = min(steps, int(max_steps))
    result = sim.run(steps, record_every=spec.record_every,
                     steady_start_s=spec.steady_start_s)
    result.spec = spec.as_dict()
    result.hull = sim.hull.as_dict()
    result.state_digest = _digest(sim, result)
    result.ledger_sha256 = ledger_sha256()
    payload = result.as_dict(include_series=True)
    payload["n_steps_requested"] = steps
    payload["wall_time_s"] = time.time() - started
    payload["anchors"] = reference_anchors(spec)
    payload["initial_overlap_m"] = sim.max_initial_overlap()
    payload["mean_floe_radius_m"] = float(sim.radius.mean())
    return payload


def _digest(sim: SteadyTransit, result) -> str:
    body = f"{sim.n}|{result.mean_kN:.9e}|{result.steps}|{ledger_sha256()}"
    return hashlib.sha256(body.encode()).hexdigest()


def _run_named(name: str) -> tuple[str, dict[str, Any]]:
    return name, run_case(case_spec(name))


def run_campaign(names: Sequence[str] | None = None, *,
                 workers: int | None = None) -> dict[str, dict[str, Any]]:
    """Run the case set (in parallel) and return ``name -> payload``."""
    names = list(CASE_SPECS) if names is None else list(names)
    unknown = [n for n in names if n not in CASE_SPECS]
    if unknown:
        raise KeyError(f"unknown cases: {unknown}")
    workers = workers or min(len(names), max(1, (os.cpu_count() or 2) - 1))
    results: dict[str, dict[str, Any]] = {}
    if workers <= 1:
        for name in names:
            results[name] = run_case(case_spec(name))
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, payload in pool.map(_run_named, names):
            results[name] = payload
    return results


# -- criteria ---------------------------------------------------------------
def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _loglog_exponent(xs: Sequence[float], ys: Sequence[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(max(y, 1e-12)) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    den = sum((x - mx) ** 2 for x in lx)
    return num / den if den > 0 else float("nan")


def _verdict(ok: bool, *, honest: bool = True) -> str:
    return "PASS" if ok else ("FAIL" if honest else "WITHHELD")


def evaluate_criteria(results: dict[str, dict[str, Any]], *,
                      test_count: int | None = None) -> dict[str, Any]:
    """Evaluate the predeclared gates E1-E10 (pure function, no I/O)."""
    standard = results["standard"]
    anchor = standard["anchors"]
    lind = anchor["lindqvist_total_kN"]
    rel_error = abs(standard["mean_kN"] - lind) / lind
    criteria: dict[str, dict[str, Any]] = {}

    criteria["E1"] = {
        "statement": "consolidated broken-ice field (C=0.90): |R_DEM - R_Lindqvist| / R_Lindqvist",
        "value": rel_error, "threshold": THRESHOLDS["E1_rel_error_vs_lindqvist"],
        "dem_kN": standard["mean_kN"], "reference_kN": lind,
        "verdict": _verdict(rel_error <= THRESHOLDS["E1_rel_error_vs_lindqvist"]),
        "note": "headline gate of the campaign node; the linear broken-ice "
                "C-scaling variant is reported in the sensitivity artefact",
    }
    criteria["E2"] = {
        "statement": ("energy-ledger residual "
                      "|W_thrust + W_num + W_inj - W_rem - W_diss - dE| / W_thrust"),
        "value": standard["ledger_residual_rel"],
        "threshold": THRESHOLDS["E2_ledger_residual_rel"],
        "verdict": _verdict(standard["ledger_residual_rel"]
                            <= THRESHOLDS["E2_ledger_residual_rel"]),
        "note": "every channel is measured from the state (contact-level "
                "potentials and dashpot/Coulomb powers), not assumed",
    }
    null = results.get("no_ice")
    if null is not None:
        criteria["E3"] = {
            "statement": "field-free control: resistance of a transit with no ice in the window",
            "value": null["mean_kN"], "threshold": THRESHOLDS["E3_null_field_kN"],
            "verdict": _verdict(abs(null["mean_kN"]) <= THRESHOLDS["E3_null_field_kN"]),
        }

    conc_names = [n for n in results if n.startswith("conc_")]
    if len(conc_names) >= 3:
        conc = sorted(((results[n]["spec"]["concentration"], results[n]["mean_kN"])
                       for n in conc_names))
        r = _pearson([c for c, _ in conc], [v for _, v in conc])
        ratio = [v / lind if lind > 0 else float("nan") for _, v in conc]
        criteria["E4"] = {
            "statement": "resistance increases with ice concentration (Pearson r)",
            "value": r, "threshold": THRESHOLDS["E4_concentration_pearson_r"],
            "points": [{"concentration": c, "R_kN": v, "R_over_lindqvist": rr}
                       for (c, v), rr in zip(conc, ratio)],
            "verdict": _verdict(r >= THRESHOLDS["E4_concentration_pearson_r"]),
        }
    thick_names = [n for n in results if n.startswith("thick_")] + ["standard"]
    if len(thick_names) >= 3:
        thick = sorted(((results[n]["spec"]["thickness_m"], results[n]["mean_kN"])
                        for n in thick_names))
        r = _pearson([h for h, _ in thick], [v for _, v in thick])
        exponent = _loglog_exponent([h for h, _ in thick], [v for _, v in thick])
        criteria["E5"] = {
            "statement": "resistance increases with ice thickness; R ~ h^m with m in [1, 3]",
            "value": r, "exponent": exponent,
            "threshold": THRESHOLDS["E5_thickness_pearson_r"],
            "exponent_band": [THRESHOLDS["E5_thickness_exponent_min"],
                              THRESHOLDS["E5_thickness_exponent_max"]],
            "points": [{"thickness_m": h, "R_kN": v} for h, v in thick],
            "verdict": _verdict(r >= THRESHOLDS["E5_thickness_pearson_r"]
                                and THRESHOLDS["E5_thickness_exponent_min"] <= exponent
                                <= THRESHOLDS["E5_thickness_exponent_max"]),
        }
    strength_names = [n for n in results if n.startswith("strength_")] + ["standard"]
    if len(strength_names) >= 3:
        strength = sorted(((results[n]["spec"]["compressive_strength_pa"],
                            results[n]["mean_kN"]) for n in strength_names))
        r = _pearson([s for s, _ in strength], [v for _, v in strength])
        criteria["E6"] = {
            "statement": "resistance increases with crushing strength (Pearson r)",
            "value": r, "threshold": THRESHOLDS["E6_strength_pearson_r"],
            "points": [{"sigma_c_Pa": s, "R_kN": v} for s, v in strength],
            "verdict": _verdict(r >= THRESHOLDS["E6_strength_pearson_r"]),
        }
    criteria["E7"] = {
        "statement": "steady state: |linear drift over the averaging window| / mean",
        "value": abs(standard["stationarity_drift"]),
        "threshold": THRESHOLDS["E7_stationarity_drift"],
        "verdict": _verdict(abs(standard["stationarity_drift"])
                            <= THRESHOLDS["E7_stationarity_drift"]),
    }
    seed_names = [n for n in results if n == "standard" or n.startswith("seed_")]
    if len(seed_names) >= 3:
        values = [results[n]["mean_kN"] for n in seed_names]
        spread = (max(values) - min(values)) / (sum(values) / len(values))
        criteria["E8"] = {
            "statement": "seed ensemble: (max - min) / mean of the steady resistance",
            "value": spread, "threshold": THRESHOLDS["E8_seed_relative_spread"],
            "seeds": {n: results[n]["spec"]["seed"] for n in seed_names},
            "verdict": _verdict(spread <= THRESHOLDS["E8_seed_relative_spread"]),
        }
    if test_count is not None:
        criteria["E9"] = {
            "statement": "pytest coverage of the new surface",
            "value": float(test_count), "threshold": THRESHOLDS["E9_min_tests"],
            "verdict": _verdict(test_count >= THRESHOLDS["E9_min_tests"]),
        }
    criteria["E10"] = {
        "statement": "transit route covers at least 10 ship lengths",
        "value": standard["route_length_m"] / 105.0,
        "threshold": THRESHOLDS["E10_min_route_ship_lengths"],
        # the route is 10.000 ship lengths by construction; only the floating
        # point representation of the last sample is excluded
        "verdict": _verdict(standard["route_length_m"] / 105.0
                            >= THRESHOLDS["E10_min_route_ship_lengths"] * (1.0 - 1e-6)),
    }
    passed = sum(1 for c in criteria.values() if c["verdict"] == "PASS")
    return {
        "criteria": criteria,
        "summary": {
            "n_criteria": len(criteria),
            "n_pass": passed,
            "n_fail": sum(1 for c in criteria.values() if c["verdict"] == "FAIL"),
            "n_withheld": sum(1 for c in criteria.values() if c["verdict"] == "WITHHELD"),
            "headline": criteria["E1"]["verdict"],
        },
        "thresholds": dict(THRESHOLDS),
        "ledger_sha256": ledger_sha256(),
    }