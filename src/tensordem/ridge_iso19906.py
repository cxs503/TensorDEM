"""ISO 19906-form grounding load vs. the DEM ridge-keel / buried-pipeline runs.

The campaign node IN155-N11 asks for the ridge-keel grounding load in the
regulatory form

    V_max = N_q * gamma' * h_k * D + N_c * c * D
    H_max = N_p * s_u * D                     (undrained lateral break-out)

and for a cross-check against the coupled DEM.  This module implements that
form exactly as a *term-by-term* decomposition, so every number in the report
can be traced to a bearing-capacity factor and an overburden depth.

Two surcharge conventions are provided, because they answer two different
questions and the difference between them is the whole physics of keel
grounding:

``"pipe_invert"`` (default, used for the DEM comparison)
    ``h_k = cover_depth + D`` is the depth of the *pipe invert* below the
    mudline.  The overburden that acts on the pipe is the soil column above it,
    so this is the bearing check of the pipe-soil system.

``"keel"`` (the literal campaign form)
    ``h_k`` is the ice-ridge keel depth.  A keel that has gouged down to that
    depth has *displaced* the soil column, so the N_q term is an upper bound
    that cannot be mobilised; the module returns it so the report can show how
    far the literal form sits above the DEM.

Evidence grade of the coefficients: the N_q / N_c / N_gamma trio and the
Skempton embedment factor are the closed forms already verified in
:mod:`tensordem.ridge_geotech` (Vesic 1973; Skempton 1951).  The ISO 19906
primary text is **not** in this worktree, so the *attribution* of the exact
regulatory coefficients is WITHHELD; the arithmetic of the campaign-cited form
is REAL and is re-derived in ``docs/ice_ridge_iso19906_grounding.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .ridge_geotech import (MohrCoulombSoil, PipeSection, bearing_capacity_factors,
                            skempton_undrained_nc)

__all__ = [
    "iso19906_vertical_load",
    "iso19906_lateral_load",
    "instability_limits",
    "critical_cover_depth",
    "compare_with_dem",
    "cover_envelope",
    "energy_convergence_table",
    "grid_convergence_table",
]


@dataclass(frozen=True)
class ISOForm:
    """Term-by-term value of ``V = N_q gamma' h_k D + N_c c D``."""

    n_q: float
    n_c: float
    n_gamma: float
    overburden_stress: float
    cohesion: float
    diameter: float
    depth: float
    q_term: float
    c_term: float
    gamma_term: float

    @property
    def total(self) -> float:
        return self.q_term + self.c_term + self.gamma_term

    def as_dict(self) -> dict:
        return {
            "n_q": self.n_q, "n_c": self.n_c, "n_gamma": self.n_gamma,
            "overburden_stress_pa": self.overburden_stress,
            "cohesion_pa": self.cohesion, "diameter_m": self.diameter,
            "depth_m": self.depth,
            "q_term_n_per_m": self.q_term, "c_term_n_per_m": self.c_term,
            "gamma_term_n_per_m": self.gamma_term,
            "total_n_per_m": self.total,
        }


def iso19906_vertical_load(soil: MohrCoulombSoil, diameter: float, *,
                           cover_depth: float,
                           keel_depth: float | None = None,
                           surcharge_mode: str = "pipe_invert",
                           include_gamma_term: bool = False,
                           include_depth_factors: bool = True) -> ISOForm:
    """ISO 19906-form vertical grounding load per metre of pipeline.

    ``surcharge_mode`` selects the overburden depth (see the module docstring).
    ``include_gamma_term`` switches the ``0.5 N_gamma gamma' D^2`` self-weight
    term, which the campaign form omits; ``include_depth_factors`` switches the
    Vesic depth factors ``d_c``/``d_q`` (drained branch only).
    """
    if diameter <= 0.0 or not math.isfinite(diameter):
        raise ValueError("diameter must be finite and positive")
    if cover_depth < 0.0 or not math.isfinite(cover_depth):
        raise ValueError("cover_depth must be finite and nonnegative")
    if surcharge_mode not in ("pipe_invert", "keel"):
        raise ValueError("surcharge_mode must be 'pipe_invert' or 'keel'")
    if surcharge_mode == "keel":
        if keel_depth is None:
            raise ValueError("surcharge_mode='keel' needs keel_depth")
        depth = float(keel_depth)
    else:
        depth = cover_depth + diameter
    gamma_prime = soil.effective_unit_weight * (1.0 - soil.pore_pressure_ratio)
    if soil.undrained:
        # total-stress branch: N_q does not apply, the surcharge is carried by
        # the undrained cohesion term through the Skempton embedment factor.
        n_c = skempton_undrained_nc(depth, diameter, length_ratio=0.0)
        n_q = 1.0
        n_gamma = 0.0
        sigma_v = 0.0
        c_term = soil.shear_strength * n_c * diameter
        q_term = 0.0
        gamma_term = 0.0
    else:
        n_c, n_q, n_gamma = bearing_capacity_factors(soil.phi_rad)
        k = min(depth / diameter, 1.0)
        d_c = 1.0 + 0.4 * k if include_depth_factors else 1.0
        d_q = (1.0 + 2.0 * math.tan(soil.phi_rad)
               * (1.0 - math.sin(soil.phi_rad)) ** 2 * k) \
            if include_depth_factors else 1.0
        sigma_v = soil.effective_vertical_stress(depth)
        c_term = soil.cohesion * n_c * d_c * diameter
        q_term = sigma_v * n_q * d_q * diameter
        gamma_term = 0.5 * gamma_prime * diameter * n_gamma * diameter \
            if include_gamma_term else 0.0
    return ISOForm(n_q=n_q, n_c=n_c,
                   n_gamma=0.0 if soil.undrained else n_gamma,
                   overburden_stress=sigma_v, cohesion=soil.shear_strength,
                   diameter=diameter, depth=depth, q_term=q_term,
                   c_term=c_term, gamma_term=gamma_term)


def iso19906_lateral_load(soil: MohrCoulombSoil, diameter: float, *,
                          cover_depth: float) -> dict:
    """ISO 19906-form lateral break-out load ``H = N_p s_u D`` per metre.

    Undrained soils use ``N_p = min(2 + 1.5 z/D, 9)`` (the lateral limit
    pressure already verified in :mod:`tensordem.ridge_geotech`); drained soils
    use the Coulomb passive wedge.  The ``N_c``-style factor and the cohesion
    term are reported separately so the criterion "H vs N_c theory" is
    traceable.
    """
    if diameter <= 0.0 or cover_depth < 0.0:
        raise ValueError("diameter must be positive and cover_depth nonnegative")
    total = soil.lateral_limit_pressure(cover_depth, diameter)
    embedment = cover_depth + 0.5 * diameter
    if soil.undrained:
        n_factor = min(2.0 + 1.5 * embedment / diameter, 9.0)
        cohesion_term = total
        surcharge_term = 0.0
    else:
        n_factor = 0.0
        k_p = math.tan(0.25 * math.pi + 0.5 * soil.phi_rad) ** 2
        gamma_prime = soil.effective_unit_weight * (1.0 - soil.pore_pressure_ratio)
        surcharge_term = 0.5 * gamma_prime * embedment ** 2 * k_p
        cohesion_term = 2.0 * soil.cohesion * embedment * math.sqrt(k_p)
    return {"total_n_per_m": total, "n_factor": n_factor,
            "cohesion_term_n_per_m": cohesion_term,
            "surcharge_term_n_per_m": surcharge_term,
            "embedment_m": embedment}


def instability_limits(section: PipeSection) -> dict:
    """Displacement limits of the cover-instability criterion."""
    diameter = section.outer_diameter
    return {"vertical": diameter / 10.0, "lateral": diameter / 5.0,
            "diameter": diameter}


def critical_cover_depth(soil: MohrCoulombSoil, diameter: float, demand: float, *,
                         lower: float = 0.0, upper: float = 40.0,
                         tol: float = 1.0e-4) -> float:
    """Smallest cover depth for which ``V_ISO(cover) >= demand`` (bisection).

    ``demand`` is the ice-side line load the shallow-cover / rigid-pipe runs
    deliver.  Returns ``0.0`` if even the mudline configuration already
    resists, and ``inf`` if the bracket does not contain a root.
    """
    if demand <= 0.0:
        return 0.0

    def capacity(cover: float) -> float:
        return iso19906_vertical_load(soil, diameter, cover_depth=cover).total

    if capacity(lower) >= demand:
        return 0.0
    if capacity(upper) < demand:
        return math.inf
    lo, hi = lower, upper
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if capacity(mid) >= demand:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def compare_with_dem(dem_load: float, iso_load: float) -> dict:
    """Signed deviation of the DEM load from the ISO-form load."""
    if iso_load == 0.0:
        raise ValueError("iso_load must be nonzero")
    deviation = (dem_load - iso_load) / iso_load
    return {"dem_n_per_m": dem_load, "iso_n_per_m": iso_load,
            "ratio": dem_load / iso_load, "deviation": deviation,
            "dem_below_iso": dem_load <= iso_load,
            "abs_deviation": abs(deviation)}


def cover_envelope(rows: Iterable[Mapping], *, diameter: float,
                   vertical_limit: float, lateral_limit: float) -> dict:
    """Turn per-cover DEM rows into the stability envelope and its verdicts.

    Each row must carry ``cover_depth``, ``dem_capacity`` (compliant pipe),
    ``iso_vertical``, ``dem_displacement``, ``dem_demand`` (rigid pipe),
    ``iso_lateral`` and ``dem_lateral_displacement``.
    """
    table = []
    for row in rows:
        ratio = row["cover_depth"] / diameter
        vertical_ratio = row["dem_displacement"] / vertical_limit
        lateral_ratio = row.get("dem_lateral_displacement", 0.0) / lateral_limit
        unstable = (vertical_ratio > 1.0) or (lateral_ratio > 1.0)
        predicted = row["dem_demand"] > row["iso_vertical"]
        table.append({
            "cover_depth_m": row["cover_depth"],
            "cover_over_d": ratio,
            "dem_capacity_n_per_m": row["dem_capacity"],
            "iso_vertical_n_per_m": row["iso_vertical"],
            "capacity_ratio": row["dem_capacity"] / row["iso_vertical"],
            "capacity_deviation": (row["dem_capacity"] - row["iso_vertical"])
            / row["iso_vertical"],
            "iso_lateral_n_per_m": row["iso_lateral"],
            "dem_lateral_n_per_m": row.get("dem_lateral", 0.0),
            "lateral_ratio": (row.get("dem_lateral", 0.0) / row["iso_lateral"]
                              if row["iso_lateral"] else float("nan")),
            "dem_demand_n_per_m": row["dem_demand"],
            "vertical_displacement_m": row["dem_displacement"],
            "vertical_displacement_over_limit": vertical_ratio,
            "lateral_displacement_m": row.get("dem_lateral_displacement", 0.0),
            "lateral_displacement_over_limit": lateral_ratio,
            "dem_unstable": unstable,
            "iso_predicted_unstable": predicted,
            "prediction_consistent": unstable == predicted,
        })
    consistent = sum(1 for row in table if row["prediction_consistent"])
    deviations = [row["capacity_deviation"] for row in table]
    return {
        "rows": table,
        "cases": len(table),
        "consistent_predictions": consistent,
        "max_abs_capacity_deviation": max(abs(d) for d in deviations) if deviations else None,
        "verdict": ("consistent" if consistent == len(table) else "inconsistent"),
    }


def energy_convergence_table(rows: Iterable[Mapping]) -> dict:
    """Closure error of the energy ledger against the time step."""
    table = []
    for row in rows:
        table.append({"dt": row["dt"], "closure_error": row["closure_error"],
                      "residual_j": row["residual"],
                      "input_j": row["sources"]["total"],
                      "steps": row.get("steps")})
    table.sort(key=lambda item: item["dt"], reverse=True)
    ratios = []
    for coarse, fine in zip(table, table[1:]):
        if fine["closure_error"] > 0.0:
            ratios.append(coarse["closure_error"] / fine["closure_error"])
    best = table[-1] if table else None
    return {"rows": table, "refinement_ratios": ratios,
            "finest_closure": best["closure_error"] if best else None,
            "satisfied": bool(best and best["closure_error"] < 0.05)}


def grid_convergence_table(rows: Iterable[Mapping]) -> dict:
    """Grid independence: capacity change when the particle count doubles."""
    table = sorted(({"particle_radius": row["particle_radius"],
                     "particles": row["particles"],
                     "capacity": row["capacity"],
                     "relative": row.get("relative")} for row in rows),
                   key=lambda item: -item["particle_radius"])
    changes = []
    for coarse, fine in zip(table, table[1:]):
        if coarse["capacity"] == 0.0:
            continue
        changes.append((fine["capacity"] - coarse["capacity"]) / coarse["capacity"])
    change = max((abs(c) for c in changes), default=None)
    return {"rows": table, "particle_ratio":
            (table[-1]["particles"] / table[0]["particles"]) if len(table) > 1 else 1.0,
            "capacity_change": change,
            "satisfied": bool(change is not None and change < 0.05)}