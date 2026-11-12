"""Multi-ship column formation resistance in a broken-ice channel (W25).

Purpose
-------
The ``formation-ice-channel`` research case asks two separate questions:

1. **Absolute caliber** -- how far is the DEM single-ship ice resistance from
   the Lindqvist (1989) three-component level-ice anchor once the anchor is
   converted to the case caliber?
2. **Relative caliber** -- how does the ice-resistance *saving* of a ``n``-ship
   column formation depend on the ship-to-ship spacing ``S`` (in ship lengths
   ``L``) and on the transit speed ``V``?

This module supplies the reusable machinery for both, so the experiment script
contains no physics and no arithmetic:

* :class:`FormationCase` -- a frozen description of one DEM run (one or more
  prescribed hulls traversing one ice field);
* :func:`build_simulation` -- constructs the field and the hulls from the case,
  including a heave-equilibrium initial condition so the run starts without a
  buoyancy transient;
* :func:`run_case` -- advances the DEM and accumulates the per-hull ice-load
  ledger, restricted to the declared measurement window;
* :func:`savings_rate` / :func:`formation_savings` -- the formation saving
  metric and its per-hull breakdown;
* :func:`gci_three_grid` -- Roache grid-convergence index for a
  three-level refinement family, with an explicit *monotone / oscillatory*
  verdict (a GCI is only meaningful when the family converges monotonically);
* :func:`lindqvist_reference` -- the same-caliber Lindqvist anchor plus the
  ``R_dem`` vs anchor deviation and the *out-of-domain* test
  ``R_dem < R_bending + R_submersion`` (see :func:`domain_verdict`);
* :func:`channel_reclosure_time` -- closes the loop on *why* the saving depends
  on spacing: the lead leaves a cleared corridor that a cross-channel drift
  re-fills in ``~w/(2 u_drift)`` seconds, which is compared with the lag
  ``S/V`` of the follower.

Measurement window (declared, not tuned)
----------------------------------------
A hull contributes samples only while its centre lies inside the ice field at
least ``L/2`` from either longitudinal edge and ``t >= warmup_s``.  The
reported per-hull load is the arithmetic mean of the hull's *upstream*
(``-x``) contact reaction over those samples; the formation total is the sum
over hulls, which is what a tug would have to pull for the column.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
    analytic_equilibrium_draught,
)
from .level_ice.formula import HullForm, IceState, lindqvist_components

__all__ = [
    "FormationCase",
    "build_simulation",
    "run_case",
    "savings_rate",
    "formation_savings",
    "gci_three_grid",
    "lindqvist_reference",
    "domain_verdict",
    "channel_reclosure_time",
    "formation_geometry",
    "carve_launch_footprints",
]

#: Nominal case caliber of ``formation-ice-channel`` (kept in one place so the
#: runner, the collector and the tests cannot drift apart).
CASE_HULL_LENGTH_M = 80.0
CASE_HULL_BEAM_M = 20.0
CASE_ICE_THICKNESS_M = 0.7
CASE_CONCENTRATION = 0.8
CASE_MEAN_RADIUS_M = 10.0
CASE_DRIFT_M_S = 0.05
CASE_SEED = 20261112


@dataclass(frozen=True)
class FormationCase:
    """One DEM run: ``n_ships`` hulls in column, spacing ``spacing_L * L``.

    ``spacing_L is None`` (or ``n_ships == 1``) is the *degenerate single-ship*
    case used for the Lindqvist reconciliation.
    """

    label: str
    n_ships: int = 3
    spacing_L: float | None = None
    speed_m_s: float = 1.0
    # hull caliber
    ship_length_m: float = CASE_HULL_LENGTH_M
    ship_beam_m: float = CASE_HULL_BEAM_M
    # ice caliber
    concentration: float = CASE_CONCENTRATION
    mean_radius_m: float = CASE_MEAN_RADIUS_M
    ice_thickness_m: float = CASE_ICE_THICKNESS_M
    field_width_m: float = 70.0
    drift_m_s: float = CASE_DRIFT_M_S
    # numerics
    dt: float = 0.008
    seed: int = CASE_SEED
    warmup_s: float = 60.0
    sample_every: int = 5
    mesh_level: int = 1            # 1 = base, 2 = N x2, 4 = N x4
    role: str = "formation"        # "formation" | "single" | "gci"
    notes: str = ""

    # -- derived geometry ---------------------------------------------------
    @property
    def spacing_m(self) -> float:
        if self.n_ships == 1:
            return 0.0
        if self.spacing_L is None:
            raise ValueError(f"{self.label}: n_ships={self.n_ships} needs spacing_L")
        return float(self.spacing_L) * self.ship_length_m

    @property
    def span_m(self) -> float:
        """Centre-to-centre length of the column (0 for a single ship)."""
        return (self.n_ships - 1) * self.spacing_m

    @property
    def field_length_m(self) -> float:
        """Ice-field length: the column span plus two hull lengths of ice.

        Two hull lengths of ice ahead of the lead guarantee that every hull
        (including the trailing one) spends ``>= L`` of travel inside the field
        after the entry transient, at every spacing in the sweep.
        """
        # the column span plus two hull lengths of ice; the single-ship case
        # (the denominator of every saving) gets a longer field so that its time
        # mean is averaged over many more floe encounters
        minimum = 6.0 if self.n_ships == 1 else 4.0
        return max(self.span_m + 2.0 * self.ship_length_m,
                   minimum * self.ship_length_m)

    @property
    def mean_radius_refined_m(self) -> float:
        """Mean floe radius at this mesh level (``N x level`` keeps coverage)."""
        return self.mean_radius_m / math.sqrt(self.mesh_level)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(
            spacing_m=self.spacing_m,
            span_m=self.span_m,
            field_length_m=self.field_length_m,
            mean_radius_refined_m=self.mean_radius_refined_m,
        )
        return out


def formation_geometry(case: FormationCase) -> dict[str, float]:
    """Hull start positions and the measurement window, in metres."""
    length = case.ship_length_m
    # trailing hull starts one hull length upstream of the ice edge so that
    # every hull performs a genuine entry; the lead starts `span` ahead of it.
    x_trailing = -0.5 * length
    x_lead = x_trailing + case.span_m
    window_lo = 0.5 * length
    window_hi = case.field_length_m - 0.5 * length
    return {
        "x_trailing_start_m": x_trailing,
        "x_lead_start_m": x_lead,
        "x_trailing_exit_m": window_hi,
        "window_lo_m": window_lo,
        "window_hi_m": window_hi,
        "window_length_m": window_hi - window_lo,
        "travel_m": window_hi - x_trailing,
    }


def build_simulation(case: FormationCase) -> tuple[IceChannelSimulation, tuple[HullGeometry, ...]]:
    """Construct the ice field and the ``n_ships`` prescribed hulls of ``case``."""
    geom = formation_geometry(case)
    water = WaterEnvironment(
        wind_velocity=(0.0, 0.0),
        current_velocity=(0.0, case.drift_m_s),
    )
    hulls = tuple(
        HullGeometry(
            length=case.ship_length_m,
            beam=case.ship_beam_m,
            speed=case.speed_m_s,
            x_start=geom["x_lead_start_m"] - index * case.spacing_m,
            y_position=0.0,
        )
        for index in range(case.n_ships)
    )
    sim = IceChannelSimulation.from_field(
        case.concentration,
        domain_length=case.field_length_m,
        domain_width=case.field_width_m,
        seed=case.seed,
        mean_radius=case.mean_radius_refined_m,
        thickness_range=(case.ice_thickness_m, case.ice_thickness_m),
        dt=case.dt,
        contact=ContactParameters(),
        water=water,
        hull=hulls[0],
        extra_hulls=hulls[1:],
    )
    # heave equilibrium initial condition: no buoyancy transient at t = 0
    d_eq = analytic_equilibrium_draught(
        sim.water.ice_density, sim.water.water_density, case.ice_thickness_m)
    sim.z = torch.full_like(sim.z, 0.5 * case.ice_thickness_m - d_eq)
    return sim, hulls


def carve_launch_footprints(sim: IceChannelSimulation, hulls: Sequence[HullGeometry],
                            *, margin_m: float = 0.0) -> tuple[IceChannelSimulation, int]:
    """Remove the floes that overlap a hull at ``t = 0``.

    A hull that starts **inside** the ice field (which happens whenever the
    column span exceeds half a ship length, i.e. for every spacing in this
    sweep) would otherwise begin the run with a deeply overlapping contact and
    launch floes at several m/s.  Removing exactly the overlapping floes is the
    minimal initial condition that represents "the ships are already in the
    corridor they have cleared"; the number of removed floes is reported so the
    local concentration loss is visible.  The reported measurement window starts
    far enough downstream that it is unaffected.
    """
    if sim.n == 0:
        return sim, 0
    keep = torch.ones(sim.n, dtype=torch.bool)
    for hull in hulls:
        x_c = hull.x_centre(0.0)
        y_c = float(hull.y_position)
        local_x = sim.pos[:, 0] - x_c
        local_y = sim.pos[:, 1] - y_c
        dx = (local_x.abs() - hull.half_length).clamp(min=0.0)
        dy = (local_y.abs() - hull.half_beam).clamp(min=0.0)
        # inside the box -> distance to the hull is zero
        inside = (local_x.abs() <= hull.half_length) & (local_y.abs() <= hull.half_beam)
        dist = torch.sqrt(dx * dx + dy * dy)
        dist = torch.where(inside, torch.zeros_like(dist), dist)
        overlaps = dist <= (sim.radius + margin_m)
        keep &= ~overlaps
    removed = int((~keep).sum())
    if removed == 0:
        return sim, 0
    rebuilt = IceChannelSimulation(
        radius=sim.radius[keep].tolist(),
        thickness=sim.thickness[keep].tolist(),
        water=sim.water,
        contact=sim.contact,
        hull=sim.hulls[0],
        extra_hulls=tuple(sim.hulls[1:]),
        dt=sim.dt,
        integrator=sim.integrator,
        dtype=sim.dtype,
    )
    rebuilt.pos = sim.pos[keep].clone()
    return rebuilt, removed


def run_case(case: FormationCase, *, progress: bool = False) -> dict[str, Any]:
    """Advance the DEM and return the per-hull ice-load ledger of ``case``.

    The returned dict holds *raw measured* quantities only (means, sample
    counts, contact fractions, and a decimated time series for plotting).  All
    derived ratios live in :func:`formation_savings` / :func:`lindqvist_reference`
    so that no interpretation leaks into the measurement step.
    """
    sim, hulls = build_simulation(case)
    sim, n_carved = carve_launch_footprints(sim, hulls)
    geom = formation_geometry(case)
    n_steps = int(round(geom["travel_m"] / (case.speed_m_s * case.dt)))

    n_hulls = len(hulls)
    series_t: list[float] = []
    series_f: list[list[float]] = [[] for _ in range(n_hulls)]
    inside_window: list[bool] = [False] * n_hulls
    load_sum = [0.0] * n_hulls
    load_abs_sum = [0.0] * n_hulls
    load_contact = [0] * n_hulls
    samples = [0] * n_hulls
    max_abs_load = 0.0
    blew_up = False

    for step in range(n_steps):
        sim.step()
        if (step + 1) % case.sample_every:
            continue
        t = sim.time
        ledger = sim.hull_contact(t)
        row: list[float] = []
        for index, entry in enumerate(ledger):
            hull = hulls[index]
            x_c = entry["x_centre_m"]
            force = entry["force_on_hull_N"]
            f_upstream = -float(force[0])          # upstream reaction = resistance
            magnitude = float(torch.linalg.norm(force))
            in_window = (t >= case.warmup_s) and (geom["window_lo_m"] <= x_c <= geom["window_hi_m"])
            inside_window[index] = in_window
            if in_window:
                load_sum[index] += f_upstream
                load_abs_sum[index] += magnitude
                if entry["contact_count"] > 0:
                    load_contact[index] += 1
                samples[index] += 1
                row.append(f_upstream)
            else:
                row.append(float("nan"))
            max_abs_load = max(max_abs_load, magnitude)
        series_t.append(t)
        for index in range(n_hulls):
            series_f[index].append(row[index])
        if not (max_abs_load == max_abs_load) or max_abs_load > 1.0e10 or sim.time != sim.time:
            blew_up = True
            break
        if progress and step % max(1, n_steps // 10) == 0:
            print(f"  {case.label}: step {step}/{n_steps} t={sim.time:.1f}s", flush=True)

    per_hull = []
    for index, hull in enumerate(hulls):
        n_s = samples[index]
        mean_load = load_sum[index] / n_s if n_s else float("nan")
        mean_abs = load_abs_sum[index] / n_s if n_s else float("nan")
        per_hull.append({
            "index": index,
            "x_start_m": hull.x_start,
            "samples": n_s,
            "mean_upstream_load_N": mean_load,
            "mean_abs_load_N": mean_abs,
            "contact_fraction": load_contact[index] / n_s if n_s else float("nan"),
        })

    total = sum(item["mean_upstream_load_N"] for item in per_hull)
    fields = sim.field_statistics()
    return {
        "label": case.label,
        "case": case.to_dict(),
        "geometry": geom,
        "steps_planned": n_steps,
        "steps_done": sim.steps,
        "sim_time_s": sim.time,
        "dt": case.dt,
        "blew_up": blew_up,
        "max_abs_load_N": max_abs_load,
        "n_floes": sim.n,
        "n_floes_carved_at_launch": n_carved,
        "realised_concentration": fields["realised_concentration"],
        "max_initial_overlap_m": fields["max_initial_overlap"],
        "per_hull": per_hull,
        "formation_total_load_N": total,
        "series": {
            "t": series_t,
            "loads_N": series_f,
        },
    }


# ---------------------------------------------------------------------------
# formation saving metric
# ---------------------------------------------------------------------------
def savings_rate(formation_total_N: float, single_ship_N: float,
                 n_ships: int) -> float:
    """Ice-resistance saving of the column relative to ``n_ships`` lone ships.

    ``1 - R_formation_total / (n_ships * R_single)``: 1.0 means the column costs
    nothing, 0.0 means the column is indistinguishable from ``n`` independent
    ships, negative means the column is worse.
    """
    for name, value in (("formation_total_N", formation_total_N),
                        ("single_ship_N", single_ship_N)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if single_ship_N <= 0.0:
        raise ValueError("single_ship_N must be positive (a zero reference "
                         "makes the saving undefined)")
    if n_ships < 1:
        raise ValueError("n_ships must be >= 1")
    return 1.0 - formation_total_N / (n_ships * single_ship_N)


def formation_savings(formation: dict[str, Any], single: dict[str, Any]) -> dict[str, Any]:
    """Saving of a formation run against a single-ship run, with breakdown."""
    case = formation["case"]
    n_ships = int(case["n_ships"])
    reference = float(single["per_hull"][0]["mean_upstream_load_N"])
    total = float(formation["formation_total_load_N"])
    per_hull_saving = [
        savings_rate(float(item["mean_upstream_load_N"]), reference, 1)
        for item in formation["per_hull"]
    ]
    return {
        "n_ships": n_ships,
        "spacing_L": case["spacing_L"],
        "spacing_m": case["spacing_m"],
        "speed_m_s": case["speed_m_s"],
        "single_ship_load_N": reference,
        "formation_total_load_N": total,
        "independent_fleet_load_N": n_ships * reference,
        "saving": savings_rate(total, reference, n_ships),
        "per_hull_saving": per_hull_saving,
        "per_hull_load_N": [float(item["mean_upstream_load_N"]) for item in formation["per_hull"]],
        "per_hull_contact_fraction": [float(item["contact_fraction"]) for item in formation["per_hull"]],
        "follower_lead_ratio": [
            float(item["mean_upstream_load_N"]) / reference for item in formation["per_hull"]
        ],
    }


def monotonicity(values: Sequence[float], *, sign: int = -1,
                 tolerance: float = 0.0) -> dict[str, Any]:
    """Test whether ``values`` are monotone in the requested direction.

    ``sign = -1`` asks for non-increasing (saving falls with spacing), ``+1``
    for non-decreasing.  ``tolerance`` is an absolute slack that is counted as
    a tie (used to keep contact-noise plateaus from reading as reversals).
    """
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    violations = [d for d in diffs if sign * d < -tolerance]
    strict = [d for d in diffs if sign * d > tolerance]
    return {
        "values": list(values),
        "differences": diffs,
        "direction": "non-increasing" if sign < 0 else "non-decreasing",
        "tolerance": tolerance,
        "violations": violations,
        "n_violations": len(violations),
        "n_strict_steps": len(strict),
        "monotone": len(violations) == 0,
    }


# ---------------------------------------------------------------------------
# grid convergence (Roache GCI) for a three-level refinement family
# ---------------------------------------------------------------------------
def gci_three_grid(fine: float, medium: float, coarse: float, *,
                   refinement: float = 2.0, safety: float = 1.25) -> dict[str, Any]:
    """Roache grid-convergence index of a three-grid family (``fine`` is finest).

    ``p = ln|(f3-f2)/(f2-f1)| / ln(r)``, ``GCI_fine = Fs |(f2-f1)/f1| / (r^p-1)``.
    An oscillatory family (ratio <= 0) returns ``converged=False`` and
    ``gci_pct = nan``: a GCI is not defined without monotone convergence.
    """
    values = [float(fine), float(medium), float(coarse)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite member in grid family {values}")
    if refinement <= 1.0:
        raise ValueError("refinement must be > 1")
    d21 = values[1] - values[0]
    d32 = values[2] - values[1]
    out = {
        "values_fine_medium_coarse": values,
        "refinement": float(refinement),
        "delta_medium_fine": d21,
        "delta_coarse_medium": d32,
        "relative_step_pct": 100.0 * abs(d21) / abs(values[0]) if values[0] else float("nan"),
        "observed_order": float("nan"),
        "gci_pct": float("nan"),
        "converged": False,
        "monotone": bool(abs(d32) >= abs(d21) and d21 * d32 > 0.0),
        "asymptotic_ratio": float("nan"),
    }
    if values[0] == 0.0 or d21 == 0.0:
        # an exactly converged pair: GCI is zero by definition
        out.update(observed_order=float("nan"), gci_pct=0.0, converged=True,
                   note="zero medium-fine difference")
        return out
    ratio = d32 / d21 if d21 else float("nan")
    if not math.isfinite(ratio) or ratio <= 0.0:
        out["note"] = "sign change between successive differences: oscillatory family"
        return out
    p = abs(math.log(ratio) / math.log(refinement))
    gci = safety * abs(d21 / values[0]) / (refinement ** p - 1.0)
    out.update(observed_order=p, gci_pct=100.0 * gci, converged=True,
               asymptotic_ratio=ratio,
               note="Roache GCI with safety factor %.2f" % safety)
    return out


# ---------------------------------------------------------------------------
# Lindqvist anchor in the case caliber
# ---------------------------------------------------------------------------
#: literature-typical stems used for the caliber band (the DEM hull is a box,
#: i.e. 90 deg, which is outside every finite-angle formula: declared a priori)
CALIBER_ALPHA_BAND_DEG = (15.0, 20.0, 30.0)
CALIBER_MU_BAND = (0.05, 0.10, 0.15)
CALIBER_ALPHA_PRIMARY_DEG = 20.0
CALIBER_MU_PRIMARY = 0.05


def lindqvist_reference(case: FormationCase, speed_m_s: float, *,
                        alpha_deg: float = CALIBER_ALPHA_PRIMARY_DEG,
                        mu: float = CALIBER_MU_PRIMARY,
                        water_density_kg_m3: float = 1025.0) -> dict[str, Any]:
    """Lindqvist three-component anchor converted to the case caliber."""
    ice = IceState(thickness_m=case.ice_thickness_m)
    hull = HullForm(
        length_m=case.ship_length_m,
        beam_m=case.ship_beam_m,
        draught_m=analytic_equilibrium_draught(917.0, water_density_kg_m3,
                                               case.ice_thickness_m) + 0.0,
        entry_angle_deg=alpha_deg,
        stem_angle_deg=alpha_deg,
        friction_coefficient=mu,
        name=f"case-hull-alpha{alpha_deg:g}",
    )
    comp = lindqvist_components(ice, hull, speed_m_s,
                                water_density_kg_m3=water_density_kg_m3)
    # units: the model returns a total force whose implicit contact width is the
    # ice thickness h (see level_ice.formula); we report the raw total and the
    # per-unit-beam value so the caliber assumption is visible in the artifact.
    return {
        "alpha_deg": alpha_deg,
        "mu": mu,
        "components": comp,
        "total_N": float(comp["total_N"]),
        "static_N": float(comp["static_N"]),
        "crushing_N": float(comp["crushing_N"]),
        "bending_N": float(comp["bending_N"]),
        "submersion_N": float(comp["submersion_N"]),
        "added_drag_N": float(comp["added_drag_N"]),
        "total_over_beam_N_per_m": float(comp["total_N"]) / case.ship_beam_m,
    }


def domain_verdict(dem_load_N: float, anchor: dict[str, Any], *,
                   tolerance_pct: float = 15.0) -> dict[str, Any]:
    """Compare a DEM single-ship load with the converted anchor and grade it.

    Besides the plain relative deviation this returns the **out-of-domain
    test**: Lindqvist's bending and submersion terms are a *floor* that any
    level-ice-capable model must exceed.  If the DEM load falls below that
    floor, the case is outside the formula's applicability domain and the
    ``<15 %`` gate cannot be met by any numerical refinement -- the gap is
    model-form (capability), not discretisation.
    """
    anchor_total = float(anchor["total_N"])
    deviation = (dem_load_N - anchor_total) / anchor_total
    floor = float(anchor["bending_N"]) + float(anchor["submersion_N"])
    return {
        "dem_load_N": float(dem_load_N),
        "anchor_total_N": anchor_total,
        "relative_deviation": deviation,
        "relative_deviation_pct": 100.0 * deviation,
        "abs_deviation_pct": abs(100.0 * deviation),
        "tolerance_pct": tolerance_pct,
        "within_tolerance": abs(100.0 * deviation) < tolerance_pct,
        "anchor_noncrushing_floor_N": floor,
        "dem_below_anchor_floor": bool(dem_load_N < floor),
        "dem_over_anchor_floor": float(dem_load_N) / floor if floor else float("nan"),
        "anchor_over_dem_ratio": anchor_total / float(dem_load_N) if dem_load_N else float("nan"),
    }


def channel_reclosure_time(case: FormationCase, *, n_hulls: int = 1) -> dict[str, float]:
    """Time for the cross-channel drift to re-fill the corridor behind a hull.

    The cleared corridor half-width is ``B/2`` (the hull's own waterline
    footprint); a drift of ``u`` closes it in ``B/(2u)``.  Comparing that with
    the follower lag ``S/V`` explains the spacing dependence of the saving.
    """
    half_width = 0.5 * case.ship_beam_m
    closure = half_width / case.drift_m_s if case.drift_m_s > 0 else float("inf")
    lag = case.spacing_m / case.speed_m_s if case.speed_m_s > 0 else float("inf")
    return {
        "corridor_half_width_m": half_width,
        "drift_m_s": case.drift_m_s,
        "reclosure_time_s": closure,
        "follower_lag_s": lag,
        "lag_over_reclosure": lag / closure if closure else float("inf"),
    }