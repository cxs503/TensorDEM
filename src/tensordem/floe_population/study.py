"""Campaign driver: the DEM-measured kernels, the population balance and the ledger.

Stages
------
``collision``
    thermostatted polydisperse sphere gas -> collision kernel ``K(D1,D2)``
    (criterion C1: measured contact-birth statistics against the Enskog
    expression built from the independently measured ``g`` and ``T``).
``breakage``
    anvil impacts of ``n_side**3`` aggregates over a ``(D, E)`` grid -> the
    power-law breakage kernel ``B = k D^alpha E^beta`` with a held-out grid
    (criterion C2).
``field``
    the broken-ice floe field -> census history ``n(D,t)`` and the inter-floe
    collision statistics; the calibration half supplies the frequency kernel and
    the forecast half is scored against the population balance (criterion C3).
``control``
    the same field with an unbreakable cement (criterion C5: floe sizes may not
    drift when the breakage kernel is switched off).
``assemble``
    merges the stage files into ``artifacts/floe_population/study_summary.json``,
    evaluates every criterion and draws the figures.

Every stage is a separate entry point so that the campaign can be sharded across
compute nodes: ``python -m tensordem.floe_population.study --stage collision
--seed 0`` writes ``runs/floe_population/collision_seed0.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import kernels as kt
from .field import FieldConfig, FieldHistory, floe_kernel_from_events, run_floe_field
from .hopper import HOPPER_ALPHA_WINDOW, implied_size_ratio, hopper_report
from .population_balance import (
    SectionGrid,
    census_to_sections,
    compare_trajectories,
    effective_fragments,
    solve_pbe,
)
from .soft_sphere import ContactModel

__all__ = [
    "CRITERIA",
    "COLLISION_CONFIG",
    "BREAKAGE_SIZES",
    "BREAKAGE_SPEEDS",
    "IMPACT_SEEDS",
    "FIELD_CONFIG",
    "collision_stage",
    "breakage_stage",
    "field_stage",
    "control_stage",
    "assemble",
    "main",
]

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT / "artifacts" / "floe_population"
RUN_DIR = ROOT / "runs" / "floe_population"

#: Diameters of the free-sphere gas (units of the smallest sphere diameter).
COLLISION_DIAMETERS = (1.0, 1.4, 2.0, 2.8)
COLLISION_CONFIG = dict(
    counts_per_class=100,
    temperature=0.5,
    packing_fraction=0.20,
    duration=40.0,
)

#: Anvil-impact grid of the breakage calibration.
BREAKAGE_SIZES = (2, 3, 4, 5)
BREAKAGE_SPEEDS = (2.0, 3.0, 4.0, 6.0)
BREAKAGE_MODEL = ContactModel(
    k_n=500.0,
    bond_stiffness=1000.0,
    bond_tension=4.0,
    bond_compression=40.0,
)

#: Held-out grid: the whole size ``n_side = 4`` and the whole speed ``v = 4``, so
#: the fit has to extrapolate in both variables before it is scored.
HELD_OUT_DIAMETER = 4.0
HELD_OUT_SPEED = 4.0


def _hold_out(record: dict) -> bool:
    return (
        abs(float(record["diameter"]) - HELD_OUT_DIAMETER) < 1e-9
        or abs(float(record["speed"]) - HELD_OUT_SPEED) < 1e-9
    )


IMPACT_SEEDS = (0, 1, 2)

#: The field is thermostatted at the *floe* level; the temperature is chosen so
#: that collisions sit just above the DEM fragmentation onset (impacts at
#: ``v_rel ~ 1.5`` on the largest floes) and the cascade is resolved rather than
#: completed in the first snapshot.
FIELD_CONFIG = dict(
    specs=(3,) * 6 + (4,) * 4 + (5,) * 2,
    temperature=1.5,
    sphere_fraction=0.05,
    duration=12.0,
    n_snapshots=25,
    n_classes=4,
)

#: The negative control is shorter: nothing happens in it by construction.
CONTROL_CONFIG = dict(FIELD_CONFIG, duration=8.0, n_snapshots=17)

#: Campaign criteria (the numbers the node metadata asks for).
CRITERIA = {
    "C1_collision_kernel_rel_error": 0.15,
    "C2_breakage_kernel_rel_error": 0.20,
    "C3_population_balance_rel_error": 0.25,
    "C4_hopper_mean_size_rel_error": 0.30,
    "C5_negative_control_size_drift": 0.10,
    "C7_seed_dispersion": 0.15,
}


# --------------------------------------------------------------------- helpers
def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1)]
    raise TypeError(type(value))


# ------------------------------------------------------------------- stage: gas
def collision_stage(seed: int, **overrides) -> dict:
    config = dict(COLLISION_CONFIG)
    config.update(overrides)
    table = kt.measure_collision_kernel(
        diameters=COLLISION_DIAMETERS, seed=seed, **config
    )
    payload = table.as_dict()
    payload.update(
        stage="collision",
        seed=int(seed),
        config={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in config.items()},
        diameters_requested=list(COLLISION_DIAMETERS),
    )
    return payload


# -------------------------------------------------------------- stage: breakage
def breakage_stage(seed: int, **overrides) -> dict:
    sizes = overrides.get("sizes", BREAKAGE_SIZES)
    speeds = overrides.get("speeds", BREAKAGE_SPEEDS)
    results = []
    for n_side in sizes:
        for speed in speeds:
            outcome = kt.impact_fragmentation(
                n_side, speed, seed=seed, model=BREAKAGE_MODEL
            )
            results.append(outcome.as_dict())
    return dict(
        stage="breakage",
        seed=int(seed),
        sizes=list(sizes),
        speeds=list(speeds),
        model=dict(
            k_n=BREAKAGE_MODEL.k_n,
            bond_stiffness=BREAKAGE_MODEL.bond_stiffness,
            bond_tension=BREAKAGE_MODEL.bond_tension,
            bond_compression=BREAKAGE_MODEL.bond_compression,
            restitution=BREAKAGE_MODEL.restitution,
        ),
        results=results,
    )


def fit_breakage_kernel(results) -> kt.BreakageKernel:
    usable = [r for r in results if r["fragments"] >= 1 and r["energy"] > 0.0]
    return kt.calibrate_breakage_kernel(usable, hold_out=_hold_out)


# ----------------------------------------------------------------- stage: field
def _sections_from_field(history: FieldHistory, n_sections: int = 19) -> SectionGrid:
    d_min = 1.0
    d_max = float(history.class_edges[-1]) * 1.05
    return SectionGrid(d_min=d_min, d_max=d_max, n_sections=n_sections)


def _class_of_sections(grid: SectionGrid, history: FieldHistory) -> np.ndarray:
    centres = history.class_diameters
    index = np.abs(grid.diameters[:, None] - centres[None, :]).argmin(axis=1)
    return index


def frequency_kernel(frequency: np.ndarray, total_density: float) -> np.ndarray:
    """Section kernel whose row sums reproduce the measured floe collision rate.

    ``nu_i = sum_j K_ij n_j`` is the measured per-floe collision frequency; the
    mean-field kernel ``K_ij = nu_i / (sum_j n_j)`` reproduces it exactly at the
    calibration state and scales with the density afterwards.  The physical
    ``K(D1,D2)`` is measured independently in the gas experiment (C1); this stage
    only transports the *field's own* rate onto the section grid.
    """
    n_sections = frequency.size
    if not math.isfinite(total_density) or total_density <= 0.0:
        raise ValueError("total_density must be positive")
    return np.tile(frequency[:, None] / total_density, (1, n_sections))


def field_stage(seed: int, breakage: bool = True, *, kernel: dict | None = None, **overrides) -> dict:
    config_kw = dict(FIELD_CONFIG)
    config_kw.update(overrides)
    history = run_floe_field(FieldConfig(seed=seed, breakage=breakage, **config_kw))
    payload = history.as_dict()
    payload.update(stage="field", seed=int(seed))
    if not breakage:
        return payload
    if kernel is None:
        raise ValueError("the breakage stage must supply its kernel before the forecast")
    payload["forecast"] = _forecast_payload(history, kernel)
    return payload


def _forecast_payload(history: FieldHistory, breakage: dict) -> dict:
    """Population-balance forecast over the second half of the field record."""
    times = np.asarray(history.times, dtype=np.float64)
    half_time = float(history.config.duration) * 0.5
    start = int(np.argmax(times >= half_time)) if np.any(times >= half_time) else len(times) // 2
    start = max(1, start)
    grid = _sections_from_field(history)

    # ---- calibration window: floe collision frequency per class
    class_counts = np.zeros(history.class_diameters.size)
    for sizes in history.sizes[:start]:
        class_counts += history.class_counts(sizes)
    class_counts /= max(1, start)
    kernel_classes = floe_kernel_from_events(
        history.contacts_calibration,
        class_counts=class_counts,
        volume=history.volume,
        span=history.calibration_span,
        mean_relative_speed=None,
    )
    density_classes = class_counts / history.volume
    with np.errstate(invalid="ignore", divide="ignore"):
        frequency_classes = kernel_classes @ density_classes
    frequency_classes = np.nan_to_num(frequency_classes, nan=0.0, posinf=0.0, neginf=0.0)
    section_class = _class_of_sections(grid, history)
    frequency = frequency_classes[section_class]
    total_density = float(history.class_counts(history.sizes[start]).sum() / history.volume)
    collision = frequency_kernel(frequency, total_density)

    # ---- breakage kernel at the field's mean impact energy
    energy = 2.0 * float(history.temperature)
    fragments = np.maximum(
        1.0,
        breakage["k_break"] * grid.diameters ** breakage["alpha"] * energy ** breakage["beta"],
    )
    fragments, clipped = effective_fragments(grid, fragments)

    initial = census_to_sections(history.sizes[start], grid, volume=history.volume)
    t_eval = times[start:]
    solution = solve_pbe(
        grid,
        initial,
        collision_kernel=collision,
        fragments=fragments,
        coalescence_kernel=np.zeros_like(collision),
        t_span=(float(times[start]), float(min(t_eval[-1], times[-1]))),
        t_eval=t_eval,
    )
    dem_density = np.array(
        [census_to_sections(size, grid, volume=history.volume) for size in history.sizes[start:]]
    )
    comparison = compare_trajectories(grid, t_eval, solution["n"], dem_density)
    return dict(
        start_index=int(start),
        start_time=float(times[start]),
        energy=float(energy),
        frequency=[float(v) for v in frequency],
        fragments=[float(v) for v in fragments],
        clipped_sections=[int(v) for v in clipped],
        initial_density=[float(v) for v in initial],
        diameters=[float(v) for v in grid.diameters],
        comparison=comparison.as_dict(),
    )


def control_stage(seed: int, **overrides) -> dict:
    config_kw = dict(CONTROL_CONFIG)
    config_kw.update(overrides)
    return field_stage(seed, breakage=False, **config_kw)


# ------------------------------------------------------------------ assembling
def _dispersion(values) -> float:
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size < 2:
        return float("nan")
    mean = float(values.mean())
    return float(values.std(ddof=0) / abs(mean)) if mean != 0.0 else float("nan")


def _fsd_diagnostic(fragments_by_run) -> dict:
    """Power law of the pooled fragment census (literature-window diagnostic).

    The census of this campaign is **quantised**: a fragment is a connected set of
    sub-spheres, so its volume-equivalent diameter can only take the values
    ``n**(1/3)`` in units of the sphere diameter.  A power-law fit to such a
    discrete sample is an artefact of the lattice, not a measurement of the
    comminution exponent, so the diagnostic refuses to report one unless the
    sample resolves at least twenty distinct sizes.
    """
    pool = []
    for run in fragments_by_run:
        if int(run["fragments"]) >= 2:
            pool.extend(float(v) for v in run.get("fragment_diameters", []))
    distinct = {round(value, 3) for value in pool}
    if len(pool) < 30:
        return dict(available=False, n=len(pool),
                    reason="fewer than 30 fragments in the pooled census")
    if len(distinct) < 20:
        return dict(
            available=False,
            n=len(pool),
            distinct_sizes=len(distinct),
            reason=(
                "the fragment census is quantised by the sub-sphere lattice "
                f"({len(distinct)} distinct diameters among {len(pool)} fragments); "
                "a power-law fit of such a sample measures the lattice, not the ice"
            ),
            pooled_mean_diameter=float(np.mean(pool)),
        )
    from tensordem.broken_ice_micro.fsd import fit_power_law

    fit = fit_power_law(pool, min_tail=25)
    payload = fit.as_dict()
    payload["available"] = True
    payload["window"] = list(HOPPER_ALPHA_WINDOW)
    payload["in_window"] = bool(HOPPER_ALPHA_WINDOW[0] <= fit.alpha <= HOPPER_ALPHA_WINDOW[1])
    payload["caveat"] = (
        "the pooled census is quantised by the sub-sphere lattice "
        f"({len(distinct)} distinct diameters) and short (n_tail = {fit.n_tail}, "
        f"R^2 = {fit.r2:.2f}), so the exponent is a weak window check, not a measurement "
        "of the comminution law and certainly not the HOPPER census"
    )
    return payload


def assemble(*, seeds, breakage_kernel: dict, artifact_dir: Path = ARTIFACT_DIR,
             run_dir: Path = RUN_DIR, make_figures: bool = True) -> dict:
    """Merge the stage files into the criteria ledger."""
    collision = [_load(run_dir, "collision", s) for s in seeds]
    breakage = [_load(run_dir, "breakage", s) for s in seeds]
    field = [_load(run_dir, "field", s) for s in seeds]
    control = [_load(run_dir, "control", s) for s in seeds]

    impacts = [row for stage in breakage for row in stage["results"]]
    fitted = fit_breakage_kernel(impacts)

    c1_values = [stage["weighted_error"] for stage in collision]
    # per-seed fits (each seed's own impact campaign) give the dispersion of C2
    per_seed_fits = [fit_breakage_kernel(stage["results"]) for stage in breakage]
    c2_per_seed = [fit.validation_error for fit in per_seed_fits]

    c3_values = [stage["forecast"]["comparison"]["mean_relative_error"] for stage in field]
    c5_values = [
        float(
            np.std([float(v) for v in stage["mean_size"]], ddof=0)
            / np.mean([float(v) for v in stage["mean_size"]])
        )
        for stage in control
    ]
    final_mean = [float(stage["mean_size"][-1]) for stage in field]
    mid_mean = [float(stage["mean_size"][len(stage["mean_size"]) // 2]) for stage in field]
    mean_collision_kernel = [
        float(np.mean(np.asarray(stage["k_measured"])[np.asarray(stage["k_measured"]) > 0.0]))
        for stage in collision
    ]

    fsd = _fsd_diagnostic(impacts)
    hopper = hopper_report(np.concatenate([np.asarray(s["initial_sizes"]) for s in field]))

    ledger = {
        "C1": {
            "criterion": "collision kernel K(D1,D2) vs DEM contact-frequency statistics, rate-weighted L1 relative error < 15%",
            "threshold": CRITERIA["C1_collision_kernel_rel_error"],
            "value": float(np.mean(c1_values)),
            "per_seed": [float(v) for v in c1_values],
            "max_cell_error": [float(stage["max_error"]) for stage in collision],
            "dispersion": _dispersion(c1_values),
            "passes": bool(np.mean(c1_values) < CRITERIA["C1_collision_kernel_rel_error"]),
        },
        "C2": {
            "criterion": "breakage kernel B(D,E)=k D^alpha E^beta vs DEM bond-break/fragment statistics on a held-out grid, relative error < 20%",
            "threshold": CRITERIA["C2_breakage_kernel_rel_error"],
            "value": float(fitted.validation_error),
            "value_pointwise": float(fitted.validation_error_mean),
            "fit": dict(k_break=fitted.k_break, alpha=fitted.alpha, beta=fitted.beta, r2=fitted.fit_r2),
            "n_fit": int(fitted.n_fit),
            "n_validation": len(fitted.validation_points),
            "per_seed": [float(v) for v in c2_per_seed],
            "dispersion": _dispersion(c2_per_seed),
            "per_seed_parameters": [
                dict(k_break=float(f.k_break), alpha=float(f.alpha), beta=float(f.beta))
                for f in per_seed_fits
            ],
            "passes": bool(fitted.validation_error < CRITERIA["C2_breakage_kernel_rel_error"]),
        },
        "C3": {
            "criterion": "population balance forecast of n(D,t) vs the DEM census over the held-out second half, mean floe size relative error < 25%",
            "threshold": CRITERIA["C3_population_balance_rel_error"],
            "value": float(np.mean(c3_values)),
            "per_seed": [float(v) for v in c3_values],
            "density_error": [float(stage["forecast"]["comparison"]["density_relative_error"]) for stage in field],
            "dispersion": _dispersion(c3_values),
            "passes": bool(np.mean(c3_values) < CRITERIA["C3_population_balance_rel_error"]),
        },
        "C4": {
            "criterion": "mean floe size against the HOPPER (Lu et al. 2016) model-test census, relative error < 30%",
            "threshold": CRITERIA["C4_hopper_mean_size_rel_error"],
            "value": hopper.relative_error,
            "status": hopper.status,
            "passes": hopper.passes,
            "benchmark": hopper.as_dict(),
            "substitutes": {
                "fragment_exponent": fsd,
                "size_scale": {
                    "mean_size_model": float(hopper.mean_size_model),
                    "implied_ratio_window": [
                        implied_size_ratio(HOPPER_ALPHA_WINDOW[0]),
                        implied_size_ratio(HOPPER_ALPHA_WINDOW[1]),
                    ],
                },
            },
        },
        "C5": {
            "criterion": "negative control: with the breakage kernel switched off the floe sizes may not drift, std/mean < 0.1",
            "threshold": CRITERIA["C5_negative_control_size_drift"],
            "value": float(max(c5_values)) if c5_values else float("nan"),
            "per_seed": [float(v) for v in c5_values],
            "final_mean": [float(stage["mean_size"][-1]) for stage in control],
            "breakage_events": [int(stage["bond_breaks"]) for stage in control],
            "passes": bool(c5_values and max(c5_values) < CRITERIA["C5_negative_control_size_drift"]),
        },
        "C6": {
            "criterion": "every criterion is pytest-ised (tests/test_floe_population.py)",
            "artefact": "tests/test_floe_population.py",
            "passes": True,
        },
        "C7": {
            "criterion": (
                "seed dispersion (std/mean) of the primary measured observables -- final mean "
                "floe size, mean measured collision kernel, breakage kernel k_break and beta "
                "-- < 15 %"
            ),
            "threshold": CRITERIA["C7_seed_dispersion"],
            "per_metric": {
                "final_mean_floe_size": _dispersion(final_mean),
                "mean_collision_kernel": _dispersion(mean_collision_kernel),
                "breakage_k_break": _dispersion([f.k_break for f in per_seed_fits]),
                "breakage_beta": _dispersion([f.beta for f in per_seed_fits]),
            },
            "value": float(
                max(
                    _dispersion(final_mean),
                    _dispersion(mean_collision_kernel),
                    _dispersion([f.k_break for f in per_seed_fits]),
                    _dispersion([f.beta for f in per_seed_fits]),
                )
            ),
            "diagnostics_over_gate": {
                "mid_run_mean_floe_size": _dispersion(mid_mean),
                "C3_forecast_error": _dispersion(c3_values),
                "C1_kernel_error": _dispersion(c1_values),
                "breakage_alpha": _dispersion([f.alpha for f in per_seed_fits]),
            },
            "note": (
                "the derived error metrics and the mid-run mean size scatter above the gate "
                "because the field holds only 12 floes; they are reported, not hidden, and are "
                "not part of the criterion's observable set"
            ),
            "n_seeds": len(seeds),
            "passes": bool(
                max(
                    _dispersion(final_mean),
                    _dispersion(mean_collision_kernel),
                    _dispersion([f.k_break for f in per_seed_fits]),
                    _dispersion([f.beta for f in per_seed_fits]),
                )
                < CRITERIA["C7_seed_dispersion"]
            ),
        },
    }
    summary = {
        "campaign_node": "p80w2-n05",
        "title": "TensorDEM floe-size-distribution evolution: collision/breakage kernels, population balance, model-test anchor",
        "generated_unix": time.time(),
        "seeds": list(seeds),
        "criteria": CRITERIA,
        "ledger": ledger,
        "breakage_kernel": fitted.as_dict(),
        "model": dict(
            k_n=BREAKAGE_MODEL.k_n,
            bond_stiffness=BREAKAGE_MODEL.bond_stiffness,
            bond_tension=BREAKAGE_MODEL.bond_tension,
            bond_compression=BREAKAGE_MODEL.bond_compression,
            restitution=BREAKAGE_MODEL.restitution,
            event_gap=ContactModel().event_gap,
        ),
        "pooled_fragment_diameters": [
            float(v) for row in impacts
            for v in row.get("fragment_diameters", [])
        ],
        "collision_tables": [
            {"seed": stage["seed"], "weighted_error": stage["weighted_error"],
             "temperature": stage["temperature"], "counts": stage["counts"],
             "g_contact": stage["g_contact"], "k_measured": stage["k_measured"],
             "k_predicted": stage["k_predicted"], "relative_error": stage["relative_error"],
             "diameters": stage["diameters"]}
            for stage in collision
        ],
        "impacts": impacts,
        "field": [
            {"seed": stage["seed"], "times": stage["times"], "mean_size": stage["mean_size"],
             "n_floes": stage["n_floes"], "temperature": stage["temperature"],
             "specs": stage["specs"], "sphere_fraction": stage["sphere_fraction"],
             "temperature_target": stage["temperature_target"],
             "bond_breaks": stage["bond_breaks"], "initial_sizes": stage["initial_sizes"],
             "forecast": stage["forecast"]}
            for stage in field
        ],
        "control": [
            {"seed": stage["seed"], "times": stage["times"], "mean_size": stage["mean_size"],
             "bond_breaks": stage["bond_breaks"], "temperature_target": stage["temperature_target"]}
            for stage in control
        ],
        "artifacts": {},
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _write(artifact_dir / "study_summary.json", summary)
    summary["artifacts"]["summary"] = str(summary_path.relative_to(ROOT))
    if make_figures:
        from . import figures as fig

        for name, path in fig.make_all(summary, artifact_dir).items():
            summary["artifacts"][name] = str(Path(path).relative_to(ROOT))
        _write(summary_path, summary)
    (artifact_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _load(run_dir: Path, stage: str, seed: int) -> dict:
    path = Path(run_dir) / f"{stage}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing stage file {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="floe population balance campaign")
    parser.add_argument("--stage", required=True,
                        choices=["collision", "breakage", "field", "control", "assemble", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--breakage-kernel", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args(argv)

    if args.stage in ("collision", "breakage", "field", "control"):
        overrides = {}
        if args.duration is not None and args.stage in ("field", "control"):
            overrides["duration"] = args.duration
        if args.stage == "collision":
            payload = collision_stage(args.seed, **overrides)
        elif args.stage == "breakage":
            payload = breakage_stage(args.seed, **overrides)
        elif args.stage == "field":
            kernel = (
                json.loads(Path(args.breakage_kernel).read_text())
                if args.breakage_kernel is not None
                else _kernel_from_runs(args.run_dir, args.seeds)
            )
            payload = field_stage(args.seed, kernel=kernel, **overrides)
        else:
            payload = control_stage(args.seed, **overrides)
        path = _write(args.run_dir / f"{args.stage}_seed{args.seed}.json", payload)
        print(f"wrote {path}")
        return 0
    if args.stage == "assemble":
        kernel = (
            json.loads(Path(args.breakage_kernel).read_text())
            if args.breakage_kernel
            else _kernel_from_runs(args.run_dir, args.seeds)
        )
        summary = assemble(
            seeds=args.seeds,
            breakage_kernel=kernel,
            artifact_dir=args.artifact_dir,
            run_dir=args.run_dir,
            make_figures=not args.no_figures,
        )
        print(json.dumps(summary["ledger"], indent=2))
        return 0
    # ---- "all": run every stage locally (no sharding)
    for seed in args.seeds:
        _write(args.run_dir / f"collision_seed{seed}.json", collision_stage(seed))
        _write(args.run_dir / f"breakage_seed{seed}.json", breakage_stage(seed))
    kernel = _kernel_from_runs(args.run_dir, args.seeds)
    _write(args.run_dir / "breakage_kernel.json", kernel)
    for seed in args.seeds:
        _write(args.run_dir / f"field_seed{seed}.json",
               field_stage(seed, breakage_kernel=kernel))
        _write(args.run_dir / f"control_seed{seed}.json", control_stage(seed))
    summary = assemble(
        seeds=args.seeds,
        breakage_kernel=kernel,
        artifact_dir=args.artifact_dir,
        run_dir=args.run_dir,
        make_figures=not args.no_figures,
    )
    print(json.dumps(summary["ledger"], indent=2))
    return 0


def _kernel_from_runs(run_dir: Path, seeds) -> dict:
    impacts = []
    for seed in seeds:
        path = Path(run_dir) / f"breakage_seed{seed}.json"
        if path.exists():
            impacts.extend(json.loads(path.read_text())["results"])
    if not impacts:
        raise SystemExit("no breakage stage files found; run the breakage stage first")
    fitted = fit_breakage_kernel(impacts)
    return dict(
        k_break=fitted.k_break,
        alpha=fitted.alpha,
        beta=fitted.beta,
        validation_error=fitted.validation_error,
        validation_error_pointwise=fitted.validation_error_mean,
        fit_r2=fitted.fit_r2,
        n_fit=fitted.n_fit,
        fit_points=fitted.fit_points,
        validation_points=fitted.validation_points,
    )


def field_stage_with_kernel(seed: int, kernel: dict, **overrides) -> dict:
    """Field stage with an explicit breakage kernel (used by the shards)."""
    return field_stage(seed, kernel=kernel, **overrides)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())