"""Run the DEM scenarios of the rubble-pile rheology calibration (P82W4-N02).

Usage
-----
    python scripts/run_rubble_rheology.py list
    python scripts/run_rubble_rheology.py run --index 3
    python scripts/run_rubble_rheology.py sweep --shard 0/8      # cluster shard
    python scripts/run_rubble_rheology.py report                 # fit + figures

One *run* = one scenario at one imposed rate and one seed.  For every snapshot
of the run the post-processing module extracts the porosity field (with its
conservation audit), the Voronoi/Delaunay pore descriptors, the force-chain
statistics and the bulk ``(I, mu)`` point; the run file stores the per-snapshot
points plus a compact summary, so the aggregation step never re-integrates DEM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.rubble_postprocess import (   # noqa: E402
    contact_view,
    force_chain_statistics,
    grid_porosity_field,
    porosity_conservation_audit,
    rheology_point,
    voronoi_local_porosity,
)
from tensordem.rubble_scenarios import (     # noqa: E402
    bulk_mask,
    run_biaxial_compression,
    run_couette_shear,
    run_ice_channel,
)

DENSITY = 917.0
DEFAULT_OUT = REPO / "artifacts" / "rubble_rheology"

#: Declared sweep (see prereg/rubble_rheology_prereg.json).  The rate ladders are
#: fixed before the production sweep; nothing here may be extended after seeing
#: the acceptance result.
SWEEP: dict[str, dict[str, Any]] = {
    "couette_shear": {
        "runner": "shear",
        "rates": [4.0, 8.0, 15.0, 25.0, 45.0],
        "seeds": [0, 1, 2, 3, 4, 5],
        "kwargs": dict(n_particles=400, mean_radius=0.014, width=0.64,
                       height=0.36, phi=0.82, strain=3.0, snapshots=12),
    },
    "biaxial_compression": {
        "runner": "compression",
        "rates": [0.5, 1.5, 4.0],
        "seeds": [0, 1, 2, 3, 4, 5],
        "kwargs": dict(n_particles=380, mean_radius=0.014, width=0.40,
                       height=0.45, phi=0.80, total_strain=0.30,
                       snapshots=12),
    },
    "ice_channel": {
        "runner": "channel",
        "rates": [0.15, 0.40, 0.90],
        "seeds": [0, 1, 2, 3],
        "kwargs": dict(n_particles=350, mean_radius=0.014, phi=0.80,
                       duration=0.60, snapshots=10),
    },
}


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    rate: float
    seed: int

    @property
    def name(self) -> str:
        return f"{self.scenario}__rate{self.rate:g}__seed{self.seed}"


def run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for scenario, cfg in SWEEP.items():
        for rate in cfg["rates"]:
            for seed in cfg["seeds"]:
                specs.append(RunSpec(scenario, float(rate), int(seed)))
    return specs


def _run_scenario(spec: RunSpec) -> dict[str, Any]:
    cfg = SWEEP[spec.scenario]
    kw = dict(cfg["kwargs"])
    runner = cfg["runner"]
    t0 = time.time()
    if runner == "shear":
        out = run_couette_shear(shear_rate=spec.rate, seed=spec.seed, **kw)
    elif runner == "compression":
        out = run_biaxial_compression(strain_rate=spec.rate, seed=spec.seed, **kw)
    elif runner == "channel":
        out = run_ice_channel(drift_speed=spec.rate, seed=spec.seed, **kw)
    else:                                            # pragma: no cover
        raise SystemExit(f"unknown runner {runner!r}")
    out["wall_time_s"] = time.time() - t0
    return out


def analyse_run(spec: RunSpec, run: dict[str, Any]) -> dict[str, Any]:
    """Post-process every snapshot of one run into ``(I, mu)`` points."""
    points = []
    poro_err, poro_mean, poro_std = [], [], []
    vor_mean, vor_std, delaunay = [], [], []
    z_all, z_strong, a_c, a_c_all, theta_c = [], [], [], [], []
    share, largest, percolate_x, percolate_y, partitions_ok = [], [], [], [], []
    field_snapshot = None
    for snap in run["snapshots"]:
        sample = snap["sample"]
        view = contact_view(snap["contacts"])
        if view.count == 0:
            continue
        margin = run.get("bulk_margin", 3.0 * float(np.mean(sample.radii)))
        mask = bulk_mask(sample, margin=margin)
        if int(np.sum(mask)) < 20:
            continue
        rate = abs(float(snap.get("profile", {}).get("slope",
                                                     snap.get("nominal_shear_rate", 0.0))))
        if not rate > 0.0:
            continue
        point = rheology_point(sample, view, shear_rate=rate, density=DENSITY,
                               particle_mask=mask)
        point["profile_r2"] = float(snap.get("profile", {}).get("profile_r2", float("nan")))
        point["phi_solid"] = float(sample.area_fraction())
        point["n_bulk_particles"] = int(np.sum(mask))
        point["time"] = float(snap["time"])
        points.append(point)

        field = grid_porosity_field(sample)
        audit = porosity_conservation_audit(field, sample)
        poro_err.append(audit["absolute_error"])
        poro_mean.append(field["phi_mean"])
        poro_std.append(field["phi_std"])
        vor = voronoi_local_porosity(sample)
        vor_mean.append(vor["phi_mean"])
        vor_std.append(vor["phi_std"])
        delaunay.append(vor["delaunay_mean_length_over_d50"])
        stats = force_chain_statistics(sample, view)
        z_all.append(stats["coordination_z"])
        z_strong.append(stats["coordination_z_strong"])
        a_c.append(stats["anisotropy_a_c"])
        a_c_all.append(stats["anisotropy_a_c_all"])
        theta_c.append(stats["theta_c_deg"])
        share.append(stats["force_share_strong"])
        largest.append(stats["strong_largest_fraction"])
        percolate_x.append(stats["percolates_x"])
        percolate_y.append(stats["percolates_y"])
        partitions_ok.append(stats["partition"]["passed"])
        if spec.seed == 0:
            field_snapshot = {
                "phi": field["phi"], "h": np.array(field["cell_size"]),
                "positions": sample.positions, "radii": sample.radii,
                "contacts": view, "grid_shape": np.array(field["grid_shape"]),
                "phi_local": vor["phi_local"],
                "angle_centres": np.array(stats["angle_bins_deg"]),
                "angle_counts": np.array(stats["angle_counts"]),
                "angle_force_fraction": np.array(stats["angle_force_fraction"]),
                "sample_domain": np.array(sample.domain),
            }
    if not points:
        raise RuntimeError(f"{spec.name}: no valid rheology point (all snapshots empty)")

    I = np.array([p["I"] for p in points])
    mu = np.array([p["mu"] for p in points])
    p_arr = np.array([p["pressure"] for p in points])
    n = I.size
    run_mean = {
        "I": float(np.mean(I)), "mu": float(np.mean(mu)), "pressure": float(np.mean(p_arr)),
        "I_se": float(np.std(I, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "mu_se": float(np.std(mu, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "shear_rate": float(np.mean([p["shear_rate"] for p in points])),
        "n_snapshots": int(n),
    }
    record = {
        "scenario": spec.scenario,
        "rate_nominal": spec.rate,
        "seed": spec.seed,
        "run_name": spec.name,
        "rate_measured": run.get("measured_shear_rate"),
        "rate_measured_std": run.get("shear_rate_std"),
        "velocity_profile_r2": run.get("velocity_profile_r2"),
        "wall_slip_fraction": run.get("wall_slip_fraction"),
        "dt": run.get("dt"), "steps": run.get("steps"),
        "wall_time_s": run.get("wall_time_s"),
        "kwargs": SWEEP[spec.scenario]["kwargs"],
        "points": points,
        "run_mean": run_mean,
        "porosity": {
            "conservation_error_max": float(np.max(poro_err)),
            "phi_mean": float(np.mean(poro_mean)),
            "phi_std_mean": float(np.mean(poro_std)),
            "h_over_d50": float(field["h_over_d50"]) if field_snapshot is not None else None,
        },
        "voronoi": {
            "phi_mean": float(np.mean(vor_mean)),
            "phi_std_mean": float(np.mean(vor_std)),
            "delaunay_mean_length_over_d50": float(np.mean(delaunay)),
        },
        "force_chain": {
            "z_mean": float(np.mean(z_all)),
            "z_strong_mean": float(np.mean(z_strong)),
            "a_c_mean": float(np.mean(a_c)),
            "a_c_all_mean": float(np.mean(a_c_all)),
            "theta_c_mean_deg": float(np.mean(theta_c)),
            "force_share_strong": float(np.mean(share)),
            "largest_component_fraction": float(np.mean(largest)),
            "percolates_x_fraction": float(np.mean(percolate_x)),
            "percolates_y_fraction": float(np.mean(percolate_y)),
            "partition_all_passed": bool(np.all(partitions_ok)),
        },
    }
    return record, field_snapshot


def write_run(spec: RunSpec, outdir: Path) -> dict[str, Any]:
    run = _run_scenario(spec)
    record, field = analyse_run(spec, run)
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{spec.name}.json").write_text(json.dumps(record, indent=1))
    if field is not None:
        snap_dir = outdir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        cv = field.pop("contacts")
        np.savez_compressed(snap_dir / f"{spec.name}.npz", **field,
                            contact_i=cv.i, contact_j=cv.j,
                            contact_fn=cv.normal_force, contact_f=cv.force_on_i)
    print(json.dumps({"run": spec.name, "I": record["run_mean"]["I"],
                      "mu": record["run_mean"]["mu"],
                      "n": record["run_mean"]["n_snapshots"],
                      "wall_time_s": round(record["wall_time_s"], 1)}), flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("list")
    p_run = sub.add_parser("run")
    p_run.add_argument("--index", type=int, required=True)
    p_run.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--shard", default="0/1",
                         help="SHARD/TOTAL; only runs with index %% TOTAL == SHARD")
    p_sweep.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    specs = run_specs()
    if args.mode == "list":
        for k, s in enumerate(specs):
            print(k, s.name)
        return 0
    if args.mode == "run":
        spec = specs[args.index]
        write_run(spec, args.outdir)
        return 0
    shard, total = (int(x) for x in str(args.shard).split("/"))
    if not 0 <= shard < total:
        raise SystemExit("shard must satisfy 0 <= shard < total")
    done = 0
    for k, spec in enumerate(specs):
        if k % total != shard:
            continue
        write_run(spec, args.outdir)
        done += 1
    print(json.dumps({"shard": shard, "total": total, "runs": done,
                      "python": platform.python_version(),
                      "numpy": np.__version__}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())