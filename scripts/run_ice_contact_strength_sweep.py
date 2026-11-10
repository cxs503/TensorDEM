#!/usr/bin/env python
"""Cluster sweep of the ice-steel interface strength law over ``T x p x v x t_c``.

The sweep is sharded by temperature so that each shard is an independent,
re-runnable unit (``--shard i --shards n``); each shard writes its own JSON, and
``--aggregate`` folds the shards into one artifact.  Every row is produced by
:func:`tensordem.ice_contact_strength.sweep_grid`, i.e. by the same code the DEM
kernel and the report tables use -- no number in the paper is typed by hand.

Usage
-----
    python scripts/run_ice_contact_strength_sweep.py --shard 0 --shards 7 \
        --out artifacts/ice_contact_strength_sweep
    python scripts/run_ice_contact_strength_sweep.py --aggregate \
        --out artifacts/ice_contact_strength_sweep
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.ice_contact_strength import (  # noqa: E402
    AdhesionModel,
    InterfaceStrength,
    MultiplicativeFrictionLaw,
    adhesion_band_check,
    direct_shear_case,
    inclined_plane_case_onset,
    interface_shear_strength,
    sweep_grid,
)

DEFAULT_TEMPERATURES = (-2.0, -5.0, -8.0, -10.0, -15.0, -20.0, -25.0, -30.0)
PRESSURE_MPA = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
SPEEDS = (1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0)
CONTACT_TIMES = (1.0, 10.0, 60.0, 600.0)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run_shard(index: int, shards: int, out: pathlib.Path) -> dict:
    if shards < 1:
        raise SystemExit("--shards must be >= 1")
    if not 0 <= index < shards:
        # tolerate 1-based shard indices (cluster templates pass {i}/{n})
        if 1 <= index <= shards:
            index -= 1
        else:
            raise SystemExit(f"shard index {index} outside [0, {shards})")
    temperatures = DEFAULT_TEMPERATURES[index::shards]
    strength = InterfaceStrength.calibrated()
    rows: list[dict] = []
    per_temperature: list[dict] = []
    for temperature in temperatures:
        grid = sweep_grid(temperatures_c=(temperature,), pressures_mpa=PRESSURE_MPA,
                          speeds_m_per_s=SPEEDS, contact_times_s=CONTACT_TIMES,
                          strength=strength)
        rows.extend(grid["rows"])
        per_temperature.append({
            "temperature_c": temperature,
            "count": grid["count"],
            "adhesion_share_mean": grid["adhesion_share_mean"],
            "adhesion_share_min": grid["adhesion_share_min"],
            "adhesion_share_max": grid["adhesion_share_max"],
            "n_cap_binding": grid["n_cap_binding"],
        })
    payload = {
        "shard": index, "shards": shards, "temperatures_c": list(temperatures),
        "rows": rows, "per_temperature": per_temperature,
        "axes": {"pressure_mpa": list(PRESSURE_MPA), "v_m_per_s": list(SPEEDS),
                 "contact_time_s": list(CONTACT_TIMES)},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{index:02d}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=1))
    return {"shard": index, "path": str(path), "rows": len(rows),
            "temperatures_c": list(temperatures)}


def _kernel_checks(out: pathlib.Path) -> dict:
    """Kernel-level checks, all through ``contact_force`` (not the curve)."""
    const = InterfaceStrength(friction=__import__(
        "tensordem.ice_contact_strength", fromlist=["ConstantFriction"]
    ).ConstantFriction(0.10))
    inclined = {}
    for temperature in (-5.0, -10.0, -20.0):
        inclined[f"{temperature:g}"] = inclined_plane_case_onset(
            mass_kg=86.0, area_m2=1.0e-4, temperature_c=temperature, strength=const,
            contact_time_s=60.0)
    shear = {}
    for normal in (1000.0, 5000.0, 20000.0):
        shear[f"{normal:g}"] = direct_shear_case(
            area_m2=0.04, normal_force_n=normal, temperature_c=-10.0, strength=const,
            contact_time_s=60.0)
    payload = {"inclined_plane": inclined, "direct_shear": shear}
    (out / "kernel_checks.json").write_text(json.dumps(_jsonable(payload), indent=1))
    return payload


def aggregate(out: pathlib.Path) -> dict:
    shard_files = sorted(out.glob("shard_*.json"))
    if not shard_files:
        raise SystemExit(f"no shards in {out}")
    rows: list[dict] = []
    per_temperature: list[dict] = []
    for path in shard_files:
        payload = json.loads(path.read_text())
        rows.extend(payload["rows"])
        per_temperature.extend(payload["per_temperature"])
    per_temperature.sort(key=lambda item: item["temperature_c"])
    strength = InterfaceStrength.calibrated()
    friction_law = MultiplicativeFrictionLaw.fit()

    # crossover pressure of every temperature: where friction term == adhesion term
    crossover = []
    for temperature in DEFAULT_TEMPERATURES:
        low, high = 1.0e-3, 50.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            value = interface_shear_strength(strength, pressure_mpa=mid,
                                             temperature_c=temperature,
                                             v_m_per_s=0.1, contact_time_s=60.0)
            if value["friction_term_pa"] > value["adhesion_term_pa"]:
                high = mid
            else:
                low = mid
        crossover.append({"temperature_c": temperature, "p_crossover_mpa": 0.5 * (low + high)})

    payload = {
        "n_rows": len(rows),
        "axes": {"temperature_c": list(DEFAULT_TEMPERATURES),
                 "pressure_mpa": list(PRESSURE_MPA), "v_m_per_s": list(SPEEDS),
                 "contact_time_s": list(CONTACT_TIMES)},
        "friction_law": friction_law.as_dict(),
        "adhesion": AdhesionModel().as_dict(),
        "adhesion_band": adhesion_band_check(),
        "per_temperature": per_temperature,
        "crossover_friction_equals_adhesion": crossover,
        "kernel_checks": _kernel_checks(out),
        "rows": rows,
        "provenance": "tensordem.ice_contact_strength.sweep_grid via "
                      "scripts/run_ice_contact_strength_sweep.py (cluster shards)",
    }
    path = out / "aggregate.json"
    path.write_text(json.dumps(_jsonable(payload), indent=1))
    return {"path": str(path), "n_rows": len(rows), "shards": len(shard_files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-range", type=str, default=None,
                        help="combined 'i/n' shard selector")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("artifacts/ice_contact_strength_sweep"))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_range:
        index_text, shards_text = args.shard_range.split("/")
        args.shard, args.shards = int(index_text), int(shards_text)
    if args.aggregate:
        print(json.dumps(aggregate(args.out)))
        return 0
    if args.shard is None:
        parser.error("either --shard i or --aggregate is required")
    print(json.dumps(run_shard(args.shard, args.shards, args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())