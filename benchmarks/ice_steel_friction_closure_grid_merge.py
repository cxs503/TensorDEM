"""Merge the cluster grid shards into the artifact and record the audit trail.

The verification grid is executed by ``run_cluster`` in shard mode
(``benchmarks/ice_steel_friction_closure.py --grid-shard i/n``); this script

1. parses the shard aggregates from the fetched shard logs (or from the shard
   JSON files the driver writes),
2. checks that the shards tile the grid exactly once,
3. compares the cross-node total against the single-process local run stored in
   the artifact (relative difference, not hashes: the nodes are loong64 while the
   control plane is x86, so the last ulp of every float can differ),
4. copies the shard logs into ``runs/ice_steel_friction_closure/cluster/`` and
   writes the merged block back into
   ``artifacts/ice_steel_friction_closure.json``.

Usage::

    PYTHONPATH=src python3 benchmarks/ice_steel_friction_closure_grid_merge.py \\
        /path/to/shard_1.log ... /path/to/shard_8.log
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "ice_steel_friction_closure.json"
RUN_DIR = ROOT / "runs" / "ice_steel_friction_closure"
CLUSTER_DIR = RUN_DIR / "cluster"


def _parse(path: Path) -> dict | None:
    for line in reversed(path.read_text().splitlines()):
        line = line.strip()
        if line.startswith("{") and "n_points" in line:
            return json.loads(line)
    return None


def main(paths: list[str]) -> dict:
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    record = json.loads(ARTIFACT.read_text())
    shards: list[dict] = []
    for raw in paths:
        source = Path(raw)
        aggregate = _parse(source)
        if aggregate is None:
            print(f"skip (no aggregate): {source}")
            continue
        shards.append(aggregate)
        shutil.copy2(source, CLUSTER_DIR / source.name)
    shards.sort(key=lambda item: item["shard"])

    total = {name: sum(int(item["regime_counts"].get(name, 0)) for item in shards)
             for name in ("boundary", "mixed", "melt")}
    n_points = sum(int(item["n_points"]) for item in shards)
    mu_sum = sum(float(item["mu_sum"]) for item in shards)
    mu_min = min(float(item["mu_min"]) for item in shards)
    mu_max = max(float(item["mu_max"]) for item in shards)
    local = record["grid_verification"]["local_full_grid"]
    relative = abs(mu_sum - float(local["mu_sum"])) / max(abs(float(local["mu_sum"])), 1e-30)

    tiling = sorted(item["shard"] for item in shards)
    expected = list(range(len(shards)))
    checks = {
        "n_shards": len(shards),
        "shard_indices": tiling,
        "shards_tile_the_grid": tiling == expected,
        "n_points_total": n_points,
        "n_grid_points": int(shards[0]["n_grid_points"]) if shards else 0,
        "points_match_grid": n_points == int(shards[0]["n_grid_points"]) if shards else False,
        "regime_counts_total": total,
        "regime_counts_match_local": total == local["regime_counts"],
        "mu_sum_cluster": mu_sum,
        "mu_sum_local": float(local["mu_sum"]),
        "mu_sum_relative_difference": relative,
        "mu_range_cluster": [mu_min, mu_max],
        "mu_range_local": [float(local["mu_min"]), float(local["mu_max"])],
        "architectures": "nodes loong64 / control plane x86_64: compare floats, "
                         "not byte hashes",
        "passes": bool(
            tiling == expected
            and n_points == int(shards[0]["n_grid_points"])
            and total == local["regime_counts"]
            and relative < 1e-9),
    }
    record["grid_verification"]["cluster_shards"] = shards
    record["grid_verification"]["cluster_consistency"] = checks
    record["acceptance"]["a3_cluster_grid_consistency"] = {
        "pass": checks["passes"],
        "n_shards": checks["n_shards"],
        "n_points": checks["n_points_total"],
        "mu_sum_relative_difference": relative,
        "regime_counts_total": total,
        "log_dir": str(CLUSTER_DIR.relative_to(ROOT)),
    }
    ARTIFACT.write_text(json.dumps(record, indent=2, allow_nan=False))
    (RUN_DIR / "grid_shards.json").write_text(json.dumps(
        {"checks": checks, "shards": shards}, indent=2))
    print(json.dumps(checks, indent=2))
    return checks


if __name__ == "__main__":
    main(sys.argv[1:])