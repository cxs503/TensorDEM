"""Shard wrapper for the R6 peak-relative ensemble (one specimen per job).

Same code path as a multi-worker run: ``_run_cached`` of
``run_r6_peak_relative_ensemble.py`` with the predeclared specimen list, so the
numbers are bit-identical to a single ``--workers 8`` run.  After all shards the
local driver re-assembles with ``--resume``.

Usage: ``python benchmarks/biaxial_shear_band/run_r6_shard.py <i>``, ``i in 0..7``
(seed order 53, 67, 79, 83, 89, 97, 101, 103).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


def main() -> int:
    i = int(sys.argv[1])
    from benchmarks.biaxial_shear_band.run_r6_peak_relative_ensemble import (
        CACHE, POST_PEAK_STRAIN, STRAIN_CAP, _configs, _run_cached)

    config = _configs(False, None)[i]
    assert config["post_peak_strain"] == POST_PEAK_STRAIN
    assert config["strain_cap"] == STRAIN_CAP
    result = _run_cached(config, CACHE, True)
    stopping = result.get("stopping") or {}
    print(f"[shard {i}] {config['name']} stop={stopping.get('stop_reason')} "
          f"eps_log={stopping.get('final_axial_strain_log')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())