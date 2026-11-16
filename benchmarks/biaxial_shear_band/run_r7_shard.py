"""Shard wrapper for the R7 wide-window ensemble (one specimen per job).

Same code path as a multi-worker run: ``_run_cached`` of
``run_r7_wide_window_ensemble.py`` with the predeclared specimen list, so the
numbers are bit-identical to a single ``--workers 8`` run.  After all shards the
assembler re-assembles with ``--resume``.

Usage: ``R4_THREADS=6 python benchmarks/biaxial_shear_band/run_r7_shard.py <i>``,
``i in 0..7`` (seed order 53, 67, 79, 83, 89, 97, 101, 103).
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


def main() -> int:
    i = int(sys.argv[1])
    from benchmarks.biaxial_shear_band.run_r7_wide_window_ensemble import (
        CACHE, LATE_WINDOW_FRACTION, POST_PEAK_STRAIN, STRAIN_CAP, _configs,
        _protocol_gate, _run_cached)

    _protocol_gate()
    config = _configs(False, None)[i]
    assert config["post_peak_strain"] == POST_PEAK_STRAIN
    assert config["strain_cap"] == STRAIN_CAP
    assert config["late_window_fraction"] == LATE_WINDOW_FRACTION
    started = time.time()
    result = _run_cached(config, CACHE, True)
    stopping = result.get("stopping") or {}
    print(f"[shard {i}] {config['name']} stop={stopping.get('stop_reason')} "
          f"eps_log={stopping.get('final_axial_strain_log'):.4f} "
          f"n_late={result.get('n_late_snapshots')} "
          f"theta={((result.get('cumulative') or {}).get('late_window') or {}).get('radon')} "
          f"resolved={((result.get('cumulative') or {}).get('late_window') or {}).get('resolved')} "
          f"wall={time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())