#!/usr/bin/env python
"""Entry point for GAP-DEM-GPU-100K benchmark.

This is the canonical entry point that maturity_gate looks for.
It delegates to the actual benchmark implementation.

Usage:
    python benchmarks/gap_dem_gpu_100k/run.py [--device cpu|cuda] [--sweep]

Output artifacts:
    benchmarks/gap_dem_gpu_100k/results.json
    benchmarks/gap_dem_gpu_100k/report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the benchmark module is importable
BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmark_gpu_throughput import main as benchmark_main


def main(argv: list[str] | None = None) -> None:
    """Run the GAP-DEM-GPU-100K throughput benchmark."""
    benchmark_main(argv)


if __name__ == "__main__":
    main()
