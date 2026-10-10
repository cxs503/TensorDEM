# DEM3D benchmark visualization outputs

## What is generated

The benchmark runner writes `visualization_trajectory_3d.pt` alongside the existing checkpoint, history and fracture-event CSV. It stores positions, alive-bond masks and prescribed indenter position at the same sampled steps used for diagnostics; it does not add integration steps or alter the benchmark signature.

Install visualization dependencies and run:

```bash
python -m pip install -e ".[viz]"
python scripts/run_dem3d_benchmark_suite.py --output results-dem3d-benchmark-suite
python scripts/visualize_dem3d_benchmarks.py --input results-dem3d-benchmark-suite --output results-dem3d-visualizations
```

For cases with saved state histories, each case folder receives `damage_initial.png`, `damage_final.png`, `displacement_final.png`, and `damage_evolution.gif`. A supported history CSV also produces `force_history.png`. `visualization_summary.json` and per-case `visualization_manifest.json` list what was actually emitted.

Damage is colored by the fraction of incident bonds that have broken at each particle. Displacement is the magnitude relative to the initial particle position. These are particle fields, not continuum stress contours. The GIF shows sampled states only and does not interpolate. If a case has no sampled trajectory, the script can still produce a force-history plot from a recognized CSV but does not fabricate a cloud map.

## Limits

This is post-processing only; it does not change solver mechanics or numerical benchmark verdicts. A visually plausible cloud is not evidence of physical calibration, convergence, or experimental validation. Matplotlib and Pillow are optional dependencies.
