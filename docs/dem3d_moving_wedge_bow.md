# DEM3D moving wedge-bow contact benchmark

## Purpose

This benchmark adds a prescribed, rigid wedge bow to the 3-D bonded-sphere DEM. The wedge is a two-sided profile in the x-z plane, extruded uniformly along y, and translates in +x at a constant prescribed speed. The model reports the opposing x reaction as ice resistance, vertical reaction, contact elastic energy, bond failures and a time history.

## Run

```bash
python scripts/benchmark_dem3d_moving_wedge_bow.py --output results-dem3d-moving-wedge-bow
python -m unittest tests.test_dem3d_moving_wedge_bow -v
```

Parameters include `--steps`, `--bow-speed` (m/s) and `--wedge-angle-deg`. Outputs are a JSON report, per-step history CSV and summary CSV. The run is repeated internally and the rounded history signature is compared for determinism.

## Sign convention and metrics

- The wedge reaction is the force of the ice on the prescribed bow.
- Reported ice resistance is `max(-bow_reaction_x_N, 0)`, i.e. the nonnegative component opposing forward motion.
- Integrated resistance work is the trapezoidal time integral of resistance multiplied by the prescribed speed.
- First-fracture time is the first sample where the cumulative broken-bond count becomes nonzero.

## Model scope and limitations

This is a **dry contact mechanics prototype**, not a complete ship-ice interaction solver. The bow has no finite beam, stem curvature, keel, or free-surface representation; the wedge is uniform across the transverse direction. The model does not include fluid dynamics, hydrostatic pressure, buoyancy, ship heave/pitch, ice floe hydrodynamics, or calibrated hull/ice parameters. The current bond lattice and failure thresholds require calibration and resolution/timestep sensitivity studies before interpreting resistance as a physical prediction.

The smoke benchmark verifies finite, deterministic execution and output persistence. It does not require fracture to occur for every parameter set, nor does a passing result imply agreement with measured ice resistance.
