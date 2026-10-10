# DEM3D moving wedge-bow contact benchmark

## Purpose

This benchmark adds a prescribed, rigid wedge bow to the 3-D bonded-sphere DEM. The x-z wedge profile can be extruded infinitely across y for legacy comparisons, or intersected with a finite transverse slab to model a finite beam. The default benchmark uses a finite bow half-width of 0.03 m. It reports the opposing x reaction as ice resistance, transverse/vertical reactions, contact elastic energy, bond failures and time histories.

## Run

```bash
python scripts/benchmark_dem3d_moving_wedge_bow.py --output results-dem3d-moving-wedge-bow
python -m unittest tests.test_dem3d_moving_wedge_bow -v
```

Parameters include `--steps`, `--bow-speed` (m/s), `--wedge-angle-deg`, and `--bow-half-width-m` (m). Set `--bow-half-width-m 0` to select the legacy infinite-width geometry. Outputs are a JSON report, per-step history CSV and summary CSV. The run is repeated internally and history signatures are compared for determinism.

## Geometry and contact

The finite-width body is represented as the intersection of the x-z wedge half-space and a transverse slab `|y - center_y| <= half_width`. The signed distance uses the maximum of the two limiting face distances. At a sharp edge/corner, the contact normal is assigned to the active limiting face; no edge smoothing or true curved stem is included. This is a practical first finite-width approximation, not a complete signed-distance representation of a ship hull.

## Sign convention and metrics

- The wedge reaction is the force of the ice on the prescribed bow.
- Reported ice resistance is `max(-bow_reaction_x_N, 0)`, i.e. the nonnegative component opposing forward motion.
- Integrated resistance work is the trapezoidal time integral of resistance multiplied by the prescribed speed.
- First-fracture time is the first sample where the cumulative broken-bond count becomes nonzero.

## Model scope and limitations

This is a **dry contact mechanics prototype**, not a complete ship-ice interaction solver. It has no stem curvature, keel, free-surface representation, fluid dynamics, hydrostatic pressure, buoyancy, ship heave/pitch, or ice floe hydrodynamics. The bond lattice and failure thresholds require calibration and resolution/timestep sensitivity studies before interpreting resistance as a physical prediction.

The smoke benchmark verifies finite, deterministic execution and output persistence. It does not require fracture to occur for every parameter set, nor does a passing result imply agreement with measured ice resistance.
