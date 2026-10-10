# DEM3D UCS loading-rate and resolution sensitivity

Run from the repository root:

```bash
python scripts/benchmark_dem3d_ucs_sensitivity.py --output results-dem3d-ucs-sensitivity
```

The campaign runs six cases to a common target axial engineering strain:

- Loading-rate sweep: 0.025, 0.05 and 0.1 m/s at fixed particle geometry and spring parameters.
- Resolution sweep: 4×4×3, 5×5×4 and 6×6×5 particle prisms. Particle radius and spring stiffness are scaled to keep the specimen dimensions approximately comparable and use a first-order stiffness scaling assumption.

The runner derives each case's step count from the configured recommended timestep, platen speed, specimen height and target strain. Every case is run twice and its complete step-history signature is compared. Outputs include a JSON report, a per-case summary CSV, and a combined step-history CSV.

## Interpretation

Peak stress deltas are reported relative to the reference loading rate or coarsest resolution. They quantify sensitivity of the current discrete configuration; they are not evidence of convergence by themselves. Resolution cases do not have exactly identical dimensions, and linear spring scaling is only a first-order assumption. Physical material calibration requires measured reference data, lattice-bias checks, and a more complete rate/timestep/resolution convergence programme.
