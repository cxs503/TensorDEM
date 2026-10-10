# DEM3D wedge-bow resistance sensitivity campaign

## Run

```bash
python scripts/benchmark_dem3d_wedge_resistance_sensitivity.py --output results-dem3d-wedge-sensitivity
python -m unittest tests.test_dem3d_wedge_resistance_sensitivity -v
```

The default campaign contains nine cases: three wedge angles (30°, 45°, 60°), three bow speeds (0.05, 0.1, 0.2 m/s), and three timestep factors (1, 1/2, 1/4). Every case is run twice. The report records deterministic signatures, finite-value checks, peak/mean opposing resistance, integrated resistance work, broken-bond count/fraction, first-fracture time, timestep, duration and bow travel. JSON, summary CSV and full history CSV are written.

## Comparison policy

- Angle sweep reference: 45° at 0.1 m/s and nominal timestep.
- Speed sweep reference: 0.1 m/s at 45° and nominal timestep.
- Timestep sweep reference: finest timestep in the selected campaign.
- Timestep cases scale step count inversely with timestep factor to approximately preserve physical duration. Angle/speed cases retain the same step count and nominal timestep.
- Relative deltas quantify numerical/model sensitivity. They are not errors against measured data, and no monotonic trend is assumed as a pass condition.

## Acceptance and limitations

The hard gate requires finite histories, repeatable CPU signatures/metrics, forward bow travel and nonnegative integrated resistance work. The workflow deliberately does not require a specific force magnitude or fracture event because those depend on the selected model parameters and simulation duration.

This remains an idealized dry-contact prototype with an infinitely extruded wedge. It excludes fluid dynamics, hydrostatic pressure, buoyancy, ship motions, finite-width hull geometry and calibrated ice/hull properties. A stable sensitivity result is necessary but not sufficient for physical validation; measured resistance and fracture data are still required.
