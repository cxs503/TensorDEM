# DEM3D specimen-response benchmark layer

Protocol: `tensordem-dem3d-specimen-response-v1`

## Implemented case

`affine_compression_patch` imposes a prescribed small axial compression on the existing bonded-sphere lattice at three engineering strain levels. It records top-layer axial resultant, nominal engineering stress, apparent secant modulus, particle/bond counts and damage state. It runs twice and compares signatures and stresses.

Run it from the repository root:

```bash
python scripts/run_dem3d_specimen_response_benchmarks.py --output results-dem3d-specimen-response
python -m unittest tests.test_dem3d_specimen_response_benchmarks -v
```

Outputs:

- `specimen_response_report.json`: checks, repeatability signature, modulus spread and scope notes;
- `specimen_stress_strain.csv`: stress-strain response rows.

## Important interpretation

This is a **kinematic patch test**, not a standard uniaxial-compression-strength (UCS) experiment. The strain field is prescribed affinely, so the reported secant modulus is an apparent network response under this kinematic constraint; it is not yet a calibrated Young's modulus. Passing demonstrates repeatability and near-linearity for the exercised configuration only.

## Next specimen-level milestones

1. Implement plane platens or controlled boundary layers with explicit reaction accounting and loading-rate control.
2. Implement standard UCS with lateral relaxation and peak-load/failure reporting.
3. Implement Brazilian splitting with roller/strip fixtures and diametral tensile failure.
4. Implement three-point bending with support rollers, loading nose and optional notch.
5. Compare force-displacement, peak load, crack path and energy measures with independent experimental or published data. Keep calibration data separate from validation data.

Hertz-Mindlin contact, tangential friction history, rolling resistance and calibrated ice constitutive response are not implied by this benchmark.
