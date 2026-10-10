# DEM3D UCS timestep convergence campaign

## Purpose

Quantify how dynamic uniaxial platen compression changes when the recommended stable timestep is refined by factors 1, 1/2 and 1/4 while keeping the nominal physical duration and loading speed approximately fixed. The campaign compares peak load, peak engineering stress, strain at peak, broken-bond count and first fracture time.

## Run

```bash
python scripts/benchmark_dem3d_ucs_timestep_convergence.py --output results-dem3d-ucs-timestep
python -m unittest tests.test_dem3d_ucs_timestep_convergence -v
```

Options include `--base-steps` (steps at timestep factor 1) and `--relative-tolerance` (peak-stress screening threshold, default 0.10). Outputs include JSON, summary CSV and full sampled histories.

## Interpretation and status

- **PASS/FAIL verdict** covers finite histories and repeatable CPU runs.
- **Convergence PASS/WARN** compares each case's peak stress with the finest timestep in the campaign. WARN means the selected screening tolerance was exceeded; it is not a solver crash or a universal physical failure criterion.
- First-fracture-time deltas are undefined if either case has no fracture event. Broken-bond counts and strain at peak remain separately reported because event timing can shift discretely with the timestep.
- The finest timestep is only a within-campaign reference, not an exact solution. Tolerance must be selected for the intended application and later justified against experiments or a higher-fidelity reference.

## Limits

This is numerical verification for a simple-cubic bonded-sphere prism with frictionless moving platens. It does not establish calibrated ice strength, fracture toughness, or full convergence in particle resolution, loading rate, or boundary conditions. No experimental measurements are invented or embedded.
