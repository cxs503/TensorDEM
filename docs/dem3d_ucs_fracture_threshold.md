# DEM3D UCS bond-failure threshold screening

Run from the repository root:

```bash
python scripts/benchmark_dem3d_ucs_fracture_threshold.py \
  --target-strain 0.015 --thresholds 0.01 0.015 0.02 \
  --output results-dem3d-ucs-fracture-threshold
```

The runner varies only the tensile bond `breaking_strain`; loading speed, particle
geometry, density, contact parameters, platen properties, and integration protocol
are held fixed. Every case is run twice and complete history signatures are compared.
The outputs contain a JSON report, summary CSV, and concatenated step-history CSV.
Reported observables include peak compressive load/stress, strain at peak stress,
broken-bond count/fraction, final bond elastic energy, kinetic energy, timestep,
and stress delta versus the reference threshold.

## Interpretation and limitations

A PASS means the numerical runs are finite, repeatable, reach the common target
strain, and report bounded fracture counts and nonnegative energies. It does not
mean a material parameter has been calibrated. The breaking-strain threshold is
not fracture energy: the energy dissipated by fracture also depends on bond
stiffness, bond geometry, failure mode, and the implemented release rule.

To calibrate against ice, supply independent measured stress-strain curves and
fracture observations at known temperature, salinity, loading rate, and specimen
geometry. Define an objective and uncertainty range before fitting; then jointly
test peak stress, strain at peak, post-peak response, broken-bond statistics, and
energy balance. Avoid fitting only one peak stress because different parameter
sets can produce similar peaks. This campaign intentionally performs no inverse
fit and makes no claim about physical ice strength or fracture toughness.
