# DEM3D fracture objectivity and energy validation

This regression campaign complements UCS curve fitting with three controlled numerical probes. It is a software/mechanics verification tool, not a calibrated ice material model.

## Run

```bash
python scripts/validate_dem3d_fracture.py --output results-dem3d-fracture-validation
python -m unittest tests.test_dem3d_fracture_objectivity -v
```

The script writes `dem3d_fracture_validation.json` with individual check results and tolerances.

## Checks

1. **Rigid-rotation objectivity:** deform a small specimen, record internal forces, apply a proper rigid rotation plus translation, and compare the rotated force field with the recomputed field. The relative force covariance error must be below `1e-8`; the probe must not create damage.
2. **Single-bond release-energy identity:** isolate one active bond, extend it beyond its tensile threshold, and verify the stored event energy against `0.5 * bond_stiffness * extension**2` evaluated at failure.
3. **Mixed-mode and cumulative ledger:** apply a combined extension/shear deformation, require at least one mixed-mode failure, and verify that the cumulative release ledger equals the sum of per-bond event energies.

## Interpretation and limits

- The single-bond probe deliberately isolates one active bond to test the event-energy formula, not to reproduce a laboratory fracture specimen.
- The mixed-mode probe verifies classification and bookkeeping under a controlled deformation; it does not validate a physical mixed-mode fracture criterion.
- The reported bond failure energy is the elastic energy stored in the bond at the failure configuration under the current linear spring law. It is **not** a calibrated material fracture energy (G_c), and it does not by itself prove that all released energy is physically dissipated.
- The audited energy-balance residual remains a discrete diagnostic affected by explicit integration, damping/work quadrature and how broken-bond energy is removed. Do not claim exact conservation from a finite residual.
- Before using the model for ship-ice resistance, validate against independent tensile/splitting, shear, compression and indentation experiments, with documented specimen geometry, temperature, salinity, loading rate and uncertainty.

A green CI run means these coded invariants passed; it does not establish experimental agreement, mesh independence or engineering predictive capability.
