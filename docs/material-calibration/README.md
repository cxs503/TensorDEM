# Independent central-spring material qualification

This benchmark diagnoses why particle refinement alone cannot qualify the present ice model. It preserves `dem.py` and previous coupling evidence. Three square grids occupy the same **0.8 × 0.4 × 0.2 m** continuum domain with total mass **58.688 kg**. Boundary control-volume weights preserve this mass; these are deliberately **not** IceDEM's overlapping disk mass convention. The material target E = 1 MPa, ν = 0.3 is a declared synthetic plane stress target, not experimentally calibrated ice.

The new module reproduces IceDEM's axial and both diagonal central-spring bonds. Tests compare exact bond indices against an actual IceDEM instance and compare actual finite extension spring energy with the linearized patch energy. A single stiffness in N/m is calibrated independently at each resolution to target C11 = E/(1−ν²). This mapping can be supplied to `DEMConfig.bond_stiffness`, but it does **not** calibrate disk mass, fracture thresholds, contact, damping or shear response. Existing wet particle sensitivity remains failed.

## Actual results

| Height cells | C11 error | C12 error | Shear modulus error | Prescribed bending energy error |
|---|---:|---:|---:|---:|
| 8 | 0% | 56.86% | 34.45% | 12.5% |
| 16 | 0% | 61.62% | 38.53% | 6.25% |
| 32 | 0% | 64.10% | 40.66% | 3.125% |

Bending here prescribes **all** displacements from u_x = −κxy, u_y = κx²/2 and compares energy to C11 κ² L t H³/24. It is not a free cantilever or transverse-relaxed bending verification. This energy discretization improves while the constitutive mismatch persists.

A separate real assembled central-spring equilibrium solve prescribes left/right x extension, leaves all transverse displacements free, and fixes one y displacement to remove rigid translation:

| Height cells | Measured E (Pa) | E error | Apparent ν | Max free-DOF residual (N) |
|---|---:|---:|---:|---:|
| 4 | 865666 | 13.43% | 0.4644 | 1.07e−14 |
| 8 | 846839 | 15.32% | 0.4824 | 1.78e−14 |
| 16 | 836138 | 16.39% | 0.4915 | 2.31e−14 |

The apparent ν uses mean top/bottom transverse strain. Reaction-based E uses net right boundary force divided by the original cross-sectional area and prescribed axial strain. Raw equilibrium displacements/reactions, geometries, masses, bonds and source hash are retained in `study.json`. All material gates remain **false** with a declared 3% tolerance.

For the infinite square network with equal spring stiffness k, C11 = C22 = 2k/t, C12 = k/t, G = k/t. The target isotropic relation G = (C11−C12)/2 does not hold. Finer particles approach this anisotropic central-spring material rather than the desired E, ν. Independent axial/diagonal stiffnesses can remove square anisotropy only under additional restrictions; a general target requires a richer constitutive mechanism (angular or rotational bonds, bending moments or a continuum-consistent force law). Those mechanisms must pass tension, shear and free bending gates before recalibrating the wet fracture case.

## Reproduce

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/material_calibration.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/material_calibration.py --audit
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m unittest discover -s tests -p test_material_calibration.py -v
```

The audit regenerates and exactly compares source hash, raw fields and metrics. This verifies reproducibility; it does not promote the material to a physically qualified ice model. Dense equilibrium solves are limited to 600 nodes. No dynamic fracture, experimental ice comparison, cantilever convergence or physical three-dimensional shell ice is claimed.

## Executable correction for the realizable ν = 1/3 material

The benchmark also implements a **separate corrected network** with the same bond topology. For E = 1 MPa and t = 0.2 m, diagonal k = 3Et/8 = **75000 N/m**, axial k = **150000 N/m**. Axial bonds lying on a rectangle boundary receive control-volume weight 1/2. Coefficients are returned per bond by `corrected_stiffness(positions, bonds, E, t)`; they cannot be represented by the present scalar `DEMConfig.bond_stiffness`.

On 4, 8 and 16 height-cell grids, C11 = 1125000 Pa, C12 = G = 375000 Pa match the ν = 1/3 target to < 4e−16 relative error. The actual free-transverse equilibrium solves produce E = 1 MPa with relative error **1.6e−15, 1.0e−14, 4.6e−14**, and apparent ν = 1/3 to machine precision. These uniform patch fields are exact at all three grids; this is patch reproduction rather than a measured asymptotic convergence rate. All three corrected patch gates pass C11/C12/G/E within 3% and absolute ν within 0.01.

Raw per-bond coefficients, equilibrium displacements and reactions are retained under `corrected_cases`. The ν = 0.3 failures remain untouched. This correction is usable as an independent material network and demonstrates the necessary mapping, but `actual_IceDEM_backend_qualified` remains false: a future per-bond stiffness interface must consistently update spring force, energy, failure release, timestep bounds and snapshot validation before replacing the live coupling model. Nonuniform deformation, free cantilever bending, fracture calibration and wet particle refinement still require qualification.
