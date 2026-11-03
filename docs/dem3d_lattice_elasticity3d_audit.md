# DEM3D lattice elastic constants: closed-form calibration and finite-sample bias

## Why this audit

`src/tensordem/dem3d.py` states that the simple-cubic bonded-sphere lattice "must be
calibrated before engineering use" its directional stiffness. Until now no number
existed for that stiffness: `C11`, `C12`, `C44`, the Poisson ratio and the shear
modulus of the 3-D bond network were undocumented, so any comparison of a simulated
stress-strain response with an ice material law was unfalsifiable.

This audit supplies the missing anchor in closed form and then measures how far the
benchmark geometries actually used in this repository sit from it.

## Method

`src/tensordem/lattice_elasticity3d.py`

* `periodic_elastic_constants` -- affine (Born) homogenisation of the solver's own bond
  topology. Each node owns thirteen bonds (three face neighbours plus six in-plane edge
  diagonals and four body diagonals, half-space filtered exactly as `IceDEM3D` builds
  them), so for an axial spring network

      C_ijkl = (1/a^3) * sum_{13 bonds} k * l^2 * n_i n_j n_k n_l .

  Term by term: one face bond of length `a` (weight `k a^2`), four length-`a*sqrt(2)`
  diagonals per plane (weight `2 k a^2`, direction cosine squared `1/2`) and four
  length-`a*sqrt(3)` body diagonals (weight `3 k a^2`, direction cosine squared `1/3`).
  This leaves the exact rationals `C11 = 13k/(3a)`, `C12 = C44 = 7k/(3a)`.
* `relaxed_uniaxial_modulus` -- Young's modulus of a finite sample with free lateral
  faces, solved as a linear equilibrium of the same axial-spring law, under two
  protocols: prescribed platen displacement (both end faces plane) and applied traction
  (load control).
* `derived_moduli` -- `E`, `nu`, `K`, Voigt/Reuss shear bounds and the Zener anisotropy.

Reference values are hand-derived from the bond enumeration above; the unit tests
assert the rationals to twelve digits, the Cauchy relation `C12 = C44`, cubic symmetry
and the linear scaling with `k` and `1/a`.

## Results (default config: r = 0.025 m, a = 0.05 m, k = 2000 N/m)

| quantity | analytic | note |
|---|---|---|
| `C11` | 13k/3a = 173.33 kPa | lateral-strain-constrained modulus |
| `C12 = C44` | 7k/3a = 93.33 kPa | Cauchy relation, exact for central forces |
| `E` (uniaxial) | 2.7 k/a = 108.0 kPa | `C11 - 2C12^2/(C11+C12)` |
| `nu` | 0.35 | fixed by topology, not a free parameter |
| `K` | 3 k/a = 120.0 kPa | |
| Zener anisotropy | 2.333 | the network is cubic, not isotropic |

Finite-sample measurements against the 108.0 kPa anchor:

| lattice | control | `E` (kPa) | error vs analytic | `nu` |
|---|---|---|---|---|
| 9x5x3 (solver default) | displacement | 162.9 | +50.9 % | 0.334 |
| 9x5x3 | load | 162.7 | +50.7 % | 0.333 |
| 13x7x5 | load | 133.6 | +23.7 % | 0.340 |
| 17x9x7 | load | 124.3 | +15.1 % | 0.342 |
| 21x11x9 | load | 120.2 | +11.3 % | 0.344 |
| 25x15x11 | load | 118.1 | +9.4 % | 0.344 |

(Exact numbers are programmatically written to
`outputs/lattice_elasticity3d/lattice_elasticity3d_samples.csv`; the table above is a
transcription of that file.)

## Findings

1. The lattice is a **central-force network**, so `C12 = C44` exactly: the shear
   modulus cannot be tuned independently of the Poisson ratio. Any calibration that
   tries to hit both ice `E` and ice `G` with a single `bond_stiffness` is over-determined.
2. The network's Poisson ratio is **0.35**, only 0.02 above mid-range sea ice (0.33),
   but that agreement is a property of the 26-neighbour topology, not a calibration
   result. It cannot be adjusted without changing the neighbour set.
3. **Platen-modelled samples are systematically stiffer than the material they
   represent.** At the solver default `9x5x3` the measured uniaxial modulus exceeds the
   infinite-lattice value by 51 %, and still by 9.4 % at `25x15x11`. The cause is the
   plane-stays-plane end constraint and the free lateral surfaces: the sample is too
   short for the homogeneous stress state to develop. Any modulus inferred from a
   platen compression test on the default lattice (see the existing platen benchmarks)
   must be divided by roughly 1.5 before it is compared with a material law, or the
   sample must be made much more slender.
4. **Scale gap.** With `k = 2000 N/m` and `a = 0.05 m` the lattice represents
   `E = 108 kPa`, i.e. 8.3e4 times softer than mid-range ice (`9 GPa`). Matching the ice
   modulus would require `k = 1.67e8 N/m`, which lowers the conservative critical time
   step from 3.37e-5 s to 3.91e-7 s (factor 86). This quantifies why the current
   benchmarks are quasi-static or artificially slow, and why mass/stiffness scaling with
   an explicit statement of the scaling law is unavoidable for transit-scale ice loads.

## Limits

* The closed-form constants are exact only for the infinite lattice; the finite-sample
  values are bracketed from above and converge slowly (roughly 1/L), so no extrapolated
  "measured" modulus is quoted.
* Central-force bond law only: no bending, torsion or explicit shear bond stiffness is
  modelled, so `C12 = C44` is a structural property of the model, not an ice property.
* No experiment is compared here. Ice reference values are mid-range literature values
  used only to express the scale gap.
* Only the simple-cubic `box` lattice is audited; `cylinder_xz` geometry has a different
  surface bond population and is not covered.

## Reproduce

```bash
PYTHONPATH=src python scripts/audit_dem3d_lattice_elasticity3d.py --output outputs/lattice_elasticity3d
PYTHONPATH=src python -m pytest tests/test_dem3d_lattice_elasticity3d.py -q   # 18 passed
```
