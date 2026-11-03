# DEM3D bond fracture energy: closed-form calibration and controlled tensile audit

## Why this exists

`docs/dem3d_material_calibration_roadmap.md` listed four open items that all touch
the same uncalibrated pair of numbers (`bond_stiffness = 2000 N/m`,
`breaking_strain = 0.015`):

* audit bond energy before failure and released energy at failure under a
  controlled test;
* establish a fracture-energy calibration instead of equating breaking strain
  with fracture energy;
* quantify the sensitivity of the parameter set to particle resolution;
* do not present a fitted breaking strain as a physical fracture property.

This increment answers them with a closed-form law, a lattice-resolved numerical
check and a controlled tensile audit of the solver's own bookkeeping. It adds
`src/tensordem/bond_fracture_energy.py`, the benchmark script
`scripts/benchmark_dem3d_bond_fracture_energy.py` and
`tests/test_dem3d_bond_fracture_energy.py`. The solver itself is not modified.

## Model recap (as implemented in `tensordem.dem3d`)

* simple-cubic lattice, spacing `a = 2 * radius`, so `a = 0.050 m` for the default radius;
* every neighbour pair inside the initial cutoff carries a linear central spring
  (`bond_stiffness`) with rest length equal to its initial distance: 26 neighbours
  per interior particle (axial, face-diagonal, body-diagonal);
* a bond fails irreversibly when `strain = (L - L0) / L0 > breaking_strain`, or
  when the reconstructed deviatoric shear strain exceeds
  `shear_breaking_strain = 0.03`;
* the solver stores, per failed bond, `0.5 * k * extension²` at the moment of
  detection, and `diagnostics()["released_bond_energy_J"]` sums that ledger.

## The law

A crack sheet of area `A`, normal to a cube axis, cuts each lattice cell
cross-section `a²` with nine bond families (`dz = +1`, `dx, dy` in `{-1, 0, 1}`),
whose rest lengths are `a * sqrt(1 + dx² + dy²)`. Summing the failure energy of
those nine families and dividing by `a²`:

```text
G_f = 0.5 * k * eps_b^2 * SUM_over(dx, dy in {-1,0,1}) (1 + dx^2 + dy^2)
    = 0.5 * k * eps_b^2 * 21
    = 10.5 * k * eps_b^2    [J/m^2]
```

The spacing cancels: bonds per unit area scale as `1/a²` while the stored energy
per bond scales as `a²`. Two consequences follow, and both are verified below:

1. the equivalent fracture energy does **not** depend on which cube axis the
   crack plane is normal to (lattice isotropy of the *energy*, not of the elastic
   stiffness);
2. refining the particle size at fixed `(k, eps_b)` does **not** change `G_f`, so a
   parameter set calibrated at one resolution stays fracture-energy equivalent at
   another. (The elastic modulus does change with resolution; that is a separate
   calibration axis.)

For the default parameters, `G_f = 10.5 * 2000 * 0.015^2 = 4.7250 J/m²`.

## Finite-specimen check

The closed form is the edge-free limit. Counting the bonds that actually cross a
mid-specimen plane in the solver's own bond list gives the table below. Two
normalisations are shown because a finite specimen has no unique "area": the
centre-to-centre footprint `(n-1)²a²` and the particle-population area `n²a²`.
Both bracket the closed form and converge as `O(1/n)`; the footprint error decays
as roughly `50/n` percent.

| n (n×n×n) | particles | bonds | crossing bonds | G_f footprint [J/m²] | G_f population [J/m²] | err footprint | err population |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 125 | 1036 | 169 | 5.3016 | 3.3930 | 12.20% | 28.19% |
| 7 | 343 | 3258 | 361 | 5.1062 | 3.7515 | 8.07% | 20.60% |
| 9 | 729 | 7448 | 625 | 5.0098 | 3.9583 | 6.03% | 16.23% |
| 13 | 2197 | 24228 | 1369 | 4.9141 | 4.1871 | 4.00% | 11.38% |
| 17 | 4913 | 56368 | 2401 | 4.8665 | 4.3108 | 2.99% | 8.77% |
| 21 | 9261 | 108860 | 3721 | 4.8381 | 4.3883 | 2.39% | 7.13% |

## Crack-plane orientation and resolution

Crack plane normal x / y / z at n = 11:

| crack plane normal | G_f [J/m²] | closed form [J/m²] |
| --- | --- | --- |
| x | 4.952250 | 4.725000 |
| y | 4.952250 | 4.725000 |
| z | 4.952250 | 4.725000 |

Particle radius sweep at n = 11 (`G_f` identical to machine precision):

| radius [m] | spacing a [m] | particles | G_f [J/m²] |
| --- | --- | --- | --- |
| 0.01 | 0.02 | 1331 | 4.952250 |
| 0.015 | 0.03 | 1331 | 4.952250 |
| 0.02 | 0.04 | 1331 | 4.952250 |
| 0.025 | 0.05 | 1331 | 4.952250 |
| 0.05 | 0.1 | 1331 | 4.952250 |

## Physical scale and calibration targets

Sea-ice fracture toughness is reported around `K_Ic = 0.05–0.15 MPa·m^0.5` with
`E = 5–10 GPa`; plane strain gives `G = K_Ic²(1 - ν²)/E`, i.e. a band of
**0.2275–4.0950 J/m²** at ν = 0.3. This is a scale for judging a parameter
set, not a measurement of the ice in any particular simulated case.

Equivalent fracture energy versus breaking strain at `k = 2000 N/m`:

| breaking strain ε_b [-] | G_f [J/m²] | inside physical band |
| --- | --- | --- |
| 0.004 | 0.3360 | yes |
| 0.006 | 0.7560 | yes |
| 0.008 | 1.3440 | yes |
| 0.01 | 2.1000 | yes |
| 0.012 | 3.0240 | yes |
| 0.015 | 4.7250 | no |
| 0.018 | 6.8040 | no |
| 0.022 | 10.1640 | no |
| 0.026 | 14.1960 | no |
| 0.03 | 18.9000 | no |

Inverse calibration (what to set for a target `G_f` at fixed `k`):

| target G_f [J/m²] | required ε_b at k = 2000 N/m |
| --- | --- |
| 0.2275 | 0.003291 |
| 2.1612 | 0.010145 |
| 4.0950 | 0.013964 |

The default set therefore represents `4.7250 J/m²`: 2.18× the band
midpoint and 15.4 % above the band ceiling. It is not absurd, but it sits
at the high end, and it was never chosen deliberately — it is the product of two
independently chosen regression defaults. Lowering `breaking_strain` from
0.015 to 0.0101 would place the model at the band midpoint.

## Controlled tensile audit

`controlled_tensile_test` prescribes an affine axial stretch on a
3×3×3 specimen in increments of `0.0005` up to
`0.0200`, with drag and contact damping disabled and the shear criterion
parked at 1.0, so neither time-integration error nor shear breakage contaminates
the ledger. Results:

| quantity | value |
| --- | --- |
| bonds in specimen | 158 |
| bonds broken | 66 |
| bonds intact | 92 |
| recorded release energy | 2.873820609e-02 J |
| ledger `released_bond_energy_J` | 2.873820609e-02 J |
| ledger vs event-sum discrepancy | 1.207e-16 |
| max relative error, recorded vs 0.5 k δ² | 0.000e+00 |
| max relative error, recorded vs 0.5 k (ε_b L0)² | 0.99 % |
| detected strain range | 0.010000 – 0.010050 |
| max overshoot above ε_b | 4.950e-05 (increment 0.0005) |

Exactly the stretched bond families break: 66 of 158 bonds have a
non-zero axial projection, while the 92 bonds normal to the loading axis
never reach the failure strain at this stretch level — a physically correct and
previously untested property of the strain criterion.

The residual recorded-vs-analytic gap is the strain overshoot of one prescribed
increment, which enters quadratically: `2 * increment / ε_b ≈ 10 %` of the failure
strain would give ~10 % energy error, and the measured value is consistent with
`2 * 4.950e-05 / 0.015`.

## Error judgement against baselines

| quantity | value | baseline | relative error | <3 %? |
| --- | --- | --- | --- | --- |
| finite-specimen G_f, n = 21 | 4.8381 J/m² | closed form 4.7250 J/m² | 2.39% | yes |
| crack-plane orientation spread (x, y, z) | 2.665e-15 J/m² | zero spread expected | 0.00% | yes |
| resolution spread, radius 0.01–0.05 m | 6.217e-15 J/m² | zero spread expected | 0.00% | yes |
| per-bond recorded release energy vs ½kδ² | recomputed from recorded extension | exact identity | 0.00e+00% | yes |
| release ledger vs sum of events | 2.873820608792e-02 J | 2.873820608792e-02 J | 1.21e-14% | yes |
| recorded energy vs analytic failure point | detection with one-increment overshoot | ½k(ε_b L0)² | 0.99% | yes |
| default G_f vs physical band ceiling | 4.7250 J/m² | band 0.2275–4.0950 J/m² | +15.4% above ceiling | no — calibration statement, not a numerical error |

Every numerically verifiable quantity is inside 3 %. The one comparison that is
*not* is the default parameter set against the physical band ceiling, and that is
a calibration statement rather than a numerical error: the band is a literature
scale from `K_Ic`/`E` ranges, not a measured `G_f` for the ice of any DEM case, so
a percent-level agreement claim would be unjustified.

## Pitfalls and limits

* The law assumes a crack sheet normal to a cube axis. Inclined or rough cracks
  redistribute energy over bond families; the axis-normal sheet uses factor 10.5
  while a body-diagonal sheet uses 7.0, which is a real orientation dependence of
  the model.
* The **shear** criterion (`shear_breaking_strain`) releases the elastic energy of a
  bond that may be barely stretched axially. It is a strain criterion without a
  mode-II energy release rate, so `G_f` above applies to tensile sheets only.
* Equating a breaking strain with a fracture energy remains invalid *per se*: the
  mapping `G_f = 10.5 k ε_b²` holds **only** because this lattice stores energy in
  linear springs of pair-independent stiffness, which makes the bond energy exactly
  quadratic in rest length. Change the bond law (constant failure force, or a
  stiffness normalised per unit bond length) and the factor changes.
* The finite-specimen normalisation ambiguity (footprint vs population area) is
  genuine: any "measured fracture energy" from a small DEM specimen must state
  which area definition it used.
* No experimental tensile specimen is simulated here; this is a law plus a
  bookkeeping audit, not a validation of ice fracture.

## Reproduction

```bash
PYTHONPATH=src python scripts/benchmark_dem3d_bond_fracture_energy.py \
    --output artifacts/bond_fracture_energy
PYTHONPATH=src python scripts/render_dem3d_bond_fracture_energy_doc.py
PYTHONPATH=src python -m pytest tests/test_dem3d_bond_fracture_energy.py -q
```

Artifacts: `artifacts/bond_fracture_energy/bond_fracture_energy.json`,
`..._size.csv`, `..._strain.csv`, `..._resolution.csv`, and the figure
`bond_fracture_energy.png` (calibration law with the physical band; finite-specimen
convergence; resolution independence).
