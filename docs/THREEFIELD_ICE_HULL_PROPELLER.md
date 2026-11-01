# Ice–hull–propeller three-field coupling: DEM contact, LBM propeller race, fragment thermodynamics

**Campaign node** W6-N02 · **repository** TensorDEM · **package** `tensordem.threefield`
**Artifacts** `artifacts/threefield/` · **Tests** `tests/test_threefield.py` (25 cases)

---

## 1. Case description

An ice-going ship breaks a level-ice sheet with its bow, producing a debris field
of small floes behind the stem. Those floes are then entrained by the propeller
race, where they are (i) further fragmented by turbulence and (ii) melted by the
warm, turbulent water. The engineering question is how much of the broken-ice
mass is removed from the channel and how the surviving floe population looks for
a following ship.

| component | configuration used here |
|---|---|
| hull–ice DEM | moving wedge bow, wedge angle 45°, bow speed 0.10 m/s, bonded simple-cubic ice sheet 9 × 4 × N<sub>z</sub> particles, particle radius 0.025 m, ice thickness 0.30 / 0.45 / 0.60 m (6 / 9 / 12 layers), ρ<sub>i</sub> = 917 kg/m³, bond stiffness 1200 N/m, contact stiffness 3500 N/m, breaking strain 8·10⁻³ / 25·10⁻³ (axial/shear) |
| reference hull (formula baselines) | L = 150 m, B = 22 m, T = 8 m, waterline entry half-angle 22°, stem angle 30°, buttock angle 20°, μ = 0.05, V = 0.5 m/s |
| ice material (baselines) | σ<sub>c</sub> = 1.5 MPa, σ<sub>f</sub> = 0.5 MPa, E = 5 GPa, h = 0.30 / 0.45 / 0.60 m |
| propeller race LBM | D3Q19 + Smagorinsky C<sub>s</sub> = 0.14, τ₀ = 0.5005 (ν = 1.67·10⁻⁴), actuator disk D = 10 lattice units (Gaussian exit profile), U<sub>in</sub> = 0.04, U<sub>jet</sub> = 0.12, Re = 7200, quarter domain 72 × 36 × 36, 2100 steps, 840 time-averaged samples |
| thermal | water 0.5 °C, ice at melting point 0 °C, k<sub>w</sub> = 0.55 W/m·K, ν = 1.79·10⁻⁶ m²/s, Pr = 12.5, L<sub>f</sub> = 3.34·10⁵ J/kg, relative velocity 0.06 m/s, residence time 30 s |
| race/fragmentation driver | race velocity 5 m/s, turbulence intensity 15 % → u′ = 0.75 m/s |

## 2. Method

1. **DEM hull–ice contact** (`hull_ice.py`): the moving-wedge bow of
   `dem3d_wedge` is run as a *unit-width* section (sheet width 0.20 m), so the bow
   reaction reduces to N/m and can be compared with per-unit-width engineering
   baselines without a geometric fudge factor. Per step the module records the
   bow reaction, the number of particles in contact, the contact area, and the
   mean/peak normal pressure (F/A over contacting particles). At the end the
   surviving bond network is decomposed into connected components → fragment
   masses and equivalent diameters; the census conserves the ice mass to 1e-12.
2. **LBM propeller race** (`wake_lbm.py`): the actuator disk is forced with a
   Gaussian radial velocity bell (zero at hub and tip) and 15 % target noise, so
   the exit profile has no single-cell shear layer. Free-slip (specular) lateral
   boundaries allow a quarter domain cut on the two symmetry planes. Statistics:
   time-averaged mean and rms velocity fields, azimuthally binned profiles.
3. **Race model comparison**: the empirical reference is the self-similar
   Gaussian race whose centreline follows exactly from momentum conservation
   (thrust = excess momentum flux) and whose width follows σ = σ₀ + β(x−x₀).
   The gate is the profile *shape* error in the race core (r ≤ 2.5σ) with the
   width identified from the resolved profile; the literature spreading rate
   β = 0.085 is used for the predictive audit.
4. **Fragmentation** (`fragmentation.py`): bending-stability length
   L<sub>crit</sub> = √(8σ<sub>f</sub>h/(3ρ<sub>w</sub>C<sub>d</sub>u′²)) and a
   mass-conserving split cascade.
5. **Thermal erosion** (`thermal.py`): Ranz–Marshall convection with the Stefan
   interface condition, RK2 integration, graded against the closed-form solution.
6. **Ledger & downstream impact** (`coupling.py`): M₀ = Σ M<sub>frag</sub> + M<sub>melt</sub>
   and the follower-ship metrics.

Toolchain: PyTorch 2.7 (CPU, 8 threads), NumPy-free, matplotlib for figures;
all numbers in this document are read from `artifacts/threefield/*.json`.

## 3. Results

### 3.1 DEM hull–ice contact (unit width)

| h [m] | mean resistance [N/m] | crushing-phase mean [N/m] | peak [N/m] | mean contact pressure [Pa] | fragments |
|---|---|---|---|---|---|
| 0.30 | 77.54 | 81.03 | 119.04 | 1328 | 9 |
| 0.45 | 36.59 | 445.36 | 800.44 | 3814 | 196 |
| 0.60 | 103.58 | 103.58 | 161.32 | 1729 | 17 |

The scatter (0.45 m giving the largest crushing-phase force while 0.60 m gives
one of the smallest) is a first-order finding, not noise: with 216–432 particles
the bow advances only 1–3 grain diameters before the sheet shatters, so the
"resistance" is a transient brittle event whose magnitude is set by which bond
network happens to be loaded. No h-trend can be established at this resolution.

### 3.2 Propeller race (lattice-Boltzmann)

| station [lu] | U<sub>c</sub> [lu] | u′<sub>rms</sub> [lu] | σ<sub>fit</sub> | σ<sub>model</sub>(β=0.085) | shape error (core) | full-domain error | R² (core) |
|---|---|---|---|---|---|---|---|
| 21 | 0.130 | 0.0030 | 2.63 | 3.35 | **2.68 %** | 12.19 % | 0.9986 |
| 31 | 0.127 | 0.0036 | 2.53 | 4.20 | **2.41 %** | 10.19 % | 0.9990 |
| 41 | 0.124 | 0.0042 | 2.40 | 5.05 | **5.80 %** | 8.90 % | 0.9948 |

Thrust (lattice) 0.0707; excess momentum flux 0.0842 / 0.0770 / 0.0753 at the
three stations (19 % above thrust at the first station, decaying to 6 % above —
the residual is the pressure term of the bounded domain). Mass flux is constant
to 0.1 % along the domain. Gaussian log-linear fit R² = 0.994–0.999, i.e. the
race *is* self-similar Gaussian. The resolved spreading, however, is
β<sub>measured</sub> = −0.012 against β = 0.085 (the resolved width slightly
*shrinks*): at 10 lattice cells per propeller diameter and Re = 7200 the race is
under-resolved and mixing is laminar-like.

### 3.3 Fragmentation and melting

| quantity | value |
|---|---|
| richest DEM census | 196 fragments (h = 0.45 m) |
| L<sub>crit</sub> at u′ = 0.75 m/s, h = 0.10 m | 13.88 m |
| split events | 0 (all debris ≪ L<sub>crit</sub>) |
| Weibull (CDF least squares) of the census | k = 0.97, A = 0.038 m, **R² = 0.030** |
| census vs experimental Weibull (k = 1.8) | **R² = −2.24**, max CDF deviation 0.92 |
| melting over 30 s residence (largest fragment) | 0.296250 m → 0.296224 m, melt fraction 2.59·10⁻⁴ |
| numeric vs closed-form Ranz–Marshall | diameter 1.9·10⁻¹⁴ %, melt fraction 1.7·10⁻¹⁰ % |
| coupled mass ledger (19.4457 kg) | residual 8.5·10⁻¹⁴ kg, relative error **4.4·10⁻¹⁵** |

**Outlook (not a gate).** Applying the same breakup criterion to a field-scale
floe population (0.5–40 m, log-uniform, 400 floes, an *assumed* field, not a
measurement) produces 577 split events, 1272 fragments and a Weibull fit with
R² = 0.980 (k = 1.07, A = 6.91 m). This shows that the cascade machinery does
reproduce a Weibull size spectrum when the initial population is physically
sized; it is deliberately excluded from the acceptance verdict.

### 3.4 Downstream impact on a following ship

| metric | value |
|---|---|
| melted fraction of the broken-ice mass | 0.247 % |
| mean surviving floe diameter / initial largest floe | 0.131 |
| floe-refinement resistance relief (1 − d/d₀) | 0.869 |
| melted-fraction resistance relief | 0.0025 |

Interpretation: over a 30 s race residence the thermal effect is negligible
(0.25 % of the mass); the channel benefit comes almost entirely from the small
floe size, i.e. from the mechanical breaking already captured by the DEM, not
from melting.

## 4. Figures

| file | content |
|---|---|
| `artifacts/threefield/figures/fig1_resistance.png` | DEM unit-width resistance vs Lindqvist/Riska baselines (log scale) and the uncalibrated error bars against the 15 % gate |
| `artifacts/threefield/figures/fig2_wake.png` | race profiles at three stations with the Gaussian model overlay, the resolved axial rms (turbulence intensity), and the width growth vs the empirical spreading law |
| `artifacts/threefield/figures/fig3_fragments_thermal.png` | fragment size CDF with the Weibull fit, the melting history (numeric vs closed form), and the mass-ledger bars |
| `artifacts/threefield/figures/fig4_coupling_summary.png` | per-gate status with the gate numbers and the downstream-impact metrics |

## 5. Notes, pitfalls and scope

1. **The DEM is a scaled toy material.** The repository's smoke-test
   micro-parameters give an effective lattice modulus of order 10⁴ Pa, three to
   five decades below sea ice. Absolute force transfer is therefore meaningless.
   Scaling the stiffness by the material ratio (×2·10⁵, with mass scaling to
   keep the time step) was tried and **rejected**: the lattice fails in a shock at
   first contact (contact pressure 3.6·10⁸ Pa, crushing-phase force 3.4·10⁷ N/m),
   i.e. the naive scaling is worse than the soft lattice, not better.
2. **Thermal melt is slow at fragment scale.** With u′ ≈ 0.06 m/s relative
   velocity, Ranz–Marshall gives Nu ≈ 12, h ≈ 130 W/m²K and dD/dt ≈ −1.4·10⁻⁶ m/s:
   melting 0.3 m of ice needs hours, not the 30 s residence. Any claim that a
   propeller race melts appreciable ice on a channel transit is unsupported by
   this model.
3. **Turbulent bending cannot break small debris.** L<sub>crit</sub> = 13.9 m for
   0.1 m thick ice at u′ = 0.75 m/s: 0.1–0.3 m fragments are mechanically stable
   in the race. Fragmentation in reality comes from blade/hull impacts, which
   this coupling does not model. Fragmentation of the DEM debris is therefore
   inert by physics, not by a coding error.
4. **The LBM race is under-resolved** (10 cells per diameter): the functional
   form is verified, the spreading law is not. Grid refinement (≥ 32 cells per
   diameter) is required before the width law can be used predictively.
5. **Formula provenance.** The baselines follow the published component
   structure of Lindqvist (1989) and Riska (1989) but the coefficients could not
   be re-verified offline (see `docs/threefield/README.md`); they are used as a
   band, never as a single authority.
6. **Census statistics.** 196 components of a 324-particle lattice are dominated
   by singletons; no size law can be validated on such a census.

## 6. Error assessment (explicit)

| gate | metric | value | threshold | < 3 %? |
|---|---|---|---|---|
| (a) resistance vs formulas, 3 thicknesses | worst raw error vs Riska | 99.57 % | < 15 % | **no** — FAIL |
| (a) after single-point calibration at h = 0.45 m | worst error (Riska) | 282.90 % | < 15 % | **no** — FAIL |
| (b) race profile vs empirical Gaussian | worst core shape error | 5.80 % | < 10 % | not all stations (2.68 %, 2.41 %, 5.80 %) — PASS at the gate, **>3 % at the third station** |
| (b) predictive (literature β, momentum amplitude) | worst error | 1033 % | — | **no** (documented, under-resolution) |
| (c) fragment size vs experimental Weibull | R² | 0.030 (fit), −2.24 (vs reference) | R² > 0.9 | **no** — FAIL |
| (d) melting vs Ranz–Marshall closed form | melt-fraction error | 1.7·10⁻¹⁰ % | < 10 % | **yes** |
| (e) mass conservation | relative ledger error | 4.4·10⁻¹⁵ | < 1e−6 | **yes** |
| (f) pytest coverage | 25 tests | all pass | ≥ 18 | **yes** |

**Verdict: FAIL** (criteria a and c). The failures are attributable, quantified
and reproducible:

* (a) fails because the DEM material is ~10³–10⁵× softer than sea ice and the
  lattice is too small to reach a steady crushing regime; a meaningful
  DEM-versus-formula comparison needs a micro-parameter calibration against the
  repository's UCS / three-point-bending specimen benchmarks (mass scaling to
  keep the time step) and ≥10⁴ particles.
* (c) fails because a 324-particle bonded lattice produces a debris census of
  single particles, which is a discretisation artefact rather than a floe-size
  distribution. A continuum population-balance model (or a ≥10⁵-particle ice
  field) is the documented next step; the cascade machinery is shown to produce
  R² = 0.98 on a physically sized population in the outlook section, which is
  *not* offered as evidence for the gate.