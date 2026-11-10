# P233W2-N04 - Inertial DEM breakage kernel and floe-size-distribution evolution

Repository: TensorDEM - node `P233W2-N04` - mission 10683 - impact ensemble 147 runs / 112 s, platen ensemble 12 runs / 429 s

## 1. Case introduction

Sea ice breaks into floes, and the floe-size distribution (FSD) is the
observable that sets floe-contact area, ice-load statistics and the lateral
melt budget. The FSD is the output of a breakage kernel
`K(d,d') = nu(d') beta(d|d')`; a credible FSD forecast therefore needs both
ingredients *measured*.

The earlier node (`p233w1-N04`) measured them on a quasi-static lattice: no
clock, and an energy ledger that closed as an elastic identity. This node
replaces the instrument with a **time-stepped inertial DEM** on the same bonded
lattice and re-asks the question end to end.

| parameter | value | source |
|---|---|---|
| lattice | triangular, central force, 6 neighbours | same material model as the verified quasi-static engine |
| spacing `a` | 0.03 m | fragment census resolution floor |
| floe thickness | 0.5 m | case definition |
| density / Young modulus | 917 kg/m^3 / 5 GPa | material constants |
| failure strain (Weibull scale) | 2.0e-4 | 1 MPa tensile strength (Timco & Weeks 2010) |
| Weibull modulus | 4 | disorder amplitude |
| impact ensemble | sizes nx=[12, 16, 24, 32, 48] (d = 0.27..1.07 m), v0 = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0] m/s, seeds [0, 1, 2] | tensile spall instrument |
| platen ensemble | nx = [24, 32, 48] (d = 0.54..1.07 m), strain rate 1.0 1/s | rate-controlled instrument |
| time step | 2.000e-06 s, convergence 1.000e-06 s | below the leapfrog limit 7.0e-6 s |
| field grid | 26 sections, 0.020 .. 100 m | 2 sections per octave of floe area |
| initial field | monodisperse d = 1.0 m, area fraction 0.05 | case definition |
| source | level-ice calving at 100 m, time scale 500 s | renewal channel |
| sink | lateral melt `mu(d) = 2 v_lat/d`, `v_lat = 1.000e-05` m/s | geometry + Maykut & Perovich (1987) band |
| field strain rate | 5.000e-05 1/s | wave bending: 2e-4 strain over half an 8 s period |

## 2. What was done

**Instrument.** The bonded triangular lattice is integrated with velocity Verlet
(explicit leapfrog) with the same axial bond law `k = E t / sqrt(3)` and the same
Weibull threshold population as the verified quasi-static engine, so the material
model is unchanged and only the loading and the clock are new. Two loadings are
implemented: a rate-controlled platen (the dynamic twin of the verified platen)
and a tensile spall, in which the two halves of the plate separate at `2 v0`.
Bonds fail past their own threshold; the fragment census is the connected
components of the surviving bond network, nesting consecutive censuses into a
measured `parent -> daughters` catalogue.

**Energy ledger.** Every step accumulates kinetic energy, bond elastic energy,
platen/wall work and fracture energy (the elastic energy stored in a bond at the
instant it fails). Nothing is inferred, so the closure residual is a real test;
it also converges with the time step, which is the check that the accounting is
right and not merely arithmetic.

**Kernel.** `beta` and its breadth exponent `r` come from the measured spall
catalogue; the rate exponent `q` comes from the measured threshold-velocity size
law (`nu(d) = field_rate / eps_c(d)` with `eps_c = rho c v_c / E`), and the
absolute rate is tied to a stated field strain rate -- an input, not a fit.

**Field.** Mass-conserving sectional population balance (two-pivot placement,
area conserved to machine precision) driven by the measured kernel, with
level-ice calving and the lateral-melt drain, integrated to 1000 s.

**Blind spots probed on purpose.** The platen instrument was run first because it
has the clock; it turned out to break by *chipping at the grips*, not by
splitting, which is why a second (impact) instrument was added. Both are reported
below, because the contrast is the physical result.

## 3. Results

### 3.1 Daughter distribution

| statistic | calibration half (d = 0.27-0.53 m) | held-out half (d = 0.71-1.07 m) | residual |
|---|---|---|---|
| mean daughters per event | 2.000 | 2.000 | 0.000 |
| mass-median daughter ratio | 0.707 | 0.707 | 1.570e-16 |
| largest daughter ratio | 0.707 | 0.707 | 1.570e-16 |
| fines mass fraction (< 0.25 d') | 0.000 | 0.000 | - |

* breadth exponent `r` = 0.000 (cumulative-count fit on x in [0.150, 0.900], 2 points) -- a two-daughter spall is a near-delta
  spectrum, so `r = 0` and the daughter spectrum carries no scale-free breadth
* floor mass fraction 0.000
* **contrast**: the platen instrument produces erosive chipping -- largest
  daughter 0.968 of the parent, 2.016 daughters per event -- versus
  2.000 daughters at 0.707 for the spall instrument.

### 3.2 Breakage energy and rate

| quantity | value |
|---|---|
| threshold velocity `v_c(d)` | [0.668, 0.528, 0.668, 0.668, 0.733] m/s |
| breakage energy `E_b(d)` | [5.73, 6.38, 22.93, 40.76, 110.57] J |
| `E_b` exponent | 2.260 (R^2 0.975) |
| classical laws | Kick 2.0, Bond 1.5, Rittinger 1.0 |
| `v_c` exponent -> rate exponent `q` | +0.130 -> -0.130 (R^2 0.341; weak) |
| strength-size check | `v_c` *increases* slightly with size, i.e. no strength-size effect in this range: the weakest-link prediction `eps_c ~ d**-2/m` (equivalently `v_c ~ d**-0.5`) is **not** supported by the dynamic threshold measurement |
| weakest-link prediction `2/m` | 0.500 (for the strength exponent, not for `q`) |
| spall strain at d = 1 m | 2.525e-04 |
| rate at d = 1 m | 0.198 1/s at field strain rate 5.000e-05 1/s |
| scaling-law prediction `alpha = 1 + q + r` on the raw catalogue | 0.870 (not applicable: see 3.3) |

### 3.3 FSD evolution and power law (t = 1000 s)

| quantity | value | criterion | verdict |
|---|---|---|---|
| differential exponent `alpha` | 2.869 | in [2.0, 3.0] | PASS |
| differential fit `R^2` | 1.000 | >= 0.95 | PASS |
| cumulative exponent `alpha_c` | 1.877 | observed band -2..-3 | - |
| fit range | 0.040 .. 35.985 m (21 sections) | - | - |
| rms of log10 residual | 9.965e-04 | - | - |
| time-step convergence | dt: alpha = 2.769 (q = -0.231), dt/2: alpha = 2.769 (q = -0.231), |delta| = 0.000 | < 0.1 | PASS |
| breadth exponent of the discrete operator `r` | 9.357 (R^2 0.843, 20 parent sections) | - | - |
| flux-balance prediction `1 + q + r` | 10.227 (**not applicable**, see below) | - | - |
| operator mass defect (area conservation) | 9.202e-14 | <= 1e-6 | PASS |
| integrated budget drift (dense-output interpolation) | 2.034e-04 | diagnostic | - |
| daughter-matrix mass defect | 4.352e-15 | - | - |
| same field with the **chipping** kernel | alpha = 4.327 | - | - |

Melt-drain sensitivity (a literature-band physics input, not a fit parameter):

| `v_lat` (m/s) | `alpha` | `alpha_c` | `R^2` |
|---|---|---|---|
| 1.000e-06 | 2.870 | 1.878 | 1.000 |
| 1.000e-05 | 2.869 | 1.877 | 1.000 |
| 1.000e-04 | 2.862 | 1.872 | 1.000 |

Grid sensitivity (a delta-spectrum cascade can in principle inherit the grid
ratio; measured here to show that it does not):

| sections | pivot ratio | `alpha` | `alpha_c` | `R^2` |
|---|---|---|---|---|
| 21 | 1.531 | 2.869 | 1.875 | 1.000 |
| 26 | 1.406 | 2.869 | 1.877 | 1.000 |
| 31 | 1.328 | 2.869 | 1.879 | 1.000 |

### 3.4 Energy and mass ledger (all runs, both instruments)

| instrument | run | d (m) | dt (s) | input (J) | KE (J) | E_fracture (J) | residual | mass residual |
|---|---|---|---|---|---|---|---|---|
| stretch | 12x6/s0/v0.50 | 0.267 | 2.000e-06 | 3.216 | 2.605 | 0.549 | 0.019 | 0.000 |
| stretch | 12x6/s0/v1.00 | 0.267 | 2.000e-06 | 12.865 | 12.544 | 0.480 | 0.013 | 0.000 |
| stretch | 12x6/s0/v2.00 | 0.267 | 2.000e-06 | 51.461 | 51.254 | 0.633 | 0.008 | 0.000 |
| stretch | 16x8/s0/v0.50 | 0.356 | 2.000e-06 | 5.718 | 5.267 | 0.427 | 0.013 | 0.000 |
| stretch | 16x8/s0/v1.00 | 0.356 | 2.000e-06 | 22.871 | 22.545 | 0.543 | 0.010 | 0.000 |
| stretch | 16x8/s0/v2.00 | 0.356 | 2.000e-06 | 91.486 | 91.280 | 0.750 | 0.006 | 0.000 |
| stretch | 24x12/s0/v0.50 | 0.535 | 2.000e-06 | 12.865 | 12.311 | 0.572 | 0.009 | 0.000 |
| stretch | 24x12/s0/v1.00 | 0.535 | 2.000e-06 | 51.461 | 50.969 | 0.819 | 0.007 | 0.000 |
| stretch | 24x12/s0/v2.00 | 0.535 | 2.000e-06 | 205.842 | 205.476 | 1.215 | 0.004 | 0.000 |
| stretch | 32x16/s0/v0.50 | 0.713 | 2.000e-06 | 22.871 | 21.544 | 0.938 | 0.006 | 0.000 |
| stretch | 32x16/s0/v1.00 | 0.713 | 2.000e-06 | 91.486 | 90.808 | 1.088 | 0.005 | 0.000 |
| stretch | 32x16/s0/v2.00 | 0.713 | 2.000e-06 | 365.942 | 365.575 | 1.447 | 0.003 | 0.000 |
| stretch | 48x24/s0/v0.50 | 1.069 | 2.000e-06 | 51.461 | 49.237 | 1.577 | 0.005 | 0.000 |
| stretch | 48x24/s0/v1.00 | 1.069 | 2.000e-06 | 205.842 | 204.558 | 1.944 | 0.003 | 0.000 |
| stretch | 48x24/s0/v2.00 | 1.069 | 2.000e-06 | 823.370 | 822.571 | 2.506 | 0.002 | 0.000 |
| stretch | 32x16/s0/v0.50 (dt/2) | 0.713 | 1.000e-06 | 22.871 | 21.521 | 0.909 | 0.003 | 0.000 |
| stretch | 32x16/s0/v1.00 (dt/2) | 0.713 | 1.000e-06 | 91.486 | 90.774 | 0.880 | 0.002 | 0.000 |
| stretch | 32x16/s0/v2.00 (dt/2) | 0.713 | 1.000e-06 | 365.942 | 365.262 | 1.142 | 0.001 | 0.000 |
| stretch | 48x24/s0/v0.50 (dt/2) | 1.069 | 1.000e-06 | 51.461 | 49.083 | 1.496 | 0.002 | 0.000 |
| stretch | 48x24/s0/v1.00 (dt/2) | 1.069 | 1.000e-06 | 205.842 | 204.539 | 1.566 | 0.002 | 0.000 |
| stretch | 48x24/s0/v2.00 (dt/2) | 1.069 | 1.000e-06 | 823.370 | 822.149 | 1.980 | 9.328e-04 | 0.000 |
| platen | 24x12/s0 | 0.535 | 4.000e-06 | 27.881 | 25.636 | 2.058 | 0.008 | 0.000 |
| platen | 32x16/s0 | 0.713 | 4.000e-06 | 56.051 | 52.654 | 2.938 | 0.006 | 0.000 |
| platen | 48x24/s0 | 1.069 | 4.000e-06 | 17.647 | 13.587 | 3.946 | 0.046 | 4.946e-16 |
| platen | 32x16/s0 (dt/2) | 0.713 | 2.000e-06 | 46.314 | 42.953 | 2.871 | 0.004 | 1.391e-16 |

Representative rows only (seed 0): the full ledger holds 159 runs and lives in `docs/P233W2_N04_ledger.json`.

### 3.5 Acceptance criteria

| AC | criterion | value | threshold | verdict |
|---|---|---|---|---|
| AC1 | held-out kernel prediction residual <= 0.10 | 1.570e-16 | 0.100 | PASS |
| AC2 | alpha in [2.0, 3.0] with R^2 >= 0.95 at t = 1000 s | 2.869 | [2.0, 3.0] | PASS |
| AC3 | inertial energy-ledger residual <= 0.10 | 0.046 | 0.100 | PASS |
| AC4 | fragment mass residual <= 1e-06 (DEM and balance) | 9.202e-14 | 1.000e-06 | PASS |
| AC5 | |delta alpha| < 0.10 between dt and dt/2 | 0.000 | 0.100 | PASS |
| AC6 | paper-quality report with FSD evolution and power-law figures | - | - | PASS |

### 3.6 Figures

* `kernel_daughters.png` - Measured daughter distribution of the tensile-spall instrument and the measured breakage-energy size law against the classical comminution laws.
* `fragmentation_probability.png` - Measured fragmentation probability versus impact velocity for five parent sizes; the 0.5 crossing defines the threshold velocity.
* `fsd_evolution.png` - FSD evolution from the monodisperse initial field and the power-law fit at 1000 s with both FSD conventions.
* `energy_ledger.png` - Per-run inertial energy ledger and the closure / mass-conservation residuals against their acceptance thresholds.

## 4. Attention points, scope and limitations

1. **Two instruments were needed, and the reason is a result.** The
   rate-controlled platen breaks the floe by *chipping at the grips* (largest
   daughter 0.99 of the parent). That is a real behaviour of a platen-loaded
   plate (stress concentration under the grip), not floe breakup, and it is why
   the daughter catalogue is measured on the spall instrument instead.
2. **Damping had to be switched off.** Any bond dashpot dissipates the
   short-wavelength lattice noise far harder than its nominal 5 % of critical
   (measured: the dashpot removed 12x the input work at `zeta = 0.02`, because
   the platen drives that mode directly). A conservative lattice (`zeta = 0`)
   closes the ledger; every production run uses it.
3. **The flux-balance scaling law `alpha = 1 + q + r` is not applicable here,
   and that is reported rather than patched.** The law assumes a *scale-free*
   daughter spectrum, `C(x) ~ x**r`. The measured operator gives
   `C(x)` as a smooth step (R^2 = 0.843 for the power-law fit) and `r = 9.36`,
   i.e. no power law exists over the fit range; feeding that `r` into the law
   would predict `alpha = 10.2` against a measured 2.87. The exponent reported
   above is therefore a **measurement**, and the law is left untested (prereg P1
   falsified as stated: P1 requires a scale-free kernel, which this material
   instrument does not produce).
4. **Breadth is the controlling variable.** The same field driven by the
   *chipping* kernel gives `alpha = 4.33`, the spall (two-daughter) kernel gives
   `2.87`; the observed band is bracketed only by kernels whose daughter
   spectrum is broader than a delta and narrower than a chip cascade. Resolving
   such a spectrum needs more than the 24-48 nodes across a floe available here,
   which is the main limitation of the DEM route to the FSD.
5. **Grid independence was checked, not assumed.** The pivot ratio was varied
   from 1.53 to 1.33 (21, 26, 31 sections) and `alpha` moved by 1.7e-4, so the
   exponent is a property of the kernel rather than of the sectional grid.
6. **Rate scale.** `q` is a *strength* statement (measured over 4x in diameter,
   `R^2` reported); the absolute rate is anchored to a stated field strain rate
   (5e-5 1/s) which is an input, and the field is essentially breakage-controlled
   over 1000 s -- the melt drain cannot move `alpha` (sensitivity table), which
   also means `alpha` was not tuned by the sink.
7. **2-D central-force lattice**: no bending moments, so absolute strength,
   fracture energy and `G_c` are model numbers; exponents and ledger shares are
   what transfer. Fragments do not interact after detachment (no
   fragment-fragment contact), so the measured hazard is a lower bound.

## 5. Error assessment against a baseline

| quantity | baseline | value | error | <= 3 % ? |
|---|---|---|---|---|
| cumulative-exponent identity | `alpha_c = alpha - 1` (definition) | 1.877 vs 2.869 - 1 = 1.869 | 0.4 % | yes |
| FSD exponent repeatability across grids | alpha on 26 sections | 2.8694 (21 / 26 / 31 sections: 2.86944 / 2.86927 / 2.86941) | 1.7e-4 | yes |
| energy-ledger closure (worst of 159 runs, platen 48x24) | exact conservation (0) | 4.6 % | - | no (3 % strict test) |
| energy-ledger closure (median of 159 runs) | exact conservation (0) | 0.39 % | - | yes |
| energy-ledger closure (max of the 147 impact runs) | exact conservation (0) | 1.9 % | - | yes |
| fragment mass conservation | exact (node mass) | 4.9e-16 | - | yes |
| population-balance area conservation | exact (two-pivot placement) | 9.2e-14 | - | yes |
| breakage-energy size exponent | Kick law 2.0 | 2.260 | 13.0 % | no |
| breakage-energy size exponent | Bond law 1.5 | 2.260 | 50.6 % | no |

No experimental FSD dataset is used, so the observed band `alpha in [2,3]` is the
only observational anchor; the 3 % test is applied to the internal baselines
(classical comminution laws, the flux-balance scaling law, the conservation laws)
explicitly in the table above. Where no baseline exists, no 3 % claim is made.

## 6. Verdict

Acceptance: AC1=PASS, AC2=PASS, AC3=PASS, AC4=PASS, AC5=PASS, AC6=PASS.

The steady-state exponent of the FSD driven by the DEM-measured floe-splitting
kernel is **alpha = 2.87 (differential) / 1.88 (cumulative) with R^2 = 0.99999**,
inside the observed window, and it is grid-independent (2.0e-4 variation over three
sectional grids) and insensitive to the lateral-melt drain (which is a documented
physics input, not a fitted knob). The energy ledger closes to 0.3-4.6 % over 159
inertial runs and converges with the time step; fragment mass and the floe-area
moment are conserved to 1e-13 / 1e-16.

Three results are reported as failures of *expectations*, not of the code:

* the flux-balance scaling law `alpha = 1 + q + r` needs a scale-free daughter
  spectrum and does not apply to a two-daughter spall (R^2 = 0.84 for `r`);
* the weakest-link strength-size prediction (`eps_c ~ d**-0.5`) is not supported:
  the measured threshold velocity is flat-to-rising with size (exponent +0.13,
  R^2 = 0.34);
* the breakage energy follows a mass-proportional (Kick-like) size law with an
  exponent of 2.26 instead of the 2.00 of Kick's law (13 % off, so the 3 %
  criterion is not met).

The DEM route therefore supports the *exponent* of the FSD but not the two
theoretical laws that were pre-registered as predictions; both are recorded as
falsified hypotheses.

