# Quasi-static limit of the DEM3D direct-shear block, and why the affine shear constant is not recovered

Generated from `results-dem3d-shear-quasistatic/summary.json` by
`scripts/benchmark_dem3d_shear_quasistatic.py` (protocol `tensordem-dem3d-shear-quasistatic-v2`); all
tables below are injected from that record, none transcribed by hand.

## 1. Case: the item that was withheld

`docs/dem3d_shear_specimen.md` calibrated the mode-II criterion of
`tensordem.dem3d` to a macroscopic strain but reported the driven-specimen
apparent shear modulus as **WITHHELD**:

> *"The response is not yet rate converged: halving the ramp speed still changes
> the traction by 20 %, and the apparent modulus by 31 % ... Reaching the
> quasi-static limit needs roughly one decade slower ramps."*

`docs/dem3d_material_calibration_roadmap.md` lists the same item as *"Still open:
quasi-static apparent modulus of a driven block"*. This page measures that limit,
settles the rate question, and reports an honest negative result for the
geometry question.

| item | value |
| --- | --- |
| lattice | simple-cubic, 26 neighbours, central linear springs |
| reference specimen | 9 x 5 x 5 = 225 particles |
| bond stiffness `k` | 2000 N/m |
| radius / spacing `a` | 0.025 m / 0.05 m |
| particle mass | 0.060018 kg (rho = 917 kg/m3) |
| affine shear constant `C44 = 7k/(3a)` | 93333.33 Pa |
| reference height `h = (nz-1)a` | 0.2000 m |
| nominal area `A = nx a . ny a` | 0.1125 m2 |
| shear-wave speed `c = a sqrt(k/m)` | 9.1274 m/s (transit 0.02191 s) |
| viscous diffusion time `drag h^2/(k a^2)` | 0.08000 s at drag = 10 N s/m |
| tensile threshold | parked at 1000 (mode isolation) |
| shear threshold | 0.03 |
| ramp speeds | 0.4 -> 0.00625 m/s (7 points, 64x span) |
| widths / heights | nx = 5, 9, 13, 17 / nz = 5, 7, 9 |
| drag series | 2, 5, 10, 20 N s/m |
| fit window | gamma in [0.0200, 0.0240] at every rate |
| time step (solver guard) | 3.2737e-05 s |

## 2. Method

1. **Two relaxation clocks in closed form.** A driven block is quasi-static only
   if the ramp outlasts the shear-wave transit `t_w = h/c` and the viscous
   diffusion time `t_d = drag h^2/(k a^2)`. With `t_ramp = gamma_max h / v` the
   criterion used throughout is `t_ramp/t_w >= 20` **and** `t_ramp/t_d >= 5`.
   The earlier series (v = 0.4 ... 0.1 m/s) satisfied neither.
2. **Fixed-strain fit window.** `shear_specimen.run_direct_shear` is driven at each
   speed; the elastic slope of `tau(gamma)` is fitted on the *same*
   engineering-strain window at every rate, so a rate comparison is not
   contaminated by moving the window. Two further windows are fitted on the same
   history and their spread is reported as the residual transient.
3. **Zero-rate extrapolation** `G_app(v) = G0 + c v` on the three slowest rates,
   repeated with two points to expose stability.
4. **Width series** at a converged rate, extrapolated linearly in `a/Lx`.
5. **Height series** with the ramp speed *chosen per height* by
   `quasi_static_velocity` (a taller block has a longer `t_d`, so one speed cannot
   serve all heights), plus a cross run at `nz = 9, nx = 5` to test whether the
   width slope is height independent.
6. **Drag series** at a converged rate to test whether the fast-ramp bias is a
   viscous stress bias.

Toolchain: `tensordem.shear_quasistatic` (new) on top of
`tensordem.shear_specimen` and `tensordem.dem3d`; `torch` float64,
`torch.set_num_threads(1)`, cases in parallel processes, matplotlib figures,
JSON + CSV records. **The solver is not modified.**

```bash
python scripts/benchmark_dem3d_shear_quasistatic.py --jobs 10
python -m pytest tests/test_dem3d_shear_quasistatic.py -q
```

## 3. Results

### 3.1 Rate convergence: the WITHHELD number, resolved

| v [m/s] | steps | transits | diffusion times | $G_{app}$ [Pa] | $G_{app}/C_{44}$ | window spread | quasi-static |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 382 | 0.57 | 0.16 | 29096 | 0.3117 | 9.15e-01 | no |
| 0.2 | 764 | 1.14 | 0.31 | 83664 | 0.8964 | 5.00e-01 | no |
| 0.1 | 1527 | 2.28 | 0.63 | 60245 | 0.6455 | 2.20e-01 | no |
| 0.05 | 3055 | 4.56 | 1.25 | 62056 | 0.6649 | 1.16e-02 | no |
| 0.025 | 6109 | 9.13 | 2.50 | 62071 | 0.6650 | 3.28e-04 | no |
| 0.0125 | 12219 | 18.25 | 5.00 | 62071 | 0.6650 | 2.17e-04 | no |
| 0.00625 | 24437 | 36.51 | 10.00 | 62071 | 0.6650 | 2.18e-04 | yes |

The apparent modulus falls from 0.3117 to 0.6650 `C44`
(-35 % of `C44`) as the ramp slows by 64x, then stops moving.
The zero-rate limit is

```
G0 = 62071.1 Pa = 0.6650 C44
fit residual 4.5e-07 (three slowest rates)
two-point stability 1.0e-06
slowest measured rate inside 2.0e-06 of G0
```

The window spread collapses from 9.1e-01 to 2.2e-04: at
v = 0.4 m/s the fit window is dominated by the load-path transient (the run even
breaks 2 bonds through the shear criterion), at the slow end the three strain
windows agree. **This closes the "not rate converged" item: the driven-specimen
apparent modulus of the 9x5x5 block is 62071 Pa, rate independent to
2.0e-06 once `t_ramp/t_w >= 18`.**

### 3.2 Width series at the converged rate (`nz = 5`)

| nx | h [m] | $a/L_x$ | $G_{app}$ [Pa] | $G_{app}/C_{44}$ | window spread |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.200 | 4.000 | 45499 | 0.4875 | 5.87e-04 |
| 9 | 0.200 | 2.222 | 62071 | 0.6650 | 3.28e-04 |
| 13 | 0.200 | 1.538 | 68413 | 0.7330 | 2.07e-04 |
| 17 | 0.200 | 1.176 | 71770 | 0.7690 | 1.36e-04 |

The width dependence is linear in `a/Lx` to 2.5e-04
(max residual over four widths), giving

```
G(a/Lx -> 0, nz = 5) = 82730 Pa = 0.8864 C44
```

i.e. 11.4 % below the affine constant. The lateral free surfaces explain
only one third of the gap between the raw block (0.6650 `C44`) and `C44`.

### 3.3 Height series: an honest negative result

| nx | nz | v [m/s] | steps | $a/h$ | $G_{app}$ [Pa] | $G_{app}/C_{44}$ | transits / diffusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 5 | 0.01141 | 13143 | 0.250 | 62071 | 0.6650 | 20.0 / 11.0 |
| 9 | 7 | 0.01141 | 19715 | 0.167 | 53353 | 0.5716 | 20.0 / 7.3 |
| 9 | 9 | 0.01141 | 26287 | 0.125 | 44623 | 0.4781 | 20.0 / 5.5 |
| 5 | 9 | 0.01141 | 26287 | 0.125 | 22320 | 0.2391 | 20.0 / 5.5 |

The ramp is quasi-static by the same criterion at every height (transits 20-36,
diffusion times 5-11) and the `nx = 9, nz = 5` height run reproduces the width run
to 4.5e-06, so the auto-selected rates are trustworthy. The modulus then
**decreases** monotonically with height: 0.6650 -> 0.5716 -> 0.4781 `C44` for
`nz = 5, 7, 9`, and the `nz = 9, nx = 5` cross run sits at 0.2391.

This is physics, not a convergence failure: a driven block with free lateral faces
localises the imposed displacement near the platen, so a taller block stores the
same `Delta x` in a thinner sheared layer and `tau/(Delta x/h)` falls. Consequently
the affine constant is **not** recovered by extrapolating the driven block: the
two-stage extrapolation `G/C44 = c0 - alpha a/Lx - beta a/h` gives

```
c0 = 0.6672 C44      alpha = -0.1171      beta = 0.8767
combined-model residual over all finite-size points: 34.2% of C44
```

The intercept is 33.3 % from `C44` and the
7-point additive model does not fit the height trend either
(residual 34.2%). **Verdict: the driven-block
direct-shear test cannot be used to validate the lattice shear constant.**
`C44` is verified by the affine probes of `docs/dem3d_shear_specimen.md`, not here.

### 3.4 The rate bias is not viscous

| drag [N s/m] | $G_{app}$ [Pa] | $G_{app}/C_{44}$ | window spread | intercept [Pa] |
| --- | --- | --- | --- | --- |
| 2 | 62760 | 0.6724 | 2.10e-02 | -26.5 |
| 5 | 62100 | 0.6654 | 2.62e-03 | -29.5 |
| 10 | 62071 | 0.6650 | 3.28e-04 | -57.6 |
| 20 | 62071 | 0.6650 | 2.16e-04 | -115.2 |

At drag = 10 and 20 N s/m (both converged) the fitted modulus differs by
1.2e-04 %, so the fast-ramp bias is not a stress offset `eta gamma_dot`
from the per-particle drag: it is the load-path transient entering the fit
window, which is exactly what the window-spread column of 3.1 measures. Lowering
the drag to 2-5 N s/m moves the converged modulus by 0.7 % but worsens
the transient, which is why the reference protocol keeps drag = 10 N s/m.

### 3.5 Machine checks

| check | value | criterion | verdict |
| --- | --- | --- | --- |
| ramp speeds span two decades (0.4 -> 0.00625 m/s) | 6.400e+01 | 5.0e+01 | **pass** |
| fastest ramp is demonstrably not quasi-static | 5.705e-01 | 2.0e+01 | **pass** |
| slowest ramp satisfies the quasi-static criterion | 3.651e+01 | 2.0e+01 | **pass** |
| zero-rate extrapolation residual | 4.472e-07 | 5.0e-03 | **pass** |
| zero-rate extrapolation stability | 1.043e-06 | 1.0e-02 | **pass** |
| slowest measured rate is inside 2 % of G0 | 2.002e-06 | 2.0e-02 | **pass** |
| fit window does not bias the converged slope | 2.176e-04 | 2.0e-02 | **pass** |
| converged modulus is drag insensitive (drag >= 10) | 1.189e-06 | 1.0e-03 | **pass** |
| height series reproduces the width series at nx = 9, nz = 5 | 4.477e-06 | 2.0e-02 | **pass** |
| width extrapolation is linear in a/Lx | 2.457e-04 | 5.0e-02 | **pass** |
| two-way additive model residual (diagnostic) | 3.419e-01 | 2.0e-02 | **fail** |
| infinite-height extrapolation is within 3 % of C44 | 3.328e-01 | 3.0e-02 | **fail** |

Two checks are recorded as **fail** on purpose: the additive two-way model does
not describe the height trend, and the extrapolated modulus is not within 3 % of
`C44`. They fail for the physical reason given in 3.3 and are reported as an
honest negative result, not as withheld data.

## 4. Figures

`rate_convergence.png` (archived in `docs/dem3d-shear-quasistatic/`): (a)
`tau(gamma)` for the seven ramp speeds with the common fit window shaded; (b)
`G_app/C44` versus `v` on a log axis with the zero-rate extrapolation and the
affine line.

`finite_size.png`: (a) converged `G_app/C44` versus `a/Lx` at `nz = 5` and
`nz = 9` with the joint-fit lines and the extrapolated intercept; (b) converged
`G_app/C44` versus `a/h` at `nx = 9`, showing the decrease with height.

## 5. Notes, pitfalls, scope

* **The previous negative result was correct.** Its three rates sat at 0.6-2.3
  wave transits; a convergence claim cannot be made from inside the transient,
  and the fix is a slower ramp with a fixed *strain* window, not a better fit.
* **The tensile threshold must be parked** (here `1000`). With the
  shipped 0.015 the same ramp trips the *axial* criterion near `gamma = 0.019`
  because the transient concentrates axial strain at the platen corners.
* **A single ramp speed cannot serve a height series**: `t_d` grows with `h^2`.
  `quasi_static_velocity` picks the cheapest admissible rate per geometry.
* **Area convention**: `A = nx a . ny a` (population). With the footprint
  convention every modulus here would be scaled by
  1.406.
* **Scope**: the numbers are for a 225-3000-particle lattice block with
  `k = 2000 N/m`. No ice-material, fracture-energy or ship-scale claim is made.
  Mode-II fracture energy remains NOT_SUPPORTED (no tangential bond stiffness).
* Cost: 182731 solver steps, 434 s wall on 10 processes.

## 6. Error judgement against baselines

| quantity | value | baseline | relative error | < 3 % ? |
| --- | --- | --- | --- | --- |
| `G0` (zero-rate limit, nx=9, nz=5) | 62071.1 Pa | `G0 + c v` extrapolation | residual 4.5e-07 | **yes** |
| slowest measured rate vs `G0` | 62071.0 Pa | 62071.1 Pa | 2.0e-04 % | **yes** |
| `G` at infinite width (`nz = 5`) | 82730 Pa | `C44` = 93333 Pa | 11.4 % | **no** — height effect remains |
| `G` at infinite size (two-stage) | 62274 Pa | `C44` | 33.3 % | **no** — the driven-block BVP does not approach homogeneous shear |
| raw reference block vs `C44` | 62071 Pa | `C44` | 33 % | **no** — reported as a boundary-value statement |
| drag insensitivity (converged) | 1.2e-04 % | 0 | drag 10 vs 20 N s/m | **yes** |
| mode-II fracture energy | not defined | — | — | **no baseline exists**; NOT_SUPPORTED |

## 7. What this closes and what it does not

Closed:

* the WITHHELD *rate* question: the driven-specimen modulus has a measured
  zero-rate limit, `62071 Pa = 0.6650 C44` for the 9x5x5 block, stable to
  2.0e-06 and reproducible across the drag series;
* the roadmap item *"quasi-static apparent modulus of a driven block"* — answered
  for this protocol, with the answer that the driven block is a boundary-value
  measurement and not a modulus calibration;
* reusable machinery: `quasi_static_margins` / `quasi_static_velocity` (is a
  manifest quasi-static?), `fit_zero_rate_limit`, `fit_infinite_width`,
  `fit_two_way_finite_size`.

Not closed: the lattice shear constant from a *driven* test (needs the affine
probe, which already passes at machine precision), mode-II fracture energy,
higher-order finite-size scaling, and any ice-material claim.
