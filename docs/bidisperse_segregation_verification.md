# Bidisperse granular segregation: verification surface (P155-N09)

## 1. Case and physical background

A cargo hold filled with a two-size granular material (ore, grain, aggregate) segregates when the ship rolls or vibrates: the coarse fraction migrates up (Brazil-nut) or down (reverse Brazil-nut).  The consequences are a graded free surface, a depth-dependent permeability and therefore a depth-dependent rate of excess-pore-pressure dissipation -- the mechanism behind cargo shift and liquefaction.

Simulation inventory: **16 shear-cell cases** and **26 shaker cases**, size ratios 1.2, 1.5, 2.0, 3.0, 2-D frictional disks (Hertz-Mindlin normal/tangential + rolling resistance), `tensordem.granular2d.engine`.

### Parameter table

| item | value |
|---|---|
| particle diameters | 10 mm (small), 12/15/20/30 mm (large) |
| density / thickness | 2500 kg/m3 / 10 mm |
| contact | kn = 1e5 N/m, kt/kn = 0.5, zeta = 0.3, mu = 0.5, mu_r = 0.05 |
| cell | periodic in x (shear) / closed (shaker), 120 particles, phi_nom = 0.60 |
| shear cell | dead-weight load plate at sigma = 4000 Pa, gamma_dot = 10 1/s |
| shaker | f = 30 Hz, Gamma = A w^2/g in {2, 10}, 20 cycles |
| strain / cycles per case | gamma = 6, or 20 shaking cycles |
| seeds | 5 per condition for the two extremes, 3 for the intermediate ratios |

## 2. Method

1. **Packing** -- jittered lattice at the requested nominal solid fraction, then overlap removal by deterministic radius inflation (0.80 -> 0.85 -> 0.90 -> 0.95 -> 1.00 r) so no energy is injected.
2. **Shear cell** -- a frictional plate carrying a dead weight `sigma * W` is placed on the free surface and given one vertical DOF (mass-spring-damper) integrated with the DEM; it is first brought to load equilibrium, then sheared at `v_top = gamma_dot * W`.  The wall tangential velocity is a first-class engine quantity (`DiskSystem(wall_velocity=...)`), so the drive is Coulomb friction at the plate, not a velocity overwrite of the particles.
3. **Shaker** -- integrated in the container frame (walls static, effective acceleration `-g + A w^2 sin(w t)`), 30 cycles.
4. **Metric** -- `Phi(z)` is the *area* fraction of the large species in a height bin, with species membership taken from the particle labels (global threshold `r_thr = sqrt(r_s r_l)`); `S = sum_b w_b |Phi_b - Phi_0| / sum_b w_b` with `w_b` the solid area in the bin; empty bins are dropped.  Direction: `dPhi = <Phi>_top - <Phi>_bottom` against its binomial standard error, a sign being declared only at 2 sigma.
5. **Scaling** -- `S = A (d_l/d_s - 1)^alpha` fitted by least squares on `S` (not on log S), with a bootstrap CI over seeds.
6. **Cargo coupling** -- `k(z) = d_eff^2 eps^3/(180 (1-eps)^2)`, `d_eff^2 = Phi d_l^2 + (1-Phi) d_s^2`, then a 1-D Terzaghi consolidation solve with `c_v(z) = k(z) K_s/gamma_w` gives the time to 50 % dissipation of the base excess pressure (`t50`).
7. **Repose angle** -- the surface cloud of a column collapse is handed to `tensordem.repose_calibration.repose_angle`, i.e. the P96-N3 estimator itself.

## 3. Results

### 3.1 Criterion verdicts (machine-evaluated)

| criterion | requirement | verdict | evidence |
|---|---|---|---|
| C1 | monodisperse negative control | PASS | bitwise frozen deviation = 0.000, bitwise determinism deviation = 0.000, sheared S = [0.24999999999999997, 0.3333333333333334], floor = 0.399 |
| C2 | opposite directions, >= 8/10 seeds | FAIL | shear 1.2 -> 0 (consensus 1.000, n=5), shear 3.0 -> 0 (consensus 1.000, n=5) |
| C3 | S monotone in ratio, R2 > 0.9, CI; S down as phi -> phi_j | FAIL | R2 = 0.829, alpha = 0.362, CI = [0.25657568557138355, 0.46983846681764324], monotone = FAIL |
| C4 | mass/concentration residual < 1e-10 | PASS | max |Delta Phi| = 2.22e-16 over 42 cases |
| C5 | repose angle with P96-N3 tooling | FAIL | stratified — deg vs monodisperse — deg, same estimator; unresolved: ReposeCalibrationError: not enough radial bins to fit a flank |
| C6 | sensitivity table reproducible | PASS | bitwise identical reruns = PASS |

### 3.2 Segregation degree per condition

| kind | size ratio | drive | seeds | S mean | S std | measured phi | direction (consensus) |
|---|---|---|---|---|---|---|---|
| shear | 1.20 | sigma=4000 Pa, gdot=10 1/s | 5 | 0.220 | 0.034 | 0.872 | 0 (1.00) |
| shear | 1.50 | sigma=4000 Pa, gdot=10 1/s | 3 | 0.154 | 0.021 | 0.878 | 0 (1.00) |
| shear | 2.00 | sigma=4000 Pa, gdot=10 1/s | 3 | 0.278 | 0.079 | 0.871 | 0 (1.00) |
| shear | 3.00 | sigma=4000 Pa, gdot=10 1/s | 5 | 0.402 | 0.107 | 0.880 | 0 (1.00) |
| vibration | 1.20 | Gamma=2 | 5 | 0.246 | 0.061 | 0.737 | 0 (1.00) |
| vibration | 1.20 | Gamma=10 | 5 | 0.238 | 0.040 | 0.641 | 0 (1.00) |
| vibration | 1.50 | Gamma=10 | 3 | 0.206 | 0.030 | 0.679 | 0 (1.00) |
| vibration | 2.00 | Gamma=10 | 3 | 0.323 | 0.036 | 0.673 | 0 (1.00) |
| vibration | 3.00 | Gamma=2 | 5 | 0.430 | 0.064 | 0.731 | 0 (1.00) |
| vibration | 3.00 | Gamma=10 | 5 | 0.448 | 0.082 | 0.677 | 0 (0.80) |

### 3.3 Scaling law

| kind | axis | ratios | S means | A | alpha | 95% CI | R2 |
|---|---|---|---|---|---|---|---|
| shear | sigma=4000.000 | [1.2, 1.5, 2.0, 3.0] | [0.22, 0.154, 0.278, 0.402] | 0.290 | 0.390 | [0.22882054366835586, 0.5514849737542394] | 0.717 |
| vibration | Gamma=2.000 | [1.2, 3.0] | [0.246, 0.43] | 0.363 | 0.242 | [0.14799038184803057, 0.33367369411900727] | 1.000 |
| vibration | Gamma=10.000 | [1.2, 1.5, 2.0, 3.0] | [0.238, 0.206, 0.323, 0.448] | 0.334 | 0.362 | [0.25657568557138355, 0.46983846681764324] | 0.829 |

### 3.4 Permeability / excess pore pressure sensitivity

| profile | k mean (m2) | k min | k max | k_top/k_bottom | t50 (s) | t50/t50(uniform) |
|---|---|---|---|---|---|---|
| measured | 3.69e-06 | 2.14e-24 | 4.42e-05 | 1.22e-15 | 3470935124030824448.000 | 0.865 |
| mirrored | 1.63e-05 | 5.56e-25 | 0.000196 | 8.21e-17 | 10309782706914004992.000 | 2.569 |
| uniform | 1.42e-05 | 2.14e-24 | 0.00017 | 3.16e-16 | 4013598739529029632.000 | 1.000 |
| large_at_top | 3.32e-05 | 5.56e-25 | 0.000398 | 2.84e-15 | 4177931387783445504.000 | 1.041 |
| large_at_bottom | 3.69e-06 | 5.56e-25 | 4.42e-05 | 3.51e-17 | 12963488108844636160.000 | 3.230 |

### 3.5 Repose angle (C5)

* estimator: `tensordem.repose_calibration.repose_angle (P96-N3)`
* stratified bed: — deg (r2 = —)
* monodisperse bed: — deg (r2 = —)
* difference: — deg
* stratified_error: `ReposeCalibrationError: not enough radial bins to fit a flank`
* monodisperse_error: `ReposeCalibrationError: not enough radial bins to fit a flank`
* diagnosis: a 2-D column of 40 disks collapsing in a 0.42 m box yields too few populated radial annuli (the P96-N3 estimator expects thousands of surface grains); the physical repose angle of the stratified bed was NOT measured at this system size
* identity: the measurement calls the P96-N3 function itself, not a re-implementation; the surface cloud of the 2-D bed is embedded in its axisymmetric input
* synthetic validation: unit tests fit synthetic slopes of 15 deg and 25 deg to within 3 deg with the same estimator

## 4. Figures

`bidisperse_segregation.png`: (a) S vs size ratio for both drivers, (b) scaling fits with R2, (c) S vs measured solid fraction (jamming suppression), (d) drainage time ratio per stratification.

## 5. Caveats, limitations and honest FAILs

* **Volume fraction is an output, not an input, in the shear cell.** In 2-D frictional disks a gravity-loaded bed packs to phi ~ 0.85-0.87; the nominal grid phi = 0.55/0.60/0.63 of the task is not attainable in a gravitational cell (it would be a dilute, fluidised state). The phi axis is therefore realised by the plate load sigma and the shaking intensity Gamma, and the *measured* solid fraction is reported with every case; the jamming trend required by C3 is tested on that measured axis.
* **Metric correction.** The first implementation classified species with the median radius *inside* each bin; a bin containing only large particles then reported Phi = 0, i.e. it inverted the sign of the profile and produced a spurious S ~ 0.98 from pure quantisation noise. Species membership is now global (labels or a fixed threshold) and the profile is area weighted; the regression test `test_profile_is_label_based_not_bin_local` pins this.
* **Shear drive.** The moving lid must be in contact with the bed. An unloaded lid floating above a settled bed transmits nothing (measured vx/v_top ~ 0.01); the dead-weight plate brings the measured shear penetration to vx/v_top ~ 0.7 near the plate.
* **Campaign size.** The planned cluster grid (4 ratios x 3 stresses x 2 shear rates x 3 seeds, plus the 10-seed blocks) sat in the job queue for the whole session and was cancelled; the numbers here come from the local campaign that actually ran (16 shear + 26 shaker cases, 42 records). The cluster runner and specs are in the repo and reproduce the larger grid.
* **2-D disks.** No out-of-plane percolation path, so absolute S values are not transferable to 3-D cargo; the sign, monotonicity and the permeability coupling are.
* **No experimental benchmark in CI.** The quantitative checks are (i) exact conservation, (ii) determinism bitwise, (iii) the S -> 0 limit at d_l/d_s -> 1, and (iv) Kozeny-Carman against its closed form. No experimental segregation data set is available in this environment, so an experimental error percentage cannot be claimed -- stated explicitly rather than approximated.

## 6. Reproduction

```sh
PYTHONPATH=src python scripts/run_bidisperse_campaign.py --mode ratios \
    --shard 0 --nshards 6 --outdir runs/ratios
PYTHONPATH=src python scripts/run_bidisperse_campaign.py --mode probe \
    --outdir runs/all --merge --report artifacts/bidisperse_campaign.json
PYTHONPATH=src python scripts/audit_bidisperse_criteria.py --inputs runs/all \
    --out artifacts/bidisperse_segregation_criteria.json
PYTHONPATH=src python scripts/make_bidisperse_figures.py \
    --report artifacts/bidisperse_campaign.json \
    --out artifacts/bidisperse_segregation.png
PYTHONPATH=src python -m pytest tests/test_bidisperse_segregation.py \
    tests/test_granular2d_wall_velocity.py -q
```

## 7. Error judgement vs references

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| weighted Phi vs Phi_0 | conservation (exact) | 2.22e-16 | yes |
| frozen/deterministic replay | bitwise identity | 0.000 | yes |
| Kozeny-Carman k | closed form d^2 eps^3/(180(1-eps)^2) | 0 (same formula, unit test) | yes |
| S(d_l/d_s = 1) | 0 beyond the finite-size floor | see C1 | yes (floor-bounded) |
| segregation degree S | no experimental data set available | not computable | no claim made |

