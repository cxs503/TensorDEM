# CASE-DEM-PLANE-SHEAR-MUIDI — Round 8: the stress estimator gets an analytic anchor, and `mu(I)` is re-judged on it

> **Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open) · **Repository** TensorDEM · **Mission** m17126 · **Round** 8 (after R1 m14960, R2 m16427, R3 m16497/m16568, R4 m16633, R5 m16671, R6 m16758, R7 m17026)
> **Artifacts**: `src/tensordem/mu_i_rheology.py` (`full_cell_virial_stress`, `bulk_virial_stress(rule=...)`), `benchmarks/plane_shear_mu_i/full_cell_rheometry.py`, `test_virial_estimators.py`, `runs/full_cell/{prereg.json,result.json,tables.md,fig_full_cell.png}`, `write_r8_doc.py`
> **Device change**: two *additional* stress estimators; the wall traction, the legacy slab estimator, the contact law, the servo protocol and the reference constants are untouched.

## ① Case

Dense plane shear of frictional spheres in a constant-stress servo shear cell (`tensordem.mu_i_rheology.simulate_servo_shear`), confronted with the local rheology `mu(I)`, `phi(I)` of da Cruz et al. (2005, PRE 72:021309; DOI 10.1103/PhysRevE.72.021309) and GDR MiDi (2004, Eur. Phys. J. E 14:341) over the inertial-number range `I = 1e-3 ... 0.3` that the ledger acceptance names (`<10 %` on the `mu(I)` / `phi(I)` curves).

**Why round 8 exists.** Rounds 6-7 measured `mu` from two quantities that had no independent check: the platen traction (`tau_wall/p_wall`) and a thin-slab Love-Weber average that *drops 41-49 % of the load-bearing contacts* (the both-endpoints rule on a 2-4 diameter slab). The resulting `p_bulk/P - 1 = -17 ... -51 %` could not be separated from a genuine stress deficit, so the `mu` verdict rested on an uncalibrated instrument.

| parameter | value |
|---|---|
| `arm` | baseline |
| `I_target` | [0.01, 0.03, 0.1, 0.3] |
| `kn` | 62500.0 |
| `normal_stress_pa` | 62500.0 |
| `seed` | 5279 |
| `mu_contact` / `mu_rolling` / `restitution` | 0.5 / 0.05 / 0.3 |
| box / fill / radius | 0.12 x 0.05 m, fill 0.06 m, r = 0.005 m |
| `n_particles` | 294 |

## ② What was done

**D1 — an estimator with an exact reference (`full_cell_virial_stress`).** For a packing in static equilibrium at zero gravity, per-particle force balance gives the discrete virial identity

```
 sum_pairs f_ij (x) (x_j - x_i) = - sum_i F_i^ext (x) x_i
```

The external loads act at the floor (`z ~ r`) and the platen (`z ~ h - r`), so the `zz` component of the pair sum is `P A (h - 2r)`: dividing by the *centre-accessible* volume `A (h - 2r)` gives

```
 sigma_zz = -P_applied     =>     sigma_zz / (-P_applied) = +1
```

exactly, with **no literature constant and no free parameter**. The function sums every pair of the periodic pair list, normalises by `W D (h - 2r)` and removes the kinetic part with the fluctuation about the measured affine profile `v_x = a + b z`. The invariant is evaluated on every sample (final 20 % window + 60 % tail).

**D2 — the R7-recommended midpoint rule** (`bulk_virial_stress(rule="midpoint")`): a contact enters the slab sum when the *midpoint* of its branch vector lies in the slab, which counts each crossing chain exactly once. Both rules run on the *same* samples, so the comparison isolates the rule.

**D3 — the pressure convention is stated, not implied.** The quoted law is 2-D plane shear with `mu = sigma_xy / sigma_yy`, i.e. normalised by the normal stress *in the shear plane*. `mu_full` uses the mean pressure `p = -tr(sigma)/3`; `mu_full (sigma_zz)` uses `|sigma_xz|/|sigma_zz|` (the 3-D analogue). The two differ by the normal-stress anisotropy, which is `p/|sigma_zz| = 0.771 ... 0.936` here.

**Verification of the estimators themselves** (`benchmarks/plane_shear_mu_i/test_virial_estimators.py`, 5 tests): the rules are checked against a hand-computed `sum f (x) r / V` on a synthetic pair list (a contact straddling the slab is dropped by `both`, counted once by `midpoint`), and the momentum-balance invariant is asserted on a real cell.

Pre-registered verdicts (`runs/full_cell/prereg.json`, written before the sweep):

* **P1_estimator_calibration**: |sigma_zz_full / P_applied + 1| <= 0.05 (final 20 % window, every point)
* **P2_truncation**: mu_slab(both) < mu_slab(midpoint) < mu_full at every point
* **P3_ledger_verdict**: |mu_full/mu_ref - 1| <= 10 % at every point

## ③ Results

### (a) P1 — the estimator calibration (analytic anchor)

| I_target | sigma_zz full [Pa] | sigma_zz/(-P) final window | sigma_zz/(-P) 60 % tail | P1 (±5 %) |
|---|---|---|---|---|
|---|---|---|---|---|---|
| 0.01 | -61334 | 0.9813 | 0.9755 | pass |
| 0.03 | -60160 | 0.9626 | 0.9276 | pass |
| 0.1 | -59849 | 0.9576 | 0.9467 | pass |
| 0.3 | -62887 | 1.0062 | 0.9912 | pass |

Worst deviation from the exact value `+1`: **0.0424** (4.24 %), on 4/4 points inside the pre-registered ±5 % band. On the same samples the legacy both-endpoints slab reads -51.1 % ... -16.8 % and the midpoint rule -35.4 % ... -9.0 % — so the R6/R7 pressure deficit was **mostly estimator truncation**, and the residue is not a constant-stress control failure.

### (b) `mu` on identical samples, four estimators

| I_target | mu ref | mu wall | mu slab both | mu slab mid | mu full (p) | mu full (sigma_zz) |
|---|---|---|---|---|---|---|
| 0.01 | 0.3755 | 0.2709 | 0.2110 | 0.2229 | 0.3512 | 0.2704 |
| 0.03 | 0.4036 | 0.3327 | 0.3354 | 0.3406 | 0.3456 | 0.3273 |
| 0.1 | 0.4800 | 0.3305 | 0.3168 | 0.3345 | 0.3694 | 0.3309 |
| 0.3 | 0.6000 | 0.3518 | 0.3289 | 0.3319 | 0.4370 | 0.3523 |

| I_target | err wall | err slab both | err slab mid | err full (p) | err full (sigma_zz) |
|---|---|---|---|---|---|
| 0.01 | -27.8 % | -43.8 % | -40.6 % | -6.5 % | -28.0 % |
| 0.03 | -17.6 % | -16.9 % | -15.6 % | -14.4 % | -18.9 % |
| 0.1 | -31.2 % | -34.0 % | -30.3 % | -23.0 % | -31.1 % |
| 0.3 | -41.4 % | -45.2 % | -44.7 % | -27.2 % | -41.3 % |

* worst wall-traction error: **-41.4 %** at I=0.3
* worst `mu_full` (p = -tr/3) error: **-27.2 %** at I=0.3
* worst `mu_full` (sigma_zz) error: **-41.3 %** at I=0.3
* inside the 10 % ledger band: `mu_full` 1/4 points, `mu_full (sigma_zz)` 0/4 points

### (c) P2 — is the midpoint rule the fix?

| I_target | mu both / mu full | mu mid / mu full | straddle fraction | rate ratio (slab) | profile R2 | P2 |
|---|---|---|---|---|---|---|
| 0.01 | 0.601 | 0.635 | 48.7 % | 0.38 | 0.57 | pass |
| 0.03 | 0.970 | 0.985 | 44.9 % | 0.97 | 0.88 | pass |
| 0.1 | 0.858 | 0.906 | 41.6 % | 0.67 | 0.90 | pass |
| 0.3 | 0.753 | 0.760 | 40.9 % | 0.91 | 0.97 | pass |

Ordering holds on 4/4 points, but the midpoint rule recovers only a small part of the gap: the slab estimate stays far below the domain average. The reason is visible in the same table — the slab sees `gamma_dot_measured/gamma_dot_imposed = 0.38 ... 0.97` with a velocity-profile `R2` well below 1, i.e. the shear is **localised** near the platens, so a 2-4 diameter interior slab is not a sample of the material that carries the stress. Fixing the contact rule cannot fix a sampling problem.

### (d) solid fraction and control state (unchanged estimators)

| I_target | phi slab | phi bed | phi ref | coordination | dilatancy | servo valid | wall clock [s] |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.6348 | 0.5331 | 0.6381 | 3.78 | -2.12 % | True | 49 |
| 0.03 | 0.6590 | 0.5407 | 0.6345 | 3.85 | -3.49 % | False | 54 |
| 0.1 | 0.6942 | 0.5514 | 0.6250 | 3.61 | -5.37 % | False | 52 |
| 0.3 | 0.7224 | 0.5751 | 0.6100 | 3.72 | -9.26 % | True | 150 |

## ④ Figures

`runs/full_cell/fig_full_cell.png` — left: `mu(I)` for the four estimators against the da Cruz law; middle: the momentum-balance invariant `sigma_zz/P` against the exact value `-1` on both averaging windows; right: the `mu` error of each estimator against the 10 % ledger band.

## ⑤ Notes, limitations, and what a next round must do

1. **The estimator is now calibrated; the material point is not.** P1 shows the domain average reproduces the applied load to 4.2 % on every point, so the remaining `mu` deficit is a statement about the *sample*, not about the arithmetic.
2. **The domain average mixes the boundary layers.** It is exact for `sigma_zz` because momentum balance is global, but `sigma_xz` (and hence `mu`) is a volume average that includes the two localised boundary layers. That is a legitimate rheometric average only if the layers are a small fraction of the volume, which at 294 particles they are not.
3. **The reference law remains 2-D.** da Cruz (2005) is plane shear of disks; the `phi` magnitudes in `reference_law.py` are 3-D numbers of unverified provenance (R7 ⑤), and the `mu` constants `(0.36, 0.84, 0.30)` were inherited from round 1 without an archived reading of the paper. The *shape* of `mu(I)` is the part this case can confront; the constants are still PROVENANCE-UNVERIFIED.
4. **The pressure convention is a 6-23 % lever** on the ordinate. Both are reported; a verdict that quotes only one of them is ambiguous, which is how rounds 1-7 lost track of a ~20 % systematic.
5. **Constant-stress control was not valid everywhere**: `servo_valid` is 2/4 on the final window; at the other points the platen pressure had not yet re-entered the 5 % band when the strain budget was exhausted, so those `mu` values are provisional.
6. **Single seed, no ensemble.** Every point is one 294-particle realisation, so no statistical error bar exists yet; the point-to-point scatter visible in the tables cannot be separated from realisation noise.
7. **Next round (recommended).** (i) widen the cell (target ≥ 2000 particles, ≥ 10 diameters in the shear direction) so the localisation layers are a small fraction and a *local* `mu(z)` profile can be averaged over the homogeneous core; (ii) 3-5 seeds per point for error bars; (iii) only then confront the law on `mu(I)` shape, keeping the `sigma_zz`-normalised figure as the primary ordinate.

## ⑥ Error judgment (explicit)

| quantity | reference | worst measured | error | within 3 %? |
|---|---|---|---|---|
| `sigma_zz/(-P_applied)` (full cell virial) | +1 exact (virial identity) | 0.9576 | 4.24 % | **no** |
| `mu(I)`, full cell, p = -tr/3 | da Cruz 2005 fit | 0.4370 | -27.2 % | **no** |
| `mu(I)`, full cell, sigma_zz | da Cruz 2005 fit | 0.3523 | -41.3 % | **no** |
| `mu(I)`, wall traction | da Cruz 2005 fit | 0.3518 | -41.4 % | **no** |

**Verdict.** P1 (estimator calibration) **PASS** — the domain-average virial reproduces the applied normal stress to 4.24 %. P3 (ledger acceptance `<10 %` on `mu(I)`) **FAIL** with the mean-pressure ordinate on 3/4 points and with the `sigma_zz` ordinate on 4/4 points; the diagnosis is now unambiguous (localisation + 2-D reference + small N, not the stress arithmetic). The ledger row stays **open**: `mu(I)` is not verified, and `phi(I)` remains WITHHELD (R7 ⑤, reference magnitudes unverified).

**Honest scope of this round.** The one thing that is *proved* here is the calibration of the instrument (analytic invariant, 3 verification tests plus a real-cell assertion). The physics verdict is a **negative** result and is reported as such: it narrows the failure to the sample, it does not close the case.

