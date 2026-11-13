# CASE-DEM-PLANE-SHEAR-MUIDI — Round 7: two definitional defects, and the `mu` verdict that survives them

> **Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open) · **Repository**: TensorDEM · **Mission**: m17026 · **Round**: 7 (after R1 m14960, R2 m16427, R3 m16497/m16568, R4 m16633, R5 m16671, R6 m16758)
> **Artifacts**: `benchmarks/plane_shear_mu_i/reference_law.py`, `phi_reference_audit.py`, `bulk_stress_rheometry.py`, `test_phi_estimator_and_reference.py`; `runs/phi_reference_audit.json`, `runs/phi_reference_audit_tables.md`, `runs/fig_phi_reference_audit.png`, `runs/bulk_rheometry/{result.json,tables.md,fig_bulk_vs_wall.png}`
> **Device change**: `tensordem.mu_i_rheology.simulate_servo_shear` (solid-fraction field, bulk-stress diagnostics). No contact law, no loading protocol, no reference constant was changed.

## ① Case

Dense plane shear of glass beads at a servo-held normal stress; the case confronts the measured effective friction `mu = tau/P` and solid fraction `phi` with the da Cruz et al. (2005, PRE 72:021309) / GDR MiDi (2004) rheology over `I = 1e-3 ... 0.3`.  The ledger acceptance is `<10 %` on the `mu(I)` and `phi(I)` curves.

| parameter | value |
|---|---|
| `arm` | baseline |
| `I_target` | [0.01, 0.03, 0.1, 0.3] |
| `kn` | 62500.0 |
| `normal_stress_pa` | 62500.0 |
| `seed` | 5279 |

## ② What was done

Round 5 showed that the high-`I` points of rounds 1-3 were taken with a servo that did not hold `P` (up to +58 % deviation) and recommended (a) a bulk-stress estimator and (b) suspicion on the `phi` comparison. Round 6 pre-registered that bulk-vs-wall test but its artifacts never materialised (`write_tables` raised `KeyError: 'I_meas'`, so only `prereg.json` was committed). This round:

**D1 — the device's `solid_fraction` was the slab porosity.** `simulate_servo_shear` sampled

```
phi = 1.0 - n_bulk*V_p/V_slab     # porosity, labelled solid_fraction
```

so every stored `phi_device` (0.36-0.39) is `1 - phi`.  Fixed to `n_bulk*V_p/V_slab`, with `porosity`, `slab_n_particles` and `slab_volume` exposed so the estimator can be re-derived from a sample.

**D2 — the `phi(I)` reference law had the sign of dilatancy inverted.** Rounds 1-5 used

```
phi(I) = 0.58 + 0.06/(I0/I+1)      # rising with I
```

A sheared bed dilates, so `phi` must *fall* with `I` (da Cruz 2005 Fig. 3; GDR MiDi 2004; the device itself reports `dilatancy > 0`).  The law now lives in one place, `reference_law.py`, as `phi = 0.64 - 0.06/(I0/I+1)` — same two magnitudes, physically admissible roles (`phi_max` at `I->0`, `phi_min` at `I->inf`).  The five benchmark scripts import it instead of carrying private copies.

**R6 completed and extended.** `bulk_stress_rheometry.py` now runs (key bug fixed, single-point mode added for the wall-clock budget), reports `mu` from wall traction *and* from the slab Love-Weber virial on the same points, and additionally reports the **dropped-contact fraction**: the fraction of slab contacts that the both-endpoints rule discards, which is the diagnostic for the pressure-balance failure.

## ③ Results

### (a) `phi` verdict under the corrected definitions (legacy rows, re-read not re-simulated)

| arm | worst abs phi err, corrected | worst abs phi err, mislabeled reading | worst abs mu err | phi monotone decreasing |
|---|---|---|---|---|
| `round1` | 4.4 % | 41.9 % | 19.5 % | False |
| `round2_baseline` | 4.4 % | 41.9 % | 19.5 % | False |
| `round2_stiff` | 4.5 % | 41.0 % | 27.0 % | False |
| `round2_thick` | 14.1 % | 34.9 % | 50.5 % | False |

The definition defect alone was worth **~35-40 percentage points** of apparent `phi` error: the rounds-1/2/3 story "the bed is 35-42 % below the reference `phi`" was an artifact of tabulating `1 - phi`.

### (b) bulk (Love-Weber) vs wall rheometry at constant stress (R6 arms, gate on)

| I target | I measured | mu wall | mu bulk | mu ref | err wall % | err bulk % | p bulk / P - 1 | dropped-contact fraction | servo valid |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.0102 | 0.2709 | 0.2110 | 0.3755 | -27.8 | -43.8 | -51.1 % | 48.7 % | True |
| 0.03 | 0.0314 | 0.3327 | 0.3354 | 0.4036 | -17.6 | -16.9 | -29.6 % | 44.9 % | False |
| 0.1 | 0.1038 | 0.3305 | 0.3168 | 0.4800 | -31.2 | -34.0 | -16.8 % | 41.6 % | False |
| 0.3 | 0.3033 | 0.3518 | 0.3289 | 0.6000 | -41.4 | -45.2 | -34.8 % | 40.9 % | True |

## ④ Figures

* `runs/fig_phi_reference_audit.png` — left: the corrected decreasing reference against the legacy rising form and the measured slab solid fraction (and the legacy `phi_device` = porosity); right: `phi` error against the 10 % band after both fixes.
* `runs/bulk_rheometry/fig_bulk_vs_wall.png` — `mu(I)` from wall traction vs slab virial, the slab pressure balance, and the strain rate seen by the slab.

## ⑤ Notes, limitations

1. **The `phi` magnitudes are still unverified.** `(0.58, 0.64)` is inherited from round 1 with no in-repo literature anchor, and da Cruz (2005) is a 2D-disk study (`phi ~ 0.84 -> 0.79`), so these are 3D numbers of unknown provenance.  What is now verified is the *direction*, the *limits* and the *single source*.  `phi` acceptance stays **WITHHELD**.
2. **The 10 % `phi` gate is not discriminative.** The reference band spans `0.640 -> 0.609` (5 %) over the whole `I` range, so almost any dense bed passes a 10 % gate.  Passing it must not be reported as verification of `phi(I)`.
3. **The slab solid fraction is geometry-sensitive.** The slab is 2-3 particle layers thick (margin `0.25*plate_z`), so `phi_slab` moves by ~10 % between protocols (legacy settled rows ≈ 0.61-0.64, the R6 gate-on rows 0.63-0.72).  A converged `phi(I)` curve needs a thicker slab / larger `N`.
4. **The bulk virial drops ~40-50 % of the slab contacts** (both-endpoints rule on a thin slab), which is the same order as the measured pressure deficit — so P1's failure is an estimator truncation, not evidence against the constant-stress control.  A midpoint-rule estimator is the fix and is recommended as the next round.
5. The legacy artifacts were not re-simulated: the `phi` correction is the exact algebraic inversion of the documented formula, and it is pinned by `test_phi_estimator_and_reference.py` against a fresh run.

## ⑥ Error judgment (explicit)

| quantity | reference | best measured | error | within 3 %? |
|---|---|---|---|---|
| `mu(I)`, wall traction, gate on | da Cruz 2005 fit | 0.2709 ... 0.3518 | worst 41.4 % | **no** |
| `mu(I)`, slab virial, gate on | da Cruz 2005 fit | 0.2110 ... 0.3289 | worst 45.2 % | **no** |
| `phi(I)`, corrected slab estimator | unverified (0.64->0.58) | 0.606-0.637 (legacy rows) | worst 4.4 % | **not answerable** (reference provenance unverified, band only 5 % wide) |

**Answer to the required question:** no key quantity is within 3 % of a verified reference.  `mu` is 41.4 % (wall) / 45.2 % (bulk virial) low at worst with the constant-stress gate on, so the ledger acceptance (`<10 %`) remains **FAIL on `mu`** and **WITHHELD on `phi`**.  The case must not be closed.  Two further honest caveats from the same table: 2 of 4 points still finish outside the 5 % servo band (`servo_valid = False`), and the slab sees as little as 38 % of the imposed shear rate (strain localisation / wall slip, P3 FAIL at `I = 0.01`), so even the `I` abscissa is measured rather than imposed.

The two definition fixes do change the *diagnosis*: `phi` is no longer the problem it was reported to be, `mu` is, and the wall-traction estimator is not the reason (wall and bulk agree to ~5 % once both exist on the same point, except at `I = 0.01` where the slab virial is weakest).

