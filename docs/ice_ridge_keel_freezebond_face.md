# Freeze-bonded ice-ridge keel — validation face (campaign node `p49w2-arb7927-n08`)

Module: `src/tensordem/ice_ridge_keel_freezebond.py`
Adapter: `src/tensordem/adfreeze_key_adapter.py`
Tests: `tests/test_ice_ridge_keel_freezebond.py` (40 tests, 3 negative controls)
Artifacts: `artifacts/ice_ridge_keel/{study.json, acceptance.json,
liquidus_defect_baseline.json, free_structure.json, fig1..fig6}.png`
Scripts: `scripts/ice_ridge_keel_{study,figures,acceptance,liquidus_baseline}.py`

---

## 1. The case

A first-year pressure-ridge keel is a submerged pile of broken ice blocks
(0.1 – 1 m, Rosin–Rammler sized) loaded from above by the sail + consolidated
layer and pushed *up* by its own buoyancy excess. Two things make it a
**structure** rather than a heap: the freeze bonds (thin necks of fresh ice grown
across block–block contacts, i.e. the platform *adfreeze* joint) and the pore
space those necks consume. The chain under test is

```
adfreeze joint strength  ->  DEM bond break force  ->  bonded rubble skeleton
        ->  measured porosity + force chains  ->  drag resistance of a rigid plate
        ->  arbitration against a rigid-plastic (Kerr-type) keel reference
```

| quantity | value used |
|---|---|
| keel depths `h_k` | 2 / 4 / 6 m |
| keel box | 6 m × 6 m (periodic in `y`), depth = `h_k` |
| blocks | 520 spheres, Rosin–Rammler, `d` = 0.1 – 1.0 m, ρ = 917 kg/m³ |
| ice temperature | −10 °C (sweep −2 … −20 °C) |
| adfreeze closure | `t_c` = 3.6 ks, `R_a` = 2.5 µm, `S` = 0 / 32 g/kg |
| numerical | `dt` = 6·10⁻⁴ s, ω_contact = 250 rad/s, μ = 0.5, plate 0.45 m/s over 0.6 m |

## 2. Method

1. **Adfreeze → DEM key.** The platform model is mirrored locally (TensorDEM
   carries no cross-repo import) and mapped as

   `F_break = tau_adfreeze(dT, t, R_a, S) · A_neck`, `A_neck = π (f r_min)²`,
   `f = 0.30`, with `dT = T_f(S) − T_i` measured against the **liquidus of the
   pore water**. `tau_adfreeze` is factorised as `tau_ref(dT)·η(t,dT)·φ_rough(R_a)·φ_salt`.
   The mapping is audited by a DEM twin-particle tensile test: the recorded peak
   bond force is compared with the analytic product for ten parameter sets.
2. **Bonded keel DEM.** RSA placement → geometric affine compaction to the design
   solid fraction (jam-guarded, overlap-audited) → freeze bonds created in the
   contact network → buoyant float-free relaxation under a lid load equal to the
   keel's flotation excess → measurement → plate drag → release.
3. **Porosity / force chains.** 3-D voxel-resolved Voronoi porosity over the keel
   envelope (bond necks counted as solid) plus the strong-force tail fit of
   `P(f)`, `f = F/⟨F⟩`.
4. **Reference.** Rigid-plastic Rankine passive pressure of a submerged rubble
   wedge with buoyant unit weight γ_b < 0, smeared bond cohesion `c` and lid
   surcharge `q`: `F = B[K_p(q h_k + ½γ_b h_k²) + 2c√K_p h_k]`,
   `K_p = (1+sinφ)/(1−sinφ)`. φ is the *macroscopic* angle measured separately in
   a plane-strain biaxial test (24.75°, below the 26.6° contact value).

## 3. Results

### 3.1 Criterion 1 — bond-key consistency (10 sets, incl. 3 sea-ice sets)

| set | T [°C] | t_c [ks] | R_a [µm] | S [g/kg] | τ_adfreeze [MPa] | A_neck [m²] | F_break [N] | rel. err |
|---|---|---|---|---|---|---|---|---|
| 1 | −2 | 3.6 | 2.5 | 0 | 0.1419 | 0.02545 | 3.610·10³ | 0 |
| 3 | −10 | 3.6 | 2.5 | 0 | 0.7325 | 0.01767 | 1.294·10⁴ | 0 |
| 5 | −10 | 0.6 | 2.5 | 0 | 0.6247 | 0.02545 | 1.590·10⁴ | 0 |
| 6 | −10 | 3.6 | 20 | 0 | 1.008 | 0.02545 | 2.565·10⁴ | 0 |
| 7 | −12 | 3.6 | 10 | 0 | 1.036 | 0.01767 | 1.831·10⁴ | 0 |
| 8 | −5 | 3.6 | 2.5 | 32 | 0.0550 | 0.01767 | 9.727·10² | 0 |
| 9 | −10 | 7.2 | 15 | 32 | 0.4022 | 0.02545 | 1.023·10⁴ | 0 |
| 10 | −20 | 3.6 | 5 | 32 | 0.8955 | 0.01131 | 1.013·10⁴ | 0 |

`max_rel_error = 0.0` over ten sets (the DEM break force equals the analytic
`τ·A` to machine precision, because the pull test integrates the same threshold);
the criterion (≤ 5 %) is met with the *whole* margin. All three sea-ice sets bond
— they did **not** before the liquidus fix (§4).

### 3.2 Criterion 3 — drag resistance vs the Kerr-type reference

| h_k [m] | F_keel (DEM) [MN] | F_Kerr (φ = 24.75°) [MN] | ratio | n_standing | bonds formed / active | c_smeared [Pa] |
|---|---|---|---|---|---|---|
| 2 | 1.108 | 0.235 | **4.72** | 0.430 | 623 / 478 | 5 645 |
| 4 | 2.296 | 0.368 | **6.24** | 0.455 | 555 / 398 | 3 627 |
| 6 | 5.648 | 0.823 | **6.87** | 0.436 | 669 / 500 | 5 436 |
| 4 (unbonded) | 1.892 | 0.095 | 19.8 | 0.449 | 0 / 0 | 0 |

The DEM is 4.7 – 6.9 × the rigid-plastic reference — **the 15 % arbitration is not
met**. See §5 for the diagnosis; note that with the previously reported (and
unphysical, §4.3) friction angle of 67° the ratio happened to be 1.26, i.e. the
"agreement" was an artifact of the wrong φ.

### 3.3 Criterion 4 — sensitivity to freeze-bond strength

| T [°C] | τ_adfreeze [MPa] | F_keel [MN] | σ(F_keel) [MN] | n_standing |
|---|---|---|---|---|
| −2 | 0.142 | 1.987 | 0.264 | 0.453 |
| −5 | 0.377 | 1.998 | 0.239 | 0.453 |
| −8 | 0.596 | 2.013 | 0.252 | 0.453 |
| −12 | 0.862 | 2.034 | 0.247 | 0.453 |
| −16 | 1.104 | 2.075 | 0.256 | 0.453 |
| −20 | 1.324 | 2.033 | 0.239 | 0.453 |

Bond strength spans 9.4 × but the resistance moves 4.4 % and is non-monotone at
−20 °C: Spearman ρ(T, F) = −0.829, i.e. +0.83 in bond strength ⇒ **criterion
(ρ ≥ 0.9) not met**. The signal is inside the fluctuation band σ/⟨F⟩ ≈ 12 %.

### 3.4 Criteria 2, 5, 6, 7, 8 — porosity, force chains, controls, conservation

| criterion | measured | verdict |
|---|---|---|
| 2 · porosity 0.30 – 0.45 | 0.430 / **0.455** / 0.436 (h = 2/4/6) | **FAIL** (h = 4 m is 1 % above the band) |
| 5 · no-bond ⇒ n ≥ 0.55 | F_break = 0 exactly; n = 0.449 (0.475 after release), 0.524 with free lateral faces | **FAIL** (bond half passes, porosity half does not) |
| 6 · force-chain tail R² ≥ 0.90 | 0.934 / 0.969 / 0.952 (α = 2.13 / 2.23 / 2.32) | PASS |
| 7 · negative controls | dT = 0 → F_break = 0; h_k = 0 → F_keel = 0.0 N and F_Kerr = 0; n → 1 → F_keel = 4.7·10⁻⁶ N | PASS |
| 8 · mass conservation | relative drift = 0.0 (76 434.095 kg) | PASS |
| S1 · platform parity | 96 grid points, max rel. error 0.0, T_f(32 g/kg) identical | PASS |

Machine-readable verdicts: `artifacts/ice_ridge_keel/acceptance.json`
(5 PASS, 4 FAIL, 0 WITHHELD). **Honest summary: the coupling and the DEM
bookkeeping are verified; the macroscopic resistance arbitration is not.**

## 4. Defects found and fixed in this revision

### 4.1 Liquidus anchor table transposed (blocking, physics)

The local mirror of `tensorphysics.ice.adfreeze` listed the seawater-liquidus
anchors with the temperature list **sorted ascending** next to the
**descending** brine-salinity list, so every anchor was mis-paired. Baseline
recomputed by `scripts/ice_ridge_keel_liquidus_baseline.py`
(`artifacts/ice_ridge_keel/liquidus_defect_baseline.json`):

| quantity | pre-fix | upstream / post-fix |
|---|---|---|
| `T_f(32 g/kg)` | **−15.86 °C** | −1.83 °C |
| `τ_adfreeze(−10 °C, S = 32)` | **0 Pa** | 3.05·10⁵ Pa |
| zeroed sea-ice grid points | **18 / 24** | 0 / 24 |
| nonzero points over the 96-point grid | 36 | 90 |
| max relative error vs platform | **1.0** | **0.0** |

Consequence before the fix: *no freeze bonds at all* in any realistic (saline)
ridge keel above −15.9 °C, silently, with no error raised — the whole salinity
branch of the campaign was unexercised. Guarded by
`test_seawater_freezing_point_freshwater_exact_and_saline_monotone`,
`test_liquidus_anchors_match_the_seawater_phase_diagram` and the 96-point parity
test.

### 4.2 `h_k = 0` negative control was not a null

The control pushed every block to `z = −1 km`, i.e. *through the floor*, where the
one-sided floor penalty springs load them: the "null" keel produced
5.27·10⁷ N. Fixed by parking the blocks behind the plate's swept path:
F_keel = 0.0 N exactly, F_Kerr(h_k = 0) = 0.0 N.

### 4.3 Macroscopic friction angle was unphysical

The biaxial upscaling took the peak stress ratio over steps in which the lateral
servo walls had not reached the confining stress (s3 ≈ 0 ⇒ ratio → 1) and
returned φ = 67.1° — above the 26.6° contact friction angle, clipped only at
sin φ = 0.95. That number went straight into the Kerr reference. The protocol now
(a) prepares the specimen isotropically (lid preload chosen so the lid load
equals the confining pressure), (b) bleeds the lateral confinement along a
deformation path, (c) counts a stress state only when the servo error is < 15 %,
and (d) **fails loudly** if the mobile ratio exceeds the contact friction angle.
Measured: φ = 24.75° (valid), and the Kerr reference is reported over the bracket
φ ∈ [15°, 24.75°] (ratio 4.72 → 5.75 / 6.24 → 7.81 / 6.87 → 8.60).

## 5. Where the physics stands (attention points)

* **Why the DEM is ~5 – 7 × the wedge formula.** (i) the DEM box is closed by a
  rigid back face and lid-loaded, so the plate compacts a *confined* assembly
  instead of failing a free passive wedge — rubble cannot ride up over the plate;
  (ii) the measured plate force carries the plowing of a jammed, bonded skeleton
  (0.38 MN per metre of width, the order of full-scale ridge ramming loads)
  whereas the Rankine integration gives ~60 kN/m for the same depth; (iii) the
  reference uses the buoyant unit weight, which shrinks the friction term to
  ~10 kN/m for `h_k` = 4 m. The module therefore reports the comparison as a
  *micro→macro consistency check that currently fails by a factor 5 – 7*, not as
  a validated match. Closing the loop needs either a free-field drag geometry
  (open front/side faces, material allowed to ride up) or a wedge model with the
  DEM's own confinement — both are follow-up work, not a parameter tweak.
* **Resistance is insensitive to freeze-bond strength** (9.4 × in τ ⇒ 4.4 % in F).
  In a jammed box the plate force is set by the packing fraction and the confining
  lid load; the bonds only decide whether the skeleton *stays* bonded
  (555 – 669 formed, 398 – 500 active at standing). The criterion that demands monotone
  sensitivity therefore encodes an assumption this geometry does not support —
  the honest reading is that the *experiment* must be re-designed (free front
  face, lower initial packing fraction) before criterion 4 can be met.
* **Porosity ceiling.** The bonded keel stands at n = 0.43 – 0.455 (band
  0.30 – 0.45), the unbonded one at 0.449 (0.475 after release, 0.524 with free
  lateral faces). The 0.55 unbonded bound is not reachable with this block-size
  distribution and lid load: the loose limit of the generated Rosin–Rammler
  packing is ~0.52.
* **Adfreeze law is WITHHELD as physics.** τ_ref and the salinity exponent are
  calibrated on four Frederking temperatures and a single comparative brine
  datum; the neck-area fraction `f = 0.30` is an engineering estimate (±50 %) and
  enters the keel cohesion *linearly*. Nothing in this face validates the
  adfreeze law itself — it validates the *mapping* and its mechanical consequence.
* **Kerr (1975) is not digitised.** The monograph text is not in the worktree, so
  the "Kerr-type" reference is the standard rigid-plastic Rankine idealisation.
  Comparing DEM against it is a micro→macro consistency check, not a
  literature-curve validation.

## 6. Error judgement (explicit)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| DEM `F_break` vs `τ_adfreeze·A_neck` | analytic product (10 sets) | **0.0 %** | **yes** |
| mirrored `τ_adfreeze` vs platform model | `tensorphysics.ice.adfreeze`, 96 points | **0.0 %** | **yes** |
| mass balance | initial total mass | **0.0 %** | **yes** |
| porosity vs campaign band | 0.30 – 0.45 | h = 4 m: +1.0 % outside the upper edge | no |
| `F_keel` vs Kerr-type reference | rigid-plastic wedge (φ = 24.75°) | **+372 % … +587 %** | no (factor 4.7 – 6.9) |
| sensitivity ρ(bond strength, F_keel) | ρ ≥ 0.90 | 0.83 | no |
| force-chain tail fit | R² ≥ 0.90 | 0.934 / 0.969 / 0.952 | yes |

No baseline exists for the *keel drag* itself (no digitised experimental curve in
the worktree), so the drag numbers are stated as DEM measurements arbitrated
against an idealisation — the honest status of criteria 3 and 4 is FAIL.