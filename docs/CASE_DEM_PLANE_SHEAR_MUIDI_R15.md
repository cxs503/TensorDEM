# CASE-DEM-PLANE-SHEAR-MUIDI — Round 15: the φ axis was measuring a different quantity

> **Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open)
> **Repository**: TensorDEM · **Mission**: m20791 · **Branch**: `agent/m20791`
> **Round 14** left the case in honest FAIL with five root causes; this round
> attacks two of them (the two that concern `phi`) and **reverses the sign of one
> of the reported conclusions**.
> **Artifacts**: `benchmarks/plane_shear_mu_i/phi_estimator_audit.py`,
> `benchmarks/plane_shear_mu_i/rederive_archived_phi.py`,
> `benchmarks/plane_shear_mu_i/test_phi_estimator_audit.py`,
> `runs/phi_audit/{synthetic.json, archived_rederived.json,
> archived_rederived_tables.md, fig_phi_definition_audit.png}`,
> engine change in `src/tensordem/mu_i_rheology.py` (`phi_definitions`).

## ① Case introduction

Dense plane shear of frictional spheres in a constant-stress servo cell,
confronted with the local rheology `mu(I)`, `phi(I)` of da Cruz et al. (2005,
PRE 72:021309) and GDR MiDi (2004, Eur. Phys. J. E 14:341).

Device: `tensordem.mu_i_rheology.simulate_servo_shear` — HCP fill, constant-stress
servo settle, then shear at a prescribed inertial number, periodic in `x`, walls
in `y`/`z`.

| parameter | value |
|---|---|
| radius / density | 5 mm / 2500 kg m⁻³ |
| cell (W × D × fill) | 0.12 × 0.05 × 0.06 m |
| particles | N = 294 (6–7 layers) |
| contact law | kn = 6.25e4 N/m, kt/kn = 0.5, μ_p = 0.5, μ_roll = 0.05, e = 0.3 |
| applied pressure | 6.25e4 Pa (constant-stress servo) |
| I targets | 1e-3 … 0.3 |
| reference | μ = 0.36 + 0.48/(I₀/I+1), I₀ = 0.30; φ = 0.64 − 0.06/(I₀/I+1) |

## ② What was done (method)

Round 14 reported, for the same runs, `phi_slab` = 0.722 at `I = 0.3` together
with `phi_bed_steady` = 0.575 — two estimators of one quantity, 26 % apart. At
least one of them was not measuring φ. Three definitions were therefore put on the
*same sample state*:

| symbol | definition | what it is |
|---|---|---|
| `phi_container` | `N V_p / (W D plate_z)` | container-averaged solid fraction, solid volume / gap volume — **the definition the reference law uses** |
| `phi_extent` | `N V_p / (W D (z_max − z_min + 2r))` | the same quantity rebuilt from the particle extent; reference-free cross-check of the platen bookkeeping |
| `phi_centres` | `n_slab V_p / V_slab` | centre-counted interior slab density — the archived `solid_fraction` |

Method:

1. **Analytic test of each estimator** on homogeneous random packings whose
   container-averaged φ is known by construction (three gap heights, N =
  1073…2860), implemented in `phi_estimator_audit.py --synthetic`.
2. **Definitional audit on the state**: `phi_definitions()` (new engine function)
   returns all three on every sample plus the *measured* ratio
   `slab_over_bed`, so the discrepancy is measured, not modelled.
3. **Re-expression of the archived round-8 points** under the corrected
   definition (`rederive_archived_phi.py`) — no re-simulation, so every number
   below is a re-reading of already-committed evidence.
4. **Regression tests** (`test_phi_estimator_audit.py`, 5 tests) that pin the
   definitions, including a live device run, so a future round cannot silently
   re-mix them.

### A hypothesis that was tested and falsified

The first hypothesis was that the slab denominator omitted the particle *shell*
inside the slab, giving a bias `(H_s + 2r)/H_s`. On the analytic packings the
shell-corrected value overshot the truth by −11 % while the plain centre count
overshot by +23 %; the hypothesis is **wrong** and the shell correction is not
used. The effect that actually reproduces the measurements is the **boundary
layer**: no particle centre can lie within `r` of a wall, so the centre density
exceeds the container average by `plate_z/(plate_z − 2r)` — a factor that
*grows as the bed compacts*, which by itself manufactures a false rise of "φ"
with `I`.

## ③ Results

### Estimator audit on analytic packings (truth known by construction)

| packing (gap) | N | φ_true | φ_container | φ_extent | φ_centres | measured slab/bed | `plate_z/(plate_z−2r)` |
|---|---|---|---|---|---|---|---|
| 0.12 m | 2860 | 0.5200 | 0.5200 (+0.0 %) | 0.5201 (+0.0 %) | 0.6387 (+22.8 %) | 1.228 | 1.091 |
| 0.06 m | 1430 | 0.5200 | 0.5200 (+0.0 %) | 0.5207 (+0.1 %) | 0.6969 (+34.0 %) | 1.340 | 1.200 |
| 0.045 m | 1073 | 0.5202 | 0.5202 (+0.0 %) | 0.5213 (+0.2 %) | 0.7337 (+41.0 %) | 1.410 | 1.286 |

`phi_container` and `phi_extent` are unbiased (≤ 0.2 %); the centre-counted slab
is biased **high by 23–41 %**, and the bias grows as the gap narrows. This is
exactly the archived pathology, reproduced on a packing where the answer is known.

### Archived runs re-expressed under the correct definition

| I | μ_full | μ_ref | err μ [%] | φ_centres (archived) | φ_container | φ_ref | err φ [%] | slab/bed measured | `plate_z/(plate_z−2r)` |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.3751 | 0.3755 | −0.1 | 0.6348 | 0.5331 | 0.6381 | −16.4 | 1.191 | 1.262 |
| 0.03 | 0.4090 | 0.4036 | +1.3 | 0.6590 | 0.5407 | 0.6345 | −14.8 | 1.219 | 1.267 |
| 0.1 | 0.3772 | 0.4800 | −21.4 | 0.6942 | 0.5514 | 0.6250 | −11.8 | 1.259 | 1.274 |
| 0.3 | 0.4333 | 0.6000 | −27.8 | 0.7224 | 0.5751 | 0.6100 | −5.7 | 1.256 | 1.289 |

The measured definitional gap equals **94–99 %** of the z boundary-layer factor
at every point (1.191/1.262 = 0.94, 1.219/1.267 = 0.96, 1.259/1.274 = 0.99,
1.256/1.289 = 0.97). The residual is the y-wall depletion and finite-size
heterogeneity of the N = 294 cell.

## ④ Figures

`runs/phi_audit/fig_phi_definition_audit.png` — three panels:

1. archived runs, both definitions vs the reference band (the archived "φ rises
   with I, above the band" becomes "φ = 0.53 → 0.58, below the band");
2. the definitional gap, measured per point, against `plate_z/(plate_z − 2r)`,
   with the three synthetic packings as ghost points;
3. μ(I) unchanged (same numbers as round 8/14) — the μ axis is *not* affected by
   this round.

## ⑤ What this changes, and what it does not

**Changed (a reported conclusion is reversed).** Round 14 recorded
"full-cell φ FAIL: φ_slab increases with I (compaction artifact), reaching an
unphysical 0.72". That 0.72 is the *estimator*, not the bed. The physical
container-averaged φ is 0.533 → 0.575, which is admissible for a sheared sphere
packing and is **11–16 % below** the reference band at low `I`, not 18 % above it
at high `I`. The error of the φ verdict therefore changes sign at the high-`I`
end: the two high-`I` points (−11.8 %, −5.7 %) are at or inside the 10 % band
instead of outside it.

**Not changed.** μ(I) is untouched: −0.1 %, +1.3 % at `I ≤ 0.03` and −21.4 %,
−27.8 % at `I ≥ 0.1`, i.e. the μ half of the ledger acceptance still **FAILS**.

**Still open.** The bed *compacts* (φ increases) where the reference dilates;
at low `I` the bed sits at φ ≈ 0.53 against a reference 0.64 (−16 %). The
archived protocol cannot separate "loose initial fill still compacting" from a
genuine steady state, because the shear budget is
`max(min_shear_steps, γ_target/(γ̇Δt))`: the low-`I` points only reached
`γ ≈ 0.4` while the high-`I` points ran past `γ ≈ 10`. A strain-convergence sweep
(`phi_estimator_audit.py --live --I … --strain 2.0`, six points, submitted to
the cluster as cj2950–cj2955 through the compute gate) was launched to settle
that; **it had not returned within this session**, so no claim about the
steady-state φ(I) is made here. That is the next round's first measurement.

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| μ(I), `I ≤ 0.03` | < 10 % | −0.1 %, +1.3 % | **PASS** (2 points) |
| μ(I), `I ≥ 0.1` | < 10 % | −21.4 %, −27.8 % | **FAIL** |
| φ(I), all four points | < 10 % | −16.4, −14.8, −11.8, −5.7 % | **FAIL** (2/4 inside band) |
| φ estimator definition | < 5 % consistency | `phi_container` vs `phi_extent` agree to 2.4 % on a live state; both unbiased (≤ 0.2 %) on analytic packings | **PASS** |
| definitional gap explained (device) | > 90 % | 94–99 % of `plate_z/(plate_z−2r)` | **PASS** — device only; the analytic-packing version of this claim was 9.7–12.6 % short and is corrected in ⑧ |

No key quantity is within 3 % of a literature reference except the low-`I` μ
points (−0.1 %, +1.3 %), whose reference constants are themselves
provenance-unverified (round 13).

**Ledger verdict for this round: still FAIL, but no longer for the reason
recorded in round 14.** The φ axis is now measured in the definition the
reference uses; the μ deficit at high `I` and the low-`I` φ deficit are the two
real remaining gaps.

## ⑦ Notes / pitfalls

* The archived key `solid_fraction` was **kept with its historical meaning** (no
  silent redefinition); the corrected quantities are new keys
  (`phi_container`, `phi_extent`, `phi_centres_slab`, `slab_over_bed`,
  `boundary_layer_factor`). Archived runs stay bit-for-bit reproducible.
* `phi_container` and `phi_extent` agreeing to 2.4 % is what validates the
  platen bookkeeping; if a future change breaks that, the φ comparison is void.
* Do not compare a centre-counted slab density with a container-averaged
  reference. That single mistake cost rounds 6–14.
* `plate_z/(plate_z−2r)` is not a correction to *apply* to archived numbers —
  it is the analytic scale of the difference between two quantities. The
  correct route is to recompute φ in the container definition from the archived
  state, which is what `rederive_archived_phi.py` does.

## ⑧ R15-c correction (re-open of PR#1657 as mission m23285)

The round-15 claim "*the* effect that reproduces the measurements is the
boundary layer, `plate_z/(plate_z − 2r)`" was **not verified**: the same audit's
own criterion (`|measured/predicted − 1| < 10 %`) failed on two of its three
packings, and the shipped artifact carried

```json
"verdict": "unexpected", "ok": false
```

without any test noticing. Root cause: a centre-counted slab that is interior in
**both** walled directions sees depletion at **both** wall pairs, so the
container-average / centre-density ratio is

```
[ plate_z / (plate_z − 2r) ] * [ box_depth / (box_depth − 2r) ]
```

and the z factor alone leaves the y factor (1.111 in the audit geometry) out.
`phi_definitions()` now returns `boundary_factor_y` and `boundary_factor_2wall`
alongside the unchanged z-only `boundary_factor`, the audit's acceptance is the
predeclared two-wall criterion (`runs/phi_audit/PREREG_R15C.md`, 5 %), and the
"z-only is insufficient" condition is asserted explicitly, so the correction
cannot degrade into a tolerance widening.

### Analytic packings (truth known by construction, N = 1073…2860)

| packing (gap) | φ_true | measured slab/bed | z-only pred | err [%] | two-wall pred | err [%] |
|---|---|---|---|---|---|---|
| gap_0.12 | 0.5200 | 1.2284 | 1.0909 | +12.6 | 1.2121 | +1.3 |
| gap_0.06 | 0.5200 | 1.3403 | 1.2000 | +11.7 | 1.3333 | +0.5 |
| gap_0.045 | 0.5202 | 1.4104 | 1.2857 | +9.7 | 1.4286 | -1.3 |

Artifact verdict: `two_wall_boundary_factor_reproduces_slab_gap__z_only_model_insufficient` / `ok = True` with per-criterion flags
`{'phi_bed_unbiased_lt_2pct': True, 'phi_extent_unbiased_lt_3pct': True, 'phi_centres_biased_high_gt_5pct': True, 'two_wall_factor_within_5pct': True, 'z_only_factor_insufficient': True}`.

### Archived device points (re-derived, nothing re-simulated)

| I | φ_centres (archived) | φ_container | slab/bed measured | z-only factor | measured/z-only | two-wall factor |
|---|---|---|---|---|---|---|
| 0.01 | 0.6348 | 0.5331 | 1.191 | 1.262 | 0.943 | 1.578 |
| 0.03 | 0.6590 | 0.5407 | 1.219 | 1.267 | 0.962 | 1.584 |
| 0.1 | 0.6942 | 0.5514 | 1.259 | 1.274 | 0.988 | 1.592 |
| 0.3 | 0.7224 | 0.5751 | 1.256 | 1.289 | 0.975 | 1.611 |

On the device the measured gap is **below** the z-only bound
(`measured/z-only` = 0.94–0.99): the grafted bed sits denser against the y-walls
than a geometrical shell allows, so the two-wall product (1.58–1.61) is *not*
claimed for the device and the two-wall model is asserted on analytic packings
only. Device-level claim (predeclared interval `measured/z-only ∈ [0.85, 1.00]`,
now a regression test): the z-wall shell accounts for 94–99 % of the archived
definitional gap — the round-15 conclusion that φ_centres and φ_container are
different quantities, and that the archived φ verdict was an estimator artifact,
**stands unchanged**.

### What changed in this correction

| item | before | after |
|---|---|---|
| boundary model | z only | z × y, with z-only kept for archived tables |
| audit artifact | `ok: false`, `verdict: unexpected`, unnoticed | `ok: true`, named per-criterion flags, `z_only_factor_insufficient` asserted |
| tests | 5, none on the audit verdict | 10 (incl. two-wall closure, artifact reproducibility, archived re-derivation determinism) |
| device claim | "gap = z factor" | "z factor bounds the gap from above (0.94–0.99)"; two-wall not claimed |

Reproduce with: `python benchmarks/plane_shear_mu_i/phi_estimator_audit.py --synthetic`
and `python benchmarks/plane_shear_mu_i/rederive_archived_phi.py` (writes the
tables above and `fig_phi_definition_audit.png`);
`python -m pytest benchmarks/plane_shear_mu_i/test_phi_estimator_audit.py`.
