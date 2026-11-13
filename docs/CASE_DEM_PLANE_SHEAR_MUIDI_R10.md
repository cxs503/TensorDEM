# CASE-DEM-PLANE-SHEAR-MUIDI — Round 10: the abscissa and window audit (no new simulation)

> **Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=**open**) · **Repository** TensorDEM · **Mission** m17694 · **Round** 10 (after R1 m14960 … R8 m17126, R9 m17546)
> **Artefacts**: `src/tensordem/mu_i_rheology.py` (`inertial_number()` helper + explicit I taxonomy), `benchmarks/plane_shear_mu_i/abscissa_audit.py`, `test_abscissa_audit.py` (15 tests), `runs/abscissa_audit/{prereg.json,result.json,tables.md,fig_abscissa.png}`
> **Zero new simulation.** Capacity discipline: `load15 = 122` on a 128-core host → light work only (analysis + tests). The round-9 wide-cell sweep (`cj1584`, 2112 particles) belongs to m17546 and is **not** re-run or duplicated here.
> **Does this overwrite anything?** No. Every round-8 number is *reproduced* (law column agreed to 0.0e+00; the published `mu_full` error at `I = 0.3` reproduced to 0.1 pp).

## ① Case

Dense plane shear of frictional spheres in a constant-stress servo shear cell, confronted with the local rheology `mu(I)` of da Cruz et al. (2005, PRE 72:021309; GDR MiDi 2004, EPJ E 14:341). Geometry and parameters are the archived round-8 sweep — nothing was re-simulated:

| parameter | value |
|---|---|
| box / fill / radius | 0.12 × 0.05 m, fill 0.06 m, r = 0.005 m (294 particles) |
| `normal_stress_pa` / `kn` | 62500 / 62500 |
| contact law | `mu_contact` 0.5, `mu_rolling` 0.05, `restitution` 0.3 |
| `I_target` grid | 0.01, 0.03, 0.1, 0.3 (single seed 5279) |
| grain density / diameter | 2500 kg m⁻³ / 0.01 m |
| ledger acceptance | `|mu/mu_ref − 1| ≤ 10 %` on the `mu(I)` curve |

**Why round 10 exists.** Round 8 closed with the sentence "`mu(I)` is −6.5 %, −14.4 %, −23.0 %, −27.2 % from the law". That sentence silently fixed three analysis choices that were never audited:

1. **the abscissa** — `mu_ref` was evaluated at `I_target`, the inertial number of the *imposed platen rate* with the *applied* pressure. The ordinate `mu_full` is a **volume** average. Localisation or wall slip makes the volume-averaged strain rate differ from the platen rate, and the device's own `shear_rate_profile` docstring says so ("`I` must use the measured rate if the `mu(I)` abscissa is to mean anything").
2. **the meaning of `I_measured`** — the archived field of that name is `gdot_imposed d / sqrt(P_measured/rho_p)`, i.e. **not** a measured I (round 8 wrote `"I_measured": res["inertial_number"]`, and `inertial_number` was built from `gamma_dot`, the imposed rate). A reader of the artefact naturally assumes a measured abscissa had been used.
3. **the averaging window** — P3 was quoted on the 60 % **tail**, while the constant-stress validity gate (`servo_valid`, round 5) is judged on the 20 % **final** window. The two verdicts are therefore not about the same sample.

## ② What was done

**D1 — the definition gets one home (`inertial_number()`).** `I = |gdot| d / sqrt(P/rho_grain)`, returning `nan` (never 0) for a non-positive pressure, with the grain-density convention stated. The historic key `inertial_number` keeps its exact value (regression-guarded by a test) and is now accompanied by

| field | meaning |
|---|---|
| `inertial_number` | unchanged, historic; **imposed rate × measured pressure** |
| `inertial_number_imposed_rate` | the same value under its true name |
| `inertial_number_measured_rate` | volume-averaged measured affine rate × measured pressure — the abscissa that pairs with a volume average |
| `inertial_number_measured_rate_final` / `shear_rate_measured_final` | the same on the validity gate's window |
| `inertial_number_bulk` / `_final` | unchanged (pre-existing: measured rate × bulk pressure) |

**D2 — the verdict becomes a matrix (`abscissa_audit.py`).** A zero-simulation re-judgement of the archived `runs/full_cell/point_*.json` and `runs/bulk_rheometry/point_*.json` over *abscissa* × *ordinate* × *gate*:

* abscissae: `I_target` (imposed rate, applied P) · `I_imposed_measP` · `I_vol` (measured rate, measured P) · `I_bulk` (measured rate, bulk P);
* ordinates: `mu_full`/`_final` (p = −tr σ/3) and `mu_full_zz`/`_final`, plus the wall-traction and slab ordinates of the preamble arm;
* gate: all four points vs the two that pass `servo_valid`.

The reference law is **rebuilt from `reference_law.ref_mu` and cross-checked against the archived `mu_ref` column** (max relative disagreement `0.0e+00`, a hard error otherwise) — the audit judges round 8 with round 8's law. Inputs are identified by `sha256` in `result.json`. The hypotheses were pre-registered in `runs/abscissa_audit/prereg.json` before the re-judgement code ran; blinding was limited because round 8 already published the errors at `I_target` (stated in the prereg).

**D3 — the abscissa budget.** For each point, `mu_ref(I_vol)/mu_ref(I_target) − 1` is the slice of the measured discrepancy that the abscissa shift alone creates, and `explained = that change / observed error`. The ratio is refused (`null`) when the observed error is under 1 %, where it is ill-conditioned.

## ③ Results

Pre-registered hypotheses, answered:

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| **H1** | the abscissa choice is not cosmetic (`max |I_vol/I_target − 1| ≥ 10 %`) | **PASS** | −61 % at I = 0.01, −30.5 % at 0.1, −8.2 % at 0.3, +1.7 % at 0.03 |
| **H2** | the abscissa explains the error (≤10 % at `I_vol` on ≥3/4 points) | **FAIL** | −4.1 / −14.5 / −18.0 / −25.9 % → 1/4 inside the band |
| **H3** | the verdict is window-stable (same sign tail vs final, 4/4) | **FAIL** | I = 0.03 flips: **−14.4 % (tail) vs +1.3 % (final)** |
| **H4** | on gate-passing points some cell is inside ±10 % | **FAIL** | best cell 1/2 (`I_vol · mu_full_final`); `mu_full_zz*` 0/2 |

**(a) The four abscissae and the rate deficit** (full_cell arm; identical for both arms since it is the same run)

| `I_target` | `I_imposed_measP` | `I_vol` | `I_bulk` | measured/imposed rate | profile R² | `servo_valid` |
|---|---|---|---|---|---|---|
| 0.01 | 0.0102 | **0.0039** | 0.0059 | 0.38 | 0.57 | True |
| 0.03 | 0.0314 | 0.0305 | 0.0350 | 0.97 | 0.88 | False |
| 0.1 | 0.1038 | **0.0695** | 0.0736 | 0.67 | 0.90 | False |
| 0.3 | 0.3033 | 0.2753 | 0.3454 | 0.91 | 0.97 | True |

At the lowest rate the cell shears **2.6× slower than commanded** (wall slip/localisation, profile R² = 0.57) — so the "I = 0.01" point is really an I ≈ 0.004 point, and every point is at a *different* I than its label.

**(b) `mu_full` error against `mu_ref` at each abscissa** [%]

| `I_target` | `I_target` | `I_imposed_measP` | `I_vol` | `I_bulk` |
|---|---|---|---|---|
| 0.01 | −6.5 | −6.5 | −4.1 | −4.9 |
| 0.03 | −14.4 | −14.8 | −14.5 | −15.7 |
| 0.1 | −23.0 | −23.6 | −18.0 | −18.7 |
| 0.3 | −27.2 | −27.3 | −25.9 | −29.2 |

**(c) the abscissa budget — how much of the deficit it explains**

| `I_target` | Δ`mu_ref` [%] | explained (tail) | explained (final) |
|---|---|---|---|
| 0.01 | −2.5 | +0.39 | ill-conditioned (obs. −0.1 %) |
| 0.03 | +0.2 | −0.01 | ill-conditioned (obs. +1.3 %) |
| 0.1 | −6.2 | +0.27 | +0.29 |
| 0.3 | −1.7 | +0.06 | +0.06 |

The abscissa shift recovers **at most 6.2 pp** of a 23 pp discrepancy (I = 0.1) and 0.2 pp of a 14.4 pp discrepancy (I = 0.03). The −27 % verdict of round 8 is **not** an abscissa artefact.

**(d) the ordinates disagree among themselves by more than the band** [%]

| `I_target` | `mu_full` (tail) | `mu_full_final` | `mu_full_zz_final` | `mu_wall_final` | spread |
|---|---|---|---|---|---|
| 0.01 | −6.5 | **−0.1** | −23.6 | **−23.3** | **23.5 pp** |
| 0.03 | −14.4 | +1.3 | −1.0 | +2.1 | 16.5 pp |
| 0.1 | −23.0 | −21.4 | −29.3 | −31.1 | 9.7 pp |
| 0.3 | −27.2 | −27.8 | −40.8 | −40.9 | 13.7 pp |

At I = 0.01 one estimator is dead on the law and another is 23 % low **on the same samples**. The ±10 % acceptance band is narrower than the instrument's own spread at two of four points — so no single estimator can carry a verdict there.

## ④ Figures

`runs/abscissa_audit/fig_abscissa.png` (generated by `abscissa_audit.py`, matplotlib/Agg):
*left* — `mu_ref(I)` curve with the five (I, mu_full) markers per abscissa and a red dashed arrow per point showing the **target → volume-averaged abscissa** move (the arrow is horizontal, so it cannot change the measured `mu`, only the reference it is compared to);
*right* — the same ordinate (`mu_full`) judged at the four abscissae, bars against the ±10 % band: the bars move by a few pp at most, i.e. the deficit survives every abscissa.

## ⑤ Notes, limitations, what the next round must do

1. **What is delivered is a corrected judging instrument, not a number that closes the case.** `mu(I)` remains **not verified**; `phi(I)` remains **WITHHELD** (reference magnitudes still PROVENANCE-UNVERIFIED). The ledger row stays `open`.
2. **A real defect was fixed, and it is the one that mattered for interpretation**: the field named `I_measured` was an imposed-rate value. Any future round reading these artefacts would have quoted an abscissa nobody computed. Values are unchanged; the names now say what they are, and a test pins the historic key.
3. **H3 is the uncomfortable result.** The verdict is not identified by this data: at I = 0.03 the sign flips with the window (tail −14.4 %, final +1.3 %). A round that quotes one window and gates on the other can produce either "law verified" or "law violated" from the same run. Future rounds must pre-register the window *and* quote both.
4. **The gate is not incidental**: 2/4 points fail the constant-stress gate, and restricting to the 2 gate-passing points never gives a full pass under any abscissa (H4). The two gate-passing points are exactly the two where the estimator spread is largest.
5. **`I_bulk` is not a rescue.** It mixes the two effects with opposite signs (measured rate lowers I, deficient bulk pressure raises it) — it lands between `I_target` and `I_vol` and inherits the uncalibrated bulk pressure of rounds 6–7 (later calibrated to 4.2 % by round 8, but only for the `zz` component).
6. **This round cannot separate "the law is wrong" from "the cell is not a material point".** That separation needs the wide cell (round 9, `cj1584`) and a height sweep; round 10 only proves that neither the abscissa nor the window choice is hiding the answer.
7. **Analysis-only scope.** No new trajectory, no new contact-law variant, no seed ensemble: the ensemble gap of round 8 (single seed) is untouched here.

## ⑥ Error judgment (explicit)

| quantity | reference | this round | error | within 3 %? |
|---|---|---|---|---|
| rebuilt reference law vs archived `mu_ref` column | round 8's law | identical on 4/4 points | **0.0e+00** | **yes** |
| published round-8 `mu_full` error at I = 0.3 | −27.2 % (R8 doc) | −27.2 % | ≤ 0.1 pp | **yes** |
| pressure reconstruction `P` from `I_measured` | `gdot d`/`I` inversion | exact to 1e-12 (test) | 0 % | **yes** (algebraic) |
| `mu(I)` vs da Cruz 2005 at the volume-averaged abscissa `I_vol` | law as used | −4.1 / −14.5 / −18.0 / −25.9 % | worst **25.9 %** | **no** (and >10 % ledger band) |
| `phi(I)` | law magnitudes (PROVENANCE-UNVERIFIED) | not judged this round | — | **no verdict** |

**Verdict.** Capability: **PASS** — the I taxonomy is explicit and regression-guarded (`inertial_number` unchanged), the audit reproduces round 8 exactly, and 15 new tests + 52 existing (3 skipped) pass. Physics: **H1 PASS, H2/H3/H4 FAIL** — the abscissa is a real identification defect (61 % at the lowest rate) that explains at most 6.2 pp of the ≥10 pp deficit, the verdict is *not* window-stable, and no gate-restricted cell is inside the ledger band. `mu(I)` stays **not verified**; `phi(I)` stays **WITHHELD**; the ledger row stays `open`. Honest scope: round 10 removes two excuses (abscissa, window) and shows neither is the answer — the wide cell remains the decisive test.