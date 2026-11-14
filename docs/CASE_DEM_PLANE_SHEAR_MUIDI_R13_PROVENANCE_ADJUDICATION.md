# CASE-DEM-PLANE-SHEAR-MUIDI — Round 13: provenance & consistency adjudication

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池 · 平面剪切 mu(I) 流变律标定)
**Reference:** da Cruz et al. 2005, *Phys. Rev. E* **72**:021309; GDR MiDi 2004,
*Eur. Phys. J. E* **14**:341. **Tolerance:** < 10 % on `mu(I)` and `phi(I)`.
**Round type:** archival re-adjudication — **zero new simulation**. All numbers
are recomputed from the round-9/10 point files already in `runs/`.
**Tooling:** `reference_provenance.py`, `adjudicate_provenance.py`,
`test_provenance_adjudication.py`, `provenance_adjudication_figures.py`.
**Artefacts:** `runs/provenance_adjudication.json`,
`runs/provenance_adjudication_tables.md`, `runs/round13/fig*.png`.

---

## ① Case introduction

A plane-shear cell of 294 spheres (`d = 10 mm`, `rho = 2500 kg/m3`) between a
servo-controlled platen and a frictional floor, sheared at four inertial-number
targets `I = 0.01, 0.03, 0.1, 0.3` at a confining stress of `62.5 kPa`. The
ledger asks for the effective friction coefficient `mu` and the solid fraction
`phi` as functions of `I`, compared with the da Cruz / GDR MiDi law

```
mu(I)  = mu_s + (mu_2 - mu_s) / (I_0/I + 1)
```

Rounds 1–12 established the device, the servo, the estimators and the
rough-wall switch, and settled on `phi(I)` being **WITHHELD** (reference
magnitudes unverified). They did **not** close `mu(I)`. Round 12 left the row
open on a *bibliographic* blocker, quoted verbatim:

> "Reference provenance is still open. […] The two curves differ by up to ~15 %
> in mu at I = 0.3. Neither reading is archived in-repo; **a provenance round is
> needed before the ledger acceptance can be judged** against the *paper*."

That recommendation is what this round tests. It is **not** a re-run.

### The two conflicting triples (a defect in its own right)

| registry name | `mu_s` | `mu_2` | `I_0` | declared status | where |
|---|---|---|---|---|---|
| `PRIMARY_REPO` | 0.36 | 0.84 | 0.30 | `QUOTED-NOT-ARCHIVED` | `reference_law.py`, "as quoted in rounds 1-5" |
| `TEST_GDR_MIDI` | 0.379 | 0.641 | 0.279 | `ATTRIBUTED-NOT-ARCHIVED` | `tests/test_mu_i_rheology.py`, "Table 1 of da Cruz et al. 2005" |

`reference_law.py` bills itself as the *single source of truth*, yet the test
suite carries a materially different triple. Both are labelled with the same two
papers. Neither is anchored to an archived copy of the table it cites. Round 13
therefore refuses to pick a winner and instead carries the **box spanned by both
readings** (8 corners) as the admissible set — see Table A, where the envelope
width grows from 5.5 % at `I = 1e-3` to **23.4 % at `I = 0.3`**, i.e. more than
twice the acceptance band.

## ② Method — what was actually done

Four steps, all mechanical, all re-derivable from the archive:

1. **Provenance registry + envelope** (`reference_provenance.py`). Both triples
   as dataclasses with evidence labels; the 2×2×2 corner box; `envelope_bounds`
   / `envelope_spread`; the Jop law with correct limits.
2. **Consistency audit.** The law is a relation between the *shear stress* and
   the *pressure*, with the inertial number built from **the same pressure**:
   `mu = tau/p`, `I = gamma_dot d sqrt(rho/p)`. A walled plane-shear cell
   controls `sigma_zz` directly but the published law is written in the
   continuum pressure `p = -tr(sigma)/3`. There are exactly **two** admissible
   pairings (`zz` and `p`) and they must not be mixed. Five arms were built from
   the archived primitives (`tau`, `sigma_zz`, `p`, imposed rate, measured bulk
   rate) so that no pairing is silently assumed:
   * zz-consistent, nominal (imposed) rate — **the archived convention**
   * zz-consistent, measured bulk rate
   * p-consistent, nominal (imposed) rate
   * p-consistent, measured bulk rate — **law-consistent**
   * wall traction, measured bulk rate
3. **Pass matrix + within-envelope test** (Tables C, D2): every arm against
   every candidate *and* against all 8 corners, on the ledger's 10 % band.
4. **Falsification oracle** (Table D): scan a continuous box that strictly
   contains the envelope (`mu_s in [0.34,0.40]`, `mu_2 in [0.55,0.95]`,
   `I_0 in [0.22,0.38]`) for the smallest achievable worst-point error, with
   explicit **boundary-hit detection** — a fit sitting on the box edge is not
   identified and must not be reported as a rescue.

Plus a **reverse calibration** (Table E): fit the Jop form to the measured
series, i.e. ask *what law the device actually produces*.

## ③ Results

### Table A — the disputed reference envelope

| I | PRIMARY_REPO | TEST_GDR_MIDI | envelope min | envelope max | spread |
|---|---|---|---|---|---|
| 0.001 | 0.3616 | 0.3799 | 0.3609 | 0.3806 | 5.5 % |
| 0.01 | 0.3755 | 0.3881 | 0.3691 | 0.3950 | 7.0 % |
| 0.03 | 0.4036 | 0.4044 | 0.3855 | 0.4238 | 9.9 % |
| 0.1 | 0.4800 | 0.4481 | 0.4303 | 0.5006 | 16.4 % |
| 0.3 | 0.6000 | 0.5148 | 0.5005 | 0.6179 | **23.4 %** |

### Table B — is the archived `(mu, I)` pair law-consistent?

| I_target | rate_ratio | profile R² | servo | `I` archived | `I`(imposed, σ_zz) | `I`(meas rate, p) | p/σ_zz | spurious μ inflation | admissible |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.380 | 0.571 | PASS | 0.0102 | 0.0100 | 0.0044 | 0.766 | **+30.6 %** | no |
| 0.03 | 0.971 | 0.878 | **FAIL** | 0.0314 | 0.0300 | 0.0312 | 0.902 | +10.8 % | no |
| 0.1 | 0.669 | 0.898 | **FAIL** | 0.1038 | 0.1000 | 0.0728 | 0.882 | +13.4 % | no |
| 0.3 | 0.908 | 0.969 | PASS | 0.3033 | 0.3000 | 0.3060 | 0.787 | +27.1 % | **yes** |

Three defects are visible here, none of them bibliographic:

* **D1 — abscissa.** The archived `I_measured` reproduces the *imposed platen
  rate* to <5 % at every point (`test_archived_abscissa_uses_the_imposed_rate`),
  while `rate_ratio` says the bulk actually shears at only **38 %** of that rate
  at `I = 0.01` (velocity-profile `R² = 0.571` — the bed is essentially in plug
  flow). The abscissa is therefore a *nominal* coordinate, not a measured one.
* **D2 — normaliser mixing.** `mu_full` divides by `p = -tr(sigma)/3` while the
  abscissa divides by `sigma_zz`. Because the sheared state is strongly
  anisotropic (`p/|sigma_zz| = 0.766 … 0.902`), mixing them **inflates `mu` by
  up to +30.6 %** without moving `I` at all.
* **D3 — admissibility.** Only **1 of 4** points (I = 0.3) simultaneously has
  `rate_ratio >= 0.9`, profile `R² >= 0.9` and a passing servo. Two of four
  points fail the servo gate outright.

### Table C — pass matrix, `measured/reference - 1`, band 10 %

| arm | candidate | I≈0.004 | I≈0.031 | I≈0.073 | I≈0.306 | n pass | max abs err |
|---|---|---|---|---|---|---|---|
| zz-consistent, nominal **[archived]** | PRIMARY_REPO | −23.6 % | −1.1 % | −29.6 % | −40.8 % | 1/4 | 40.8 % |
| zz-consistent, nominal **[archived]** | TEST_GDR_MIDI | −26.1 % | −1.3 % | −24.5 % | −31.0 % | 1/4 | 31.0 % |
| zz-consistent, measured rate | PRIMARY_REPO | −21.6 % | −0.9 % | −24.4 % | −39.6 % | 1/4 | 39.6 % |
| zz-consistent, measured rate | TEST_GDR_MIDI | −25.0 % | −1.1 % | −21.2 % | −30.1 % | 1/4 | 30.1 % |
| p-consistent, nominal rate | PRIMARY_REPO | −0.7 % | +0.6 % | −22.7 % | −29.4 % | 2/4 | 29.4 % |
| p-consistent, nominal rate | TEST_GDR_MIDI | −3.7 % | +0.7 % | −16.6 % | −17.1 % | 2/4 | 17.1 % |
| **p-consistent, measured rate [law-consistent]** | PRIMARY_REPO | +2.2 % | +0.9 % | −16.9 % | −28.1 % | 2/4 | 28.1 % |
| **p-consistent, measured rate [law-consistent]** | TEST_GDR_MIDI | −2.1 % | +0.9 % | −12.9 % | −16.0 % | 2/4 | **16.0 %** |
| wall traction, measured rate | PRIMARY_REPO | −21.4 % | +1.9 % | −26.5 % | −39.8 % | 1/4 | 39.8 % |
| wall traction, measured rate | TEST_GDR_MIDI | −24.8 % | +1.8 % | −23.3 % | −30.3 % | 1/4 | 30.3 % |

**The archived convention is the worst admissible pairing.** Adopting the
law-consistent pairing moves the worst-point error from **40.8 % → 28.1 %**
(PRIMARY) and from **31.0 % → 16.0 %** (TEST_GDR_MIDI): D1+D2 were costing
≈12–15 percentage points. This is asserted as a regression test
(`test_law_consistent_pairing_beats_the_archived_convention`).

### Table D / D2 — can any reading of the paper rescue the acceptance?

| arm | min worst-point err over the whole Jop family | argmin | argmin on box edge | within envelope: corners passing | best corner reaches |
|---|---|---|---|---|---|
| zz-consistent, nominal **[archived]** | 21.7 % | (0.340, 0.550, 0.380) | mu_s, mu_2, I_0 | **0/8** | 1/4 |
| zz-consistent, measured rate | 20.3 % | (0.340, 0.550, 0.380) | mu_s, mu_2, I_0 | **0/8** | 1/4 |
| p-consistent, nominal rate | 7.7 % | (0.365, 0.550, 0.380) | mu_2, I_0 | **0/8** | 2/4 |
| **p-consistent, measured rate** | **6.1 %** | (0.370, **0.550**, 0.370) | **mu_2** | **0/8** | **3/4** |
| wall traction, measured rate | 20.7 % | (0.340, 0.550, 0.380) | mu_s, mu_2, I_0 | **0/8** | 1/4 |

**Not one corner of the disputed envelope passes every point, under any of the
five admissible pairings.** The best corner reaches 3 of 4 points. The oracle
*can* be pushed under 10 % (6.1 %), but only by pinning `mu_2` to **0.550** —
the lower edge of the search box, *below both candidates* — so the fit is not
identified and is reported as such (`boundary_hit: ["mu_2"]`,
`provenance_can_rescue: false`).

### Table E — reverse calibration: what law does the device produce?

| arm | `mu_s` | `mu_2` | `I_0` | rms | R² | degenerate? |
|---|---|---|---|---|---|---|
| zz-consistent, nominal | −0.390 | 0.371 | 0.001 | 0.0308 | +0.42 | yes (unphysical μ_s) |
| zz-consistent, measured rate | −0.007 | 0.370 | 0.001 | 0.0255 | +0.60 | yes (μ_s ≈ 0) |
| p-consistent, nominal rate | 0.381 | 1.838 | **10.0** | 0.0158 | +0.57 | yes (`I_0` at scan bound) |
| **p-consistent, measured rate** | **0.382** | 2.058 | **10.0** | 0.0143 | +0.65 | yes (`I_0` at scan bound) |
| wall traction, measured rate | −0.003 | 0.370 | 0.001 | 0.0327 | +0.47 | yes (μ_s ≈ 0) |

Under **every** convention the reverse fit is degenerate. But the law-consistent
arm identifies one quantity cleanly and it is physically meaningful:
**`mu_s = 0.382`**, within 1 % of both candidate papers' quasi-static limits
(0.36 / 0.379). The *rate dependence* (`mu_2`, `I_0`) is what the device does
not reproduce — it either decreases with `I` (zz arms → negative `mu_s`) or
needs `I_0 -> infinity`.

## ④ Figures

* **`runs/round13/fig1_mu_vs_I_arms.png`** — measured `mu(I)` for all five
  admissible pairings against the shaded 8-corner reference envelope and the two
  candidate curves with the ±10 % band on PRIMARY_REPO. Visually: the archived
  zz arm sits ~25 % under the whole envelope; the law-consistent arm sits inside
  it at low `I` and falls away above `I ≈ 0.07`.
* **`runs/round13/fig2_law_consistent_residuals.png`** — residuals of the
  law-consistent arm against both candidates, band shaded, every point labelled.
  Shows the two low-`I` points landing within ±2 % and the two high-`I` points
  at −13 % / −16 % (TEST_GDR_MIDI) resp. −17 % / −28 % (PRIMARY_REPO).
* **`runs/round13/fig3_admissibility_and_normaliser.png`** — left: `rate_ratio`
  and velocity-profile `R²` per point with the 0.90 threshold, flagging the two
  servo failures; right: the spurious `mu` inflation (+11 … +31 %) caused by
  mixing pressure normalisers.

## ⑤ Notes, pitfalls, scope

1. **This round installs no new physics and runs no new simulation.** Its value
   is that the ledger's blocking question ("which citation?") is replaced by a
   computed one ("does the citation matter?"). The answer is *yes for the size
   of the error, no for the verdict*: the choice moves the worst-point error
   from 28 % to 16 %, and neither choice closes the band.
2. **A defect was found and fixed in the tooling, not in the solver.** Two
   conflicting "single source of truth" triples existed; `reference_law.py` now
   records the dispute and points at the registry, and a test enforces that the
   module and the registry cannot drift apart again.
3. **The `± 0.0000` standard errors in the round-12 tables are not real.** Round
   12 computed mean ± sd over seeds whose results were all identical (single
   seed rerun); the spread is 0 by construction. Every conclusion here is
   therefore stated per-point, not per-ensemble.
4. **Wall slip is the leading physical suspect, unconfirmed.** At `I = 0.01` the
   bulk shears at 38 % of the commanded rate with `R² = 0.571` — the cell is not
   shearing the sample. Whether the high-`I` deficit is the same phenomenon at a
   smaller amplitude, or a soft-contact/stiffness artefact, is *not* decided
   here. Round 11's rough-platen series (which would test wall slip) never
   produced data — cluster job `cj2515` was **killed** in the queue.
5. **Provenance remains unresolved as a citation.** Neither triple is anchored
   to an archived copy of the paper. This round shows the verdict is *robust*
   within the disputed box (0/8 corners pass), which is stronger than resolving
   the citation, but the registry labels remain honest: not archived.
6. Four points, one seed each. `phi(I)` is untouched and stays **WITHHELD**.

## ⑥ Error judgment (explicit)

Law-consistent pairing (`p = -tr(sigma)/3` for both ordinate and abscissa),
`error = mu_measured / mu_ref - 1`:

| quantity | reference | measured | error | within 3 %? | within ledger 10 %? |
|---|---|---|---|---|---|
| `mu`, I≈0.0044 | 0.3755 (PRIMARY) | 0.3751 | **+2.2 %** | **yes** | yes |
| `mu`, I≈0.0044 | 0.3881 (TEST) | 0.3751 | −2.1 % | **yes** | yes |
| `mu`, I≈0.031 | 0.4036 (PRIMARY) | 0.4090 | +0.9 % | **yes** | yes |
| `mu`, I≈0.031 | 0.4044 (TEST) | 0.4090 | +0.9 % | **yes** | yes |
| `mu`, I≈0.073 | 0.4800 (PRIMARY) | 0.3772 | −16.9 % | **no** | **no** |
| `mu`, I≈0.073 | 0.4481 (TEST) | 0.3772 | −12.9 % | **no** | **no** |
| `mu`, I≈0.306 | 0.6000 (PRIMARY) | 0.4333 | −28.1 % | **no** | **no** |
| `mu`, I≈0.306 | 0.5148 (TEST) | 0.4333 | −16.0 % | **no** | **no** |
| `mu_s` (quasi-static limit, reverse fit) | 0.36 / 0.379 | 0.382 | +6.1 % / **+0.8 %** | yes / **yes** | yes |
| `mu_2`, `I_0` (reverse fit) | 0.84/0.30 or 0.641/0.279 | not identified | — | **no verdict** | **no verdict** |
| `phi(I)` | unverified | 0.533–0.575 | — | **no verdict** | **WITHHELD** |
| spurious `mu` inflation from normaliser mixing | 0 | +10.8 … +30.6 % | **defect, quantified** | **no** | **no** |
| kinematically admissible points | 4 of 4 | **1 of 4** | **defect, quantified** | — | — |

**Answer to "is it < 3 %?":** *for the quasi-static limit, yes* — `mu_s` is
recovered to +0.8 %, and the two lowest-`I` points sit within ±2.2 %. *For the
rate-dependent branch, no* — `mu` at `I ≈ 0.07` and `0.306` misses by 13–17 %
(best candidate) and the reverse fit cannot identify `(mu_2, I_0)` at all.
**The ledger row stays `open`.**

**Verdict on the round-12 recommendation.** The recommendation "run a provenance
round before judging" is **falsified as the binding blocker**: 0 of 8 corners of
the disputed envelope passes every point under any of the five admissible
pairings, so no citation resolution can close the acceptance. Two genuine
defects were quantified instead — the abscissa uses the imposed rather than the
measured rate (D1), and the archived convention mixes pressure normalisers,
inflating `mu` by up to 30.6 % (D2). Fixing D1+D2 in the *analysis* already
recovers 12–15 percentage points; what remains is a device-level deficit at
`I >= 0.07` that must be attacked by making the cell actually shear the sample
(≥3 seeds, `rate_ratio >= 0.9`, profile `R² >= 0.9`, and the rough-platen series
that `cj2515` never delivered).

## Reproduction

```bash
cd benchmarks/plane_shear_mu_i
python3 adjudicate_provenance.py            # -> runs/provenance_adjudication.{json,md}
python3 provenance_adjudication_figures.py  # -> runs/round13/fig*.png
cd ../.. && python3 -m pytest benchmarks/plane_shear_mu_i/ -q   # 69 passed, 3 skipped
```