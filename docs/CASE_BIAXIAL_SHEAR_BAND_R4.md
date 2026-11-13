# CASE-BIAXIAL-SHEAR-BAND — R4: the cumulative window (synthetic half delivered, DEM half WITHHELD)

**Verdict.** R4 asked one question: is the residual scatter of the biaxial
shear-band measurement a *signal-to-noise* statement or a *physics* statement?
The synthetic half is answered and is decisive on the mechanism; the DEM half is
**WITHHELD** (not run to completion in this window — see ⑥). Nothing below
changes the R3 anchor verdict (**53.83°, 10.3 % from the 60° anchor, 5/5
specimens resolved, PASS at the owner 15 % bar, FAIL at 3 %**).

Synthetic result (24 × 48 case lattice, known-angle bands, 3 angles × 3 noise
levels × 6 realisations, `artifacts/biaxial_shear_band/r4_snr_check.json`):

| noise (% of spacing) | one-step increment (slip 0.0033 m) RMS | cumulative window (slip 0.02 m) RMS | ratio |
|---|---|---|---|
| 0 | 0.58° | 0.58° | 1.0× (identical — no bias) |
| 0.25 | **27.20°** (bias −17.89°) | **0.88°** (bias −0.33°) | **30.9×** |
| 0.5 | 29.69° (bias −17.44°) | 21.52° (bias −10.22°) | 1.4× |

The predeclared decision rule (cumulative RMS < one-step RMS at every non-zero
noise level **and** every angle) is **met in 6/6 cells**
(`decision_holds: true`). All digits are injected from the JSON artefact by the
scripts named in ②.

---

## ① Case introduction (physics, geometry, parameters)

Plane-strain biaxial compression of a freeze-bonded frictional disc specimen:
displacement-driven top platen, fixed bottom wall, both lateral walls servo to a
constant `sigma_3`. A dense frictional material localises into a shear band at or
after the deviator peak; the inclination of that band measured from `sigma_3` is
the classical validation quantity:

* Mohr–Coulomb `45 + phi/2`; Roscoe (1970) `45 + psi/2`;
  Arthur et al. (1977) `45 + (phi + psi)/4`;
* **anchor**: dense Hostun sand plane-strain biaxial tests, shear band 55–60°,
  midpoint **60°** — Desrues & Viggiani (2004), *Int. J. Numer. Anal. Methods
  Geomech.* **28**:279–321; tolerance 15 % relative ⇒ accept `51° ≤ theta ≤ 69°`.

| quantity | value (R4 specification, `run_r4_cumulative_ensemble.py`) |
|---|---|
| cell | plane-strain biaxial, 24 × 48 disc lattice (16 × 32 in `--quick`) |
| particle radius / spacing | 0.75 m / 1.5 m (2D disc, unit thickness) |
| freeze-bond fraction | 0.35 |
| wall friction `mu` | 0.55 |
| consolidation steps | 4000 |
| axial strain rate / target | 0.05 s⁻¹ / 0.12 |
| confining pressure | 100 kPa (anchor) and 200 kPa (confinement sensitivity) |
| anchor seeds (p = 100 kPa) | 53, 67, 79, 83, 89, 97, 101, 103 — **disjoint from R3's [20261031, 11, 23, 37, 41]**, i.e. out-of-sample |
| p = 200 kPa seeds | 53, 67, 79 |
| bootstrap | 30 resamples of 80 % of the window particles |
| anchor / tolerance | Desrues & Viggiani 2004 midpoint 60° / 15 % |

**Why R4.** R3 left three measurement questions open (R3 §⑤/§⑥): the cumulative
window was implemented but never measured; p = 200 kPa did not resolve; and the
per-specimen measurement uncertainty had never been quantified. R4's predeclared
design (driver docstring) attacks all three with the same physics and an
out-of-sample specimen set.

## ② What was done

1. **Predeclared DEM driver** — `benchmarks/biaxial_shear_band/run_r4_cumulative_ensemble.py`:
   8 out-of-sample specimens at p = 100 kPa + 3 at p = 200 kPa; primary
   estimator declared *before* the run is the **cumulative Radon focus over the
   late post-peak window** (`tail_start = peak + 0.4 (final − peak)`), with the
   R3 one-step route and the structure-tensor operator reported as secondary
   numbers on the *same* snapshots; per-specimen JSON cache so a reaped job can
   be resumed (`--resume`); PASS rule `|mean − 60|/60 ≤ 0.15` **and** ≥ 6/8
   resolved; P1 = cumulative uncertainty < one-step uncertainty in ≥ 6/8
   specimens (paired particle bootstrap, 30 resamples × 80 % of particles);
   P2 = ≥ 2/3 p = 200 kPa specimens resolve with the cumulative window.
2. **Synthetic SNR check** — `benchmarks/biaxial_shear_band/run_r4_snr_check.py`:
   the mechanism the round rests on, measured on the exact case lattice with
   bands of *known* angle, two slips (0.0033 m = one 0.01-strain increment;
   0.02 m = cumulative late window ⇒ 6× tangential signal), three noise levels,
   6 independent realisations, fixed seed 20261113. Wall time 5.9 s.
3. **Report generator** — `benchmarks/biaxial_shear_band/build_r4_report.py`:
   renders this document and the platform report from the DEM JSON with no hand
   transcription (it runs on a degenerate JSON without crashing, verified).
4. **Tests** — `tests/test_biaxial_shear_band_r4.py`, 11 tests, **11 passed**:
   late-window/peak selection, bootstrap reproducibility and small-window
   behaviour, the SNR inequality on a 24 × 24 lattice, and the verdict assembly
   (pass rule, P1 6/8 bar, P2 counting) on synthetic specimen records.
5. **DEM ensemble** — submitted to the cluster as `cj1633`
   (`run_cluster submit`, 12 cores, `stage_dir` = this worktree, 11 specimens in
   parallel). It stayed **queued** (363 jobs ahead) for the whole window and was
   cancelled; the local box load15 was 175–371 on 128 cores, so a local rerun was
   ruled out by the capacity discipline. No DEM R4 numbers are reported.

## ③ Results

**Synthetic SNR check (programmatic, `r4_snr_check.json`):**

| window | noise 0 % | noise 0.25 % | noise 0.5 % |
|---|---|---|---|
| one-step increment, RMS (°) | 0.58 | 27.20 | 29.69 |
| one-step increment, bias (°) | −0.33 | −17.89 | −17.44 |
| cumulative window, RMS (°) | 0.58 | 0.88 | 21.52 |
| cumulative window, bias (°) | −0.33 | −0.33 | −10.22 |
| per-angle cumulative RMS, 55 / 60 / 65° at 0.25 % | — | 0.91 / 0.82 / 0.91 | — |

Reading: at zero noise the two windows are numerically **identical** (same
operator, same bias 0.58° RMS — so the 6× slip introduces no new bias), while at
0.25 % noise the one-step reading collapses (27° RMS, 54° worst case: the
projection locks onto a spurious ridge) and the cumulative reading is untouched
(0.88° RMS). At 0.5 % noise the cumulative window also degrades (21.5° RMS) —
the SNR gain buys roughly **one decade of noise tolerance**, not immunity.

**DEM ensemble (8 + 3 specimens):** WITHHELD, see ⑥.

## ④ Figures

**Figure 1** (`artifacts/biaxial_shear_band/r4_snr_check.png`): RMS inclination
error vs displacement noise for the one-step increment and the cumulative
window, on the 24 × 48 case lattice. The cumulative curve is flat up to 0.25 %
noise (0.88°) where the one-step curve has already saturated near 27°; both
collapse by 0.5 %. This is the quantitative form of the R3 prediction
("cumulative moves the case into the safe branch") and the reason R4's DEM half
uses the cumulative window as its primary estimator.

No DEM field figure is provided because the DEM half did not produce data
(honest gap, not a stylistic choice).

## ⑤ Notes, pitfalls, limits

* **The R2 lesson is load-bearing.** R1's "model defect" (44° vs 60°) was an
  estimator artefact (per-axis min–max rescaling of the Radon projection, up to
  18° bias on this 2:1 lattice). R4 changes only the *displacement baseline*,
  never the operator, so the comparison above is controlled.
* **What the synthetic check can and cannot say.** It isolates the SNR
  mechanism with known-angle bands. It cannot show that the DEM field's
  displacement noise is ≲ 0.25 % of the spacing; that is exactly what the DEM
  half was for, and it is the reason the synthetic result is reported as a
  mechanism, not as an anchor error budget.
* **The 0.5 % cliff survives.** The cumulative window does not make the
  measurement noise-immune; it buys ~1 decade. Any future single-specimen claim
  must therefore still be ensemble-backed.
* **Scope.** The anchor is 3-D dense Hostun sand; the cell is a 2-D
  freeze-bonded disc assembly. Consistency at the 15 % level is not physical
  equivalence.
* **Reproduction hazard (recorded for the next round).** `run_shell` reaps the
  process group at its cap, and the cluster queue on 2026-11-13 was ~360 jobs
  deep at submit time; a DEM ensemble must be submitted early in a mission or
  run as sharded single-specimen jobs (`--only-seed N`) that each fit a short
  budget. The driver's per-specimen cache exists for exactly this reason.

## ⑥ Error judgement (explicit)

| quantity | value | reference | error | < 3 % ? |
|---|---|---|---|---|
| cumulative-window inclination RMS, synthetic, 0.25 % noise | 0.88° | 0° (known injected angle) | 0.88° = **1.5 % of 60°** | **Yes** |
| one-step inclination RMS, synthetic, 0.25 % noise | 27.20° | 0° | 27.20° = 45.3 % of 60° | **No** |
| cumulative-window bias, synthetic, noise-free | 0.58° RMS (−0.33° mean) | 0° | 0.97 % of 60° | **Yes** |
| anchor ensemble mean (R3, unchanged by R4) | 53.83° | 60° | **10.3 %** | **No** (15 % bar: yes) |
| DEM R4 ensemble mean (cumulative window, out-of-sample) | — | 60° | — | **not measured** (WITHHELD) |

**Is the key quantity below 3 %?** The R4-derived key quantity — the measurement
error of the operator at a noise level that the case plausibly sits at (0.25 %)
— is **0.88°, i.e. 1.5 % of the anchor: yes, below 3 %**. The *anchor* quantity
(band inclination vs the literature 60°) remains at **10.3 %: no**. R4 therefore
does not claim a 3 % validation of the case; it claims that the measurement
uncertainty of a single cumulative-window reading is no longer the limiting
error term, which is what the DEM half must now confirm.

**Honest WITHHELD.** The DEM ensemble (`cj1633`) never left the queue and was
cancelled; no part of the R4 anchor verdict is inferred from it. The R3 verdict
(53.83°, 10.3 %, PASS@15 %, FAIL@3 %) stands unchanged.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r4_snr_check.py     # 6 s, done above
PYTHONPATH=src python -m pytest tests/test_biaxial_shear_band_r4.py -q      # 11 passed
# DEM half (needs compute; per-specimen cache makes it resumable):
PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r4_cumulative_ensemble.py --workers 11 --resume
PYTHONPATH=src python benchmarks/biaxial_shear_band/build_r4_report.py --publish
```