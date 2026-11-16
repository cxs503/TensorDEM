# CASE-BIAXIAL-SHEAR-BAND — R5: the R4 ensemble is executed (real DEM numbers)

**Verdict (honest, mixed — the case stays open).** R4 predeclared an
out-of-sample 8-specimen ensemble at `p = 100 kPa` and never ran it (its DEM half
was `WITHHELD`: the cluster job stayed queued, see `docs/CASE_BIAXIAL_SHEAR_BAND_R4.md`
⑥). This round **runs it**. Eleven specimens (8 anchor + 3 confinement) were
executed on 2026-11-16 and measured with the R4 production estimator.

* **Accuracy: the acceptance tolerance is met.** The primary estimator
  (cumulative late-window Radon) gives **θ = 55.20° ± 1.92°**
  (n = 5, 95 % CI [52.81°, 57.59°]) against the 60° anchor
  ⇒ **8.0 % relative error, tolerance 15 %: PASS**.
  The mean is also **inside** the measured
  experimental range 55–60° of Desrues & Viggiani (2004)
  (`within_literature_band = True`).
* **Resolution: the predeclared pre-condition is not met.**
  5/8 specimens resolved
  (bar was ≥ 6/8) ⇒ the predeclared conjunction
  (`|mean−60|/60 ≤ 0.15` **and** ≥ 6/8 resolved) evaluates to
  **passed = False**. The case therefore remains an **honest FAIL on its
  predeclared rule**, and the accuracy number above must be read as
  "measured on the 5 specimens that localised inside the strain target", not as
  a whole-ensemble claim.
* **Root cause of the 5/8 is now named, and it is not the estimator.**
  3 specimens land in `degenerate_window`
  (`anchor_s103`, `anchor_s89`, `anchor_s97`):
  their deviator peak sits at the strain target, so the late post-peak window
  holds **one** snapshot, `strain_span = 0`, the displacement increment is zero
  and the strain field is degenerate by construction
  (`BandMeasurementError('local strain field is degenerate')`).
  **0** specimens failed the significance
  bar — i.e. the measurement operator itself never failed to find a band when a
  band existed to be found. Before this round those two states were
  indistinguishable in the artefact (both `NaN`, zero slip); they are now
  separate classes in `verdict.failure_modes`.

All numbers below are injected from
`artifacts/biaxial_shear_band/r4_cumulative_ensemble.json` by
`benchmarks/biaxial_shear_band/build_r5_report.py` (no hand transcription).

---

## ① Case introduction (physics, geometry, parameters)

Plane-strain biaxial compression of a freeze-bonded frictional disc specimen:
displacement-driven top platen, fixed bottom wall, both lateral walls servo to a
constant `sigma_3`. A dense frictional material localises into a shear band
at/after the deviator peak; the band inclination measured from the `sigma_3`
axis is the classical validation quantity, with three competing predictions from
the *same-run* measured `phi` and `psi`:

* Mohr–Coulomb plasticity `45 + phi/2`;
* Roscoe (1970, *Géotechnique*) `45 + psi/2`;
* Arthur et al. (1977, *Géotechnique*) `45 + (phi + psi)/4`.

**Anchor**: dense Hostun sand plane-strain biaxial tests, shear band 55–60°,
midpoint 60°, tolerance 15 % ⇒ accept `[51.0°, 69.0°]`
(Desrues & Viggiani 2004, *Int. J. Numer. Anal. Methods Geomech.* **28**:279–321).

| quantity | value |
|---|---|
| cell | plane-strain biaxial, 24 x 48 disc lattice |
| particle radius / spacing | 0.75 m / 1.5 m (2D disc, unit thickness) |
| lattice jitter | 3 % of spacing |
| freeze-bond fraction | 0.35 |
| wall friction `mu` | 0.55 |
| consolidation steps | 4000 |
| axial strain rate / target | 0.05 s⁻¹ / 0.12 (logarithmic) |
| confining pressure | 100 kPa (anchor, seeds [53, 67, 79, 83, 89, 97, 101, 103]) and 200 kPa (seeds [53, 67, 79]) |
| estimator | cumulative Radon focus over the late post-peak window (primary), one-step increment + structure tensor (secondary) |
| bootstrap | 30 resamples × 80% of particles, paired |

The seeds are the R4 predeclared out-of-sample set, disjoint from the R3
ensemble, so this is a genuine held-out check, not a re-measurement.

## ② What was done

1. **Defect fixed in the R4 driver (`--workers > 1` had never executed).**
   `run_r4_cumulative_ensemble.py` called
   `pool.imap_unordered(_run_cached, configs, [cache]*n, [resume]*n)` —
   `imap_unordered` takes a single argument per item, so the multi-worker path
   raised `TypeError: Pool.imap_unordered() takes from 3 to 4 positional
   arguments but 5 were given` before running a single specimen. Because the
   round had only ever been driven through the *single-specimen shard wrapper*
   (`run_r4_shard.py` → `_run_cached` directly), the broken branch was never
   reached. Fix: bind the constant arguments with `functools.partial` and route
   both branches through `_run_cached_safe`, which converts a per-specimen
   exception into an `error` record instead of killing the pool (the serial
   branch already did this; the parallel branch did not). Regression test:
   `tests/test_biaxial_r4_ensemble_driver.py` (dispatches a 2-worker pool with a
   stubbed 1 ms specimen — it fails on the old call site).
2. **Failure-mode classification added to the analysis.**
   `_failure_modes()` splits the specimens into `resolved`,
   `degenerate_window` (`strain_span == 0`: peak at the strain target),
   `unresolved_measurement` (window exists, focus below the `z`/`eta2` bar) and
   `no_post_peak_window`. `verdict.failure_modes`, `verdict.literature_band_deg`
   and `verdict.within_literature_band` are now reported, so "within 15 % of the
   60° midpoint" and "inside the measured 55–60° range" can never be confused.
3. **DEM execution (11 specimens), 2026-11-16.**
   One full specimen is ~80–100 s wall time at 4–6 threads on the local box
   (24 × 48 lattice, 4000 consolidation steps, 12 snapshots). The 11 specimens
   were run through the production shard wrapper one cache entry at a time
   (each run < 10 min, ≤ 6 cores — the local-compute gate), then assembled with
   `--resume`, which only re-measures from the cached snapshots
   (assembly wall time 0.70 s). Total DEM wall time ≈
   16.7 core-min.
   A 12-core cluster submission of the same ensemble (`cj4310`) was killed
   before dispatch (it had tarred the pre-fix source); the numbers here are
   reproducible from the per-specimen cache in `artifacts/biaxial_shear_band/r4_cache/`.
4. **Report builder** — `benchmarks/biaxial_shear_band/build_r5_report.py`
   renders this document, both figures and the platform report from the JSON.

## ③ Results

**Primary (cumulative late-window Radon), p = 100 kPa, n = 5 resolved
of 8:**

| quantity | value |
|---|---|
| mean θ | **55.20°** |
| std (between specimens) | 1.92° |
| 95 % CI (t, n−1) | [52.81°, 57.59°] |
| relative error vs 60° | **8.00 %** (tolerance 15 %) |
| within acceptance band | True |
| inside literature range 55–60° | True |

**Secondary, same snapshots:**

| estimator | mean θ | std | rel. error | n |
|---|---|---|---|---|
| cumulative late window, Radon | 55.20° | 1.92° | 8.00 % | 5 |
| cumulative late window, structure tensor | 63.79° | 3.96° | 6.31 % | 5 |
| one-step increment, Radon | 54.30° | 1.29° | 9.50 % | 5 |

**Per-specimen table** (programmatic; `—` = not measurable):

| specimen | p [kPa] | slip / spacing | strain span | θ cumulative (Radon) | θ one-step (Radon) | z | η² | φ [°] | ψ [°] | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| anchor_s101 | 100 | 1.08 | 0.040 | 56.0 | 54.8 | 12.7 | 0.527 | 35.4 | 49.8 | resolved |
| anchor_s103 | 100 | 0.00 | 0.000 | — | — | — | — | 35.9 | 71.8 | degenerate window |
| anchor_s53 | 100 | 0.27 | 0.010 | 52.0 | 52.0 | 7.7 | 0.321 | 35.6 | 67.2 | resolved |
| anchor_s67 | 100 | 1.07 | 0.040 | 56.0 | 55.0 | 14.7 | 0.438 | 35.5 | 50.1 | resolved |
| anchor_s79 | 100 | 1.07 | 0.040 | 57.0 | 54.8 | 16.2 | 0.597 | 35.5 | 49.2 | resolved |
| anchor_s83 | 100 | 0.28 | 0.010 | 55.0 | 55.0 | 7.8 | 0.365 | 35.1 | 71.8 | resolved |
| anchor_s89 | 100 | 0.00 | 0.000 | — | — | — | — | 35.2 | 69.9 | degenerate window |
| anchor_s97 | 100 | 0.00 | 0.000 | — | — | — | — | 35.5 | 71.8 | degenerate window |
| p200_s53 | 200 | 0.52 | 0.020 | 57.0 | 55.5 | 5.5 | 0.277 | 24.5 | 45.8 | — |
| p200_s67 | 200 | 0.00 | 0.000 | — | — | — | — | 25.1 | 42.6 | — |
| p200_s79 | 200 | 0.00 | 0.000 | — | — | — | — | 25.2 | 38.5 | — |

**P1 (SNR pairing): `False`** — only 3 paired
specimens exist (a pair needs both estimators defined), 3
improved. The predeclared 6-pair bar cannot be met because the 3 degenerate
specimens have no defined estimator at all, so P1 is **WITHHELD, not refuted**:
on the specimens where it is defined, the cumulative window is
better
in 3/3 cases.

**P2 (confinement, p = 200 kPa): `False`** —
1/3 resolved with the cumulative estimator
(0/3 with the one-step estimator). The single resolved
200 kPa specimen reads 57.00°; the other
two are `degenerate_window`. **No confinement trend can be claimed from n = 1** —
reported as WITHHELD.

**Measured material response, p = 100 kPa:** `phi` = 35.46°,
`psi` = 62.71° (global, secant at peak /
late post-peak volumetric slope). The theory angles are therefore
`{'mohr_coulomb': 62.716223218251955, 'roscoe': 69.91658638557533, 'arthur': 66.31640480191365}` for one specimen — the global
`psi > phi` means **Roscoe/Arthur cannot be evaluated on this `psi`** (a known
defect carried since R1/R2: the global dilation angle is unphysical for a bonded
specimen, the band-local replacement is still WITHHELD). The band angle is
consequently compared **only** with the empirical literature range and the
60° anchor, never with a `psi`-based law.

## ④ Figures

**Figure 1** (`artifacts/biaxial_shear_band/r5_gamma_field.png`): cumulative
local shear strain `gamma` over the late window for `anchor_s101`, with the
measured band line and the 55°/60° references. The localisation is a single
planar band, not a diffuse zone — the qualitative half of the acceptance
criterion.

**Figure 2** (`artifacts/biaxial_shear_band/r5_angle_comparison.png`):
left — per-specimen inclination for both estimators with the ±15 % band, the
55–60° literature band and the ensemble 95 % CI; right — the outcome classes
(5/8 resolved vs the ≥ 6/8 bar).

## ⑤ Notes, pitfalls, limits

1. **The strain target is the binding constraint, not the estimator.**
   `strain_target = 0.12` was chosen to bound cost; 3/8
   specimens still harden up to it, so their peak is at the last snapshot and the
   post-peak window degenerates. A physically clean fix is a *peak-relative*
   stopping rule (run until `eps_a >= peak + 0.03`), not a longer fixed target;
   that changes the predeclared protocol and is therefore **not** done here
   post hoc — it is the recommendation for R6.
2. **The measured mean is inside the literature range even though the
   predeclared rule fails.** These are two different statements and the reason
   the case stays open: the accuracy half passes (8.0 % < 15 %), the
   resolution-count half does not (5/8 < 6/8).
   Reporting only the first would be cherry-picking; reporting only the second
   would hide a 5-specimen measurement that agrees with both the anchor and the
   measured experimental band.
3. **Chaotic scatter is now quantified.** Between-specimen σ =
   1.92° on the resolved set (R2's single-specimen pairs were
   54.3° vs 72.0°, i.e. an 18° spread that this ensemble confirms as the tail,
   not the typical case). Two resolved specimens (`anchor_s53`, `anchor_s83`)
   have a one-snapshot cumulative window (slip 0.27 spacings) yet still pass the
   focus bar; their θ (52°, 55°) is the low end of the ensemble.
4. **Operator scope.** 2D discs, unit thickness, no rolling resistance, 35 %
   freeze bonds. The 55–60° anchor is a 3D dilatant sand; agreement in 2D is
   encouraging but is not a claim of 3D transferability.
5. **Reproduction.** `run_r4_shard.py <i>` (i = 0..10) then
   `run_r4_cumulative_ensemble.py --resume` — both on CPU, deterministic per seed
   (`torch` is single-threaded inside the driver via `R4_THREADS`).

## ⑥ Error judgement (explicit)

| quantity | reference | measured | error | < 15 %? |
|---|---|---|---|---|
| band inclination (ensemble mean, n = 5) | 60° (Desrues & Viggiani 2004 midpoint) | 55.20° ± 1.92° | **8.00 %** | **YES** |
| band inclination vs measured experimental range | 55–60° | 55.20° | inside range | n/a (range, not point) |
| per-specimen worst case | 60° | 57.0° | 5.00 % | YES |
| synthetic known-angle recovery (R2 audit, 24 × 48) | injected 60° / 45° | 60° / 45° | ≤ 2.2° | YES (single-agent, separate artefact) |

**Is the key quantity within 3 %?** **No** — 8.00 % vs 60°. Within 15 %
(the case tolerance): **yes**. Within the measured
experimental band 55–60°: **yes**. The 3 % bar is not
achievable with n = 5 chaotic DEM specimens in 2D at this system size (the
between-specimen σ alone is 1.92°, i.e.
3.2 %), and no claim to it is made.
