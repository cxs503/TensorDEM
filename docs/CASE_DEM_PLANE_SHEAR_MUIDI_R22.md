# CASE-DEM-PLANE-SHEAR-MUIDI — Round 22: the reference law, finally read out of the papers

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池 · 平面剪切 mu(I) 流变律标定)
**References:** da Cruz, Emam, Prochnow, Roux & Chevoir, *Phys. Rev. E* **72**:021309
(2005), arXiv:cond-mat/0503682; GDR MiDi, *Eur. Phys. J. E* **14**:341 (2004),
arXiv:cond-mat/0312502. **Tolerance:** < 10 % on `mu(I)` and `phi(I)`.
**Round type:** bibliographic + re-judgement round — **zero new simulation**.
All measured numbers are the four archived servo-cell points
(`runs/full_cell/point_*.json`, round 14); all reference numbers are parsed out
of archived copies of the two papers.
**Tooling:** `reference_archive.py` (archive + parsers), `build_reference_archive.py`,
`analyse_r22.py`, `figures_r22.py`, `test_reference_archive_r22.py` (12 tests).
**Artefacts:** `runs/provenance_archive/{sources.json,*_excerpts.txt,*_numbers.txt}`,
`runs/round22/{analysis_r22.json,tables_r22.md,fig_r22_*.png}`.

---

## ① Case introduction

A plane-shear cell of 294 spheres (`d = 10 mm`, `rho_p = 2500 kg/m3`) between a
servo-controlled rough platen and a frictional floor, sheared under a
constant confining stress `P = 62.5 kPa` at four inertial-number targets
`I = 0.01, 0.03, 0.1, 0.3`. The ledger asks whether the measured effective
friction `mu = tau/P` and solid fraction `phi` reproduce the `mu(I)` / `phi(I)`
laws of the two cited papers within 10 %.

Rounds 1–21 built the device, the servo, the estimators and the whole-gap
abscissa, and left the ledger **FAIL / WITHHELD** for two reasons:

1. two mutually inconsistent Jop-law triples live in the repository —
   `reference_law.py` uses `(mu_s, mu_2, I_0) = (0.36, 0.84, 0.30)` ("quoted in
   rounds 1–5") and `tests/test_mu_i_rheology.py` uses `(0.379, 0.641, 0.279)`
   ("Table 1 of da Cruz et al. 2005") — and **neither is anchored to a reading
   of any paper** (round 13 carried the 8-corner envelope as a stop-gap);
2. the measurement itself sat 13–30 % off every corner of that envelope.

Round 22 attacks reason 1 directly: it archives the papers and reads them.

## ② What was done

1. **Archive** (`reference_archive.py`, `build_reference_archive.py`): both
   papers are fetched from arXiv, the PDF identity is pinned by SHA-256, the
   text is extracted, and verbatim excerpts are **sliced mechanically** around
   analyst-chosen anchor strings (`EXCERPT_ANCHORS`). A per-source inventory of
   every decimal literal in the text is committed too
   (`*_numbers.txt`), so an offline reader can check "is this number actually
   printed in the paper?" without needing the PDF.
2. **Parse, do not transcribe** (`parse_dacruz`, `parse_gdr_midi`): the
   constants are capture groups of regular expressions applied to those
   excerpts. A build-time failure is raised if an anchor string is not found —
   so a silent drift of the archived text cannot pass as a constant.
3. **Model the law in the form the paper publishes it** (`DaCruzLaw`):
   `mu(I) = mu_min + b I` is quoted **only** up to `I_sat`; above it
   `mu()` returns `None` and the caller must ask for `mu_extrapolated()`
   explicitly, so no module can compare against an extrapolation by accident.
4. **Re-judge** (`analyse_r22.py`): the four archived points are compared to
   (a) the archived da Cruz law, (b) `PRIMARY_REPO`, (c) `TEST_GDR_MIDI`,
   across five stress ordinates and two `phi` ordinates; monotonicity of the
   measured `mu(I)` is checked; and the Jop form is least-squares fitted to the
   *archived law* to test whether either in-repo triple could be a faithful
   transcription of it (`fit_jop_to_law`).
5. **Gate it** (`test_reference_archive_r22.py`, 12 tests): every constant used
   by the model must be a decimal literal printed in the paper; the two triples
   must be absent; the re-judged errors must recompute bit-for-bit from the
   archived inputs; the round must not claim a pass.

## ③ Results

### Table A — the archived reference law (parsed, zero hand-copy)

| constant | value | where in the paper |
|---|---|---|
| `mu_min` | 0.220 | da Cruz 2005 §IV B Eq. (14), "for mu ≠ 0" |
| `b` | 1.000 | same equation |
| `I_sat` | 0.200 | §IV B: "tends to saturate for I ≥ 0.2" |
| `nu_max` | 0.810 | §IV A Eq. (13), at `mu_p = 0.4` |
| `a` | 0.300 | same equation |
| `mu_min` (frictionless) | 0.110 | §V E: the `mu_p = 0` case is singular |

`mu(I) = 0.22 + 1.00 I` and `phi(I) = 0.81 − 0.30 I`, both valid for
`I ≤ 0.2`. GDR MiDi 2004 publish no fitted triple at all: their packing
fraction runs `0.85 → 0.80` over `I = 0 → 0.1`, and `mu_eff(I)` is described
qualitatively (rising from `mu_S`, saturating). **The `mu(I)` triple is a
Jop-form fit to this body of data, not a table in either cited paper.**

### Table B — the citation defect (falsified, not "unverified")

| source | `0.36` | `0.84` | `0.30` | `0.379` | `0.641` | `0.279` |
|---|---|---|---|---|---|---|
| da Cruz 2005 (90 174 chars, 69 decimals) | absent | **printed as phi** | **printed as the dilatancy slope `a` = 0.3** | absent | absent | absent |
| GDR MiDi 2004 (100 960 chars, 74 decimals) | absent | absent | absent | absent | absent | absent |

The defining `mu_s` of each triple (0.36 / 0.379) is printed **nowhere** in
either paper, and the pytest-suite triple attributed to "Table 1 of da Cruz et
al. 2005" is absent in all three of its numbers. Both attributions are
therefore **mis-attributions**, and round 13's "envelope of two readings" was
an envelope of two different non-transcriptions.

### Table C — can the Jop form represent da Cruz at all? (reconciliation)

| object | `mu_s` | `mu_2` | `I_0` | max err vs the archived law |
|---|---|---|---|---|
| LSQ Jop fit to da Cruz's own law, `I = 1e-3..0.2` | 0.215 | 1.080 | 0.678 | **2.1 %** |
| `PRIMARY_REPO` | 0.36 (+67 %) | 0.84 (−22 %) | 0.30 (−56 %) | **63.6 %** |
| `TEST_GDR_MIDI` | 0.379 (+76 %) | 0.641 (−41 %) | 0.279 (−59 %) | **71.9 %** |

The Jop form is flexible enough to reproduce da Cruz to 2 % — with a triple
that resembles neither in-repo triple. The two triples are 63 % / 72 % away
from the archived law in `mu` over the ledger's own `I` range: they cannot be
readings of these papers.

### Table D — re-judgement of the four archived points

Ordinate `mu_full_zz_final` (whole-cell virial stress), `I_bulk` = measured bulk rate:

| `I_bulk` | in law validity | measured `mu` | vs da Cruz law | vs `PRIMARY_REPO` | vs `TEST_GDR_MIDI` |
|---|---|---|---|---|---|
| 0.0059 | yes | 0.2869 | +27.0 % | −22.3 % | −25.4 % |
| 0.0350 | yes | 0.3998 | **+56.8 %** | −2.5 % | −2.1 % |
| 0.0736 | yes | 0.3394 | +15.6 % | −25.3 % | −21.8 % |
| 0.3454 | **no** (`I > I_sat`) | 0.3552 | −37.2 % (extrapolation, not judged) | −42.4 % | −32.2 % |

Pass/fail over the three points inside the law's validity, all five stress
ordinates:

| candidate | within 10 % | max abs err | verdict |
|---|---|---|---|
| da Cruz 2005 archived law | 0/3 (`mu_full_zz`), 2/3 at best (`mu_slab_both`) | +56.8 % | **FAIL** |
| `PRIMARY_REPO` | 1/3 | +25.3 % | FAIL |
| `TEST_GDR_MIDI` | 1/3 | +25.4 % | FAIL |

`phi`: 0/3 within 10 % for `phi_slab` (max 21.5 %) and 0/3 for `phi_bed_steady`
(max 34.0 %). The archived dilatancy law gives `phi(0.01) = 0.807`, the
measured bed packs at `≈0.53` — a factor-1.5 gap that is **not** a law error
but a definition error (see ⑤).

Monotonicity: the measured `mu(I)` is **not** monotone in either ordinate —
`(0.006, 0.287) → (0.035, 0.400) → (0.074, 0.339)` is a −15.1 % step
(`mu_full_zz`), −19.7 % at the wall, −19.0 % on the slab. Every `mu(I)` law is
monotone increasing, so the 10 % acceptance cannot be met by *any* choice of
constants while that dip exists: it is the same size as the tolerance band.

## ④ Figures

`runs/round22/`:

| file | caption |
|---|---|
| `fig_r22_mu_I.png` | `mu(I)` over `I = 1e-3..0.35`: the archived da Cruz law (solid, with its ±10 % band and its validity limit `I_sat = 0.2`), the same fit extrapolated (dashed, not judged), the two in-repo triples, and the four measured points. The measured sequence is visibly non-monotone and sits above the archived law at `I = 0.035`. |
| `fig_r22_phi_I.png` | `phi(I)`: archived dilatancy law `0.81 − 0.30 I`, GDR MiDi's `0.85 → 0.80` packing path, and the two measured `phi` ordinates, which lie far below both — the `phi` definition mismatch, drawn. |
| `fig_r22_attribution.png` | (a) per-point `|error|` of every candidate against the 10 % band (hatched = point outside the law's validity); (b) the Jop form fitted to da Cruz's own law stays inside ±3 %, while both in-repo triples leave the band from `I ≈ 0.004` — the graphical form of Table C. |

## ⑤ Note points (pitfalls, limits, scope)

* **The `phi` comparison is not yet a comparison.** da Cruz measure 2-D disks
  (`nu_max = 0.81` at `mu_p = 0.4`); our cell is 3-D spheres and the archived
  ordinates are `phi_slab` (sheared slab) and `phi_bed_steady` (bed volume).
  A 0.53 vs 0.81 gap is dominated by those definitional differences, so the
  `phi` row stays **WITHHELD** — round 22 changes its *provenance* status, not
  its verdict, and does not pretend the numbers are commensurable.
* **`I = 0.3` is outside the archived law's validity.** The ledger's own
  acceptance range (`I = 1e-3..0.3`) runs past the point where the source's
  linear fit breaks down; the archived law is therefore only judged on
  `I ≤ 0.2`, and the fourth point is reported as an extrapolation. This is a
  defect of the acceptance text, not of the simulation.
* **One seed per point, four points.** Nothing here quantifies seed spread; the
  −15 % dip between `I = 0.035` and `0.074` is one realisation each and could
  itself be noise. It is reported because *until it is resolved* the 10 %
  criterion is unfalsifiable in either direction.
* **arXiv copies.** The published PRE / EPJ E versions are not open access;
  the constants are read from the arXiv v1 manuscripts whose SHA-256 is
  recorded in `sources.json`. A future reader can verify the identity of the
  exact text this round used, but not that the journal version is
  byte-identical.
* **Number-presence is decided by exact decimal-token membership**, not by
  substring search: `0.84` and `0.3` *are* printed in da Cruz (as a packing
  fraction and as the dilatancy slope), which is exactly the false positive
  that a naive grep would produce. `test_the_shared_numbers_are_a_documented_false_positive`
  pins that interpretation.
* Anchor strings are matched against whitespace-normalised text; if the arXiv
  copy were re-typeset the build would raise rather than silently mis-parse.

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| reference-law provenance | a citable reading of the paper | da Cruz Eq. (13)/(14) parsed from an SHA-256-pinned copy | **RESOLVED** (this round) |
| in-repo triples | transcription of the cited papers | 0.36 / 0.379 (`mu_s`) printed nowhere; pytest triple absent entirely | **MIS-ATTRIBUTED** |
| `mu(I)` vs archived da Cruz law (`I ≤ 0.2`) | < 10 % | 0/3 points within 10 % on `mu_full_zz`; max +56.8 % | **FAIL** |
| `mu(I)` vs either in-repo triple | < 10 % | 1/3 within 10 % (`I = 0.035` only); max 25.3 % / 25.4 % | **FAIL** |
| `phi(I)` vs archived dilatancy law | < 10 % | 0/3; max 21.5 % (`phi_slab`) / 34.0 % (`phi_bed_steady`) | **WITHHELD** (definitions not commensurable) |
| monotonicity of measured `mu(I)` | strictly increasing | one decreasing step, −15.1 % … −19.7 % | **FAIL** |

**Is any key quantity within 3 % of its reference? No.** The archived reference
law is reproduced by a Jop-form triple to 2.1 % (Table C), but that is a
statement about the *law's representation*, not about the simulation: no
measured `mu` is within 3 % of anything, the best single point being −2.5 %
against `PRIMARY_REPO` at `I = 0.035` while its neighbours miss by 20–25 %.

**What changed for the ledger.** The blocker is no longer bibliographic. Round
13 could argue that an unknown envelope of readings might rescue the
acceptance; that argument is now closed — the correct yardstick is
`mu = 0.22 + 1.00 I`, `phi = 0.81 − 0.30 I` (`I ≤ 0.2`), and against it the
cell misses by 16–57 % with a non-monotone `mu(I)`. The remaining work is a
measurement/definition problem (contact friction calibration, whole-gap vs slab
estimator, `phi` definition), not a citation problem.

## ⑦ Recommended next steps

1. **Recalibrate the contact friction against the archived law.** da Cruz use
   `mu_p = 0.4` and obtain `mu_min ≈ 0.22`; our device runs `mu_p = 0.5,
   mu_r = 0.05` and gives `0.29–0.40` at the same `I`. A `mu_p` ladder judged
   against `0.22 + 1.00 I` (not against `0.36 + …`) is the missing measurement.
2. **Resolve the `mu(I)` dip** (`I = 0.035 → 0.074`, −15 %) with 3–5 seeds per
   point before any further law fitting: it is the same size as the tolerance.
3. **Fix the `phi` definition** (bed volume vs sheared slab vs 3-D packing) so
   that `phi(I)` can be judged at all; compare 3-D spheres with a 3-D source
   (GDR MiDi's `0.85 → 0.80` path) rather than with 2-D disks.
4. **Amend the ledger acceptance text**: name the source equation
   (`da Cruz 2005 Eq. 14`, valid `I ≤ 0.2`) and drop `I = 0.3`, or state that
   the saturating branch is what is judged there.