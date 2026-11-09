# ISO 19906 external-anchor ledger — ice-hull-propeller-threefield, exp-1

**Task** `paper-exp-ice-hull-propeller-threefield-1` · **repository** TensorDEM
**Artifact** `benchmarks/anchors/ice-hull-propeller-threefield.json` (script-injected;
no number in it is typed by hand)
**Regenerate** `python3 scripts/build_ice_hull_propeller_anchor.py`
**Harness** `src/tensordem/threefield/iso_anchor.py` · **Tests** `tests/test_threefield_iso_anchor.py` (19 cases)
**Figure** `docs/img/threefield_iso_anchor.png` — pressure-area law with the case calibre points (both panels stamped *candidate triple — NOT a verified anchor*)

---

## 1. What this experiment was asked to do

The study's research plan compares the DEM hull–ice / LBM race / thermal three-field
results with the ISO 19906:2019 pressure–area practice. Before any such comparison is
allowed, the external *anchors* must be (i) transcribed verbatim with table/clause/page,
(ii) converted to the case calibre by a script that can be recomputed, and (iii) screened
by dimensional-homogeneity and internal-consistency checks. This artefact reports that
verification, not the domain comparison (that is exp-2).

## 2. Verdict

| criterion | requirement | status | measured |
|---|---|---|---|
| C1 verbatim transcription | quote + table/clause/page + exact value in the artifact | **FAIL** | 0 / 2 anchors verified; ISO 19906:2019 primary text absent from the worktree |
| C2 conversion recomputable | chain landed as script, recompute residual < 1e-9 | PASS | worst relative residual **1.26e-16** (incl. an independent algebraic route `p = C_R (A/A_R)^ex`) |
| C3 dimensional + consistency | homogeneous equations, internal consistency machine-checked | PASS | 6 dimension equations, 6 consistency checks, all pass |
| C4 unverified not cited | candidate values only usable as cross-check references | PASS | claim gate refuses both registered anchors; `claims_permitted = []` |

**Overall: HONEST_FAIL** on C1, i.e. the research plan's blocking condition is not met:
the ISO anchors are **not** verified, so no comparison against them is admissible yet
(`downstream_gate.domain_comparison_allowed = false`).

Why an honest FAIL and not a plausible-looking pass: a three-number pressure–area law
(exponent, reference pressure, reference area) reproduces a smooth monotone curve for a
wide range of triples, so numerics cannot detect a wrong-but-remembered transcription.
Only the verbatim quote plus a locator can, therefore the ledger refuses to carry numbers
it cannot source. The in-repo record agrees that this has never been digitised here:
`docs/ICE_CONTACT_PRESSURE_ROLLING_PROXY.md:170` — *"the pressure–area exponent was not
digitised from the standard here"*.

## 3. What *is* established (REAL)

Registered anchors, all with `quote = null`, `locator = null`, `usable_for_claim = false`:

| id | quantity | status | registered value |
|---|---|---|---|
| `A1_pressure_area_exponent_and_reference` | pressure–area exponent + reference pressure/area | UNVERIFIED_CANDIDATE (recall prior) | ex = −0.3, C_R = 1.2 MPa, A_R = 2.5 m² |
| `A2_design_global_ice_force` | design global ice force (clause-level action format) | LOCATOR_ONLY | **no value registered on purpose** |

The conversion chain, which is what the artefact does own:

```
(ex, C_R, A_R) --1--> C_A = C_R A_R^(-ex)   [Pa·m^(-2ex)]
               --2--> p(A) = C_A A^ex        [Pa]
               --3--> q    = p h             [N/m]   <- the unit the unit-width DEM reports
               --4--> F    = q w = p A       [N]
```

Calibre levels are read from the code that runs the case, not from prose: ice thicknesses
0.30 m (`HullIceCase`) and 0.60 m (`IceProperties`), unit width 1 m (the DEM's unit-width
normalisation, `HullIceCase.model_width_m = 0.20 m` is the realised strip). Each converted
point records its area ratio to the reference area and an `outside_reference_area` flag
(0.5–2 window); both unit-width levels and the full-beam upper bound fall outside it, i.e.
any use of the law at this calibre is an **extrapolation** and must be reported as such.

Checks that pass with numbers, not assertions: `q = p h` and `F = p A` are cross-multiplied
to machine precision; `p(A_R)` recovers `C_R` to 0 relative error; the numeric log-slope
`d ln p / d ln A` equals the exponent to < 1e-9; scaling `A → λ²A` scales `p` by exactly
`λ^{2·ex}`; a deliberately corrupted stored pressure is caught (residual > 1e-3), so the
recompute criterion is a live test rather than a tautology.

## 4. Machine-found provenance defect (worth its own line)

The case document declares a reference hull `L = 150 m, B = 22 m, T = 8 m`
(`docs/THREEFIELD_ICE_HULL_PROPELLER.md:20`), while the executed code defaults are
`HullGeometry(length_m=120, beam_m=20, draught_m=7)`. The artifact records both and uses
the executed values (`doc_vs_code_hull_check.consistency_ok = false`). Draught, length and
beam all differ; entry/stem/buttock angles match. Any formula baseline quoted from the
document therefore does not describe the default configuration. Fixing the drift is a
documentation/design decision, deliberately not taken inside an anchor-verification unit.

## 5. What closes C1

1. Obtain the ISO 19906:2019 primary text and transcribe (a) the pressure–area exponent,
   (b) the reference pressure/area, with annex/table/page, and (c) locate the global ice
   action clauses (Clause 14/15 in the 2010 edition — unconfirmed for 2019).
2. Replace the candidate triple with the transcribed values in
   `pressure_area_candidate(...)`; paste the literal sentences into `quote` and the
   table/page into `locator`, flip `verification_status` to `VERIFIED`. The checks, the
   calibre conversion and the claim gate need no change, and
   `test_artifact_schema_and_verdict` automatically switches to the PASS branch.
3. If the standard supplies an action *format* rather than a tabulated global force,
   register the format and the ice-strength/action-factor tables instead of a number —
   do not invent a number to fill the field.

## 6. Limitations

* The dimensional algebra checks structure only; a dimensionally perfect but
  mis-transcribed equation passes it. That is exactly why C1 is the gate.
* The full-beam level is a geometric upper bound (the bow does not contact the full
  breadth); it exists to keep one calibre level inside the reference-area window and must
  never be read as a predicted bow load.
* `recompute` residual of 0.0 on some points is bit-identical re-evaluation; the
  independent reference-form route (`1.26e-16`) is the informative number.