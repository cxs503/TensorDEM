# Verification-evidence ledger: what the repo's verification notes are actually backed by

Status: audit note, 2026-11-03. Tool: `scripts/verification_evidence_ledger.py`.
Test: `tests/test_verification_evidence_ledger.py`. Artefacts: `outputs/verification_ledger/`
(`ledger.json`, `ledger.md`, `evidence_grades.png`).

## 1. Why a ledger

TensorDEM publishes 73 notes under `docs/`, 67 of which cite at least one executable
(`pytest` module or reproduction script). The repository has already learned, the hard way,
that a merged doc plus a merged test module is not evidence: in the m2536 incident the
documentation *and* the tests of five analytic contracts landed on `main` while the matching
`src/tensordem/*.py` hunks were dropped by the delivery gate, leaving verification tables that
nobody could reproduce (`docs/dem_analytic_verification_debt_restoration.md`). Those claims
were, in practice, `MOCK`-grade evidence sitting inside `REAL`-looking documents.

That failure mode is detectable mechanically, and it recurs every time a PR is partially
delivered. This note installs a standing detector instead of relying on somebody noticing.
It grades every doc by its **executable backing**, not by its prose.

## 2. Method

For each `docs/*.md` the tool extracts four kinds of citation:

| citation | pattern | resolved by |
|---|---|---|
| test module | `tests/test_*.py` | file exists **and** pytest collects it |
| script | `scripts/`, `benchmarks/`, `examples/` `*.py` | file exists |
| library symbol | `tensordem.<ident>` | submodule file/dir, or a public top-level name of any submodule (AST scan of `src/tensordem`) |
| artefact | `outputs/`, `artifacts/`, `results/` paths | file exists (reported, not graded) |

Grading, one grade per doc:

| grade | meaning |
|---|---|
| `REAL_BACKED` | cites an executable and every citation resolves and collects |
| `PARTIAL` | cites an executable, but a cited path is missing |
| `WITHHELD` | cites an executable that **cannot run** (collection error) or a symbol that exists nowhere in the library — the m2536 signature |
| `UNBACKED` | makes a *quantified* verification claim (percentage figure, `rel. error`, `±`) but cites nothing runnable |
| `NO_CLAIM` | design/charter note: no quantified claim, nothing cited |

The symbol resolver is static (AST over `src/tensordem/**/*.py`) so it describes *this*
checkout rather than whatever editable install happens to be active, and so it still audits
the names defined by a module that itself fails to import.

## 3. Result

| grade | docs | share |
|---|---|---|
| `REAL_BACKED` | 64 | 87.7 % |
| `NO_CLAIM` | 6 | 8.2 % |
| `WITHHELD` | 3 | 4.1 % |
| `PARTIAL` | 0 | 0.0 % |
| `UNBACKED` | 0 | 0.0 % |

Citation totals across the 73 notes: 64 test-module references (49 distinct modules),
105 script references (77 distinct scripts), 23 distinct `tensordem` symbols, 74 artefact
paths. Collection state: **97 test modules, 1 collection error**
(`tests/test_fracture_gen_mass_footprint.py`, the deferred m2536 debt).
**All 23 cited library symbols resolve** on this checkout.

The three `WITHHELD` documents are:

| doc | reason |
|---|---|
| `docs/fracture_gen_mass_footprint_audit.md` | its audit test `tests/test_fracture_gen_mass_footprint.py` fails to collect (`MASS_RADIUS_FACTOR`, `face_footprint_radius` absent from `tensordem.fracture_gen`) |
| `docs/dem_analytic_verification_debt_restoration.md` | cites the same uncollectable module (as the debt it describes) |
| `docs/verification_evidence_ledger.md` | this note, which cites the same uncollectable module as its worked example |

The last row is **not** a true defect, and it is kept rather than hidden because it is the
ledger's own known false-positive class: doc-level grading cannot distinguish a note that
*depends* on a broken citation from one that *diagnoses* it. Both read as "cites something
that cannot run". The debt-restoration note in row 2 is ambiguous in exactly the same way.
A claim-level (rather than doc-level) ledger would need to parse the sentence that introduces
the citation — out of scope here, and the conservative direction is the right one: a doc that
mentions an uncollectable module should invite a human look.

The six `NO_CLAIM` notes are charter/design documents (`PROJECT_CHARTER`,
`ecosystem-integration`, `S3_ARTIFACT_STORAGE`, `fracture_event_tracking`,
`indentation_benchmark_protocol`, `polyline_hull_contact`): they contain no percentage or
`rel. error` figure, so there is nothing to reproduce. Note that
`indentation_benchmark_protocol` and `polyline_hull_contact` *specify* protocols and
therefore carry a real debt of a different kind — they are not wrong, they are simply not
claimed results. Any future number added to them will flip them to `UNBACKED` until a test
or script is cited.

### Side finding: artefact citations are unversioned

Of the 74 cited artefact paths, **1 exists in this checkout** (the one this note writes) — the
rest live under `outputs/`, `artifacts/`, `results/`, which are not versioned. A reader
therefore cannot re-derive a published table from the repository alone; they can only re-run
the cited script. That is acceptable *if* the script is versioned and cited (it is, in 77
distinct cases), but it means doc tables and script output can drift silently. It is reported
in `ledger.json` per doc (`n_artifacts` / `artifacts_present`) rather than graded, because
failing a doc for an unversioned PNG would punish the wrong thing.

## 4. Checking the detector against known answers

A detector that always answers "fine" and a detector that always answers "broken" both pass
any single sample, so the tool is pinned in
`tests/test_verification_evidence_ledger.py` against three independent sets of cases:

| case class | source of truth | expected | observed |
|---|---|---|---|
| known answers, restored contracts | m2536 §3 lists 4 restored contracts | 4/4 `REAL_BACKED` | 4/4 |
| known answer, deferred contract | m2536 §4 says it is deliberately not restored | 1/1 `WITHHELD` | 1/1 |
| negative controls (ghost test / ghost symbol / non-collecting module / unbacked claim) | constructed | 4/4 flagged | 4/4 |
| precision controls (schema tag `tensordem.restart/1`, policy prose) | constructed | 2/2 not flagged | 2/2 |

The precision controls exist because both false-positive classes were hit while building the
tool (see §5); without them the detector silently inflates the `WITHHELD` count.

## 5. Pitfalls found while building this

1. **A global symbol map graded every doc.** The symbol check was computed over the union of
   all citations and then attached to each doc, so a single unresolved submodule name marked
   all 72 notes `WITHHELD`. Fix: subset the resolver per doc. Symptom to remember — a graded
   population that is 100 % in one bucket is a bug, not a finding.
2. **File-schema tags look like symbol references.** `tensordem.restart/1` and
   `tensordem.ice-floe-physics/2` are schema identifiers, not API paths. The regex now
   refuses a match followed by `/`, `-` or more identifier characters; without that,
   `accounting_ledger_validation.md` and `ice_floe_physics_validation.md` were graded
   `WITHHELD` for symbols that never existed as claims.
3. **Policy prose reads as a claim.** "Report exactly what was run, what passed or failed"
   contains `passed`; requiring a *number* (percentage, `rel. error`, `±`) before calling
   something a claim keeps `PROJECT_CHARTER` out of the ledger.
4. **The ledger is self-referential.** Adding this note changed the population it grades, so
   the published table had to be regenerated to a fixpoint: the note must not cite a symbol
   that does not exist (row 3 of §3 shows what happens when it cites a test that does not
   collect).

## 6. Interpretation limits (what this ledger does **not** say)

`REAL_BACKED` means *the cited evidence is present and executable on this checkout*. It does
**not** mean the cited tests pass, and it says nothing about whether the numbers in the doc
are correct. The ledger is a **reachability** audit, deliberately static:

- No test module is executed by the ledger; pass/fail comes from a separate campaign
  (`docs/verification_campaigns.md`). A doc citing a module whose tests all fail is
  `REAL_BACKED` here. Closing that gap (recording runtime status per citation) is the natural
  next increment, and is why the ledger records module names rather than counts.
- The library-symbol check is name-level: it proves `tensordem.<name>` exists somewhere, not
  that the signature matches the doc's usage.
- Grades are per doc, not per claim. A doc with ten tables and one working citation is
  `REAL_BACKED`; the grade is a floor on provenance, not a certificate.

## 7. Reproduction

    PYTHONPATH=src python scripts/verification_evidence_ledger.py --output outputs/verification_ledger
    PYTHONPATH=src python -m pytest tests/test_verification_evidence_ledger.py -q

Runtime is ~20 s (dominated by the `pytest --collect-only` probe) and the tool is read-only
with respect to `src/`, `tests/` and `docs/`.