# Verification-evidence ledger: how to read it

This directory holds a **generated** index of the evidence behind every module in
`src/tensordem`. It exists because the repository has far more test files than any
one reviewer can hold in mind, and because "there is a test" and "there is a
quantitative reference" are very different claims.

Files (all regenerated, never hand-edited):

| file | content |
| --- | --- |
| `verification_ledger.json` | full machine-readable ledger (schema `tensordem-verification-evidence-ledger-v1`) |
| `verification_ledger_modules.csv` | one row per source module: grade, attribution, tests, docs |
| `verification_ledger_tests.csv` | one row per test file: counts of each evidence class |
| `verification_ledger.md` | the same tables rendered for reading |

Regenerate with:

```bash
python scripts/audit_verification_evidence.py
python -m pytest tests/test_audit_verification_evidence.py -q
```

The audit is **static**: it reads text and never imports the solver, never runs a
simulation and never touches a GPU. It runs in about a second, so it can be run
on every change.

## What each test is classified as

The classification looks at assertion *style*, strongest first:

| class | recognised by | meaning |
| --- | --- | --- |
| `QUANTITATIVE` | `assertAlmostEqual`, `pytest.approx`, `assert_allclose`, `isclose`, relative-error helpers, `< 1e-9`, or an ordering assertion carrying a numeric/fractional tolerance | a number is compared against a tolerance or reference |
| `EXACT_VALUE` | `assertEqual`, `assert x == y`, checksum comparisons | an exact identity is compared (e.g. SHA-256) |
| `ORDERING` | `assertGreater`/`assertLess`, or a bare `>`/`<` in an assertion | monotonicity/sign only, no declared tolerance |
| `ERROR_PATH` | `assertRaises`, `pytest.raises` | invalid input is rejected |
| `SMOKE` | any other assertion | truth/finiteness/type only |
| `NONE` | no assertion | the test cannot fail on a value |

## What each module is graded as

| grade | rule |
| --- | --- |
| `QUANTITATIVE_DOCUMENTED` | strongest class is `QUANTITATIVE` and the module is named somewhere in `docs/` |
| `QUANTITATIVE_UNDOCUMENTED` | strongest class is `QUANTITATIVE` but no documentation names the module |
| `EXACT_ONLY` | only exact-value comparisons, no tolerance/reference |
| `ORDERING_ONLY` | only inequality assertions |
| `ERROR_PATH_ONLY` | only invalid-input assertions |
| `SMOKE_ONLY` | only bare assertions |
| `NO_TEST` | no test function references the module |

## Interpretation contract (read before quoting a grade)

1. **A grade is not a validation statement.** It summarises assertion style. A
   module can hold `QUANTITATIVE_DOCUMENTED` while its "reference" is a copy of
   its own previous output, an internally derived formula, or a fitted value. The
   ledger cannot detect that; only review of the cited reference can.
2. **Attribution is name-based.** A test is attributed `direct` when the module
   name appears in the test function body, and `file` when the module only appears
   elsewhere in the same file (typically a file-scope import). `file` attribution
   credits every test in that file, so it is coarse: use it as a weak signal.
   `direct_test_count` and `file_only_test_count` are reported for exactly this
   reason.
3. **`reference_hint_tests`** counts crediting tests whose body also contains
   `analytic`, `closed-form`, `literature`, `reference`, `hand calculation` or
   `benchmark`. It is a reviewer hint with known false positives and negatives
   (the word "reference" also appears in ordinary variable names).
4. **Ordering versus tolerance.** `assert abs(err) < tol` with a *named* tolerance
   is counted as `ORDERING`, not `QUANTITATIVE`. This deliberately under-credits:
   a named tolerance can be 1e-12 or 10.
5. **Static parsing limits.** Test functions are found with a line regex; a
   `def test_` inside a multi-line string would be a false record. Tests that are
   collected but skipped (for example optional-dependency integration tests) still
   count as evidence.
6. **Nothing here executes.** A module whose quantitative tests are currently
   failing or skipped still receives its grade. Use `pytest` output for pass/fail;
   use this ledger only for *what kind* of evidence exists.

## Snapshot at the time of writing

Generated from 45 source modules, 97 test files, 999 test functions, 91
documentation files and 89 scripts:

- `QUANTITATIVE_DOCUMENTED`: 36 modules
- `QUANTITATIVE_UNDOCUMENTED`: 8 modules
- `EXACT_ONLY`: 1 module (`run_manifest`, checksum identities)
- `ORDERING_ONLY`, `ERROR_PATH_ONLY`, `SMOKE_ONLY`, `NO_TEST`: 0 modules
- attribution: 8 modules `direct`, 37 modules `file`

Test-function evidence distribution: 571 `QUANTITATIVE`, 182 `EXACT_VALUE`, 131
`ERROR_PATH`, 61 `ORDERING`, 47 `NONE`, 7 `SMOKE`.

These numbers are a snapshot; the CSV and Markdown files are authoritative and are
regenerated on demand. The debts this snapshot exposes are tracked in
`../verification_debt_register.md`.