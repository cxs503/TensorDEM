# Fracture-field generator audit: recurrence of the merge-gate revert and its guard

## 1. What happened

The m2245 audit (`tests/test_fracture_gen_mass_footprint.py`,
`benchmarks/fracture_gen_mass_footprint_audit.py`, `MASS_RADIUS_FACTOR`,
`mass_mapping="lattice_cell"`, `footprint_aware`, `on_plane_tolerance`) shipped its
tests without the implementation they audit. Commit `c9c07cd` restored it and added
`scripts/verify_fracture_gen_restore.py` with seven closed-form criteria.

The merge `eb26f6a` (PR #131, m2560) then resolved `src/tensordem/fracture_gen.py`,
`tests/test_fracture_gen_mass_footprint.py`, `tests/test_fracture_gen_audit.py` and
`benchmarks/fracture_gen_mass_footprint_audit.py` from a base that predates `c9c07cd`
(long-lived branch merged after the fix landed), and dropped the audit's documentation
and verifier outright. Net effect measured on the merge commit:

| artefact | at `c9c07cd` / `26e3447` | at merge `eb26f6a` (HEAD before this task) |
|---|---|---|
| `src/tensordem/fracture_gen.py` | 443-line audit API (`MASS_RADIUS_FACTOR`, `face_footprint_radius`, `mass_accounting`, `lattice_cell`) | pre-audit 426-line module, API absent |
| `tests/test_fracture_gen_mass_footprint.py` | collectable | **collection error** -> the whole suite of 1039 tests was unrunnable |
| `scripts/verify_fracture_gen_restore.py` | present | deleted |
| `docs/fracture_gen_restore_verification.md` + `docs/fracture_gen_restore/` | present | deleted |

A collection error is the most expensive failure mode in this repository: it makes every
other red test invisible, so seven honest-debt reports were produced while the suite could
not run at all.

## 2. What this task restores and fixes

1. `src/tensordem/fracture_gen.py` restored to the audited version
   (frozen legacy mapping `equal_volume_spheres` + mass-conserving `cell_volume`/
   `lattice_cell`, footprint-aware cutting, `on_plane_tolerance`, `FailureFace.extent_radius`,
   `FractureLedger.mass_accounting`).
2. **Default mapping decided by evidence, not by taste.** `tests/test_fracture_gen.py`
   asserts the *volume-consistent* mass of a 3x3x3 lattice (`rho*N*h^3`), i.e. it requires
   the default to be `cell_volume`; the m2245 audit explicitly selects the frozen mapping
   where it audits it. Setting `DEFAULT_MASS_MAPPING = "cell_volume"` satisfies both, and the
   two audit call sites (`benchmarks/fracture_gen_mass_footprint_audit.py`,
   `scripts/verify_fracture_gen_restore.py`) are made explicit so their numbers no longer
   depend on the default.
3. `tests/test_fracture_gen_audit.py` restored to the `c9c07cd` version: the quantisation
   criterion was checking the wrong *sign* of `relative_error`.
4. `scripts/verify_fracture_gen_restore.py` and `docs/fracture_gen_restore/*` restored and
   re-run; all criteria pass with zero deviation at double precision.

## 3. Guard against the next occurrence

The audit contracts are now pinned by tests that fail at *collection* time if the API is
removed (`tests/test_fracture_gen_mass_footprint.py` imports the symbols at module import),
and by `TestAuditReproducibility`, which re-executes
`benchmarks/fracture_gen_mass_footprint_audit.py::run_audit` inside pytest and compares its
numbers with the module API. Anyone reverting the implementation can no longer leave the
repository in a state where the suite silently cannot start: the failure is immediate and
names the missing symbol.

## 4. Verification run in this task

```
python -m pytest tests/test_fracture_gen.py tests/test_fracture_gen_audit.py \
    tests/test_fracture_gen_mass_footprint.py -q   # 48 passed
python scripts/verify_fracture_gen_restore.py      # ALL CRITERIA PASS
python -m pytest tests -q                          # full suite, see report
```
