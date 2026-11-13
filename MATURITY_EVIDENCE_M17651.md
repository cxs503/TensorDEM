# MATURITY EVIDENCE m17651 - CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE

Case: `CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE` (TensorDEM). Anchor: shear-band inclination of a plane-strain biaxial cell, 24 x 48 spacings, `sigma_3` = 100 kPa, wall friction 0.55, 4000 consolidation steps; reference 60 deg (Desrues & Viggiani 2004, 55-60 deg), tolerance 15 %.

**Outcome: PASS** - `theta` = 55.50 deg vs 60 deg = 7.50 % error, localisation z = 15.7 (needs > 3), known-angle operator error max 0.00 deg. Maturity gate replay: **L4** (requested L4, passed=True, gaps=[]).

## Evidence files (hashes are of the delivered tree)

| role | path | bytes | sha256[:16] |
|---|---|---|---|
| implementation (case) | `src/tensordem/biaxial_anchor_case.py` | 21556 | `a1f7b43daf621554` |
| implementation (DEM cell + measurement chain) | `src/tensordem/biaxial_shear_band.py` | 31250 | `191cf01a881b6f17` |
| entry point | `benchmarks/biaxial_shear_band/run_anchor_case.py` | 3808 | `2cd7856deea8a238` |
| benchmark (4 configurations) | `benchmarks/biaxial_shear_band/run_benchmark.py` | 8955 | `62b2569e18cf3abf` |
| benchmark artefact (4 configurations) | `artifacts/biaxial_shear_band/results.json` | 48439 | `42cf93c39cddf34a` |
| run artefact | `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json` | 88510 | `1e6de71e074a2c3c` |
| figure | `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/shear_band_case.png` | 254819 | `aa203058783ff1cc` |
| doc generator | `benchmarks/biaxial_shear_band/make_anchor_doc.py` | 17732 | `36f4054a11cde818` |
| generated report | `docs/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE.md` | 10715 | `282ec850f54afaa6` |
| integration caller / maturity gate | `src/tensordem/integration_callers.py` | 27183 | `b5fd63831f42eec4` |
| tests | `tests/test_biaxial_anchor_case.py` | 6768 | `ec492c6ba590d59b` |

## L2 - implementation + entry + run artefacts

* Implementation: `src/tensordem/biaxial_anchor_case.py` (known-angle benchmark, anchor run, pre-declared acceptance arithmetic, figure) on top of the production DEM cell `src/tensordem/biaxial_shear_band.py`; no placeholder code.
* Entry: `benchmarks/biaxial_shear_band/run_anchor_case.py` (`--quick` smoke, `--no-dem` known-angle only, exit code = verdict).
* Run artefacts: `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json` (full band history, config echo, environment echo, repro command) and `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/shear_band_case.png`.

## L3 - integration caller (platform side)

`tensordem.integration_callers.run_biaxial_shear_band_case(output_dir=..., quick=False)` returns the machine-checkable verdict dict; `load_biaxial_case_result()` replays a stored artefact **without** re-running the DEM; `check_biaxial_maturity_gate(result, 'L4')` re-derives the ladder from files on disk. CLI: `python -m tensordem.integration_callers --biaxial-case`.

Replay of the delivered artefact:

```json
{
 "overall": "PASS",
 "verdict": {
  "ref_deg": 60.0,
  "got_deg": 55.5,
  "relative_error": 0.075,
  "tolerance": 0.15,
  "within_tolerance": true,
  "resolved": true,
  "late_snapshots": 4,
  "angle_std_deg": 1.118033988749895,
  "z_score": 15.748587723375298,
  "eta2": 0.5193431435858201,
  "localised": true,
  "passed": true
 },
 "gate": {
  "requested_level": "L4",
  "level_achieved": "L4",
  "passed": true,
  "gaps": []
 }
}
```

## L4 - benchmark (reference vs got) + repro command

| quantity | reference | got | error |
|---|---|---|---|
| band inclination (deg) | 60.0 (literature) | 55.50 | 7.50 % |
| known-angle band, injected 30 deg (deg) | 30.0 (analytic) | 30.0 | 0.00 deg |
| known-angle band, injected 45 deg (deg) | 45.0 (analytic) | 45.0 | 0.00 deg |
| known-angle band, injected 60 deg (deg) | 60.0 (analytic) | 60.0 | 0.00 deg |
| known-angle band, injected 70 deg (deg) | 70.0 (analytic) | 70.0 | 0.00 deg |

```
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/biaxial_shear_band/run_anchor_case.py --out-dir artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE
```

## Test evidence

Commands: `PYTHONPATH=src python -m pytest tests/test_biaxial_anchor_case.py tests/test_biaxial_shear_band.py tests/test_biaxial_band_diagnostics.py -q`

Result: `39 passed, 1 warning in 29.43s` (exit 0)

Documented pitfalls / limits: global `psi` of the servo cell is unphysical (`psi > phi`), so Roscoe/Arthur rows are not evidence; the band-local dilation estimator stays WITHHELD; 16 x 32 specimens do not resolve a band and are reported as `theta = NaN`, never as a pass.

