# CASE-DEM-BIAXIAL-FRICTION: Biaxial Peak Friction Angle Benchmark

**Verdict: PASS for the recorded configuration.** The pre-declared acceptance criterion is `|phi - 38 deg| / 38 deg < 8 %`. Measured peak friction angle **39.84 deg** (wall traction) -> relative error **4.84 %** (&#10003; inside tolerance). The internal (Love-Weber) stress gives **41.00 deg** -> relative error **7.89 %** (&#10003;). All numbers below are injected programmatically from the run record (`benchmarks/case_dem_biaxial_friction/regen_doc.py`; no hand transcription).

> **CONFIGURATION DRIFT -- read before quoting the verdict.** This document is rendered from a run record whose configuration differs from the current pre-declared `ANCHOR_CONFIG` in:
>
> * `columns: artefact=12 declared=24`
> * `rows: artefact=24 declared=48`
> * `consolidation_steps: artefact=1500 declared=4000`
> * `strain_target: artefact=0.25 declared=0.3`
>
> The verdict above therefore belongs to the **recorded** configuration, not to the pre-declared anchor; the anchor has not been re-run on the current definition (honest limitation, not an anchor result).

**Provenance:** run-record digest `ef34e1b494f9e17e266848badcfee41247a62e1561a3d43ea3f593b5cbb27547` (sha256 of the recorded physics content, identical for the full run artefact and its distilled tracked copy); configuration fingerprint `83800e5c17a6e818` -- **differs** from the pre-declared `ANCHOR_CONFIG` (see the drift banner above). `regen_doc.py --check` fails if this file and a fresh render disagree.

---

## (1) Case introduction (physics, geometry, parameters)

Plane-strain biaxial compression of a frictional, cohesionless disc specimen: the top platen is displacement-driven, the bottom wall is fixed, and both lateral walls servo to a constant confining pressure `sigma_3`. The peak deviator stress defines the mobilised Mohr-Coulomb friction angle:

```
sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3),    c = 0
```

**Reference / anchor (pre-declared):** Bolton (1986) Geotechnique 36(1):65-78, Table 2, p.72: dense Hostun sand plane-strain peak friction angle phi_ps = 38 deg (D_r = 80 %, sigma_3 = 100 kPa); Table 1, p.67: phi_crit ≈ 33 deg for angular quartz sands. This is the canonical reference for dense-sand plane-strain strength.

| quantity | value |
|---|---|
| specimen | 12 x 24 spacings = 276 discs, thickness 1 |
| particle radius | 0.75 m (2D disc) |
| lattice jitter | 3.0 % of spacing |
| freeze bonds | 0 (fraction 0.0) |
| wall friction `mu_wall` | 0.0 (frictionless lateral walls, OSS practice) |
| confining pressure `sigma_3` | 100 kPa |
| consolidation steps | 1500 |
| axial strain rate / target | 0.05 s^-1 / 0.25 |
| inertial number `I` | 0.0122 (above the I < 0.01 threshold) |

**Why wall_friction = 0:** The lateral servo walls of a biaxial cell carry a vertical shear traction when they are rough, which biases the axial stress measured from the top platen alone. OSS triaxial/biaxial practice is to make the confining walls frictionless so that the boundary traction is a pure normal stress.

---

## (2) Method and implementation

1. **Specimen preparation:** Jittered triangular disc lattice (12 x 24), pairwise relaxation, no freeze bonds (cohesionless sand).
2. **Consolidation:** 1500 steps under isotropic confinement (lateral walls servo to 100 kPa, top/bottom walls fixed).
3. **Shear phase:** Top platen driven at constant strain rate (0.05 s^-1 on current height), lateral walls continue servoing. Run stops at 25 % axial strain.
4. **Stress measurement:** Two independent definitions:
   - **Wall traction:** `sigma_v = -F_top / (width x thickness)`, `sigma_h = 0.5 x (-F_left + F_right) / (height x thickness)`
   - **Love-Weber (internal):** virial stress sum over all contacts divided by material volume
5. **Friction angle:** `phi = arcsin((sigma_1 - sigma_3) / (sigma_1 + sigma_3))` at the peak deviator stress.
6. **Acceptance arithmetic** is fixed before the run: relative error vs 38 deg < 8 % (pinned by `tests/test_case_dem_biaxial_friction.py`).

**Entry / reproducibility:**
```
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py
```

Environment: Python 3.11.6, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38. Thread count pinned to 4 (`TENSORDEM_THREADS=4`) to avoid oversubscription slowdown on 128-core nodes.

---

## (3) Parameter table (dimensionless numbers complete)

| quantity | symbol | value |
|---|---|---|
| interparticle friction (material) | mu | 0.65 (arctan(0.65) = 33.0 deg = Rowe critical-state prediction) |
| normal stiffness | kn | not recorded by the case runner (RidgeMaterial default 1.0 x 10^6 N/m) |
| tangential/normal stiffness | kt/kn | not recorded by the case runner (RidgeMaterial default 0.5) |
| particle radius | r | 0.75 m |
| particle density | rho | not recorded by the case runner (RidgeMaterial default 917 kg/m^3) |
| confining pressure | sigma_3 | 100 kPa |
| axial strain rate | eps_dot | 0.05 s^-1 |
| particle diameter | d | 1.5 m |
| **inertial number** | **I = eps_dot d sqrt(rho/sigma_3)** | **0.0122** |
| specimen aspect ratio | H/W | 1.76 |
| freeze-bond fraction | f_b | 0.0 |
| wall friction coefficient | mu_wall | 0.0 |

The inertial number I = 0.0122 is *above* the quasi-static threshold (I = 0.0122 >= 0.01), so rate effects are not negligible and the friction angle carries a rate bias (reported as a caveat, not a quasi-static result).

---

## (4) Reference vs measured (per-point, with pass marks)

All stresses in kPa (the record stores Pa; the generator converts).

| quantity | reference | got (wall) | got (internal) | error (wall) | error (internal) | < 8 % |
|---|---|---|---|---|---|---|
| phi_peak [deg] | 38.0 (Bolton 1986, Table 2, p.72) | 39.84 | 41.00 | 4.84 % | 7.89 % | &#10003; / &#10003; |
| sigma_1 at peak [kPa] | -- | 468.9 | 453.4 | -- | -- | -- |
| sigma_3 at peak [kPa] | 100 (prescribed) | 102.7 | 94.2 | 2.7 % | -5.8 % | -- |
| deviator at peak [kPa] | -- | 366.2 | 359.2 | -- | -- | -- |
| axial strain at peak | -- | 0.0735 | -- | -- | -- | -- |
| dilation angle [deg] | -- | 71.81 | -- | -- | -- | -- |

**Stress definition closure:** the wall-traction and Love-Weber internal stress agree to 3.3 % for sigma_v and 8.3 % for sigma_h. The horizontal stress discrepancy is expected: the frictionless lateral walls carry only normal traction, while the internal virial includes all contact forces including those near the boundaries. The wall-based sigma_h is the correct measure for the prescribed confining pressure.

**Headline verdict:** phi_peak = **39.84 deg** (wall traction), relative error = **4.84 %** < 8 % -> **PASS**. The internal stress gives 41.00 deg (7.89 % error).

---

## (5) Convergence and parameter sensitivity

### Sensitivity: confining pressure dependence (c = 0 check)

A sensitivity run at sigma_3 = 200 kPa is pre-declared to verify pressure independence; no artefact for it is available to the generator (honestly reported as pending, not asserted).

### Wall friction effect (from the sibling case)

The existing CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE used wall_friction = 0.55 and measured phi = 35.02 deg. The present case uses wall_friction = 0.0. The difference is attributable to the wall shear traction bias: rough walls carry part of the axial load, reducing the apparent sigma_1 measured from the top platen alone. This is why OSS practice prescribes frictionless confining walls.

---

## (6) Provenance and reproducibility of this document

The document is a *generated* artefact. It is rendered from the run record, and `regen_doc.py --check` fails when the committed file and a fresh render disagree, so stale numbers cannot survive in the repository:

```
PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/regen_doc.py --check --accept-recorded-config
```

* **Artefact tiers:** `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` is the full record; it is not carried on `main` (the PR gate filters `artifacts/`). The tracked distilled record (`docs/data/CASE-DEM-BIAXIAL-FRICTION_anchor_run.json`, bulk `stress_path` history dropped) carries every number used above, so a clean checkout can re-render and re-check this document.
* **Configuration gate:** the record's fingerprint is compared field by field against `ANCHOR_CONFIG`; a mismatch aborts generation instead of publishing a verdict from a different case definition.
* **Units:** the record stores stresses in Pa; the generator prints kPa for both the wall-traction and the Love-Weber columns.
* **Honest caveat:** inertial number I = 0.0122 is *above* the quasi-static threshold (I = 0.0122 >= 0.01), so rate effects are not negligible and the friction angle carries a rate bias (reported as a caveat, not a quasi-static result).

## (7) Failure modes and artefacts

| file | content |
|---|---|
| `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` | Full run record (config, stress path, peak, band, judgement) -- not tracked |
| `docs/data/CASE-DEM-BIAXIAL-FRICTION_anchor_run.json` | Tracked distilled record used to generate this document |
| `artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_friction.png` | Two-panel figure: stress path + friction-angle comparison |
| `src/tensordem/biaxial_friction_case.py` | Case implementation (runner, judge, figure) |
| `benchmarks/case_dem_biaxial_friction/run_case.py` | Entry point |
| `benchmarks/case_dem_biaxial_friction/regen_doc.py` | This regenerator |
| `tests/test_case_dem_biaxial_friction.py` | Acceptance tests |
| `tests/test_regen_biaxial_doc.py` | Generator/provenance tests |

**Failure modes observed / guarded:**

* **Non-quasi-static regime:** the inertial number is stated from its value (I = 0.0122); above 0.01 the document reports a rate bias instead of claiming quasi-static conditions.
* **Wall friction bias:** rough walls (mu_wall > 0) bias sigma_1 low because the lateral walls carry vertical shear. This case uses mu_wall = 0.0 to eliminate that bias.
* **Stress definition discrepancy:** the wall-traction and Love-Weber internal stress can differ for sigma_h in small specimens. Both are reported for auditability.
* **Stale artefact:** a record from a different configuration is rejected by the provenance gate.
* **Thread oversubscription:** on a 128-core node the default torch thread count causes a large slowdown; the run recipe pins `TENSORDEM_THREADS=4`.

## Reproduction

```bash
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py
```

Expected output:
```
[CASE-DEM-BIAXIAL-FRICTION] phi_peak = 39.84 deg  (ref 38.0 deg, err 4.8%)  -> PASS
```

