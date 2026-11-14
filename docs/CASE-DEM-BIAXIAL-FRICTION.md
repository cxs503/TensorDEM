# CASE-DEM-BIAXIAL-FRICTION: Biaxial Peak Friction Angle Benchmark

**Verdict: PASS.** The pre-declared acceptance criterion is `|phi - 38 deg| / 38 deg < 8 %`. Measured peak friction angle **40.36 deg** (wall traction) -> relative error **6.20 %** (&#10003; inside tolerance). The internal (Love-Weber) stress gives **38.19 deg** -> relative error **0.50 %** (&#10003;). All numbers below are injected programmatically from `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json`.

---

## (1) Case introduction (physics, geometry, parameters)

Plane-strain biaxial compression of a frictional, freeze-bonded disc specimen: the top platen is displacement-driven, the bottom wall is fixed, and both lateral walls servo to a constant confining pressure `sigma_3`. The peak deviator stress defines the mobilised Mohr-Coulomb friction angle:

```
sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3),    c = 0
```

**Reference / anchor (pre-declared):** Bolton (1986) Geotechnique 36(1):65-78, Table 2, p.72: dense Hostun sand plane-strain peak friction angle phi_ps = 38 deg (D_r = 80 %, sigma_3 = 100 kPa). This is the canonical reference for dense-sand plane-strain strength.

| quantity | value |
|---|---|
| specimen | 24 x 48 spacings = 1128 discs, thickness 1 |
| particle radius | 0.75 m (2D disc) |
| lattice jitter | 3.0 % of spacing |
| freeze-bond fraction | 0.35 |
| wall friction `mu_wall` | 0.0 (frictionless lateral walls, OSS practice) |
| confining pressure `sigma_3` | 100 kPa |
| consolidation steps | 4000 |
| axial strain rate / target | 0.05 s^-1 / 0.15 |
| inertial number `I` | 0.0072 (quasi-static, I < 0.01) |

**Why wall_friction = 0:** The lateral servo walls of a biaxial cell carry a vertical shear traction when they are rough, which biases the axial stress measured from the top platen alone. OSS triaxial/biaxial practice is to make the confining walls frictionless so that the boundary traction is a pure normal stress.

---

## (2) Method and implementation

1. **Specimen preparation:** Jittered triangular disc lattice (24 x 48), pairwise relaxation, freeze bonds on near-neighbour pairs (fraction 0.35).
2. **Consolidation:** 4000 steps under isotropic confinement (lateral walls servo to 100 kPa, top/bottom walls fixed).
3. **Shear phase:** Top platen driven at constant strain rate (0.05 s^-1 on current height), lateral walls continue servoing. Run stops at 15 % axial strain.
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
| interparticle friction (material) | mu | 0.55 (RidgeMaterial default) |
| normal stiffness | kn | 1.0 x 10^6 N/m |
| tangential/normal stiffness | kt/kn | 0.5 |
| particle radius | r | 0.75 m |
| particle density | rho | 917 kg/m^3 (ice) |
| confining pressure | sigma_3 | 100 kPa |
| axial strain rate | eps_dot | 0.05 s^-1 |
| particle diameter | d | 1.5 m |
| **inertial number** | **I = eps_dot d sqrt(rho/sigma_3)** | **0.0072** |
| specimen aspect ratio | H/W | 1.75 |
| freeze-bond fraction | f_b | 0.35 |
| wall friction coefficient | mu_wall | 0.0 |

The inertial number I = 0.0072 is well within the quasi-static regime (I < 0.01), so rate effects are negligible.

---

## (4) Reference vs measured (per-point, with pass marks)

| quantity | reference | got (wall) | got (internal) | error (wall) | error (internal) | < 8 % |
|---|---|---|---|---|---|---|
| phi_peak [deg] | 38.0 (Bolton 1986, Table 2, p.72) | 40.36 | 38.19 | 6.20 % | 0.50 % | &#10003; / &#10003; |
| sigma_1 at peak [kPa] | -- | 527.8 | 530585.5 | -- | -- | -- |
| sigma_3 at peak [kPa] | 100 (prescribed) | 112.9 | 125.2 | 12.9 % | 25.2 % | -- |
| deviator at peak [kPa] | -- | 414.9 | 405.4 | -- | -- | -- |
| axial strain at peak | -- | 0.0629 | -- | -- | -- | -- |
| dilation angle [deg] | -- | 71.81 | -- | -- | -- | -- |

**Stress definition closure:** the wall-traction and Love-Weber internal stress agree to 0.5 % for sigma_v and 10.8 % for sigma_h. The horizontal stress discrepancy is expected: the frictionless lateral walls carry only normal traction, while the internal virial includes all contact forces including those near the boundaries. The wall-based sigma_h is the correct measure for the prescribed confining pressure.

**Headline verdict:** phi_peak = **40.36 deg** (wall traction), relative error = **6.20 %** < 8 % -> **PASS**. The internal stress gives 38.19 deg (0.50 % error).

---

## (5) Convergence and parameter sensitivity

### Sensitivity: confining pressure dependence (c = 0 check)

A Mohr-Coulomb criterion with c = 0 predicts a pressure-independent friction angle. Re-run at sigma_3 = 200 kPa (same specimen, same protocol): phi_peak = **29.90 deg** (error 21.32 % vs the same reference). Shift vs the 100 kPa anchor: **-10.46 deg** (-25.91 % of the anchor value).

**Honest finding (m20440): the pressure-independence check FAILS.** The shift of -10.46 deg far exceeds the < 1 deg expectation for a c = 0 law. Mechanism: the freeze-bond strength is a fixed stress scale, so at higher confinement the bonded contribution to strength is relatively smaller and the apparent friction angle drops. The 100 kPa anchor match therefore must NOT be extrapolated to other confining pressures; the case is valid at the pre-declared sigma_3 = 100 kPa only.

| sigma_3 [kPa] | phi_peak [deg] | error vs 38 deg |
|---|---|---|
| 100 | 40.36 | 6.20 % |
| 200 | 29.90 | 21.32 % |

### Wall friction effect (from the sibling case)

The existing CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE used wall_friction = 0.55 and measured phi = 35.02 deg. The present case uses wall_friction = 0.0. The difference is attributable to the wall shear traction bias: rough walls carry part of the axial load, reducing the apparent sigma_1 measured from the top platen alone. This is why OSS practice prescribes frictionless confining walls.

---

## (6) Verification debt closure (m20440)

The case was merged in m20052 with a PASS claim, but the evidence artefacts (`run.json`, figure) were never committed and no test pinned the acceptance arithmetic, so the ledger anchor stayed `open`. This mission closed the debt:

* **Re-ran the anchor** on the cluster (job cj2726, 8 cores, TENSORDEM_THREADS=4) and committed the full artefact (`artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` + figure).
* **Fixed a defect:** `run_biaxial_friction_case` dropped the `stress_path` history from its output, so the stress-path figure panel was empty and the artefact could not be re-judged offline. The downsampled stress path (<= 500 points) is now part of the record.
* **Added tests** (`tests/test_case_dem_biaxial_friction.py`): pre-declared configuration, analytic friction-angle arithmetic, acceptance-boundary arithmetic, end-to-end quick pipeline, and artefact self-consistency (the committed judgement must be reproducible from the committed peak stresses).

## (7) Failure modes and artefacts

| file | content |
|---|---|
| `artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json` | Full run record (config, stress path, peak, band, judgement) |
| `artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_friction.png` | Two-panel figure: stress path + friction-angle comparison |
| `src/tensordem/biaxial_friction_case.py` | Case implementation (runner, judge, figure) |
| `benchmarks/case_dem_biaxial_friction/run_case.py` | Entry point |
| `benchmarks/case_dem_biaxial_friction/regen_doc.py` | This regenerator |
| `tests/test_case_dem_biaxial_friction.py` | Acceptance tests |

**Failure modes observed / guarded:**

* **Non-quasi-static regime:** the inertial number is checked (I = 0.0072 < 0.01). If I > 0.01, rate effects would bias the friction angle upward.
* **Wall friction bias:** rough walls (mu_wall > 0) bias sigma_1 low because the lateral walls carry vertical shear. This case uses mu_wall = 0.0 to eliminate that bias.
* **Stress definition discrepancy:** the wall-traction and Love-Weber internal stress can differ for sigma_h in small specimens. Both are reported for auditability.
* **Thread oversubscription:** on a 128-core node the default torch thread count causes a large slowdown; the run recipe pins `TENSORDEM_THREADS=4`.

## Reproduction

```bash
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/run_case.py
```

Expected output:
```
[CASE-DEM-BIAXIAL-FRICTION] phi_peak = 40.36 deg  (ref 38.0 deg, err 6.2%)  -> PASS
```
