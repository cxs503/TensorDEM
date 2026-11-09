# Trickle-injection repose driver with rate control (TensorDEM, selfstart m7569)

**Status: driver + campaign harness implemented and verified at the contact-law and
estimator level; the heap-angle campaign itself was NOT executed in this session
(a single 90-grain case exceeded the 110 s per-call budget) — that is an honest
gap, recorded in §6 and in `docs/dem_calibration_gap_ledger.md`. No heap angle is
quoted anywhere in this document.**

Module: `src/tensordem/repose_injection.py`
Campaign script: `scripts/verify_repose_injection.py`
Tests: `tests/test_repose_injection.py` (13 cases, all passing)
Artifact: `artifacts/repose_injection/verification.json`
Figure: `artifacts/repose_injection/figures/anchors_and_estimator.png`

---

## 1. Case introduction — the gap this closes

The repository's only repose driver is the lifting-cylinder column collapse of
`repose_calibration.py` (m4970). It is monodisperse, wall-confined and
sphere-only. The shape-representation verdict
(`docs/dem_nonspherical_representation_verdict.md` §5.4) records the consequence
as an explicit open item:

> "the repository's calibrated repose driver (`repose_calibration.py`, lifting
> cylinder) is sphere-only and therefore cannot serve the clump/superquadric
> comparison. Closing this needs a *trickle-injection* driver with rate control."

A point-source pour is the standard laboratory reflex for that: grains are
released one at a time at a controlled rate onto a frictional floor, they fall,
roll and arrest, and the free-surface angle of the resulting open heap is the
observable. Because there is no confining wall and no prescribed packing, the
same driver accepts **any set of grain radii** — the other limitation the column
driver declares ("no particle-size distribution").

| quantity | value used in the driver defaults |
|---|---|
| grain radius | 6.0 mm (uniform draw of half-width `polydispersity`) |
| density | 2500 kg/m³ |
| normal stiffness `k_n` | 6.0e4 N/m (linear spring-dashpot) |
| tangential stiffness `k_t` | 0.5 `k_n`, Cundall–Strack, saturated at `mu F_n` |
| restitution `e` | 0.05 → damping ratio `zeta = 0.6901` |
| rolling resistance `mu_r` | 0.20 (swept 0 … 0.30) |
| sliding friction `mu` | 0.50 (grain–grain and grain–floor) |
| outlet | disc radius 15 mm, ≥ 2.5 grain radii above the *settled* apex |
| release rate | 120 grains/s (swept 60 … 240) |
| timestep | contact period / 24 = 3.59e-5 s at `r = 6` mm |

## 2. Method

1. **Release.** One grain at a time, interval `1/injection_rate`, uniform in the
   outlet disc. The release height is keyed to the apex of the *settled* grains
   (`|v| < 0.05 m/s`), never to the highest grain in flight — otherwise a stream
   of falling grains ratchets the source upwards and no heap forms (this failure
   mode was hit and is documented in §5).
2. **Broad phase.** Verlet pair list over a skin of `0.3 × 2r`, rebuilt by dense
   distance blocks when the *cumulative* displacement since the last rebuild
   exceeds `skin/2`; tangential/rolling spring history is carried across rebuilds
   by an `(i, j)` key lookup, so a rebuild is invisible to the physics.
3. **Contacts.** Identical laws to the column driver (normal spring-dashpot with
   `zeta(e)`, Cundall–Strack tangential spring, isotropic rolling spring
   saturated at `mu_r R_eff F_n`), generalised to per-pair effective mass,
   stiffness and radius so polydisperse sets are admissible.
4. **Settling gate.** The run only reports an angle after the maximum grain speed
   has stayed below 2 mm/s for 5 ms *and* `settle_time` has elapsed since the last
   release; a still-flowing heap yields `angle_error`, not a number.
5. **Two independent angle estimators.** (a) the merged radial-bin envelope
   estimator `repose_calibration.repose_angle` (bin maxima per azimuthal sector,
   2σ clipping); (b) `surface_flank_angle`, a new free-surface point fit: grains
   with no neighbour above them within `0.8 (r_i + r_j)` only, line fit over the
   flank band `[0.30, 0.85] r_toe`, plus per-sector fits for the azimuthal
   spread. They share no binning or extremum logic, so their agreement is
   evidence about the *measurement*, not about one code path.

## 3. Results — what was actually measured

### 3.1 Contact-law anchors (single grain, tilted plane)

In a frame rotated with the plane, `gravity = g(sin θ, 0, −cos θ)`; the grain
rests on `z = 0`; a 30 ms hold decides "moves" (`|Δx| > 5e-5 m`).

| anchor | bracket [deg] | analytic reference | travel at lo / hi [m] | bracketed? |
|---|---|---|---|---|
| A1 sliding, rotation frozen, `mu = 0.5` | 23.909 … 29.222 | `atan(mu) = 26.565` | 1.14e-6 / 2.55e-4 | **yes** |
| A2 rolling, `mu_r = 0.20`, rotation free | 10.179 … 12.441 | `atan(mu_r) = 11.310` | 1.35e-6 / 8.05e-5 | **yes** |
| A2b ideal rolling, `mu_r = 0` | 1.328 … 5.313 | threshold ≈ 0 (ideal sphere rolls at any tilt) | 3.25e-3 already moving at the low end | threshold ≈ 0 confirmed |
| A3 legacy floor convention, `mu_r = 0` | 23.909 … 29.222 | correct: ≈ 0 | 1.14e-6 / 2.55e-4 | **no** — moves only above `atan(mu)` |
| A3b legacy floor convention, `mu_r = 0.20` | 10.179 … 12.441 | correct: `atan(mu_r) = 11.310` | 2.33e-7 / 2.84e-7 | **no** — never rolls |

### 3.2 Angle estimator on synthetic cones (analytic reference)

Grains laid on a cone of known flank angle (600 grains, `r = 6` mm, toe radius
100 mm); `surface_flank_angle` must recover the angle it was given.

| cone angle [deg] | fitted [deg] | rel. error | flank `r²` | sector σ [deg] |
|---|---|---|---|---|
| 8 | 8.0330 | 0.41 % | 0.9893 | 0.021 |
| 15 | 15.0598 | 0.40 % | 0.9893 | 0.038 |
| 25 | 25.0103 | 0.041 % | 0.9891 | 0.089 |
| 35 | 35.1169 | 0.33 % | 0.9894 | 0.165 |

### 3.3 Broad phase cost

`build_pair_list` on 1000 grains (random box, skin 3.6 mm): 880 candidates,
0.180 s per rebuild. Per-step contact work is then `O(N × coordination)`; the
rebuild is the only `O(N²)` term and it is amortised over ~100 steps.

### 3.4 Heap campaign — not run

`scripts/verify_repose_injection.py` implements the full matrix (baseline,
rate 60/240, `mu_r ∈ {0, 0.10, 0.30}`, `polydispersity = 0.25`, `mu = 0.35`, and
the legacy-floor control) with multiprocessing and figure/CSV export. On this
host a single 90-grain case did not finish inside the 110 s per-call budget, so
**no heap repose angle was measured and none is claimed**. The script runs
unchanged on the 702 cluster (128 cores, `--workers 8`).

## 4. Notes — what was got wrong, and what it means

1. **A defect in the merged column driver's floor contact.** `repose_calibration.simulate`
   builds the floor tangential history from the grain **centre** velocity
   (`radial_v = velocities[idx]`, `radial_v[:, 2] = 0`), omitting the spin term
   `omega × (−R z)` that the *particle–particle* branch of the same function does
   include. Consequence, measured here with the same law (A3/A3b):
   * a grain with `mu = 0.5`, `mu_r = 0` needs **26.57°** before it moves instead
     of rolling at any tilt — a spurious static threshold of `atan(mu)`;
   * a grain with `mu_r = 0.20` **never rolls at all**, even at 12.44° (2.8e-7 m
     of travel), because the spurious sliding friction caps the rolling mode.
   Physics is unaffected only in the friction-frozen limit. The published
   `mu → theta_r` table of m4970 is therefore a table of *this* wall law, not of
   Cundall–Strack contact; it should be re-measured with the spin-corrected
   floor branch before being quoted as a material calibration. The new driver
   defaults to the spin-corrected law and keeps `floor_spin_term=False` as an
   explicit reproduction switch — the merged module was **not** modified in this
   task (its regression artefacts stay bit-identical).
2. **Silent blow-up from a stale neighbour list.** The first implementation
   compared the *per-step* displacement with `skin/2`; the list then never
   rebuilt, grains tunneled, and the resulting deep overlaps launched grains at
   4–14 m/s (a heap that "exploded" and still returned a number). Fixed by
   comparing the **cumulative** displacement since the last rebuild and by
   forcing a rebuild on every release. Guard: any non-finite or
   still-flowing state now returns `angle_error` instead of an angle.
3. **Source-height ratchet.** Keying the outlet to the highest grain (including
   airborne ones) makes the source climb with each release; a 150-grain run
   finished with "grains" at `z = 7.7 m`. Keyed to the settled apex instead.
4. **Estimator floors.** A heap below ~100 grains has no measurable flank: both
   estimators refuse (this is asserted by a test), so small smoke runs cannot
   masquerade as repose measurements.
5. **Applicability.** The driver is a numerical device. Its angles belong to the
   contact laws above; they are not laboratory angles for a specific bulk
   material, and non-spherical grains are still out of reach until the clump /
   superquadric contact machinery is wired to this loop (the next node).

## 5. Error judgement against references

| quantity | measured | reference | error | < 3 %? |
|---|---|---|---|---|
| A1 sliding onset | bracket 23.909–29.222° | `atan(0.5) = 26.565°` | analytic inside bracket; bracket half-width 9.9 % | yes as a bracket; ±10 % localisation |
| A2 rolling onset | bracket 10.179–12.441° | `atan(0.2) = 11.310°` | analytic inside bracket; half-width 9.9 % | yes as a bracket |
| A2b ideal rolling | moves at 1.33° | threshold 0° | absolute 1.33° | yes (threshold < 5°) |
| angle estimator, 8–35° cones | 8.033 / 15.060 / 25.010 / 35.117° | the cones' own angles | 0.41 / 0.40 / 0.041 / 0.33 % | **yes** |
| heap repose angle `theta_r` | **not measured** | — | — | **no basis — NOT_CERTIFIED** |

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_repose_injection.py -q        # 13 passed
PYTHONPATH=src python scripts/verify_repose_injection.py --workers 8    # full matrix (~10 min on 8 workers)
PYTHONPATH=src python scripts/verify_repose_injection.py --figures-only --heap-figure
```

**Open (honest):** the heap matrix has not been run, so the driver has no
measured repose angle yet; and the clump/superquadric path (clump contacts +
rigid-body integration inside this loop) is not implemented. Both are recorded in
`docs/dem_calibration_gap_ledger.md`.