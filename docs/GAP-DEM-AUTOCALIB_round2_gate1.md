# GAP-DEM-AUTOCALIB round 2, gate 1 — is the repose response a material function?

**Mission m14868 · repository TensorDEM · branch agent/m14866 · domain: DEM 自动标定（GAP-DEM-AUTOCALIB）**

Round 1 (`docs/GAP-DEM-AUTOCALIB_round1.md`) closed with an honest FAIL of the
calibration criterion and an ordered round-2 plan whose **first** item was:

> *Size convergence: θ_r(N) for N ∈ {2k, 5k, 10k} …; accept the protocol only
> when θ_r changes < 2° between the two largest N.*

This mission executes that gate (shifted to the sizes the shipped protocol can
actually reach: 240–1920 grains) and adds the two questions round 1 could not
answer at a single system size:

* is the measured θ_r **reproducible** (seed-to-seed) at all?
* does it respond to the **material** knob (grain friction) more than to a
  **protocol-geometry** knob (injection radius)?

All thresholds were declared as module constants in
`src/tensordem/dem_calibration/repose_qualification.py` *before* the sweep was
launched and are not fitted afterwards.

## 1. Case and parameters

Forward model: `tensordem.dem_calibration.heap.run_pluviation` (rain deposition,
linear spring–dashpot normal law, Cundall–Strack tangential spring, rolling
spring, Velocity–Verlet, seeded lattice jitter). Material: the shipped `sand`
class of `tensordem.dem_calibration.materials` (twin truth μ = 0.52, e = 0.38,
k_n/k_s = 2.2, μ_r = 0.10, r = 3 mm, ρ = 2650 kg/m³, k_n = 800 N/m).

| quantity | value |
|---|---|
| protocol | pluviation, `layer_size` 12 (clipped to the disk capacity), `settle_per_layer` 200, `drop_height_factor` 1.5 |
| system size N | 240 / 480 / 960 / 1920 grains |
| seeds | 0, 1 (two replicas per configuration) |
| injection radius knob (IRF) | 3.5 r / 7 r (shipped) / 14 r |
| friction knob μ | 0.30 / 0.52 (sand truth) / 0.70 |
| literature anchor | dry-sand repose band **30–37°** (shipped material spec; Beakawi Al-Hashemi & Baghabra Al-Amoudi 2018) |
| platform | loong64, 128 cores shared control plane (6 worker processes; wall times inflated by co-tenancy) |

## 2. Method

Pre-declared cell design (`scripts/m14868_autocalib_r2_sweep.py`, one forward
DEM call per cell, JSONL appended as each future completes, resume-safe):

* **SIZE** — N ∈ {240, 480, 960, 1920} × seeds {0, 1} at μ = 0.52, IRF = 7.
* **DESIGN** — a 3-point factorial at N = 960, seeds {0, 1}:
  (μ, IRF) ∈ {(0.30, 3.5), (0.70, 3.5), (0.30, 14)} → friction signal at fixed
  geometry and geometry signal at fixed friction.
* **REPEAT** — bitwise re-run of the smallest cell (digest witness).

Gates (`repose_qualification.evaluate`, all machine-checkable):

| gate | criterion |
|---|---|
| G1 flank quality | min `r2` of the flank fit ≥ 0.95 |
| G2 solid fraction | median bulk solid fraction ≥ 0.30 (a heap, not a sheet) |
| G3 size convergence | \|θ(N_max) − θ(N_prev)\| ≤ max(2°, 5 %) |
| G4a signal SNR | \|θ(μ=0.70) − θ(μ=0.30)\| / σ_seed ≥ 3 |
| G4b material dominance | material Δθ / geometry Δθ ≥ 3 |
| determinism | identical SHA-256 position digest on an exact re-run |

Artefacts: `artifacts/gap_dem_autocalib_r2/cells.jsonl` (15 cells, raw),
`artifacts/gap_dem_autocalib_r2/qualification.json` (gates + cells + environment),
`artifacts/gap_dem_autocalib_r2/snapshot_n960.json` (deposit cloud with speeds),
figure `docs/verification/figures/autocalib_r2_gate1.png`.

## 3. Results

### 3.1 Size sweep (μ = 0.52, IRF = 7 r)

| N | θ seed 0 [°] | θ seed 1 [°] | seed mean [°] | φ_solid | r² | wall [s] |
|---|---|---|---|---|---|---|
| 240 | 6.34 | 29.47 | **14.05** | 0.021 / 0.033 | 0.564 / 0.936 | 12 |
| 480 | 47.33 | 49.15 | **48.24** | 0.0154 / 0.0073 | 0.913 / 0.758 | 28 / 31 |
| 960 | 66.14 | 22.04 | **44.09** | 0.0065 / 0.0058 | 0.836 / 0.897 | 89 / 96 |
| 1920 | 64.27 | 17.18 | **40.73** | 0.0017 / 0.0015 | 0.684 / 0.940 | 307 / 295 |

Pooled seed noise of θ_r: **σ = 18.77°**. Plateau check on the two largest
sizes: \|40.73 − 44.09\| = **3.37°** versus a 2.04° tolerance → **not converged**.

### 3.2 Knob sensitivities at N = 960 (seeds 0/1)

| (μ, IRF) | θ [°] | mean [°] | φ_solid | r² |
|---|---|---|---|---|
| (0.30, 3.5) | 0.79 / 0.34 | 0.57 | ~1e-4 | 0.138 / 0.052 |
| (0.70, 3.5) | 0.48 / 0.98 | 0.73 | 0.0 / 1e-4 | 0.055 / 0.343 |
| (0.30, 14) | 26.54 / 11.30 | 18.92 | 0.0018 / 0.0022 | 0.684 / 0.524 |

Material signal \|Δθ\| = **0.16°** (SNR 0.009), geometry signal \|Δθ\| =
**18.35°** (SNR 0.98), ratio **0.009**.

### 3.3 Determinism

Exact re-run of the N = 240, seed 0 cell: identical digest
`757b2049…6d84d`, identical θ = 6.338°. The model is **bitwise deterministic**;
the 18.77° spread above is therefore *physical* deposition stochasticity
(which grains roll where), not an integrator or RNG defect.

### 3.4 Gate verdict

| gate | value | threshold | pass |
|---|---|---|---|
| G1 flank quality (min r²) | 0.052 | ≥ 0.95 | **NO** |
| G2 solid fraction (median) | 0.002 | ≥ 0.30 | **NO** |
| G3 size convergence | 3.37° | ≤ 2.04° | **NO** |
| G4a signal SNR | 0.009 | ≥ 3 | **NO** |
| G4b material dominance | 0.009 | ≥ 3 | **NO** |
| determinism | digest equal | equal | yes |

**verdict = `not_qualified`** (5/5 substantive gates failed).

## 4. Error verdict versus the literature anchor — explicit

| key quantity | reference | got | error | < 3 %? |
|---|---|---|---|---|
| θ_r(sand) at μ = 0.52 | 30–37° | seed means 14.05 / 48.24 / 44.09 / 40.73° | −53 % … +30 % (sign flips with N) | **NO** |
| seed reproducibility of θ_r | noise ≪ signal | σ = 18.77° vs band width 7° | 2.7× the whole literature band | **NO** |
| size plateau \|θ(1920) − θ(960)\| | ≤ 2° | 3.37° | 68 % over | **NO** |
| deposition density | a heap (φ ≈ 0.55–0.60) | φ = 0.002–0.033 | 95–99 % low (a sheet) | **NO** |
| friction monotonicity claim (round-1 P1) | 0 inversions | unresolvable: \|Δθ(μ)\| = 0.16° < σ/117 | — | n/a |
| bitwise reproducibility | identical | identical digest | 0 | **yes** |

No acceptance threshold of the campaign node (calibrated-angle error < 10 %) can
be met, and — more importantly — the *measurement* it would rest on is not a
material property at any size tested.

## 5. What changed in the diagnosis (and a correction to round 1)

1. **Round 1's "not monotone in μ" is not established; the map is
   noise-dominated.** Round 1's P1 θ_r ∈ [9.92, 18.45] over μ ∈ [0.2, 0.85] was
   read as non-monotonicity. The replicated measurement here gives σ = 18.77°
   at fixed parameters, i.e. the whole round-1 μ-response (8.5°) is smaller than
   one standard deviation of the noise. A monotonicity test on that map is
   unfalsifiable, and so is the derived `literature_reachable = false`.
2. **The reachability question is the wrong question at these sizes.** At
   N ≥ 480 the seed-mean θ_r (40.7–48.2°) is *above* the 30–37° band; at 240 it
   is far below. The response is not monotone in N either, so "the model cannot
   reach the band" should be restated as **"θ_r(N) of this protocol is not a
   converged function of anything"**.
3. **Root cause is the deposition density, not the contact law.** φ_solid ≤ 3 %
   at every size (gate G2), with the N = 960 snapshot showing ~31 grain layers
   of material spread over a 113 mm footprint — the pile is a fan/sheet whose
   slope records where grains landed, not where friction stopped them. Round-1's
   step 2 ("upgrade to Hertz–Mindlin + rolling resistance") would therefore not
   have addressed the failure; the *protocol geometry* (fixed 1.5 r drop height,
   injection disk of 7 r while the footprint grows to ~38 r) has to be fixed
   first. This reorders round-2: **protocol redesign before contact-law upgrade
   before optimiser**.

### Defect fixed on the way (independent of the science)

`run_pluviation` validated `layer_size` against a hexagon-**area** estimate of
the injection disk but generates a staggered-square lattice clipped to that disk.
For a small injection radius the estimate over-predicts (10 vs the true 7 sites
at IRF = 3.5), the pour then produced a short array and the run died with
`ValueError: shape mismatch: value array of shape (7,3) could not be broadcast to
indexing result of shape (10,3)`. Fixed:

* `heap.pluviation_layer_capacity(case)` — exact lattice capacity (public),
* a short layer now raises `HeapError` naming the capacity, the requested
  `layer_size` and the injection radius,
* `run_pluviation(..., return_positions=True)` additionally returns `speeds`
  (used by the report figure),
* regression tests `tests/test_pluviation_layer_capacity.py` (7 tests, incl. the
  over-prediction 10 vs 7 and the shipped protocol being unaffected).

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_repose_qualification.py \
    tests/test_pluviation_layer_capacity.py -q          # 18 tests, ~11 s
PYTHONPATH=src python scripts/m14868_autocalib_r2_sweep.py --workers 6   # ~13 min, resume-safe
PYTHONPATH=src python scripts/m14868_autocalib_r2_snapshot.py            # ~90 s
PYTHONPATH=src python scripts/m14868_autocalib_r2_figure.py
PYTHONPATH=src python scripts/m14868_autocalib_r2_sweep.py --evaluate-only
```

## 7. Round-3 plan (ordered, each step gated)

1. **Protocol redesign with a size-independent aspect ratio**: injection height
   proportional to the current heap height and injection disk proportional to the
   *current* footprint (or a confined lifting-cylinder with H/R fixed), gate G2
   (φ_solid ≥ 0.30) and a ≥ 10-layer deposit must pass before any θ is quoted.
2. **Convergence with replicas**: σ_seed ≤ 2° and plateau ≤ 2° at two sizes
   (the gates in `repose_qualification` are reusable verbatim; only the cell
   design changes).
3. **Contact law**: only after 1–2 pass, compare linear spring vs
   Hertz–Mindlin + rolling resistance against the 30–37° band.
4. **Optimiser**: BayesOpt / Nelder–Mead (`dem_calibration.inverse`, verified on
   an analytic function) with geometry A/B transfer.

## 8. Limitations

* Sizes reached 1920 grains (single forward call 300 s on the shared control
  plane); round 1's 5k–10k proposal was not affordable in this mission's window
  and is unnecessary until G2 passes — a sheet does not become a heap by adding
  grains at fixed geometry.
* Two seeds per configuration: adequate to *reject* (σ ≫ signal), weak to
  *estimate* σ precisely; the claim used is only "noise exceeds signal by two
  orders of magnitude".
* Wall times reflect a co-tenanted 128-core control plane, not the model.
* The literature band is taken from the shipped material spec; no new
  experimental data was introduced.