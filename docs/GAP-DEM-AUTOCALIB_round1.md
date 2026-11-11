# GAP-DEM-AUTOCALIB — round 1: is the repose forward model calibratable at all?

**Mission m14680 · repository TensorDEM · status: honest FAIL of the round-1 acceptance
criterion, with a quantified root cause and a concrete round-2 plan.**

The mission asks for the closed loop

> micro parameters (stiffness / friction / damping) → macro responses (angle of repose /
> elastic modulus) → optimiser (Bayesian or Nelder–Mead) against **literature**
> experimental values → **calibrated angle prediction error < 10 %**

Round 1 deliberately spent its budget on the step that allBayesOpt/MCMC machinery
inherits: *is the forward map (i) measured at rest, (ii) monotone in the parameter that is
supposed to explain the response, and (iii) able to reach the literature band at all?*
The answer, measured below, is **no to (ii) and (iii)**, so the optimiser was not pointed at
literature data in this round; doing so would have produced a "calibrated" number that is a
property of the search box, not of the model (the exact failure mode the previous
`docs/p360w1_n03_bayesian_calibration_report.md` §163 already suspected but did not
quantify).

## 1. Case and parameters

Forward model: `tensordem.dem_calibration.heap` (linear spring–dashpot normal law,
Cundall–Strack tangential spring, rolling spring; Velocity–Verlet, sparse `cKDTree` pairs).
Material: the shipped `sand` class of `tensordem.dem_calibration.materials`.

| quantity | value |
|---|---|
| grain radius `r` | 3.0 mm |
| density ρ | 2650 kg/m³ |
| rolling friction μ_r | 0.10 |
| contact stiffness `k_n` | 800 N/m (gravity-scaled soft sphere) |
| container radius | 21 mm, fill height 42 mm (**7 grain diameters**) |
| particles | **275** (pluviation protocol of the shipped pipeline) |
| varied | μ ∈ {0.20, 0.35, 0.52, 0.70, 0.85}; e = 0.38, k_n/k_s = 2.2 fixed |
| literature anchor | dry sand angle of repose **30–37°** (band used by the shipped material spec; a review of granular repose measurements, Beakawi Al-Hashemi & Baghabra Al-Amoudi 2018, gives 30–35° for dry sand) |

## 2. Method

Three pre-declared, seeded experiments (`scripts/m14680_autocalib_diagnosis.py`, all
numbers written programmatically to `artifacts/gap_dem_autocalib/diagnosis.json`):

* **P1 friction response** — θ_r(μ) at fixed e, k_n/k_s. A calibratable map must be
  strictly increasing: rank correlation > 0, zero inversions.
* **P2 replica noise** — same parameters, three seeds. The spread is the noise floor; a
  signal smaller than it cannot be inverted.
* **P3 rest criterion** — lifting-cylinder collapse measured (a) on the fixed
  `hold+lift+settle` window of the shipped pipeline and (b) with a new adaptive
  termination `ke_stop` (`K / released PE < ke_stop`), i.e. "the heap is at rest".

New capability added for P3 (minimal, backwards compatible):
`run_heap(..., ke_stop=None, settle_check_every=25)` in
`src/tensordem/dem_calibration/heap.py` returns `settled`, `steps_used`, `ke_stop`; with
the default `None` the trajectory and the SHA-256 digest are bit-identical to before
(asserted in `tests/test_dem_repose_settle_termination.py`, 5 tests).

## 3. Results

**P1 — the map is not monotone** (artifact fields, zero hand-copying):

| μ | 0.20 | 0.35 | 0.52 | 0.70 | 0.85 |
|---|---|---|---|---|---|
| θ_r [°] | 13.83 | **9.92** | **18.45** | 15.88 | 14.89 |
| `ke_ratio` | 0.295 | 0.250 | 0.294 | 0.326 | 0.329 |

Spearman(μ, θ_r) = **0.50**, 3 of 4 intervals non-increasing, total simulated band
**9.92–18.45°** versus the literature band **30–37°** → `literature_reachable = false`.

**P2 — noise floor almost equals the signal**: replica θ_r at identical parameters =
18.45 / 17.27 / 13.45° → spread **5.00°**, while the whole μ-sweep spans **8.53°** →
signal-to-noise **1.70**. Any least-squares fit through this map attributes seed noise to
μ.

**P3 — the fixed window is not at rest, and the settled state is not a heap**:

| μ | fixed window (shipped) | `ke_stop = 0.02` |
|---|---|---|
| 0.40 | θ = 0.78°, k_e/PE = 1.1e-3, r² = 0.42 | θ = 2.09°, k_e/PE = 1.9e-2, r² = 0.36 |
| 0.65 | θ = 1.18°, k_e/PE = 5.7e-4, r² = 0.59 | θ = 4.53°, k_e/PE = 2.0e-2, r² = 0.53 |

The settled deposit has a bulk solid fraction of **1–3 %** — a monolayer; 275 grains at a
7-grain-diameter column height do not contain enough layers to form a standing flank. The
shipped pluviation heap is the only protocol that produces a fittable flank, and it is
simultaneously (a) never at rest at the fixed window (`ke_ratio ≈ 0.25–0.33`, i.e. 25–33 %
of the released potential energy is still kinetic when θ_r is "measured") and (b)
noise-dominated.

Figure: `artifacts/gap_dem_autocalib/fig1_repose_diagnosis.png`.

## 4. Error verdict — explicit

| key quantity | reference | got | error | < 3 %? |
|---|---|---|---|---|
| angle of repose, sand, best simulated value | 30–37° (literature) | 18.45° | **38–50 % low** | **NO** |
| best achievable after free choice of μ ∈ [0.1, 0.8] | 30–37° | 18.45° | ≥ 38 % | **NO** |
| monotonicity of θ_r(μ) | 0 inversions | 3 inversions | — | **NO** |
| signal-to-noise of θ_r(μ) | > 3 (minimum for a stable inversion) | 1.70 | — | **NO** |
| `ke_stop` termination vs fixed window (code) | criterion met ⇔ `settled = True` | 5/5 tests | 0 coding defects found | yes (unit level only) |

Calibrated-angle error < 10 % is therefore **not demonstrated**; the 10 % criterion is
unreachable while the forward model's whole reachable band lies below the literature band.

## 5. `fail_stage` / `failure_reason` (input to the round-2 research order)

```json
{"fail_stage": "forward_model_qualification (before the optimiser stage)",
 "failure_reason": "the shipped repose forward model cannot reach the literature band and is not monotone in mu: theta_r in 9.92-18.45 deg vs 30-37 deg literature; Spearman(mu,theta)=0.50 with 3/4 non-increasing intervals; replica spread 5.00 deg vs signal 8.53 deg (SNR 1.70); the pluviation heap is not at rest when measured (ke_ratio 0.25-0.33) and the lifting-cylinder collapse at n=275 grains settles into a monolayer (phi_solid 1-3 %, theta 0.8-4.5 deg).",
 "tried": ["5-point mu sweep on the pluviation protocol (300-800 s wall)",
           "3-seed replica study at fixed parameters",
           "lifting-cylinder collapse: fixed window vs adaptive ke_stop termination",
           "cone-relaxation protocol (does not relax: 50.9 deg = initial cone angle)"],
 "missing": ["system size: >= 5000 grains so a flank has >= 10 grain layers (runtime ~1e2 s per forward call -> cluster_queue sharding, not the 99 control plane)",
             "a size-convergence gate (theta_r(N) plateau) before any optimisation",
             "reported model form: soft-sphere k_n=800 N/m gives phi_solid ~0.25 in a dense pack, far from a real sand (0.6); a Hertz-Mindlin/rolling-resistance law with realistic stiffness is probably needed for the literature band",
             "a literature anchor for the second macro response (elastic modulus) - the shipped pipeline returns phi_solid, not E, so the 'elastic modulus' leg of the mission has no forward model yet"]}
```

## 6. What *was* delivered in round 1

1. **`run_heap(..., ke_stop=...)` adaptive rest termination** (+ `settled`, `steps_used`,
   `ke_stop` in the result) — a measured response can now be qualified as "at rest"
   instead of "at the end of an arbitrary clock". `tests/test_dem_repose_settle_termination.py`
   (5 tests, 15 s): digest invariance, never-met, budget cap, tolerance monotonicity,
   check-interval invariance.
2. **Reusable diagnosis harness** — `scripts/m14680_autocalib_diagnosis.py` (≈115 s,
   deterministic, writes the JSON artefact) and `scripts/m14680_autocalib_figure.py`.
3. **Quantified root cause** that turns the vague remark "simplified DEM produces
   unrealistically low angles" (p360w1_n03 §163) into four machine-checkable gates
   (reach, monotonicity, SNR, at-rest) that any future calibration must pass first.

## 7. Round-2 plan (ordered, each step with its own gate)

1. **Size convergence**: θ_r(N) for N ∈ {2k, 5k, 10k} grains, pluviation and lifting
   cylinder, `ke_stop = 2 %`; accept the protocol only when θ_r changes < 2° between the
   two largest N (shard over `run_cluster`, one forward call per shard).
2. **Contact-law upgrade**: Hertz–Mindlin normal + rolling resistance at realistic
   stiffness, re-run the same gates (reach / monotone / SNR ≥ 3 / at rest).
3. **Literature anchoring**: dry sand θ_r = 34° ± 2° and a macro elastic modulus from a
   uniaxial confined-compression forward model (new `dem_calibration/uniaxial.py`,
   E = dσ/dε over the elastic range) anchored to the sand modulus band 30–50 MPa.
4. **Only then** the optimiser: GP/EI BayesOpt (`dem_calibration.inverse`, already
   verified on an analytic function) or Nelder–Mead, calibrated on geometry A, verified on
   geometry B (fixed 10 % angle criterion) — plus the calibration curve and the
   convergence table as artefacts.

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_dem_repose_settle_termination.py -q
PYTHONPATH=src python scripts/m14680_autocalib_diagnosis.py     # ~115 s, writes the artefact
PYTHONPATH=src python scripts/m14680_autocalib_figure.py
```