# CASE-DEM-SILO-BEVERLOO / m22628 — orifice scan: inverting (C, k, p) instead of assuming k

**Mission**: m22628 (self-start, generalist seat) · **Anchor**: `CASE-DEM-SILO-BEVERLOO`
(ledger status `open` at checkout — verified, not redone) · **Repo**: TensorDEM ·
**Report**: platform id **20353** · **Cluster job**: `cj4266` (**queued, no data returned**)

**Evidence grade**: harness + estimator = **REAL** (verified against a planted
ground-truth law, 11 tests green). The DEM numbers for this scan = **NOT_RUN**.
No accuracy claim of any kind is made.

---

## 1. Case introduction

Flat-bottom cylindrical silo, centred circular orifice, gravity discharge.
Steady mass flow rate follows Beverloo–Leniger–van de Velde (1961, *Chem. Eng.
Sci.* **15**, 260–269, Eq. (4)):

```
W = C · rho_b · sqrt(g) · (D − k·d)**(5/2)      C = 0.58 (0.55–0.60), k = 1.4 (1.2–1.6)
```

Machine-readable reference: `benchmarks/m20055_beverloo/ref_beverloo_1961.json`
(cited, **NOT re-digitized** — the 1961 paper is not in the repo and there is no
outbound network; `C` therefore carries ~±4 % and `k` ~±14 % of provenance risk).

### The defect this mission targets

Six earlier rounds (m20055 / m20707 / m21033 / m21274 / m22124 …) all evaluated
the reference with **`k = 1.4` treated as known** and compared the **amplitude**
at a single orifice. That design is blind to two of the law's three falsifiable
statements:

| # | statement | why the old design misses it |
|---|---|---|
| S1 | shape: exponent is 5/2 | one point carries no slope |
| S2 | offset: `k·d`, `k ∈ [1.2, 1.6]` | `k` was an input, never an output |
| S3 | independence of filling height `H` | `H` never varied |

Folding all three into one amplitude lets a wrong `k` masquerade as a
`D`-dependent amplitude error — precisely the campaign's fingerprint
(**+20 %** at `D/d = 4`, **−6.9 % / +5.9 %** at `D/d = 5 / 6`).

## 2. What was done

1. **Ledger check** before any work: `CASE-DEM-SILO-BEVERLOO` status `open`;
   anchor title/acceptance untouched.
2. **Pre-registration** (`benchmarks/m22628_beverloo_scan/prereg.md`) written
   **before** any data existed: S1–S5 statements with tolerances, predictions
   P1–P5, and an honest-FAIL clause (no case added or dropped afterwards, no
   re-tuning `k` until the exponent "looks right", a refitted `C` may **not**
   rescue the ledger tolerance).
3. **Estimator first, then physics** (`benchmarks/m22628_beverloo_scan/analyse.py`):
   four nested log-space least-squares models with `C` profiled out analytically
   and standard errors from the numeric Hessian of the profiled objective.

   | model | free parameters | what it tests |
   |---|---|---|
   | A | none (0.58 / 1.4 / 5/2) | point-by-point amplitude error |
   | B | `C` | can one amplitude rescale every orifice? |
   | C | `C`, `k` | is the offset recovered in [1.2, 1.6]? |
   | D | `C`, `k`, `p` | is the exponent 5/2 within ±0.2? |
4. **Kernel**: `tensordem.silo_flow` (Cundall–Strack wall tangential history,
   disordered packing, per-step energy ledger) — **not** `silo_discharge`, whose
   own audit records a systematic +20 % over-prediction where `silo_D/D` is small.
5. **Scan submitted**: 8 shards, `run_compute`, `stage_dir` direct to node,
   8 cores, timeout 7200 s → job **`cj4266`**.
6. **Domain proof**: every one of the 8 cases satisfies `D/d ≥ 4`,
   `silo_D/D ≥ 2.5`, `H/D ≥ 1.5` (`scan.py --list`, asserted by a test).

## 3. Results

### 3.1 DEM discharge rates — **NOT_RUN**

`cj4266` remained `queued` for the whole mission budget (~41 jobs ahead,
scheduler never dispatched a node; re-checked 12 min after submission, including
a 600 s blocking wait). **No DEM flow rate exists for this scan.** S1–S5 are all
undetermined and are recorded as `NOT_RUN`, not as passes.

A local substitute was deliberately **not** used: one case is ≈400 s on one core
(≈76 600 steps × 4 882 particles at `dt = 1.87e-5 s`), i.e. heavy compute that
belongs on the scheduler, not on the 99 control plane.

| case | D/d | W_dem | verdict |
|---|---|---|---|
| D40 | 4.00 | not_run | **NOT_RUN** |
| D45 | 4.50 | not_run | **NOT_RUN** |
| D50 | 5.00 | not_run | **NOT_RUN** |
| D55 | 5.50 | not_run | **NOT_RUN** |
| D60 | 6.00 | not_run | **NOT_RUN** |
| D50_H25 | 5.00 | not_run | **NOT_RUN** |
| D50_H20 | 5.00 | not_run | **NOT_RUN** |
| D50_s2 | 5.00 | not_run | **NOT_RUN** |

### 3.2 What *is* established: the estimator recovers a planted truth

Synthetic data generated from C = 0.56, k = 1.2, p = 5/2, fed through the same
code path the DEM data will take:

| model | C | k | p | max abs rel err |
|---|---|---|---|---|
| B (C only) | 0.644438 | 1.4000 | 2.5000 | 4.692 % |
| C (C, k) | 0.560000 | 1.2000 | 2.5000 | 1.05e-07 % |
| D (C, k, p) | 0.560000 | 1.2000 | 2.5000 | 5.54e-07 % |

Recovery to machine precision — **a self-check of the estimator, not validation
of the Beverloo law** (the truth was planted by the harness itself).

### 3.3 The estimator must be able to fail

`tests/test_m22628_beverloo_scan.py`, 11 tests, all green:

| planted "truth" | required behaviour | result |
|---|---|---|
| p = 2.0 | 3-parameter fit reports p ≈ 2.0 and S1 fails | pass |
| k = 0.0, `k` pinned at 1.4 | residuals must be D-dependent, no single `C` fixes them | pass |
| same, `k` released | residuals collapse below 1e-6 and `C` is inflated | pass |
| W climbs 20 % with `H` | filling-height statement fails | pass |
| W uniformly +20 % | ledger 10 % tolerance fails, reported honestly | pass |
| noise ×10 | `se_p` grows | pass |

The second/third rows **reproduce the mechanism of the campaign's defect**: with
`k` pinned above its true value, `(D − k d)` shrinks, so the compensating `C` is
biased upward. Comparing amplitudes therefore neither detects a wrong `k` nor
exposes it — it just returns a plausible-looking `C`.

## 4. Figures

Figures 1–2 (in report 20353) are generated from the **synthetic ground truth**
above, to show how the fit path and the residual comparison behave. They are
**estimator evidence, not DEM results**. `render_figures.py` (delivered) will
produce the DEM log-log plot, residual bars, `k`-profile, filling-height plot and
per-window steadiness series from `runs/m22628_beverloo_scan/` once `cj4266` lands.

## 5. Notes, limitations, scope

1. **Blocking item is scheduling, not method.** Re-running needs one command;
   all parameters are frozen, reproducible and in-domain.
2. **Reference coefficients are cited, not re-digitized** (§1).
3. **Identifiability of `p`**: 5 points with 3 free parameters leaves 2 dof, so
   `se_p` is optimistic; narrowing the exponent needs more orifices or more seeds.
4. **Documentation inconsistency found (audit)**: `docs/CASE-DEM-SILO-BEVERLOO_m22124.md`
   tabulates d = 2 mm, D = 10 mm, while `benchmarks/m20707_beverloo/campaign.py`
   has `BASE.particle_diameter = 3 mm` and `D5.orifice_diameter = 15 mm`. That
   table is inconsistent with the current code and **should not be cited as a
   numerical basis** any more. This mission generates all numbers from code and
   compares nothing digit-by-digit against those tables.
5. **Soft contacts**: `kn = 5e3 N/m` (1–3 % r overlap), far softer than practice
   (1e5–1e6). The bias this introduces in `C` is not isolated here; m20055's dt
   ladder is circumstantial evidence (refining dt moved +20.6 % → +8.9 %).
6. **Scope**: monodisperse spheres, flat bottom, smooth wall, `D/d ∈ [4, 6]`,
   `H/D ∈ [2, 3.75]`. With non-zero wall friction the silo enters the Janssen
   regime and `H`-independence should break — the best falsifiable follow-up.

## 6. Error verdict

Ledger tolerance is **< 10 %**; the report discipline additionally asks, item by
item, whether **< 3 %** holds.

| quantity | value | basis | rel. error | < 10 %? | **< 3 %?** |
|---|---|---|---|---|---|
| W (D/d = 5, primary) | **not_run** | 0.58·ρ_b·√g·(D−1.4d)^2.5 | — | **NOT_RUN** | **undecidable** |
| W (other 4 orifices) | **not_run** | idem | — | **NOT_RUN** | **undecidable** |
| exponent p | **not_run** | 2.5 | — | **NOT_RUN** | **undecidable** |
| offset k | **not_run** | 1.4 | — | **NOT_RUN** | **undecidable** |
| H-independence | **not_run** | 0 (no H in the law) | — | **NOT_RUN** | **undecidable** |
| estimator inversion (synthetic) | §3.2 | planted C/k/p | ≤ 1e-6 | yes | yes (**not** validation evidence) |

**Explicit answer**: this mission can claim **neither <10 % nor <3 %** — there is
no DEM data. The only sub-3 % number is the estimator's recovery of a truth it
planted itself; it is a self-check and is therefore **excluded from any accuracy
claim**. A refitted `C` is not used to rescue the ledger tolerance, and synthetic
data is never presented as a case result.

## 7. Reproduction

```bash
PYTHONPATH=src python benchmarks/m22628_beverloo_scan/scan.py --list
PYTHONPATH=src python benchmarks/m22628_beverloo_scan/scan.py --shard 1 --nshards 8
PYTHONPATH=src python benchmarks/m22628_beverloo_scan/analyse.py
PYTHONPATH=src python benchmarks/m22628_beverloo_scan/render_figures.py
PYTHONPATH=src python -m pytest tests/test_m22628_beverloo_scan.py -q
```

**Artifacts**: `benchmarks/m22628_beverloo_scan/{prereg.md,scan.py,analyse.py,render_figures.py}`,
`tests/test_m22628_beverloo_scan.py`, `post_report_m22628.py`,
`docs/figures/m22628_validation_{loglog,residuals}.png`,
`runs/m22628_beverloo_scan/` (populated when `cj4266` lands).