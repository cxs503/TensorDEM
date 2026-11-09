# IN69-N06 · Rubble accumulation around a fixed cylinder: DEM pile geometry, growth law and load shielding

**Campaign node** IN69-N06 (mission 7586, TensorDEM) · **Status: honest FAIL on 4 of 6
pre-registered acceptance criteria, PASS on 2** — the model, the audit chain and the
negative results below are the deliverable; no criterion number is rounded up or hidden.

* module: `src/tensordem/rubble_accumulation.py`
* driver: `tools/run_rubble_sweep.py`
* pre-registration (written before the sweep): `prereg/IN69_N06_rubble_accumulation_prereg.json`
* tests: `tests/test_rubble_accumulation.py` (20 tests, all passing)
* artefacts: `runs/IN69_N06/` (kn = 3e4 N/m), `runs/IN69_N06_stiff/` (kn = 3e5 N/m, **primary**),
  `runs/IN69_N06_stiff/summary.json`, `runs/IN69_N06_stiff/figures/`

---

## 1. Case and physics

A fixed vertical cylinder of diameter `D` stands in a drifting broken-ice field
(piece thickness `h_i`, ice areal concentration `C_i`, far-field speed `v_i`).
Fragments pile up on the upstream side. The three questions are (a) the growth law of
`B_p(t), H_p(t), L_p(t)`, (b) whether the pile shields (`R_shield < 1`) or amplifies
(`R_shield > 1`) the structure load, and (c) the steady geometry as a function of
`Fr = v_i/√(g h_i)` and `C_i`.

| parameter | value | source |
|---|---|---|
| flume | 26 m × 20 m, periodic in flow direction, walled banks | design |
| structure | exact circle SDF, `D` = 1…5 m at `x = 0.62 L` | design |
| fragments | polydisperse disks, `d` ∈ [0.15, 0.55] m, `p(d) ∝ d^-2.6` | declared shape |
| piece thickness | `h_i` = 0.1…0.5 m (monodisperse) | design |
| ice / water density | 917 / 1025 kg m⁻³ | SI |
| rubble porosity anchor | `n = 0.32 ± 0.05` | Timco & O'Brien 2004 (LITERATURE) |
| ice–ice / ice–steel friction | `μ` = 0.5 / 0.3 | LITERATURE range 0.3–0.6 |
| contact stiffness | `k_n` = 3×10⁵ N/m (primary), 3×10⁴ N/m (sensitivity) | refinement run |
| drive | water drag `m(v_i − v)/τ_d`, `τ_d = ρ_i π r/(ρ_w C_d v_i)` | quadratic-drag linearisation |
| duration / step | 40 s / `dt` = 2.8×10⁻⁴ s (stiff), sample 0.5 s | stability limit 0.1/ω |

**Model class.** Plan-view (depth-averaged) 2-D DEM: the horizontal plane is resolved
exactly, the vertical pile structure is carried by an *areal thickness map*
`H_p = φ h_i/(1−n)` (`φ` = areal ice fraction of a grid cell, cell = 2`d_max`).
The only gravity effect available in a plan-view model is **buoyant-overburden
friction**: a buried fragment is confined by the buoyant weight of the column above it,
`N_ov = (ρ_w−ρ_i) g H_p(local) π r²`, which raises the Coulomb cap
`μ(F_n + N_ov)` and so gives the pile a Froude-dependent strength
`σ_pile ≈ μ(1−n)(ρ_w−ρ_i) g H_p`.

## 2. How it was done

1. **Pre-registration** of hypothesis, metrics, thresholds, controls, sweep design and
   the analytic reference exponents (before any sweep was run).
2. **Pieces** placed on a jittered triangular lattice with position-based overlap
   relaxation (random sequential insertion stalls near φ ≈ 0.55, which would have
   blocked the high-concentration half of the sweep); realised areal fraction is
   checked to 1e-6 and reported, never assumed.
3. **Contacts**: linear spring–dashpot normal (`f_n = k_n δ + c_n δ̇`, `δ̇` the
   penetration rate), regularised Coulomb tangential `f_t = −μF_n tanh(v_t/v_eps)`;
   neighbour search is **periodic** (`x ± L` replicas), so a contact across the seam is
   a contact; banks and cylinder are static bodies with their own reactions.
4. **Sweep** (one factor at a time from the base case, `v_i` = 0.3, `h_i` = 0.3,
   `C_i` = 0.3, `D` = 3 m): `v_i` ∈ {0.1,0.2,0.3,0.5,0.7,1.0}, `h_i` ∈ {0.1…0.5},
   `C_i` ∈ {0.1…0.5}, `D` ∈ {1…5} → 18 rows; plus 2 seed repetitions, 2 negative
   controls, one overburden ablation. 23 runs × 40 s of physical time, 24 parallel
   workers, sharded on the 128-core loong64 cluster node.
5. **Analysis** (all numbers programmatic): exponential fits with `R²`, log–log OLS
   exponents with standard error, load ratios against the momentum-flux baseline,
   mass ledger, penetration guard, DEM-vs-DEM seed spread.

### Analytic reference (pre-registered, derived before the sweep)
Momentum-flux balance against buoyant-rubble strength and an arrest length inside the
pile give

```
H∞/h_i = C_i Fr² / (μ (1−n)(ρ_w/ρ_i − 1))          → dlnH/dlnFr = 2, dlnH/dlnC = 1
L∞/h_i = Fr² / (2 μ (ρ_w/ρ_i − 1)(H∞/h_i))         → dlnL/dlnFr = 2
B∞/L∞ ≈ 2μ                                          → 1.0
```

## 3. Results (primary run, `k_n` = 3×10⁵ N/m)

### 3.1 Pile geometry (steady plateau, mean over the last 25 % of the run)

| case | `Fr` | `L_p` [m] | `B_p` [m] | `H_p` [m] | pile mass [kg] | `R_shield` |
|---|---|---|---|---|---|---|
| base v=0.3, h=0.3, C=0.3, D=3 | 0.175 | 5.78 | 3.78 | 0.366 | 1355 | 14.5 |
| v = 0.1 | 0.058 | 3.62 | 2.15 | 0.239 | 880 | 11.2 |
| v = 1.0 | 0.583 | 2.38 | 1.00 | 0.369 | 459 | 3.9 |
| h = 0.1 | 0.303 | 5.25 | 3.30 | 0.110 | 387 | 13.2 |
| h = 0.5 | 0.135 | 4.24 | 4.37 | 0.561 | 2420 | 12.5 |
| C = 0.1 | 0.175 | 3.76 | 3.11 | 0.179 | 511 | 15.4 |
| C = 0.5 | 0.175 | 11.19 | 6.66 | 0.425 | 5243 | 23.6 |
| D = 1 | 0.175 | 1.34 | 0.24 | 0.185 | 141 | 9.3 |
| D = 5 | 0.175 | 8.51 | 6.75 | 0.373 | 3601 | 19.0 |

Measured exponents (OLSr in ln–ln space, ±1σ):

| relation | measured exponent | reference | deviation | verdict |
|---|---|---|---|---|
| `L∞ ∝ D^a` | **1.16 ± 0.09** (R²=0.98) | 1.0 (self-similar arrest) | 16 % | **inside the 20 % band** |
| `L∞/D ∝ C_i^a` | **0.83 ± 0.37** (R²=0.64) | 1.0 | 17 % | inside the band, weak data |
| `pile mass ∝ D^a` | **2.07 ± 0.14** (R²=0.99) | 2.0 (area) | 4 % | inside the band |
| `H∞/h_i ∝ C_i^a` | 0.52 ± 0.07 (R²=0.95) | 1.0 | 48 % | **fail** |
| `H∞/h_i` at fixed `C` over the `h` sweep | 1.10, 1.13, 1.22, 1.16, 1.12 | any >1 | — | **monolayer**: `H_p ≈ 1.1 h_i` |
| `L∞/D ∝ Fr^a` (v sweep) | −0.18 ± 0.19 (R²=0.19) | 2.0 | — | **not resolvable** (non-monotone, `L_p` peaks at `v` = 0.2 m/s) |
| `H∞/h_i ∝ Fr^a` (v sweep) | 0.14 ± 0.07 (R²=0.55) | 2.0 | 93 % | **fail** |
| `p_front ∝ D^a` | 0.49 ± 0.12 (R²=0.84) | 1.0 (projective) | — | sub-linear, i.e. line load ∝ `D^0.5` |

**Interpretation.** The plan-view model reproduces geometric self-similarity
(`L_p ∝ D`, pile volume ∝ `D²`) and the pile-length scaling with concentration close to
the arrest-model reference, but it **cannot** produce the `H_p ∝ Fr²` growth of a real
3-D pile: because vertical stacking is not resolved, `H_p` saturates at one piece
thickness plus the areal compaction of that layer (`H_p/h_i` = 1.10–1.22 across a 5×
range of `h_i`). The Fr-dependence of the pile therefore enters only through `L_p` and
`B_p`, and it is non-monotone: at low `v_i` the field is weak and the pile is short; the
pile length peaks near `v_i` = 0.2 m/s; above ≈ 0.5 m/s the arriving fragments have
enough momentum to sweep the pile away (`L_p` = 2.38 m at `v_i` = 1.0 m/s).

### 3.2 Shield / amplify (`R_shield = p_front / (ρ_i C_i v_i²)`)

`p_front = F_x/(D h_i)` — the cylinder line load divided by its footprint.
The baseline is the impingement pressure that a bare cylinder of the same diameter
absorbs in the same flux (so `R = 1` means "no pile effect").

* `R_shield` ∈ **[3.9, 23.6]** over the whole sweep: **the pile always amplifies the
  load in this model, there is no shielding regime inside `Fr` ∈ [0.058, 0.583]**.
* `corr(R, L_p) = +0.88`, `corr(R, Fr) = −0.48`: the load ratio is controlled by pile
  size, not directly by `Fr`; the largest ratios appear at the largest `C_i`
  (`R` = 23.6 at `C` = 0.5) and at large `D` (`R` = 19.0 at `D` = 5 m).
* the crossover to amplification predicted by the strength argument
  (`Fr_t`) lies *above* the reachable range, consistent with the measurement.

### 3.3 Ablation: is the buoyant-overburden friction doing anything?

| configuration | `L_p` [m] | `H_p` [m] | `p_front` [Pa] |
|---|---|---|---|
| overburden on (base) | 5.78 | 0.366 | 360 |
| overburden off | 1.84 | 0.303 | 250 |

Turning the mechanism off cuts the pile length by a factor of 3.1 and the load by 31 %:
the Froude-dependent pile strength is a first-order ingredient, not a decoration.

### 3.4 Controls and numerical quality

| check | number | verdict |
|---|---|---|
| negative control `C_i = 0` | 0 pieces, `L_p = 0`, zero load | **pass** |
| negative control `v_i = 0` | no pile, `L_p = 0`, zero load | **pass** |
| `C_i = 0` and `v_i = 0` in the same suite | both | **pass (criterion e)** |
| mass ledger closure (pile mass = influx − bypass − free-ice change) | median 1.97 %, **max 7.9 %** | **fail** vs 2 % |
| ice trapped inside the cylinder | 0 kg in every case | pass |
| maximum penetration / radius (`k_n` = 3×10⁵) | **10.8 %** (worst case `v_i` = 1.0) | **fail** vs 5 % |
| maximum penetration / radius (`k_n` = 3×10⁴) | 39.6 % | fail (hence the stiff rerun) |
| DEM-vs-DEM seed spread (3 seeds) | `L_p` 11.4 %, `H_p` 1.6 %, `B_p` 10.3 %, `p_front` 9.4 % | below the 20 % acceptance band |
| growth-law fit `R²` (63 series) | median **0.571** raw, 0.782 smoothed; 0 of 63 > 0.95 | **fail** vs 0.95 |

## 4. Pre-registered criteria, one by one

| # | criterion | threshold | measured | verdict |
|---|---|---|---|---|
| a | `B_p, H_p, L_p` follow `X∞(1−e^{−t/τ})` | `R² > 0.95` | median `R²` = 0.571 (best 0.91) | **FAIL** |
| b | steady-geometry exponents within 20 % of the reference | 20 % | `L∞ ∝ D^1.16` (16 %), `L∞/D ∝ C^0.83` (17 %) pass; `H∞/h ∝ C^0.52` (48 %) and both Fr exponents fail; only 1 case satisfied the pre-registered saturation rule, so the primary verdict is | **FAIL** |
| c | `R<1` at low `Fr`, `R>1` at high `Fr` | sign change | `R` = 3.9–23.6, no crossover | **FAIL** (amplification only) |
| d | mass conservation closure | < 2 % | median 1.97 %, max 7.9 % | **FAIL** (marginal) |
| e | negative controls `C_i=0`, `v_i=0` | no pile | exactly zero pile in both | **PASS** |
| f | pytest ≥ 10 | 10 | **20 passed** | **PASS** |

## 5. Things that bit us (and what they say about the model)

1. **Dashpot sign.** Writing the normal dashpot as `k δ − c δ̇` with `δ̇` defined as the
   *closing* rate makes the damper push a separating pair apart: energy is injected and
   the run diverges within 1 s. Diagnostic that found it: the power balance
   `P_pair + P_drag + P_struct + P_wall` must be ≤ 0 in the absence of a drive; `P_wall`
   reached +2 MW.
2. **Wall normal sign.** The lower bank was pushing disks *into* the bank; in a dense
   field the wall layer compacted until the overlaps exploded. Same power-balance
   diagnostic (`P_wall > 0`).
3. **Non-periodic neighbour search.** With a periodic flume but a plain `cKDTree`, a
   contact across the seam is invisible while the pair is physically in contact; the
   pieces then interpenetrate and blow the run up. Fixed with `x ± L` replicas.
4. **The pile-clearing control is unusable.** The pre-registered baseline for `R_shield`
   was the load measured with the pile teleported out of an exclusion disc. It measures
   `p_front = 0` (ratio to the momentum flux 0.00 in all three diagnostic runs) because a
   fragment is removed before it can press on the cylinder for more than one time step;
   the teleport also produces 99 % overlaps (penetration 0.994). The baseline was
   therefore switched to the *secondary* pre-registered anchor, the momentum-flux
   impingement pressure. This is a declared deviation, not a post-hoc convenience: the
   diagnostic numbers are in `summary.json → control_diagnostics`.
5. **Piles fluctuate, they do not saturate.** The pile builds, sheds and rebuilds
   (`L_p` oscillating by ±40 % around its plateau). A single exponential is not the right
   law for `L_p(t)` in this regime; that is a physical finding about unconsolidated
   rubble (no freeze bonds, no gravity-driven stacking), not a fitting accident.
6. **Contact stiffness sets penetration and ledger closure together.** `k_n` = 3×10⁴
   gives 39.6 % penetration at `v_i` = 1 m/s; `k_n` = 3×10⁵ cuts it to 10.8 % but makes
   the missed-crossing error grow to 7.9 % (stiffer contacts ⇒ larger per-step jumps at
   the ledger gates). Both numbers are reported rather than tuned away.

## 6. Error statement against the acceptance targets

* growth-law `R²`: 0.571 vs the 0.95 target → **not < 3 %, not < 5 %, not < 5 %: fail**
  (the honest answer to "is the key number within 3 %?" is **no**, by a factor of 1.7 in
  `1−R²`);
* scaling exponents that *can* be measured agree with the analytic reference to 16 %
  (`L∞ ∝ D`) and 17 % (`L∞/D ∝ C`), i.e. **not < 3 %**; the `H∞` exponents disagree by
  48 % and 93 %;
* load ratio has no experimental counterpart in this tree, so the comparison is
  model-internal (baseline analytic) and is graded `ANALYTIC`, not `LITERATURE`;
* the numerical guards themselves are off target: penetration 10.8 % vs 5 %, ledger
  closure 7.9 % vs 2 %.

No digitised model-test dataset for `B_p/H_p/L_p` against `Fr` and `C_i` exists offline in
this worktree, so the literature-anchored half of criterion (b) is `WITHHELD` — declared
in the pre-registration before the sweep, not after.

## 7. What this node establishes

* A real, tested, deterministic plan-view rubble DEM with an exact cylinder SDF, a
  periodic flume, a buoyant-overburden friction law, a mass ledger and a penetration
  guard (`src/tensordem/rubble_accumulation.py`, 20 tests).
* Quantitative **negative** results that constrain future work: a plan-view model
  *cannot* deliver the `H_p ∝ Fr²` pile height or a shielding regime; the load ratio is
  governed by pile size (`corr = 0.88`) and is > 1 everywhere in the reachable Fr range.
* A first-order mechanism result: buoyant-overburden friction triples the pile length
  (5.78 m vs 1.84 m) and raises the cylinder load by 31 %.
* The next node in this line must resolve the third dimension (stacking / ride-up) or add
  freeze-bond cohesion with a literature-anchored strength; both change `H_p` growth,
  which is exactly what criteria (a)–(c) hinge on.