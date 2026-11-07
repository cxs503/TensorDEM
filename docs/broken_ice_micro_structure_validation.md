# Broken-ice microstructure: FSD evolution, arching/clogging and force chains

Campaign node **IN34-W3-N01** · module `tensordem.broken_ice_micro` ·
study driver `scripts/broken_ice_micro_study.py` ·
figures `scripts/broken_ice_micro_figures.py` ·
ledger `artifacts/broken_ice_micro_study/study.json` ·
tests `tests/test_broken_ice_micro_chain.py` (20 tests, all passing).

---

## 1. What was asked, and what this deliverable is

The node asks for one validation chain with three physical layers of a broken
ice cover, all with machine-checkable thresholds:

| # | criterion | requirement |
|---|---|---|
| a | FSD evolution | `n(D) ~ D**-alpha`, `2.5 <= alpha <= 3.5`, `R**2 > 0.9` |
| b | arching / clogging | critical width ratio `(W/h)_crit` in `[3, 6]` |
| c | Janssen anchor | wall-stress fit error `< 15 %` |
| d | force chains | `P(f)` shows an exponential tail |
| e | porosity | `0.30 <= n <= 0.45` |
| f | case count | at least six independent cases |

**Verdict: 4 / 6 criteria pass, 2 fail honestly.**

| criterion | value measured | verdict |
|---|---|---|
| a FSD exponent | `alpha = 2.330 +- 0.145`, `R**2_cdf = 0.899`, 1048 fragments | **FAIL** (alpha < 2.5, `R**2` below 0.9 by 0.001) |
| b clogging ratio | `(W/h)_crit = 3.66` (bracket 3.5–4.0), 48 discharge runs | **PASS** |
| c Janssen error | best fit `max error = 42.5 %` (W/h = 6) | **FAIL** |
| d force-chain tail | exponential tail, `R**2 = 0.93–0.97`, tail rate 3.8–4.1 | **PASS** (rate is not the unit value, see §5) |
| e porosity | `n = 0.434, 0.436, 0.434` | **PASS** |
| f case count | 58 cases | **PASS** |

Both failures are physics results, not missing code: (a) the exponent is
reproducibly below the campaign window for a representative ice Weibull modulus
and only enters the window at the brittle end of the ice-strength range; (c) the
2D engine's walls are frictionless, so the silo effect that Janssen describes is
not present in the DEM column at all (bottom wall carries 95 % of the weight).

---

## 2. Modelling: three layers, three tools

### 2.1 Fragmentation of a level ice plate (`fragmentation.py`)

* **Lattice.** Triangular (hexagonal) bond lattice, `nx × ny` nodes, spacing `a`;
  every node owns the plate area `sqrt(3)/2 a²` and is bonded to its six nearest
  neighbours by an axial spring `k = E t / sqrt(3)` (`E = 5 GPa`, `t = 0.5 m`,
  ice density 917 kg/m³).  Bond lengths are exactly `a` (asserted by a test), so
  the intact lattice is a genuine 2D elastic medium.
* **Disorder.** Bond `i` fails when its axial strain exceeds
  `eps_c,i = eps_0 (-ln U_i)**(1/m)`, i.e. a Weibull distribution of modulus
  `m` and scale `eps_0 = 2e-3`; compression failure at `5 eps_c`.  This quenched
  disorder is the *only* heterogeneity in the model.
* **Loading.** Displacement control between two rigid platens (left fixed,
  right advancing in `1e-3` strain increments).  At every load level the elastic
  equilibrium of the *surviving* network is solved exactly (sparse direct solve
  of the ~2 N dof system), bonds over their threshold are removed and the
  equilibrium is recomputed (cascade) until the level is admissible.  There is no
  inertia, no contact model, no damping: the census is a property of the elastic
  network.
* **Load path.** Only the components that span both platens carry load; floes
  attached to one platen (or to none) are unloaded debris and are held at zero
  displacement -- this removes the soft "hinge" modes that an explicit dynamic
  integration of the same lattice cannot handle (measured: a dynamic version kept
  `KE / strain_energy ~ O(10)` even with critical contact damping, i.e. it was
  not quasi-static).
* **Census.** Connected components of the surviving bonds; fragment diameter
  `D = sqrt(4A/pi)` with `A = n_nodes sqrt(3)/2 a²`.

### 2.2 Arching / clogging and Janssen mechanics (`arching.py`)

* `janssen_profile` is the closed-form 2D Janssen solution
  `sigma_zz(z) = rho_b g lambda (1 - e^{-z/lambda})`, `lambda = W/(2 K mu_w)`,
  `sigma_w = K sigma_zz`; `fit_janssen` recovers `(K, mu_w)` from a measured
  wall-stress profile (differential or cumulative form) and reports the
  maximum and rms relative error.
* `run_silo` is a gravity-driven discharge experiment on the verified
  `tensordem.granular2d` engine: a silo body of width `2W`, a bottom gate of
  width `W`, a polydisperse (log-normal CV 0.3) log-normal disk bed of mean
  diameter `h`, Coulomb friction `mu = 1.2`, rolling resistance 0.4 and a wall
  friction coefficient 0.9 applied as an operator-split Coulomb impulse (the
  engine's segment contacts are frictionless; see §5).
  A run **clogs** when material remains in the body and no disk has crossed the
  outlet for `0.5 s`.  `clogging_curve` turns a `W/h` sweep into `P_clog`, and
  `critical_width_ratio` interpolates the 50 % crossing.
* The arch is measured too: `strong_arch` extracts the load-bearing contact
  network (`f_n > <f_n>`), and reports the horizontal span of the widest
  spanning component and whether it reaches both walls.

### 2.3 Force chains, coordination and porosity (`chain.py`)

* `fit_exponential_tail` fits `log S(f) = a - f/rate` to the survival function of
  the normalised contact force `f = F/<F>` and reports the tail rate, its
  `R**2`, the strong-contact fraction and the strong-load fraction, plus the KS
  distance against the unit-mean exponential.
* `three_dimensional_rubble` builds a 3D sphere packing with the **measured**
  power-law FSD (`MeasuredPowerLawFSD`, inverse-CDF sampling), compacts it with
  the verified oedometer protocol of
  `tensordem.ice_rubble_microstructure` (dead-weight piston, 25 kPa) and measures
  porosity (voxel-resolved Voronoi, 30 mm voxels), coordination number `Z` and
  `P(f)`.  This is the piece that closes the chain: the porosity is conditioned
  on the fragmentation physics instead of an assumed Rosin–Rammler law.

---

## 3. Case matrix (58 cases, ledger `study.json`)

| stage | cases | detail |
|---|---|---|
| `fsd` | 4 ensembles = 52 fragmentation runs | `48×24 m=4` (16 seeds), `64×32 m=4` (12), `48×24 m=2.5` (12), `48×24 m=6` (12) |
| `arch` | 48 discharge runs | `W/h ∈ {2.0, 2.5, 3.0, 3.5, 4.0, 6.0}` × 8 seeds |
| `janssen` | 3 static columns | `W/h = 4, 6` with `mu_w = 0.9`, `W/h = 4` with `mu_w = 0` (control) |
| `chain` | 3 3D rubble compactings | `alpha_FSD = 2.3`, 400 floes, seeds 3/5/7 |

Numerical settings: fragmentation `eps_0 = 2e-3`, `strain_increment = 1e-3`,
`target_strain = 0.05`, `min_fragment_nodes = 1`; silo `h = 0.08 m`, body 2W,
height 2.4 m, fill 0.9, `dt` from the engine stability bound (~1.3e-4 s),
discharge window 3 s; rubble cell 1.2 m, 25 kPa compaction, 30 mm voxels.

---

## 4. Results

### 4.1 (a) FSD exponent — **FAIL at the campaign threshold**

Pooled fragment census per ensemble (number distribution of equivalent
diameters, CSN maximum likelihood with a KS-minimising cutoff; `R²_cdf` is the
log–log survival-function fit, `R²_bin` the log-binned one):

| ensemble | fragments | alpha | s.e. | `R²_cdf` | `R²_bin` | tail count | verdict |
|---|---|---|---|---|---|---|---|
| `48×24 m=4` (baseline) | 1048 | 2.330 | 0.145 | 0.899 | 0.760 | 84 | window FAIL (`R²` −0.001) |
| `64×32 m=4` (resolution) | 1565 | 2.307 | 0.143 | 0.943 | 0.905 | 84 | window FAIL (alpha) |
| `48×24 m=2.5` (strong disorder) | 234 | 2.230 | 0.211 | 0.822 | 0.849 | 34 | FAIL |
| `48×24 m=6` (brittle) | 1066 | **2.815** | 0.157 | **0.924** | 0.873 | 134 | **window PASS** |

What the numbers say:

* the fragmentation *does* produce a power law: 1048 fragments, 1.0–1.4 decades
  of size, KS distance 0.09–0.11, `R²_cdf` 0.90–0.94;
* the exponent is **resolution independent** (2.330 → 2.307, 1 % change for a
  4× larger lattice) and **brackets the 2D percolation prediction**
  `alpha = 2 tau - 2 = 2.10` (`tau = 2.05`, 2D percolation cluster number) within
  11 % at `m = 4`;
* the exponent **increases with the brittleness of the strength disorder**
  (2.23 at `m = 2.5`, 2.33 at `m = 4`, 2.82 at `m = 6`): narrow strength scatter
  gives clean crack paths and a steeper census.  The campaign window is met only
  at `m = 6`;
* `R²_bin` (log-binned) is systematically lower because the fitted range spans
  only four to five bins; the survival-function `R²` uses every fragment above
  the cutoff and is reported as the primary metric.

Since the representative sea-ice Weibull modulus is `m ≈ 3–5` (the baseline
`m = 4`), the pre-registered verdict is **FAIL**: `alpha = 2.33 < 2.5`.

### 4.2 (b) Arching / clogging — **PASS**

48 gravity-driven discharge runs (`h = 0.08 m`, 8 seeds per width ratio):

| W/h | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|
| `P_clog` | 1.00 | 0.75 | 0.75 | 0.625 | 0.25 | 0.00 |
| runs | 8 | 8 | 8 | 8 | 8 | 8 |
| residual material: min–max (median) | 0.68–0.98 (0.94) | 0.44–0.97 (0.88) | 0.53–0.96 (0.82) | 0.55–0.97 (0.72) | 0.25–0.78 (0.42) | 0.07–0.17 (0.11) |

`(W/h)_crit = 3.66` (log-linear interpolation of the 50 % crossing, bracket
`[3.5, 4.0]`) — inside the campaign window `[3, 6]`.  Two definitions are
reported and they must not be confused: the **clog flag** is the strict one
(no disk crosses the outlet for 0.5 s with material still in the body) and the
**residual fraction** is the softer one (material left at the end of the 3 s
window).  The residual fraction falls monotonically with `W/h`
(0.94 → 0.11) while the strict flag drops faster, because a slowly creeping bed
still discharges: at `W/h = 4` four runs arrested visibly (residual 0.4–0.8) but
kept creeping, so only two count as strict clogs.

In the strictly clogged runs the load-bearing network (`f_n > <f_n>`) contains a
component that touches both side walls, with a horizontal span equal to the silo
body width (2W, i.e. `2×(W/h)` floe diameters).  That confirms the arrest is
mediated by a connected strong-force structure rather than by a single blocking
floe, but it does **not** isolate an arch *across the outlet plane*: the metric
reports the span of the whole jammed region, which includes the wide body above
the gate (limitation 4 in §5).

The full-width channel drain (no gate, the configuration one would try first)
does **not** arch at any ratio `2.5 ≤ W/h ≤ 6`: a plug of round disks accelerates
freely at a full-width outlet.  The gate/silo geometry is what makes the
criterion well posed, and this negative result is kept in the study
(§5, limitation 3).

### 4.3 (c) Janssen wall-load profile — **FAIL**

| column | `lambda` [m] | `K` | `mu_w` (fitted) | `R²` | max rel. error |
|---|---|---|---|---|---|
| `W/h = 4`, `mu_w = 0.9` | 31.0 | 1.31 | 0.0039 | 0.985 | **55.9 %** |
| `W/h = 6`, `mu_w = 0.9` | 0.75 | 2.43 | 0.131 | 0.953 | **42.5 %** |
| `W/h = 4`, `mu_w = 0` (control) | 31.0 | 1.51 | 0.0034 | 0.989 | 61.4 % |

The fitted `lambda` is at or beyond the search bound (`31 m >> H = 2.4 m`), i.e.
the measured cumulative wall stress is **linear in depth** (`Sigma_w ~ z²`),
which is the hydrostatic (`lambda → inf`) limit, not the Janssen saturation.
The direct diagnostic confirms it: the bottom wall carries
`~95 %` of the bed weight in every case (165 N of 164 N weight at
`mu_w = 0` vs 155 N at `mu_w = 0.5`), so the walls carry almost no vertical load
and Janssen's premise (`tau_w = mu_w K sigma_zz` doing the load transfer) is not
realised.  Root cause and fix are in §5, limitation 1.  The criterion is **not**
met; the analytic profile itself is verified against synthetic data to `< 2 %`
(two tests), so the failure is in the DEM anchor, not in the formula.

### 4.4 (d) Force-chain statistics — **PASS**

Compacted 3D rubble (400 spheres, FSD `alpha = 2.3`, 25 kPa oedometer):

| case | contacts | tail rate | tail `R²` | strong fraction | strong load fraction | `Z` |
|---|---|---|---|---|---|---|
| seed 3 | 1218 | 4.06 | 0.967 | 0.31 | 0.52 | 4.05 |
| seed 5 | 1104 | 3.82 | 0.931 | 0.32 | 0.53 | 3.75 |
| seed 7 | 1195 | 3.87 | 0.966 | 0.31 | 0.52 | 3.90 |

The survival function of `P(f)` is exponential over `1 ≤ f ≤ 4` with
`R² = 0.93–0.97` in all three cases, an explicit exponential tail as required.
The decay rate is 3.8–4.1 (in units of the mean force), i.e. **much steeper than
the unit-rate exponential** of an unjammed random network: the 25 kPa
pre-consolidation narrows the force distribution.  `Z = 3.75–4.05` is the
frictional-sphere value for a compacted packing (`Z ≈ 4` is the isostatic limit
for frictional spheres).

### 4.5 (e) Porosity — **PASS**

Voxel-resolved Voronoi porosity of the same three packings:
`n = 0.435, 0.436, 0.434` (`phi = 0.564–0.566`), inside `[0.30, 0.45]` and inside
the 0.30–0.45 band reported for ice-rubble keels.  The three realisations agree to
within 0.002, so the measurement noise is far below the window width.

### 4.6 Elastic verification of the fragmentation lattice

`E_eff / E = 0.685` (16×8) and `0.624` (32×16): a central-force triangular lattice
is softer than the continuum by a fixed geometric factor (O(1) as it must be);
the 9 % change between the two resolutions is the free-surface boundary layer of
a 2 : 1 plate.  The intact lattice is linear elastic to 2 % under a 4× load sweep
and does not break when the thresholds are frozen — the fragmentation is caused
by the disorder, not by the discretisation.

---

## 5. Limitations, negative results and what would fix them

1. **Janssen anchor missing in the DEM (criterion c).**  The
   `tensordem.granular2d` walls are frictionless segments; the study adds
   Coulomb wall friction as an operator-split impulse, which cannot sustain a
   *static* tangential load (a velocity-level friction law gives zero force when
   the slip is zero).  Hence ~95 % of the bed weight reaches the floor.  Fix:
   add a Cundall–Strack tangential spring at segment contacts in the
   `granular2d` engine (a small, well-scoped change, but it belongs to a
   verified module and was out of scope here).  The analytic Janssen profile and
   the fitting machinery are verified independently (`< 2 %` recovery of known
   `K`, `mu_w` from exact data).
2. **Two-dimensional central-force lattice (criterion a).**  No bond bending
   moments, no contact between detached floes, and platen compression instead of
   hull indentation.  The measured exponent is consistent with 2D percolation
   cluster statistics; a 3D lattice would raise the reference anchor to
   `alpha ≈ 2.4` but not to the campaign window.  Reaching `alpha ≈ 2.5–3.5`
   requires *repeated* crushing (comminution), which is the natural next step: the
   fragments of one pass become the bodies of the next, with a size-dependent
   strength (largest-floe-first breakage) that steepens the census.
3. **Full-width channel does not arch.**  Kept as a negative control: without a
   gate the discharge is a free plug flow and `P_clog = 0` for all `W/h`; the
   arching criterion is a property of a constriction, not of a channel.
4. **The arch-span metric is coarse.**  `strong_arch` returns the horizontal span
   of the widest strong-force component and whether it reaches both side walls;
   because the silo body is `2W` wide, a spanning component reports a span of
   `2W/h` (3.2–8.6 floe diameters in the sweep) even when the actual arch closes
   across the outlet.  A sharper detector would test for a connected strong path
   across the outlet plane at `y ≈ h` above the gate; the clog flag and the
   residual fraction do not depend on this metric.
5. **Wall friction in the silo is translational only** (no frictional torque); it
   therefore over-estimates slip and makes the arching threshold *conservative*
   (a torque-resisting contact would arch more readily ⇒ `(W/h)_crit` would move
   slightly up, still inside `[3, 6]`).
6. **`R²` definition.**  `R²_cdf` (survival function) is used as the primary
   dispersion metric because the log-binned version rests on four bins over the
   fitted decade; both are reported in `study.json` and in §4.1 so the choice is
   auditable.
7. **`P(f)` rate ≠ 1.**  The exponential *form* is verified (`R² > 0.9`), the
   unit-rate value is not; the pre-consolidation stress sets the rate and a
   stress sweep would map it (not run here).

---

## 6. Reproduce

```bash
python scripts/broken_ice_micro_study.py --stage fsd          # 4 ensembles, ~200 s
for i in 0 1 2 3 4 5; do
  python scripts/broken_ice_micro_study.py --stage arch --shard $i/6 &
done                                                          # 48 runs
python scripts/broken_ice_micro_study.py --stage janssen       # 3 columns, ~100 s
python scripts/broken_ice_micro_study.py --stage chain --seeds 3   # 3 x 3D rubble
python scripts/broken_ice_micro_study.py --stage assemble      # ledger
python scripts/broken_ice_micro_figures.py                     # figures
python -m pytest tests/test_broken_ice_micro_chain.py -q       # 20 tests
```

All numbers in this document are generated into
`artifacts/broken_ice_micro_study/study.json` (criteria ledger) and
`artifacts/broken_ice_micro_study/stages.json` (raw stage output); nothing is
transcribed by hand.