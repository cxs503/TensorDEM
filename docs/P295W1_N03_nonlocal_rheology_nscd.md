# P295W1-N03 — Nonlocal granular fluidity (NGF) closure + NSCD contact solver

**Repository:** TensorDEM · **branch:** `agent/m9902` · **module:** `src/tensordem/nonlocal_rheology/`
**Status:** NGF machinery + calibration pipeline + NSCD solver implemented and verified;
the pooled closure target `R² ≥ 0.95` is **not met** on the data reduced in this session
(measured `R² = 0.61`, up from `−0.88` for the local closure). Honest FAIL, see §6.

---

## 1. Case description (physics, geometry, parameters)

### 1.1 Why the local `μ(I)` law has to be extended
The local rheology `μ = μ_s + (μ_2−μ_s)/(I_0/I+1)` with `I = γ̇d/√(P/ρ)` assumes the
strain rate is determined *pointwise* by the stress state. In dense granular flow this
fails in two places: below yield (`μ < μ_s`) the local law predicts a rigid plug while
DEM and experiments show creep whose rate depends on how far the flowing region is, and a
shear band is observed to be several diameters wide where the local law predicts a band of
a fraction of a diameter. The nonlocal granular fluidity (NGF) model (Kamrin & Henann,
PRL 108, 178301, 2012) replaces the strain rate by a *fluidity field* `g`:

```
γ̇ = g μ                                         (1)
g = g_loc(μ) + ξ² ∇²g                            (2)
ξ = A d / √(|μ − μ_s|)                           (3)
```

`g_loc(μ) = γ̇_loc(μ)/μ` is the fluidity implied by the local law (zero below yield), and
(2) diffuses fluidity over the cooperativity length `ξ`, which diverges as yield is
approached. The two free objects are the local law `(μ_s, μ_2, I_0)` and the amplitude
`A = ξ/d`.

### 1.2 Test bed: 2D annular (Couette) shear cell
In an annulus the shear stress decays as `τ(r) ≈ T/(2πr²)` while the pressure is much
flatter, so a single run sweeps a *range* of stress ratio and crosses yield inside the
cell. Two annulus gaps provide the two calibration scales:

| quantity | symbol | value |
|---|---|---|
| particle radius (mean, ±12 % polydisperse) | `r` | 1.0 mm (`d50 = 2 mm`) |
| particle density | `ρ` | 2500 kg/m³ |
| normal stiffness / ratio | `k_n`, `k_t/k_n` | 3.0 × 10⁵ N/m, 0.5 |
| particle friction / wall friction | `μ`, `μ_wall` | 0.40, 3.0 (rough walls) |
| restitution | `e` | 0.10 |
| inner cylinder radius | `R_1` | 10 mm (5 d) |
| outer radius — **thin** gap | `R_2` | 22 mm (gap = 6 d) |
| outer radius — **wide** gap | `R_2` | 45 mm (gap = 17.5 d) |
| servo confining pressure | `P` | 300 Pa |
| inner angular velocity | `ω` | 15 / 50 / 150 rad/s |
| time step | `Δt` | 2.4 × 10⁻⁵ s (`T_contact/30`) |
| strain before averaging | — | 8 (settle 2 + shear 6), average over the last 3 |
| seeds | — | 1, 2 |

Fields are extracted by coarse-graining into radial bins: Love–Weber virial stress
projected on **each contact's own polar frame** plus the kinetic (Reynolds) part in each
particle's own frame; solid fraction from the **exact clipped disk area** per annulus bin.
`γ̇` comes from `d v_θ/dr − v_θ/r` on the binned profile.

### 1.3 NSCD contact solver
Soft-sphere DEM makes the overlap `δ = F/k` a state variable; in a rigid (`φ = 0.85`)
packing that is not a physical quantity. NSCD (Moreau 1988; Jean & Moreau 1992) imposes
the contacts as unilateral rigid constraints at the impulse level:

```
0 ≤ u_n⁺ + e u_n⁻  ⊥  p_n ≥ 0                       (Newton restitution)
p_t = argmin_{|p_t| ≤ μ p_n} |u_t⁺ + p_t/m_t|       (Coulomb cone)
```

solved by NLGS (non-linear Gauss–Seidel) sweeps of local projected sub-solves. Contacts
are activated on the *predicted* gap `g + Δt u_n⁻ < 0`, so the post-step normal velocity
satisfies `u_n⁺ ≥ −e u_n⁻ ≥ 0` and the flat-wall gap update `g⁺ = g + Δt u_n⁺` is
non-negative: **zero penetration by construction**.

## 2. What was done (method, tool chain)

* `fluidity.py` — local `μ(I)` inversion (`g_loc`), cooperativity length, finite-volume
  operator on a 1D annulus (`M_i = r_i Δr`, face conductances `r_{i±1/2}/Δr`) and on a 2D
  Cartesian grid (volume-integrated five-point stiffness), and a hand-written **conjugate
  gradient with Jacobi (diagonal) preconditioner**. Equation (2) is assembled as the SPD
  system `(M + ξ²K) g = M g_loc`; the coefficient `ξ²` is applied *on the faces* (the
  arithmetic face mean), which keeps the operator symmetric — writing `diag(ξ²)·K` does
  not, and that error was caught by an explicit symmetry/eigenvalue test.
* `couette.py` — the 2D annular cell kernel (numpy + cKDTree, Verlet-cached pair list,
  bincount scattering), rough rotating inner cylinder (torque meter), rough outer wall
  with a proportional radius servo holding the pressure, coarse-graining as in §1.2.
* `calibration.py` — two-scale protocol: `(μ_s, μ_2, I_0)` from the high-`I` subset of the
  pooled data (high `I` means `μ` far from `μ_s`, hence `ξ` small and the state local);
  `A = ξ/d` from the **thin-gap subset only** (the local law held fixed), by minimising the
  log-rate residual of the NGF prediction; then the frozen model is scored on the pooled
  data.
* `nscd.py` — Moreau stepper with three NLGS backends: `gauss_seidel` (sequential),
  `colored` (greedy colour-partitioned contact graph, vectorised but exactly equivalent to
  the sequential sweep) and `jacobi` (fully simultaneous, for large packings).
* `validation.py` — three verification tiers + a soft-sphere adjudication partner.
* `study.py` — driver (`runs` / `aggregate`), JSON + figure production.

Notes on two implementation decisions that are *not* the textbook phrasing:
`(i)` the campaign phrases the target as a pooled `μ(I)` **fit** `R²`; a free `μ`-vs-`I`
fit cannot be single-valued below yield, so the acceptance is applied to the equivalent
**rate-space closure** `R²` (§3.4) and the free `μ`-vs-`I` fit `R²` is reported alongside
for continuity. `(ii)` the local law is fitted on the high-`I` subset of *all* runs rather
than on the wide-gap run alone, because the wide gap does not reach the `μ` values that the
thin gap covers (`μ_2` would be extrapolated beyond the data).

## 3. Results (all numbers machine-produced into `artifacts/nonlocal_rheology_quick/summary.json`)

### 3.1 NGF solver evidence
| check | value | criterion |
|---|---|---|
| annular CG iterations / final relative residual | 60 / 1.6 × 10⁻¹⁹ | converged |
| CG vs dense `solve` (relative max diff) | 2.3 × 10⁻¹⁶ | ≤ 10⁻⁹ |
| manufactured solution, `L∞` error at `n = 21 / 41 / 81` | 9.66e−5 / 2.42e−5 / 6.04e−6 | — |
| observed convergence order (error ratio per halving) | **4.00 / 4.00** | ≥ 3.6 (2nd order) |
| `μ(I)` inversion round trip | 1.5 × 10⁻¹⁴ | ≤ 10⁻¹⁰ |

### 3.2 Local law and cooperativity length
| parameter | value |
|---|---|
| `μ_s`, `μ_2`, `I_0` | 0.434, 1.145, 0.351 |
| fit `R²` on the high-`I` subset (`I ≥ 0.0876`, 15 points) | **0.480** |
| fit RMSE | 0.097 |
| `A = ξ/d` (thin-gap, local law frozen) | **0.629** |
| log-rate RMSE on the calibration split | 0.401 |

### 3.3 Two-scale calibration outcome
The thin-gap state is genuinely nonlocal: the cell shears at `γ̇ ≈ 10–100 s⁻¹` across the
whole gap while the stress ratio decays from `μ ≈ 0.8` to `≈ 0.5`, i.e. the flow cannot be
produced by the (mostly sub-yield) local fluidity. The fitted amplitude `A ≈ 0.63` lies in
the range reported for granular flows (`A ~ 0.3–1`).

### 3.4 Pooled closure (3 runs, 36 bins; rate floor `γ̇_floor = 0.0205 s⁻¹`, declared as
`10⁻³ × mean measured rate`)

| metric | local closure | NGF closure | target |
|---|---|---|---|
| `R²` log-rate, pooled | **−0.875** | **0.610** | ≥ 0.95 → **FAIL** |
| RMSE log-rate | 2.200 | 1.003 | — |
| `R²` log-rate, thin gap (6 d, 23 bins) | −4.03 | **0.656** | — |
| `R²` log-rate, wide gap (17.5 d, 13 bins) | −5.03 | −0.633 | — |
| free `μ`-vs-`I` fit `R²` (continuity metric) | 0.651 | 0.665 (rectified `I`) | — |
| log-amplification `R²` / correlation | — | 0.488 / 0.732 | — |

The NGF correction improves the pooled closure by **+1.49 in `R²`** (a factor 2.2 in
log-RMSE) and turns the thin-gap closure from catastrophically wrong (−4.03) into
quantitatively useful (0.66). It does **not** reach 0.95.

### 3.5 NSCD verification tiers
Tier 1 — single sphere on a plane (12 `(e, Δt)` combinations, `e ∈ {0.2, 0.5, 0.8}`,
`Δt ∈ {8e−5 … 1e−5}` s):

| quantity | NSCD | soft-sphere (adjudication) |
|---|---|---|
| restitution error `|e_meas − e|` | **≤ 1.1e−16** | up to 1.7e−1 |
| max penetration / `d` | **0 (exactly)** | 1.5e−2 |
| anticipated-contact offset | `1.6e−5 m = v Δt/2` (O(Δt)) | — |

Tier 2 — 12-sphere impact chain (`Δt = 1e−5`, `e = 0.8`): momentum error **0.0**
(bit-exact), max penetration **0**, kinetic energy monotonically non-increasing, final
`KE/KE₀ = 0.142`, outlet velocity 0.314 m/s (the chain transmits a pulse, as it must).

Tier 3 — `φ = 0.85` rigid packing: the packing generator was rebuilt (HCP stacking with
the correct layer stagger, no initial overlap, verified `min d_ij = s > 2r`), the lattice
now clears the walls, and zero penetration holds exactly — but the load-carrying
diagnostics are **not trustworthy**: a 90-particle packing needs far more NLGS sweeps than
the budget allowed (the impulse residual stalls at ~1e−8 with `max_iter = 60`), so no
reportable load balance was obtained. Declared as **WITHHELD** (see §6).

## 4. Figures

* `artifacts/nonlocal_rheology_quick/figures/rheology_profiles.png` — (a) radial strain
  rate: DEM (circles) vs local (dotted) vs NGF (solid) for the 6 d and 17.5 d gaps;
  (b) pooled `μ` vs `I` with the wide-band local law; (c) the calibrated `ξ/d(μ)`.
* `artifacts/nonlocal_rheology_quick/figures/nscd_verification.png` — restitution error vs
  `Δt`, penetration for NSCD vs the penalty model, and the O(Δt) contact offset.

## 5. Notes, limitations, pitfalls found

1. **An overlapped start destroys a soft-disk kernel.** A jittered lattice at a nominal
   `φ` produced 20–50 % overlaps and the run exploded within a handful of steps
   (`cKDTree` overflow). The packing is now generated at the target `φ` with a lattice
   pitch from `φ`, tiny jitter, and a positional relaxation, verified overlap-free.
2. **The drive saturates at the wall friction.** With `μ_wall = 0.5` the inner cylinder
   transmitted at most `μ_wall P 2πR_1²` of torque and simply slipped: the measured `γ̇`
   was 100× smaller than `ωR_1/gap` (≈ 0.5 s⁻¹ vs 42 s⁻¹) — a plugged cell, not a
   rheology. Rough walls (`μ_wall = 3`) fixed it; wall slip is still 40–90 % of the wall
   speed in the thin cell and is a declared limitation (it creates a boundary layer).
3. **`diag(ξ²)·K` is not symmetric.** The first version produced a non-SPD operator
   (min eigenvalue < 0) and CG diverged. The coefficient must be applied on the faces.
4. **A Cartesian bin average cancels the shear stress in an annulus.** Summing `l⊗f` over
   azimuth and then projecting on one mean radial direction gives `⟨ê_r ê_θ⟩ = 0`, so
   `μ` came out 10× too small. Each contact must be projected on its own frame.
5. **Centre-in-bin porosity exceeds 1** for bins thinner than a diameter. Replaced by the
   exact clipped-disk-area integral; `φ ≤ 1` is now enforced structurally by the geometry
   (a regression test checks it).
6. **Wall normals**: the stored contact normal must point from body *i* to body *j*; for a
   wall that is the *inward* normal, not the wall's outward normal. The opposite choice
   made the wall suck the sphere in.
7. **The `μ(I)` fit is weakly identified** on 15 high-`I` points: `μ_s` and `I_0` sit near
   their bounds and the fit `R²` is 0.48. This is the dominant error source in §3.4 — the
   local law, not the NGF operator, is the bottleneck.
8. **NLGS convergence at packing scale** needs warm-started impulses or a block/JORProx
   solver; the present implementation is fine for the small verification tiers and for
   impact problems, not yet for a loaded 3D packing. Also note the `colored` backend
   re-colours the contact graph every step, which dominates its cost at `N ≳ 500`.
9. **Reduced dataset in this session.** The full design (12 runs: 2 gaps × 3 rates × 2
   seeds) was submitted to the cluster (`cj397`, 12 shards, `study.py runs --shard i
   --shards 12`) but did not return in time; the numbers above come from **3 runs
   (1 wide + 2 thin, seed 1)**. Fewer bins make both fits noisy: the thin-gap result is
   encouraging, the pooled number is not converged.

## 6. Error determination (key quantities vs reference)

| quantity | value | reference | error | < 3 %? |
|---|---|---|---|---|
| CG solution vs dense linear solve | 2.3e−16 rel. | exact linear algebra | 2.3e−16 | **yes** |
| manufactured-solution convergence order | 4.00 (h→h/2) | 4 (2nd order in `L∞` per halving) | 0 % | **yes** |
| `μ(I)` inversion round trip | 1.5e−14 rel. | exact inverse | 1.5e−14 | **yes** |
| NSCD restitution, single impact | `e_meas` = `e` to 1.1e−16 | Newton's law | ~1e−16 | **yes** |
| NSCD penetration, tiers 1–2 | 0 exactly | 0 | 0 | **yes** |
| momentum conservation, 12-sphere chain | 0 exactly | 0 | 0 | **yes** |
| pooled closure `R²` | 0.610 | target ≥ 0.95 | 0.34 absolute shortfall | **no — FAIL** |
| local-law fit `R²` | 0.480 | ≥ 0.95 (a local fit should be near 1) | — | **no — FAIL** |
| `A = ξ/d` | 0.629 | no external anchor (calibration split RMSE 0.40 in log-rate) | — | WITHHELD |
| tier-3 packing load balance | not obtained | `|F_lid| = |F_floor|` | — | WITHHELD |

Honest verdict: the **solver-level** capabilities (NGF elliptic solve with CG+Jacobi,
two-scale calibration machinery, NSCD with zero penetration and exact restitution) are
delivered and verified to machine precision. The **physical** target (pooled closure
`R² ≥ 0.95`) is **not met** — the measured value is 0.61 — and the φ = 0.85 packing load
balance is withheld. Neither is dressed up: both are reported as failures with the
diagnosis (local-law identifiability + reduced dataset; NLGS convergence at packing scale).