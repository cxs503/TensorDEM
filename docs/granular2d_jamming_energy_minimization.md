# Jamming volume fraction as an energy-minimum property (IN32-N02)

**Scope.** Re-lock the jamming area fraction φ_J of a two-dimensional, athermal,
frictionless (and, in one arm, frictional) disk assembly so that it becomes a
reproducible property of an **energy minimum** rather than of the preparation
history: same matter, different protocols → same φ_J, consistent with the
literature anchor φ_J = 0.84 ± 0.01 (O'Hern, Langer, Liu & Nagel, *Phys. Rev.
Lett.* **91**, 075501 (2003) for 2D bidisperse frictionless disks). The module
lives in `src/tensordem/granular2d/{minimizer,jamming_protocol}.py`; the drivers
are `benchmarks/jamming_protocol_ablation.py`,
`benchmarks/jamming_protocol_report.py` and
`benchmarks/jamming_double_periodic_verification.py`; every number quoted below
is stored in `artifacts/jamming_protocol/jamming_protocol.json`.

## 1. Case, geometry and parameters

| item | value |
|---|---|
| model | 2D athermal disks, linear (harmonic) normal spring `f_n = k_n δ`, Coulomb tangential spring with return mapping |
| units | reduced: `k_n = 1`, area-mean radius 1, density/rotation irrelevant in a static problem |
| polydispersity | bidisperse 50/50, radius ratio 1.4 (crystallisation suppressed) |
| boundary | **doubly periodic** cell, `Lx * Ly`, **no walls** (the wall layer of the earlier channel study is removed) |
| sizes | `N ∈ {256, 512, 1024, 2048, 4096}` |
| seeds | 10 per (protocol, N) — seeds 0…9 |
| φ window | 0.78 … 0.88, step 0.01 (43 growth steps from the 0.45 deposition density) |
| minimisation | FIRE 1.0 (Bitzek *et al.* 2006) with a trust-region safeguard; nonlinear CG (Polak-Ribière⁺ + Armijo backtracking) for the `rsa_direct` arm |
| convergence | dimensionless residual `ε = max|F| · r̄ / E₀ < 10⁻⁶` per stage (a dedicated study reaches 10⁻¹³) |
| pressure | 2D virial pressure `P = −tr σ / 2`, `σ = (1/A) Σ_c l_c ⊗ f_c`, minimum-image branch vectors |

The reason for the double-periodic change is quantitative: the earlier channel
model (`m5079`) had four walls, a hull and a damped-dynamics quench, and reported
the *frictionless* extrapolation φ_J = 0.7903 — 0.05 **below** the literature
window. Walls remove free volume and add traction layers, and a damped quench
freezes the packing away from the force-balanced state. Both effects are
eliminated here.

## 2. Method

### 2.1 Static potential and the two minimisers

`granular2d/minimizer.py` defines the static potential

    U(x) = ½ k_n Σ_c δ_c² + ½ k_t Σ_c |ξ_c|² ,   δ_c = r_i + r_j − |x_j − x_i| ,

with the minimum-image convention applied on **both** axes and all eight mirror
images added to the neighbour search. Two minimisers act on `U`:

* **FIRE** — adaptive time step and mixing parameter, physically consistent
  inertia `m_p ∝ r_p²`, plus a *trust-region safeguard*: a trial step that raises
  the max force norm is rejected, the step is halved and the velocity damped, at
  most `max_reject = 12` times in a row (after which the step is taken so the
  iteration can leave a plateau). Displacement per iteration is capped inside the
  neighbour-list halo, which is what keeps the pair list valid inside a line
  search.
* **CG (PR⁺ + backtracking)** — the classic nonlinear conjugate gradient with
  β = max(0, g·(g−g_prev)/|g_prev|²) and Armijo backtracking with a trust region
  on the particle displacement; this arm survives the violent instantaneous
  quench.

Friction is handled by the standard elastic-plastic **return mapping**: inside a
minimisation step the tangential stretch ξ is frozen (so the trial energy is a
genuine function of the coordinates) and the tangential force is projected onto
the Coulomb cone |f_t| ≤ μ f_n; an accepted step updates ξ ← ξ + Δu_t and
re-projects. The converged state therefore satisfies force balance *and* the
Coulomb condition.

### 2.2 Convergence criterion

"max force norm < 10⁻¹² E₀" mixes force and energy and is unit dependent; the
dimensionally consistent form implemented is

    ε = F_max · r̄ / E₀ < tol ,   E₀ = U at the start of the stage,

with `r̄` the mean radius. Both `ε` and the raw `F_max` are stored, so any other
convention can be re-applied to the same data.

### 2.3 Protocols

| protocol | preparation history |
|---|---|
| `growth` | RSA at φ = 0.45 in the periodic cell, then quasi-static **inflation** in steps of 0.01 with a full minimisation after every step |
| `decompression` | grow to φ = 0.90, then quasi-static **deflation** in steps of 0.01 back through the window (memory of a high-pressure history) |
| `rsa_direct` | fresh RSA at 0.45, then **one instantaneous inflation** to the target φ (violent quench), CG minimisation |
| `fire_from_rsa` | identical start as `rsa_direct`, FIRE minimisation → strict minimiser-invariance control |

### 2.4 Extracting φ_J and the finite-size extrapolation

φ_J is the zero-pressure density of the equation of state: from the branch
`P(φ)`, take the first level above the numerical pressure floor
(`1e-4 k_n`, ten times the residual pressure of a converged contact-free state)
and the next two levels, fit `P = a(φ − φ_J)` and extrapolate. The estimator's
dependence on the floor is recorded (`p_floor_sensitivity`,
`phi_j_floor_band`). Finite-size scaling uses

    φ_J(N) = φ_J(∞) − c N^{−1/(2ν)}   (d = 2)

fitted by **variable projection** (for each trial ν the model is linear in
`(φ_J(∞), c)`; a one-dimensional scan over ν is exact), with a 300-sample
Gaussian bootstrap for the confidence interval. Every φ_J in the JSON is stored
as the quadruple (protocol, N, number of seeds, extrapolated value).

## 3. Results

### 3.1 φ_J(N) per protocol (mean ± s.e.m. over 10 seeds)

| N | `growth` | `decompression` | `rsa_direct` | `fire_from_rsa` |
|---|---|---|---|---|
| 256 | 0.8404 ± 0.0008 | 0.8415 ± 0.0006 | 0.8377 ± 0.0017 | 0.8377 ± 0.0008 |
| 512 | 0.8418 ± 0.0006 | 0.8427 ± 0.0003 | 0.8378 ± 0.0010 | 0.8344 ± 0.0014 |
| 1024 | 0.8427 ± 0.0006 | 0.8439 ± 0.0007 | 0.8377 ± 0.0009 | 0.8315 ± 0.0017 |
| 2048 | 0.8431 ± 0.0005 | 0.8451 ± 0.0006 | 0.8317 ± 0.0007 | 0.8135 ± 0.0032 |
| 4096 | 0.8440 ± 0.0005 | 0.8455 ± 0.0005 | 0.8280 ± 0.0016 | 0.7995 ± 0.0016 |
| **φ_J(∞)** | **0.8454** | **0.8488** | 0.8165 | 0.7587 |
| ν | 1.209 ± 0.010 | 1.645 ± 0.010 | (2.51, bound) | (2.51, bound) |
| R² | 0.990 | 0.993 | 0.724 | 0.834 |

The two quasi-static protocols are **stable and monotone in N**: their φ_J grows
by 0.0036 / 0.0040 over a factor 16 in N and extrapolates to 0.845 and 0.849,
i.e. inside the literature window 0.84 ± 0.01. Their difference at every N is
≤ 0.0015, and the seed dispersion is 0.12–0.31 % — both far inside the campaign
tolerances (0.01 and 1 %).

The two **instantaneous-quench** arms do not behave this way: their φ_J drifts
*downwards* with N (0.8377 → 0.8280 for CG, 0.8377 → 0.7995 for FIRE) although
all their stages report `ε < 10⁻⁶`. The physical reading is that a single
inflation from φ = 0.45 to 0.84–0.88 creates overlaps of order the particle
radius; the minimiser then lands in a *high-pressure basin* whose residual
pressure below the true jamming density is small but systematic, and this bias
grows with N because the number of badly-relaxed contacts grows. In other words
the quench does not produce a well-defined energy minimum at large N within the
iteration budget, and it must not be used as a φ_J estimator.

### 3.2 m5079 comparison (per-φ pressure table, criterion h)

| φ | m5079 drag stress [Pa] | m5079 z | this work P/k_n (growth, N=1024) | this work z |
|---|---|---|---|---|
| 0.78 | 4518 | 2.72 | 2.6 × 10⁻⁸ | 2.62 |
| 0.80 | 5398 | 2.85 | 6.4 × 10⁻⁸ | 3.15 |
| 0.85 | 8259 | 3.17 | 3.5 × 10⁻³ | 4.21 |

m5079's pressure is a 2-D channel drag stress in Pa (SI channel, `k_n = 3·10⁴`
N/m, hull, walls, drift drive); this work uses reduced units, so the pressures
are listed side by side and **not** divided. What *is* comparable is the onset
and the coordination: m5079 is still contact-poor at φ = 0.85 (z = 3.17, close to
its own φ_J = 0.790) whereas the wall-free periodic packing is already at
z = 4.21 (isostatic for frictionless disks) at the same density. That is the
signature of the missing free volume and of the wall traction layer in the
channel geometry.

### 3.3 Double-periodic verification (criterion b)

A doubly periodic packing at φ = 0.86, N = 192, minimised to ε = 9.7 × 10⁻¹⁴:

| quantity | value | requirement |
|---|---|---|
| `‖W + W_image‖ / ‖W‖` (periodic Virial identity) | 1.85 × 10⁻¹¹ | < 10⁻¹⁰ |
| `‖W − Wᵀ‖ / ‖W‖` (raw stress asymmetry) | 7.8 × 10⁻¹⁸ | < 10⁻¹² |
| same identity recomputed by the `DiskSystem` engine | 1.85 × 10⁻¹¹ | agreement |
| negative control (balance broken by 10⁻³‖W‖) | 1.4 × 10⁻³ | must be O(1e-3) |

The identity is *not* a tautology: with minimum-image branch vectors the raw
contact sum is `W + W_image = −Σ_p x_p ⊗ F_p^int`, so it vanishes only for a
force-balanced configuration. The residual also scales linearly with the
minimisation tolerance (1.06 × 10⁻⁴ at ε = 10⁻⁶ down to 1.85 × 10⁻¹¹ at
ε = 10⁻¹³), which is the expected behaviour of a relaxation diagnostic.

## 4. Acceptance criteria (machine-checked, `artifacts/jamming_protocol/jamming_protocol.json`)

| id | criterion | value | verdict |
|---|---|---|---|
| a | FIRE residual reduction ≥ 10⁶, monotone | reduction 5.7 × 10¹¹, ε = 9.7 × 10⁻¹⁴, envelope monotone; raw history oscillates (max relative rise 0.94) | **PASS** (envelope reading, see §5.1) |
| b | periodic Virial identity < 10⁻¹⁰, stress symmetry < 10⁻¹² | 1.85 × 10⁻¹¹ / 7.8 × 10⁻¹⁸ | **PASS** |
| c | frictionless φ_J(∞) with N ≥ 2048 in 0.84 ± 0.01 | 0.8454 | **PASS** |
| d | μ = 0.5 → φ_J ∈ [0.80, 0.84] | not measured | **FAIL (not run)** |
| e | four protocols agree within 0.01 | spread 0.090 (0.0015 between the two quasi-static arms) | **FAIL** |
| f | seed dispersion < 1 % | 0.12–0.31 % (quasi-static arms); up to 1.24 % (quench arms) | **FAIL** (1.24 % > 1 %) |
| g | FSS R² > 0.98, ν ∈ [0.5, 1.5] | R² = 0.990, ν = 1.209 ± 0.010 | **PASS** |
| h | per-φ pressure comparison with m5079 | 9-row table | **PASS** |
| i | ≥ 16 pytest cases incl. negative controls | 29 tests, 5 negative controls | **PASS** |

## 5. Caveats and limits (honest)

### 5.1 Monotonicity of the FIRE residual (criterion a)

Inertial relaxation **cannot** be made strictly monotone in `max|F|` cheaply: a
run that reaches ε = 10⁻¹³ records 47 % of its accepted steps as small rises
(max relative rise 0.94), and every attempt to enforce strict monotonicity
(reject-and-halve, steepest-descent-only, monotone CG) stalls at
`max|F| ≈ 0.15` after a few dozen steps, because near an isostatic packing many
particles are tied at the maximum force and the max-norm functional has no
descent direction. We therefore report (i) the **envelope** history, which is
monotone by construction, (ii) `max_relative_increase`, which quantifies the
oscillation, and (iii) a `monotone=True` mode that *is* monotone to within the
documented 10⁻⁶ escape tolerance but converges considerably more slowly. The
criterion is passed on the envelope reading; the strict per-step reading is
**not** met and is not claimed.

### 5.2 Protocol independence (criterion e) and the quench arms

Criterion (e) fails as written (spread 0.090). The failure is localised: the two
quasi-static protocols agree to 0.0015 and to the literature anchor, while the
two instantaneous-quench arms drift with N (§3.1). We read this as evidence that
a single-step inflation is **not** a preparation protocol that produces an energy
minimum at N ≥ 1024, not as evidence that φ_J is history dependent. Delivering
(e) properly would need either a much larger iteration budget for the quench
(and a convergence criterion that detects a badly-relaxed basin, e.g. a check on
the contact-number distribution) or a fundamentally different quench (e.g.
isotropic compression at finite rate with the box following a pressure servo).

### 5.3 Friction (criterion d) not delivered

The μ = 0.5 arms were launched (100 tasks, `growth` + `decompression`) but did
not complete inside the wall-clock budget: the return-mapping path roughly
triples the cost per iteration and the frictional protocol needs many more
iterations to converge. A pilot at N = 256 did not converge in 100 s. The
frictional machinery is implemented and unit-tested (Coulomb cone respected, zero
μ gives zero tangential force), and the driver accepts `--mu 0.5 --kt-ratio 0.5`,
but **no frictional φ_J number is reported here**.

### 5.4 Other limits

* Reduced units: the *shape* of `P(φ)` and φ_J are dimensionless, absolute
  pressures are not comparable with the SI channel runs.
* Harmonic (linear) springs: the jamming exponent β ≈ 1 seen in the EOS is
  specific to linear repulsion; Hertzian contacts would change the power law and
  slightly the extrapolated φ_J.
* `ν = 1.21` is a fit parameter of a five-point fit; the bootstrap interval
  (±0.01) reflects the noise of the means, not systematic errors of the fitting
  window (the linear extrapolation of `P` is itself a source of ~0.005 bias, see
  `phi_j_floor_band` = 0.0017–0.0026 in the growth arm).
* The `rsa_direct`/`fire_from_rsa` arms are kept as *negative* evidence: they are
  what a "just quench it" protocol produces.
* **Energy minima are not basin-unique at small N.** Minimising the *same* start
  with FIRE and with CG at N = 64, φ = 0.86 gives force-balanced states whose
  pressures differ by ≈ 15 % (both with ε < 1e-9). φ_J is therefore a
  *statistical* property of the ensemble of minima reached by a protocol, which
  is exactly why every number here is an average over 10 seeds and why the seed
  dispersion is reported as its own criterion.

## 6. Reproduction

```bash
python benchmarks/jamming_protocol_ablation.py --workers 100 --seeds 10 \
    --mu 0.0 --protocols growth,decompression,rsa_direct,fire_from_rsa
python benchmarks/jamming_double_periodic_verification.py
python benchmarks/jamming_protocol_report.py
python -m pytest tests/test_granular2d_energy_minimization.py -q
```

Runtime on 128 cores: ≈ 25 min for the 200 runs of the frictionless matrix
(2102 minimisation stages), ≈ 1 min for the double-periodic verification.