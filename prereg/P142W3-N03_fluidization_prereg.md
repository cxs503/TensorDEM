# P142W3-N03 — pre-registration: fluidised-bed minimum verifiable unit

**Written before any production number of this node was produced** (the only
prior numbers consulted were the packing/porosity bookkeeping of the existing
`IN143-N09` module, which is reused, not re-scored).

## 1. Decision this study must support

> Is the TensorDEM point-particle granular–fluid coupling layer
> (`tensordem.bedcoupling`) a *faithful* unresolved DEM–CFD closure set — i.e.
> does it reproduce the four classical anchors (Ergun pressure drop,
> single-particle terminal velocity, Richardson–Zaki hindered settling,
> Wen–Yu onset of fluidisation) and does its three-force momentum ledger close
> — to the tolerances pre-declared below?

**Alternatives considered** (and why not chosen as the primary route):

*A1.* Re-derive the closures from a resolved pore-scale CFD field
(lattice-Boltzmann on the packing). Rejected as out of scope for a *minimum
verifiable unit*: it adds a second solver whose own error budget would obscure
the closure audit, and the classical anchors already exist as analytic
reference values.

*A2.* Report only the closure algebra (dp/dz from Ergun evaluated twice).
Rejected: it is tautological — nothing is integrated, so no ledger and no
hindered-settling behaviour is exercised.

*A3.* (chosen) Drive the closures through a **time-integrated DEM particle
system** (velocity-Verlet, no contacts, closed/periodic column) and audit the
result against analytic anchors *and* against a per-step momentum ledger.

## 2. Hypothesis

H1. With the Ergun/Wen–Yu (Gidaspow) blend evaluated on the bed's *local*
voidage, the bed-averaged `dp/dz` reproduces the analytic Ergun relation over
`Re_s ∈ [1, 1000]` within 10 %.

H2. A single particle settling through the same closure reaches the Stokes
terminal velocity within 5 % (Stokes regime, `Re_t < 0.01`) and the
Schiller–Naumann terminal velocity within 10 % (transition regime).

H3. The simulated hindered settling speed obeys `u_p(ε) = u_t ε^n` with the
Richardson–Zaki index within the reported band, and each point of the curve is
within 10 % of the RZ prediction.

H4. The onset velocity found from the *closure's own* pressure drop equalling
the bed weight per area reproduces the Wen–Yu (1966) correlation within 10 %.

H5. The three-force ledger (`F_drag + F_pressure + F_gravity`) closes against
`d(mv)/dt` to ≤1 % of the bed weight, both step-wise (bookkeeping identity) and
in the steady (equilibrium) window.

H6. Three resolutions (N = 600 / 1200 / 2400 particles) agree within 5 % on
`dp/dz` at three Reynolds numbers, on `U_mf` and on the steady settling speed.

## 3. Metrics, unit of analysis, thresholds (pre-declared)

| id | metric | unit of analysis | threshold | source of threshold |
|----|--------|------------------|-----------|---------------------|
| C1 | `max_U \|dp/dz_closure − dp/dz_Ergun\| / dp/dz_Ergun` | one U in a 13-point log sweep, `Re_s∈[1,1000]` | ≤ 10 % | node spec ① |
| C2a | `\|u_settle − u_Stokes\| / u_Stokes` | one single-particle run, `Re_t<0.01` | ≤ 5 % | node spec ② |
| C2b | `\|u_settle − u_SchillerNaumann\| / u_SN` | one single-particle run, `Re_t≈O(1–100)` | ≤ 10 % | node spec ② |
| C2c | `max_ε \|u_p(ε) − u_t ε^n_RZ\| / (u_t ε^n_RZ)` | four porosities ε=0.80/0.86/0.92/0.96 | ≤ 10 % | node spec ② |
| C3a | `max_step \|Δp/Δt − ½(F_n+F_{n+½})\| / W_bed` | every step of the ledger run | ≤ 1 % | node spec ③ |
| C3b | `\|Σ F\|_{steady} / W_bed` | steady window (last 25 % of steps) | ≤ 1 % | node spec ③ |
| C4 | `\|U_mf,closure − U_mf,WenYu\| / U_mf,WenYu` | one fluidisation case | ≤ 10 % | node spec ④ |
| C5 | `max over quantities of (max−min)/mean across N∈{600,1200,2400}` | three resolutions | ≤ 5 % | node spec ⑤ |

Aggregate rule: the node **passes** iff all of C1–C5 pass. Any failure is
reported as an honest FAIL with its number; no threshold is moved afterwards.

## 4. Baselines / controls / negative controls (all reported, none deleted)

* **Analytic anchors** (baselines): Ergun (1952); Stokes `(ρ_p−ρ_f) g d²/18μ`;
  Schiller–Naumann `C_D = 24/Re(1+0.15Re^0.687)`; Richardson–Zaki power law with
  the tabulated piecewise `n(Re)`; Wen–Yu onset correlation
  `Re_mf = sqrt(33.7² + 0.0408 Ar) − 33.7`.
* **N1 (buoyancy variant)**: hydrostatic `∇p = −ρ_f g` instead of the mixture
  field `∇p = −(ερ_f+(1−ε)ρ_p) g`. The two differ by one power of ε and hence
  by ≈1 unit of the RZ index; the hydrostatic variant is *expected to fail* the
  RZ comparison — that is the demonstration that the comparison discriminates.
* **N2 (dropped ledger term)**: run the dynamics with the drag term omitted from
  the *applied* force while the ledger keeps recording it. The ledger residual
  must blow up to O(1); if it stays ≤1 % the check has no teeth and C3 must be
  declared invalid.
* **N3 (wrong drag branch)**: Wen–Yu branch only, at ε = 0.38. Expected large
  error against Ergun (it is the wrong branch for a dense bed) — reported as a
  discrimination check for C1.
* **N4 (time-step halving)**: the discretisation-level residual
  `max|F̄_centred − dp/dt|/W` must shrink under `dt → dt/2`; a non-shrinking
  residual would indicate an integration error rather than truncation.

## 5. Controlled vs changed variables

*Controlled*: particle diameter 1 mm (packed bed) / 0.2 mm (Stokes case),
densities, viscosity, gravity, packing target porosity 0.38, seeds
(20261110 for packings, 11 for dynamics), local-voidage probe radius 1.5 d
(all resolutions), integrator (velocity-Verlet), timestep rule `dt = 0.1 τ`.
*Changed one at a time*: superficial velocity (C1 sweep), bed resolution
(C5), buoyancy model (N1), applied-force set (N2), drag branch (N3), timestep
(N4).

## 6. Confounders / nuisance variables

* Local-voidage kernel bias (small kernels oversample the cage around a
  particle) → fixed probe radius across resolutions, and the mean-voidage
  assembly is reported alongside.
* Periodic-cell finite size (box 8–13 diameters) → C5 explicitly measures it.
* Gidaspow branch discontinuity at ε = 0.8 → the voidages used are far from the
  switch (0.38 for Ergun, ≥0.80 for RZ with the Wen–Yu branch) and the jump is
  reported rather than smoothed.
* Discretisation: `dt = 0.1τ` → N4 measures the residual's dt-scaling.
* Random packing seed → one seed per resolution; the seed spread is bounded by
  the packing porosity statistics (std ≤ 0.02 reported).

## 7. Repetitions / sample size

Deterministic runs (no stochastic dynamics beyond the initial velocity
perturbation of amplitude 1e-4 m/s, which decays in <1 τ). One run per case is
therefore sufficient; variability is reported through (i) the steady-window
standard deviation and (ii) the 13-point velocity sweep, not through repeated
identical runs. The resolution sweep (3 sizes) is the sample-size device for
the discretisation uncertainty.

## 8. Environment, versions, commands

Repo `TensorDEM`, branch `agent/m9019`, Python 3.10, numpy/scipy/torch from
`uv.lock`. Produce: `python scripts/p142_n03_study.py --out-dir artifacts/P142W3-N03`
→ `results.json` + 3 figures. Tests: `python -m pytest tests/test_bedcoupling_fluidization.py -q`.
Both executed on the single control node (runs are ~3 min single core, so no
cluster submission was necessary; the cluster route is recorded in the report
as the escalation path for the N→∞ refinement).

## 9. Analysis plan (fixed in advance)

All criteria are computed by the study script from run output; the report only
quotes the JSON. Aggregation: max over the sweep for C1/C2c, steady window mean
for C3b, (max−min)/mean for C5. No outlier removal. No post-hoc threshold
changes. Honest FAIL is reported as FAIL.

## 10. Stopping rule

Stop when C1–C5 are computed once at the declared settings. If any criterion
fails, one diagnostic pass (larger N or smaller dt) is allowed to attribute the
failure, and both the failing number and the diagnostic are reported.

## 11. Safety / resource limits

No contacts, no fracture → no blow-up risk; instability is detected by a
finite-value check on the velocity record (test asserts it). Resource budget:
≤ 5 min single core, ≤ 1 GB RAM.